"""
ft_09_fn_masking — Linq+LoRA with false-negative masking via moral similarity matrix.

Standard InfoNCE treats every in-batch fable as a negative. But when two morals
in the same batch are near-duplicates (e.g. "Union is strength" / "Union gives
strength"), the model gets penalised for retrieving either fable — even though
both answers are valid. This experiment masks those pairs out of the denominator.

The masking matrix is precomputed once from BGE-large embeddings of all 709
morals and cached at data/processed/moral_sim_matrix.npy. Any in-batch moral
pair with cosine similarity > fn_mask_threshold is excluded from the loss.

Compared to ft_08 (clean sanity check), ft_09 trains on the full 709 queries
but with the false-negative signal removed rather than discarding ambiguous data.

Usage
-----
    ./run.sh finetuning/ft_09_fn_masking/train.py --remote --gpu 0
    ./run.sh finetuning/ft_09_fn_masking/train.py --fold 0 --remote --gpu 0
    ./run.sh finetuning/ft_09_fn_masking/train.py --force --remote --gpu 0
"""
import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Reduce CUDA memory fragmentation — needed when a resident process on the GPU
# leaves <100 MB headroom (allocator can satisfy requests from smaller segments).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml

EXP_DIR     = Path(__file__).parent
ROOT        = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from transformers import TrainerCallback

from finetuning.lib import notify
from finetuning.lib.data import load_pairs
from finetuning.lib.eval import evaluate

CACHE_DIR   = EXP_DIR / "cache"
RESULTS_DIR = EXP_DIR / "results"
CONFIG_PATH = EXP_DIR / "config.yaml"

SIM_MATRIX_PATH = ROOT / "data/processed/moral_sim_matrix.npy"


# ── False-negative-aware InfoNCE loss ─────────────────────────────────────────

class MaskedInfoNCELoss(nn.Module):
    """
    InfoNCE loss with false-negative masking via a precomputed moral sim matrix.

    Labels passed in forward() are corpus-level moral indices (0..708).
    Any in-batch pair (i, j) where moral_sim_matrix[label_i, label_j] > fn_threshold
    is masked out of the softmax denominator — the model is not penalised for
    being attracted to fables whose morals are semantically equivalent to the query.
    """

    def __init__(self, model, temperature: float = 0.05, moral_sim_matrix: torch.Tensor = None, fn_threshold: float = 0.85):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.fn_threshold = fn_threshold
        self.register_buffer("moral_sim_matrix", moral_sim_matrix.float())

    def forward(self, sentence_features: list, labels=None) -> torch.Tensor:
        moral_emb = F.normalize(self.model(sentence_features[0])["sentence_embedding"], dim=-1)
        pos_emb   = F.normalize(self.model(sentence_features[1])["sentence_embedding"], dim=-1)

        B      = moral_emb.size(0)
        device = moral_emb.device
        τ      = self.temperature

        # ── Type 1: moral_i vs all in-batch fables ────────────────────────────
        fable_sim = torch.mm(moral_emb, pos_emb.T) / τ  # (B, B)

        # ── False-negative masking ─────────────────────────────────────────────
        # labels = corpus-level moral indices; look up pairwise sim for the batch
        if labels is not None:
            batch_sim = self.moral_sim_matrix[labels][:, labels]  # (B, B)
            false_neg = batch_sim > self.fn_threshold
            false_neg.fill_diagonal_(False)                        # keep the positive
            fable_sim = fable_sim.masked_fill(false_neg, float("-inf"))

        # ── Type 2: moral_i vs other morals in the batch ─────────────────────
        moral_sim = torch.mm(moral_emb, moral_emb.T) / τ  # (B, B)
        moral_sim = moral_sim.masked_fill(torch.eye(B, dtype=torch.bool, device=device), float("-inf"))

        all_logits = torch.cat([fable_sim, moral_sim], dim=1)  # (B, 2B)
        targets    = torch.arange(B, device=device)
        return F.cross_entropy(all_logits, targets)


# ── Precompute moral sim matrix ────────────────────────────────────────────────

def load_or_build_sim_matrix(moral_texts: list[str]) -> torch.Tensor:
    """BGE-large cosine similarity matrix over all 709 morals. Cached to disk."""
    if SIM_MATRIX_PATH.exists():
        print(f"  [sim matrix] Loading from cache ← {SIM_MATRIX_PATH}")
        return torch.from_numpy(np.load(SIM_MATRIX_PATH))

    print("  [sim matrix] Computing BGE-large embeddings for all morals...")
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer("BAAI/bge-large-en-v1.5")
    prefixed = ["Represent this sentence: " + t for t in moral_texts]
    emb = bge.encode(prefixed, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.array(emb, dtype=np.float32)
    sim = np.dot(emb, emb.T).astype(np.float32)

    SIM_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(SIM_MATRIX_PATH, sim)
    print(f"  [sim matrix] Saved → {SIM_MATRIX_PATH}")
    del bge
    gc.collect()
    return torch.from_numpy(sim)


# ── Best adapter callback ──────────────────────────────────────────────────────

class BestAdapterCallback(TrainerCallback):
    def __init__(self, peft_model, metric_key: str):
        self.peft_model = peft_model
        self.metric_key = metric_key
        self.best_mrr   = -1.0
        self.best_state: dict | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        mrr = metrics.get(self.metric_key)
        if mrr is not None and float(mrr) > self.best_mrr:
            self.best_mrr = float(mrr)
            self.best_state = {
                k: v.detach().cpu().clone()
                for k, v in self.peft_model.state_dict().items()
                if "lora" in k.lower()
            }
            print(f"    [best] MRR={mrr:.4f} — adapter saved to CPU")

    def restore(self, peft_model) -> float:
        if self.best_state is None:
            return self.best_mrr
        device = next(peft_model.parameters()).device
        state  = peft_model.state_dict()
        for k, v in self.best_state.items():
            state[k] = v.to(device)
        peft_model.load_state_dict(state)
        print(f"    [best] Restored best adapter (MRR={self.best_mrr:.4f})")
        return self.best_mrr


# ── Model builder ──────────────────────────────────────────────────────────────

def build_model(config: dict):
    from sentence_transformers import SentenceTransformer
    from peft import LoraConfig, TaskType, get_peft_model

    model_kwargs = config.get("model_kwargs") or {}
    model = SentenceTransformer(config["model_name"], model_kwargs=model_kwargs)
    model.max_seq_length = config["max_seq_length"]

    lora_cfg = config["lora"]
    model[0].auto_model = get_peft_model(
        model[0].auto_model,
        LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        ),
    )
    model[0].auto_model.print_trainable_parameters()
    return model


# ── Fold generation ────────────────────────────────────────────────────────────

def make_folds(moral_texts: list[str], n_folds: int, seed: int) -> list[dict]:
    from sklearn.model_selection import GroupKFold

    text_to_group: dict[str, int] = {}
    groups = []
    for t in moral_texts:
        if t not in text_to_group:
            text_to_group[t] = len(text_to_group)
        groups.append(text_to_group[t])

    groups  = np.array(groups)
    indices = np.arange(len(moral_texts))
    folds   = []
    for fold_i, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=n_folds).split(indices, groups=groups)
    ):
        folds.append({"fold": fold_i, "train": train_idx.tolist(), "test": test_idx.tolist()})
    return folds


# ── Single fold ────────────────────────────────────────────────────────────────

def run_fold(
    fold:          dict,
    moral_texts:   list[str],
    doc_texts:     list[str],
    ground_truth:  dict[int, int],
    sim_matrix:    torch.Tensor,
    config:        dict,
    force:         bool,
    use_wandb:     bool,
) -> dict:
    fold_idx  = fold["fold"]
    train_idx = fold["train"]
    test_idx  = fold["test"]
    n_train, n_test = len(train_idx), len(test_idx)
    print(f"\n  Fold {fold_idx + 1}/{config['n_folds']}  train={n_train}  test={n_test}")

    instruction  = config.get("query_instruction", "")
    train_morals = [f"{instruction}{moral_texts[i]}" for i in train_idx]
    train_docs   = [doc_texts[ground_truth[i]]       for i in train_idx]
    test_morals  = [f"{instruction}{moral_texts[i]}" for i in test_idx]
    test_gt      = {j: ground_truth[i] for j, i in enumerate(test_idx)}

    _model_root    = Path(config["model_output_dir"])
    model_cache    = _model_root / f"fold_{fold_idx}"
    checkpoint_dir = CACHE_DIR / "checkpoints" / f"fold_{fold_idx}"
    emb_cache      = CACHE_DIR / "embeddings"   / f"fold_{fold_idx}"

    best_metric_key = f"eval_fold_{fold_idx}_cosine_mrr@10"

    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    evaluator = InformationRetrievalEvaluator(
        queries={str(j): q for j, q in enumerate(test_morals)},
        corpus={str(i): d for i, d in enumerate(doc_texts)},
        relevant_docs={str(j): {str(test_gt[j])} for j in range(n_test)},
        mrr_at_k=[10], ndcg_at_k=[10], accuracy_at_k=[1, 5, 10],
        name=f"fold_{fold_idx}",
    )

    if use_wandb:
        wandb.init(
            project=config["wandb"]["project"],
            name=f"ft_09_fold_{fold_idx}",
            group="ft_09_fn_masking",
            tags=["ft_09_fn_masking", f"fold_{fold_idx}"],
            config={k: v for k, v in config.items() if k != "wandb"},
        )

    if model_cache.exists() and not force:
        from sentence_transformers import SentenceTransformer
        model_kwargs = config.get("model_kwargs") or {}
        print(f"    [cache hit] ← {model_cache}")
        model = SentenceTransformer(str(model_cache), model_kwargs=model_kwargs)
    else:
        import shutil
        from datasets import Dataset
        from sentence_transformers.trainer import SentenceTransformerTrainer
        from sentence_transformers.training_args import SentenceTransformerTrainingArguments
        from transformers import EarlyStoppingCallback
        from transformers.trainer_utils import get_last_checkpoint

        if force and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        checkpoint_to_resume = None
        if checkpoint_dir.exists():
            last = get_last_checkpoint(str(checkpoint_dir))
            if last:
                checkpoint_to_resume = last
                print(f"    [resume] ← {checkpoint_to_resume}")

        model      = build_model(config)
        # Pass corpus-level moral indices as labels so the loss can look up
        # the precomputed sim matrix for each in-batch pair.
        loss = MaskedInfoNCELoss(
            model,
            temperature=config["temperature"],
            moral_sim_matrix=sim_matrix,
            fn_threshold=config["fn_mask_threshold"],
        )
        adapter_cb   = BestAdapterCallback(model[0].auto_model, best_metric_key)
        steps_per_ep = max(1, n_train // config["batch_size"])

        dataset = Dataset.from_dict({
            "anchor":   train_morals,
            "positive": train_docs,
            "label":    train_idx,   # corpus-level moral indices (0..708)
        })

        callbacks = [
            adapter_cb,
            EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"]),
        ]
        import os
        if os.getenv("TG_BOT_TOKEN") and os.getenv("TG_CHAT_ID"):
            from finetuning.lib.notify import TelegramCallback
            callbacks.append(TelegramCallback(label=f"ft_09/fold_{fold_idx}"))

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer_args = SentenceTransformerTrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            gradient_checkpointing=config["gradient_checkpointing"],
            learning_rate=float(config["learning_rate"]),
            seed=config["seed"],
            save_strategy="epoch",
            save_total_limit=2,
            eval_strategy="epoch",
            load_best_model_at_end=False,
            metric_for_best_model=best_metric_key,
            greater_is_better=True,
            dataloader_pin_memory=False,
            logging_steps=max(1, steps_per_ep // 2),
            report_to="wandb" if (use_wandb and wandb.run is not None) else "none",
        )

        SentenceTransformerTrainer(
            model=model, args=trainer_args,
            train_dataset=dataset, evaluator=evaluator,
            loss=loss, callbacks=callbacks,
        ).train(resume_from_checkpoint=checkpoint_to_resume)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        adapter_cb.restore(model[0].auto_model)
        model[0].auto_model = model[0].auto_model.merge_and_unload()
        model_cache.mkdir(parents=True, exist_ok=True)
        model.save(str(model_cache))
        print(f"    [saved] → {model_cache}")

        if use_wandb and wandb.run is not None:
            wandb.finish()

    metrics = evaluate(model, test_morals, doc_texts, test_gt,
                       cache_dir=emb_cache, force=force)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"  MRR={metrics['MRR']:.4f}  "
        f"R@1={metrics['Recall@1']:.4f}  "
        f"R@5={metrics['Recall@5']:.4f}  "
        f"R@10={metrics['Recall@10']:.4f}"
    )
    result = {"fold": fold_idx, "n_train": n_train, "n_test": n_test, **metrics}
    # Save per-fold result so the parent subprocess-orchestrator can collect it
    RESULTS_DIR.mkdir(exist_ok=True)
    fold_out = RESULTS_DIR / f"fold_{fold_idx}.json"
    with open(fold_out, "w") as f:
        json.dump(result, f, indent=2)
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ft_09 false-negative masking")
    parser.add_argument("--fold",     type=int, default=None)
    parser.add_argument("--force",    action="store_true")
    parser.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    use_wandb = config.get("wandb", {}).get("enabled", False) and not args.no_wandb

    print(f"\n[ft_09_fn_masking]  threshold={config['fn_mask_threshold']}  doc_mode={config['doc_mode']}")

    moral_texts, doc_texts, ground_truth = load_pairs(config["doc_mode"])
    print(f"  Queries: {len(moral_texts)}  Docs: {len(doc_texts)}")

    sim_matrix = load_or_build_sim_matrix(moral_texts)
    print(f"  Sim matrix: {sim_matrix.shape}  threshold={config['fn_mask_threshold']}")

    folds = make_folds(moral_texts, config["n_folds"], config["seed"])
    if args.fold is not None:
        folds = [f for f in folds if f["fold"] == args.fold]

    notify.send(
        f"🚀 ft_09 starting\n"
        f"folds: {[f['fold'] for f in folds]}  queries: {len(moral_texts)}\n"
        f"fn_threshold: {config['fn_mask_threshold']}  doc_mode: {config['doc_mode']}\n"
        f"model: Linq+LoRA  lr: {config['learning_rate']}  τ: {config['temperature']}"
    )

    # When running multiple folds, execute each as a fresh subprocess so Linq's
    # 7B weights are fully released from GPU between folds (avoids OOM).
    if args.fold is None and len(folds) > 1:
        import subprocess
        extra = ["--force"] if args.force else []
        if not use_wandb:
            extra.append("--no-wandb")
        for fold in folds:
            print(f"\n  [subprocess] Launching fold {fold['fold']} as fresh process...")
            subprocess.run(
                [sys.executable, __file__, "--fold", str(fold["fold"])] + extra,
                check=True,
            )
        # Collect results saved by each subprocess fold run
        fold_results = []
        for result_file in sorted(RESULTS_DIR.glob("*_fold_*.json")):
            with open(result_file) as f:
                fold_results.append(json.load(f))
    else:
        fold_results = []
        for fold in folds:
            result = run_fold(
                fold, moral_texts, doc_texts, ground_truth,
                sim_matrix, config, args.force, use_wandb,
            )
            fold_results.append(result)

    mrrs = [r["MRR"] for r in fold_results]
    mean_mrr, std_mrr = float(np.mean(mrrs)), float(np.std(mrrs))
    print(f"\n  ── ft_09 results ──")
    print(f"  MRR@10:  {mean_mrr:.4f} ± {std_mrr:.4f}")
    for r in fold_results:
        print(f"  fold {r['fold']}: MRR={r['MRR']:.4f}  R@1={r['Recall@1']:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = RESULTS_DIR / f"{ts}_fn_masking.json"
    with open(out, "w") as f:
        json.dump({
            "fn_mask_threshold": config["fn_mask_threshold"],
            "n_queries":         len(moral_texts),
            "doc_mode":          config["doc_mode"],
            "config":            config,
            "fold_results":      fold_results,
            "mean_mrr":          mean_mrr,
            "std_mrr":           std_mrr,
        }, f, indent=2)
    print(f"  Results → {out}")

    fold_mrrs = [f"{r['MRR']:.4f}" for r in fold_results]
    notify.send(
        f"✅ ft_09 done\n"
        f"MRR@10: {mean_mrr:.4f} ± {std_mrr:.4f}\n"
        f"folds: {fold_mrrs}\n"
        f"fn_threshold: {config['fn_mask_threshold']}"
    )


if __name__ == "__main__":
    main()
