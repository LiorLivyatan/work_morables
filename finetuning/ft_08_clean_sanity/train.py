"""
ft_08_clean_sanity — Linq+LoRA fine-tuned on MORABLES, 5-fold CV.

Sanity check for annotation noise: run once with --clean (615 unambiguous queries)
and once without (all 709, noisy baseline). The MRR difference directly measures
how much duplicate/ambiguous annotations suppress the reported performance.

Both runs use the same model (Linq+LoRA, ft_07 hyperparams) and the same evaluation
setup (test morals retrieve from the full 709-fable corpus). The only thing that
changes is which queries are used for training and evaluation.

Usage
-----
    # Clean variant (sanity check — removes 94 ambiguous queries):
    ./run.sh finetuning/ft_08_clean_sanity/train.py --clean --remote --gpu 2

    # Noisy baseline (all 709 queries — apples-to-apples comparison):
    ./run.sh finetuning/ft_08_clean_sanity/train.py --remote --gpu 2

    # Single fold (quick test):
    ./run.sh finetuning/ft_08_clean_sanity/train.py --clean --fold 0 --remote --gpu 2

    # Force re-train:
    ./run.sh finetuning/ft_08_clean_sanity/train.py --clean --force --remote --gpu 2
"""
import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

EXP_DIR     = Path(__file__).parent
ROOT        = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from transformers import TrainerCallback

from finetuning.lib import notify
from finetuning.lib.data import load_pairs, load_clean_pairs
from finetuning.lib.eval import evaluate
from finetuning.lib.losses import InfoNCELoss

CACHE_DIR   = EXP_DIR / "cache"
RESULTS_DIR = EXP_DIR / "results"
CONFIG_PATH = EXP_DIR / "config.yaml"


# ── Best adapter callback (saves LoRA weights to CPU on each new best MRR) ────

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
    """GroupKFold on moral text — keeps identical morals in the same fold."""
    from sklearn.model_selection import GroupKFold

    n = len(moral_texts)
    # group id = unique moral text index (prevents same moral leaking across folds)
    text_to_group: dict[str, int] = {}
    groups = []
    for t in moral_texts:
        if t not in text_to_group:
            text_to_group[t] = len(text_to_group)
        groups.append(text_to_group[t])

    groups = np.array(groups)
    indices = np.arange(n)

    folds = []
    for fold_i, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=n_folds).split(indices, groups=groups)
    ):
        folds.append({
            "fold":  fold_i,
            "train": train_idx.tolist(),
            "test":  test_idx.tolist(),
        })
    return folds


# ── Single fold training + evaluation ─────────────────────────────────────────

def run_fold(
    fold:         dict,
    moral_texts:  list[str],
    doc_texts:    list[str],
    ground_truth: dict[int, int],
    config:       dict,
    variant:      str,
    force:        bool,
    use_wandb:    bool,
) -> dict:
    fold_idx  = fold["fold"]
    train_idx = fold["train"]
    test_idx  = fold["test"]
    n_train, n_test = len(train_idx), len(test_idx)
    print(f"\n  Fold {fold_idx + 1}/{config['n_folds']}  train={n_train}  test={n_test}")

    instruction = config.get("query_instruction", "")

    train_morals = [f"{instruction}{moral_texts[i]}" for i in train_idx]
    train_docs   = [doc_texts[ground_truth[i]]       for i in train_idx]
    test_morals  = [f"{instruction}{moral_texts[i]}" for i in test_idx]
    test_gt      = {j: ground_truth[i] for j, i in enumerate(test_idx)}

    _model_root    = Path(config["model_output_dir"]) / variant
    model_cache    = _model_root / f"fold_{fold_idx}"
    checkpoint_dir = CACHE_DIR / "checkpoints" / variant / f"fold_{fold_idx}"
    emb_cache      = CACHE_DIR / "embeddings"   / variant / f"fold_{fold_idx}"

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
            name=f"{variant}_fold_{fold_idx}",
            group=f"ft_08_clean_sanity/{variant}",
            tags=["ft_08_clean_sanity", variant, f"fold_{fold_idx}"],
            config={k: v for k, v in config.items() if k != "wandb"},
        )

    # ── Train ────────────────────────────────────────────────────────────────
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

        model        = build_model(config)
        loss         = InfoNCELoss(model, temperature=config["temperature"])
        adapter_cb   = BestAdapterCallback(model[0].auto_model, best_metric_key)
        steps_per_ep = max(1, n_train // config["batch_size"])

        labels = list(range(n_train))
        dataset = Dataset.from_dict({
            "anchor": train_morals, "positive": train_docs, "label": labels,
        })

        callbacks = [
            adapter_cb,
            EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"]),
        ]
        import os
        if os.getenv("TG_BOT_TOKEN") and os.getenv("TG_CHAT_ID"):
            from finetuning.lib.notify import TelegramCallback
            callbacks.append(TelegramCallback(label=f"ft_08/{variant}/fold_{fold_idx}"))

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

    # ── Evaluate ─────────────────────────────────────────────────────────────
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
    return {"fold": fold_idx, "n_train": n_train, "n_test": n_test, **metrics}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ft_08 clean sanity check")
    parser.add_argument("--clean",    action="store_true",
                        help="Use clean 615-query subset (removes 94 ambiguous queries)")
    parser.add_argument("--fold",     type=int, default=None,
                        help="Run a single fold (0-based). Default: run all 5.")
    parser.add_argument("--force",    action="store_true", help="Re-train even if cached")
    parser.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    variant   = "clean" if args.clean else "noisy"
    use_wandb = config.get("wandb", {}).get("enabled", False) and not args.no_wandb

    print(f"\n[ft_08_clean_sanity]  variant={variant}  doc_mode={config['doc_mode']}")

    # ── Load data ──────────────────────────────────────────────────────────────
    if args.clean:
        from finetuning.lib.data import load_clean_pairs
        moral_texts, doc_texts, ground_truth, kept = load_clean_pairs(config["doc_mode"])
        print(f"  Queries: {len(moral_texts)} clean (removed {709 - len(moral_texts)} ambiguous)")
    else:
        moral_texts, doc_texts, ground_truth = load_pairs(config["doc_mode"])
        print(f"  Queries: {len(moral_texts)} (full noisy dataset)")

    # ── Folds ──────────────────────────────────────────────────────────────────
    folds = make_folds(moral_texts, config["n_folds"], config["seed"])
    if args.fold is not None:
        folds = [f for f in folds if f["fold"] == args.fold]

    # ── Notify start ──────────────────────────────────────────────────────────
    notify.send(
        f"🚀 ft_08 starting\n"
        f"variant: {variant}  folds: {[f['fold'] for f in folds]}\n"
        f"queries: {len(moral_texts)}  doc_mode: {config['doc_mode']}\n"
        f"model: Linq+LoRA  lr: {config['learning_rate']}  τ: {config['temperature']}"
    )

    # ── Run folds ──────────────────────────────────────────────────────────────
    fold_results = []
    for fold in folds:
        result = run_fold(
            fold, moral_texts, doc_texts, ground_truth,
            config, variant, args.force, use_wandb,
        )
        fold_results.append(result)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    mrrs = [r["MRR"] for r in fold_results]
    mean_mrr, std_mrr = float(np.mean(mrrs)), float(np.std(mrrs))
    print(f"\n  ── {variant} results ──")
    print(f"  MRR@10:  {mean_mrr:.4f} ± {std_mrr:.4f}")
    for r in fold_results:
        print(f"  fold {r['fold']}: MRR={r['MRR']:.4f}  R@1={r['Recall@1']:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = RESULTS_DIR / f"{ts}_{variant}.json"
    with open(out, "w") as f:
        json.dump({
            "variant":   variant,
            "n_queries": len(moral_texts),
            "doc_mode":  config["doc_mode"],
            "config":    config,
            "fold_results": fold_results,
            "mean_mrr":  mean_mrr,
            "std_mrr":   std_mrr,
        }, f, indent=2)
    print(f"  Results → {out}")

    fold_mrrs = [f"{r['MRR']:.4f}" for r in fold_results]
    notify.send(
        f"✅ ft_08 done  variant={variant}\n"
        f"MRR@10: {mean_mrr:.4f} ± {std_mrr:.4f}\n"
        f"folds: {fold_mrrs}"
    )


if __name__ == "__main__":
    main()
