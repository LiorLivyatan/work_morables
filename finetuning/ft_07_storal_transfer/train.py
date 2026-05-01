"""
ft_07_storal_transfer — Cross-domain transfer: train on STORAL, evaluate on MORABLES.

Unlike ft_01–06 (which train on MORABLES with 5-fold CV), this experiment uses STORAL
(1,675 clean moral-story pairs, Aesop duplicates removed) as the only training data and
evaluates on all 709 MORABLES fables. Tests whether out-of-domain moral-story training
signal transfers to Aesop fable retrieval.

Three models (bge, linq, qwen3) × three dataset sizes (s500, s1000, sfull) = 9 runs.
Length-matched sampling for s500/s1000 ensures stories are similar in length to MORABLES.

Usage
-----
    # BGE-base, full dataset:
    ./run.sh finetuning/ft_07_storal_transfer/train.py --model bge --size sfull

    # Linq, 500 length-matched stories:
    ./run.sh finetuning/ft_07_storal_transfer/train.py --model linq --size s500

    # Qwen3-8B, 1000 stories, remote GPU 1:
    ./run.sh finetuning/ft_07_storal_transfer/train.py --model qwen3 --size s1000 --remote --gpu 1

    # All sizes for one model (sequential):
    for SIZE in s500 s1000 sfull; do
      ./run.sh finetuning/ft_07_storal_transfer/train.py --model bge --size $SIZE --remote --gpu 1
    done
"""
import argparse
import gc
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from transformers import TrainerCallback

from finetuning.lib import notify
from finetuning.lib.data import load_pairs
from finetuning.lib.losses import InfoNCELoss

CACHE_DIR   = EXP_DIR / "cache"
RESULTS_DIR = EXP_DIR / "results"
CONFIG_PATH = EXP_DIR / "config.yaml"
STORAL_PATH = ROOT / "data/external/storal/processed/storal_pairs.json"


# ── STORAL data loading & sampling ────────────────────────────────────────────

def load_storal(size_cfg: dict, seed: int) -> list[dict]:
    """
    Load clean STORAL pairs, optionally filtered by story length and down-sampled.

    Args:
        size_cfg  dataset_sizes entry: {n, length_range}
        seed      RNG seed for reproducible sub-sampling

    Returns:
        list of {moral, story} dicts
    """
    all_pairs = json.loads(STORAL_PATH.read_text())
    clean = [p for p in all_pairs if not p.get("is_duplicate")]

    length_range = size_cfg.get("length_range")
    if length_range:
        lo, hi = length_range
        clean = [p for p in clean if lo <= len(p["story"].split()) <= hi]

    n = size_cfg.get("n")
    if n is not None and len(clean) > n:
        rng = random.Random(seed)
        clean = rng.sample(clean, n)

    return clean


# ── Model building ────────────────────────────────────────────────────────────

class BestAdapterCallback(TrainerCallback):
    """
    Saves LoRA adapter weights to CPU RAM at each new best MRR.
    Avoids the OOM that load_best_model_at_end=True causes on 24 GB cards.
    """
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


def build_model(model_cfg: dict):
    from sentence_transformers import SentenceTransformer

    st_kwargs: dict = {}
    if model_cfg.get("trust_remote_code"):
        st_kwargs["trust_remote_code"] = True
    model_kwargs = model_cfg.get("model_kwargs") or {}
    if model_kwargs:
        st_kwargs["model_kwargs"] = model_kwargs

    model = SentenceTransformer(model_cfg["model_name"], **st_kwargs)
    if model_cfg.get("max_seq_length"):
        model.max_seq_length = model_cfg["max_seq_length"]

    lora_cfg = model_cfg.get("lora")
    if lora_cfg:
        from peft import LoraConfig, TaskType, get_peft_model
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


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model,
    train_dataset,
    evaluator,
    config: dict,
    model_cfg: dict,
    checkpoint_dir: Path,
    model_cache: Path,
    run_tag: str,
    force: bool,
    use_wandb: bool,
):
    import shutil
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from transformers import EarlyStoppingCallback
    from transformers.trainer_utils import get_last_checkpoint

    if force and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    checkpoint_to_resume = None
    if checkpoint_dir.exists():
        last_ckpt = get_last_checkpoint(str(checkpoint_dir))
        if last_ckpt:
            checkpoint_to_resume = last_ckpt
            print(f"    [resume] ← {checkpoint_to_resume}")

    τ = config.get("temperature", 0.05)
    loss = InfoNCELoss(model, temperature=τ)
    lora_cfg = model_cfg.get("lora")

    steps_per_epoch  = max(1, len(train_dataset) // model_cfg["batch_size"])
    best_metric_key  = "eval_morables_cosine_mrr@10"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=model_cfg["batch_size"],
        gradient_accumulation_steps=model_cfg.get("gradient_accumulation_steps", 1),
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
        learning_rate=float(model_cfg["learning_rate"]),
        seed=config["seed"],
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model=best_metric_key,
        greater_is_better=True,
        dataloader_pin_memory=False,
        logging_steps=max(1, steps_per_epoch // 2),
        report_to="wandb" if (use_wandb and wandb.run is not None) else "none",
    )

    callbacks = []
    if config.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"]))

    best_adapter_cb = None
    if lora_cfg:
        best_adapter_cb = BestAdapterCallback(model[0].auto_model, best_metric_key)
        callbacks.append(best_adapter_cb)

    import os
    if os.getenv("TG_BOT_TOKEN") and os.getenv("TG_CHAT_ID"):
        from finetuning.lib.notify import TelegramCallback
        callbacks.append(TelegramCallback(label=f"ft_07/{run_tag}"))

    SentenceTransformerTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        evaluator=evaluator,
        loss=loss,
        callbacks=callbacks or None,
    ).train(resume_from_checkpoint=checkpoint_to_resume)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if best_adapter_cb is not None:
        best_adapter_cb.restore(model[0].auto_model)

    if lora_cfg:
        model[0].auto_model = model[0].auto_model.merge_and_unload()

    model_cache.mkdir(parents=True, exist_ok=True)
    model.save(str(model_cache))
    print(f"    [saved] → {model_cache}")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_on_morables(model, instruction: str, force: bool, cache_base: Path) -> dict:
    """Evaluate on all 709 MORABLES fables for both doc modes."""
    from finetuning.lib.eval import evaluate

    results = {}
    for doc_mode in ("raw", "fable_plus_summary"):
        moral_texts, doc_texts, ground_truth = load_pairs(doc_mode)
        query_texts = [f"{instruction}{m}" for m in moral_texts]
        cache_dir = cache_base / doc_mode
        metrics = evaluate(
            model, query_texts, doc_texts, ground_truth,
            cache_dir=cache_dir, force=force,
        )
        results[doc_mode] = metrics
        print(
            f"  [{doc_mode}] MRR={metrics['MRR']:.4f}  "
            f"R@1={metrics['Recall@1']:.4f}  "
            f"R@5={metrics['Recall@5']:.4f}  "
            f"R@10={metrics['Recall@10']:.4f}"
        )
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ft_07 STORAL→MORABLES transfer")
    parser.add_argument("--model", required=True, choices=["bge", "linq", "qwen3"],
                        help="Which model config to use")
    parser.add_argument("--size",  required=True,
                        help="Dataset size key from config (e.g. s500, s1000, sfull)")
    parser.add_argument("--force",    action="store_true", help="Re-train even if cached")
    parser.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    model_cfg = config["models"][args.model]
    size_cfg  = config["dataset_sizes"][args.size]
    instruction = model_cfg.get("query_instruction", "")
    τ = config.get("temperature", 0.05)

    print(
        f"\n[ft_07_storal_transfer]  model={args.model}  size={args.size}  "
        f"τ={τ}  epochs={config['epochs']}"
    )

    # ── Load STORAL training data ──────────────────────────────────────────────
    storal_pairs = load_storal(size_cfg, config["seed"])
    print(
        f"  STORAL: {len(storal_pairs)} pairs  "
        f"(length_range={size_cfg.get('length_range')}, n={size_cfg.get('n')})"
    )

    notify.send(
        f"🚀 ft_07 starting\n"
        f"model: {args.model}  size: {args.size}\n"
        f"storal pairs: {len(storal_pairs)}  τ: {τ}\n"
        f"epochs: {config['epochs']}"
    )

    # ── Build training dataset ─────────────────────────────────────────────────
    from datasets import Dataset

    anchors   = [f"{instruction}{p['moral']}" for p in storal_pairs]
    positives = [p["story"] for p in storal_pairs]
    # All morals are unique in STORAL so each sample is its own group
    labels    = list(range(len(storal_pairs)))
    train_dataset = Dataset.from_dict({
        "anchor": anchors, "positive": positives, "label": labels
    })

    # ── MORABLES evaluator (used during training for early stopping) ───────────
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    # Use fable_plus_summary as the training-time eval corpus (best zero-shot baseline)
    eval_moral_texts, eval_doc_texts, eval_gt = load_pairs("fable_plus_summary")
    eval_queries = [f"{instruction}{m}" for m in eval_moral_texts]
    evaluator = InformationRetrievalEvaluator(
        queries={str(i): q for i, q in enumerate(eval_queries)},
        corpus={str(i): d for i, d in enumerate(eval_doc_texts)},
        relevant_docs={str(i): {str(eval_gt[i])} for i in range(len(eval_queries))},
        mrr_at_k=[10], ndcg_at_k=[10], accuracy_at_k=[1, 5, 10],
        name="morables",
    )

    # ── Paths ──────────────────────────────────────────────────────────────────
    _model_root   = Path(model_cfg["model_output_dir"])
    run_tag       = f"{args.model}_{args.size}"
    model_cache   = _model_root / run_tag
    checkpoint_dir = CACHE_DIR / "checkpoints" / run_tag
    emb_cache     = CACHE_DIR / "embeddings" / run_tag

    use_wandb = config.get("wandb", {}).get("enabled", False) and not args.no_wandb

    # ── Train or load from cache ───────────────────────────────────────────────
    if model_cache.exists() and not args.force:
        from sentence_transformers import SentenceTransformer
        print(f"  [cache hit] Loading model ← {model_cache}")
        st_kwargs: dict = {}
        if model_cfg.get("trust_remote_code"):
            st_kwargs["trust_remote_code"] = True
        model = SentenceTransformer(str(model_cache), **st_kwargs)
    else:
        if use_wandb:
            wandb.init(
                project=config["wandb"]["project"],
                name=run_tag,
                group="ft_07_storal_transfer",
                tags=["ft_07", args.model, args.size],
                config={k: v for k, v in config.items() if k != "wandb"},
            )
        model = build_model(model_cfg)
        model = train(
            model, train_dataset, evaluator, config, model_cfg,
            checkpoint_dir, model_cache, run_tag, args.force, use_wandb,
        )
        if use_wandb and wandb.run is not None:
            wandb.finish()

    # ── Final evaluation on all MORABLES (raw + fable_plus_summary) ───────────
    print(f"\n  Evaluating on all 709 MORABLES fables …")
    doc_mode_results = eval_on_morables(model, instruction, args.force, emb_cache)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Save results ───────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = RESULTS_DIR / f"{ts}_{run_tag}.json"
    with open(out, "w") as f:
        json.dump({
            "model":    args.model,
            "size":     args.size,
            "n_storal": len(storal_pairs),
            "length_range": size_cfg.get("length_range"),
            "tau": τ,
            "config": config,
            "doc_mode_results": doc_mode_results,
        }, f, indent=2)
    print(f"  Results → {out}")

    mrr_raw  = doc_mode_results["raw"]["MRR"]
    mrr_summ = doc_mode_results["fable_plus_summary"]["MRR"]
    notify.send(
        f"✅ ft_07 done\n"
        f"model: {args.model}  size: {args.size}\n"
        f"MRR raw: {mrr_raw:.4f}  MRR fable+summary: {mrr_summ:.4f}"
    )


if __name__ == "__main__":
    main()
