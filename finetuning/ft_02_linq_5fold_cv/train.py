"""
ft_02_linq_5fold_cv — 5-fold cross-validation fine-tuning Linq-Embed-Mistral on MORABLES.

Fine-tunes the same model used in the retrieval pipeline (Linq-Embed-Mistral)
with LoRA adapters, using the identical 5-fold splits as ft_01 for a fair
comparison. Early stopping prevents overfitting on the small 567-pair training
set by halting when per-epoch MRR stops improving.

Key differences vs ft_01:
  - Model: Linq-Embed-Mistral (7B) vs bge-base-en-v1.5 (110M)
  - LoRA: only ~1% of parameters are trained (adapter fine-tuning)
  - Query instruction: prepended to moral texts for encoding
  - Early stopping: training halts when MRR plateaus (patience=3)
  - Shared splits: uses ft_01's folds.json for direct comparison

Usage
-----
    # Run all 5 folds:
    python finetuning/ft_02_linq_5fold_cv/train.py

    # Run with fable+summary documents:
    python finetuning/ft_02_linq_5fold_cv/train.py --doc_mode fable_plus_summary

    # Run a single fold (e.g. to test quickly):
    python finetuning/ft_02_linq_5fold_cv/train.py --fold 0

    # Re-train even if cached models exist:
    python finetuning/ft_02_linq_5fold_cv/train.py --force

    # Disable wandb logging:
    python finetuning/ft_02_linq_5fold_cv/train.py --no-wandb
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import wandb
import yaml

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from finetuning.lib.data import load_pairs
from finetuning.lib.eval import evaluate
from finetuning.lib.trainer import train_model

CACHE_DIR = EXP_DIR / "cache"
RESULTS_DIR = EXP_DIR / "results"
CONFIG_PATH = EXP_DIR / "config.yaml"

# Reuse ft_01's splits for a fair apples-to-apples comparison.
SPLITS_PATH = EXP_DIR.parent / "ft_01_5fold_cv" / "cache" / "splits" / "folds.json"

_EVAL_KEYS = ("MRR", "Recall@1", "Recall@5", "Recall@10")


def run_fold(
    fold: dict,
    moral_texts: list[str],
    doc_texts: list[str],
    ground_truth: dict[int, int],
    config: dict,
    force: bool,
    use_wandb: bool,
) -> dict:
    fold_idx = fold["fold"]
    train_idx, test_idx = fold["train"], fold["test"]
    print(f"\n  Fold {fold_idx + 1}/5  train={len(train_idx)}  test={len(test_idx)}")

    instruction = config.get("query_instruction", "")

    # Prepend the task instruction to anchor texts so the model encodes them
    # as retrieval queries, matching the instruction format it was trained on.
    train_morals = [f"{instruction}{moral_texts[i]}" for i in train_idx]
    train_docs = [doc_texts[ground_truth[i]] for i in train_idx]

    # Test morals retrieve from the FULL 709-fable corpus — not just the test fold.
    test_morals_raw = [moral_texts[i] for i in test_idx]
    test_morals = [f"{instruction}{t}" for t in test_morals_raw]
    test_gt = {j: ground_truth[i] for j, i in enumerate(test_idx)}

    model_cache = CACHE_DIR / "models" / config["doc_mode"] / f"fold_{fold_idx}"
    checkpoint_dir = CACHE_DIR / "checkpoints" / config["doc_mode"] / f"fold_{fold_idx}"
    emb_cache = CACHE_DIR / "embeddings" / config["doc_mode"] / f"fold_{fold_idx}"

    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    evaluator = InformationRetrievalEvaluator(
        queries={str(j): text for j, text in enumerate(test_morals)},
        corpus={str(i): text for i, text in enumerate(doc_texts)},
        relevant_docs={str(j): {str(test_gt[j])} for j in range(len(test_morals))},
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        name=f"fold_{fold_idx}",
    )

    if use_wandb:
        wandb.init(
            project=config["wandb"]["project"],
            name=f"fold_{fold_idx}_{config['doc_mode']}",
            group=f"ft_02_linq_5fold_cv/{config['doc_mode']}",
            tags=["ft_02_linq_5fold_cv", config["doc_mode"], f"fold_{fold_idx}"],
            config={k: v for k, v in config.items() if k != "wandb"},
        )

    model = train_model(
        train_morals=train_morals,
        train_docs=train_docs,
        config=config,
        model_cache=model_cache,
        checkpoint_dir=checkpoint_dir,
        evaluator=evaluator,
        force=force,
    )

    # evaluate() uses the same instructed morals as queries; doc_texts are plain fables
    metrics = evaluate(
        model, test_morals, doc_texts, test_gt,
        cache_dir=emb_cache,
        force=force,
    )

    print(
        f"  MRR={metrics['MRR']:.4f}  "
        f"R@1={metrics['Recall@1']:.4f}  "
        f"R@5={metrics['Recall@5']:.4f}  "
        f"R@10={metrics['Recall@10']:.4f}"
    )
    if use_wandb:
        wandb.log({
            "fold": fold_idx,
            **{f"eval/{k.lower().replace('@', '_at_')}": metrics[k] for k in _EVAL_KEYS},
        })
        wandb.finish()

    # Explicitly free GPU memory before the next fold loads a fresh 7B model.
    # Without this, the 3090's 24GB fills up and the next fold crashes with OOM.
    import gc
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold CV fine-tuning Linq-Embed-Mistral on MORABLES")
    parser.add_argument("--doc_mode", choices=["raw", "fable_plus_summary"], help="Override config doc_mode")
    parser.add_argument("--fold", type=int, choices=range(5), metavar="0-4", help="Run a single fold instead of all 5")
    parser.add_argument("--force", action="store_true", help="Re-train and re-encode even if cached")
    parser.add_argument("--no-wandb", dest="no_wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    if args.doc_mode:
        config["doc_mode"] = args.doc_mode

    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"Splits not found: {SPLITS_PATH}\n"
            "Run first:  python finetuning/ft_01_5fold_cv/prepare_data.py"
        )
    with open(SPLITS_PATH) as f:
        all_folds = json.load(f)

    wandb_cfg = config.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", False) and not args.no_wandb

    lora_r = config.get("lora", {}).get("r", "none")
    print(
        f"\n[ft_02_linq_5fold_cv]  "
        f"model={config['model_name']}  "
        f"doc_mode={config['doc_mode']}  "
        f"epochs={config['epochs']}  "
        f"LoRA r={lora_r}  "
        f"early_stop={config.get('early_stopping_patience', 'off')}  "
        f"wandb={'on' if use_wandb else 'off'}"
    )

    notify.send(
        f"🚀 ft_02_linq_5fold_cv starting\n"
        f"model: {config['model_name']}\n"
        f"doc_mode: {config['doc_mode']}  epochs: {config['epochs']}  LoRA r={lora_r}\n"
        f"folds: {[args.fold] if args.fold is not None else 'all 5'}"
    )

    moral_texts, doc_texts, ground_truth = load_pairs(config["doc_mode"])

    fold_indices = [args.fold] if args.fold is not None else list(range(len(all_folds)))
    fold_metrics = [
        run_fold(all_folds[i], moral_texts, doc_texts, ground_truth, config, args.force, use_wandb)
        for i in fold_indices
    ]

    # ── Aggregate summary ─────────────────────────────────────────────────────
    mrr_scores = [m["MRR"] for m in fold_metrics]
    mean_mrr = float(np.mean(mrr_scores))
    std_mrr = float(np.std(mrr_scores))
    print(f"\n  Final MRR: {mean_mrr:.4f} ± {std_mrr:.4f}")

    if use_wandb and len(fold_metrics) > 1:
        wandb.init(
            project=wandb_cfg["project"],
            name=f"summary_{config['doc_mode']}",
            group=f"ft_02_linq_5fold_cv/{config['doc_mode']}",
            tags=["ft_02_linq_5fold_cv", config["doc_mode"], "summary"],
            config={k: v for k, v in config.items() if k != "wandb"},
        )
        table = wandb.Table(columns=["fold", "MRR", "R@1", "R@5", "R@10"])
        for i, m in zip(fold_indices, fold_metrics):
            table.add_data(i, m["MRR"], m["Recall@1"], m["Recall@5"], m["Recall@10"])
        wandb.log({
            "folds": table,
            "summary/mean_mrr": mean_mrr,
            "summary/std_mrr": std_mrr,
        })
        wandb.finish()

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"fold{args.fold}" if args.fold is not None else "all_folds"
    out = RESULTS_DIR / f"{ts}_{config['doc_mode']}_{suffix}.json"
    with open(out, "w") as f:
        json.dump({
            "config": config,
            "folds_run": fold_indices,
            "mean_mrr": mean_mrr,
            "std_mrr": std_mrr,
            "fold_mrrs": mrr_scores,
            "fold_metrics": fold_metrics,
        }, f, indent=2)
    print(f"  Results → {out}")


if __name__ == "__main__":
    main()
