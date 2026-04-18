"""
ft_01_5fold_cv — 5-fold cross-validation on MORABLES.

Trains on ~567 pairs per fold, evaluates against the full 709-fable corpus on
~142 held-out morals. Reports per-fold and mean ± std MRR.

Each fold gets its own wandb run (clean loss curves per fold), grouped under
ft_01_5fold_cv/<doc_mode> for easy comparison in the wandb dashboard.

Usage
-----
    # Generate splits first (one-time):
    python finetuning/ft_01_5fold_cv/prepare_data.py

    # Run all 5 folds (raw fables):
    python finetuning/ft_01_5fold_cv/train.py

    # Run with fable+summary documents:
    python finetuning/ft_01_5fold_cv/train.py --doc_mode fable_plus_summary

    # Run a single fold (e.g. to test quickly):
    python finetuning/ft_01_5fold_cv/train.py --fold 0

    # Re-train even if cached models exist:
    python finetuning/ft_01_5fold_cv/train.py --force

    # Disable wandb logging:
    python finetuning/ft_01_5fold_cv/train.py --no-wandb
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
SPLITS_PATH = CACHE_DIR / "splits" / "folds.json"
CONFIG_PATH = EXP_DIR / "config.yaml"

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

    train_morals = [moral_texts[i] for i in train_idx]
    train_docs = [doc_texts[ground_truth[i]] for i in train_idx]

    # Test morals retrieve from the FULL 709-fable corpus — not just the test fold
    test_morals = [moral_texts[i] for i in test_idx]
    test_gt = {j: ground_truth[i] for j, i in enumerate(test_idx)}

    model_cache = CACHE_DIR / "models" / config["doc_mode"] / f"fold_{fold_idx}"
    checkpoint_dir = CACHE_DIR / "checkpoints" / config["doc_mode"] / f"fold_{fold_idx}"
    emb_cache = CACHE_DIR / "embeddings" / config["doc_mode"] / f"fold_{fold_idx}"

    # Build an evaluator so we can track MRR after each epoch and restore the
    # best-epoch model at the end rather than always using the final epoch.
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
            group=f"ft_01_5fold_cv/{config['doc_mode']}",
            tags=["ft_01_5fold_cv", config["doc_mode"], f"fold_{fold_idx}"],
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

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold CV fine-tuning on MORABLES")
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

    print(
        f"\n[ft_01_5fold_cv]  "
        f"model={config['model_name']}  "
        f"doc_mode={config['doc_mode']}  "
        f"epochs={config['epochs']}  "
        f"wandb={'on' if use_wandb else 'off'}"
    )
    notify.send(
        f"🚀 ft_01_5fold_cv starting\n"
        f"model: {config['model_name']}\n"
        f"doc_mode: {config['doc_mode']}  epochs: {config['epochs']}\n"
        f"folds: {fold_indices if args.fold is not None else 'all 5'}"
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

    # Log fold summary as a wandb Table for easy dashboard comparison
    if use_wandb and len(fold_metrics) > 1:
        wandb.init(
            project=wandb_cfg["project"],
            name=f"summary_{config['doc_mode']}",
            group=f"ft_01_5fold_cv/{config['doc_mode']}",
            tags=["ft_01_5fold_cv", config["doc_mode"], "summary"],
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
