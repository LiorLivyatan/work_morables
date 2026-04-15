"""
ft_02_linq_5fold_cv — standalone evaluation.

Loads cached per-fold models and evaluates each on its held-out test split
against the full 709-fable corpus. Also runs the baseline (untrained
Linq-Embed-Mistral) on every fold for a direct side-by-side comparison.

Use this to:
  - Re-evaluate all folds without re-running training
  - Compare fine-tuned vs baseline across all folds
  - Evaluate a single fold quickly

Usage
-----
    python finetuning/ft_02_linq_5fold_cv/evaluate.py
    python finetuning/ft_02_linq_5fold_cv/evaluate.py --fold 0
    python finetuning/ft_02_linq_5fold_cv/evaluate.py --doc_mode fable_plus_summary
    python finetuning/ft_02_linq_5fold_cv/evaluate.py --wandb
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

from finetuning.lib.data import load_pairs
from finetuning.lib.eval import evaluate

CACHE_DIR = EXP_DIR / "cache"
RESULTS_DIR = EXP_DIR / "results"
CONFIG_PATH = EXP_DIR / "config.yaml"
SPLITS_PATH = EXP_DIR.parent / "ft_01_5fold_cv" / "cache" / "splits" / "folds.json"

_METRICS = ("MRR", "Recall@1", "Recall@5", "Recall@10")
_W = 14


def _divider(width: int = 70) -> str:
    return "  " + "─" * width


def print_fold_table(
    fold_indices: list[int],
    baseline_fold_metrics: list[dict],
    finetuned_fold_metrics: list[dict],
    config: dict,
) -> None:
    print(f"\n{'─' * 70}")
    print(f"  ft_02_linq_5fold_cv  |  doc_mode={config['doc_mode']}  |  LoRA r={config.get('lora', {}).get('r', 'none')}")
    print(_divider())
    print(f"  {'Fold':<6}  {'Baseline MRR':>{_W}}  {'Fine-tuned MRR':>{_W}}  {'Δ MRR'}")
    print(_divider())

    for fold_idx, bm, fm in zip(fold_indices, baseline_fold_metrics, finetuned_fold_metrics):
        delta = fm["MRR"] - bm["MRR"]
        sign = "+" if delta >= 0 else ""
        print(f"  {fold_idx:<6}  {bm['MRR']:{_W}.4f}  {fm['MRR']:{_W}.4f}  {sign}{delta:.4f}")

    if len(fold_indices) > 1:
        print(_divider())
        b_mean = np.mean([m["MRR"] for m in baseline_fold_metrics])
        b_std = np.std([m["MRR"] for m in baseline_fold_metrics])
        f_mean = np.mean([m["MRR"] for m in finetuned_fold_metrics])
        f_std = np.std([m["MRR"] for m in finetuned_fold_metrics])
        delta_mean = f_mean - b_mean
        sign = "+" if delta_mean >= 0 else ""
        print(f"  {'Mean':<6}  {b_mean:{_W}.4f}  {f_mean:{_W}.4f}  {sign}{delta_mean:.4f}")
        print(f"  {'Std':<6}  {b_std:{_W}.4f}  {f_std:{_W}.4f}")

    print(_divider())


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone evaluation for ft_02_linq_5fold_cv")
    parser.add_argument("--doc_mode", choices=["raw", "fable_plus_summary"], help="Override config doc_mode")
    parser.add_argument("--fold", type=int, choices=range(5), metavar="0-4", help="Evaluate a single fold")
    parser.add_argument("--wandb", dest="use_wandb", action="store_true", help="Log results to wandb")
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

    fold_indices = [args.fold] if args.fold is not None else list(range(len(all_folds)))

    missing = [
        i for i in fold_indices
        if not (CACHE_DIR / "models" / config["doc_mode"] / f"fold_{i}").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"No cached model(s) for fold(s) {missing} with doc_mode='{config['doc_mode']}'.\n"
            f"Run training first:  python finetuning/ft_02_linq_5fold_cv/train.py --doc_mode {config['doc_mode']}"
        )

    print(
        f"\n[ft_02_linq_5fold_cv / evaluate]  "
        f"doc_mode={config['doc_mode']}  "
        f"folds={fold_indices}"
    )

    moral_texts, doc_texts, ground_truth = load_pairs(config["doc_mode"])
    instruction = config.get("query_instruction", "")

    from sentence_transformers import SentenceTransformer
    print("\n  Loading baseline (untrained Linq-Embed-Mistral)...")
    baseline_model = SentenceTransformer(config["model_name"], model_kwargs=config.get("model_kwargs") or {})

    baseline_fold_metrics: list[dict] = []
    finetuned_fold_metrics: list[dict] = []

    for fold_idx in fold_indices:
        test_idx = all_folds[fold_idx]["test"]
        test_morals = [f"{instruction}{moral_texts[i]}" for i in test_idx]
        test_gt = {j: ground_truth[i] for j, i in enumerate(test_idx)}

        print(f"\n  Fold {fold_idx + 1}/{len(fold_indices)}  (test={len(test_idx)})")

        print("    Baseline...")
        bm = evaluate(baseline_model, test_morals, doc_texts, test_gt)
        baseline_fold_metrics.append(bm)
        print(f"    Baseline  MRR={bm['MRR']:.4f}  R@1={bm['Recall@1']:.4f}  R@5={bm['Recall@5']:.4f}")

        print("    Fine-tuned (from cache)...")
        model_cache = CACHE_DIR / "models" / config["doc_mode"] / f"fold_{fold_idx}"
        emb_cache = CACHE_DIR / "embeddings" / config["doc_mode"] / f"fold_{fold_idx}"
        # Merged model is a standard SentenceTransformer — no model_kwargs needed
        ft_model = SentenceTransformer(str(model_cache))
        fm = evaluate(ft_model, test_morals, doc_texts, test_gt, cache_dir=emb_cache)
        finetuned_fold_metrics.append(fm)
        print(f"    Fine-tuned MRR={fm['MRR']:.4f}  R@1={fm['Recall@1']:.4f}  R@5={fm['Recall@5']:.4f}")
        del ft_model

    del baseline_model

    print_fold_table(fold_indices, baseline_fold_metrics, finetuned_fold_metrics, config)

    if args.use_wandb:
        wandb_cfg = config.get("wandb", {})
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            project=wandb_cfg.get("project", "morables-finetuning"),
            name=f"eval_5fold_{config['doc_mode']}_{ts}",
            group=f"ft_02_linq_5fold_cv/{config['doc_mode']}",
            tags=["ft_02_linq_5fold_cv", config["doc_mode"], "evaluate"],
            config={k: v for k, v in config.items() if k != "wandb"},
        )
        table = wandb.Table(columns=["fold", "baseline_mrr", "finetuned_mrr", "delta_mrr"])
        for fold_idx, bm, fm in zip(fold_indices, baseline_fold_metrics, finetuned_fold_metrics):
            table.add_data(fold_idx, bm["MRR"], fm["MRR"], fm["MRR"] - bm["MRR"])

        f_mrrs = [m["MRR"] for m in finetuned_fold_metrics]
        b_mrrs = [m["MRR"] for m in baseline_fold_metrics]
        wandb.log({
            "folds": table,
            "summary/mean_finetuned_mrr": float(np.mean(f_mrrs)),
            "summary/std_finetuned_mrr": float(np.std(f_mrrs)),
            "summary/mean_baseline_mrr": float(np.mean(b_mrrs)),
            "summary/mean_delta_mrr": float(np.mean(f_mrrs) - np.mean(b_mrrs)),
        })
        wandb.finish()

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"fold{args.fold}" if args.fold is not None else "all_folds"
    out = RESULTS_DIR / f"{ts}_{config['doc_mode']}_{suffix}_evaluate.json"
    with open(out, "w") as f:
        json.dump({
            "config": config,
            "folds_evaluated": fold_indices,
            "baseline_fold_metrics": baseline_fold_metrics,
            "finetuned_fold_metrics": finetuned_fold_metrics,
            "mean_finetuned_mrr": float(np.mean([m["MRR"] for m in finetuned_fold_metrics])),
            "std_finetuned_mrr": float(np.std([m["MRR"] for m in finetuned_fold_metrics])),
            "mean_baseline_mrr": float(np.mean([m["MRR"] for m in baseline_fold_metrics])),
        }, f, indent=2)
    print(f"\n  Results → {out}")


if __name__ == "__main__":
    main()
