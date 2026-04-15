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
    parser.add_argument("--doc_mode", choices=["raw", "fable_plus_summary"], help="Override config doc_mode (also sets model source unless --train_doc_mode is given)")
    parser.add_argument("--train_doc_mode", choices=["raw", "fable_plus_summary"], help="doc_mode the model was trained on (controls which cached model is loaded); defaults to --doc_mode")
    parser.add_argument("--fold", type=int, choices=range(5), metavar="0-4", help="Evaluate a single fold")
    parser.add_argument("--force", action="store_true", help="Re-encode even if embeddings are cached")
    parser.add_argument("--wandb", dest="use_wandb", action="store_true", help="Log results to wandb")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    if args.doc_mode:
        config["doc_mode"] = args.doc_mode
    # doc_mode used to locate cached models (may differ from eval doc_mode)
    train_doc_mode = args.train_doc_mode or config["doc_mode"]

    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"Splits not found: {SPLITS_PATH}\n"
            "Run first:  python finetuning/ft_01_5fold_cv/prepare_data.py"
        )
    with open(SPLITS_PATH) as f:
        all_folds = json.load(f)

    fold_indices = [args.fold] if args.fold is not None else list(range(len(all_folds)))

    _model_root = Path(config["model_output_dir"]) if config.get("model_output_dir") else CACHE_DIR / "models"
    missing = [
        i for i in fold_indices
        if not (_model_root / train_doc_mode / f"fold_{i}").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"No cached model(s) for fold(s) {missing} with train_doc_mode='{train_doc_mode}'.\n"
            f"Expected at: {_model_root / train_doc_mode}\n"
            f"Run training first:  python finetuning/ft_02_linq_5fold_cv/train.py --doc_mode {train_doc_mode}"
        )

    train_label = f"trained_on={train_doc_mode}"
    eval_label = f"eval_on={config['doc_mode']}"
    cross_eval = train_doc_mode != config["doc_mode"]
    print(
        f"\n[ft_02_linq_5fold_cv / evaluate]  "
        f"{train_label}  {eval_label}  "
        f"{'⚡ cross-eval  ' if cross_eval else ''}"
        f"folds={fold_indices}"
    )

    moral_texts, doc_texts, ground_truth = load_pairs(config["doc_mode"])

    # Fine-tuned model instruction (from config — how it was trained)
    ft_instruction = config.get("query_instruction", "")

    # Baseline instruction matches exp07 exactly (run_all_variants.py):
    # "Instruct: Given a text, retrieve the most relevant passage that answers the query\nQuery: {t}"
    # Using a different instruction for baseline vs FT is intentional — the baseline
    # is an untuned model and should be evaluated the same way exp07 evaluated it.
    BASELINE_INSTRUCTION = "Instruct: Given a text, retrieve the most relevant passage that answers the query\nQuery: "

    import gc
    import torch
    from sentence_transformers import SentenceTransformer

    # Build per-fold query/gt dicts for both baseline and FT (different instructions)
    fold_data = []
    for fold_idx in fold_indices:
        test_idx = all_folds[fold_idx]["test"]
        baseline_morals = [f"{BASELINE_INSTRUCTION}{moral_texts[i]}" for i in test_idx]
        ft_morals = [f"{ft_instruction}{moral_texts[i]}" for i in test_idx]
        test_gt = {j: ground_truth[i] for j, i in enumerate(test_idx)}
        fold_data.append((fold_idx, baseline_morals, ft_morals, test_gt))

    # ── Pass 1: baseline (one model load for all folds, then freed) ───────────
    # Keeping both baseline and FT model in memory simultaneously would require
    # ~28GB (2 × 7B @ bfloat16), exceeding the 3090's 24GB. Run them in two
    # separate passes so only one 7B model is resident at a time.
    baseline_fold_metrics: list[dict] = []
    print("\n  [Pass 1/2] Baseline (untrained Linq-Embed-Mistral)...")
    baseline_model = SentenceTransformer(config["model_name"], model_kwargs=config.get("model_kwargs") or {})
    for fold_idx, baseline_morals, ft_morals, test_gt in fold_data:
        print(f"\n  Fold {fold_idx + 1}/{len(fold_indices)}  (test={len(baseline_morals)})")
        bm = evaluate(baseline_model, baseline_morals, doc_texts, test_gt, force=args.force)
        baseline_fold_metrics.append(bm)
        print(f"    Baseline  MRR={bm['MRR']:.4f}  R@1={bm['Recall@1']:.4f}  R@5={bm['Recall@5']:.4f}")
    del baseline_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Pass 2: fine-tuned models (one per fold, freed after each) ────────────
    finetuned_fold_metrics: list[dict] = []
    print("\n  [Pass 2/2] Fine-tuned models...")
    for fold_idx, baseline_morals, ft_morals, test_gt in fold_data:
        print(f"\n  Fold {fold_idx + 1}/{len(fold_indices)}  (test={len(ft_morals)})")
        model_cache = _model_root / train_doc_mode / f"fold_{fold_idx}"
        emb_cache = CACHE_DIR / "embeddings" / f"{train_doc_mode}_eval_{config['doc_mode']}" / f"fold_{fold_idx}"
        ft_model = SentenceTransformer(str(model_cache), model_kwargs=config.get("model_kwargs") or {})
        fm = evaluate(ft_model, ft_morals, doc_texts, test_gt, cache_dir=emb_cache, force=args.force)
        finetuned_fold_metrics.append(fm)
        print(f"    Fine-tuned MRR={fm['MRR']:.4f}  R@1={fm['Recall@1']:.4f}  R@5={fm['Recall@5']:.4f}")
        del ft_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    label = f"{train_doc_mode}_eval_{config['doc_mode']}" if cross_eval else config["doc_mode"]
    out = RESULTS_DIR / f"{ts}_{label}_{suffix}_evaluate.json"
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
