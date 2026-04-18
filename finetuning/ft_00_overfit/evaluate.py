"""
ft_00_overfit — standalone evaluation.

Loads the cached fine-tuned model and compares it against the baseline
(untrained model) on the full 709-pair dataset. No training is performed.

Use this to:
  - Re-evaluate a cached model without re-running training
  - Try a different doc_mode than the one used during training
  - Get a quick sanity check on a specific checkpoint

Usage
-----
    python finetuning/ft_00_overfit/evaluate.py
    python finetuning/ft_00_overfit/evaluate.py --doc_mode fable_plus_summary
    python finetuning/ft_00_overfit/evaluate.py --wandb     # log results to wandb
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

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

_METRICS = ("MRR", "Recall@1", "Recall@5", "Recall@10")
_W = 14  # column width


def _row(label: str, baseline: float, finetuned: float) -> str:
    delta = finetuned - baseline
    sign = "+" if delta >= 0 else ""
    return f"  {label:<12}  {baseline:{_W}.4f}  {finetuned:{_W}.4f}  {sign}{delta:.4f}"


def _divider() -> str:
    return "  " + "─" * (12 + _W * 2 + 14)


def print_comparison(baseline_metrics: dict, trained_metrics: dict, config: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"  ft_00_overfit  |  doc_mode={config['doc_mode']}  |  model={config['model_name']}")
    print(_divider())
    print(f"  {'Metric':<12}  {'Baseline':>{_W}}  {'Fine-tuned':>{_W}}  {'Δ'}")
    print(_divider())
    for key in _METRICS:
        print(_row(key, baseline_metrics[key], trained_metrics[key]))
    print(_divider())


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone evaluation for ft_00_overfit")
    parser.add_argument("--doc_mode", choices=["raw", "fable_plus_summary"], help="Override config doc_mode")
    parser.add_argument("--wandb", dest="use_wandb", action="store_true", help="Log results to wandb")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    if args.doc_mode:
        config["doc_mode"] = args.doc_mode

    model_cache = CACHE_DIR / "models" / config["doc_mode"]
    if not model_cache.exists():
        raise FileNotFoundError(
            f"No cached model found at {model_cache}\n"
            f"Run training first:  python finetuning/ft_00_overfit/train.py --doc_mode {config['doc_mode']}"
        )

    print(
        f"\n[ft_00_overfit / evaluate]  "
        f"doc_mode={config['doc_mode']}  "
        f"model={config['model_name']}"
    )

    moral_texts, doc_texts, ground_truth = load_pairs(config["doc_mode"])

    # ── Baseline ──────────────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    print("\n  Evaluating baseline (untrained model)...")
    baseline = SentenceTransformer(config["model_name"])
    baseline_metrics = evaluate(baseline, moral_texts, doc_texts, ground_truth)
    del baseline

    # ── Fine-tuned ────────────────────────────────────────────────────────────
    print("  Evaluating fine-tuned model (from cache)...")
    emb_cache = CACHE_DIR / "embeddings" / config["doc_mode"]
    finetuned = SentenceTransformer(str(model_cache))
    trained_metrics = evaluate(finetuned, moral_texts, doc_texts, ground_truth, cache_dir=emb_cache)

    # ── Results ───────────────────────────────────────────────────────────────
    print_comparison(baseline_metrics, trained_metrics, config)

    if args.use_wandb:
        wandb_cfg = config.get("wandb", {})
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            project=wandb_cfg.get("project", "morables-finetuning"),
            name=f"eval_overfit_{config['doc_mode']}_{ts}",
            group="ft_00_overfit",
            tags=["ft_00_overfit", config["doc_mode"], "evaluate"],
            config={k: v for k, v in config.items() if k != "wandb"},
        )
        wandb.log({
            **{f"baseline/{k.lower().replace('@', '_at_')}": baseline_metrics[k] for k in _METRICS},
            **{f"eval/{k.lower().replace('@', '_at_')}": trained_metrics[k] for k in _METRICS},
            "eval/delta_mrr": trained_metrics["MRR"] - baseline_metrics["MRR"],
        })
        wandb.finish()

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = RESULTS_DIR / f"{ts}_{config['doc_mode']}_evaluate.json"
    with open(out, "w") as f:
        json.dump({
            "config": config,
            "baseline_metrics": baseline_metrics,
            "trained_metrics": trained_metrics,
            "delta_mrr": trained_metrics["MRR"] - baseline_metrics["MRR"],
        }, f, indent=2)
    print(f"\n  Results → {out}")


if __name__ == "__main__":
    main()
