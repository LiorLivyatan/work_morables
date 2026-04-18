"""
ft_00_overfit — train and evaluate on the full 709-pair dataset.

Purpose
-------
Trains the model on all 709 (moral, fable) pairs and evaluates on the same set.
Expected outcome: MRR → ~1.0 (memorisation). Serves as an upper-bound sanity
check that the training pipeline is working before running cross-validation.

Usage
-----
    python finetuning/ft_00_overfit/train.py
    python finetuning/ft_00_overfit/train.py --doc_mode fable_plus_summary
    python finetuning/ft_00_overfit/train.py --force       # re-train even if cached
    python finetuning/ft_00_overfit/train.py --no-wandb    # disable wandb logging
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
from finetuning.lib.trainer import train_model

CACHE_DIR = EXP_DIR / "cache"
RESULTS_DIR = EXP_DIR / "results"
CONFIG_PATH = EXP_DIR / "config.yaml"

_EVAL_KEYS = ("MRR", "Recall@1", "Recall@5", "Recall@10")


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit fine-tuning — upper bound check")
    parser.add_argument("--doc_mode", choices=["raw", "fable_plus_summary"], help="Override config doc_mode")
    parser.add_argument("--force", action="store_true", help="Re-train and re-encode even if cached")
    parser.add_argument("--no-wandb", dest="no_wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    if args.doc_mode:
        config["doc_mode"] = args.doc_mode

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_cfg = config.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", False) and not args.no_wandb

    print(
        f"\n[ft_00_overfit]  "
        f"model={config['model_name']}  "
        f"doc_mode={config['doc_mode']}  "
        f"epochs={config['epochs']}  "
        f"wandb={'on' if use_wandb else 'off'}"
    )

    if use_wandb:
        wandb.init(
            project=wandb_cfg["project"],
            name=f"overfit_{config['doc_mode']}_{ts}",
            group="ft_00_overfit",
            tags=["ft_00_overfit", config["doc_mode"]],
            config={k: v for k, v in config.items() if k != "wandb"},
        )

    moral_texts, doc_texts, ground_truth = load_pairs(config["doc_mode"])

    # ── Baseline (untrained model) ────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    print("\n  Evaluating baseline (no fine-tuning)...")
    baseline = SentenceTransformer(config["model_name"])
    baseline_metrics = evaluate(baseline, moral_texts, doc_texts, ground_truth)
    print(f"  Baseline MRR : {baseline_metrics['MRR']:.4f}")
    if use_wandb:
        wandb.log({f"baseline/{k.lower().replace('@', '_at_')}": baseline_metrics[k] for k in _EVAL_KEYS})
    del baseline

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    train_morals = moral_texts
    train_docs = [doc_texts[ground_truth[i]] for i in range(len(moral_texts))]

    print("\n  Fine-tuning...")
    model_cache = CACHE_DIR / "models" / config["doc_mode"]
    checkpoint_dir = CACHE_DIR / "checkpoints" / config["doc_mode"]
    model = train_model(
        train_morals=train_morals,
        train_docs=train_docs,
        config=config,
        model_cache=model_cache,
        checkpoint_dir=checkpoint_dir,
        force=args.force,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n  Evaluating fine-tuned model...")
    emb_cache = CACHE_DIR / "embeddings" / config["doc_mode"]
    trained_metrics = evaluate(
        model, moral_texts, doc_texts, ground_truth,
        cache_dir=emb_cache,
        force=args.force,
    )
    print(f"  Trained MRR  : {trained_metrics['MRR']:.4f}")
    if use_wandb:
        wandb.log({
            **{f"eval/{k.lower().replace('@', '_at_')}": trained_metrics[k] for k in _EVAL_KEYS},
            "eval/delta_mrr": trained_metrics["MRR"] - baseline_metrics["MRR"],
        })

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{ts}_{config['doc_mode']}.json"
    with open(out, "w") as f:
        json.dump({
            "config": config,
            "baseline_mrr": baseline_metrics["MRR"],
            "trained_mrr": trained_metrics["MRR"],
            "baseline_metrics": baseline_metrics,
            "trained_metrics": trained_metrics,
        }, f, indent=2)
    print(f"\n  Results → {out}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
