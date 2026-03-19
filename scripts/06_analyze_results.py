# scripts/06_analyze_results.py
"""
Analyse and visualise model comparison results.

Loads from the most recent run directory (or --run-dir), and writes into it:
  results.csv           — all metrics, sorted by MRR
  plot_ranking.png      — horizontal bar chart: all runs sorted by MRR
  plot_instruction.png  — grouped bars: instruction-variant effect per model

Usage:
  python scripts/06_analyze_results.py                        # latest run
  python scripts/06_analyze_results.py --run-dir results/runs/2026-03-15_08-39-45
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"

VARIANT_COLORS = {
    "no_instr": "#4C72B0",
    "plain":    "#4C72B0",
    "general":  "#DD8452",
    "specific": "#55A868",
    "retrieval": "#C44E52",
    "prefixed":  "#8172B2",
    "prompted":  "#8172B2",
}
DEFAULT_COLOR = "#937860"


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    with open(run_dir / "results.json") as f:
        results = json.load(f)
    with open(run_dir / "metadata.json") as f:
        metadata = json.load(f)
    df = pd.DataFrame([r for r in results if "error" not in r])
    df = df.sort_values("MRR", ascending=False).reset_index(drop=True)
    return df, metadata


def save_csv(df: pd.DataFrame, run_dir: Path):
    col_order = [
        "run_key", "model",
        "MRR", "NDCG@10", "Mean Rank", "Median Rank",
        "Recall@1", "Recall@5", "Recall@10", "Recall@50",
        "P@1", "P@10",
        "NDCG@1", "NDCG@5", "NDCG@50",
        "R-Precision", "MAP", "n_queries", "encoding_time_s",
    ]
    cols = [c for c in col_order if c in df.columns]
    out = run_dir / "results.csv"
    df[cols].to_csv(out, index=False, float_format="%.6f")
    print(f"  Saved {out}")


def plot_ranking(df: pd.DataFrame, run_dir: Path):
    """Horizontal bar chart of all runs sorted by MRR."""
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(df) * 0.35)))
    fig.suptitle("Model Comparison — Moral → Fable Retrieval", fontsize=13, y=1.01)

    for ax, metric in zip(axes, ["MRR", "Recall@10"]):
        vals = df[metric].values
        labels = df["run_key"].values
        # colour by instruction variant (suffix after last __)
        colors = [
            VARIANT_COLORS.get(k.rsplit("__", 1)[-1], DEFAULT_COLOR)
            for k in labels
        ]
        bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(metric)
        ax.set_title(metric)
        ax.axvline(df[metric].mean(), color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=6)

    # Legend for variant colours
    from matplotlib.patches import Patch
    legend_items = [Patch(color=c, label=k) for k, c in VARIANT_COLORS.items()]
    fig.legend(handles=legend_items, title="Variant", loc="lower center",
               ncol=len(VARIANT_COLORS), bbox_to_anchor=(0.5, -0.02), fontsize=8)

    plt.tight_layout()
    out = run_dir / "plot_ranking.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_instruction_effect(df: pd.DataFrame, run_dir: Path):
    """
    Grouped bar chart showing no_instr / general / specific per model.
    Only includes models that have more than one instruction variant.
    """
    # Extract base model name (everything before last __)
    df = df.copy()
    df["base"] = df["run_key"].str.rsplit("__", n=1).str[0]
    df["variant"] = df["run_key"].str.rsplit("__", n=1).str[1]

    # Keep only models with multiple variants
    multi = df.groupby("base").filter(lambda g: len(g) > 1)
    if multi.empty:
        print("  No multi-variant models yet, skipping instruction plot.")
        return

    bases = multi["base"].unique()
    variants = ["no_instr", "plain", "general", "specific", "retrieval", "prefixed", "prompted"]
    present_variants = [v for v in variants if v in multi["variant"].values]

    x = np.arange(len(bases))
    width = 0.8 / max(len(present_variants), 1)

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(bases) * 0.9), 10))
    fig.suptitle("Instruction Variant Effect per Model", fontsize=13)

    for ax, metric in zip(axes, ["MRR", "Recall@10"]):
        for j, variant in enumerate(present_variants):
            subset = multi[multi["variant"] == variant].set_index("base")
            vals = [subset.loc[b, metric] if b in subset.index else 0.0 for b in bases]
            offset = (j - len(present_variants) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width=width * 0.9,
                   label=variant, color=VARIANT_COLORS.get(variant, DEFAULT_COLOR),
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(bases, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(title="Variant", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = run_dir / "plot_instruction.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse model comparison results.")
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Path to a specific run directory. Defaults to the most recent run.",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dirs = sorted(RUNS_DIR.iterdir())
        if not run_dirs:
            raise RuntimeError(f"No run directories found in {RUNS_DIR}")
        run_dir = run_dirs[-1]

    print(f"Analysing run: {run_dir}")
    df, metadata = load_run(run_dir)
    print(f"  {len(df)} completed runs  |  device: {metadata['device']}")
    print(f"  Top-3 by MRR:")
    for _, row in df.head(3).iterrows():
        print(f"    {row['run_key']:<45} MRR={row['MRR']:.4f}  NDCG@10={row['NDCG@10']:.4f}")

    save_csv(df, run_dir)
    plot_ranking(df, run_dir)
    plot_instruction_effect(df, run_dir)

    print(f"\nDone. Outputs written to {run_dir}/")
