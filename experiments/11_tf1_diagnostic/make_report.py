"""
Generate report figures from a TF1 IoU diagnostic run.

Reads summary.json (and the MORABLES corpus for the IoU histogram), writes
PNGs into <run_dir>/figures/. PNGs are gitignored — regenerate with:

    ./run.sh experiments/11_tf1_diagnostic/make_report.py \
        --run experiments/11_tf1_diagnostic/results/runs/20260502_184314
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the exact tokenization from check_iou.py for the IoU recompute
WORD_RE = re.compile(r"\b\w+\b")
STOP_WORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "and", "but", "or", "nor", "not", "so", "if",
    "than", "that", "this", "it", "its", "his", "her", "their", "who",
    "which", "what", "when", "where", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "only",
    "own", "same", "he", "she", "they", "them", "him", "we", "you", "i",
    "me", "my", "your", "our",
}


def word_set(t: str) -> set[str]:
    return set(WORD_RE.findall(t.lower()))


def iou_no_stop(moral: str, fable: str) -> float:
    a = word_set(moral) - STOP_WORDS
    b = word_set(fable) - STOP_WORDS
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "font.family": "sans-serif",
        "font.size": 10,
    })


C_MORABLES = "#D7263D"
C_TF1 = "#1B9AAA"
C_NEUTRAL = "#5F6368"


def fig_plateau(summary: dict, out: Path) -> None:
    """Cumulative unique morals vs rows seen — the plateau curve."""
    growth = summary["tf1"]["unique_growth_curve"]
    xs = [g[0] for g in growth]
    ys = [g[1] for g in growth]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, color=C_TF1, linewidth=2.2, marker="o", markersize=3)
    ax.axhline(100, color=C_NEUTRAL, linestyle="--", alpha=0.5)
    ax.text(
        max(xs) * 0.98, 100, "  global pool = 100",
        va="bottom", ha="right", color=C_NEUTRAL, fontsize=9,
    )

    # Mark chunk boundaries
    chunks = summary["tf1"]["chunks"]
    chunk_size = summary["tf1"]["chunk_size"]
    for c in range(1, chunks):
        ax.axvline(c * chunk_size, color="#ccc", linewidth=0.6, zorder=0)

    ax.set_xlabel("Cumulative TF1 rows streamed (10 stratified chunks across 3M)")
    ax.set_ylabel("Unique morals discovered")
    ax.set_title(
        "TF1 moral pool saturates immediately and never grows\n"
        "10 chunks × 5,000 rows spanning the full 3M-row dataset"
    )
    ax.set_ylim(0, 120)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_iou_distribution(summary: dict, morables_path: Path, samples_path: Path, out: Path) -> None:
    """IoU (no-stopwords) distributions: MORABLES vs TF1."""
    fables = json.loads(morables_path.read_text())
    morables_iou = [iou_no_stop(f["moral"], f["story"]) for f in fables]

    tf1_iou: list[float] = []
    with samples_path.open() as f:
        for line in f:
            tf1_iou.append(json.loads(line)["iou_no_stop"])

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 0.12, 50)
    ax.hist(morables_iou, bins=bins, alpha=0.55, color=C_MORABLES,
            label=f"MORABLES (n={len(morables_iou)})  μ={np.mean(morables_iou):.4f}",
            density=True)
    ax.hist(tf1_iou, bins=bins, alpha=0.55, color=C_TF1,
            label=f"TF1-EN-3M sample (n={len(tf1_iou)})  μ={np.mean(tf1_iou):.4f}",
            density=True)

    ax.axvline(np.mean(morables_iou), color=C_MORABLES, linestyle="--", alpha=0.8, linewidth=1)
    ax.axvline(np.mean(tf1_iou), color=C_TF1, linestyle="--", alpha=0.8, linewidth=1)

    ax.set_xlabel("IoU between moral and fable (content words only)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Both datasets have low lexical overlap — TF1 doesn't 'leak' moral words into fables\n"
        "TF1 mean is ~2× MORABLES, but both well under 0.05 — the task remains semantic"
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_moral_frequency(summary: dict, out: Path, top_k: int = 30) -> None:
    """Top-K of TF1's 100 morals by occurrence in the 50K sample."""
    items = summary["tf1"]["all_unique_morals_with_counts"][:top_k]
    morals = [m for m, _ in items][::-1]
    counts = [c for _, c in items][::-1]

    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(morals, counts, color=C_TF1, alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8, color="#333")

    ax.set_xlabel(f"Occurrences in 50,000-row stratified sample (extrapolates to ~30K per moral in the 3M)")
    ax.set_title(
        f"Top {top_k} of 100 TF1 morals — distribution is roughly flat (2× spread top-to-bottom)\n"
        "Implication: each moral has ~30,000 fables in the 3M dataset"
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_chunk_consistency(summary: dict, out: Path) -> None:
    """Heatmap: chunks × top-12 morals, values = count in that chunk."""
    per_chunk = summary["tf1"]["per_chunk"]
    top_morals = [m for m, _ in summary["tf1"]["all_unique_morals_with_counts"][:12]]

    samples_path = Path(out).parent.parent / "samples.jsonl"
    chunk_moral_counts: dict[int, Counter] = {c["chunk"]: Counter() for c in per_chunk}
    with samples_path.open() as f:
        for line in f:
            r = json.loads(line)
            chunk_moral_counts[r["chunk"]][r["moral"].lower()] += 1

    matrix = np.array([
        [chunk_moral_counts[c["chunk"]].get(m, 0) for m in top_morals]
        for c in per_chunk
    ])

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0)

    ax.set_xticks(range(len(top_morals)))
    ax.set_xticklabels(top_morals, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(per_chunk)))
    ax.set_yticklabels(
        [f"chunk {c['chunk']+1}  (offset {c['offset']:,})" for c in per_chunk],
        fontsize=10,
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=9, color="black" if matrix[i, j] < matrix.max() * 0.6 else "white")

    cbar = fig.colorbar(im, ax=ax, label="count in chunk (out of 5,000)", fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        "Same 12 morals dominate every chunk across the 3M dataset\n"
        "Confirms a global (not shard-local) 100-moral pool",
        pad=15,
    )
    fig.subplots_adjust(left=0.18, right=0.95, top=0.90, bottom=0.30)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_summary_table(summary: dict, out: Path) -> None:
    """Side-by-side stats card: MORABLES vs TF1."""
    m = summary["morables"]
    t = summary["tf1"]
    rows = [
        ["Source", "MORABLES (curated literary)", "TF1-EN-3M (synthetic)"],
        ["Pairs", f"{m['n_pairs']:,}", f"{t['rows_streamed']:,} sampled / 3,000,000 total"],
        ["Unique morals", "678 / 709 fables (≈1:1)", "100 / 50,000 sampled (≈30,000:1)"],
        ["IoU mean (no-stop)", f"{m['iou_no_stopwords']['mean']:.4f}", f"{t['iou_no_stopwords']['mean']:.4f}"],
        ["IoU median (no-stop)", f"{m['iou_no_stopwords']['median']:.4f}", f"{t['iou_no_stopwords']['median']:.4f}"],
        ["IoU 99th pct (no-stop)", f"{m['iou_no_stopwords']['p99']:.4f}", f"{t['iou_no_stopwords']['p99']:.4f}"],
        ["Source of morals", "Aesop / Gibbs / Perry / Abstemius", "Llama-3.1-8B seeded from 100-phrase pool"],
        ["License", "(thesis-internal)", "MIT"],
    ]

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows[1:], colLabels=rows[0],
        loc="center", cellLoc="left", colLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    for col_idx, color in enumerate(["white", "#FCEDEF", "#E8F5F8"]):
        for row_idx in range(len(rows)):
            cell = tbl[row_idx, col_idx]
            if row_idx == 0:
                cell.set_facecolor("#222")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor(color)
    ax.set_title("MORABLES ↔ TF1-EN-3M side-by-side", pad=20, fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="path to a results/runs/<timestamp> dir")
    args = parser.parse_args()

    run_dir = Path(args.run)
    summary = json.loads((run_dir / "summary.json").read_text())
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)

    morables_path = Path(__file__).parent.parent.parent / "data" / "raw" / "fables.json"
    samples_path = run_dir / "samples.jsonl"

    setup_style()

    fig_plateau(summary, figures / "01_plateau.png")
    fig_iou_distribution(summary, morables_path, samples_path, figures / "02_iou_distribution.png")
    fig_moral_frequency(summary, figures / "03_moral_frequency.png")
    fig_chunk_consistency(summary, figures / "04_chunk_consistency.png")
    fig_summary_table(summary, figures / "05_summary_table.png")

    for f in sorted(figures.glob("*.png")):
        print(f"  wrote {f}")


if __name__ == "__main__":
    main()
