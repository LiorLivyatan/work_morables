"""
03_score_gap_distribution/analyze.py
======================================
How large is the gap between rank-1 and the ground-truth fable's score?
Near-misses (gap ≈ 0.01) vs confident mistakes (gap ≈ 0.1+) require different fixes.

Outputs (all saved to --output_dir):
  score_gaps.csv            — raw per-query data
  score_gap_histogram.png   — distribution of score gaps for all misranked queries
  score_gap_by_rank.png     — box plot: gap grouped by gt_rank bucket

FIXED:
  - Gap = score(rank-1) - score(gt) for queries where gt_rank > 1

CONFIGURABLE:
  --moral_embs   path to moral_embs.npy
  --doc_embs     path to doc_embs.npy
  --label        experiment name
  --compare      space-separated list of "label:moral_embs:doc_embs" for overlay plots
                 e.g. --compare "zero-shot:exp12/moral.npy:exp12/doc.npy"
  --output_dir   where to save outputs (default: results/)

Usage
-----
# Single experiment:
./run.sh analysis/03_score_gap_distribution/analyze.py \\
    --moral_embs finetuning/ft_07_storal_transfer/cache/embeddings/linq_s500/fable_plus_summary/moral_embs.npy \\
    --doc_embs   finetuning/ft_07_storal_transfer/cache/embeddings/linq_s500/fable_plus_summary/doc_embs.npy \\
    --label      "ft07-linq-s500"

# Overlay zero-shot vs fine-tuned:
./run.sh analysis/03_score_gap_distribution/analyze.py \\
    --moral_embs finetuning/ft_07_storal_transfer/cache/embeddings/linq_s500/fable_plus_summary/moral_embs.npy \\
    --doc_embs   finetuning/ft_07_storal_transfer/cache/embeddings/linq_s500/fable_plus_summary/doc_embs.npy \\
    --label      "ft07-linq-s500" \\
    --compare    "zero-shot:experiments/12_.../moral_embs.npy:experiments/12_.../doc_embs.npy"
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.lib.loader import ExperimentConfig, load_dataset, load_embeddings, compute_rankings
from analysis.lib.plotting import setup_style, save_fig, COLOURS, PALETTE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--moral_embs", required=True)
    p.add_argument("--doc_embs",   required=True)
    p.add_argument("--label",      required=True)
    p.add_argument("--compare",    nargs="*", default=[],
                   help="Additional experiments: 'label:moral_embs:doc_embs'")
    p.add_argument("--output_dir", default=str(Path(__file__).parent / "results"))
    return p.parse_args()


def gaps_for_config(cfg, qrels) -> list[float]:
    moral_embs, doc_embs = load_embeddings(cfg)
    rankings = compute_rankings(moral_embs, doc_embs, qrels)
    return [r["score_gap"] for r in rankings if r["gt_rank"] > 1]


def main():
    args = parse_args()
    setup_style()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[03_score_gap_distribution] {args.label}")
    fables, morals, qrels = load_dataset()

    cfg = ExperimentConfig(
        moral_embs_path=args.moral_embs,
        doc_embs_path=args.doc_embs,
        label=args.label,
    )
    moral_embs, doc_embs = load_embeddings(cfg)
    rankings = compute_rankings(moral_embs, doc_embs, qrels)

    # ── CSV ───────────────────────────────────────────────────────────────────
    with open(out / "score_gaps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_idx", "gt_rank", "score_gap", "gt_score", "top1_score"])
        for r in rankings:
            if r["gt_rank"] > 1:
                w.writerow([
                    r["query_idx"], r["gt_rank"],
                    f"{r['score_gap']:.4f}",
                    f"{r['gt_score']:.4f}",
                    f"{r['top1_score']:.4f}",
                ])
    print(f"  [saved] {out / 'score_gaps.csv'}")

    # ── Histogram (with optional overlays) ───────────────────────────────────
    all_series = [(args.label, [r["score_gap"] for r in rankings if r["gt_rank"] > 1])]
    for spec in args.compare:
        lbl, m_path, d_path = spec.split(":")
        c = ExperimentConfig(moral_embs_path=m_path, doc_embs_path=d_path, label=lbl)
        all_series.append((lbl, gaps_for_config(c, qrels)))

    fig, ax = plt.subplots()
    bins = np.linspace(0, max(max(s) for _, s in all_series) + 0.01, 40)
    for (lbl, gaps), colour in zip(all_series, PALETTE):
        ax.hist(gaps, bins=bins, alpha=0.6, color=colour, label=lbl, density=True)
        ax.axvline(np.mean(gaps), color=colour, linestyle="--", linewidth=1.5,
                   label=f"{lbl} mean = {np.mean(gaps):.3f}")
    ax.set_xlabel("Score gap (rank-1 score − ground-truth score)")
    ax.set_ylabel("Density")
    ax.set_title("Score gap distribution (misranked queries only)")
    ax.legend()
    save_fig(str(out / "score_gap_histogram.png"), fig)

    # ── Box plot: gap by gt_rank bucket ──────────────────────────────────────
    misranked = [r for r in rankings if r["gt_rank"] > 1]
    rank_buckets = {2: [], 3: [], "4-5": [], "6-10": [], "11+": []}
    for r in misranked:
        rk = r["gt_rank"]
        if   rk == 2:           rank_buckets[2].append(r["score_gap"])
        elif rk == 3:           rank_buckets[3].append(r["score_gap"])
        elif 4 <= rk <= 5:      rank_buckets["4-5"].append(r["score_gap"])
        elif 6 <= rk <= 10:     rank_buckets["6-10"].append(r["score_gap"])
        else:                   rank_buckets["11+"].append(r["score_gap"])

    labels = [str(k) for k in rank_buckets]
    data   = [rank_buckets[k] for k in rank_buckets]
    fig, ax = plt.subplots()
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, colour in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)
    ax.set_xlabel("Ground-truth fable rank")
    ax.set_ylabel("Score gap")
    ax.set_title(f"Score gap by gt_rank — {args.label}")
    save_fig(str(out / "score_gap_by_rank.png"), fig)

    print("  Done.\n")


if __name__ == "__main__":
    main()
