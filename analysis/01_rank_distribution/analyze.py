"""
01_rank_distribution/analyze.py
================================
Where does the ground-truth fable actually land in the ranking?

Outputs (all saved to --output_dir):
  rank_distribution.png  — histogram of ground-truth ranks
  rank_distribution.csv  — raw counts per rank bucket
  cumulative_recall.png  — R@K curve (K = 1..top_k)
  summary.txt            — MRR@10 and key percentiles

FIXED across all experiments:
  - Rank computed from dot-product similarity on L2-normalised embeddings
  - Dataset: 709 fables × 709 morals (lib/data.py)

CONFIGURABLE per run:
  --moral_embs   path to moral_embs.npy
  --doc_embs     path to doc_embs.npy
  --label        experiment name for plot titles
  --top_k        cutoff for R@K curve (default: 50)
  --output_dir   where to save outputs (default: results/)

Usage
-----
./run.sh analysis/01_rank_distribution/analyze.py \\
    --moral_embs finetuning/ft_07_storal_transfer/cache/embeddings/linq_s500/fable_plus_summary/moral_embs.npy \\
    --doc_embs   finetuning/ft_07_storal_transfer/cache/embeddings/linq_s500/fable_plus_summary/doc_embs.npy \\
    --label      "ft07-linq-s500-fable+summary"
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.lib.loader import ExperimentConfig, load_dataset, load_embeddings, compute_rankings, compute_mrr
from analysis.lib.plotting import setup_style, save_fig, COLOURS

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--moral_embs", required=True)
    p.add_argument("--doc_embs",   required=True)
    p.add_argument("--label",      required=True)
    p.add_argument("--top_k",      type=int, default=50)
    p.add_argument("--output_dir", default=str(Path(__file__).parent / "results"))
    return p.parse_args()


def main():
    args   = parse_args()
    setup_style()
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        moral_embs_path=args.moral_embs,
        doc_embs_path=args.doc_embs,
        label=args.label,
    )

    print(f"\n[01_rank_distribution] {args.label}")
    fables, morals, qrels = load_dataset()
    moral_embs, doc_embs  = load_embeddings(cfg)
    rankings              = compute_rankings(moral_embs, doc_embs, qrels)

    gt_ranks = [r["gt_rank"] for r in rankings]
    mrr      = compute_mrr(rankings)
    print(f"  MRR@10 = {mrr:.4f}")
    print(f"  Rank-1 = {sum(r == 1 for r in gt_ranks)} / {len(gt_ranks)}")

    # ── Rank distribution histogram ──────────────────────────────────────────
    # Bucket: 1, 2, 3, 4, 5, 6-10, 11-50, 51+
    buckets = [1, 2, 3, 4, 5, "6-10", "11-50", "51+"]
    counts  = [
        sum(r == 1 for r in gt_ranks),
        sum(r == 2 for r in gt_ranks),
        sum(r == 3 for r in gt_ranks),
        sum(r == 4 for r in gt_ranks),
        sum(r == 5 for r in gt_ranks),
        sum(6  <= r <= 10  for r in gt_ranks),
        sum(11 <= r <= 50  for r in gt_ranks),
        sum(r  > 50        for r in gt_ranks),
    ]
    fig, ax = plt.subplots()
    bars = ax.bar([str(b) for b in buckets], counts,
                  color=[COLOURS["gt"]] + [COLOURS["primary"]] * 4 +
                        [COLOURS["neutral"]] * 3)
    ax.set_xlabel("Ground-truth fable rank")
    ax.set_ylabel("Number of queries")
    ax.set_title(f"Rank distribution — {args.label}\nMRR@10 = {mrr:.4f}")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha="center", va="bottom", fontsize=9)
    save_fig(str(out / "rank_distribution.png"), fig)

    # ── CSV ──────────────────────────────────────────────────────────────────
    with open(out / "rank_distribution.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank_bucket", "count", "pct"])
        total = len(gt_ranks)
        for b, c in zip(buckets, counts):
            w.writerow([b, c, f"{100*c/total:.1f}"])
    print(f"  [saved] {out / 'rank_distribution.csv'}")

    # ── Cumulative recall (R@K) ───────────────────────────────────────────────
    ks      = list(range(1, args.top_k + 1))
    recalls = [sum(r <= k for r in gt_ranks) / len(gt_ranks) for k in ks]
    fig, ax = plt.subplots()
    ax.plot(ks, recalls, color=COLOURS["primary"], linewidth=2)
    ax.axhline(recalls[0],  color=COLOURS["gt"],      linestyle="--", alpha=0.7,
               label=f"R@1 = {recalls[0]:.3f}")
    ax.axhline(recalls[4],  color=COLOURS["neutral"],  linestyle="--", alpha=0.7,
               label=f"R@5 = {recalls[4]:.3f}")
    ax.axhline(recalls[9],  color=COLOURS["secondary"], linestyle="--", alpha=0.7,
               label=f"R@10 = {recalls[9]:.3f}")
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title(f"Cumulative recall — {args.label}")
    ax.legend()
    save_fig(str(out / "cumulative_recall.png"), fig)

    # ── Summary text ─────────────────────────────────────────────────────────
    with open(out / "summary.txt", "w") as f:
        f.write(f"Experiment: {args.label}\n")
        f.write(f"MRR@10:     {mrr:.4f}\n")
        f.write(f"R@1:        {recalls[0]:.4f}\n")
        f.write(f"R@5:        {recalls[4]:.4f}\n")
        f.write(f"R@10:       {recalls[9]:.4f}\n")
        f.write(f"Median rank: {int(np.median(gt_ranks))}\n")
        f.write(f"Mean rank:   {np.mean(gt_ranks):.1f}\n")
    print(f"  [saved] {out / 'summary.txt'}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
