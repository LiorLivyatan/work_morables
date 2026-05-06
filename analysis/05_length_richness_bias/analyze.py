"""
05_length_richness_bias/analyze.py
=====================================
Do longer/richer fables systematically appear at rank-1 for the WRONG moral?
"Attractor fables" dominate embedding space and absorb nearby morals.

Outputs:
  false_positive_fables.csv  — per fable: word_count, n_times_rank1, n_times_false_positive
  length_vs_fp_rate.png      — scatter: fable length vs false-positive rate
  top_attractor_fables.md    — fables that most often appear as rank-1 for wrong morals

FIXED:
  - False positive = fable appears at rank-1 but is NOT the ground-truth fable
  - Word count = len(text.split()) as richness proxy

CONFIGURABLE:
  --moral_embs       path to moral_embs.npy
  --doc_embs         path to doc_embs.npy
  --label            experiment name
  --richness_metric  word_count | unique_words | sentence_count (default: word_count)
  --n_top_fp         how many top attractor fables to show in .md (default: 20)
  --output_dir       where to save outputs (default: results/)
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
from analysis.lib.plotting import setup_style, save_fig, COLOURS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--moral_embs",      required=True)
    p.add_argument("--doc_embs",        required=True)
    p.add_argument("--label",           required=True)
    p.add_argument("--richness_metric", default="word_count",
                   choices=["word_count", "unique_words", "sentence_count"])
    p.add_argument("--n_top_fp",        type=int, default=20)
    p.add_argument("--output_dir",      default=str(Path(__file__).parent / "results"))
    return p.parse_args()


def richness(text: str, metric: str) -> int:
    if metric == "word_count":     return len(text.split())
    if metric == "unique_words":   return len(set(text.lower().split()))
    if metric == "sentence_count": return text.count(".") + text.count("!") + text.count("?")
    return len(text.split())


def main():
    args = parse_args()
    setup_style()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[05_length_richness_bias] {args.label}")
    fables, morals, qrels = load_dataset()

    cfg = ExperimentConfig(
        moral_embs_path=args.moral_embs,
        doc_embs_path=args.doc_embs,
        label=args.label,
    )
    moral_embs, doc_embs = load_embeddings(cfg)
    rankings = compute_rankings(moral_embs, doc_embs, qrels)

    fable_texts  = [f["text"]  for f in fables]
    fable_titles = [f.get("title", f["doc_id"]) for f in fables]
    n_fables     = len(fables)

    rich = [richness(t, args.richness_metric) for t in fable_texts]

    # Per-fable tallies
    n_rank1   = [0] * n_fables   # times this fable appeared at rank-1 (any query)
    n_fp      = [0] * n_fables   # times rank-1 AND not the gt (false positive)
    sum_rank  = [0] * n_fables   # sum of ranks (for mean rank as a corpus item)
    n_queries = len(rankings)

    for r in rankings:
        top1 = r["ranked_indices"][0]
        gt   = r["gt_fable_idx"]
        n_rank1[top1] += 1
        if top1 != gt:
            n_fp[top1] += 1
        for rank, fable_idx in enumerate(r["ranked_indices"], 1):
            sum_rank[fable_idx] += rank

    mean_rank = [s / n_queries for s in sum_rank]
    fp_rate   = [fp / n_queries for fp in n_fp]

    # ── CSV ───────────────────────────────────────────────────────────────────
    with open(out / "false_positive_fables.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fable_idx", "title", args.richness_metric,
                    "n_times_rank1", "n_times_false_positive", "fp_rate", "mean_rank"])
        for i in range(n_fables):
            w.writerow([i, fable_titles[i], rich[i],
                        n_rank1[i], n_fp[i], f"{fp_rate[i]:.4f}", f"{mean_rank[i]:.1f}"])
    print(f"  [saved] {out / 'false_positive_fables.csv'}")

    # ── Scatter: richness vs false-positive rate ──────────────────────────────
    fig, ax = plt.subplots()
    sc = ax.scatter(rich, fp_rate, alpha=0.5, c=n_rank1,
                    cmap="RdYlGn_r", s=20, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Times appeared at rank-1")
    ax.set_xlabel(f"Fable {args.richness_metric.replace('_', ' ')}")
    ax.set_ylabel("False-positive rate (rank-1 for wrong moral)")
    ax.set_title(f"Richness vs false-positive rate — {args.label}")
    save_fig(str(out / "length_vs_fp_rate.png"), fig)

    # ── Top attractor fables ──────────────────────────────────────────────────
    top_fp = sorted(range(n_fables), key=lambda i: n_fp[i], reverse=True)[:args.n_top_fp]

    lines = [
        f"# Top Attractor Fables — {args.label}",
        f"\nFables that most often appear at rank-1 for the WRONG moral.",
        f"High false-positive count = this fable 'absorbs' many morals in embedding space.",
        "",
    ]
    for rank_i, fable_i in enumerate(top_fp, 1):
        lines += [
            f"---",
            f"## {rank_i}. {fable_titles[fable_i]}",
            f"- **False positives:** {n_fp[fable_i]} / {n_queries} queries",
            f"- **Total rank-1 appearances:** {n_rank1[fable_i]}",
            f"- **{args.richness_metric}:** {rich[fable_i]}",
            f"- **Mean rank across all queries:** {mean_rank[fable_i]:.1f}",
            f"",
            f"> {' '.join(fable_texts[fable_i].split()[:60])} …",
            f"",
        ]
    with open(out / "top_attractor_fables.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [saved] {out / 'top_attractor_fables.md'}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
