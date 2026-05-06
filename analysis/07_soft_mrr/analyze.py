"""
07_soft_mrr/analyze.py
======================
Two analyses in one:

(A) Soft MRR — correct any fable whose moral is semantically similar (sim >= threshold)
    to the query moral. Gives the "true" model performance after removing dataset ambiguity.

(B) Clean subset MRR — exclude ambiguous queries entirely and recompute MRR on the
    remaining queries only. Shows what the model achieves on unambiguous examples.

Outputs:
  soft_mrr_summary.txt        — soft MRR@10, clean subset MRR, ambiguity stats
  ambiguous_queries.csv       — list of excluded query indices with their moral pairs
  mrr_comparison.png          — bar chart: standard vs soft vs clean-subset MRR

CONFIGURABLE:
  --moral_embs        path to moral_embs.npy
  --doc_embs          path to doc_embs.npy
  --label             experiment name
  --sim_threshold     moral-moral similarity threshold for "same moral" (default: 0.90)
  --output_dir        where to save outputs (default: results/)
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.lib.loader import ExperimentConfig, load_dataset, load_embeddings, compute_rankings, compute_mrr
from analysis.lib.plotting import setup_style, save_fig, COLOURS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--moral_embs",    required=True)
    p.add_argument("--doc_embs",      required=True)
    p.add_argument("--label",         required=True)
    p.add_argument("--sim_threshold", type=float, default=0.90)
    p.add_argument("--output_dir",    default=str(Path(__file__).parent / "results"))
    return p.parse_args()


def compute_soft_mrr(rankings, moral_embs, qrels, threshold, k=10):
    """
    For each query, a ranked fable is counted as correct if either:
      (a) it IS the ground-truth fable, OR
      (b) the moral attached to the wrong fable has cosine similarity >= threshold
          to the query moral.

    Returns soft_mrr, and a list of (query_idx, is_ambiguous) flags.
    """
    fables, morals, _ = load_dataset()
    # Build fable_idx -> moral_emb mapping via qrels (reverse: fable_idx -> moral_idx)
    fable_to_moral_emb = {}
    for moral_idx, fable_idx in qrels.items():
        fable_to_moral_emb[fable_idx] = moral_embs[moral_idx]

    scores = []
    ambiguous_flags = []  # True if any fable in top-k was accepted via soft match

    for r in rankings:
        q_idx      = r["query_idx"]
        gt_fable   = r["gt_fable_idx"]
        q_moral_emb = moral_embs[q_idx]

        hit = 0.0
        was_ambiguous = False
        for rank, fable_idx in enumerate(r["ranked_indices"][:k], 1):
            if fable_idx == gt_fable:
                hit = 1.0 / rank
                break
            # Check if this fable's own moral is semantically equivalent
            if fable_idx in fable_to_moral_emb:
                sim = float(q_moral_emb @ fable_to_moral_emb[fable_idx])
                if sim >= threshold:
                    hit = 1.0 / rank
                    was_ambiguous = True
                    break

        scores.append(hit)
        ambiguous_flags.append(was_ambiguous)

    return float(np.mean(scores)), ambiguous_flags


def main():
    args = parse_args()
    setup_style()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[07_soft_mrr] {args.label}")

    fables, morals, qrels = load_dataset()
    cfg = ExperimentConfig(
        moral_embs_path=args.moral_embs,
        doc_embs_path=args.doc_embs,
        label=args.label,
    )
    moral_embs, doc_embs = load_embeddings(cfg)
    rankings = compute_rankings(moral_embs, doc_embs, qrels)

    # Attach query_idx to each ranking entry
    moral_indices = list(qrels.keys())
    for i, r in enumerate(rankings):
        r["query_idx"] = moral_indices[i]

    # ── Standard MRR ─────────────────────────────────────────────────────────
    standard_mrr = compute_mrr(rankings)
    n_total = len(rankings)

    # ── Soft MRR ──────────────────────────────────────────────────────────────
    soft_mrr, ambiguous_flags = compute_soft_mrr(
        rankings, moral_embs, qrels, args.sim_threshold
    )
    n_ambiguous_rescued = sum(ambiguous_flags)
    print(f"  Standard MRR@10:        {standard_mrr:.4f}")
    print(f"  Soft MRR@10 (t={args.sim_threshold}):  {soft_mrr:.4f}")

    # ── Identify ambiguous queries (to exclude for clean subset) ──────────────
    # An "ambiguous" query is one where at least one other fable in the corpus
    # shares a moral with sim >= threshold to this query's moral
    all_moral_embs = moral_embs  # shape (n_morals, D)
    ambiguous_query_indices = set()
    ambiguous_rows = []

    for q_idx in moral_indices:
        q_emb = all_moral_embs[q_idx]
        # Check against all other morals (excluding self)
        sims = all_moral_embs @ q_emb          # shape (n_morals,)
        sims[q_idx] = 0.0                      # exclude self
        best_other_idx = int(np.argmax(sims))
        best_sim       = float(sims[best_other_idx])
        if best_sim >= args.sim_threshold:
            ambiguous_query_indices.add(q_idx)
            ambiguous_rows.append({
                "query_idx":         q_idx,
                "query_moral":       morals[q_idx]["text"],
                "similar_moral_idx": best_other_idx,
                "similar_moral":     morals[best_other_idx]["text"],
                "sim":               f"{best_sim:.4f}",
            })

    n_ambiguous = len(ambiguous_query_indices)
    print(f"  Ambiguous queries:      {n_ambiguous} / {n_total} ({100*n_ambiguous/n_total:.1f}%)")

    # ── Clean subset MRR ──────────────────────────────────────────────────────
    clean_rankings = [r for r in rankings if r["query_idx"] not in ambiguous_query_indices]
    clean_mrr = compute_mrr(clean_rankings)
    n_clean = len(clean_rankings)
    print(f"  Clean subset MRR@10:    {clean_mrr:.4f}  (n={n_clean})")

    # R@1 / R@5 / R@10 on clean subset
    clean_r1  = sum(1 for r in clean_rankings if r["gt_rank"] == 1) / n_clean
    clean_r5  = sum(1 for r in clean_rankings if r["gt_rank"] <= 5) / n_clean
    clean_r10 = sum(1 for r in clean_rankings if r["gt_rank"] <= 10) / n_clean

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = [
        f"Experiment: {args.label}",
        f"Similarity threshold: {args.sim_threshold}",
        f"",
        f"Standard MRR@10:               {standard_mrr:.4f}  (n={n_total})",
        f"Soft MRR@10:                   {soft_mrr:.4f}  (ambiguity-corrected)",
        f"  Queries rescued by soft:     {n_ambiguous_rescued}",
        f"",
        f"Ambiguous queries excluded:    {n_ambiguous} / {n_total} ({100*n_ambiguous/n_total:.1f}%)",
        f"Clean subset MRR@10:           {clean_mrr:.4f}  (n={n_clean})",
        f"Clean subset R@1:              {clean_r1:.4f}",
        f"Clean subset R@5:              {clean_r5:.4f}",
        f"Clean subset R@10:             {clean_r10:.4f}",
        f"",
        f"MRR gain (soft vs standard):   +{soft_mrr - standard_mrr:.4f}",
        f"MRR gain (clean vs standard):  +{clean_mrr - standard_mrr:.4f}",
    ]
    with open(out / "soft_mrr_summary.txt", "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"  [saved] {out / 'soft_mrr_summary.txt'}")

    # ── CSV of ambiguous queries ───────────────────────────────────────────────
    with open(out / "ambiguous_queries.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query_idx", "query_moral", "similar_moral_idx", "similar_moral", "sim"])
        w.writeheader()
        w.writerows(sorted(ambiguous_rows, key=lambda x: -float(x["sim"])))
    print(f"  [saved] {out / 'ambiguous_queries.csv'}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    labels = ["Standard\nMRR@10", f"Soft MRR@10\n(t={args.sim_threshold})", "Clean Subset\nMRR@10"]
    values = [standard_mrr, soft_mrr, clean_mrr]
    colours = [COLOURS["neutral"], COLOURS["primary"], COLOURS["gt"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colours, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, min(1.0, max(values) * 1.25))
    ax.set_ylabel("MRR@10")
    ax.set_title(f"MRR comparison — {args.label}\n(clean subset excludes {n_ambiguous} ambiguous queries)")
    save_fig(str(out / "mrr_comparison.png"), fig)

    print("  Done.\n")


if __name__ == "__main__":
    main()
