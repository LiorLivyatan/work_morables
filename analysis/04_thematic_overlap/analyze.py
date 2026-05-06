"""
04_thematic_overlap/analyze.py
================================
Does the rank-1 wrong fable's OWN ground-truth moral overlap with the query moral?
If yes → dataset ambiguity. If no → genuine model error.

Outputs:
  thematic_overlap.csv          — per confusion case: query_moral, wrong_fable_moral, semantic_overlap
  ambiguity_distribution.png    — moral-moral cosine sim: confused pairs vs random pairs
  ambiguous_pairs.md            — pairs where moral-moral sim > threshold (potential annotation issues)

FIXED:
  - Moral-moral semantic similarity = cosine sim of the SAME moral embeddings
    (uses the same embedding model that produced the retrieval rankings)
  - Random baseline = random pairs of morals (500 samples)

CONFIGURABLE:
  --moral_embs          path to moral_embs.npy (used BOTH for retrieval AND moral-moral sim)
  --doc_embs            path to doc_embs.npy
  --label               experiment name
  --ambiguity_threshold moral-moral cosine sim above which we flag as ambiguous (default: 0.85)
  --output_dir          where to save outputs (default: results/)

NOTE: moral-moral similarity uses the query moral embeddings — this means the similarity
is computed in the SAME space as the retrieval, which is the most relevant measure.
If you want model-agnostic thematic overlap, replace with TF-IDF cosine on raw text
(add --overlap_method tfidf flag when implementing).
"""
import argparse
import csv
import random
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
    p.add_argument("--moral_embs",          required=True)
    p.add_argument("--doc_embs",            required=True)
    p.add_argument("--label",               required=True)
    p.add_argument("--ambiguity_threshold", type=float, default=0.85)
    p.add_argument("--output_dir",          default=str(Path(__file__).parent / "results"))
    return p.parse_args()


def main():
    args = parse_args()
    setup_style()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[04_thematic_overlap] {args.label}")
    fables, morals, qrels = load_dataset()

    cfg = ExperimentConfig(
        moral_embs_path=args.moral_embs,
        doc_embs_path=args.doc_embs,
        label=args.label,
    )
    moral_embs, doc_embs = load_embeddings(cfg)
    rankings = compute_rankings(moral_embs, doc_embs, qrels)

    # fable_idx → moral_idx (reverse qrels)
    fable_to_moral = {v: k for k, v in qrels.items()}
    moral_texts    = [morals[i]["text"] for i in range(len(morals))]

    # moral-moral similarity matrix (cosine, already L2-normed)
    moral_sim = moral_embs @ moral_embs.T  # (709, 709)

    misranked = [r for r in rankings if r["gt_rank"] > 1]

    # ── CSV ───────────────────────────────────────────────────────────────────
    confused_sims = []
    rows = []
    for r in misranked:
        q_moral_idx   = r["query_idx"]
        top1_fable    = r["ranked_indices"][0]
        top1_moral_idx = fable_to_moral.get(top1_fable)

        if top1_moral_idx is None:
            continue  # should not happen

        sim = float(moral_sim[q_moral_idx, top1_moral_idx])
        confused_sims.append(sim)
        rows.append({
            "query_moral_idx":  q_moral_idx,
            "query_moral":      moral_texts[q_moral_idx],
            "top1_fable_idx":   top1_fable,
            "top1_moral_idx":   top1_moral_idx,
            "top1_moral":       moral_texts[top1_moral_idx],
            "moral_moral_sim":  f"{sim:.4f}",
            "gt_rank":          r["gt_rank"],
            "score_gap":        f"{r['score_gap']:.4f}",
        })

    rows.sort(key=lambda x: float(x["moral_moral_sim"]), reverse=True)
    with open(out / "thematic_overlap.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  [saved] {out / 'thematic_overlap.csv'}")

    # ── Random baseline ───────────────────────────────────────────────────────
    rng = random.Random(42)
    n_morals = len(morals)
    random_sims = [
        float(moral_sim[rng.randint(0, n_morals - 1), rng.randint(0, n_morals - 1)])
        for _ in range(500)
    ]

    # ── Ambiguity distribution plot ───────────────────────────────────────────
    fig, ax = plt.subplots()
    bins = np.linspace(-0.1, 1.0, 40)
    ax.hist(random_sims,   bins=bins, alpha=0.5, color=COLOURS["neutral"],
            label="Random pairs", density=True)
    ax.hist(confused_sims, bins=bins, alpha=0.7, color=COLOURS["secondary"],
            label="Confused pairs (rank-1 wrong fable)", density=True)
    ax.axvline(args.ambiguity_threshold, color=COLOURS["highlight"], linestyle="--",
               label=f"Ambiguity threshold = {args.ambiguity_threshold}")
    ax.set_xlabel("Moral-moral cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"Thematic overlap — {args.label}\n"
                 f"Confused pairs vs random (mean confused = {np.mean(confused_sims):.3f})")
    ax.legend()
    save_fig(str(out / "ambiguity_distribution.png"), fig)

    # ── Ambiguous pairs markdown ──────────────────────────────────────────────
    ambiguous = [r for r in rows if float(r["moral_moral_sim"]) >= args.ambiguity_threshold]
    print(f"  Ambiguous pairs (sim ≥ {args.ambiguity_threshold}): {len(ambiguous)}")

    lines = [
        f"# Ambiguous Moral Pairs — {args.label}",
        f"\nPairs where the confused fable's own moral is semantically similar "
        f"(sim ≥ {args.ambiguity_threshold}) to the query moral.",
        f"These may indicate genuine annotation ambiguity in the dataset.",
        "",
    ]
    for i, r in enumerate(ambiguous, 1):
        lines += [
            f"---",
            f"## {i}. sim = {r['moral_moral_sim']} | gt_rank = {r['gt_rank']}",
            f"**Query moral:** {r['query_moral']}",
            f"**Wrong fable's moral:** {r['top1_moral']}",
            "",
        ]
    with open(out / "ambiguous_pairs.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [saved] {out / 'ambiguous_pairs.md'}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
