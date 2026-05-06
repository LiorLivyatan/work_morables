"""
02_nearest_neighbor_confusion/analyze.py
=========================================
For each misranked query, what fable beat the ground truth, and why?

Outputs (all saved to --output_dir):
  confusion_cases.csv  — every query: moral, gt fable, rank, top-1 fable, scores
  hardest_cases.md     — top-N hardest cases rendered as human-readable prose

FIXED across all experiments:
  - Side-by-side comparison: moral | top-1 wrong fable | gt fable | scores
  - Only queries where gt_rank > 1 are included

CONFIGURABLE per run:
  --moral_embs       path to moral_embs.npy
  --doc_embs         path to doc_embs.npy
  --label            experiment name for headers
  --n_cases          how many hardest cases to render in .md (default: 30)
  --rank_threshold   only include queries where gt_rank >= this (default: 2)
  --output_dir       where to save outputs (default: results/)

Usage
-----
./run.sh analysis/02_nearest_neighbor_confusion/analyze.py \\
    --moral_embs finetuning/ft_07_storal_transfer/cache/embeddings/linq_s500/fable_plus_summary/moral_embs.npy \\
    --doc_embs   finetuning/ft_07_storal_transfer/cache/embeddings/linq_s500/fable_plus_summary/doc_embs.npy \\
    --label      "ft07-linq-s500-fable+summary" \\
    --n_cases    30
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.lib.loader import ExperimentConfig, load_dataset, load_embeddings, compute_rankings
from analysis.lib.plotting import setup_style


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--moral_embs",      required=True)
    p.add_argument("--doc_embs",        required=True)
    p.add_argument("--label",           required=True)
    p.add_argument("--n_cases",         type=int, default=30)
    p.add_argument("--rank_threshold",  type=int, default=2)
    p.add_argument("--output_dir",      default=str(Path(__file__).parent / "results"))
    return p.parse_args()


def _truncate(text: str, max_words: int = 60) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


def main():
    args = parse_args()
    setup_style()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        moral_embs_path=args.moral_embs,
        doc_embs_path=args.doc_embs,
        label=args.label,
    )

    print(f"\n[02_nearest_neighbor_confusion] {args.label}")
    fables, morals, qrels = load_dataset()
    moral_embs, doc_embs  = load_embeddings(cfg)
    rankings              = compute_rankings(moral_embs, doc_embs, qrels)

    fable_texts  = [f["text"]  for f in fables]
    fable_titles = [f.get("title", f["doc_id"]) for f in fables]
    moral_texts  = {r["query_idx"]: morals[r["query_idx"]]["text"] for r in rankings}

    misranked = [r for r in rankings if r["gt_rank"] >= args.rank_threshold]
    print(f"  Misranked queries (gt_rank ≥ {args.rank_threshold}): {len(misranked)} / {len(rankings)}")

    # ── CSV: all confusion cases ──────────────────────────────────────────────
    with open(out / "confusion_cases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "query_idx", "moral_text",
            "gt_fable_idx", "gt_fable_title", "gt_rank", "gt_score",
            "top1_fable_idx", "top1_fable_title", "top1_score",
            "score_gap",
        ])
        for r in sorted(misranked, key=lambda x: x["score_gap"], reverse=True):
            q   = r["query_idx"]
            gt  = r["gt_fable_idx"]
            top1 = r["ranked_indices"][0]
            w.writerow([
                q,
                moral_texts[q],
                gt,
                fable_titles[gt],
                r["gt_rank"],
                f"{r['gt_score']:.4f}",
                top1,
                fable_titles[top1],
                f"{r['top1_score']:.4f}",
                f"{r['score_gap']:.4f}",
            ])
    print(f"  [saved] {out / 'confusion_cases.csv'}")

    # ── Markdown: hardest cases (largest score gap) ───────────────────────────
    hardest = sorted(misranked, key=lambda x: x["score_gap"], reverse=True)[:args.n_cases]

    lines = [
        f"# Nearest-Neighbor Confusion — {args.label}",
        f"\nTop {args.n_cases} hardest misranked cases (largest score gap between rank-1 and ground truth).",
        "",
    ]
    for i, r in enumerate(hardest, 1):
        q    = r["query_idx"]
        gt   = r["gt_fable_idx"]
        top1 = r["ranked_indices"][0]

        lines += [
            f"---",
            f"## Case {i} — Query {q} | gt_rank={r['gt_rank']} | gap={r['score_gap']:.4f}",
            f"",
            f"**Moral (query):**",
            f"> {moral_texts[q]}",
            f"",
            f"**Rank-1 (wrong) — `{fable_titles[top1]}` — score {r['top1_score']:.4f}:**",
            f"> {_truncate(fable_texts[top1], 80)}",
            f"",
            f"**Ground truth (rank {r['gt_rank']}) — `{fable_titles[gt]}` — score {r['gt_score']:.4f}:**",
            f"> {_truncate(fable_texts[gt], 80)}",
            f"",
        ]

    with open(out / "hardest_cases.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [saved] {out / 'hardest_cases.md'}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
