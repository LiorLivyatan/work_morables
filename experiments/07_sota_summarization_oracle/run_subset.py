"""
run_subset.py — Evaluate cot_proverb summaries on a subset of fables.

Runs retrieval using Linq-Embed-Mistral on the first N fables that have
golden summaries. Tests three configurations:
  - Baseline: raw fable text
  - Config A: summary only
  - Config B: fable + summary concatenated

Usage:
  python experiments/07_sota_summarization_oracle/run_subset.py
  python experiments/07_sota_summarization_oracle/run_subset.py --summaries-path path/to/golden_summaries.json
  python experiments/07_sota_summarization_oracle/run_subset.py --variant proverb
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "lib"))
from data import load_fables, load_morals, load_qrels_moral_to_fable
from retrieval_utils import compute_metrics

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
GENERATION_RUNS_DIR = RESULTS_DIR / "generation_runs"

# ── Embedding model ──────────────────────────────────────────────────────────
EMBED_MODEL_ID = "Linq-AI-Research/Linq-Embed-Mistral"
QUERY_INSTRUCTION = "Given a text, retrieve the most relevant passage that answers the query"


def find_latest_summaries() -> Path:
    if not GENERATION_RUNS_DIR.exists():
        raise FileNotFoundError(f"No generation runs at {GENERATION_RUNS_DIR}")
    run_dirs = sorted(GENERATION_RUNS_DIR.iterdir())
    if not run_dirs:
        raise FileNotFoundError("No generation runs found. Run generate_summaries.py first.")
    return run_dirs[-1] / "golden_summaries.json"


def main():
    parser = argparse.ArgumentParser(description="Evaluate summaries on fable subset")
    parser.add_argument(
        "--summaries-path", type=Path, default=None,
        help="Path to golden_summaries.json (default: latest generation run).",
    )
    parser.add_argument(
        "--variant", type=str, default="cot_proverb",
        help="Which summary variant to evaluate (default: cot_proverb).",
    )
    args = parser.parse_args()

    # Load golden summaries
    summaries_path = args.summaries_path or find_latest_summaries()
    print(f"\nLoading summaries from: {summaries_path}")
    with open(summaries_path) as f:
        golden = json.load(f)

    variant = args.variant
    available = set(golden[0]["summaries"].keys())
    if variant not in available:
        print(f"ERROR: variant '{variant}' not found. Available: {available}")
        sys.exit(1)

    # Build summary lookup: fable_idx -> summary text
    summary_lookup = {
        int(item["id"].split("_")[1]): item["summaries"][variant]
        for item in golden
    }
    subset_fable_indices = sorted(summary_lookup.keys())
    n_fables = max(subset_fable_indices) + 1  # corpus size (0 to max index)

    print(f"  {len(golden)} fables with summaries (indices 0-{max(subset_fable_indices)})")
    print(f"  Variant: {variant}")
    print(f"  Sample: \"{summary_lookup[subset_fable_indices[0]]}\"")

    # Load full dataset
    fables = load_fables()
    morals = load_morals()
    gt_m2f = load_qrels_moral_to_fable()

    # Subset: only fables we have summaries for, and morals that map to them
    fable_texts = [fables[i]["text"] for i in range(n_fables)]
    summaries = [summary_lookup.get(i, "") for i in range(n_fables)]

    moral_indices = sorted(gt_m2f.keys())
    subset_morals = []
    subset_gt = {}
    q_idx = 0
    for m_idx in moral_indices:
        fable_idx = gt_m2f[m_idx]
        if fable_idx in summary_lookup:
            subset_morals.append(morals[m_idx]["text"])
            subset_gt[q_idx] = fable_idx
            q_idx += 1

    print(f"  {n_fables} fables in corpus, {len(subset_morals)} moral queries")

    # Load embedding model
    print(f"\nLoading {EMBED_MODEL_ID}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL_ID)

    def encode(texts, is_query=False, batch_size=32):
        if is_query:
            texts = [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {t}" for t in texts]
        return model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)

    # Encode morals once
    print("\nEncoding morals...")
    moral_embs = encode(subset_morals, is_query=True)

    results = {}

    # Baseline: raw fable
    print("\n--- Baseline: raw fable ---")
    fable_embs = encode(fable_texts)
    m = compute_metrics(moral_embs, fable_embs, subset_gt)
    print(f"  MRR={m['MRR']:.4f}  R@1={m['Recall@1']:.3f}  "
          f"R@5={m['Recall@5']:.3f}  R@10={m['Recall@10']:.3f}")
    results["baseline_raw_fable"] = m

    # Config A: summary only
    print(f"\n--- Config A: {variant} summary only ---")
    sum_embs = encode(summaries)
    m_a = compute_metrics(moral_embs, sum_embs, subset_gt)
    print(f"  MRR={m_a['MRR']:.4f}  R@1={m_a['Recall@1']:.3f}  "
          f"R@5={m_a['Recall@5']:.3f}  R@10={m_a['Recall@10']:.3f}")
    results["config_A_summary_only"] = m_a

    # Config B: fable + summary
    print(f"\n--- Config B: fable + {variant} summary ---")
    enriched = [f"{fable_texts[i]}\n\nMoral summary: {summaries[i]}" for i in range(n_fables)]
    enr_embs = encode(enriched)
    m_b = compute_metrics(moral_embs, enr_embs, subset_gt)
    print(f"  MRR={m_b['MRR']:.4f}  R@1={m_b['Recall@1']:.3f}  "
          f"R@5={m_b['Recall@5']:.3f}  R@10={m_b['Recall@10']:.3f}")
    results["config_B_fable_plus_summary"] = m_b

    # Summary table
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY (subset: {len(subset_morals)} queries, {n_fables} fables)")
    print(f"{'═' * 70}")
    print(f"  {'Config':<35} {'MRR':>8} {'R@1':>8} {'R@10':>8}")
    print(f"  {'─' * 60}")
    for label, r in results.items():
        print(f"  {label:<35} {r['MRR']:>8.4f} {r['Recall@1']:>7.1%} {r['Recall@10']:>7.1%}")

    # Save
    run_dir = summaries_path.parent
    output = {
        "variant": variant,
        "n_fables": n_fables,
        "n_morals": len(subset_morals),
        "summaries_path": str(summaries_path),
        **results,
    }
    out_path = run_dir / f"retrieval_results_{variant}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
