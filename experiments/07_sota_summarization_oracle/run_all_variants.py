"""
run_all_variants.py — Evaluate all summary variants in one go.

Loads the embedding model once, then runs retrieval for each variant
across all three configs (baseline, summary only, fable+summary).

Usage:
  python experiments/07_sota_summarization_oracle/run_all_variants.py
  python experiments/07_sota_summarization_oracle/run_all_variants.py --summaries-path path/to/golden_summaries.json
  python experiments/07_sota_summarization_oracle/run_all_variants.py --variants direct_moral proverb
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "lib"))
from data import load_fables, load_morals, load_qrels_moral_to_fable
from retrieval_utils import compute_metrics
from embedding_cache import encode_with_cache

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
GENERATION_RUNS_DIR = RESULTS_DIR / "generation_runs"
CACHE_DIR = ROOT_DIR / "cache" / "embeddings"

# ── Embedding model ──────────────────────────────────────────────────────────
EMBED_MODEL_ID = "Linq-AI-Research/Linq-Embed-Mistral"
QUERY_INSTRUCTION = "Given a text, retrieve the most relevant passage that answers the query"


def find_latest_summaries() -> Path:
    if not GENERATION_RUNS_DIR.exists():
        raise FileNotFoundError(f"No generation runs at {GENERATION_RUNS_DIR}")
    run_dirs = sorted(GENERATION_RUNS_DIR.iterdir())
    if not run_dirs:
        raise FileNotFoundError("No generation runs found.")
    return run_dirs[-1] / "golden_summaries.json"


def main():
    parser = argparse.ArgumentParser(description="Evaluate all summary variants")
    parser.add_argument("--summaries-path", type=Path, default=None)
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Variants to evaluate (default: all in file).")
    args = parser.parse_args()

    # Load golden summaries
    summaries_path = args.summaries_path or find_latest_summaries()
    print(f"\nLoading summaries from: {summaries_path}")
    with open(summaries_path) as f:
        golden = json.load(f)

    available_variants = list(golden[0]["summaries"].keys())
    variants = args.variants or available_variants
    variants = [v for v in variants if v in available_variants]

    print(f"  {len(golden)} fables, variants: {variants}")

    # Build subset
    fables = load_fables()
    morals = load_morals()
    gt_m2f = load_qrels_moral_to_fable()

    summary_indices = {int(item["id"].split("_")[1]) for item in golden}
    n_fables = max(summary_indices) + 1
    fable_texts = [fables[i]["text"] for i in range(n_fables)]

    moral_indices = sorted(gt_m2f.keys())
    subset_morals = []
    subset_gt = {}
    q_idx = 0
    for m_idx in moral_indices:
        fable_idx = gt_m2f[m_idx]
        if fable_idx in summary_indices:
            subset_morals.append(morals[m_idx]["text"])
            subset_gt[q_idx] = fable_idx
            q_idx += 1

    print(f"  {n_fables} fables in corpus, {len(subset_morals)} moral queries")

    # Load embedding model once
    print(f"\nLoading {EMBED_MODEL_ID}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL_ID)

    def enc(texts, label, query_instruction=None):
        return encode_with_cache(
            model=model, texts=texts, model_id=EMBED_MODEL_ID,
            cache_dir=CACHE_DIR, query_instruction=query_instruction, label=label,
        )

    # Encode morals once
    print("\nEncoding morals...")
    moral_embs = enc(subset_morals, label="moral queries", query_instruction=QUERY_INSTRUCTION)

    # Baseline once
    print("\n--- Baseline: raw fable ---")
    fable_embs = enc(fable_texts, label="raw fables")
    baseline = compute_metrics(moral_embs, fable_embs, subset_gt)
    print(f"  MRR={baseline['MRR']:.4f}  R@1={baseline['Recall@1']:.3f}  R@10={baseline['Recall@10']:.3f}")

    all_results = {"baseline_raw_fable": baseline}

    # Evaluate each variant
    for variant in variants:
        lookup = {int(item["id"].split("_")[1]): item["summaries"][variant] for item in golden}
        summaries = [lookup.get(i, "") for i in range(n_fables)]

        # Config A: summary only
        print(f"\n--- {variant}: summary only ---")
        sum_embs = enc(summaries, label=f"{variant} summaries")
        m_a = compute_metrics(moral_embs, sum_embs, subset_gt)
        print(f"  MRR={m_a['MRR']:.4f}  R@1={m_a['Recall@1']:.3f}  R@10={m_a['Recall@10']:.3f}")

        # Config B: fable + summary
        print(f"--- {variant}: fable + summary ---")
        enriched = [f"{fable_texts[i]}\n\nMoral summary: {summaries[i]}" for i in range(n_fables)]
        enr_embs = enc(enriched, label=f"{variant} fable+summary")
        m_b = compute_metrics(moral_embs, enr_embs, subset_gt)
        print(f"  MRR={m_b['MRR']:.4f}  R@1={m_b['Recall@1']:.3f}  R@10={m_b['Recall@10']:.3f}")

        all_results[f"{variant}__summary_only"] = m_a
        all_results[f"{variant}__fable_plus_summary"] = m_b

    # Summary table
    print(f"\n{'═' * 80}")
    print(f"  SUMMARY ({len(subset_morals)} queries, {n_fables} fables)")
    print(f"{'═' * 80}")
    print(f"  {'Config':<45} {'MRR':>8} {'R@1':>8} {'R@10':>8}")
    print(f"  {'─' * 72}")

    for label, r in sorted(all_results.items(), key=lambda x: x[1]["MRR"], reverse=True):
        delta = r["MRR"] - baseline["MRR"]
        d = f"{'+' if delta >= 0 else ''}{delta:.3f}" if label != "baseline_raw_fable" else "—"
        print(f"  {label:<45} {r['MRR']:>8.4f} {r['Recall@1']:>7.1%} {r['Recall@10']:>7.1%}  {d:>7}")

    # Save
    run_dir = summaries_path.parent
    out_path = run_dir / "retrieval_results_all_variants.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
