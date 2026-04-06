"""
run.py — Evaluate symmetric moral matching for 10-fable pilot.

Step 0: Re-run exp07's best config (conceptual_abstract__summary_only)
        on the same 10-fable subset to establish the comparison baseline.

New configs:
  A:        ground_truth_style summary only, original moral query
  B:        declarative_universal summary only, original moral query
  A+expand: ground_truth_style summary only, max-score over 4 query paraphrases
  B+expand: declarative_universal summary only, max-score over 4 query paraphrases
  RRF-all:  Reciprocal Rank Fusion over ranked lists of all 5 configs above

Usage:
  python experiments/08_symmetric_moral_matching/run.py
  python experiments/08_symmetric_moral_matching/run.py --run-dir path/to/run_dir
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent

# ── Paths ────────────────────────────────────────────────────────────────────
EXP07_SUMMARIES = (
    ROOT_DIR / "experiments" / "07_sota_summarization_oracle"
    / "results" / "generation_runs" / "full_709" / "golden_summaries.json"
)
RUNS_DIR = Path(__file__).parent / "results" / "generation_runs"

EMBED_MODEL_ID = "Linq-AI-Research/Linq-Embed-Mistral"
QUERY_INSTRUCTION = "Given a text, retrieve the most relevant passage that answers the query"


# ── Fusion helpers ────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(score_matrices: list, k: int = 60) -> np.ndarray:
    """
    Merge score matrices using Reciprocal Rank Fusion.

    Args:
        score_matrices: list of (n_queries, n_docs) float arrays, higher = better
        k: RRF constant (default 60)

    Returns:
        fused: (n_queries, n_docs) float64 array, higher = better
    """
    n_queries, n_docs = score_matrices[0].shape
    fused = np.zeros((n_queries, n_docs), dtype=np.float64)

    for scores in score_matrices:
        ranked = np.argsort(-scores, axis=1)
        rank_matrix = np.empty_like(ranked)
        for q in range(n_queries):
            rank_matrix[q, ranked[q]] = np.arange(1, n_docs + 1)
        fused += 1.0 / (k + rank_matrix)

    return fused


def max_score_fusion(score_matrices: list) -> np.ndarray:
    """
    Return elementwise maximum across score matrices.

    Args:
        score_matrices: list of (n_queries, n_docs) float arrays

    Returns:
        (n_queries, n_docs) float array, elementwise max
    """
    return np.maximum.reduce(score_matrices)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_lib():
    """Lazily load lib modules to avoid import errors when only fusion helpers are used."""
    sys.path.insert(0, str(ROOT_DIR / "lib"))
    from data import load_fables, load_morals, load_qrels_moral_to_fable
    from retrieval_utils import compute_metrics
    return load_fables, load_morals, load_qrels_moral_to_fable, compute_metrics


def build_subset(n_fables: int):
    """
    Return data restricted to the first n_fables fable indices.

    Returns:
        fable_texts:   list[str], length n_fables
        moral_texts:   list[str], one per matching moral
        ground_truth:  dict[int, int], {query_idx: fable_idx}, 0-based contiguous
        moral_indices: list[int], original moral indices (for loading expansions)
    """
    load_fables, load_morals, load_qrels_moral_to_fable, _ = _load_lib()
    fables = load_fables()
    morals = load_morals()
    gt_m2f = load_qrels_moral_to_fable()

    fable_texts = [fables[i]["text"] for i in range(n_fables)]
    target_fable_indices = set(range(n_fables))

    moral_entries = sorted(
        [(moral_idx, fable_idx) for moral_idx, fable_idx in gt_m2f.items()
         if fable_idx in target_fable_indices],
        key=lambda x: x[0]
    )

    moral_texts = [morals[m_idx]["text"] for m_idx, _ in moral_entries]
    moral_indices = [m_idx for m_idx, _ in moral_entries]
    ground_truth = {i: fable_idx for i, (_, fable_idx) in enumerate(moral_entries)}

    return fable_texts, moral_texts, ground_truth, moral_indices


def find_latest_run_dir() -> Path:
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"No run dirs at {RUNS_DIR}")
    run_dirs = sorted(RUNS_DIR.iterdir())
    if not run_dirs:
        raise FileNotFoundError("No run dirs found.")
    return run_dirs[-1]


# ── Encoding ─────────────────────────────────────────────────────────────────

def encode(model, texts: list, is_query: bool = False) -> np.ndarray:
    """Encode texts, optionally prepending query instruction."""
    if is_query:
        texts = [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {t}" for t in texts]
    return model.encode(
        texts, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    ).astype(np.float32)


# ── Metrics from pre-computed score matrix ────────────────────────────────────

def _metrics_from_scores(score_matrix: np.ndarray, ground_truth: dict) -> dict:
    """Compute retrieval metrics directly from a pre-computed score matrix."""
    rankings = np.argsort(-score_matrix, axis=1)
    reciprocal_ranks = []
    recall_at_1 = []

    for q_idx, correct_doc_idx in ground_truth.items():
        ranked = rankings[q_idx]
        rank = int(np.where(ranked == correct_doc_idx)[0][0])
        reciprocal_ranks.append(1.0 / (rank + 1))
        recall_at_1.append(1.0 if rank == 0 else 0.0)

    return {
        "MRR": float(np.mean(reciprocal_ranks)),
        "Recall@1": float(np.mean(recall_at_1)),
        "n_queries": len(reciprocal_ranks),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run experiment 08 retrieval evaluation")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Run dir with corpus_summaries.json + query_expansions.json")
    parser.add_argument("--n-fables", type=int, default=10,
                        help="Pilot size (default: 10)")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir()
    n_fables = args.n_fables

    print(f"\n08_symmetric_moral_matching — Retrieval Evaluation")
    print(f"  Pilot size: {n_fables} fables")
    print(f"  Run dir: {run_dir}")

    _, _, _, compute_metrics = _load_lib()

    fable_texts, moral_texts, ground_truth, moral_indices = build_subset(n_fables)
    print(f"  {len(fable_texts)} fables, {len(moral_texts)} moral queries")

    with open(run_dir / "corpus_summaries.json") as f:
        corpus_data = json.load(f)
    with open(run_dir / "query_expansions.json") as f:
        expansion_data = json.load(f)

    corpus_lookup = {int(item["id"].split("_")[1]): item["summaries"]
                     for item in corpus_data}
    expansion_lookup = {item["moral_idx"]: item["paraphrases"]
                        for item in expansion_data}

    def get_corpus_texts(variant: str) -> list:
        return [corpus_lookup.get(i, {}).get(variant, "") for i in range(n_fables)]

    print(f"\n  Loading {EMBED_MODEL_ID}...")
    import torch
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")
    model = SentenceTransformer(EMBED_MODEL_ID, device=device)

    # Step 0: exp07 baseline
    print("\n─── Step 0: Exp 07 baseline (conceptual_abstract__summary_only) ───")
    with open(EXP07_SUMMARIES) as f:
        exp07_golden = json.load(f)
    exp07_lookup = {int(item["id"].split("_")[1]): item["summaries"].get("conceptual_abstract", "")
                    for item in exp07_golden}
    exp07_corpus_texts = [exp07_lookup.get(i, "") for i in range(n_fables)]

    moral_embs = encode(model, moral_texts, is_query=True)
    exp07_corpus_embs = encode(model, exp07_corpus_texts)
    step0_metrics = compute_metrics(moral_embs, exp07_corpus_embs, ground_truth)
    print(f"  Baseline R@1={step0_metrics['Recall@1']:.3f}  MRR={step0_metrics['MRR']:.4f}")

    all_results = {"step0_baseline_exp07": step0_metrics}
    score_matrices_for_rrf = []

    configs = [
        ("A", "ground_truth_style", False),
        ("B", "declarative_universal", False),
        ("A_expand", "ground_truth_style", True),
        ("B_expand", "declarative_universal", True),
    ]

    for config_name, variant, use_expansion in configs:
        print(f"\n─── Config {config_name}: {variant}"
              f"{' + query expansion' if use_expansion else ''} ───")

        corpus_embs = encode(model, get_corpus_texts(variant))
        score_matrix = moral_embs @ corpus_embs.T

        if use_expansion:
            paraphrase_matrices = [score_matrix]
            for para_variant in ["moral_rephrase", "moral_elaborate", "moral_abstract"]:
                para_texts = [
                    expansion_lookup.get(moral_indices[q_idx], {}).get(para_variant, moral_texts[q_idx])
                    for q_idx in range(len(moral_texts))
                ]
                para_embs = encode(model, para_texts, is_query=True)
                paraphrase_matrices.append(para_embs @ corpus_embs.T)
            score_matrix = max_score_fusion(paraphrase_matrices)

        metrics = _metrics_from_scores(score_matrix, ground_truth)
        delta = metrics["Recall@1"] - step0_metrics["Recall@1"]
        print(f"  R@1={metrics['Recall@1']:.3f}  MRR={metrics['MRR']:.4f}"
              f"  (vs baseline: {'+' if delta >= 0 else ''}{delta:.3f})")

        all_results[config_name] = metrics
        score_matrices_for_rrf.append(score_matrix)

    # RRF fusion
    print("\n─── RRF-all: Fusion of all 4 configs ───")
    fused_scores = reciprocal_rank_fusion(score_matrices_for_rrf, k=60)
    rrf_metrics = _metrics_from_scores(fused_scores, ground_truth)
    delta = rrf_metrics["Recall@1"] - step0_metrics["Recall@1"]
    print(f"  R@1={rrf_metrics['Recall@1']:.3f}  MRR={rrf_metrics['MRR']:.4f}"
          f"  (vs baseline: {'+' if delta >= 0 else ''}{delta:.3f})")
    all_results["RRF_all"] = rrf_metrics

    # Summary table
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY ({n_fables}-fable pilot)")
    print(f"{'═' * 70}")
    print(f"  {'Config':<30} {'R@1':>8} {'MRR':>8} {'vs baseline':>12}")
    print(f"  {'─' * 60}")
    for label, r in sorted(all_results.items(), key=lambda x: x[1]["Recall@1"], reverse=True):
        delta = r["Recall@1"] - step0_metrics["Recall@1"]
        d_str = f"{'+' if delta >= 0 else ''}{delta:.3f}" if label != "step0_baseline_exp07" else "—"
        print(f"  {label:<30} {r['Recall@1']:>7.1%} {r['MRR']:>8.4f} {d_str:>12}")

    out_path = run_dir / "retrieval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
