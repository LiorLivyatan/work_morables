# scripts/10_sentence_level_retrieval.py
"""
Sentence-level chunking retrieval: moral → fable.

Instead of one embedding per fable, embed each sentence separately.
A moral query matches against individual sentences; the fable with the
best-matching sentence wins.

Chunking strategies:
  - sentence:     Split on sentence boundaries (nltk or regex)
  - last_n:       Only embed the last N sentences (where morals are often stated)
  - sliding:      Sliding window of W sentences with stride S

Aggregation strategies:
  - max:          Fable score = max similarity across its chunks
  - top_k_mean:   Fable score = mean of top-k chunk similarities
  - weighted:     Weight later chunks higher (moral tends to be at the end)

Usage:
  python scripts/10_sentence_level_retrieval.py
  python scripts/10_sentence_level_retrieval.py --strategies sentence__max sentence__last3
  python scripts/10_sentence_level_retrieval.py --sample 50
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from retrieval_utils import compute_rankings

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Baselines for comparison ─────────────────────────────────────────────────
BASELINES = {
    "Linq-Embed-Mistral (full fable)": {"MRR": 0.2105, "Recall@1": 0.141},
}

# ── Sentence splitting ───────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (avoids nltk dependency)."""
    # Split on period, exclamation, question mark followed by space or end
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings and very short fragments
    return [s.strip() for s in sents if len(s.strip()) > 5]


def chunk_fable(text: str, strategy: str) -> list[str]:
    """
    Chunk a fable text according to the given strategy.

    Returns a list of text chunks to embed.
    """
    sentences = split_sentences(text)
    if not sentences:
        return [text]  # fallback: whole text as one chunk

    if strategy == "sentence":
        # Each sentence is a chunk
        return sentences

    elif strategy.startswith("last_"):
        # last_N: only the last N sentences
        n = int(strategy.split("_")[1])
        return sentences[-n:] if len(sentences) >= n else sentences

    elif strategy.startswith("sliding_"):
        # sliding_W_S: window of W sentences, stride S
        parts = strategy.split("_")
        window = int(parts[1])
        stride = int(parts[2]) if len(parts) > 2 else 1
        chunks = []
        for i in range(0, len(sentences), stride):
            chunk = " ".join(sentences[i:i + window])
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text]

    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def aggregate_scores(chunk_scores: np.ndarray, method: str) -> float:
    """
    Aggregate similarity scores from multiple chunks into a single fable score.

    Args:
        chunk_scores: array of cosine similarities between query and each chunk
        method: aggregation method
    """
    if method == "max":
        return float(np.max(chunk_scores))

    elif method.startswith("top_") and method.endswith("_mean"):
        # top_k_mean
        k = int(method.split("_")[1])
        top_k = np.sort(chunk_scores)[::-1][:k]
        return float(np.mean(top_k))

    elif method == "weighted":
        # Weight later chunks higher (linear ramp)
        n = len(chunk_scores)
        weights = np.linspace(0.5, 1.0, n)
        return float(np.max(chunk_scores * weights))

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ── Strategies to test ───────────────────────────────────────────────────────

STRATEGIES = {
    # (chunking_strategy, aggregation_method, description)
    "sentence__max":        ("sentence",    "max",          "Each sentence, take max"),
    "sentence__top_3_mean": ("sentence",    "top_3_mean",   "Each sentence, mean of top-3"),
    "last_1__max":          ("last_1",      "max",          "Last sentence only"),
    "last_3__max":          ("last_3",      "max",          "Last 3 sentences, take max"),
    "last_5__max":          ("last_5",      "max",          "Last 5 sentences, take max"),
    "sliding_3_2__max":     ("sliding_3_2", "max",          "3-sentence window stride 2, max"),
    "sentence__weighted":   ("sentence",    "weighted",     "Each sentence, weighted toward end"),
}


# ── Metrics (adapted for chunk-level retrieval) ─────────────────────────────

def compute_chunked_metrics(
    query_embeddings: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunk_to_fable: list[int],
    ground_truth: dict[int, int],
    aggregation: str,
    ks: tuple = (1, 5, 10, 50),
) -> dict:
    """
    Compute retrieval metrics with chunk-level corpus.

    For each query, compute similarity to all chunks, aggregate per-fable,
    then rank fables.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # query (N) x chunk (C) similarity matrix
    sim_matrix = cosine_similarity(query_embeddings, chunk_embeddings)

    # Build fable → chunk indices mapping
    n_fables = max(chunk_to_fable) + 1
    fable_chunks: dict[int, list[int]] = {f: [] for f in range(n_fables)}
    for chunk_idx, fable_idx in enumerate(chunk_to_fable):
        fable_chunks[fable_idx].append(chunk_idx)

    reciprocal_ranks = []
    recall_at_k = {k: [] for k in ks}
    precision_at_k = {k: [] for k in ks}
    ndcg_at_k = {k: [] for k in ks}
    r_precisions = []
    all_ranks = []

    for q_idx in range(len(query_embeddings)):
        if q_idx not in ground_truth:
            continue
        correct_fable = ground_truth[q_idx]

        # Aggregate chunk scores per fable
        fable_scores = np.zeros(n_fables)
        for fable_idx in range(n_fables):
            c_indices = fable_chunks[fable_idx]
            if c_indices:
                fable_scores[fable_idx] = aggregate_scores(
                    sim_matrix[q_idx, c_indices], aggregation
                )

        # Rank fables by aggregated score
        ranked_fables = np.argsort(-fable_scores)
        rank = int(np.where(ranked_fables == correct_fable)[0][0])

        reciprocal_ranks.append(1.0 / (rank + 1))
        all_ranks.append(rank + 1)
        for k in ks:
            hit = 1.0 if rank < k else 0.0
            recall_at_k[k].append(hit)
            precision_at_k[k].append(hit / k)
            ndcg_at_k[k].append(1.0 / np.log2(rank + 2) if rank < k else 0.0)
        r_precisions.append(1.0 if rank == 0 else 0.0)

    results = {
        "MRR": float(np.mean(reciprocal_ranks)),
        "MAP": float(np.mean(reciprocal_ranks)),
        "R-Precision": float(np.mean(r_precisions)),
        "Mean Rank": float(np.mean(all_ranks)),
        "Median Rank": float(np.median(all_ranks)),
        "n_queries": len(reciprocal_ranks),
    }
    for k in ks:
        results[f"Recall@{k}"] = float(np.mean(recall_at_k[k]))
        results[f"P@{k}"] = float(np.mean(precision_at_k[k]))
        results[f"NDCG@{k}"] = float(np.mean(ndcg_at_k[k]))
    return results


# ── Main experiment ──────────────────────────────────────────────────────────

def run_experiment(strategy_keys: list[str], sample: int | None = None):
    """Run sentence-level retrieval with specified strategies."""

    # Load data
    with open(DATA_DIR / "fables_corpus.json") as f:
        fables_corpus = json.load(f)
    with open(DATA_DIR / "morals_corpus.json") as f:
        morals_corpus = json.load(f)
    with open(DATA_DIR / "qrels_moral_to_fable.json") as f:
        qrels = json.load(f)
        gt_m2f = {
            int(q["query_id"].split("_")[1]): int(q["doc_id"].split("_")[1])
            for q in qrels
        }

    fable_texts = [f["text"] for f in fables_corpus]
    moral_indices = sorted(gt_m2f.keys())
    moral_texts = [morals_corpus[i]["text"] for i in moral_indices]
    gt_subset = {idx: gt_m2f[idx] for idx in moral_indices}

    if sample:
        moral_indices = moral_indices[:sample]
        moral_texts = moral_texts[:sample]
        gt_subset = {i: gt_subset[moral_indices[i]]
                     for i, idx in enumerate(moral_indices)}
        # Remap to 0-based
        gt_subset = {i: gt_m2f[moral_indices[i]] for i in range(len(moral_indices))}

    print(f"  {len(fable_texts)} fables, {len(moral_texts)} moral queries")

    # Load embedding model (Linq-Embed-Mistral — our best baseline)
    print("\n  Loading Linq-Embed-Mistral...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("Linq-AI-Research/Linq-Embed-Mistral")

    TASK = "Given a text, retrieve the most relevant passage that answers the query"

    def encode(texts: list[str], is_query: bool = False, batch_size: int = 32):
        if is_query:
            texts = [f"Instruct: {TASK}\nQuery: {t}" for t in texts]
        return model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)

    # Encode moral queries once
    print("\n  Encoding moral queries...")
    moral_embs = encode(moral_texts, is_query=True)
    print(f"  Moral embeddings: {moral_embs.shape}")

    # Also compute baseline (full-fable) for direct comparison
    print("\n  Encoding full fables (baseline)...")
    t0 = time.time()
    full_fable_embs = encode(fable_texts)
    baseline_time = time.time() - t0

    from retrieval_utils import compute_metrics
    baseline_metrics = compute_metrics(moral_embs, full_fable_embs, gt_subset)
    print(f"  Baseline: MRR={baseline_metrics['MRR']:.4f}  "
          f"R@1={baseline_metrics['Recall@1']:.3f}  "
          f"R@10={baseline_metrics['Recall@10']:.3f}")

    all_results = [{
        "strategy": "full_fable (baseline)",
        "chunking": "none",
        "aggregation": "none",
        "description": "Full fable text, single embedding per fable",
        "n_chunks": len(fable_texts),
        "avg_chunks_per_fable": 1.0,
        "encode_time_s": round(baseline_time, 1),
        **baseline_metrics,
    }]

    # Run each chunking strategy
    for strat_key in strategy_keys:
        if strat_key not in STRATEGIES:
            print(f"\n  WARNING: Unknown strategy '{strat_key}', skipping.")
            continue

        chunking, aggregation, description = STRATEGIES[strat_key]
        print(f"\n  {'═' * 60}")
        print(f"  Strategy: {strat_key}")
        print(f"    Chunking: {chunking}  |  Aggregation: {aggregation}")
        print(f"    {description}")

        # Chunk all fables
        all_chunks = []
        chunk_to_fable = []  # maps chunk index → fable index
        for fable_idx, fable_text in enumerate(fable_texts):
            chunks = chunk_fable(fable_text, chunking)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_to_fable.append(fable_idx)

        n_chunks = len(all_chunks)
        avg_chunks = n_chunks / len(fable_texts)
        print(f"    Total chunks: {n_chunks}  "
              f"(avg {avg_chunks:.1f} per fable)")

        # Encode chunks
        t0 = time.time()
        chunk_embs = encode(all_chunks)
        encode_time = time.time() - t0
        print(f"    Encoded in {encode_time:.1f}s  ({chunk_embs.shape})")

        # Compute metrics
        metrics = compute_chunked_metrics(
            moral_embs, chunk_embs, chunk_to_fable, gt_subset, aggregation
        )

        print(f"    MRR={metrics['MRR']:.4f}  "
              f"R@1={metrics['Recall@1']:.3f}  "
              f"R@10={metrics['Recall@10']:.3f}  "
              f"Med.Rank={metrics['Median Rank']:.0f}")

        delta_mrr = metrics["MRR"] - baseline_metrics["MRR"]
        print(f"    vs baseline: {'+'if delta_mrr>=0 else ''}{delta_mrr:.4f} MRR")

        all_results.append({
            "strategy": strat_key,
            "chunking": chunking,
            "aggregation": aggregation,
            "description": description,
            "n_chunks": n_chunks,
            "avg_chunks_per_fable": round(avg_chunks, 1),
            "encode_time_s": round(encode_time, 1),
            **metrics,
        })

    return all_results


def make_run_dir():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d = RESULTS_DIR / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentence-level chunking retrieval")
    parser.add_argument(
        "--strategies", nargs="+", choices=list(STRATEGIES.keys()),
        default=list(STRATEGIES.keys()),
        help="Which strategies to run (default: all).",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Evaluate only N queries for quick testing.",
    )
    args = parser.parse_args()

    print(f"\nSentence-Level Chunking Retrieval Experiment")
    print(f"  {len(args.strategies)} strategies to test")
    if args.sample:
        print(f"  Sample mode: {args.sample} queries")
    print()

    results = run_experiment(args.strategies, args.sample)

    # Save results
    run_dir = make_run_dir()
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'═' * 80}")
    print(f"  SUMMARY")
    print(f"{'═' * 80}")
    print(f"  {'Strategy':<30} {'MRR':>8} {'R@1':>8} {'R@10':>8} {'Chunks':>8}")
    print(f"  {'─' * 66}")
    for r in sorted(results, key=lambda x: x["MRR"], reverse=True):
        print(f"  {r['strategy']:<30} {r['MRR']:>8.4f} "
              f"{r['Recall@1']:>7.1%} {r['Recall@10']:>7.1%} "
              f"{r['n_chunks']:>8}")

    print(f"\n  Results saved to {run_dir}")
