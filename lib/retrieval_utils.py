# scripts/retrieval_utils.py
"""Shared retrieval metrics used across experiment scripts."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_metrics(query_embeddings, corpus_embeddings, ground_truth, ks=(1, 5, 10, 50)):
    """
    Compute retrieval metrics for a set of queries against a corpus.

    Args:
        query_embeddings: (N, D) float32 array, L2-normalized
        corpus_embeddings: (M, D) float32 array, L2-normalized
        ground_truth: dict mapping query_idx (int) -> correct corpus_idx (int)
        ks: tuple of k values for Recall@k, P@k, NDCG@k

    Returns:
        dict with keys: MRR, MAP, R-Precision, Mean Rank, Median Rank,
                        Recall@k, P@k, NDCG@k for each k, n_queries

    Notes:
        - MAP == MRR when each query has exactly 1 relevant document.
        - R-Precision == Recall@1 when R=1.
        - P@k = (1/k) if correct doc in top-k, else 0.  ≠ Recall@k for k > 1.
        - NDCG@k uses binary relevance; ideal score is 1/log2(2) = 1 at rank 0.
    """
    sim_matrix = cosine_similarity(query_embeddings, corpus_embeddings)
    rankings = np.argsort(-sim_matrix, axis=1)  # descending by similarity

    reciprocal_ranks = []
    recall_at_k = {k: [] for k in ks}
    precision_at_k = {k: [] for k in ks}
    ndcg_at_k = {k: [] for k in ks}
    r_precisions = []

    for q_idx in range(len(query_embeddings)):
        if q_idx not in ground_truth:
            continue
        correct_idx = ground_truth[q_idx]
        ranked = rankings[q_idx]
        rank = int(np.where(ranked == correct_idx)[0][0])  # 0-indexed

        reciprocal_ranks.append(1.0 / (rank + 1))
        for k in ks:
            hit = 1.0 if rank < k else 0.0
            recall_at_k[k].append(hit)
            # P@k: fraction of top-k that are relevant (1/k when the doc is found)
            precision_at_k[k].append(hit / k)
            # NDCG@k: gain=1, discount=log2(rank+2); ideal DCG = 1/log2(2) = 1
            ndcg_at_k[k].append(1.0 / np.log2(rank + 2) if rank < k else 0.0)
        r_precisions.append(1.0 if rank == 0 else 0.0)

    ranks_1indexed = [1.0 / rr for rr in reciprocal_ranks]
    results = {
        "MRR": float(np.mean(reciprocal_ranks)),
        # MAP == MRR when each query has exactly 1 relevant doc; computed explicitly
        "MAP": float(np.mean(reciprocal_ranks)),
        "R-Precision": float(np.mean(r_precisions)),
        "Mean Rank": float(np.mean(ranks_1indexed)),
        "Median Rank": float(np.median(ranks_1indexed)),
        "n_queries": len(reciprocal_ranks),
    }
    for k in ks:
        results[f"Recall@{k}"] = float(np.mean(recall_at_k[k]))
        results[f"P@{k}"] = float(np.mean(precision_at_k[k]))
        results[f"NDCG@{k}"] = float(np.mean(ndcg_at_k[k]))
    return results


def compute_metrics_from_matrix(
    score_matrix: np.ndarray,
    ground_truth: dict,
    ks=(1, 5, 10, 50),
) -> dict:
    """
    Compute retrieval metrics from a pre-computed score matrix.

    Identical metric keys to compute_metrics(). Use this for fused score matrices
    where embeddings are not available directly.

    Args:
        score_matrix: (N_queries, N_docs) float array, higher = more relevant
        ground_truth: dict mapping query_idx (int) -> correct corpus_idx (int)
        ks: tuple of k values for Recall@k, P@k, NDCG@k

    Returns:
        dict with same keys as compute_metrics()
    """
    rankings = np.argsort(-score_matrix, axis=1)

    reciprocal_ranks = []
    recall_at_k = {k: [] for k in ks}
    precision_at_k = {k: [] for k in ks}
    ndcg_at_k = {k: [] for k in ks}
    r_precisions = []

    for q_idx, correct_idx in ground_truth.items():
        ranked = rankings[q_idx]
        rank = int(np.where(ranked == correct_idx)[0][0])

        reciprocal_ranks.append(1.0 / (rank + 1))
        for k in ks:
            hit = 1.0 if rank < k else 0.0
            recall_at_k[k].append(hit)
            precision_at_k[k].append(hit / k)
            ndcg_at_k[k].append(1.0 / np.log2(rank + 2) if rank < k else 0.0)
        r_precisions.append(1.0 if rank == 0 else 0.0)

    ranks_1indexed = [1.0 / rr for rr in reciprocal_ranks]
    results = {
        "MRR": float(np.mean(reciprocal_ranks)),
        "MAP": float(np.mean(reciprocal_ranks)),
        "R-Precision": float(np.mean(r_precisions)),
        "Mean Rank": float(np.mean(ranks_1indexed)),
        "Median Rank": float(np.median(ranks_1indexed)),
        "n_queries": len(reciprocal_ranks),
    }
    for k in ks:
        results[f"Recall@{k}"] = float(np.mean(recall_at_k[k]))
        results[f"P@{k}"] = float(np.mean(precision_at_k[k]))
        results[f"NDCG@{k}"] = float(np.mean(ndcg_at_k[k]))
    return results


def compute_rankings(query_embeddings, corpus_embeddings, top_k=100):
    """
    Return top-k corpus indices and similarity scores for each query.

    Args:
        query_embeddings: (N, D) float32 array, L2-normalized
        corpus_embeddings: (M, D) float32 array, L2-normalized
        top_k: number of top results to return per query

    Returns:
        list of dicts (one per query) with keys:
            'indices': list[int]  — top-k corpus indices, best first
            'scores':  list[float] — corresponding cosine similarities (rounded to 6 dp)
    """
    sim_matrix = cosine_similarity(query_embeddings, corpus_embeddings)
    top_k = min(top_k, sim_matrix.shape[1])
    results = []
    for q_idx in range(len(query_embeddings)):
        ranked_indices = np.argsort(-sim_matrix[q_idx])[:top_k]
        scores = sim_matrix[q_idx][ranked_indices]
        results.append({
            "indices": ranked_indices.tolist(),
            "scores": [round(float(s), 6) for s in scores],
        })
    return results


def rank_analysis(query_embeddings, corpus_embeddings, ground_truth):
    """Return 0-indexed rank of the correct doc for each query."""
    sim_matrix = cosine_similarity(query_embeddings, corpus_embeddings)
    rankings = np.argsort(-sim_matrix, axis=1)
    ranks = []
    for q_idx in range(len(query_embeddings)):
        if q_idx not in ground_truth:
            continue
        correct_idx = ground_truth[q_idx]
        rank = int(np.where(rankings[q_idx] == correct_idx)[0][0])
        ranks.append(rank)
    return np.array(ranks)
