"""Pure numpy: cosine similarity, ranking, group MRR. No torch, no transformers."""
from __future__ import annotations
import numpy as np


def compute_rankings(query_embs: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """Return (n_queries, n_docs) array of doc indices in descending similarity order.

    Embeddings are assumed L2-normalised so similarity = inner product.
    """
    sims = query_embs @ doc_embs.T
    return np.argsort(-sims, axis=1)


def mrr_at_k(rankings: np.ndarray, gt_indices: np.ndarray, k: int = 10) -> float:
    """Mean Reciprocal Rank at k. rankings[i] is the doc-id ranking for query i."""
    n_q = rankings.shape[0]
    rr = np.zeros(n_q, dtype=np.float64)
    for i in range(n_q):
        positions = np.where(rankings[i, :k] == gt_indices[i])[0]
        if len(positions) == 1:
            rr[i] = 1.0 / (positions[0] + 1)
    return float(rr.mean())


def group_mrr(rankings: np.ndarray, gt_indices: np.ndarray,
              target_query_mask: np.ndarray, k: int = 10) -> tuple[float, float]:
    """MRR separately on queries whose GT fable is target-tagged vs not.

    Returns (mrr_target, mrr_control).
    """
    target_idx = np.where(target_query_mask)[0]
    ctrl_idx   = np.where(~target_query_mask)[0]

    mrr_t = mrr_at_k(rankings[target_idx], gt_indices[target_idx], k=k) if len(target_idx) else float("nan")
    mrr_c = mrr_at_k(rankings[ctrl_idx],  gt_indices[ctrl_idx],  k=k) if len(ctrl_idx)  else float("nan")
    return mrr_t, mrr_c
