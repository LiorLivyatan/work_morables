"""Pure numpy: cosine similarity, ranking, group MRR. No torch, no transformers.

Multi-target qrels: gt may be a 1D array of single indices (legacy single-target)
or a list of int-lists (one set of relevant doc indices per query). All metrics
treat multi-target as "best rank among relevant" — standard IR semantics.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np


def compute_rankings(query_embs: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """Return (n_queries, n_docs) array of doc indices in descending similarity order.

    Embeddings are assumed L2-normalised so similarity = inner product.
    """
    sims = query_embs @ doc_embs.T
    return np.argsort(-sims, axis=1)


def _gt_as_lists(gt) -> list[set[int]]:
    """Normalise gt input to a list of int sets, one per query.

    Accepts: 1D np.ndarray of ints (legacy), list[int], list[list[int]],
    list[set[int]], or object-dtype np.ndarray of any of the above.
    """
    if isinstance(gt, np.ndarray) and gt.dtype != object and gt.ndim == 1:
        return [{int(x)} for x in gt]
    out: list[set[int]] = []
    for g in gt:
        if isinstance(g, (int, np.integer)):
            out.append({int(g)})
        else:
            out.append({int(x) for x in g})
    return out


def mrr_at_k(rankings: np.ndarray, gt_indices, k: int = 10) -> float:
    """Mean Reciprocal Rank at k. rankings[i] is the doc-id ranking for query i.

    Multi-target: rr_i = 1 / (best_position+1) over the relevant set; 0 if no
    relevant doc appears in the top-k.
    """
    gt = _gt_as_lists(gt_indices)
    n_q = rankings.shape[0]
    if len(gt) != n_q:
        raise ValueError(f"gt length {len(gt)} != rankings rows {n_q}")
    rr = np.zeros(n_q, dtype=np.float64)
    for i in range(n_q):
        gt_set = gt[i]
        # Walk the top-k ranking and stop at the first hit.
        for pos in range(min(k, rankings.shape[1])):
            if int(rankings[i, pos]) in gt_set:
                rr[i] = 1.0 / (pos + 1)
                break
    return float(rr.mean())


def group_mrr(rankings: np.ndarray, gt_indices,
              target_query_mask: np.ndarray, k: int = 10) -> tuple[float, float]:
    """MRR separately on queries whose GT relevant set intersects the target tag vs not.

    target_query_mask is precomputed by the caller (any-of-relevant logic lives there).
    Returns (mrr_target, mrr_control).
    """
    gt = _gt_as_lists(gt_indices)
    target_idx = np.where(target_query_mask)[0]
    ctrl_idx   = np.where(~target_query_mask)[0]

    def _sub(idx):
        return [gt[i] for i in idx]

    mrr_t = mrr_at_k(rankings[target_idx], _sub(target_idx), k=k) if len(target_idx) else float("nan")
    mrr_c = mrr_at_k(rankings[ctrl_idx],   _sub(ctrl_idx),   k=k) if len(ctrl_idx)  else float("nan")
    return mrr_t, mrr_c
