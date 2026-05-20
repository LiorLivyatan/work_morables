"""
Step 2 of the pipeline: find tag values whose presence is statistically
overrepresented in retrieval failures. Uses Fisher's exact test (one-sided,
greater-than) and BH-FDR for multiple comparisons.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def failure_overrep_per_tag(tag_present: np.ndarray, failed: np.ndarray) -> float:
    """One-sided Fisher's exact: is failure overrepresented among tagged fables?"""
    a = int((tag_present & failed).sum())
    b = int((tag_present & ~failed).sum())
    c = int((~tag_present & failed).sum())
    d = int((~tag_present & ~failed).sum())
    _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
    return float(pval)


def bh_fdr(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg. Returns boolean array of rejection decisions."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool)
    cutoff_rank = np.where(below)[0].max()
    rejected_sorted = np.zeros(n, dtype=bool)
    rejected_sorted[: cutoff_rank + 1] = True
    rejected = np.zeros(n, dtype=bool)
    rejected[order] = rejected_sorted
    return rejected


def rank_problematic_concepts(
    *,
    tag_index: dict[str, dict[str, set[str]]],
    fable_doc_ids: list[str],
    failed_doc_ids: set[str],
    min_tagged_fables: int,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        field, value, n_tagged, n_failed_in_tag, p_value, fdr_significant
    sorted by p_value ascending.
    """
    fable_idx = {fid: i for i, fid in enumerate(fable_doc_ids)}
    n_total = len(fable_doc_ids)
    failed_mask = np.zeros(n_total, dtype=bool)
    for fid in failed_doc_ids:
        if fid in fable_idx:
            failed_mask[fable_idx[fid]] = True

    rows = []
    for field, tag_map in tag_index.items():
        for value, doc_set in tag_map.items():
            if len(doc_set) < min_tagged_fables:
                continue
            tag_mask = np.zeros(n_total, dtype=bool)
            for fid in doc_set:
                if fid in fable_idx:
                    tag_mask[fable_idx[fid]] = True
            p = failure_overrep_per_tag(tag_mask, failed_mask)
            rows.append({
                "field": field,
                "value": value,
                "n_tagged": int(tag_mask.sum()),
                "n_failed_in_tag": int((tag_mask & failed_mask).sum()),
                "p_value": p,
            })

    df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    if len(df):
        df["fdr_significant"] = bh_fdr(df["p_value"].to_numpy(), alpha=fdr_alpha)
    return df


def per_tag_baseline_mrr(
    *,
    tag_index: dict[str, dict[str, set[str]]],
    fable_doc_ids: list[str],
    moral_gt_idx,
    rankings: np.ndarray,
    min_tagged_fables: int,
    k: int = 10,
) -> pd.DataFrame:
    """For each (field, value) with sample size above min_tagged, compute MRR@k
    on queries whose ground-truth fable is tagged with that value. Used to pick
    the difficulty-matched placebo.

    moral_gt_idx may be list[int] (legacy single-target) or list[list[int]]
    (multi-target). For multi-target, a moral is "tagged" iff any of its
    relevant fables is tagged."""
    from .retrieval import mrr_at_k
    # Normalise to list[list[int]]
    if len(moral_gt_idx) and isinstance(moral_gt_idx[0], (int, np.integer)):
        gts = [[int(g)] for g in moral_gt_idx]
    else:
        gts = [[int(x) for x in g] for g in moral_gt_idx]
    fable_idx = {fid: i for i, fid in enumerate(fable_doc_ids)}
    rows = []
    for field, tag_map in tag_index.items():
        for value, doc_set in tag_map.items():
            if len(doc_set) < min_tagged_fables:
                continue
            tagged_fable_indices = {fable_idx[fid] for fid in doc_set if fid in fable_idx}
            target_query_mask = np.array([
                any(g in tagged_fable_indices for g in gt) for gt in gts
            ])
            if target_query_mask.sum() == 0:
                continue
            sub_gt = [gts[i] for i in np.where(target_query_mask)[0]]
            mrr = mrr_at_k(rankings[target_query_mask], sub_gt, k=k)
            rows.append({
                "field": field, "value": value,
                "n_tagged_queries": int(target_query_mask.sum()),
                "baseline_mrr": mrr,
            })
    return pd.DataFrame(rows)
