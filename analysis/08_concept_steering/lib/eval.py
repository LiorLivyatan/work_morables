"""
Statistical tests for the specificity gap S = ΔMRR_target − ΔMRR_control.
Also: summarize_run (build the dict the plotting module consumes) and
stage2_go_no_go (the §9 four-condition decision rule).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def reciprocal_rank_per_query(rankings: np.ndarray, gt_indices,
                                k: int = 10) -> np.ndarray:
    """Per-query 1/best-rank capped at k (rr=0 if no relevant doc in top-k).
    Multi-target: gt_indices can be 1D array of ints or list of int-lists.
    Default k=10 so aggregate matches MRR@10 (keep the `mrr_at_10` label truthful)."""
    from .retrieval import _gt_as_lists
    gt = _gt_as_lists(gt_indices)
    n_q = rankings.shape[0]
    if len(gt) != n_q:
        raise ValueError(f"gt length {len(gt)} != rankings rows {n_q}")
    rr = np.zeros(n_q, dtype=np.float64)
    for i in range(n_q):
        gt_set = gt[i]
        for pos in range(min(k, rankings.shape[1])):
            if int(rankings[i, pos]) in gt_set:
                rr[i] = 1.0 / (pos + 1)
                break
    return rr


def best_rank_per_query(rankings: np.ndarray, gt_indices) -> np.ndarray:
    """Per-query best (1-based) rank of any relevant fable; len(rankings[i])+1 if none in topN.
    Used by ranks_baseline.json's gt_rank field under multi-target semantics."""
    from .retrieval import _gt_as_lists
    gt = _gt_as_lists(gt_indices)
    n_q = rankings.shape[0]
    n_d = rankings.shape[1]
    out = np.full(n_q, n_d + 1, dtype=np.int32)
    for i in range(n_q):
        gt_set = gt[i]
        for pos in range(n_d):
            if int(rankings[i, pos]) in gt_set:
                out[i] = pos + 1
                break
    return out


def paired_bootstrap_ci_specificity_gap(
    *,
    rr_target_base: np.ndarray,
    rr_target_intv: np.ndarray,
    rr_control_base: np.ndarray,
    rr_control_intv: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI of S = ΔMRR_target − ΔMRR_control.

    Resamples WITH REPLACEMENT independently within target and control groups.
    Pairing within group is preserved (per-query Δ first, then averages).
    """
    rng = np.random.default_rng() if rng is None else rng
    n_t, n_c = len(rr_target_base), len(rr_control_base)
    delta_t = rr_target_intv - rr_target_base
    delta_c = rr_control_intv - rr_control_base

    bootstrap_S = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx_t = rng.integers(0, n_t, size=n_t)
        idx_c = rng.integers(0, n_c, size=n_c)
        bootstrap_S[b] = delta_t[idx_t].mean() - delta_c[idx_c].mean()

    lo = float(np.quantile(bootstrap_S, alpha / 2))
    hi = float(np.quantile(bootstrap_S, 1 - alpha / 2))
    return lo, hi


def ci_excludes_zero_negative(ci: tuple[float, float]) -> bool:
    """True iff the CI lies entirely below 0 (concept-specific suppression)."""
    return ci[1] < 0.0


def ci_excludes_zero(ci: tuple[float, float]) -> bool:
    """Sign-agnostic: True iff CI excludes 0 (either entirely above or entirely below)."""
    lo, hi = ci
    return lo > 0.0 or hi < 0.0


def passing_lift(S_ci_lo: float, S_ci_hi: float) -> bool:
    """Sign-agnostic 'this cell produced a specificity gap': CI excludes 0."""
    return ci_excludes_zero((S_ci_lo, S_ci_hi))


def passing_null(S_median: float, null_lo, null_hi) -> bool | None:
    """True iff |S_median| beats the null envelope (shuffled-tag primary).
    Returns None if null is unavailable (cell did not run as candidate)."""
    if null_lo is None or null_hi is None:
        return None
    return (S_median > null_hi) or (S_median < null_lo)


def exploratory_decision(summary: dict) -> dict:
    """Replacement for stage2_go_no_go when concepts.placebo is empty.

    Computes per-concept passing flags from the summary dict; emits a no-gate
    'go: null' verdict for stage2_decision.json.
    """
    per_concept: dict[str, dict] = {}
    for concept, cells in summary.get("cells", {}).items():
        passing_strict_cells: list[tuple[int, float]] = []
        for layer_key, lc in cells.items():
            alphas = summary["alphas"]
            for j, alpha in enumerate(alphas):
                lo = lc["S_ci_lo"][j]; hi = lc["S_ci_hi"][j]
                pl = passing_lift(lo, hi)
                S_med = lc["S_median"][j]
                pn = passing_null(S_med, lc.get("null_lo", [None]*len(alphas))[j],
                                          lc.get("null_hi", [None]*len(alphas))[j])
                if pl and (pn is True):
                    layer_int = int(layer_key) if layer_key.lstrip("-").isdigit() else layer_key
                    passing_strict_cells.append((layer_int, alpha))
        per_concept[concept] = {
            "passing_strict_cells": passing_strict_cells,
            "n_passing_strict": len(passing_strict_cells),
        }
    return {
        "mode": "exploratory",
        "go": None,
        "reasons": ["exploratory landscape; per-concept verdicts in specificity_summary.json"],
        "per_concept": per_concept,
    }


def _load_rr_for_cell(cells: list[dict], layer: int, alpha: float) -> np.ndarray:
    for c in cells:
        if c["layer"] == layer and abs(c["alpha"] - alpha) < 1e-9:
            with open(c["rr_per_query_path"]) as f:
                return np.array(json.load(f)["rr_per_query"], dtype=np.float64)
    raise KeyError(f"cell layer={layer} alpha={alpha} not found")


def summarize_run(
    *,
    cells_per_concept: dict[str, list[dict]],
    target_query_mask_per_concept: dict[str, np.ndarray],
    rr_baseline: np.ndarray,
    layers: list[int],
    alphas: list[float],
    null_envelopes: dict | None = None,
    n_bootstrap: int = 10000,
    rng_seed: int = 0,
    singleton_moral_mask: np.ndarray | None = None,
) -> dict:
    """Build the dict consumed by plotting.plot_specificity_summary.

    If singleton_moral_mask is provided (multi-target qrels), additionally
    compute the per-cell specificity gap restricted to morals with exactly 1
    relevant fable. A cell whose verdict flips under this restriction is
    flagged as cluster-fanout-sensitive (spec §7).
    """
    rng = np.random.default_rng(rng_seed)
    summary: dict = {
        "concept_order": list(cells_per_concept.keys()),
        "layers": [(l if l >= 0 else "last") for l in layers],
        "alphas": alphas,
        "cells":  {},
    }
    for concept, cells in cells_per_concept.items():
        target_mask = target_query_mask_per_concept[concept]
        cell_dict: dict[str, dict] = {}
        for layer in layers:
            layer_key = str(layer)
            S_median: list[float] = []
            S_lo:     list[float] = []
            S_hi:     list[float] = []
            null_lo:  list = []
            null_hi:  list = []
            S_singleton: list = []      # S restricted to singleton-cluster morals (cluster-fanout flag)
            for alpha in alphas:
                rr_int = _load_rr_for_cell(cells, layer, alpha)
                lo, hi = paired_bootstrap_ci_specificity_gap(
                    rr_target_base=rr_baseline[target_mask],
                    rr_target_intv=rr_int[target_mask],
                    rr_control_base=rr_baseline[~target_mask],
                    rr_control_intv=rr_int[~target_mask],
                    n_bootstrap=n_bootstrap, alpha=0.05, rng=rng,
                )
                S = (rr_int[target_mask] - rr_baseline[target_mask]).mean() \
                    - (rr_int[~target_mask] - rr_baseline[~target_mask]).mean()
                S_median.append(float(S))
                S_lo.append(lo); S_hi.append(hi)
                if null_envelopes and concept in null_envelopes \
                        and layer in null_envelopes[concept] \
                        and alpha in null_envelopes[concept][layer]:
                    nl, nh = null_envelopes[concept][layer][alpha]
                else:
                    nl = nh = None
                null_lo.append(nl); null_hi.append(nh)
                # Cluster-fanout sensitivity: same S but restricted to morals with
                # exactly 1 relevant fable. If singleton-only S is sign-flipped or
                # near-zero while the full S is the opposite, the cell's verdict
                # is driven by multi-target morals — flagged downstream.
                if singleton_moral_mask is not None:
                    tm_s = target_mask & singleton_moral_mask
                    cm_s = (~target_mask) & singleton_moral_mask
                    if tm_s.any() and cm_s.any():
                        S_s = (rr_int[tm_s] - rr_baseline[tm_s]).mean() \
                              - (rr_int[cm_s] - rr_baseline[cm_s]).mean()
                        S_singleton.append(float(S_s))
                    else:
                        S_singleton.append(None)
                else:
                    S_singleton.append(None)
            cell_dict[layer_key] = {
                "S_median": S_median, "S_ci_lo": S_lo, "S_ci_hi": S_hi,
                "null_lo": null_lo, "null_hi": null_hi,
                "S_singleton_only": S_singleton,
            }
        summary["cells"][concept] = cell_dict
    return summary


def stage2_go_no_go(
    summary: dict, *,
    target_concepts: list[str],
    placebo_concepts: list[str],
    pooled_cos_threshold: float = 0.99,
    min_targets_passing: int = 2,
) -> dict:
    """Apply spec §9 four-condition decision rule.

    Returns {go: bool, reasons: list[str], targets_passing: list[str]}.
    """
    targets_passing = []
    for concept in target_concepts:
        if concept not in summary["cells"]:
            continue
        cells = summary["cells"][concept]
        if any(any(hi is not None and hi < 0.0 for hi in lc["S_ci_hi"])
               for lc in cells.values()):
            targets_passing.append(concept)

    cond1 = len(targets_passing) >= min_targets_passing
    cond2 = all(
        not any(any(hi is not None and hi < 0.0 for hi in lc["S_ci_hi"])
                for lc in summary["cells"].get(p, {}).values())
        for p in placebo_concepts
    )
    # cond3 has three states: True (passed), False (failed), None (N/A — no candidate cells)
    rdwn = summary.get("random_dir_within_null")
    cond3 = bool(rdwn) if rdwn is not None else (not cond1)
    # When no targets passed (cond1 False), the random-direction check was never
    # run; treat cond3 as vacuously satisfied so the reasons list doesn't include
    # a misleading random-direction message on top of the real failure.
    cond4 = bool(summary.get("passing_pooled_cosine_max", 1.0) < pooled_cos_threshold)

    reasons = []
    if not cond1:
        reasons.append(f"only {len(targets_passing)}/{len(target_concepts)} targets pass")
    if not cond2:
        reasons.append("placebo passed specificity (criterion 2 failed)")
    if cond1 and rdwn is False:
        # Only report this when the check actually ran and failed.
        reasons.append("random-direction control did not stay within null envelope")
    if cond1 and not cond4:
        reasons.append("pooled cosine ≥ 0.99 (intervention may be magnitude-only)")
    return {
        "go": all([cond1, cond2, cond3, cond4]),
        "reasons": reasons,
        "targets_passing": targets_passing,
    }
