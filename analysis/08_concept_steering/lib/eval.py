"""
Statistical tests for the specificity gap S = ΔMRR_target − ΔMRR_control.
Also: summarize_run (build the dict the plotting module consumes) and
stage2_go_no_go (the §9 four-condition decision rule).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def reciprocal_rank_per_query(rankings: np.ndarray, gt_indices: np.ndarray,
                                k: int = 10) -> np.ndarray:
    """Per-query 1/rank capped at k (rr=0 if gt-rank > k). Default k=10 so the
    aggregate matches MRR@10 — keep the `mrr_at_10` label in artifacts truthful."""
    n_q = rankings.shape[0]
    rr = np.zeros(n_q, dtype=np.float64)
    for i in range(n_q):
        positions = np.where(rankings[i, :k] == gt_indices[i])[0]
        if len(positions) == 1:
            rr[i] = 1.0 / (positions[0] + 1)
    return rr


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
) -> dict:
    """Build the dict consumed by plotting.plot_specificity_summary."""
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
            cell_dict[layer_key] = {
                "S_median": S_median, "S_ci_lo": S_lo, "S_ci_hi": S_hi,
                "null_lo": null_lo, "null_hi": null_hi,
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
    cond3 = bool(summary.get("random_dir_within_null", False))
    cond4 = bool(summary.get("passing_pooled_cosine_max", 1.0) < pooled_cos_threshold)

    reasons = []
    if not cond1: reasons.append(f"only {len(targets_passing)}/{len(target_concepts)} targets pass")
    if not cond2: reasons.append("placebo passed specificity (criterion 2 failed)")
    if not cond3: reasons.append("random-direction control did not stay within null envelope")
    if not cond4: reasons.append("pooled cosine ≥ 0.99 (intervention may be magnitude-only)")
    return {
        "go": all([cond1, cond2, cond3, cond4]),
        "reasons": reasons,
        "targets_passing": targets_passing,
    }
