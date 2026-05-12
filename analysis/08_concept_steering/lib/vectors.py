"""
Concept-vector construction from hidden states.

Public surface:
    build_matched_pairs(...) -> list[(pos_id, neg_id)]
    build_caa_vector(pos_hidden_per_layer, neg_hidden_per_layer) -> dict[layer, ndarray]
    build_mean_diff_vector(hidden_per_layer, pos_indices, neg_indices) -> dict[layer, ndarray]
    cosine(a, b) -> float
    matched_pair_quality_metrics(...) -> dict
"""
from __future__ import annotations
import numpy as np


class MatchingFailure(RuntimeError):
    pass


def build_matched_pairs(
    *,
    positives: set[str],
    fable_doc_ids: list[str],
    metadata: list[dict],
    fable_token_lengths: list[int],
    match_fields: list[str],
    cross_field: str | None,
    length_tolerance: float,
) -> list[tuple[str, str]]:
    """For each positive fable, greedy-pick a negative matched on the requested fields.

    Each negative is used at most once. Pairs are formed in deterministic doc-id order.
    """
    md_by_id = {m["doc_id"]: m for m in metadata}
    len_by_id = dict(zip(fable_doc_ids, fable_token_lengths))
    used_negatives: set[str] = set()
    pairs: list[tuple[str, str]] = []

    for pos_id in sorted(positives):
        if pos_id not in md_by_id:
            continue
        pos_md = md_by_id[pos_id]
        pos_len = len_by_id.get(pos_id)
        if pos_len is None:
            continue

        for cand_id in fable_doc_ids:
            if cand_id == pos_id or cand_id in positives or cand_id in used_negatives:
                continue
            if cand_id not in md_by_id:
                continue
            cand_md = md_by_id[cand_id]
            if not _fields_match(pos_md, cand_md, match_fields):
                continue
            if cross_field and not _list_fields_share_value(pos_md, cand_md, cross_field):
                continue
            cand_len = len_by_id.get(cand_id)
            if cand_len is None or cand_len == 0 or pos_len == 0:
                continue
            ratio = min(pos_len, cand_len) / max(pos_len, cand_len)
            if ratio < (1 - length_tolerance):
                continue
            pairs.append((pos_id, cand_id))
            used_negatives.add(cand_id)
            break

    return pairs


def _fields_match(a: dict, b: dict, fields: list[str]) -> bool:
    for f in fields:
        if a.get(f) != b.get(f):
            return False
    return True


def _list_fields_share_value(a: dict, b: dict, field: str) -> bool:
    """True iff a and b share at least one value under `field`.
    Handles list, dict (compares value sets — e.g. character_roles), and scalar shapes."""
    av, bv = a.get(field), b.get(field)
    if av is None or bv is None:
        return False
    if isinstance(av, dict) and isinstance(bv, dict):
        return bool(set(av.values()) & set(bv.values()))
    if isinstance(av, list) and isinstance(bv, list):
        return bool(set(av) & set(bv))
    return av == bv


def build_caa_vector(
    pos_hidden_per_layer: dict[int, np.ndarray],
    neg_hidden_per_layer: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """v_C[layer] = mean over pairs of (h_pos − h_neg). Shape: (hidden_dim,)."""
    out = {}
    for layer in pos_hidden_per_layer:
        diffs = pos_hidden_per_layer[layer] - neg_hidden_per_layer[layer]
        out[layer] = diffs.mean(axis=0)
    return out


def build_mean_diff_vector(
    hidden_per_layer: dict[int, np.ndarray],
    pos_indices: np.ndarray,
    neg_indices: np.ndarray,
) -> dict[int, np.ndarray]:
    """v_C[layer] = mean(h | pos) − mean(h | neg). Shape: (hidden_dim,)."""
    out = {}
    for layer, h in hidden_per_layer.items():
        out[layer] = h[pos_indices].mean(axis=0) - h[neg_indices].mean(axis=0)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def matched_pair_quality_metrics(
    *,
    pairs: list[tuple[str, str]],
    metadata: list[dict],
    fable_token_lengths: list[int],
    fable_doc_ids: list[str],
    cross_field: str | None,
    pos_baseline_mrr: float,
    neg_baseline_mrr: float,
    cos_caa_meandiff_per_layer: dict[int, float],
) -> dict:
    md = {m["doc_id"]: m for m in metadata}
    lens = dict(zip(fable_doc_ids, fable_token_lengths))
    if not pairs:
        return {"n_matched_pairs": 0}

    setting_match = sum(1 for p, n in pairs if md[p].get("setting") == md[n].get("setting")) / len(pairs)
    type_match    = sum(1 for p, n in pairs if md[p].get("fable_type") == md[n].get("fable_type")) / len(pairs)
    length_ratios = [
        min(lens[p], lens[n]) / max(lens[p], lens[n]) for p, n in pairs
    ]
    cross_overlap = float("nan")
    if cross_field:
        cross_overlap = sum(
            1 for p, n in pairs if _list_fields_share_value(md[p], md[n], cross_field)
        ) / len(pairs)

    return {
        "n_matched_pairs": len(pairs),
        "mean_length_ratio": float(np.mean(length_ratios)),
        "setting_match_rate": float(setting_match),
        "fable_type_match_rate": float(type_match),
        "cross_field_overlap_rate": float(cross_overlap),
        "pos_baseline_mrr": float(pos_baseline_mrr),
        "neg_baseline_mrr": float(neg_baseline_mrr),
        "cos_caa_meandiff_per_layer": {str(l): float(c)
                                        for l, c in cos_caa_meandiff_per_layer.items()},
    }
