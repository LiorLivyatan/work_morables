"""
Orchestrate the (concept × layer × α) intervention sweep.

For each cell, runs encode_with_intervention on all fables, computes
moral→fable rankings, computes per-query reciprocal rank, and persists.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .model import EncoderHandle, encode_with_intervention
from .retrieval import compute_rankings
from .eval import reciprocal_rank_per_query
from .io import save_json


def sweep_concept(
    *,
    handle: EncoderHandle,
    fable_texts: list[str],
    moral_embs: np.ndarray,
    gt_indices: np.ndarray,
    concept_name: str,
    direction_per_layer: dict[int, np.ndarray],
    layers: list[int],
    alphas: list[float],
    output_dir: Path,
    batch_size: int = 4,
) -> dict:
    """For one concept, run the full (layer × α) sweep."""
    summary: dict = {"concept": concept_name, "cells": []}

    for layer in layers:
        layer_resolved = handle.n_layers + layer if layer < 0 else layer
        direction = direction_per_layer[layer_resolved]
        for alpha in alphas:
            embs, pooled_cos = encode_with_intervention(
                handle, fable_texts,
                layer_idx=layer, direction=direction, alpha=alpha,
                batch_size=batch_size, renormalize=True,
            )
            rankings = compute_rankings(moral_embs, embs)
            rr = reciprocal_rank_per_query(rankings, gt_indices)

            ranks_int = np.zeros(len(gt_indices), dtype=np.int32)
            for i in range(len(gt_indices)):
                pos = np.where(rankings[i] == gt_indices[i])[0]
                ranks_int[i] = (pos[0] + 1) if len(pos) else len(rankings[i]) + 1

            cell_path = Path(output_dir) / f"{concept_name}_layer{layer_resolved}_alpha{alpha:+.2f}.json"
            save_json(cell_path, {
                "concept": concept_name, "layer": int(layer_resolved), "alpha": float(alpha),
                "mrr_at_10": float(rr.mean()),
                "rr_per_query": rr.tolist(),
                "ranks_intervened": ranks_int.tolist(),
                "pooled_cosine_mean": float(pooled_cos.mean()),
                "pooled_cosine_min":  float(pooled_cos.min()),
            })
            summary["cells"].append({
                "layer": int(layer_resolved), "alpha": float(alpha),
                "rr_per_query_path": str(cell_path),
                "mrr_at_10": float(rr.mean()),
                "pooled_cosine_mean": float(pooled_cos.mean()),
            })
    return summary
