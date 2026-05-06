"""
analysis/lib/loader.py — shared data loading for all analysis scripts.

Every analysis script imports from here so embedding + dataset loading is
consistent across experiments. The only thing that changes between runs is
which .npy files you point at.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.data import (
    load_fables,
    load_morals,
    load_qrels_moral_to_fable,
)


# ── Experiment config ──────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Bundles everything needed to analyse one experiment run.

    Configurable per run:
      - moral_embs / doc_embs: paths to the .npy embedding files
      - label: human-readable name for plots and filenames
      - corpus_config: which document format was embedded (for labelling only)
      - result_json: optional path to the experiment's result JSON (for cross-ref)

    Fixed across all experiments:
      - The dataset (fables, morals, qrels) — always loaded from lib/data.py
    """
    moral_embs_path: str | Path
    doc_embs_path:   str | Path
    label:           str                  # e.g. "ft07-linq-s500-fable+summary"
    corpus_config:   str = "unknown"      # e.g. "fable_plus_summary", "raw"
    result_json:     str | Path | None = None

    def __post_init__(self):
        self.moral_embs_path = Path(self.moral_embs_path)
        self.doc_embs_path   = Path(self.doc_embs_path)
        if self.result_json:
            self.result_json = Path(self.result_json)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset():
    """Load the fixed MORABLES dataset.

    Returns
    -------
    fables : list[dict]   — 709 fables, each {doc_id, title, text, alias}
    morals : list[dict]   — 709 morals, each {doc_id, text, fable_id}
    qrels  : dict[int,int]— {moral_idx: fable_idx}, 0-based contiguous indices
    """
    raw_fables = load_fables()
    raw_morals = load_morals()
    raw_qrels  = load_qrels_moral_to_fable()

    # qrels keys are raw moral indices; re-index to contiguous 0-based
    moral_indices = sorted(raw_qrels.keys())
    qrels = {i: raw_qrels[idx] for i, idx in enumerate(moral_indices)}

    return raw_fables, raw_morals, qrels


def load_embeddings(cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load and return (moral_embs, doc_embs) as float32 numpy arrays.

    Assumes embeddings are already L2-normalised (cosine sim = dot product).
    Shape: (n_queries, D) and (n_docs, D).
    """
    moral_embs = np.load(cfg.moral_embs_path).astype(np.float32)
    doc_embs   = np.load(cfg.doc_embs_path).astype(np.float32)
    return moral_embs, doc_embs


# ── Ranking ────────────────────────────────────────────────────────────────────

def compute_rankings(
    moral_embs: np.ndarray,
    doc_embs:   np.ndarray,
    qrels:      dict[int, int],
) -> list[dict]:
    """Compute per-query retrieval results via dot-product similarity.

    Returns a list of dicts (one per query), each with:
      - query_idx      : int
      - gt_fable_idx   : int
      - ranked_indices : list[int]   — all fable indices, best-first
      - scores         : list[float] — corresponding similarity scores
      - gt_rank        : int         — 1-based rank of ground-truth fable
      - gt_score       : float       — similarity score of ground-truth fable
      - top1_score     : float       — similarity score of rank-1 fable
      - score_gap      : float       — top1_score - gt_score (0 if gt is rank 1)

    Fixed: dot-product similarity (assumes L2-normalised embeddings).
    """
    # Full similarity matrix: (n_queries, n_docs)
    sim = moral_embs @ doc_embs.T  # shape (709, 709)

    results = []
    for q_idx, gt_fable in qrels.items():
        scores_row = sim[q_idx]  # (n_docs,)
        ranked = np.argsort(-scores_row).tolist()
        ranked_scores = scores_row[ranked].tolist()

        gt_rank  = ranked.index(gt_fable) + 1   # 1-based
        gt_score = float(scores_row[gt_fable])
        top1_score = ranked_scores[0]

        results.append({
            "query_idx":      q_idx,
            "gt_fable_idx":   gt_fable,
            "ranked_indices": ranked,
            "scores":         ranked_scores,
            "gt_rank":        gt_rank,
            "gt_score":       gt_score,
            "top1_score":     top1_score,
            "score_gap":      top1_score - gt_score,
        })

    return results


def compute_mrr(rankings: list[dict], k: int = 10) -> float:
    """Compute MRR@k from the output of compute_rankings()."""
    rr_sum = 0.0
    for r in rankings:
        if r["gt_rank"] <= k:
            rr_sum += 1.0 / r["gt_rank"]
    return rr_sum / len(rankings)
