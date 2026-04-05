# tests/test_retrieval_utils.py
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from retrieval_utils import compute_metrics, rank_analysis
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.retrieval_utils import compute_metrics_from_matrix


def test_perfect_retrieval():
    """Identity matrix: each query matches exactly its own corpus item."""
    q = np.eye(4, dtype=np.float32)
    c = np.eye(4, dtype=np.float32)
    gt = {0: 0, 1: 1, 2: 2, 3: 3}
    m = compute_metrics(q, c, gt)
    assert m["MRR"] == pytest.approx(1.0)
    assert m["MAP"] == pytest.approx(1.0)
    assert m["R-Precision"] == pytest.approx(1.0)
    assert m["Recall@1"] == pytest.approx(1.0)
    assert m["n_queries"] == 4


def test_worst_retrieval():
    """Query matches corpus item ranked last (rank 2 out of 3)."""
    q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    c = np.array([
        [0.99, 0.0, 0.0],  # rank 0 - wrong
        [0.98, 0.0, 0.0],  # rank 1 - wrong
        [0.0,  1.0, 0.0],  # rank 2 - correct but least similar
    ], dtype=np.float32)
    gt = {0: 2}
    m = compute_metrics(q, c, gt, ks=(1, 3))
    assert m["Recall@1"] == pytest.approx(0.0)
    assert m["Recall@3"] == pytest.approx(1.0)
    assert m["MRR"] == pytest.approx(1.0 / 3)


def test_recall_at_k_boundary():
    """Correct item at rank k is NOT within top-k; rank k-1 IS."""
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    c = np.array([
        [0.95, 0.0],  # rank 0
        [0.90, 0.0],  # rank 1
        [0.85, 0.0],  # rank 2
        [0.80, 0.0],  # rank 3
        [0.75, 0.0],  # rank 4 - correct
    ], dtype=np.float32)
    gt = {0: 4}
    m = compute_metrics(q, c, gt, ks=(1, 4, 5))
    assert m["Recall@1"] == pytest.approx(0.0)
    assert m["Recall@4"] == pytest.approx(0.0)   # rank 4 not in top-4 (0-3)
    assert m["Recall@5"] == pytest.approx(1.0)   # rank 4 IS in top-5


def test_rank_analysis():
    q = np.eye(3, dtype=np.float32)
    c = np.eye(3, dtype=np.float32)
    gt = {0: 0, 1: 1, 2: 2}
    ranks = rank_analysis(q, c, gt)
    assert list(ranks) == [0, 0, 0]


def test_skips_missing_queries():
    """Queries not in ground_truth are silently skipped."""
    q = np.eye(3, dtype=np.float32)
    c = np.eye(3, dtype=np.float32)
    gt = {0: 0, 2: 2}  # query 1 missing
    m = compute_metrics(q, c, gt)
    assert m["n_queries"] == 2
    assert m["MRR"] == pytest.approx(1.0)


def test_compute_metrics_from_matrix_perfect():
    """Identity score matrix: every query hits rank 0."""
    matrix = np.eye(4, dtype=np.float32)
    gt = {0: 0, 1: 1, 2: 2, 3: 3}
    m = compute_metrics_from_matrix(matrix, gt)
    assert m["MRR"] == pytest.approx(1.0)
    assert m["Recall@1"] == pytest.approx(1.0)
    assert m["n_queries"] == 4


def test_compute_metrics_from_matrix_matches_compute_metrics():
    """Results must be identical to compute_metrics() on normalized embeddings."""
    rng = np.random.default_rng(42)
    q = rng.random((5, 8)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    c = rng.random((10, 8)).astype(np.float32)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    gt = {0: 0, 1: 3, 2: 7, 3: 1, 4: 9}

    from sklearn.metrics.pairwise import cosine_similarity
    matrix = cosine_similarity(q, c).astype(np.float32)

    m_emb = compute_metrics(q, c, gt)
    m_mat = compute_metrics_from_matrix(matrix, gt)

    for key in ["MRR", "MAP", "R-Precision", "Recall@1", "Recall@5", "n_queries"]:
        assert m_emb[key] == pytest.approx(m_mat[key], abs=1e-5), f"Mismatch on {key}"
