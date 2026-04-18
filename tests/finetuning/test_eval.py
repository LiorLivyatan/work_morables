"""Tests for finetuning/lib/eval.py"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib.eval import evaluate


def _mock_model(*embedding_batches: np.ndarray) -> MagicMock:
    """Model whose encode() returns each array in sequence."""
    model = MagicMock()
    model.encode.side_effect = list(embedding_batches)
    return model


def _identity_embeddings(n: int) -> np.ndarray:
    """n×n identity matrix: each text maps to a unique unit vector."""
    return np.eye(n, dtype=np.float32)


# ── Correctness ───────────────────────────────────────────────────────────────


def test_perfect_retrieval():
    """Identity embeddings → every moral retrieves its fable at rank 0."""
    n = 4
    model = _mock_model(_identity_embeddings(n), _identity_embeddings(n))
    gt = {i: i for i in range(n)}
    metrics = evaluate(model, [f"m{i}" for i in range(n)], [f"d{i}" for i in range(n)], gt)
    assert metrics["MRR"] == pytest.approx(1.0)
    assert metrics["Recall@1"] == pytest.approx(1.0)


def test_worst_case_retrieval():
    """Correct doc has the lowest cosine similarity — lands at last rank."""
    # moral points along x-axis; wrong docs are close in x, correct doc is orthogonal
    moral_embs = np.array([[1.0, 0.0]], dtype=np.float32)
    doc_embs = np.array([
        [0.99, 0.0],   # rank 0 — most similar, wrong
        [0.98, 0.0],   # rank 1 — wrong
        [0.0,  1.0],   # rank 2 — correct, orthogonal (lowest similarity)
    ], dtype=np.float32)
    model = _mock_model(moral_embs, doc_embs)
    metrics = evaluate(model, ["moral"], ["d0", "d1", "d2"], {0: 2})
    assert metrics["MRR"] == pytest.approx(1.0 / 3)
    assert metrics["Recall@1"] == pytest.approx(0.0)
    assert metrics["Recall@5"] == pytest.approx(1.0)   # top-5 covers all 3 docs


def test_returns_expected_metric_keys():
    n = 3
    model = _mock_model(_identity_embeddings(n), _identity_embeddings(n))
    metrics = evaluate(model, ["a", "b", "c"], ["x", "y", "z"], {0: 0, 1: 1, 2: 2})
    for key in ("MRR", "MAP", "R-Precision", "Mean Rank", "Recall@1", "Recall@5", "n_queries"):
        assert key in metrics


# ── Embedding caching ─────────────────────────────────────────────────────────


def test_saves_embeddings_to_cache(tmp_path):
    n = 3
    model = _mock_model(_identity_embeddings(n), _identity_embeddings(n))
    evaluate(model, ["a", "b", "c"], ["x", "y", "z"], {0: 0, 1: 1, 2: 2}, cache_dir=tmp_path)
    assert (tmp_path / "moral_embs.npy").exists()
    assert (tmp_path / "doc_embs.npy").exists()


def test_loads_embeddings_from_cache(tmp_path):
    """Once cached, encode() should not be called again."""
    n = 3
    embs = _identity_embeddings(n)
    np.save(tmp_path / "moral_embs.npy", embs)
    np.save(tmp_path / "doc_embs.npy", embs)

    model = MagicMock()
    evaluate(model, ["a", "b", "c"], ["x", "y", "z"], {0: 0, 1: 1, 2: 2}, cache_dir=tmp_path)

    model.encode.assert_not_called()


def test_cached_embeddings_give_same_metrics(tmp_path):
    """Results must be identical whether embeddings are freshly computed or cached."""
    n = 4
    embs = _identity_embeddings(n)
    moral_texts = [f"m{i}" for i in range(n)]
    doc_texts = [f"d{i}" for i in range(n)]
    gt = {i: i for i in range(n)}

    # First run — computes and caches
    first_metrics = evaluate(
        _mock_model(embs.copy(), embs.copy()), moral_texts, doc_texts, gt, cache_dir=tmp_path
    )
    # Second run — loads from cache
    second_metrics = evaluate(
        MagicMock(), moral_texts, doc_texts, gt, cache_dir=tmp_path
    )

    assert first_metrics["MRR"] == pytest.approx(second_metrics["MRR"])
    assert first_metrics["Recall@1"] == pytest.approx(second_metrics["Recall@1"])


def test_force_reencodes_ignoring_cache(tmp_path):
    """force=True must call encode() even when cache files exist."""
    n = 3
    embs = _identity_embeddings(n)
    np.save(tmp_path / "moral_embs.npy", embs)
    np.save(tmp_path / "doc_embs.npy", embs)

    model = _mock_model(embs.copy(), embs.copy())
    evaluate(model, ["a", "b", "c"], ["x", "y", "z"], {0: 0, 1: 1, 2: 2}, cache_dir=tmp_path, force=True)

    assert model.encode.call_count == 2


def test_no_cache_dir_writes_no_files(tmp_path):
    """Without cache_dir, no files should be written anywhere."""
    n = 3
    model = _mock_model(_identity_embeddings(n), _identity_embeddings(n))
    evaluate(model, ["a", "b", "c"], ["x", "y", "z"], {0: 0, 1: 1, 2: 2}, cache_dir=None)
    assert not list(tmp_path.iterdir())
