"""Tests for exp08 run.py helper functions."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments" / "08_symmetric_moral_matching"))
from run import reciprocal_rank_fusion, max_score_fusion


class TestReciprocalRankFusion:

    def test_single_matrix_identity(self):
        """With one score matrix, RRF ranks match cosine ranking."""
        scores = np.array([
            [1.0, 0.5, 0.2],
            [0.3, 1.0, 0.4],
            [0.1, 0.2, 1.0],
        ], dtype=np.float32)

        fused = reciprocal_rank_fusion([scores], k=60)

        assert np.argmax(fused[0]) == 0
        assert np.argmax(fused[1]) == 1
        assert np.argmax(fused[2]) == 2

    def test_two_agreeing_matrices_amplify(self):
        """Two matrices that agree should produce the same ranking but higher fused scores."""
        scores = np.array([
            [1.0, 0.5],
            [0.3, 1.0],
        ], dtype=np.float32)

        fused_one = reciprocal_rank_fusion([scores], k=60)
        fused_two = reciprocal_rank_fusion([scores, scores], k=60)

        assert np.argmax(fused_one[0]) == np.argmax(fused_two[0])
        assert np.argmax(fused_one[1]) == np.argmax(fused_two[1])
        np.testing.assert_allclose(fused_two, 2 * fused_one)

    def test_two_disagreeing_matrices_merge(self):
        """Two opposing rankings produce a merged result."""
        scores_a = np.array([[1.0, 0.1]], dtype=np.float32)
        scores_b = np.array([[0.1, 1.0]], dtype=np.float32)

        fused = reciprocal_rank_fusion([scores_a, scores_b], k=60)

        assert abs(fused[0, 0] - fused[0, 1]) < 0.001

    def test_output_shape(self):
        """Output shape matches input."""
        scores = np.random.rand(5, 8).astype(np.float32)
        fused = reciprocal_rank_fusion([scores, scores], k=60)
        assert fused.shape == (5, 8)

    def test_k_parameter_affects_scores(self):
        """Larger k reduces the spread between high and low rank scores."""
        scores = np.array([[1.0, 0.5, 0.1]], dtype=np.float32)
        fused_k10 = reciprocal_rank_fusion([scores], k=10)
        fused_k100 = reciprocal_rank_fusion([scores], k=100)

        spread_k10 = float(np.max(fused_k10) - np.min(fused_k10))
        spread_k100 = float(np.max(fused_k100) - np.min(fused_k100))
        assert spread_k10 > spread_k100


class TestMaxScoreFusion:

    def test_elementwise_max(self):
        """Returns elementwise max across all matrices."""
        a = np.array([[0.9, 0.2], [0.1, 0.8]], dtype=np.float32)
        b = np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float32)

        fused = max_score_fusion([a, b])

        np.testing.assert_allclose(fused, np.array([[0.9, 0.7], [0.6, 0.8]]))

    def test_single_matrix_passthrough(self):
        """Single matrix is returned unchanged."""
        a = np.array([[0.5, 0.3]], dtype=np.float32)
        fused = max_score_fusion([a])
        np.testing.assert_allclose(fused, a)

    def test_output_shape(self):
        matrices = [np.random.rand(4, 6).astype(np.float32) for _ in range(3)]
        fused = max_score_fusion(matrices)
        assert fused.shape == (4, 6)
