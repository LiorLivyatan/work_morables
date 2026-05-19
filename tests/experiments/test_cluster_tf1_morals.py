import sys
import importlib.util
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PATH = Path(__file__).parent.parent.parent / "experiments" / "11_tf1_diagnostic" / "cluster_tf1_morals.py"
cl = _load("cluster_tf1_morals", _PATH)


def test_agglomerative_clusters_merges_above_threshold():
    # 3 items: 0 and 1 similar, 2 far
    sim = np.array([
        [1.0, 0.95, 0.10],
        [0.95, 1.0, 0.05],
        [0.10, 0.05, 1.0],
    ])
    clusters = cl.agglomerative_clusters(sim, threshold=0.80)
    sets = {frozenset(c) for c in clusters}
    assert sets == {frozenset({0, 1}), frozenset({2})}


def test_agglomerative_clusters_all_singletons_below_threshold():
    sim = np.array([[1.0, 0.5], [0.5, 1.0]])
    clusters = cl.agglomerative_clusters(sim, threshold=0.80)
    sets = {frozenset(c) for c in clusters}
    assert sets == {frozenset({0}), frozenset({1})}


def test_classify_cluster_type_singleton():
    sim = np.eye(3)
    assert cl.classify_cluster_type([0], sim) == "singleton"


def test_classify_cluster_type_exact():
    sim = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert cl.classify_cluster_type([0, 1], sim) == "exact"


def test_classify_cluster_type_near():
    sim = np.array([[1.0, 0.85], [0.85, 1.0]])
    assert cl.classify_cluster_type([0, 1], sim) == "near"


def test_pick_canonical_text_picks_highest_count():
    texts = ["alpha", "beta", "gamma"]
    counts = {"alpha": 10, "beta": 50, "gamma": 5}
    assert cl.pick_canonical_text([0, 1, 2], texts, counts) == "beta"


def test_pick_canonical_text_tie_breaks_by_lowest_index():
    texts = ["alpha", "beta", "gamma"]
    counts = {"alpha": 10, "beta": 10, "gamma": 5}
    assert cl.pick_canonical_text([0, 1, 2], texts, counts) == "alpha"


def test_classify_cluster_type_mixed_near_and_exact_is_near():
    # A-B is exact (1.0), A-C is near (0.85), B-C is near (0.85)
    # Cluster {A, B, C} should be "near", not "exact"
    sim = np.array([
        [1.0, 1.0, 0.85],
        [1.0, 1.0, 0.85],
        [0.85, 0.85, 1.0],
    ])
    assert cl.classify_cluster_type([0, 1, 2], sim) == "near"


def test_classify_cluster_type_all_three_exact_is_exact():
    sim = np.ones((3, 3))
    assert cl.classify_cluster_type([0, 1, 2], sim) == "exact"


def test_agglomerative_clusters_empty_matrix_returns_empty():
    assert cl.agglomerative_clusters(np.empty((0, 0)), threshold=0.8) == []


def test_agglomerative_clusters_single_item_returns_singleton_cluster():
    assert cl.agglomerative_clusters(np.array([[1.0]]), threshold=0.8) == [[0]]


def test_build_clustered_outputs_basic():
    morals = [
        {"doc_id": "moral_tf1_000", "text": "honesty is best"},
        {"doc_id": "moral_tf1_001", "text": "honesty wins"},
        {"doc_id": "moral_tf1_002", "text": "greed leads to downfall"},
    ]
    qmf_processed = [
        {"query_id": "moral_tf1_000", "doc_id": "fable_tf1_00000", "relevance": 1},
        {"query_id": "moral_tf1_000", "doc_id": "fable_tf1_00001", "relevance": 1},
        {"query_id": "moral_tf1_001", "doc_id": "fable_tf1_00002", "relevance": 1},
        {"query_id": "moral_tf1_002", "doc_id": "fable_tf1_00003", "relevance": 1},
    ]
    sim = np.array([
        [1.0, 0.92, 0.10],
        [0.92, 1.0, 0.15],
        [0.10, 0.15, 1.0],
    ])
    counts = {"honesty is best": 30, "honesty wins": 50, "greed leads to downfall": 100}

    out = cl.build_clustered_outputs(
        morals=morals, qmf=qmf_processed, sim_matrix=sim,
        counts=counts, threshold=0.80,
    )

    # 2 clusters expected: {honesty is best, honesty wins}, {greed}
    assert len(out["cluster_mapping.json"]) == 2
    canonical_texts = {u["text"] for u in out["morals_unique_corpus.json"]}
    # higher-count "honesty wins" should be canonical for the merged cluster
    assert "honesty wins" in canonical_texts and "honesty is best" not in canonical_texts
    assert "greed leads to downfall" in canonical_texts

    # Each fable maps to exactly one cluster moral in the inverse qrels
    qfm = out["qrels_fable_to_moral_clustered.json"]
    fable_ids = {q["query_id"] for q in qfm}
    assert fable_ids == {f["doc_id"] for f in qmf_processed}
    counts_per_fable = Counter(q["query_id"] for q in qfm)
    assert all(c == 1 for c in counts_per_fable.values())
