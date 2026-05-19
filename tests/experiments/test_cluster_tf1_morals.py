import sys
import importlib.util
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
