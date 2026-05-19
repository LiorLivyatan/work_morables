"""
Cluster the 100 unique morals from data/external/tf1_synthetic/processed/ into
semantic groups and emit data/external/tf1_synthetic/clustered/, mirroring
data/clustered/ for MORABLES. See the design spec.
"""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).parent.parent.parent
DEFAULT_IN = ROOT / "data" / "external" / "tf1_synthetic"
DEFAULT_INSPECT_ROOT = ROOT / "experiments" / "11_tf1_diagnostic" / "cluster_inspection"
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
EXACT_THRESHOLD = 0.999


def agglomerative_clusters(sim_matrix: np.ndarray, threshold: float) -> list[list[int]]:
    n = sim_matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    dist = np.clip(1.0 - sim_matrix, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="single")
    labels = fcluster(z, t=1.0 - threshold, criterion="distance")
    groups: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(idx)
    return list(groups.values())


def classify_cluster_type(members: list[int], sim_matrix: np.ndarray) -> str:
    if len(members) == 1:
        return "singleton"
    all_exact = all(
        sim_matrix[a, b] >= EXACT_THRESHOLD
        for i, a in enumerate(members)
        for b in members[i + 1:]
    )
    return "exact" if all_exact else "near"


def pick_canonical_text(members: list[int], moral_texts: list[str], counts: dict[str, int]) -> str:
    def key(i: int) -> tuple[int, int]:
        return (-counts.get(moral_texts[i], 0), i)
    best = sorted(members, key=key)[0]
    return moral_texts[best]
