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


def build_clustered_outputs(
    morals: list[dict],
    qmf: list[dict],
    sim_matrix: np.ndarray,
    counts: dict[str, int],
    threshold: float,
) -> dict[str, list[dict]]:
    moral_texts = [m["text"] for m in morals]
    moral_ids = [m["doc_id"] for m in morals]

    # fable_id grouped by source moral_id (from input qrels)
    fables_for_moral: dict[str, list[str]] = {}
    for row in qmf:
        fables_for_moral.setdefault(row["query_id"], []).append(row["doc_id"])

    clusters = agglomerative_clusters(sim_matrix, threshold=threshold)

    morals_unique: list[dict] = []
    cluster_mapping: list[dict] = []
    moral_to_cluster: list[dict] = []
    qrels_mtf_clustered: list[dict] = []
    qrels_ftm_clustered: list[dict] = []

    for cluster_idx, members in enumerate(sorted(clusters, key=lambda c: (-len(c), c[0]))):
        cluster_id = f"cluster_{cluster_idx:03d}"
        cluster_type = classify_cluster_type(members, sim_matrix)
        canonical_text = pick_canonical_text(members, moral_texts, counts)
        member_moral_ids = [moral_ids[i] for i in members]
        member_texts = [moral_texts[i] for i in members]
        relevant_fable_ids = [
            fid for mid in member_moral_ids for fid in fables_for_moral.get(mid, [])
        ]

        unique_doc_id = f"moral_tf1_unique_{cluster_idx:04d}"
        morals_unique.append({
            "doc_id": unique_doc_id,
            "text": canonical_text,
            "cluster_id": cluster_id,
            "cluster_type": cluster_type,
            "relevant_fable_ids": relevant_fable_ids,
            "cluster_moral_set": member_texts,
        })
        cluster_mapping.append({
            "cluster_id": cluster_id,
            "type": cluster_type,
            "moral_set": member_texts,
            "fables": relevant_fable_ids,
            "n_morals": len(member_texts),
            "n_fables": len(relevant_fable_ids),
        })
        for mid, text in zip(member_moral_ids, member_texts):
            moral_to_cluster.append({
                "query_id": mid,
                "text": text,
                "cluster_id": cluster_id,
            })
        for fid in relevant_fable_ids:
            qrels_mtf_clustered.append({
                "query_id": unique_doc_id, "doc_id": fid, "relevance": 1,
            })
            qrels_ftm_clustered.append({
                "query_id": fid, "doc_id": unique_doc_id, "relevance": 1,
            })

    return {
        "morals_unique_corpus.json": morals_unique,
        "cluster_mapping.json": cluster_mapping,
        "moral_to_cluster.json": moral_to_cluster,
        "qrels_moral_to_fable_clustered.json": qrels_mtf_clustered,
        "qrels_fable_to_moral_clustered.json": qrels_ftm_clustered,
    }
