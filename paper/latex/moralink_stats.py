#!/usr/bin/env python3
"""Compute dataset statistics reported in the MoraLink paper draft.

Run from the repository root:
    python paper/latex/moralink_stats.py

This script is intentionally read-only. It writes a JSON summary next to this
file so paper numbers can be traced back to the local benchmark artifacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "fables.json"
CLUSTERED_DIR = ROOT / "data" / "clustered"
OUT_PATH = Path(__file__).with_name("moralink_stats.json")

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def load_json(path: Path):
    return json.loads(path.read_text())


def word_count(text: str) -> int:
    return len(text.strip().split()) if text and text.strip() else 0


def token_set(text: str, *, content_only: bool) -> set[str]:
    tokens = {tok.lower() for tok in TOKEN_RE.findall(text or "")}
    if content_only:
        tokens = {tok for tok in tokens if tok not in ENGLISH_STOP_WORDS and len(tok) > 1}
    return tokens


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def stats(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 0:
        raise ValueError("Cannot summarize an empty list")
    median = (
        sorted_values[n // 2]
        if n % 2
        else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    )
    return {
        "n": n,
        "mean": sum(sorted_values) / n,
        "median": median,
        "min": sorted_values[0],
        "max": sorted_values[-1],
    }


def main() -> None:
    raw = load_json(RAW_PATH)
    fables = load_json(CLUSTERED_DIR / "fables_corpus.json")
    morals = load_json(CLUSTERED_DIR / "morals_unique_corpus.json")
    qrels = load_json(CLUSTERED_DIR / "qrels_moral_to_fable_clustered.json")
    cluster_mapping = load_json(CLUSTERED_DIR / "cluster_mapping.json")

    fable_by_id = {row["doc_id"]: row for row in fables}
    moral_by_id = {row["doc_id"]: row for row in morals}

    raw_all_iou = [
        jaccard(token_set(row["moral"], content_only=False), token_set(row["story"], content_only=False))
        for row in raw
    ]
    raw_content_iou = [
        jaccard(token_set(row["moral"], content_only=True), token_set(row["story"], content_only=True))
        for row in raw
    ]

    qrels_by_query: dict[str, int] = {}
    relevant_all_iou = []
    relevant_content_iou = []
    for row in qrels:
        qrels_by_query[row["query_id"]] = qrels_by_query.get(row["query_id"], 0) + 1
        query_text = moral_by_id[row["query_id"]]["text"]
        fable_text = fable_by_id[row["doc_id"]]["text"]
        relevant_all_iou.append(
            jaccard(token_set(query_text, content_only=False), token_set(fable_text, content_only=False))
        )
        relevant_content_iou.append(
            jaccard(token_set(query_text, content_only=True), token_set(fable_text, content_only=True))
        )

    cluster_type_counts: dict[str, int] = {}
    for row in morals:
        cluster_type = row["cluster_type"]
        cluster_type_counts[cluster_type] = cluster_type_counts.get(cluster_type, 0) + 1

    cluster_counts: dict[str, int] = {}
    for row in cluster_mapping:
        cluster_type = row["type"]
        cluster_counts[cluster_type] = cluster_counts.get(cluster_type, 0) + 1

    summary = {
        "source_files": {
            "raw_fables": str(RAW_PATH.relative_to(ROOT)),
            "clustered_fables": str((CLUSTERED_DIR / "fables_corpus.json").relative_to(ROOT)),
            "clustered_morals": str((CLUSTERED_DIR / "morals_unique_corpus.json").relative_to(ROOT)),
            "clustered_qrels": str((CLUSTERED_DIR / "qrels_moral_to_fable_clustered.json").relative_to(ROOT)),
            "cluster_mapping": str((CLUSTERED_DIR / "cluster_mapping.json").relative_to(ROOT)),
        },
        "raw_morables": {
            "pairs": len(raw),
            "unique_moral_texts_exact": len({row["moral"] for row in raw}),
            "fable_words": stats([word_count(row["story"]) for row in raw]),
            "moral_words": stats([word_count(row["moral"]) for row in raw]),
            "all_token_iou": stats(raw_all_iou),
            "content_token_iou": stats(raw_content_iou),
        },
        "clustered_moralink": {
            "unique_moral_queries": len(morals),
            "fables": len(fables),
            "moral_clusters": len(cluster_mapping),
            "relevant_query_fable_pairs": len(qrels),
            "avg_relevant_fables_per_query": len(qrels) / len(morals),
            "relevant_fables_per_query": stats(list(qrels_by_query.values())),
            "query_moral_words": stats([word_count(row["text"]) for row in morals]),
            "cluster_type_query_counts": cluster_type_counts,
            "cluster_type_cluster_counts": cluster_counts,
            "relevant_pair_all_token_iou": stats(relevant_all_iou),
            "relevant_pair_content_token_iou": stats(relevant_content_iou),
        },
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
