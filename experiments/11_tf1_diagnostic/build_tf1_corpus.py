"""
Build a MORABLES-shaped derivative of TF1-EN-3M under
data/external/tf1_synthetic/processed/ from the diagnostic's samples.jsonl
cache. See docs/superpowers/specs/2026-05-06-tf1-synthetic-corpus-design.md.
"""
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DEFAULT_RUNS_DIR = ROOT / "experiments" / "11_tf1_diagnostic" / "results" / "runs"
DEFAULT_OUT = ROOT / "data" / "external" / "tf1_synthetic"


def group_by_moral(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        key = r["moral"].lower().strip()
        out.setdefault(key, []).append(r)
    return out


def first_seen_order(rows: list[dict]) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for r in rows:
        key = r["moral"].lower().strip()
        if key not in seen:
            seen.add(key)
            order.append(key)
    return order


def assign_moral_ids(unique_morals: list[str]) -> dict[str, str]:
    return {m: f"moral_tf1_{i:03d}" for i, m in enumerate(unique_morals)}


def sample_n_per_moral(grouped: dict[str, list[dict]], n: int, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {}
    for moral, rows in grouped.items():
        if len(rows) < n:
            raise ValueError(
                f"Moral has only {len(rows)} cached rows, need {n}: {moral!r}"
            )
        sorted_rows = sorted(rows, key=lambda r: r["idx"])
        out[moral] = rng.sample(sorted_rows, n)
    return out


def build_morals_corpus(unique_morals: list[str], moral_ids: dict[str, str]) -> list[dict]:
    return [{"doc_id": moral_ids[m], "text": m} for m in unique_morals]


def build_fables_corpus(
    sampled: dict[str, list[dict]],
    unique_morals: list[str],
    moral_ids: dict[str, str],
    n: int,
) -> list[dict]:
    out: list[dict] = []
    for moral_idx, moral_text in enumerate(unique_morals):
        rows = sampled[moral_text]
        assert len(rows) == n, (
            f"Expected {n} rows for moral {moral_text!r}, got {len(rows)}"
        )
        for i, row in enumerate(rows):
            fable_id = f"fable_tf1_{moral_idx * n + i:05d}"
            out.append({
                "doc_id": fable_id,
                "text": row["fable"],
                "moral_id": moral_ids[moral_text],
                "source_idx": row["idx"],
                "source_chunk": row["chunk"],
                "prompt_hash": row["prompt_hash"],
            })
    return out


def build_qrels_moral_to_fable(fables_corpus: list[dict]) -> list[dict]:
    return [
        {"query_id": f["moral_id"], "doc_id": f["doc_id"], "relevance": 1}
        for f in fables_corpus
    ]


def build_qrels_fable_to_moral(fables_corpus: list[dict]) -> list[dict]:
    return [
        {"query_id": f["doc_id"], "doc_id": f["moral_id"], "relevance": 1}
        for f in fables_corpus
    ]
