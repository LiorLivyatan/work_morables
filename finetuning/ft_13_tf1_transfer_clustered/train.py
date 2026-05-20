"""ft_13 — TF1-synthetic transfer evaluated on clustered MORABLES.

See docs/superpowers/specs/2026-05-20-ft-13-tf1-transfer-clustered-design.md
for the full design. This file mirrors ft_12_storal_transfer_clustered/train.py
with surgical replacements in the data layer (TF1 source instead of STORAL,
size-as-morals-and-fables-per-moral instead of size-as-row-count, label=moral_id).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def _subsample_morals(pairs: list[dict], size_cfg: dict, seed: int) -> list[dict]:
    """Sample n_morals distinct morals (or keep all), then take the first
    n_fables_per_moral fables for each.

    Sub-sampling is deterministic for a given (pairs, size_cfg, seed). Because
    the low-IoU corpus stores fables per moral in ascending-IoU order, taking
    the first K is equivalent to taking the K lowest-IoU fables for that moral.
    """
    by_moral: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_moral[p["moral_id"]].append(p)
    moral_ids = sorted(by_moral.keys())

    n_morals = size_cfg.get("n_morals")
    if n_morals is not None and n_morals < len(moral_ids):
        rng = random.Random(seed)
        moral_ids = rng.sample(moral_ids, n_morals)

    n_fables = size_cfg["n_fables_per_moral"]
    return [p for mid in moral_ids for p in by_moral[mid][:n_fables]]


def split_tf1_groups(
    pairs: list[dict], seed: int, validation_ratio: float
) -> tuple[list[dict], list[dict]]:
    """Group-aware train/val split: morals (not rows) are sampled into the
    validation set; all fables of a given moral go together. Prevents within-
    moral leakage.
    """
    by_moral: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_moral[p["moral_id"]].append(p)
    moral_ids = sorted(by_moral.keys())
    if len(moral_ids) < 2:
        raise ValueError(
            f"split_tf1_groups requires at least 2 distinct morals, got {len(moral_ids)}"
        )
    rng = random.Random(seed)
    rng.shuffle(moral_ids)
    n_val = max(1, round(len(moral_ids) * validation_ratio))
    train_rows = [p for mid in moral_ids[n_val:] for p in by_moral[mid]]
    val_rows = [p for mid in moral_ids[:n_val] for p in by_moral[mid]]
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def make_tf1_dataset(pairs: list[dict], instruction: str):
    """Build the 3-column training Dataset: anchor=instruction+moral,
    positive=fable, label=integer-per-moral (for InfoNCELoss multi-positive
    masking). In exact-cluster mode, label = moral_id integer index.
    """
    from datasets import Dataset

    moral_to_label: dict[str, int] = {}
    labels: list[int] = []
    for p in pairs:
        if p["moral_id"] not in moral_to_label:
            moral_to_label[p["moral_id"]] = len(moral_to_label)
        labels.append(moral_to_label[p["moral_id"]])

    return Dataset.from_dict({
        "anchor":   [f"{instruction}{p['moral']}" for p in pairs],
        "positive": [p["story"] for p in pairs],
        "label":    labels,
    })


def load_tf1_synthetic_exact(
    size_cfg: dict, seed: int, source_dir: Path
) -> tuple[list[dict], dict]:
    """Load TF1 exact-cluster pairs from source_dir/processed/, then sub-sample
    per the size config.

    Returns (pairs, stats):
        pairs: list of dicts with keys {moral, story, moral_id, fable_id}
        stats: {raw_total, selected_rows, selected_morals, selection_strategy,
                size_config}
    """
    morals = json.loads((source_dir / "processed" / "morals_corpus.json").read_text())
    fables = json.loads((source_dir / "processed" / "fables_corpus.json").read_text())
    qrels = json.loads((source_dir / "processed" / "qrels_moral_to_fable.json").read_text())

    moral_by_id = {m["doc_id"]: m["text"] for m in morals}
    fable_by_id = {f["doc_id"]: f["text"] for f in fables}
    pairs = [
        {
            "moral": moral_by_id[q["query_id"]],
            "story": fable_by_id[q["doc_id"]],
            "moral_id": q["query_id"],
            "fable_id": q["doc_id"],
        }
        for q in qrels
    ]
    raw_total = len(pairs)
    pairs = _subsample_morals(pairs, size_cfg, seed)
    selected_morals = len({p["moral_id"] for p in pairs})

    stats = {
        "raw_total": raw_total,
        "selected_rows": len(pairs),
        "selected_morals": selected_morals,
        "selection_strategy": source_dir.name,
        "size_config": dict(size_cfg),
    }
    return pairs, stats
