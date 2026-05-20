"""ft_13 — TF1-synthetic transfer evaluated on clustered MORABLES.

See docs/superpowers/specs/2026-05-20-ft-13-tf1-transfer-clustered-design.md
for the full design. This file mirrors ft_12_storal_transfer_clustered/train.py
with surgical replacements in the data layer (TF1 source instead of STORAL,
size-as-morals-and-fables-per-moral instead of size-as-row-count, label=moral_id).
"""
from __future__ import annotations

import random
from collections import defaultdict


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
