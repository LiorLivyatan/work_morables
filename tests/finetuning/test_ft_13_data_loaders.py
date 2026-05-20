import sys
import json
import importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PATH = (
    Path(__file__).parent.parent.parent
    / "finetuning" / "ft_13_tf1_transfer_clustered" / "train.py"
)
train = _load("ft_13_train", _PATH)


def _pair(idx, moral_id="moral_tf1_000", fable_id_suffix=0):
    return {
        "moral": f"moral text {moral_id}",
        "story": f"fable text {fable_id_suffix}",
        "moral_id": moral_id,
        "fable_id": f"fable_tf1_{fable_id_suffix:05d}",
    }


def test_subsample_morals_keeps_all_when_unrestricted():
    pairs = [_pair(i, "moral_tf1_000", i) for i in range(10)]
    out = train._subsample_morals(pairs, {"n_morals": None, "n_fables_per_moral": 10}, seed=42)
    assert len(out) == 10


def test_subsample_morals_caps_fables_per_moral():
    pairs = [_pair(i, "moral_tf1_000", i) for i in range(20)]
    out = train._subsample_morals(pairs, {"n_morals": None, "n_fables_per_moral": 5}, seed=42)
    assert len(out) == 5
    # First 5 by file order (insertion preserved)
    assert [p["fable_id"] for p in out] == [f"fable_tf1_{i:05d}" for i in range(5)]


def test_subsample_morals_samples_morals_then_takes_all_fables():
    pairs = []
    for moral_idx in range(4):
        for fable_idx in range(10):
            pairs.append(_pair(moral_idx * 10 + fable_idx,
                               moral_id=f"moral_tf1_{moral_idx:03d}",
                               fable_id_suffix=moral_idx * 10 + fable_idx))
    out = train._subsample_morals(pairs, {"n_morals": 2, "n_fables_per_moral": 10}, seed=42)
    morals = {p["moral_id"] for p in out}
    assert len(morals) == 2
    assert len(out) == 20  # 2 morals x 10 fables


def test_subsample_morals_is_seed_stable():
    pairs = []
    for moral_idx in range(10):
        for fable_idx in range(10):
            pairs.append(_pair(moral_idx * 10 + fable_idx,
                               moral_id=f"moral_tf1_{moral_idx:03d}",
                               fable_id_suffix=moral_idx * 10 + fable_idx))
    a = train._subsample_morals(pairs, {"n_morals": 3, "n_fables_per_moral": 5}, seed=42)
    b = train._subsample_morals(pairs, {"n_morals": 3, "n_fables_per_moral": 5}, seed=42)
    assert [(p["moral_id"], p["fable_id"]) for p in a] == [(p["moral_id"], p["fable_id"]) for p in b]
