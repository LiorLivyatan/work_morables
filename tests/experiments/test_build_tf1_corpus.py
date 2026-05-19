# tests/experiments/test_build_tf1_corpus.py
import sys
import importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PATH = Path(__file__).parent.parent.parent / "experiments" / "11_tf1_diagnostic" / "build_tf1_corpus.py"
build = _load("build_tf1_corpus", _PATH)
group_by_moral = build.group_by_moral
first_seen_order = build.first_seen_order
assign_moral_ids = build.assign_moral_ids
sample_n_per_moral = build.sample_n_per_moral


def _row(idx, moral, fable="...", chunk=0, prompt_hash="h"):
    return {"idx": idx, "chunk": chunk, "prompt_hash": prompt_hash, "moral": moral, "fable": fable}


def test_group_by_moral_lowercases_and_strips():
    rows = [_row(0, " Greed Leads To Downfall "), _row(1, "greed leads to downfall")]
    groups = group_by_moral(rows)
    assert list(groups.keys()) == ["greed leads to downfall"]
    assert len(groups["greed leads to downfall"]) == 2


def test_first_seen_order_preserves_insertion():
    rows = [_row(0, "B"), _row(1, "A"), _row(2, "B"), _row(3, "C")]
    assert first_seen_order(rows) == ["b", "a", "c"]


def test_assign_moral_ids_zero_padded():
    ids = assign_moral_ids(["a", "b", "c"])
    assert ids == {"a": "moral_tf1_000", "b": "moral_tf1_001", "c": "moral_tf1_002"}


def test_sample_n_per_moral_is_deterministic_with_same_seed():
    grouped = {"m": [_row(i, "m") for i in range(20)]}
    out_a = sample_n_per_moral(grouped, n=5, seed=42)
    out_b = sample_n_per_moral(grouped, n=5, seed=42)
    assert [r["idx"] for r in out_a["m"]] == [r["idx"] for r in out_b["m"]]


def test_sample_n_per_moral_differs_with_different_seed():
    grouped = {"m": [_row(i, "m") for i in range(100)]}
    out_a = sample_n_per_moral(grouped, n=10, seed=1)
    out_b = sample_n_per_moral(grouped, n=10, seed=2)
    assert [r["idx"] for r in out_a["m"]] != [r["idx"] for r in out_b["m"]]


def test_sample_n_per_moral_errors_when_not_enough_rows():
    grouped = {"m": [_row(i, "m") for i in range(3)]}
    with pytest.raises(ValueError, match="only 3"):
        sample_n_per_moral(grouped, n=10, seed=42)
