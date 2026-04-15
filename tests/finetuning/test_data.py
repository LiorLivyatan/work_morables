"""Tests for finetuning/lib/data.py"""
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib.data import DOC_MODES, build_doc_text, load_pairs

# ── Fixtures ──────────────────────────────────────────────────────────────────

_FABLES = [
    {"alias": "fable_0", "text": "A fox and grapes."},
    {"alias": "fable_1", "text": "A crow and cheese."},
]
_MORALS = [
    {"text": "It is easy to despise what you cannot get."},
    {"text": "Beware of flattery."},
]
_QRELS = {0: 0, 1: 1}
_SUMMARIES = {
    "fable_0": "Sour grapes.",
    "fable_1": "Flattery fools the vain.",
}


def _patched_loaders(summaries=_SUMMARIES):
    return ExitStack().__enter__  # dummy — use the context manager below


def _loader_patches(summaries=_SUMMARIES):
    return [
        patch("finetuning.lib.data.load_fables", return_value=_FABLES),
        patch("finetuning.lib.data.load_morals", return_value=_MORALS),
        patch("finetuning.lib.data.load_qrels_moral_to_fable", return_value=_QRELS),
        patch("finetuning.lib.data._load_summaries", return_value=summaries),
    ]


def _load(doc_mode, summaries=_SUMMARIES):
    with ExitStack() as stack:
        for p in _loader_patches(summaries):
            stack.enter_context(p)
        return load_pairs(doc_mode)


# ── build_doc_text ────────────────────────────────────────────────────────────


def test_build_doc_text_raw_returns_fable_unchanged():
    assert build_doc_text("Once a fox.", None, "raw") == "Once a fox."


def test_build_doc_text_raw_ignores_summary():
    assert build_doc_text("Once a fox.", "Pride comes before a fall.", "raw") == "Once a fox."


def test_build_doc_text_fable_plus_summary_contains_both():
    result = build_doc_text("Once a fox.", "Pride comes before a fall.", "fable_plus_summary")
    assert "Once a fox." in result
    assert "Pride comes before a fall." in result


def test_build_doc_text_fable_plus_summary_fable_comes_first():
    result = build_doc_text("Once a fox.", "Pride comes before a fall.", "fable_plus_summary")
    assert result.index("Once a fox.") < result.index("Pride comes before a fall.")


def test_build_doc_text_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown doc_mode"):
        build_doc_text("text", "summary", "nonexistent_mode")


def test_all_declared_doc_modes_do_not_raise():
    for mode in DOC_MODES:
        build_doc_text("fable text", "summary text", mode)


# ── load_pairs ────────────────────────────────────────────────────────────────


def test_load_pairs_raw_returns_correct_lengths():
    morals, docs, gt = _load("raw")
    assert len(morals) == 2
    assert len(docs) == 2
    assert len(gt) == 2


def test_load_pairs_raw_moral_texts():
    morals, _, _ = _load("raw")
    assert morals[0] == "It is easy to despise what you cannot get."
    assert morals[1] == "Beware of flattery."


def test_load_pairs_raw_doc_texts_are_fable_texts():
    _, docs, _ = _load("raw")
    assert docs[0] == "A fox and grapes."
    assert docs[1] == "A crow and cheese."


def test_load_pairs_raw_ground_truth():
    _, _, gt = _load("raw")
    assert gt == {0: 0, 1: 1}


def test_load_pairs_fable_plus_summary_docs_contain_summaries():
    _, docs, _ = _load("fable_plus_summary")
    assert "Sour grapes." in docs[0]
    assert "Flattery fools the vain." in docs[1]


def test_load_pairs_raw_does_not_call_load_summaries():
    with ExitStack() as stack:
        for p in _loader_patches()[:3]:
            stack.enter_context(p)
        mock_summaries = stack.enter_context(patch("finetuning.lib.data._load_summaries"))
        load_pairs("raw")
    mock_summaries.assert_not_called()


def test_load_pairs_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown doc_mode"):
        load_pairs("bad_mode")


def test_load_pairs_ground_truth_fable_indices_are_valid():
    _, docs, gt = _load("raw")
    for fable_idx in gt.values():
        assert 0 <= fable_idx < len(docs)
