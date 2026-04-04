"""Tests for generate_summaries.py pure functions.

Run from repo root:
    pytest experiments/09_gemma4_summarization/tests/test_generate.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import generate_summaries as gs


# ── postprocess_summary ──────────────────────────────────────────────────────

def test_postprocess_direct_moral_returns_first_line():
    raw = "Slow and steady wins the race.\nSome extra text."
    assert gs.postprocess_summary(raw, "direct_moral") == "Slow and steady wins the race."


def test_postprocess_conceptual_abstract_returns_last_line():
    raw = "1. The hare was overconfident.\n2. The tortoise persisted.\nPersistence overcomes arrogance."
    assert gs.postprocess_summary(raw, "conceptual_abstract") == "Persistence overcomes arrogance."


def test_postprocess_strips_think_blocks():
    raw = "<think>some reasoning</think>Vanity blinds us to real danger."
    assert gs.postprocess_summary(raw, "direct_moral") == "Vanity blinds us to real danger."


def test_postprocess_handles_empty_lines():
    raw = "\n\nThe strong should not mock the weak.\n\n"
    assert gs.postprocess_summary(raw, "narrative_distillation") == "The strong should not mock the weak."


def test_postprocess_multiline_think_block():
    raw = "<think>\nstep 1\nstep 2\n</think>\nKindness is repaid in kind."
    assert gs.postprocess_summary(raw, "direct_moral") == "Kindness is repaid in kind."


# ── build_corpus_item ────────────────────────────────────────────────────────

def test_build_corpus_item_schema():
    fable = {"doc_id": "fable_001", "alias": "aesop_001", "text": "A fox and a crow..."}
    summaries = {"direct_moral": "Flattery is dangerous.", "narrative_distillation": "Watch for flatterers."}
    item = gs.build_corpus_item(1, fable, "Do not trust flatterers.", summaries, "mlx-community/gemma-4-e2b-it-4bit")

    assert item["id"] == "item_001"
    assert item["original_fable_id"] == "aesop_001"
    assert item["fable_text"] == "A fox and a crow..."
    assert item["ground_truth_moral"] == "Do not trust flatterers."
    assert item["summaries"] == summaries
    assert item["metadata"]["model"] == "mlx-community/gemma-4-e2b-it-4bit"
    assert item["metadata"]["source"] == "aesop"
    assert item["metadata"]["word_count_fable"] == 5


def test_build_corpus_item_id_zero_padded():
    fable = {"doc_id": "fable_007", "text": "Short fable."}
    item = gs.build_corpus_item(7, fable, "moral", {}, "model-id")
    assert item["id"] == "item_007"


def test_build_corpus_item_fallback_when_no_alias():
    fable = {"doc_id": "fable_042", "text": "Another fable."}
    item = gs.build_corpus_item(42, fable, "moral", {}, "model-id")
    assert item["original_fable_id"] == "fable_042"
    assert item["metadata"]["source"] == "fable"
