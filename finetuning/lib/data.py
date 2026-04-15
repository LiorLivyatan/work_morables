"""
MORABLES pair loading and document representation for fine-tuning.

Public API
----------
load_pairs(doc_mode) -> (moral_texts, doc_texts, ground_truth)
build_doc_text(fable_text, summary, doc_mode) -> str
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.data import load_fables, load_morals, load_qrels_moral_to_fable

_SUMMARIES_PATH = (
    ROOT
    / "experiments/07_sota_summarization_oracle"
    / "results/generation_runs/full_709/golden_summaries.json"
)

DOC_MODES = ("raw", "fable_plus_summary")


# ── Document representation ───────────────────────────────────────────────────


def build_doc_text(fable_text: str, summary: str | None, doc_mode: str) -> str:
    """
    Build the document string to encode for a given fable.

    Args:
        fable_text  raw fable narrative (~130 words on average)
        summary     cot_proverb summary from Exp 07 (short proverb/sentence);
                    None when doc_mode="raw"
        doc_mode    "raw" | "fable_plus_summary"

    Returns:
        the string the model encodes as the corpus document
    """
    if doc_mode == "raw":
        return fable_text
    if doc_mode == "fable_plus_summary":
        # Matches exp07's exact format (run_all_variants.py line 123).
        # "Moral summary:" label is critical — without it the baseline model cannot
        # distinguish the appended proverb from the narrative, tanking MRR by ~0.17.
        return f"{fable_text}\n\nMoral summary: {summary}"
    raise ValueError(f"Unknown doc_mode: {doc_mode!r}. Expected one of {DOC_MODES}.")


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_summaries() -> dict[str, str]:
    """cot_proverb summaries from Exp 07, keyed by fable alias."""
    with open(_SUMMARIES_PATH) as f:
        data = json.load(f)
    return {entry["original_fable_id"]: entry["summaries"]["cot_proverb"] for entry in data}


def load_pairs(doc_mode: str) -> tuple[list[str], list[str], dict[int, int]]:
    """
    Load all 709 (moral, document) pairs.

    Returns:
        moral_texts   list[str], length 709, ordered by moral index
        doc_texts     list[str], length 709, ordered by fable index
        ground_truth  dict {moral_idx: fable_idx}, 0-based contiguous indices
    """
    if doc_mode not in DOC_MODES:
        raise ValueError(f"Unknown doc_mode: {doc_mode!r}. Expected one of {DOC_MODES}.")

    fables = load_fables()
    morals = load_morals()
    qrels = load_qrels_moral_to_fable()
    summaries = _load_summaries() if doc_mode == "fable_plus_summary" else {}

    moral_indices = sorted(qrels.keys())
    moral_texts = [morals[i]["text"] for i in moral_indices]
    doc_texts = [
        build_doc_text(fable["text"], summaries.get(fable["alias"]), doc_mode)
        for fable in fables
    ]
    ground_truth = {i: qrels[idx] for i, idx in enumerate(moral_indices)}

    return moral_texts, doc_texts, ground_truth
