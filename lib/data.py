# lib/data.py
"""Shared data loading for MORABLES experiments."""
import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"


def load_fables() -> list[dict]:
    """Load fables corpus. Each entry: {doc_id, title, text, alias}."""
    with open(DATA_DIR / "fables_corpus.json") as f:
        return json.load(f)


def load_morals() -> list[dict]:
    """Load morals corpus. Each entry: {doc_id, text, fable_id}."""
    with open(DATA_DIR / "morals_corpus.json") as f:
        return json.load(f)


def load_qrels_moral_to_fable() -> dict[int, int]:
    """Load moral-to-fable ground truth as {moral_idx: fable_idx}."""
    with open(DATA_DIR / "qrels_moral_to_fable.json") as f:
        qrels = json.load(f)
    return {
        int(q["query_id"].split("_")[1]): int(q["doc_id"].split("_")[1])
        for q in qrels
    }


def load_qrels_fable_to_moral() -> dict[int, int]:
    """Load fable-to-moral ground truth as {fable_idx: moral_idx}."""
    with open(DATA_DIR / "qrels_fable_to_moral.json") as f:
        qrels = json.load(f)
    return {
        int(q["query_id"].split("_")[1]): int(q["doc_id"].split("_")[1])
        for q in qrels
    }


def load_moral_to_fable_retrieval_data():
    """
    Convenience loader for the standard moral-to-fable retrieval setup.

    Returns:
        fable_texts: list[str] — 709 fable texts
        moral_texts: list[str] — 709 moral texts (ordered by moral index)
        ground_truth: dict[int, int] — {query_idx: fable_idx} (0-based, contiguous)
    """
    fables = load_fables()
    morals = load_morals()
    gt_m2f = load_qrels_moral_to_fable()

    fable_texts = [f["text"] for f in fables]
    moral_indices = sorted(gt_m2f.keys())
    moral_texts = [morals[i]["text"] for i in moral_indices]
    # Ground truth with contiguous 0-based query indices
    ground_truth = {i: gt_m2f[idx] for i, idx in enumerate(moral_indices)}

    return fable_texts, moral_texts, ground_truth
