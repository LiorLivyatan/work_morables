import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_retrieval.lib.corpus import build_corpus_block


def test_format_single_fable():
    fables = [{"doc_id": "fable_0001", "text": "A fox lost its tail."}]
    block = build_corpus_block(fables)
    assert block == "[fable_0001] A fox lost its tail."


def test_format_multiple_fables():
    fables = [
        {"doc_id": "fable_0001", "text": "A fox."},
        {"doc_id": "fable_0002", "text": "A crow."},
    ]
    block = build_corpus_block(fables)
    lines = block.split("\n\n")
    assert len(lines) == 2
    assert lines[0] == "[fable_0001] A fox."
    assert lines[1] == "[fable_0002] A crow."


def test_empty_corpus():
    assert build_corpus_block([]) == ""


def test_preserves_order():
    fables = [{"doc_id": f"fable_{i:04d}", "text": f"Story {i}."} for i in range(5)]
    block = build_corpus_block(fables)
    lines = block.split("\n\n")
    for i, line in enumerate(lines):
        assert line.startswith(f"[fable_{i:04d}]")
