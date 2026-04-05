"""Tests for lib/pipeline/corpus_generator.py"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline.corpus_generator import generate_corpus_summaries


def _make_fables(n: int) -> list[dict]:
    return [
        {"doc_id": f"fable_{i:03d}", "title": f"Fable {i}", "text": f"Once upon a time {i}."}
        for i in range(n)
    ]


def _make_variants(names: list[str]) -> list[dict]:
    return [
        {"name": n, "system_prompt": f"Sys prompt for {n}.", "user_prompt_template": "Fable: {text}"}
        for n in names
    ]


def _mock_client(text: str = "Honesty is best.") -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.text = text
    usage = MagicMock()
    usage.prompt_token_count = 10
    usage.candidates_token_count = 5
    usage.thoughts_token_count = 0
    usage.total_token_count = 15
    resp.usage_metadata = usage
    client.models.generate_content.return_value = resp
    return client


def test_generate_corpus_summaries_creates_output_files(tmp_path):
    with patch("time.sleep"):
        generate_corpus_summaries(
            client=_mock_client(),
            fables=_make_fables(2),
            variants=_make_variants(["style_a"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    assert (tmp_path / "corpus_summaries.json").exists()
    assert (tmp_path / "token_usage.json").exists()


def test_generate_corpus_summaries_json_schema(tmp_path):
    with patch("time.sleep"):
        generate_corpus_summaries(
            client=_mock_client("Virtue triumphs."),
            fables=_make_fables(2),
            variants=_make_variants(["style_a", "style_b"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    with open(tmp_path / "corpus_summaries.json") as f:
        data = json.load(f)
    assert len(data) == 2
    item = data[0]
    assert "id" in item
    assert "summaries" in item
    assert "style_a" in item["summaries"]
    assert "style_b" in item["summaries"]
    assert item["summaries"]["style_a"] == "Virtue triumphs."


def test_generate_corpus_summaries_idempotent(tmp_path):
    with patch("time.sleep"):
        generate_corpus_summaries(
            client=_mock_client(),
            fables=_make_fables(3),
            variants=_make_variants(["s"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    new_client = _mock_client()
    with patch("time.sleep"):
        generate_corpus_summaries(
            client=new_client,
            fables=_make_fables(3),
            variants=_make_variants(["s"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
            force=False,
        )
    assert new_client.models.generate_content.call_count == 0


def test_generate_corpus_summaries_force_reruns(tmp_path):
    with patch("time.sleep"):
        generate_corpus_summaries(
            client=_mock_client(),
            fables=_make_fables(2),
            variants=_make_variants(["s"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    new_client = _mock_client()
    with patch("time.sleep"):
        generate_corpus_summaries(
            client=new_client,
            fables=_make_fables(2),
            variants=_make_variants(["s"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
            force=True,
        )
    assert new_client.models.generate_content.call_count == 2


def test_generate_corpus_summaries_token_usage_json(tmp_path):
    with patch("time.sleep"):
        generate_corpus_summaries(
            client=_mock_client(),
            fables=_make_fables(3),
            variants=_make_variants(["s"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    with open(tmp_path / "token_usage.json") as f:
        usage = json.load(f)
    assert usage["n_fables"] == 3
    assert usage["model"] == "gemini-flash"
    assert usage["total_input_tokens"] == 30
