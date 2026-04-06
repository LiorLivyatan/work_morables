"""Tests for lib/pipeline/query_expander.py"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline.query_expander import generate_query_expansions


def _mock_client(text: str = "Honesty is best.") -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.text = text
    usage = MagicMock()
    usage.prompt_token_count = 8
    usage.candidates_token_count = 4
    usage.thoughts_token_count = 0
    usage.total_token_count = 12
    resp.usage_metadata = usage
    client.models.generate_content.return_value = resp
    return client


def _make_morals(indices: list[int]) -> list[dict]:
    return [{"doc_id": f"moral_{i:03d}", "text": f"Moral text {i}."} for i in range(max(indices) + 1)]


def _make_variants(names: list[str]) -> list[dict]:
    return [
        {"name": n, "system_prompt": f"Rephrase: {n}", "user_prompt_template": "Moral: {text}"}
        for n in names
    ]


def test_generate_query_expansions_creates_output(tmp_path):
    moral_entries = [(0, 0), (1, 1)]
    morals = _make_morals([0, 1])
    with patch("time.sleep"):
        generate_query_expansions(
            client=_mock_client(),
            moral_entries=moral_entries,
            morals=morals,
            variants=_make_variants(["rephrase"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    assert (tmp_path / "query_expansions.json").exists()
    assert (tmp_path / "query_expansion_token_usage.json").exists()


def test_generate_query_expansions_json_schema(tmp_path):
    moral_entries = [(2, 0), (5, 1)]
    morals = _make_morals([2, 5])
    with patch("time.sleep"):
        generate_query_expansions(
            client=_mock_client("Virtue wins."),
            moral_entries=moral_entries,
            morals=morals,
            variants=_make_variants(["rephrase", "abstract"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    with open(tmp_path / "query_expansions.json") as f:
        data = json.load(f)
    assert len(data) == 2
    item = data[0]
    assert item["moral_idx"] == 2
    assert item["fable_idx"] == 0
    assert "rephrase" in item["paraphrases"]
    assert "abstract" in item["paraphrases"]
    assert item["paraphrases"]["rephrase"] == "Virtue wins."


def test_generate_query_expansions_idempotent(tmp_path):
    moral_entries = [(0, 0)]
    morals = _make_morals([0])
    with patch("time.sleep"):
        generate_query_expansions(
            client=_mock_client(),
            moral_entries=moral_entries,
            morals=morals,
            variants=_make_variants(["r"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    new_client = _mock_client()
    with patch("time.sleep"):
        generate_query_expansions(
            client=new_client,
            moral_entries=moral_entries,
            morals=morals,
            variants=_make_variants(["r"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
            force=False,
        )
    assert new_client.models.generate_content.call_count == 0


def test_generate_query_expansions_token_usage(tmp_path):
    moral_entries = [(0, 0), (1, 1), (2, 2)]
    morals = _make_morals([0, 1, 2])
    with patch("time.sleep"):
        generate_query_expansions(
            client=_mock_client(),
            moral_entries=moral_entries,
            morals=morals,
            variants=_make_variants(["r"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    with open(tmp_path / "query_expansion_token_usage.json") as f:
        usage = json.load(f)
    assert usage["n_morals"] == 3
    assert usage["total_input_tokens"] == 24
