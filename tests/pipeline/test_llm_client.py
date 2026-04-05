"""Tests for lib/pipeline/llm_client.py"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline.llm_client import call


def _mock_response(text: str, prompt_tokens=10, candidate_tokens=5):
    resp = MagicMock()
    resp.text = text
    usage = MagicMock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = candidate_tokens
    usage.thoughts_token_count = 0
    usage.total_token_count = prompt_tokens + candidate_tokens
    resp.usage_metadata = usage
    return resp


def test_call_returns_expected_keys():
    client = MagicMock()
    client.models.generate_content.return_value = _mock_response("Honesty is the best policy.")
    result = call(client, "gemini-flash", "sys prompt", "user prompt")
    assert set(result.keys()) == {"text", "input_tokens", "output_tokens", "thinking_tokens", "total_tokens"}


def test_call_returns_stripped_text():
    client = MagicMock()
    client.models.generate_content.return_value = _mock_response("  Honesty wins.  ")
    result = call(client, "gemini-flash", "sys", "user")
    assert result["text"] == "Honesty wins."


def test_call_returns_token_counts():
    client = MagicMock()
    client.models.generate_content.return_value = _mock_response("ok", prompt_tokens=20, candidate_tokens=8)
    result = call(client, "gemini-flash", "sys", "user")
    assert result["input_tokens"] == 20
    assert result["output_tokens"] == 8
    assert result["total_tokens"] == 28


def test_call_retries_on_rate_limit():
    client = MagicMock()
    rate_error = Exception("Error 429: rate limit exceeded")
    client.models.generate_content.side_effect = [
        rate_error,
        rate_error,
        _mock_response("Success after retry"),
    ]
    with patch("time.sleep"):
        result = call(client, "gemini-flash", "sys", "user", max_retries=5)
    assert result["text"] == "Success after retry"
    assert client.models.generate_content.call_count == 3


def test_call_returns_error_after_max_retries():
    client = MagicMock()
    client.models.generate_content.side_effect = Exception("429: quota exceeded")
    with patch("time.sleep"):
        result = call(client, "gemini-flash", "sys", "user", max_retries=3)
    assert result["text"].startswith("[ERROR")
    assert result["input_tokens"] == 0


def test_call_passes_system_instruction():
    client = MagicMock()
    client.models.generate_content.return_value = _mock_response("ok")
    call(client, "gemini-flash", "Be concise.", "Tell me a moral.")
    _, kwargs = client.models.generate_content.call_args
    assert kwargs["config"]["system_instruction"] == "Be concise."
