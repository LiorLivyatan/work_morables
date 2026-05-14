import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_retrieval.lib.prompt import render_prompt


VARIANT = {
    "label": "minimal",
    "system": "You are a retrieval system.",
    "user_template": "Moral: {moral}\n\nCorpus:\n{corpus}\n\nReturn the {top_k} most relevant fable IDs.",
}


def test_renders_system_unchanged():
    system, _ = render_prompt("Be kind.", "corpus text", VARIANT, top_k=10)
    assert system == "You are a retrieval system."


def test_substitutes_moral():
    _, user = render_prompt("Be kind.", "corpus text", VARIANT, top_k=10)
    assert "Be kind." in user


def test_substitutes_corpus():
    _, user = render_prompt("Be kind.", "[fable_0001] A story.", VARIANT, top_k=10)
    assert "[fable_0001] A story." in user


def test_substitutes_top_k():
    _, user = render_prompt("Be kind.", "corpus", VARIANT, top_k=5)
    assert "5" in user


def test_missing_placeholder_raises():
    bad_variant = {
        "label": "bad",
        "system": "sys",
        "user_template": "Moral: {moral}",  # missing {corpus} and {top_k}
    }
    import pytest
    with pytest.raises(KeyError):
        render_prompt("Be kind.", "corpus", bad_variant, top_k=10)
