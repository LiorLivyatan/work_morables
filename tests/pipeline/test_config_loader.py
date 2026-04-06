"""Tests for lib/pipeline load_config()"""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline import load_config


def _write_config(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f)
    return p


def test_load_config_inline_prompt(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "n_fables": 5,
        "corpus_variants": [{"name": "my_style", "prompt": "State the moral briefly."}],
    })
    cfg = load_config(cfg_path)
    assert cfg["corpus_variants"][0]["system_prompt"] == "State the moral briefly."


def test_load_config_prompt_file(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "custom.txt").write_text("Custom prompt from file.\n")
    cfg_path = _write_config(tmp_path, {
        "corpus_variants": [{"name": "custom", "prompt_file": "prompts/custom.txt"}],
    })
    cfg = load_config(cfg_path)
    assert cfg["corpus_variants"][0]["system_prompt"] == "Custom prompt from file."


def test_load_config_prompt_key(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "corpus_variants": [{"name": "gt", "prompt_key": "ground_truth_style"}],
    })
    cfg = load_config(cfg_path)
    assert "aphorism" in cfg["corpus_variants"][0]["system_prompt"]


def test_load_config_missing_prompt_raises(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "corpus_variants": [{"name": "orphan"}],
    })
    with pytest.raises(ValueError, match="must have one of"):
        load_config(cfg_path)


def test_load_config_unknown_prompt_key_raises(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "corpus_variants": [{"name": "x", "prompt_key": "nonexistent_key"}],
    })
    with pytest.raises(ValueError, match="Unknown prompt_key"):
        load_config(cfg_path)


def test_load_config_default_user_prompt_template(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "corpus_variants": [{"name": "s", "prompt": "Summarize."}],
        "query_expansion_variants": [{"name": "r", "prompt": "Rephrase."}],
    })
    cfg = load_config(cfg_path)
    assert cfg["corpus_variants"][0]["user_prompt_template"] == "Fable: {text}"
    assert cfg["query_expansion_variants"][0]["user_prompt_template"] == "Moral: {text}"


def test_load_config_custom_user_prompt_template(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "corpus_variants": [{
            "name": "s",
            "prompt": "Summarize.",
            "user_prompt_template": "Story: {text}",
        }],
    })
    cfg = load_config(cfg_path)
    assert cfg["corpus_variants"][0]["user_prompt_template"] == "Story: {text}"


def test_load_config_merges_scalars_over_defaults(tmp_path):
    cfg_path = _write_config(tmp_path, {"n_fables": 42})
    cfg = load_config(cfg_path)
    assert cfg["n_fables"] == 42
    assert cfg["embed_model"] == "Linq-AI-Research/Linq-Embed-Mistral"


def test_load_config_steps_deep_merge(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "steps": {"generate_corpus_summaries": False},
    })
    cfg = load_config(cfg_path)
    assert cfg["steps"]["generate_corpus_summaries"] is False
    assert cfg["steps"]["generate_query_expansions"] is True
    assert cfg["steps"]["run_retrieval_eval"] is True


def test_load_config_list_replaces_not_extends(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "corpus_variants": [{"name": "only_one", "prompt": "Single variant."}],
    })
    cfg = load_config(cfg_path)
    assert len(cfg["corpus_variants"]) == 1
    assert cfg["corpus_variants"][0]["name"] == "only_one"
