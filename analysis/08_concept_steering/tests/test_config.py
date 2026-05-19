import sys
from pathlib import Path
import yaml
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.config import load_config, ConfigError


def _minimal_valid_dict():
    return {
        "model": {"hf_id": "x", "pooling": "auto", "device": "cuda",
                   "dtype": "bfloat16", "batch_size": 8},
        "data": {"morals_path": "a", "fables_path": "b", "metadata_path": "c"},
        "discovery": {"failure_definition": "rank_gt_1", "metadata_fields": ["x"],
                       "min_tagged_fables": 15, "fdr_alpha": 0.05},
        "concepts": {"targets": [], "placebo": []},
        "vectors": {"layers": [4],
                     "methods": {"primary": "caa_matched", "sanity_byproducts": []},
                     "matching": {"fields": [], "cross_field_matching": False,
                                  "length_tolerance": 0.2,
                                  "min_positive_examples": 15,
                                  "min_matched_pairs": 15},
                     "quality_log": []},
        "intervention": {"alphas": [0.0], "hook_position": "residual_stream",
                          "renormalize": True},
        "null_controls": {"run_mode": "candidate_only",
                           "random_direction": {"n_seeds": 1, "norm_match": "caa_matched"},
                           "shuffled_tag_caa": {"n_permutations": 1}},
        "eval": {"metrics": ["mrr_at_10"], "group_by": [],
                  "primary_statistic": {"name": "specificity_gap",
                                         "test": "paired_bootstrap_ci_over_morals",
                                         "n_bootstrap": 100, "alpha": 0.05},
                  "diagnostics": {"pooled_cosine_pre_post": True,
                                   "rank_change_listing": True,
                                   "null_envelope": True}},
        "output": {"results_dir": "x", "cache_dir": "y",
                    "save_intermediate_embeddings": False},
    }


def test_load_config_returns_dict_for_valid_file(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(_minimal_valid_dict()))
    cfg = load_config(cfg_path)
    assert cfg["model"]["hf_id"] == "x"


def test_load_config_raises_on_missing_required_key(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump({"model": {}}))
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_load_config_raises_when_alpha_zero_missing(tmp_path):
    base = _minimal_valid_dict()
    base["intervention"]["alphas"] = [-1.0, 1.0]
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.dump(base))
    with pytest.raises(ConfigError, match="0.0"):
        load_config(cfg_path)


def test_load_config_raises_on_invalid_run_mode(tmp_path):
    base = _minimal_valid_dict()
    base["null_controls"]["run_mode"] = "bogus"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.dump(base))
    with pytest.raises(ConfigError, match="run_mode"):
        load_config(cfg_path)


def test_load_config_raises_on_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_real_repo_config():
    """Sanity: the actual repo config.yaml loads without errors."""
    repo = Path(__file__).resolve().parents[3]
    cfg = load_config(repo / "analysis/08_concept_steering/config.yaml")
    assert cfg["model"]["hf_id"] == "Linq-AI-Research/Linq-Embed-Mistral"
