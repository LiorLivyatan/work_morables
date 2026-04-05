"""lib/pipeline — Generic experiment pipeline for NLP-morables."""
import copy
from pathlib import Path
from typing import Optional

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"
_CORPUS_USER_TEMPLATE = "Fable: {text}"
_QUERY_USER_TEMPLATE = "Moral: {text}"


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base. Dicts recurse; lists and scalars replace."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _resolve_prompt(variant_cfg: dict, experiment_dir: Path) -> str:
    """Resolve a variant's system prompt: inline > file > key."""
    if "prompt" in variant_cfg:
        return variant_cfg["prompt"].strip()
    if "prompt_file" in variant_cfg:
        path = Path(experiment_dir) / variant_cfg["prompt_file"]
        return path.read_text().strip()
    if "prompt_key" in variant_cfg:
        from lib.pipeline.prompts import PROMPTS
        key = variant_cfg["prompt_key"]
        if key not in PROMPTS:
            raise ValueError(
                f"Unknown prompt_key: {key!r}. Available: {sorted(PROMPTS)}"
            )
        return PROMPTS[key]
    raise ValueError(
        f"Variant {variant_cfg.get('name', '?')!r} must have one of: "
        "prompt, prompt_file, prompt_key"
    )


def load_config(config_path: Path, experiment_dir: Optional[Path] = None) -> dict:
    """
    Load experiment config, merge over defaults, resolve all variant prompts.

    Args:
        config_path:     Path to experiment's config.yaml
        experiment_dir:  Base dir for prompt_file resolution (default: config_path.parent)

    Returns:
        Fully resolved config dict. Each variant entry has 'system_prompt' and
        'user_prompt_template' keys added.
    """
    experiment_dir = Path(experiment_dir or Path(config_path).parent)

    with open(_DEFAULT_CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}

    with open(config_path) as f:
        override = yaml.safe_load(f) or {}

    config = _deep_merge(config, override)

    for variant in config.get("corpus_variants", []):
        variant["system_prompt"] = _resolve_prompt(variant, experiment_dir)
        variant.setdefault("user_prompt_template", _CORPUS_USER_TEMPLATE)

    for variant in config.get("query_expansion_variants", []):
        variant["system_prompt"] = _resolve_prompt(variant, experiment_dir)
        variant.setdefault("user_prompt_template", _QUERY_USER_TEMPLATE)

    return config
