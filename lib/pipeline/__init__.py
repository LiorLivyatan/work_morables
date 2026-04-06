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


def run_experiment(
    config_path: Path,
    run_dir: Optional[Path] = None,
    force: bool = False,
) -> None:
    """
    Run the full pipeline for an experiment.

    Args:
        config_path: Path to experiment's config.yaml
        run_dir:     Existing run dir to continue (default: create new timestamped dir)
        force:       Re-run steps even if their output already exists
    """
    import sys as _sys
    _ROOT = Path(__file__).parent.parent.parent
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))

    from lib.pipeline import (
        corpus_generator,
        query_expander,
        retrieval_eval,
        llm_client as _lc,
    )
    from lib.pipeline.run_utils import load_env, make_run_dir, write_manifest
    from lib.data import load_fables, load_morals, load_qrels_moral_to_fable

    config_path = Path(config_path)
    experiment_dir = config_path.parent

    load_env(_ROOT)
    config = load_config(config_path, experiment_dir)

    n_fables = config.get("n_fables")
    tag = f"sample{n_fables}" if n_fables else "full"

    if run_dir is None:
        base = experiment_dir / "results" / "pipeline_runs"
        run_dir = make_run_dir(base, tag)
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Pipeline: {experiment_dir.name}")
    print(f"  Run dir:  {run_dir}")
    print(f"  n_fables: {n_fables or 'all'}")
    print(f"{'=' * 60}")

    # Load data subset
    fables = load_fables()
    morals = load_morals()
    gt_m2f = load_qrels_moral_to_fable()

    if n_fables:
        fables = fables[:n_fables]
    else:
        n_fables = len(fables)

    target_fable_indices = set(range(n_fables))
    moral_entries = sorted(
        [(m_idx, f_idx) for m_idx, f_idx in gt_m2f.items()
         if f_idx in target_fable_indices],
        key=lambda x: x[0],
    )

    fable_texts = [f["text"] for f in fables]
    moral_texts = [morals[m_idx]["text"] for m_idx, _ in moral_entries]
    moral_indices = [m_idx for m_idx, _ in moral_entries]
    ground_truth = {i: f_idx for i, (_, f_idx) in enumerate(moral_entries)}

    steps = config.get("steps", {})

    # Step 1: generate corpus summaries
    if steps.get("generate_corpus_summaries", True) and config.get("corpus_variants"):
        print("\n── Step 1: Generate corpus summaries ──────────────────")
        client = _lc.create_client()
        corpus_generator.generate_corpus_summaries(
            client=client,
            fables=fables,
            variants=config["corpus_variants"],
            model_id=config["corpus_generation_model"],
            run_dir=run_dir,
            delay=config.get("api_delay_seconds", 0.5),
            force=force,
        )
        write_manifest(run_dir, "generate_corpus_summaries", config)

    # Step 2: generate query expansions
    if steps.get("generate_query_expansions", True) and config.get("query_expansion_variants"):
        print("\n── Step 2: Generate query expansions ──────────────────")
        client = _lc.create_client()
        query_expander.generate_query_expansions(
            client=client,
            moral_entries=moral_entries,
            morals=morals,
            variants=config["query_expansion_variants"],
            model_id=config["query_expansion_model"],
            run_dir=run_dir,
            delay=config.get("api_delay_seconds", 0.5),
            force=force,
        )
        write_manifest(run_dir, "generate_query_expansions", config)

    # Step 3: retrieval eval
    if steps.get("run_retrieval_eval", True) and config.get("retrieval_configs"):
        print("\n── Step 3: Retrieval evaluation ───────────────────────")
        retrieval_eval.run_retrieval_eval(
            run_dir=run_dir,
            config=config,
            fable_texts=fable_texts,
            moral_texts=moral_texts,
            ground_truth=ground_truth,
            moral_indices=moral_indices,
            force=force,
        )
        write_manifest(run_dir, "run_retrieval_eval", config)

    print(f"\n  Done. Results in {run_dir}")
