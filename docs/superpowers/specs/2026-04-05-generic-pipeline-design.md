# Generic Experiment Pipeline Design

**Date:** 2026-04-05  
**Status:** Approved  
**Scope:** Refactor exp08 pipeline into a reusable, configurable framework in `lib/pipeline/`

---

## Problem

`experiments/08_symmetric_moral_matching/` has three scripts with hardcoded values for model IDs, fable counts, prompt variants, and retrieval configs. Every new experiment (exp09, exp10, …) would copy-paste and patch this code. The goal is a shared pipeline that any experiment can drive via a thin config file.

---

## Architecture

### Shared location: `lib/pipeline/`

All reusable pipeline code lives in `lib/pipeline/`, alongside the existing `lib/data.py`, `lib/retrieval_utils.py`, and `lib/embedding_cache.py`. Experiments reference it by config; they do not duplicate it.

```
lib/
  pipeline/
    __init__.py             # run_experiment() orchestrator
    llm_client.py           # Gemini API: client creation, call with retry/backoff
    corpus_generator.py     # step 1: fable × variant → corpus_summaries.json
    query_expander.py       # step 2: moral × variant → query_expansions.json
    retrieval_eval.py       # step 3: embed + score + fuse + metrics → retrieval_results.json
    run_utils.py            # run dir creation, .env loading, manifest I/O, token tracking
    prompts.py              # ALL system prompt strings, referenced by key
    default_config.yaml     # base defaults; experiments override only what differs
```

---

## Module Responsibilities

### `llm_client.py`
- Creates a Gemini `genai.Client` from `GEMINI_API_KEY`
- Single `call(client, model_id, system_prompt, user_prompt, max_retries)` function
- Handles rate-limit (429) and general errors with exponential backoff
- Returns `{text, input_tokens, output_tokens, thinking_tokens, total_tokens}`
- Consolidates the duplicated retry logic from both generation scripts

### `prompts.py`
- Defines shared/default system prompt strings as module-level constants, keyed by name
- Acts as the shared library of reusable prompts — used when a variant references `prompt_key`
- Experiment-local prompts do NOT need to go here; they are defined directly in config

### `corpus_generator.py`
- Accepts: fable list, list of resolved variant configs `{name, system_prompt, user_prompt_template}`, model ID, output path, delay
- Iterates fable × variant, calls `llm_client.call()`
- Writes `corpus_summaries.json` (same schema as exp08 today)
- Writes `token_usage.json`
- Idempotent: skips if `corpus_summaries.json` already exists in run dir (unless `--force`)

### `query_expander.py`
- Accepts: moral list, list of resolved variant configs `{name, system_prompt, user_prompt_template}`, model ID, output path, delay
- Iterates moral × variant, calls `llm_client.call()`
- Writes `query_expansions.json`
- Writes `query_expansion_token_usage.json`
- Idempotent: skips if output already exists

### `retrieval_eval.py`
- Accepts: run dir, retrieval config list, ground truth, n_fables, embed model, query instruction, optional baseline config, cache dir
- Uses `lib/embedding_cache.py` (`encode_with_cache()`) for all embedding calls — no re-encoding across runs
- Uses `lib/retrieval_utils.compute_metrics()` exclusively — the richer metric set (MRR, R@1, MAP, NDCG@k, P@k)
- Validates at load time that every `corpus_variant` in retrieval configs exists as a key in `corpus_summaries.json` — fails fast with clear error if not
- Supports max-score fusion and RRF fusion (carried over from exp08)
- Writes `retrieval_results.json`

### `run_utils.py`
- `load_env(root_dir)` — loads `.env` into `os.environ`
- `make_run_dir(base_dir, tag)` — creates `<timestamp>_<tag>/` and returns path
- `write_manifest(run_dir, step, config)` — writes/updates `run_manifest.json` after each step completes
- `read_manifest(run_dir)` — reads manifest for step coupling
- `find_latest_run_dir(base_dir)` — fallback for CLI convenience

### `__init__.py` — `run_experiment()`
1. Loads default config, deep-merges experiment config on top
2. Resolves each variant's prompt using this priority order:
   - `prompt:` inline text in config (highest priority)
   - `prompt_file:` path relative to experiment dir
   - `prompt_key:` name looked up in `lib/pipeline/prompts.py`
3. Creates or validates run dir
4. Runs enabled steps in order; writes manifest entry after each step
5. Passes run dir explicitly between steps (no directory-listing heuristics)

---

## Config Schema

### Prompt resolution for variants

Every variant in `corpus_variants` and `query_expansion_variants` supports three ways to specify its prompt. Use whichever fits:

```yaml
corpus_variants:
  # 1. Inline — write the prompt directly in config (most convenient)
  - name: ground_truth_style
    prompt: "State the fable's moral as a concise aphorism of 5 to 15 words. Abstract and universal. Output ONLY the moral."

  # 2. Multi-line inline — same as above, just cleaner for longer prompts
  - name: declarative_universal
    prompt: |
      You are an expert in moral philosophy.
      Distill the fable's lesson into one declarative sentence of 5 to 15 words.
      Universal and timeless — no character names. Output ONLY the sentence.

  # 3. File reference — load from a .txt file relative to the experiment dir
  - name: my_custom_style
    prompt_file: prompts/my_custom_style.txt

  # 4. Shared key — reuse a prompt already defined in lib/pipeline/prompts.py
  - name: moral_abstract
    prompt_key: moral_abstract
```

**Priority:** `prompt` > `prompt_file` > `prompt_key`. Error if none is provided.

Each variant also supports an optional `user_prompt_template` to control how the input text is formatted in the user turn:

```yaml
query_expansion_variants:
  - name: moral_rephrase
    prompt: "Rephrase this moral using different words. Keep it under 15 words."
    user_prompt_template: "Moral to rephrase: {text}"   # default: "Moral: {text}"
```

Default templates: `"Fable: {text}"` for corpus variants, `"Moral: {text}"` for query expansion variants.

---

### `lib/pipeline/default_config.yaml` — base defaults

```yaml
# Models
corpus_generation_model: gemini-3-flash-preview
query_expansion_model: gemini-3-flash-preview
embed_model: Linq-AI-Research/Linq-Embed-Mistral
embed_query_instruction: "Given a text, retrieve the most relevant passage that answers the query"

# Scale
n_fables: null   # null = all 709

# Steps (all on by default)
steps:
  generate_corpus_summaries: true
  generate_query_expansions: true
  run_retrieval_eval: true

# Corpus summary variants — defined per experiment, no global default
corpus_variants: []

# Query expansion variants — defined per experiment, no global default
query_expansion_variants: []

# Retrieval configs — defined per experiment, no global default
retrieval_configs: []

# Baseline (optional)
baseline: null

# Embedding cache
cache_dir: null   # null = <run_dir>/embedding_cache/

# Rate limiting
api_delay_seconds: 0.5
```

### Experiment config — `experiments/08_symmetric_moral_matching/config.yaml`

```yaml
n_fables: 10

corpus_variants:
  - name: ground_truth_style
    prompt: |
      You are an expert in fables. State the moral as a concise aphorism of 5 to 15 words.
      Use no character names. Be abstract and universal.
      Examples: "Appearances are deceptive." / "Vices are their own punishment."
      Output ONLY the moral.
  - name: declarative_universal
    prompt: |
      You are an expert in moral philosophy. Distill the fable's lesson into one declarative
      sentence of 5 to 15 words. Universal and timeless — no character names.
      Examples: "Those who envy others invite their own misfortune."
      Output ONLY the sentence.

query_expansion_variants:
  - name: moral_rephrase
    prompt: "Rephrase this moral using different words while preserving the exact meaning. At most 15 words. Abstract and universal only."
  - name: moral_elaborate
    prompt: "Broaden this moral slightly to express the same principle in a wider context. At most 20 words. Abstract and universal."
  - name: moral_abstract
    prompt: "Strip this moral to its most concise and abstract form. At most 10 words."

retrieval_configs:
  - name: A
    corpus_variant: ground_truth_style
    use_expansion: false
  - name: B
    corpus_variant: declarative_universal
    use_expansion: false
  - name: A_expand
    corpus_variant: ground_truth_style
    use_expansion: true
    expansion_variants: [moral_rephrase, moral_elaborate, moral_abstract]
  - name: B_expand
    corpus_variant: declarative_universal
    use_expansion: true
    expansion_variants: [moral_rephrase, moral_elaborate, moral_abstract]
  - name: RRF_all
    fusion: rrf
    source_configs: [A, B, A_expand, B_expand]
    k: 60

baseline:
  path: experiments/07_sota_summarization_oracle/results/generation_runs/full_709/golden_summaries.json
  variant: conceptual_abstract
```

### Example exp09 config — fewer variants, different model, reuse generation

```yaml
n_fables: 709
corpus_generation_model: gemini-2-flash

steps:
  generate_corpus_summaries: false   # reuse run dir from exp08
  generate_query_expansions: false
  run_retrieval_eval: true

query_expansion_variants:
  - name: moral_rephrase
    prompt: "Same meaning, different wording. Under 15 words."
  - name: moral_dramatic
    prompt: |
      Rewrite this moral with emotional intensity and dramatic flair.
      Keep it under 20 words. No character names.

retrieval_configs:
  - name: rephrase_only
    corpus_variant: ground_truth_style
    use_expansion: true
    expansion_variants: [moral_rephrase]
  - name: both_variants
    corpus_variant: ground_truth_style
    use_expansion: true
    expansion_variants: [moral_rephrase, moral_dramatic]
```

---

## Step Coupling: Run Dir Resolution

Steps are coupled through an explicit `run_manifest.json` written after each step:

```json
{
  "run_dir": "/path/to/results/pipeline_runs/2026-04-05_10-00-00_sample10",
  "n_fables": 10,
  "steps_completed": ["generate_corpus_summaries", "generate_query_expansions"],
  "config_snapshot": { "...": "..." }
}
```

When `--run-dir` is passed to `run_pipeline.py`, the orchestrator reads the manifest and uses that run dir for all steps. No directory listing, no fragile "latest" heuristics.

---

## Run Directory Layout

New pipeline writes to `results/pipeline_runs/` (separate from the old `results/generation_runs/`) to avoid confusion during the transition period:

```
experiments/08_symmetric_moral_matching/results/
  generation_runs/          # old scripts (untouched)
  pipeline_runs/            # new run_pipeline.py
    2026-04-05_10-00-00_sample10/
      run_manifest.json
      corpus_summaries.json
      token_usage.json
      query_expansions.json
      query_expansion_token_usage.json
      retrieval_results.json
      embedding_cache/
```

---

## Entry Point Per Experiment

```python
# experiments/09_whatever/run_pipeline.py
import argparse
from pathlib import Path
from lib.pipeline import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, default=None)
parser.add_argument("--force", action="store_true", help="Re-run steps even if output exists")
args = parser.parse_args()

run_experiment(
    config_path=Path(__file__).parent / "config.yaml",
    run_dir=args.run_dir,
    force=args.force,
)
```

---

## Backward Compatibility

The existing exp08 scripts (`generate_corpus_summaries.py`, `generate_query_expansions.py`, `run.py`) are **not deleted**. They continue to work as today. The new pipeline is additive.

---

## Issues Resolved from Code Review

| # | Issue | Resolution |
|---|-------|-----------|
| 1 | Step-to-step run dir coupling | `run_manifest.json` + explicit `--run-dir` CLI arg |
| 2 | `_metrics_from_scores()` inconsistency | `retrieval_eval.py` uses `lib/retrieval_utils.compute_metrics()` only; local copy deleted |
| 3 | Multi-line prompts in YAML | Prompts defined inline in config (preferred), or via file/key reference |
| 4 | No variant name validation | `retrieval_eval.py` validates variant names against loaded JSON at startup |
| 5 | Embedding cache not wired in | `retrieval_eval.py` uses `encode_with_cache()` from `lib/embedding_cache.py` |
| 6 | Baseline n_fables slicing | Documented: `n_fables` governs baseline slicing, no separate field needed |
| 7 | Shared run dir confusion | New pipeline writes to `pipeline_runs/`, old scripts keep `generation_runs/` |
| 8 | Single generation model field | Two fields: `corpus_generation_model`, `query_expansion_model` |
