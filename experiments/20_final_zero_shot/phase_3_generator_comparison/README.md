# Phase 3: Generator Comparison

## Goal

Compare summary generators only on the most important embedding models so the
experiment stays tractable.

## Planned Scope

- Models:
  - `Linq-Embed-Mistral`
  - `Qwen3-Embedding-8B`
- Instruction: best instruction from Phase 1
- Generators:
  - `gemini`
  - `gemma4-E2B`
  - `gemma4-E4B`
  - `gemma4-26B-A4B`
  - `gemma4-31B`
- Prompt variants:
  - `direct_moral`
  - `cot_proverb`
  - `conceptual_abstract`
  - `proverb`
- Layouts:
  - `summary`
  - `fable+summary`
  - `summary+fable`

## Planned Run Count

`2 models x 5 generators x 4 prompt variants x 3 layouts = 120` runs.

## Decision After This Phase

Choose final configs for paper reporting and clustered-data confirmation.

## Runner

The runner reads `docs/zero_shot_full_tracking_matrix.csv` and excludes `raw`
configs by default. It can run any subset of generators, prompt variants,
layouts, instructions, and retrieval models.

```bash
./run.sh experiments/20_final_zero_shot/phase_3_generator_comparison/run.py --dry-run
./run.sh experiments/20_final_zero_shot/phase_3_generator_comparison/run.py --models Linq-Embed-Mistral Qwen3-Embedding-8B --instructions general --dry-run
./run.sh experiments/20_final_zero_shot/phase_3_generator_comparison/run.py --generators gemini gemma4-31B --prompt-variants direct_moral conceptual_abstract --corpus-configs fable_direct_moral direct_moral_only --remote --gpu 0
```

Useful filters:

- `--models`
- `--instructions`
- `--generators`
- `--generator-models`
- `--prompt-variants`
- `--corpus-configs`
- `--limit`
- `--force`

Summary file paths are configured in `../summary_sources.json`.
