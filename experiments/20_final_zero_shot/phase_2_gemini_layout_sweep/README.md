# Phase 2: Gemini Layout Sweep

## Goal

Measure the effect of summary prompt and corpus layout using one stable generator:
Gemini.

## Planned Scope

- Models: all 31 retrieval models
- Instruction: best instruction per model from Phase 1
- Generator: `gemini`
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

`31 models x 4 prompt variants x 3 layouts = 372` runs.

## Decision After This Phase

Select the best prompt/layout families and identify the strongest model families
before doing generator comparisons.

## Runner

The runner reads `docs/zero_shot_full_tracking_matrix.csv` and filters to
`summary_generator=gemini` by default. It is intentionally customizable:

```bash
./run.sh experiments/20_final_zero_shot/phase_2_gemini_layout_sweep/run.py --dry-run
./run.sh experiments/20_final_zero_shot/phase_2_gemini_layout_sweep/run.py --models Linq-Embed-Mistral --instructions general --dry-run
./run.sh experiments/20_final_zero_shot/phase_2_gemini_layout_sweep/run.py --models Linq-Embed-Mistral --prompt-variants direct_moral --corpus-configs fable_direct_moral --remote --gpu 0
```

Useful filters:

- `--models`
- `--instructions`
- `--prompt-variants`
- `--corpus-configs`
- `--limit`
- `--force`

Each completed row writes aggregate metrics and compact ranking JSONs.
