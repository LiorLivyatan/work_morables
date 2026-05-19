# Phase 1: Raw Instruction Sweep

## Goal

Choose the best query instruction format for each embedding/retrieval model before
running expensive summary/layout/generator sweeps.

## Scope

- Corpus layout: `raw` only (`{fable}`)
- Queries: clustered unique morals
- Corpus: all 709 fables
- Metrics: multi-label metrics over clustered qrels
- Instructions:
  - `no_instr`
  - `general`
  - `moral_specific`
  - `default` only for models where the tracking matrix defines a native/default
    behavior

## Why This Comes First

Instructions multiply the rest of the grid. Running them on raw fables first is
a cheap screening step. The winning instruction per model is then used in Phase 2.

## Outputs

Results are written under `results/phase_1_raw_instruction_sweep/<timestamp>/`:

- `metrics.csv`
- `metrics.json`
- `rankings/<run_id>.json`
- `run_config.json`

The ranking JSON stores the full ranked fable order for every clustered moral
query.

## Example

```bash
./run.sh experiments/20_final_zero_shot/phase_1_raw_instruction_sweep/run.py --models all-MiniLM-L6-v2
```

