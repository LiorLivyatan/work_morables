# Experiment 20: Final Zero-Shot Retrieval

This experiment is the final staged zero-shot retrieval run plan for the clustered
MORABLES data.

The full tracking matrix lives at:

- `docs/zero_shot_full_tracking_matrix.csv`
- `docs/zero_shot_config_inventory.docx`
- `docs/zero_shot_model_inventory.csv`

The full matrix is intentionally large. This experiment runs it in phases so we
can make decisions after each stage instead of evaluating every combination.

## Data

All phases must use the clustered benchmark:

- Queries: `data/clustered/morals_unique_corpus.json`
- Corpus: `data/clustered/fables_corpus.json`
- Qrels: `data/clustered/qrels_moral_to_fable_clustered.json`

The qrels are multi-label: one clustered moral query may have multiple relevant
fables. Metrics must therefore use multi-label scoring.

## Phases

1. `phase_1_raw_instruction_sweep`
   - Run raw fable retrieval for all models across relevant instruction variants.
   - Choose the best instruction per model.

2. `phase_2_gemini_layout_sweep`
   - Use the best instruction per model from Phase 1.
   - Run Gemini summaries across prompt variants and layouts.

3. `phase_3_generator_comparison`
   - Use the most important models only, initially Linq and Qwen3-Embedding-8B.
   - Compare Gemini and all Gemma 4 generator variants.

## Output Policy

Each run must save:

- Aggregate metrics.
- A full ranking file with the ranked fable order for every clustered moral query.
- The resolved config used for the run.

Ranking files use the compact `ranking_v1` schema. They intentionally do not
repeat query text, fable text, or qrels because those are stable shared files in
`data/clustered/`.

```json
{
  "schema_version": "ranking_v1",
  "run_id": "Linq-Embed-Mistral__general__fable_direct_moral__gemini__direct_moral",
  "query_ids": ["moral_unique_0000"],
  "ranked_fable_ids": [["fable_0656", "fable_0285"]]
}
```

This is enough to recalculate future metrics by joining with
`data/clustered/qrels_moral_to_fable_clustered.json`.

## Shared Configuration

Summary source paths live in `summary_sources.json`. Edit that file if we choose
a different Gemma duplicate run or add a missing generator source.
