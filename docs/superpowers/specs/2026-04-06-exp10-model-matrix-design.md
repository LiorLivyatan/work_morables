# Exp10 — Local Model Matrix Design

**Date:** 2026-04-06  
**Experiment:** `experiments/10_model_matrix`  
**Status:** Approved

---

## Goal

Evaluate all combinations of local generation models × embedding models on the MORABLES retrieval task (fable → moral matching). Identify the best generation model, best embedding model, and best pair. Answer whether performance depends more on generation quality or embedding quality.

---

## Constraints

- No cloud LLM APIs — all generation runs locally via HuggingFace Transformers
- Hardware: Mac M2 Pro, MPS backend, bfloat16
- One model loaded at a time (avoid OOM)
- Everything cached — no recomputation on reruns
- Fully config-driven — new experiments need only a new `config.yaml`

---

## Models

### Generation models (summarization + paraphrasing)

| Alias | HuggingFace ID |
|---|---|
| Qwen3-8B | Qwen/Qwen3-8B-Instruct |
| Gemma-4B | google/gemma-3-4b-it |
| Phi-3.5-mini | microsoft/Phi-3.5-mini-instruct |

Same model handles both corpus summarization and query paraphrasing.

### Embedding models

| Alias | HuggingFace ID |
|---|---|
| Qwen3-Embed-8B | Qwen/Qwen3-Embedding-8B |
| BGE-M3 | BAAI/bge-m3 |
| pplx-embed-v1 | (update to exact HF ID when confirmed) |

---

## Pipeline Stages

### Stage 1 — Local Generation (per gen model, fully cached)

**Corpus summarization:** fable text → 1 declarative moral summary  
**Query paraphrasing:** moral text → 3 rephrases

Each gen model runs once. Results cached in `gen_cache/<alias>/`. Skipped on reruns unless `prompt_version` changes or `--force` is passed.

Model loading order to avoid MPS OOM:
```
load gen_model_1 → generate all fables + morals → unload
load gen_model_2 → generate all → unload
load gen_model_3 → generate all → unload
```

### Stage 2 — Matrix Retrieval Eval (N×M combos × 4 ablations)

For each `(gen_model, embed_model)` pair, run 4 ablation modes:

| Ablation | Corpus | Query |
|---|---|---|
| `raw_raw` | raw fable text | raw moral |
| `summary_only` | gen model summary | raw moral |
| `paraphrase_only` | raw fable text | original + filtered rephrases |
| `full` | gen model summary | original + filtered rephrases |

Retrieval: cosine similarity. For paraphrase modes: max-score fusion over `[original_emb, r1_emb, r2_emb, r3_emb]`.

Embedding model loading order (embed all gen outputs before switching):
```
load embed_model_1 → embed all gen_model combos → unload
load embed_model_2 → ...
```

### Stage 3 — Aggregation

Reads all `retrieval_results/`, produces `matrix_summary.json` and `rankings.json`.

---

## Prompts

### Summarization prompt

```
You are an expert in moral philosophy. Distill the lesson of the following
fable into one declarative sentence of 5 to 15 words. The statement must be
universal and timeless — no character names, no reference to story events.
Output ONLY the moral sentence. No explanation.
```

Applied via each model's tokenizer chat template (no manual formatting).  
`prompt_version: "v1"` — bump to bust generation cache.

### Paraphrase prompt

```
You are given a moral statement. Write exactly 3 different rephrasings using
different words while preserving the exact same meaning. Each must be abstract
and universal, 5 to 15 words.
Output ONLY the 3 rephrasings, one per line. No numbers, no labels.
```

---

## Post-Processing & Quality Gates

### Word count enforcement
- Range: 5–15 words
- Action: `flag` (log out-of-range outputs, do not silently truncate)
- Out-of-range items recorded in `diagnostics.json`

### Paraphrase filtering
- Embed each rephrase + original moral using `BAAI/bge-m3`
- Drop any rephrase with cosine similarity < 0.85 to original
- Filtered rephrases logged in `diagnostics.json` (not silently dropped)
- If all 3 rephrases are filtered, fall back to original moral only

### Summary diagnostics
- `unique_ratio`: `len(set(summaries)) / n_fables` — detects model collapse
- `duplicate_rate`: fraction of near-identical summaries (sim > 0.95)
- `generic_matches`: flag outputs matching known degenerate phrases
- Written to `gen_cache/<alias>/diagnostics.json`

---

## Caching Design

Three independent cache layers:

| Layer | Key | Location |
|---|---|---|
| Generation | `(alias, prompt_version, item_id)` | `gen_cache/<alias>/*.json` |
| Embedding | `sha256(model_id + instruction + text)` | `embedding_cache/<alias>/<hash>.npy` |
| Paraphrase filter | same as embedding (BGE-M3 embeddings) | `embedding_cache/bge-m3/<hash>.npy` |

Cache bust for generation: change `prompt_version` in config.  
Cache bust for embeddings: model alias or instruction change automatically produces different keys.

---

## Run Directory Structure

```
experiments/10_model_matrix/results/pipeline_runs/<timestamp>_sample50/
  gen_cache/
    Qwen3-8B/
      corpus_summaries.json
      query_paraphrases.json
      diagnostics.json
    Gemma-4B/
      ...
    Phi-3.5-mini/
      ...
  embedding_cache/
    Qwen3-Embed-8B/<hash>.npy
    BGE-M3/<hash>.npy
    ...
  retrieval_results/
    Qwen3-8B__Qwen3-Embed-8B__full.json
    Qwen3-8B__Qwen3-Embed-8B__raw_raw.json
    ...  (N × M × 4 ablations = 36 files for 3×3 grid)
  predictions/
    Qwen3-8B__Qwen3-Embed-8B__full.json
    ...
  matrix_summary.json
  rankings.json
  run_manifest.json
```

---

## Output Schema

### `retrieval_results/<combo>__<ablation>.json`
```json
{
  "gen_model": "Qwen3-8B",
  "embed_model": "BGE-M3",
  "ablation": "full",
  "n_queries": 50,
  "metrics": {
    "Recall@1": 0.72,
    "Recall@5": 0.91,
    "MRR": 0.801
  }
}
```

### `predictions/<combo>__<ablation>.json`
```json
{
  "combo": "Qwen3-8B__BGE-M3__full",
  "queries": [
    {
      "query_idx": 0,
      "moral_text": "Appearances are deceptive.",
      "correct_fable_idx": 3,
      "correct_rank": 1,
      "top_k_indices": [3, 7, 12],
      "top_k_scores": [0.91, 0.83, 0.71]
    }
  ]
}
```

### `matrix_summary.json`
```json
{
  "ablation": "full",
  "metric": "Recall@1",
  "matrix": {
    "rows": ["Qwen3-8B", "Gemma-4B", "Phi-3.5-mini"],
    "cols": ["Qwen3-Embed-8B", "BGE-M3", "pplx-embed-v1"],
    "values": [[0.72, 0.68, 0.65], [0.61, 0.58, 0.54], [0.66, 0.63, 0.60]]
  },
  "all_ablations": {}
}
```

### `rankings.json`
```json
{
  "best_gen_model": { "name": "Qwen3-8B", "avg_recall_at_1": 0.683, "avg_mrr": 0.751 },
  "best_embed_model": { "name": "Qwen3-Embed-8B", "avg_recall_at_1": 0.663, "avg_mrr": 0.730 },
  "best_pair": { "gen_model": "Qwen3-8B", "embed_model": "Qwen3-Embed-8B", "ablation": "full", "Recall@1": 0.72, "MRR": 0.801 },
  "impact_analysis": {
    "gen_model_variance": 0.042,
    "embed_model_variance": 0.019,
    "dominant_factor": "generation"
  }
}
```

---

## New Files

### `lib/pipeline/`

| File | Responsibility |
|---|---|
| `local_llm.py` | Load HF model to MPS, apply chat template, generate, unload |
| `local_corpus_generator.py` | fable → summary per gen model, writes `corpus_summaries.json` + `diagnostics.json` |
| `local_query_paraphraser.py` | moral → 3 rephrases, filters, writes `query_paraphrases.json` |
| `paraphrase_filter.py` | Embedding similarity filter + word count post-processor |
| `matrix_runner.py` | N×M orchestrator — loops gen models, then embed models |
| `matrix_aggregator.py` | Reads all retrieval results, builds matrix + rankings |

### Modified files

| File | Change |
|---|---|
| `lib/pipeline/retrieval_eval.py` | Accept `embed_model_id` override + `ablation_mode` param |
| `lib/pipeline/default_config.yaml` | Add `generation_models: []`, `embed_models: []` keys |
| `lib/pipeline/__init__.py` | Export `run_matrix_experiment` |

### Exp10

```
experiments/10_model_matrix/
  config.yaml
  run_pipeline.py
  README.md
  results/pipeline_runs/
```

`run_pipeline.py` is a 6-line shim:
```python
from lib.pipeline import run_matrix_experiment
run_matrix_experiment(config_path=Path(__file__).parent / "config.yaml", ...)
```

---

## Unchanged Files

`corpus_generator.py`, `query_expander.py`, `llm_client.py`, `run_utils.py`, and all existing experiments are untouched.

---

## Extensibility

To run a future grid experiment:
1. Create `experiments/11_.../config.yaml` with new `generation_models` and `embed_models` lists
2. Create a 6-line `run_pipeline.py` calling `run_matrix_experiment()`
3. No pipeline code changes needed

To extend an existing experiment:
- Add a model to `generation_models` or `embed_models` in config and rerun — cached results are preserved, only new models execute
