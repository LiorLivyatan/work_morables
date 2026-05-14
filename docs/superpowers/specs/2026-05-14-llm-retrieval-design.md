# LLM Retrieval Experiment — Design Spec

**Date:** 2026-05-14  
**Experiment dir:** `llm_retrieval/` (top-level, separate from `experiments/`)  
**Goal:** Test whether frontier LLMs can directly solve the moral-to-fable retrieval task when the full corpus is placed in context. Serves as a "naive LLM oracle baseline" against fine-tuned bi-encoder results.

---

## Research Question

Given a moral statement and all 709 fables enumerated in the prompt, can an LLM correctly rank the relevant fable in its top-10? Expected finding: even frontier models underperform fine-tuned retrievers due to position bias, lost-in-the-middle effects, and the semantic subtlety of the task.

---

## Directory Structure

```
llm_retrieval/
├── README.md                        # overview, how to run, results table
├── config.yaml                      # all models, variants, settings
├── run.py                           # main entry point
├── lib/
│   ├── __init__.py
│   ├── corpus.py                    # builds the [fable_id] + text prompt block
│   ├── prompt.py                    # renders prompt variants from config + corpus
│   ├── providers.py                 # Agno model factory (openai/anthropic/google/openrouter)
│   ├── eval.py                      # MRR@10, R@1/5/10, NDCG@10
│   └── results.py                   # incremental per-run CSV + unified CSV merge
└── results/
    ├── runs/                        # one CSV per (model × variant) run
    └── unified.csv                  # merged summary across all runs
```

---

## Task Formulation

- **Input to LLM:** moral statement + full corpus block of 709 fables, each formatted as `[fable_XXXX] <full text>`
- **Output from LLM:** structured JSON array of exactly `top_k` fable IDs (default: 10), enforced via Agno + Pydantic
- **Structured output model:**
  ```python
  class RankedFables(BaseModel):
      ids: list[str]  # e.g. ["fable_0042", "fable_0107", ...]
  ```
- No titles, no metadata — raw text only, consistent with how embedding models (Linq, BGE, etc.) see the corpus

---

## Config Shape

```yaml
top_k: 10
test_sizes: [50, 100, 200]   # documented suggested values; --test N accepts any integer

prompt_variants:
  - label: minimal
    system: "You are a retrieval system."
    user_template: |
      Moral: {moral}

      Corpus:
      {corpus}

      Return the {top_k} fable IDs most relevant to this moral as a JSON array.

  - label: detailed
    system: "..."
    user_template: "..."    # fuller task description + relevance definition

  - label: cot
    system: "Think step by step before answering."
    user_template: "..."    # asks model to reason before ranking

models:
  - alias: GPT-4o-mini
    id: gpt-4o-mini
    provider: openai
    concurrency: 20

  - alias: GPT-4o
    id: gpt-4o
    provider: openai
    concurrency: 10

  - alias: GPT-5.5
    id: gpt-5.5
    provider: openai
    concurrency: 5

  - alias: GPT-OSS-20B
    id: VERIFY_OPENROUTER_ID   # verify exact OpenRouter model slug before running
    provider: openrouter
    concurrency: 10

  - alias: GPT-OSS-120B
    id: VERIFY_OPENROUTER_ID   # verify exact OpenRouter model slug before running
    provider: openrouter
    concurrency: 5

  - alias: Claude-Haiku
    id: claude-haiku-4-5-20251001
    provider: anthropic
    concurrency: 10

  - alias: Claude-Sonnet
    id: claude-sonnet-4-6
    provider: anthropic
    concurrency: 5

  - alias: Claude-Opus
    id: claude-opus-4-7
    provider: anthropic
    concurrency: 3

  - alias: Gemini-2.5-Pro
    id: gemini-2.5-pro
    provider: google
    concurrency: 5

  - alias: Gemini-3-Flash
    id: gemini-3-flash         # update when available
    provider: google
    concurrency: 10

  - alias: Gemini-3.1-Pro
    id: gemini-3.1-pro         # update when available
    provider: google
    concurrency: 5

  - alias: Qwen3.5-Flash
    id: qwen/qwen3.5-flash
    provider: openrouter
    concurrency: 10

  - alias: Qwen3.6-Plus
    id: qwen/qwen3.6-plus
    provider: openrouter
    concurrency: 5

  - alias: Qwen3.6-27B
    id: qwen/qwen3.6-27b
    provider: openrouter
    concurrency: 5

  - alias: Llama-4-Scout
    id: meta-llama/Llama-4-Scout-17B-Instruct
    provider: openrouter
    concurrency: 10
```

---

## Execution Flow

```
run.py --models GPT-4o --variants minimal --test 100
  │
  ├─ Load config.yaml
  ├─ Load morals + qrels (all 709, or N if --test N)
  ├─ Build corpus block once (shared across all calls for this run)
  │
  └─ For each (model, variant) pair:
        │
        ├─ Init Agno model client (provider-specific)
        ├─ Create asyncio semaphore(model.concurrency)
        │
        ├─ Dispatch all N moral calls concurrently (throttled)
        │    Each call:
        │      → render prompt(moral, corpus, variant)
        │      → Agno structured call → RankedFables
        │      → compute per-query metrics immediately
        │      → append row to results/runs/YYYY-MM-DD_<alias>_<variant>.csv
        │
        └─ Aggregate metrics → append summary row to results/unified.csv
```

**Resumability:** `--skip-existing` checks for a completed run file in `results/runs/` before starting. A run file is considered complete when its row count equals the expected query count; if fewer rows exist, the run is treated as partial and resumed from where it left off.

---

## CLI

```bash
# Full run, all models, all variants
./run.sh llm_retrieval/run.py

# Test mode
./run.sh llm_retrieval/run.py --test 100 --models GPT-4o-mini --variants minimal

# Multiple models, specific variants
./run.sh llm_retrieval/run.py --models GPT-4o Claude-Sonnet --variants minimal detailed

# Skip already-completed runs
./run.sh llm_retrieval/run.py --skip-existing

# Force re-run
./run.sh llm_retrieval/run.py --force
```

---

## Results Format

**Per-run CSV** (`results/runs/YYYY-MM-DD_<alias>_<variant>.csv`):
```
moral_id | moral_text | relevant_fable | ranked_ids | reciprocal_rank | r@1 | r@5 | r@10
```

**Unified CSV** (`results/unified.csv`):
```
run_date | model_alias | model_id | provider | variant_label |
n_queries | MRR@10 | R@1 | R@5 | R@10 | NDCG@10 |
Mean_Rank | Median_Rank | avg_latency_s | total_cost_usd_est
```

---

## Metrics

Same as existing experiments: **MRR@10, R@1, R@5, R@10, NDCG@10, Mean_Rank, Median_Rank**.  
Computed via the existing `finetuning.lib.eval` utilities where possible.

---

## Dependencies to Add

- `agno` — unified LLM client + structured output
- `openai`, `anthropic`, `google-generativeai` — native SDKs (some already present)
- API keys required in `.env`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, `TOGETHER_API_KEY`

---

## Out of Scope

- OSS models served locally via vLLM (API-only for now)
- Corpus variants (titles, snippets) — raw text only in this experiment
- Few-shot prompting — zero-shot only
