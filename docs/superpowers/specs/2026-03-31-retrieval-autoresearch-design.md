# Retrieval Autoresearch Loop — Design Spec
**Date:** 2026-03-31
**Status:** Approved

---

## Overview

Adapt the autoresearch autonomous experimentation loop for the MORABLES moral-to-fable retrieval benchmark. An agent (Claude Code) iteratively modifies a single retrieval pipeline file, runs experiments, measures MRR on the fixed 709-fable validation set, and keeps only improvements — exactly mirroring the autoresearch philosophy but targeting retrieval instead of LLM training.

**Starting point:** 0.210 MRR (Linq-Embed-Mistral, raw embedding, no augmentation)
**Oracle ceiling:** 0.893 MRR (direct fable+moral concatenation)
**Target:** Maximize MRR on the raw embedding baseline; gains will compound with the Gemini summary pipeline (Exp 07)

---

## Directory Structure

```
experiments/08_autoresearch/
├── retrieval_pipeline.py    ← AGENT EDITS THIS ONLY
├── runner.py                ← read-only loop orchestrator
├── program.md               ← agent instructions and research guidelines
└── results.tsv              ← auto-generated experiment log
```

**No repo clone needed.** The autoresearch repo is used only for conceptual inspiration. All infrastructure reuses existing `lib/` utilities.

---

## `retrieval_pipeline.py` — The Editable File

Exposes a single entry point `run_pipeline() -> dict` that returns a metrics dict with at minimum an `"MRR"` key. Starts as the current best embedding baseline.

### Initial State (Baseline)

```python
MODEL_ID = "Linq-AI-Research/Linq-Embed-Mistral"
QUERY_INSTRUCTION = "Given a text, retrieve the most relevant passage"
CORPUS_INSTRUCTION = None

CHUNKING = "full"           # "full" | "sentences" | "sliding_N_stride_S"
CHUNK_AGG = "max"           # "max" | "mean" (used only when CHUNKING != "full")
SPARSE_WEIGHT = 0.0         # 0.0 = dense only; blend with BM25 when > 0
RERANKER_ID = None          # None | cross-encoder model path/id
RERANK_TOP_K = 50           # candidates fed to reranker

def rewrite_query(moral: str) -> str:
    return moral

def run_pipeline() -> dict:
    # loads data, encodes, retrieves, evaluates
    # prints: mrr: 0.XXXX
    # returns: {"MRR": float, "R@1": float, "R@5": float, "R@10": float}
```

### What the Agent May Modify
- `MODEL_ID` — swap to any locally available embedding model
- `QUERY_INSTRUCTION` / `CORPUS_INSTRUCTION` — instruction prefix tuning
- `CHUNKING` / `CHUNK_AGG` — chunking strategy and score aggregation
- `SPARSE_WEIGHT` — BM25 fusion coefficient (0 = pure dense)
- `RERANKER_ID` — add cross-encoder reranking stage
- `RERANK_TOP_K` — reranker candidate pool size
- `rewrite_query()` — query expansion or paraphrasing (no API calls)
- Any additional logic inside `run_pipeline()` (e.g., score normalization, ensemble)

### What the Agent Must Not Change
- The `run_pipeline() -> dict` return contract
- The `"mrr: 0.XXXX"` stdout format (parsed by runner)
- Imports from `lib/` (retrieval_utils, embedding_cache, data)
- Files outside `experiments/08_autoresearch/`

---

## `runner.py` — The Loop Orchestrator

Read-only. Never modified. Runs the autonomous experimentation loop:

```
loop forever:
  1. Read program.md + last 20 rows of results.tsv + current retrieval_pipeline.py
  2. Call Claude API: "propose next experiment as a diff to retrieval_pipeline.py"
  3. Apply diff → git commit
  4. Subprocess: python retrieval_pipeline.py (timeout: 10 min)
  5. Parse stdout for "mrr: 0.XXXX"
  6. If MRR > best_so_far:  git keep, record "keep"
     If MRR ≤ best_so_far:  git reset --hard HEAD~1, record "discard"
     If crash/timeout:       git reset --hard HEAD~1, record "crash"
  7. Append to results.tsv
```

### `results.tsv` Schema
```
commit    MRR       R@1     R@5     status    description
a1b2c3d   0.2100    0.140   0.364   baseline  Linq-Embed-Mistral, raw embedding
b2c3d4e   0.2250    0.155   0.378   keep      swap to e5-large-v2-instruct
c3d4e5f   0.2190    0.148   0.370   discard   sentence chunking with max agg
```

### Safety Details
- **Timeout:** 10 minutes per experiment (covers embedding 709×2 texts + eval on MPS)
- **Crash recovery:** `git reset --hard HEAD~1` on exception, timeout, or parse failure
- **Context window:** Last 20 results + current pipeline file fed to Claude each iteration
- **Metric printed as:** `mrr: 0.XXXX` on its own stdout line (single parseable token)

---

## `program.md` — Agent Research Guidelines

Instructs the agent on:

### Exploration Priority (ordered)
1. **Query instructions** — test 5-10 different instruction phrasings (most impactful, zero cost)
2. **Embedding models** — E5-large-instruct, BGE-large, GTE-Qwen2, Nomic-embed, etc.
3. **Query rewriting** — template expansions like "This moral means: {moral}" or abstractive reformulations
4. **Chunking strategies** — sliding window with stride, sentence-level with max/mean aggregation
5. **Sparse fusion** — BM25 weight blending (rank fusion or score fusion)
6. **Reranking** — cross-encoder models on top-50 candidates

### Heuristics
- Try the simplest change first (one axis at a time)
- Combine near-misses (two discarded ideas together may improve)
- Don't repeat exact failures
- If stuck after 5 consecutive discards, try a radical change
- Simpler is better when MRR improvement is marginal

### Hard Constraints
- No `pip install` or new packages
- No API calls inside `retrieval_pipeline.py` (local inference only)
- No modifying `lib/` utilities
- No modifying `runner.py` or `program.md`

---

## Evaluation Setup

- **Corpus:** 709 fables (fixed), 709 morals as queries (fixed)
- **Ground truth:** `data/processed/qrels_moral_to_fable.json`
- **Metric:** MRR (primary), R@1, R@5, R@10 (secondary, logged but not used for keep/discard)
- **Embedding cache:** reused from `lib/embedding_cache.py` (MD5-keyed `.npy` files)
- **Cache location:** `experiments/08_autoresearch/results/embedding_cache/`

---

## Connection to Experiment 07

Improvements to the raw embedding baseline (0.210 → X) are expected to compound additively when the Gemini summary pipeline from Exp 07 is re-run on top of the new baseline. Experiment 08 runs independently; once it converges, a follow-up experiment (09) can combine the best pipeline from 08 with Gemini summaries.

---

## Out of Scope

- LLM-generated query rewriting inside the loop (API cost, not locally runnable)
- Fine-tuning embedding models (separate experiment, not a pipeline change)
- Changing the validation set or evaluation metric
- Multi-GPU or distributed retrieval
