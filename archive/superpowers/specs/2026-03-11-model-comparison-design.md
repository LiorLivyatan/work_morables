# Multi-Model Retrieval Comparison — Design Spec

**Date**: 2026-03-11
**Branch**: direction-1-improved-retrieval

---

## Overview

Evaluate 10 embedding models on the moral→fable retrieval task (morals as queries, fables as the indexed corpus). Instruction-capable models are tested in multiple variants (no instruction, general instruction, domain-specific instruction) to measure the gain from task-aware prompting. Each model uses its correct encoding protocol (prefixes, instructions, pooling strategy, normalization).

---

## Files

| Action | Path |
|--------|------|
| Create | `scripts/retrieval_utils.py` — shared `compute_metrics`, `rank_analysis` |
| Create | `scripts/05_model_comparison.py` — main experiment script |
| Rename | `05_shared_morals_plot.py` → `06_shared_morals_plot.py` |
| Rename | `06_improved_retrieval.py` → `07_improved_retrieval.py` |
| Rename | `07_llm_reranking.py` → `08_llm_reranking.py` |
| Output | `results/model_comparison_results.json` |

---

## Task

**Moral → Fable (clean)**
- 709 moral texts as queries
- 709 fable texts as corpus
- Ground truth: `qrels_fable_to_moral.json` — file maps `fable_idx → moral_idx`; invert at load time to get the `moral_idx → fable_idx` mapping needed for this task (same as the `gt_m2f = {v: k for k, v in gt_f2m.items()}` pattern in script 04)

---

## Metrics

All computed in `retrieval_utils.compute_metrics()`:

| Metric | Notes |
|--------|-------|
| MRR | Mean Reciprocal Rank |
| R-Precision | With R=1 per query, equivalent to Recall@1 |
| Recall@1 | |
| Recall@5 | |
| Recall@10 | |
| Recall@50 | |
| MAP | Mean Average Precision — equals MRR when each query has exactly 1 relevant doc; computed explicitly for completeness |

Signature: `compute_metrics(query_embs, corpus_embs, ground_truth, ks=[1,5,10,50]) -> dict`

---

## Device

Primary device: **MPS** (Mac Silicon). Detect at startup:
```python
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
```
Pass `device` to all adapters. If an adapter raises an `AcceleratorError` on MPS, fall back to CPU with a printed warning (known issue for some models).

---

## Instruction Variants

Two shared task description strings used across all instruction-capable models:

```python
TASK_GENERAL  = "Given a text, retrieve the most relevant passage that answers the query"
TASK_SPECIFIC = "Given a moral principle or lesson, retrieve the fable that illustrates it"
```

Each instruction-capable model is run **three times**: `no_instr`, `general`, `specific`.
Models with a fixed-parameter instruction API (Jina, TART) run **two times**: `no_instr` + their best available mode.
Plain models run **once**.

| Model | Variants | Notes |
|-------|----------|-------|
| `facebook/contriever` | `plain` | No instruction support |
| `orionweller/tart-dual-contriever-msmarco` | `no_instr`, `specific` | Uses `[SEP]` delimiter; specific = domain instruction |
| `hkunlp/instructor-base` | `no_instr`, `general`, `specific` | List-of-pairs API; each variant has its own instruction text |
| `BAAI/bge-base-en-v1.5` | `plain` | Fixed query prefix |
| `intfloat/e5-base-v2` | `plain` | Fixed `query:` / `passage:` prefix |
| `intfloat/multilingual-e5-large` | `plain` | Fixed `query:` / `passage:` prefix |
| `intfloat/multilingual-e5-large-instruct` | `no_instr`, `general`, `specific` | `f'Instruct: {task}\nQuery: {query}'` |
| `BAAI/bge-m3` | `plain` | No instruction support in dense mode |
| `jinaai/jina-embeddings-v3` | `no_instr`, `retrieval` | `no_instr` = no task kwarg; `retrieval` = `task="retrieval.query"` |
| `Qwen/Qwen3-Embedding-0.6B` | `no_instr`, `general`, `specific` | `f'Instruct: {task}\nQuery:{query}'` (no space before query) |

**Result key naming**: `{model_shortname}__{variant}`, e.g. `multilingual-e5-large-instruct__specific`.

---

## Instruction Formats Per Model

Each model family has a **unique instruction API** — adapters handle this internally.

### `multilingual-e5-large-instruct` and `Qwen3-Embedding`

```python
# E5-instruct
f'Instruct: {task}\nQuery: {query}'

# Qwen3 (note: NO space between "Query:" and query text)
f'Instruct: {task}\nQuery:{query}'
```

General variant: `task = TASK_GENERAL`
Specific variant: `task = TASK_SPECIFIC`
No-instr variant: encode plain text, no prefix

### `instructor-base`

```python
# List-of-pairs API
[[instruction, text], ...]

# General variant instructions:
query_instruction   = "Represent the text for retrieval:"
corpus_instruction  = "Represent the passage:"

# Specific variant instructions:
query_instruction   = "Represent the moral for retrieving a fable:"
corpus_instruction  = "Represent the fable:"

# No-instr variant:
query_instruction   = ""   # empty string — model still uses list-of-pairs format
corpus_instruction  = ""
```

### `tart-dual-contriever-msmarco`

TART was trained on short imperative instructions; use a concise domain instruction rather than the shared `TASK_SPECIFIC` string:

```python
TART_INSTRUCTION = "retrieve a fable that illustrates the following moral"

# With instruction (specific):
f"{TART_INSTRUCTION} [SEP] {query}"

# No instruction:
query text as-is (plain, no [SEP])
```

### `jina-embeddings-v3`

```python
# retrieval variant:
model.encode(queries, task="retrieval.query")
model.encode(corpus, task="retrieval.passage")

# no_instr variant:
model.encode(queries)   # no task kwarg
model.encode(corpus)
```

---

## Models & Registry

```python
MODEL_REGISTRY = [
    # (run_key, model_name, adapter_cls, adapter_kwargs, batch_size)
    ("contriever__plain",                  "facebook/contriever",                        HFMeanPoolAdapter,      {},                                    64),
    ("tart__no_instr",                     "orionweller/tart-dual-contriever-msmarco",   TARTAdapter,            {"instruction": None},                 64),
    ("tart__specific",                     "orionweller/tart-dual-contriever-msmarco",   TARTAdapter,            {"instruction": TART_INSTRUCTION},     64),
    ("instructor__no_instr",               "hkunlp/instructor-base",                     InstructorAdapter,      {"query_instr": "", "corpus_instr": ""},32),
    ("instructor__general",                "hkunlp/instructor-base",                     InstructorAdapter,      {"query_instr": "Represent the text for retrieval:", "corpus_instr": "Represent the passage:"}, 32),
    ("instructor__specific",               "hkunlp/instructor-base",                     InstructorAdapter,      {"query_instr": "Represent the moral for retrieving a fable:", "corpus_instr": "Represent the fable:"}, 32),
    ("bge-base__plain",                    "BAAI/bge-base-en-v1.5",                      SentenceTransformerAdapter, {"query_prompt": "Represent this sentence for searching relevant passages: "}, 64),
    ("e5-base__plain",                     "intfloat/e5-base-v2",                        E5Adapter,              {},                                    64),
    ("multilingual-e5__plain",             "intfloat/multilingual-e5-large",             E5Adapter,              {},                                    32),
    ("multilingual-e5-instruct__no_instr", "intfloat/multilingual-e5-large-instruct",    E5InstructAdapter,      {"task": None},                        32),
    ("multilingual-e5-instruct__general",  "intfloat/multilingual-e5-large-instruct",    E5InstructAdapter,      {"task": TASK_GENERAL},                32),
    ("multilingual-e5-instruct__specific", "intfloat/multilingual-e5-large-instruct",    E5InstructAdapter,      {"task": TASK_SPECIFIC},               32),
    ("bge-m3__plain",                      "BAAI/bge-m3",                                SentenceTransformerAdapter, {},                                32),
    ("jina-v3__no_instr",                  "jinaai/jina-embeddings-v3",                  JinaAdapter,            {"use_task": False},                   32),
    ("jina-v3__retrieval",                 "jinaai/jina-embeddings-v3",                  JinaAdapter,            {"use_task": True},                    32),
    ("qwen3__no_instr",                    "Qwen/Qwen3-Embedding-0.6B",                  QwenAdapter,            {"task": None},                        16),
    ("qwen3__general",                     "Qwen/Qwen3-Embedding-0.6B",                  QwenAdapter,            {"task": TASK_GENERAL},                16),
    ("qwen3__specific",                    "Qwen/Qwen3-Embedding-0.6B",                  QwenAdapter,            {"task": TASK_SPECIFIC},               16),
]
```

Models that share the same `model_name` are loaded once and reused across their variants.

---

## Adapter Architecture

All adapters implement:

```python
class BaseAdapter:
    def encode_queries(self, texts: list[str], batch_size: int) -> np.ndarray: ...
    def encode_corpus(self, texts: list[str], batch_size: int) -> np.ndarray: ...
```

**Contract**: all adapters return **L2-normalized** float32 numpy arrays of shape `(N, D)`.

---

### Adapter Details

**`SentenceTransformerAdapter`**
Wraps `SentenceTransformer(..., device=device)`. Passes `normalize_embeddings=True`.
Optional `query_prompt` string prepended to query texts only.

**`HFMeanPoolAdapter`**
Loads via `AutoTokenizer` + `AutoModel`. Computes mean pooling over last hidden state weighted by attention mask. Applies `F.normalize(..., p=2, dim=1)`.

**`TARTAdapter`**
Extends `HFMeanPoolAdapter`. With `instruction` set: formats queries as `f"{instruction} [SEP] {query}"`. With `instruction=None`: encodes queries plain.

**`InstructorAdapter`**
Wraps `InstructorEmbedding.INSTRUCTOR`. Always uses list-of-pairs format `[[instr, text]]`. With empty string instructions, the model behaves as a plain encoder.

**`E5Adapter`**
Wraps `SentenceTransformer`. Always prepends `"query: "` / `"passage: "`. `normalize_embeddings=True`.

**`E5InstructAdapter`**
Wraps `SentenceTransformer`. With `task` set: `f'Instruct: {task}\nQuery: {query}'`. With `task=None`: plain encoding. Corpus always plain. `normalize_embeddings=True`.

**`JinaAdapter`**
Loads `SentenceTransformer(..., trust_remote_code=True)`. With `use_task=True`: passes `task="retrieval.query"` / `task="retrieval.passage"` to `encode()`. With `use_task=False`: no task kwarg.

**`QwenAdapter`**
Loads via `AutoTokenizer` + `AutoModel`. Uses **last-token pooling**:
```python
def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        seq_lengths = attention_mask.sum(dim=1) - 1
        return last_hidden_states[torch.arange(len(last_hidden_states), device=last_hidden_states.device), seq_lengths]
```
With `task` set: `f'Instruct: {task}\nQuery:{query}'` (no space before query text). With `task=None`: plain encoding. Applies `F.normalize(..., p=2, dim=1)`.

---

## Main Loop

Load each unique model once; reuse across its variants:

```python
device = detect_device()
loaded_models = {}  # model_name -> adapter instance (reused across variants of the same model)

for i, (run_key, model_name, adapter_cls, kwargs, batch_size) in enumerate(MODEL_REGISTRY):
    if model_name not in loaded_models:
        loaded_models[model_name] = adapter_cls(model_name, device=device)
    adapter = loaded_models[model_name]
    adapter.set_kwargs(**kwargs)  # update instruction/task for this variant

    t0 = time.time()
    query_embs  = adapter.encode_queries(moral_texts, batch_size=batch_size)
    corpus_embs = adapter.encode_corpus(fable_texts, batch_size=batch_size)
    elapsed = time.time() - t0

    metrics = compute_metrics(query_embs, corpus_embs, gt_m2f, ks=[1, 5, 10, 50])
    metrics["run_key"] = run_key
    metrics["model"] = model_name
    metrics["encoding_time_s"] = round(elapsed, 2)
    all_results.append(metrics)

    # Evict model from memory when all its variants are done
    next_model = MODEL_REGISTRY[i + 1][1] if i + 1 < len(MODEL_REGISTRY) else None
    if next_model != model_name:
        del loaded_models[model_name]
        torch.mps.empty_cache() if device == "mps" else None
```

---

## Output

`results/model_comparison_results.json` — list of dicts, one per run:
```json
{
  "run_key": "multilingual-e5-instruct__specific",
  "model": "intfloat/multilingual-e5-large-instruct",
  "MRR": 0.51,
  "MAP": 0.51,
  "R-Precision": 0.47,
  "Recall@1": 0.47,
  "Recall@5": 0.68,
  "Recall@10": 0.75,
  "Recall@50": 0.89,
  "encoding_time_s": 18.3
}
```

Summary table printed at end, sorted by MRR descending.

---

## Dependencies to add to `requirements.txt`

- `InstructorEmbedding` — for instructor-base
- `einops` — required by some models
- `torch` — explicit pin (used by HF adapters)
- `transformers` — explicit pin
