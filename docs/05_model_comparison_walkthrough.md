# Deep Walkthrough: `scripts/05_model_comparison.py`

This script is the **central experiment engine** for your thesis. It benchmarks many different text embedding models on a single task: given a **moral** (query), retrieve the correct **fable** from a corpus. Every design decision in the file serves that goal.

---

## 1. Purpose and High-Level Flow

```
Morals (queries) ──► encode_queries() ──► query embeddings (N, D)
                                                    │
                                                    │  cosine similarity
                                                    ▼
Fables (corpus)  ──► encode_corpus()  ──► corpus embeddings (M, D)
                                                    │
                                                    ▼
                                          gt_m2f (ground truth)
                                                    │
                                                    ▼
                                          MRR, NDCG, Recall@k ...
```

The script loops over a **registry** of (model, instruction-variant) pairs, loads each model once, runs both encode steps, computes metrics, saves predictions to disk, then moves to the next variant. Results are saved **incrementally** so a crash doesn't lose everything.

---

## 2. Imports

```python
import argparse, json, sys, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from retrieval_utils import compute_metrics, compute_rankings, rank_analysis
```

- `argparse` — parses `--models` and `--run-keys` CLI flags.
- `torch` / `torch.nn.functional` — used only in the low-level adapters (HFMeanPool, Qwen) that manage raw HuggingFace models. Most other adapters use `sentence-transformers` which handles PyTorch internally.
- `tqdm` — progress bars inside manual encoding loops.
- `sys.path.insert(0, ...)` — adds the `scripts/` directory to Python's import path so `retrieval_utils` can be imported as a module without installing it.
- `retrieval_utils` — shared math (metrics, ranking). Covered in section 10.

---

## 3. Global Constants

```python
DATA_DIR    = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TASK_GENERAL  = "Given a text, retrieve the most relevant passage that answers the query"
TASK_SPECIFIC = "Given a moral principle or lesson, retrieve the fable that illustrates it"
TART_INSTRUCTION = "Retrieve a fable that illustrates the following moral"
```

- `DATA_DIR` / `RESULTS_DIR` — constructed relative to the script file, not the CWD. This means you can run the script from any directory.
- The three **instruction strings** are the prompt variants used throughout the registry. They encode a key research question: *does giving the model a task description improve retrieval?*
  - `TASK_GENERAL` — generic IR instruction (domain-agnostic).
  - `TASK_SPECIFIC` — fable-domain instruction (domain-specific).
  - `TART_INSTRUCTION` — shorter, imperative form used only by the TART model.

---

## 4. Device Detection

```python
def detect_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

Returns a string tag used when constructing adapters. Priority order: Apple Silicon GPU (MPS) > NVIDIA GPU (CUDA) > CPU. The string is passed to each adapter's `__init__` and most adapters pass it directly to `SentenceTransformer(..., device=device)` or `.to(device)`.

---

## 5. The Adapter Pattern

### Why Adapters?

Each embedding model has a different API for encoding queries vs. documents:
- Some need special prefix strings (`query: `, `passage: `).
- Some need an instruction prepended in a specific format.
- Some use a `prompt_name=` keyword in `encode()`.
- Some come from `sentence-transformers`, some from raw HuggingFace `transformers`.
- Some use mean pooling, some use last-token pooling.

Adapters **normalize all of this into one interface**: `encode_queries(texts)` and `encode_corpus(texts)` — both returning an `(N, D)` `float32` numpy array that is **L2-normalized**.

### Base Class

```python
class BaseAdapter:
    def set_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def encode_queries(self, texts, batch_size=32): raise NotImplementedError
    def encode_corpus(self, texts, batch_size=32):  raise NotImplementedError
```

- `set_kwargs()` is the key enabler of **model reuse across variants**. After loading a model once, the main loop calls `adapter.set_kwargs(**kwargs)` to reconfigure the instruction/task before each run, without reloading the model weights. This saves enormous amounts of time.

---

### `SentenceTransformerAdapter`

```python
class SentenceTransformerAdapter(BaseAdapter):
    def __init__(self, model_name, device="cpu", query_prompt=None):
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prompt = query_prompt

    def encode_queries(self, texts, batch_size=32):
        if self.query_prompt:
            texts = [self.query_prompt + t for t in texts]
        return self._encode(texts, batch_size)

    def encode_corpus(self, texts, batch_size=32):
        return self._encode(texts, batch_size)
```

The generic wrapper. `query_prompt` is a string prepended to query text only (not corpus). Used by `bge-base` which expects a specific prefix string on queries. Corpus is always encoded plain.

---

### `PromptNameAdapter`

```python
class PromptNameAdapter(BaseAdapter):
    def __init__(self, model_name, device="cpu", query_prompt_name=None):
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.query_prompt_name = query_prompt_name

    def encode_queries(self, texts, batch_size=32):
        return self._encode(texts, batch_size, prompt_name=self.query_prompt_name)
```

Some models (facebook/drama-1b, Stella) define their prompts **inside the model's config file** under named keys. Instead of string-concatenation, you pass `prompt_name="s2p_query"` to `encode()` and the model looks up and applies the correct prompt internally. `trust_remote_code=True` is needed because these models ship custom Python code.

---

### `E5Adapter`

```python
class E5Adapter(BaseAdapter):
    def encode_queries(self, texts, batch_size=32):
        return self._encode(["query: " + t for t in texts], batch_size)

    def encode_corpus(self, texts, batch_size=32):
        return self._encode(["passage: " + t for t in texts], batch_size)
```

E5 models were fine-tuned with hard-coded prefix tokens `"query: "` and `"passage: "` as part of the training data. Without these prefixes, the model does not produce meaningful asymmetric embeddings. **Both** queries and corpus require prefixes (unlike most other models where only queries get a special treatment).

---

### `E5InstructAdapter`

```python
class E5InstructAdapter(BaseAdapter):
    def encode_queries(self, texts, batch_size=32):
        if self.task:
            texts = [f"Instruct: {self.task}\nQuery: {t}" for t in texts]
        return self._encode(texts, batch_size)
```

The instruct variants of E5, GTE-Qwen2, Mistral-based models use the format `"Instruct: {task}\nQuery: {text}"`. Note the **space** after `"Query:"` — this matters because these models were fine-tuned with that exact template. Corpus is always plain. `task=None` tests the model without instruction (plain mode).

---

### `HFMeanPoolAdapter`

```python
class HFMeanPoolAdapter(BaseAdapter):
    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def _encode(self, texts, batch_size):
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="encoding"):
            batch = texts[i: i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**encoded)
            embs = self._mean_pool(out.last_hidden_state, encoded["attention_mask"])
            embs = F.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)
```

Uses raw HuggingFace `AutoModel` instead of `sentence-transformers`. This is necessary for models like `facebook/contriever` which don't have a `sentence-transformers` wrapper.

**Mean pooling** explained:
- The transformer outputs one hidden vector per input token.
- We want a single vector for the whole sentence.
- Mean pooling = average the token vectors, **but only over real tokens** (not padding).
- `attention_mask` is 1 for real tokens, 0 for padding.
- `unsqueeze(-1)` broadcasts the mask to match the hidden dimension.
- `.clamp(min=1e-9)` prevents division-by-zero on empty sequences.

After pooling, `F.normalize(embs, p=2, dim=1)` L2-normalizes each vector so cosine similarity == dot product.

`with torch.no_grad()` disables gradient tracking — we're only doing inference, not training, so this saves memory and speeds up computation.

---

### `TARTAdapter`

```python
class TARTAdapter(HFMeanPoolAdapter):
    def encode_queries(self, texts, batch_size=32):
        if self.instruction:
            texts = [f"{self.instruction} [SEP] {t}" for t in texts]
        return self._encode(texts, batch_size)
```

TART (Task-Aware Retrieval with Instructions) is built on Contriever and uses the literal string `[SEP]` as a delimiter between the instruction and the query. This is different from the `[SEP]` token — it's the **text string** `" [SEP] "`, which TART was trained to recognize as a separator. Inherits the mean-pool encoding from `HFMeanPoolAdapter`.

---

### `InstructorAdapter`

```python
class InstructorAdapter(BaseAdapter):
    def _encode_pairs(self, instr, texts, batch_size):
        pairs = [[instr, t] for t in texts]
        embs = self.model.encode(pairs, batch_size=batch_size, show_progress_bar=True)
        ...
```

INSTRUCTOR uses a completely different API: instead of string concatenation, it takes a **list of `[instruction, text]` pairs**. The model was specifically designed this way to separate the instruction embedding from the text embedding. An empty string `""` as instruction is equivalent to no instruction.

---

### `JinaAdapter`

```python
class JinaAdapter(BaseAdapter):
    def encode_queries(self, texts, batch_size=32):
        return self._encode(texts, batch_size, task="retrieval.query" if self.use_task else None)

    def encode_corpus(self, texts, batch_size=32):
        return self._encode(texts, batch_size, task="retrieval.passage" if self.use_task else None)
```

Jina v3 uses a `task=` parameter in the `encode()` call to activate different task-specific LoRA adapters within the same model. The task strings are from Jina's API: `"retrieval.query"` and `"retrieval.passage"`. When `use_task=False`, no task is passed and the model uses its default (general) behavior.

---

### `QwenAdapter`

```python
class QwenAdapter(BaseAdapter):
    def __init__(self, model_name, device="cpu", task=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        ...

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        seq_lengths = attention_mask.sum(dim=1) - 1
        return last_hidden_states[
            torch.arange(len(last_hidden_states), device=last_hidden_states.device),
            seq_lengths,
        ]

    def encode_queries(self, texts, batch_size=16):
        if self.task:
            # NOTE: no space between "Query:" and query text
            texts = [f"Instruct: {self.task}\nQuery:{t}" for t in texts]
        return self._encode(texts, batch_size)
```

Qwen3-Embedding and lychee-embed are **causal (decoder-only) LLMs** used as embedders. This is fundamentally different from encoder models like BERT.

Key differences vs. encoder models:
1. **Left padding** (`padding_side="left"`): Causal models process tokens left-to-right, and the "end" of the sequence is semantically meaningful. Left-padding ensures the actual text ends at the same position for all sequences in a batch.
2. **Last-token pooling**: Instead of averaging all token vectors, you take only the **last real token's** hidden state. This works because in a causal LM, the last token has "attended to" all previous tokens — it's a summary of the whole sequence.
3. The pooling function handles both left-padded and right-padded cases by checking if the last column of `attention_mask` is all 1s (left-padding) or by computing each sequence's actual last position (right-padding).
4. The query format has **no space** after `"Query:"` — this is specific to how Qwen3/lychee were fine-tuned and is different from E5Instruct's `"Query: "` (with space).

---

### `NomicAdapter`

```python
class NomicAdapter(BaseAdapter):
    def encode_queries(self, texts, batch_size=32):
        if self.use_prefixes:
            texts = ["search_query: " + t for t in texts]

    def encode_corpus(self, texts, batch_size=32):
        if self.use_prefixes:
            texts = ["search_document: " + t for t in texts]
```

Nomic uses a prefix-based approach similar to E5, but with longer descriptive prefixes (`"search_query: "`, `"search_document: "`). The model is a **Mixture of Experts (MoE)** architecture, meaning different "expert" sub-networks activate depending on the input.

---

### `GritLMAdapter`

```python
class GritLMAdapter(BaseAdapter):
    def __init__(self, model_name, device="cpu", instruction=None):
        from gritlm import GritLM
        self.model = GritLM(model_name, torch_dtype=torch.bfloat16)
        self.instruction = instruction

    @staticmethod
    def _wrap(instruction):
        return f"<|user|>\n{instruction}\n<|embed|>\n" if instruction else "<|embed|>\n"

    def encode_queries(self, texts, batch_size=8):
        embs = self.model.encode(texts, instruction=self._wrap(self.instruction), ...)
```

GritLM-7B is a **generative + embedding** model — it can both generate text and produce embeddings. It uses special tokens from its chat template: `<|user|>` and `<|embed|>` mark the boundaries of the instruction and query. The library handles the actual tokenization and pooling. `torch.bfloat16` uses half-precision to fit the 7B model in memory.

For the corpus, `_wrap("")` returns `"<|embed|>\n"` — the empty instruction wrapper — so corpus items get no instruction but still use the embedding format.

---

### `BGEGemmaAdapter`

```python
def encode_queries(self, texts, batch_size=8):
    if self.task:
        texts = [f"<instruct>{self.task}\n<query>{t}" for t in texts]
```

BGE-Multilingual-Gemma2 uses HTML-like tags `<instruct>` and `<query>` as delimiters. These aren't HTML — they're literal strings the model was fine-tuned to recognize as separators between the task description and the query text.

---

### `NVEmbedAdapter`

Same format as E5Instruct (`"Instruct: {task}\nQuery: {text}"`). Used for NVIDIA's NV-Embed-v2. Small batch size (4) because it's a 7B Mistral-based model.

---

## 6. The MODEL_REGISTRY

```python
MODEL_REGISTRY = [
    ("run_key",  "hf_model_name",  AdapterClass,  {kwargs},  batch_size),
    ...
]
```

Each tuple is a **run specification**:

| Field | Purpose |
|---|---|
| `run_key` | Unique string ID like `"qwen3-0.6b__specific"`. Double underscore separates model from variant. |
| `model_name` | HuggingFace model ID (used for downloading and as a cache key in `loaded_models`). |
| `adapter_cls` | Which adapter class to instantiate. |
| `kwargs` | Passed to `set_kwargs()` before each run (instruction/task/prompt variant). |
| `batch_size` | Tuned per model. Larger models need smaller batches to fit in memory. |

**Key design**: multiple rows can share the same `model_name`. For example:

```python
("qwen3-0.6b__no_instr",  "Qwen/Qwen3-Embedding-0.6B", QwenAdapter, {"task": None},          16),
("qwen3-0.6b__general",   "Qwen/Qwen3-Embedding-0.6B", QwenAdapter, {"task": TASK_GENERAL},  16),
("qwen3-0.6b__specific",  "Qwen/Qwen3-Embedding-0.6B", QwenAdapter, {"task": TASK_SPECIFIC}, 16),
```

All three load the same Qwen3-0.6B weights once and run three encoding passes with different task strings. This is the most computationally expensive part — loading weights — done only once per model.

**Batch sizes** reflect model sizes:
- Small models (base-sized, ~100-300M params): 32–64
- Medium models (1-2B params): 16–32
- Large models (7-8B params): 4–8

---

## 7. Main Block — Setup

### CLI Parsing

```python
parser.add_argument("--models",    nargs="+", metavar="PATTERN")
parser.add_argument("--run-keys",  nargs="+", metavar="KEY")
```

Two filter modes:
- `--run-keys contriever__plain tart__specific` — exact match on `run_key` strings.
- `--models qwen3` — substring match: any `run_key` containing `"qwen3"` is included.
- No flags: run everything.

```python
if args.run_keys:
    key_set = set(args.run_keys)
    active_registry = [r for r in MODEL_REGISTRY if r[0] in key_set]
    unknown = key_set - {r[0] for r in MODEL_REGISTRY}
    if unknown:
        print(f"WARNING: unknown run keys: {unknown}")
elif args.models:
    patterns = args.models
    active_registry = [r for r in MODEL_REGISTRY if any(p in r[0] for p in patterns)]
```

`--run-keys` takes priority over `--models`. Unknown keys produce a warning but don't crash.

---

### Data Loading

```python
with open(DATA_DIR / "fables_corpus.json") as f:
    fables_corpus = json.load(f)
with open(DATA_DIR / "morals_corpus.json") as f:
    morals_corpus = json.load(f)
with open(DATA_DIR / "qrels_fable_to_moral.json") as f:
    qrels_f2m = json.load(f)

fable_texts = [fb["text"] for fb in fables_corpus]
moral_texts = [m["text"]  for m in morals_corpus]
```

Three files are loaded:
- `fables_corpus.json` — list of fable objects with a `"text"` field. This is the **corpus** (documents to retrieve from).
- `morals_corpus.json` — list of moral objects with a `"text"` field. These are the **queries**.
- `qrels_fable_to_moral.json` — the ground truth relevance judgments.

---

### Ground Truth Inversion

```python
gt_f2m = {}
for qrel in qrels_f2m:
    fable_idx = int(qrel["query_id"].split("_")[1])
    moral_idx = int(qrel["doc_id"].split("_")[1])
    gt_f2m[fable_idx] = moral_idx
gt_m2f = {v: k for k, v in gt_f2m.items()}
```

The `qrels` file is stored as **fable → moral** (fable_idx → moral_idx). But the task in this script is **moral → fable** (given a moral, find the fable).

So the ground truth is **inverted**: `gt_m2f = {moral_idx: fable_idx}`.

This dict is the central evaluation object. For every query (moral), it tells you which corpus item (fable) is the correct answer.

---

### Run Directory

```python
run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = RESULTS_DIR / "runs" / run_ts
preds_dir = run_dir / "predictions"
```

Each invocation creates a **unique timestamped directory**:
```
results/runs/2026-03-15_14-30-00/
    results.json       ← metrics for all runs
    metadata.json      ← config snapshot
    predictions/
        contriever__plain.json
        qwen3-0.6b__specific.json
        ...
```

This means you can run the script multiple times without overwriting previous experiments.

---

### Metadata

```python
metadata = {
    "timestamp": datetime.now().isoformat(),
    "device": device,
    "cli_args": {"models": args.models, "run_keys": args.run_keys},
    "n_queries": len(moral_texts),
    "corpus_size": len(fable_texts),
    "n_runs": len(active_registry),
    "run_keys": [r[0] for r in active_registry],
}
```

Saved immediately before the run loop. This documents **what was run and how**, useful when you return to results weeks later. Especially important for reproducibility in thesis work.

---

## 8. The Main Run Loop

```python
loaded_models = {}      # model_name -> adapter instance
skipped_models = set()  # model names that failed to load

for i, (run_key, model_name, adapter_cls, kwargs, batch_size) in enumerate(active_registry):
```

### Model Loading (with caching)

```python
if model_name in skipped_models:
    # skip all variants of a failed model
    ...
    continue

if model_name not in loaded_models:
    try:
        loaded_models[model_name] = adapter_cls(model_name, device=device)
    except Exception as e:
        if device == "mps" and _is_mps_error(e):
            # MPS fallback: retry on CPU
            loaded_models[model_name] = adapter_cls(model_name, device="cpu")
        else:
            skipped_models.add(model_name)
            ...
            continue
```

- `loaded_models` caches the adapter by `model_name`. If the next run uses the same model (different instruction variant), the model is already in memory.
- `skipped_models` prevents retrying a model that already failed — all variants of that model are marked as failed.
- **MPS fallback**: Apple Silicon's MPS backend doesn't support all operations of all models. The `_is_mps_error()` function checks if the error message mentions MPS/Metal device strings. If so, the model is reloaded on CPU instead of crashing entirely.

### Reconfiguring Adapter

```python
adapter = loaded_models[model_name]
adapter.set_kwargs(**kwargs)
```

After loading (or retrieving from cache), `set_kwargs` updates the instruction/task/prompt for this specific run variant. The model weights are unchanged — only the string formatting changes.

### Encoding

```python
t0 = time.time()
query_embs = adapter.encode_queries(moral_texts, batch_size=batch_size)
corpus_embs = adapter.encode_corpus(fable_texts, batch_size=batch_size)
elapsed = time.time() - t0
```

Both encode calls return `(N, D)` float32 numpy arrays, L2-normalized. Time is measured for the combined query + corpus encoding.

There's a second MPS fallback here too — if encoding fails on MPS (some models partially load on MPS but fail during inference), the entire adapter is reloaded on CPU.

---

### Metrics Computation

```python
metrics = compute_metrics(query_embs, corpus_embs, gt_m2f, ks=(1, 5, 10, 50))
metrics["run_key"] = run_key
metrics["model"] = model_name
metrics["encoding_time_s"] = round(elapsed, 2)
all_results.append(metrics)
```

See section 10 for how `compute_metrics` works. The returned dict includes MRR, NDCG@10, Mean/Median Rank, Recall@k, P@k for k in {1,5,10,50}.

---

### Saving Predictions

```python
rankings_data = compute_rankings(query_embs, corpus_embs, top_k=len(fable_texts))
ranks_arr = rank_analysis(query_embs, corpus_embs, gt_m2f)
gt_sorted_qidx = sorted(gt_m2f.keys())
rank_by_qidx = {q: int(r) + 1 for q, r in zip(gt_sorted_qidx, ranks_arr)}  # 1-indexed

pred_records = []
for q_idx, ranking in enumerate(rankings_data):
    correct_idx = gt_m2f.get(q_idx)
    pred_records.append({
        "query_idx": q_idx,
        "correct_idx": correct_idx,
        "correct_rank": rank_by_qidx.get(q_idx),
        "top_k_indices": ranking["indices"],
        "top_k_scores": ranking["scores"],
    })
```

For each query, the prediction file stores:
- `query_idx` — index into `morals_corpus`.
- `correct_idx` — the ground truth fable index.
- `correct_rank` — where the correct fable appeared in this model's ranking (1 = top result).
- `top_k_indices` — full ranked list of fable indices.
- `top_k_scores` — corresponding cosine similarity scores.

`top_k=len(fable_texts)` means the **entire corpus** is ranked, not just top-100. This allows post-hoc analysis at any cutoff.

`rank_by_qidx` converts from 0-indexed (from `rank_analysis`) to **1-indexed** (more natural for humans: rank 1 = first result).

---

### Incremental Save

```python
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
```

After each model run, `results.json` is **overwritten** with all results accumulated so far. If the script crashes at run 10 of 50, you still have the first 9 results.

---

### Model Eviction

```python
next_model = active_registry[i + 1][1] if i + 1 < len(active_registry) else None
if next_model != model_name:
    del loaded_models[model_name]
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
```

When the next entry in the registry uses a different model, the current model is deleted from the cache and GPU memory is freed. This is critical for large models (7-8B parameters) — you cannot fit two 7B models in memory at the same time. The registry is intentionally ordered so all variants of the same model are consecutive.

---

## 9. Summary Table

```python
printable = [r for r in all_results if "error" not in r]
sorted_results = sorted(printable, key=lambda r: r.get("MRR", 0), reverse=True)
```

At the end, all successful results are sorted by MRR (descending) and printed as a formatted table. Failed runs are listed separately. This gives you an immediate visual ranking of all models.

---

## 10. `retrieval_utils.py` — The Math

### `compute_metrics()`

```python
sim_matrix = cosine_similarity(query_embeddings, corpus_embeddings)
rankings = np.argsort(-sim_matrix, axis=1)  # descending by similarity
```

`sim_matrix[i, j]` = cosine similarity between query `i` and document `j`. Since embeddings are L2-normalized, cosine similarity = dot product. `argsort(-sim_matrix)` sorts each row from highest to lowest similarity, giving the ranked document list per query.

For each query:

```python
rank = int(np.where(ranked == correct_idx)[0][0])  # 0-indexed
```

Find where the correct document appears in the ranked list.

**MRR (Mean Reciprocal Rank)**:
```
MRR = mean(1 / rank) over all queries
```
If the correct fable is rank 1: contributes 1.0. Rank 2: 0.5. Rank 10: 0.1. Rank 100: 0.01. MRR heavily penalizes models that rank the answer low.

**Recall@k**:
```
Recall@k = 1 if correct_doc in top-k, else 0
```
Binary: did the correct answer appear in the top k results? Averaged over queries.

**P@k (Precision at k)**:
```
P@k = (number of relevant docs in top-k) / k
```
Since each query has exactly 1 relevant document, `P@k = 1/k` if hit, else 0. This is always ≤ Recall@k.

**NDCG@k (Normalized Discounted Cumulative Gain)**:
```
DCG@k    = 1 / log2(rank + 2)   if correct doc in top-k, else 0
NDCG@k   = DCG@k / ideal_DCG    where ideal_DCG = 1/log2(2) = 1
```
So `NDCG@k = 1/log2(rank+2)` when the correct doc is in top-k. This rewards finding the answer at rank 1 more than rank 5, with a logarithmic discount. With binary relevance and one relevant doc, NDCG@k = DCG@k.

**Mean/Median Rank**:
The actual 1-indexed position of the correct document. Lower is better. Mean rank is sensitive to outliers (one very bad result pulls it up). Median rank is more robust.

**R-Precision**:
With R=1 relevant document, R-Precision = Recall@1 = 1 if correct doc is rank 1, else 0.

---

### `compute_rankings()`

Returns the full ranked list for each query (indices + scores). Used to save the prediction files.

---

### `rank_analysis()`

Returns the 0-indexed rank of the correct document for each query. Used to build `rank_by_qidx` for prediction files. Returns a numpy array ordered by sorted query indices from `gt_m2f.keys()`.

---

## 11. Key Design Decisions Summary

| Decision | Why |
|---|---|
| Adapter pattern | Normalizes wildly different model APIs into one interface |
| `set_kwargs()` for variants | Load model once, run multiple instruction variants cheaply |
| L2-normalized numpy arrays | Cosine similarity = dot product; consistent interface regardless of backend |
| Incremental JSON save | Crash-safe: partial results are always preserved |
| Timestamped run directories | Preserve all experiments; never overwrite previous results |
| MPS fallback to CPU | Handles incomplete MPS support gracefully without crashing the whole run |
| Model eviction after last variant | Required to fit large models (7B) in memory when running sequentially |
| Registry as data structure | New models can be added with zero code changes — just add a row |
