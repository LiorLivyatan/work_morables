# Multi-Model Retrieval Comparison Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate 10 embedding models (18 total runs with instruction variants) on the moral→fable retrieval task, producing a ranked comparison table with MRR, MAP, Recall@1/5/10/50.

**Architecture:** Shared `retrieval_utils.py` holds metrics functions; `05_model_comparison.py` holds all adapter classes + MODEL_REGISTRY + main loop. Each adapter wraps a different model family's encoding API and exposes a uniform `encode_queries` / `encode_corpus` interface returning L2-normalized numpy arrays. Models are loaded once per `model_name` and reused across instruction variants.

**Tech Stack:** Python, sentence-transformers, transformers (HuggingFace), InstructorEmbedding, torch (MPS), numpy, scikit-learn

**Spec:** `docs/superpowers/specs/2026-03-11-model-comparison-design.md`

---

## Chunk 1: Foundation — rename scripts, extract utils, update deps

### Task 1: Rename existing scripts to make room for 05

**Files:**
- Rename: `scripts/05_shared_morals_plot.py` → `scripts/06_shared_morals_plot.py`
- Rename: `scripts/06_improved_retrieval.py` → `scripts/07_improved_retrieval.py`
- Rename: `scripts/07_llm_reranking.py` → `scripts/08_llm_reranking.py`

- [ ] **Step 1: Rename the three scripts**

```bash
cd /Users/liorlivyatan/LocalProjects/Thesis/work_morables/scripts
mv 05_shared_morals_plot.py 06_shared_morals_plot.py
mv 06_improved_retrieval.py 07_improved_retrieval.py
mv 07_llm_reranking.py 08_llm_reranking.py
```

- [ ] **Step 2: Verify**

```bash
ls scripts/0*.py
```
Expected: `04_baseline_retrieval.py`, `06_shared_morals_plot.py`, `07_improved_retrieval.py`, `08_llm_reranking.py` — no `05_`, `06_improved_`, or `07_llm_` files.

- [ ] **Step 3: Commit**

```bash
git add -A scripts/
git commit -m "chore: bump script numbering 05-07 to 06-08 to make room for 05_model_comparison"
```

---

### Task 2: Create `retrieval_utils.py` with metrics functions

Extract and extend `compute_metrics` and `rank_analysis` from `04_baseline_retrieval.py`. Adds `Recall@50` and `MAP`.

**Files:**
- Create: `scripts/retrieval_utils.py`

- [ ] **Step 1: Create the file**

```python
# scripts/retrieval_utils.py
"""Shared retrieval metrics used across experiment scripts."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_metrics(query_embeddings, corpus_embeddings, ground_truth, ks=(1, 5, 10, 50)):
    """
    Compute retrieval metrics for a set of queries against a corpus.

    Args:
        query_embeddings: (N, D) float32 array, L2-normalized
        corpus_embeddings: (M, D) float32 array, L2-normalized
        ground_truth: dict mapping query_idx (int) -> correct corpus_idx (int)
        ks: tuple of k values for Recall@k

    Returns:
        dict with keys: MRR, MAP, R-Precision, Recall@k for each k, n_queries
    """
    sim_matrix = cosine_similarity(query_embeddings, corpus_embeddings)
    rankings = np.argsort(-sim_matrix, axis=1)  # descending by similarity

    reciprocal_ranks = []
    recall_at_k = {k: [] for k in ks}
    r_precisions = []

    for q_idx in range(len(query_embeddings)):
        if q_idx not in ground_truth:
            continue
        correct_idx = ground_truth[q_idx]
        ranked = rankings[q_idx]
        rank = int(np.where(ranked == correct_idx)[0][0])  # 0-indexed

        reciprocal_ranks.append(1.0 / (rank + 1))
        for k in ks:
            recall_at_k[k].append(1.0 if rank < k else 0.0)
        r_precisions.append(1.0 if rank == 0 else 0.0)

    results = {
        "MRR": float(np.mean(reciprocal_ranks)),
        # MAP == MRR when each query has exactly 1 relevant doc; computed explicitly
        "MAP": float(np.mean(reciprocal_ranks)),
        "R-Precision": float(np.mean(r_precisions)),
        "n_queries": len(reciprocal_ranks),
    }
    for k in ks:
        results[f"Recall@{k}"] = float(np.mean(recall_at_k[k]))
    return results


def rank_analysis(query_embeddings, corpus_embeddings, ground_truth):
    """Return 0-indexed rank of the correct doc for each query."""
    sim_matrix = cosine_similarity(query_embeddings, corpus_embeddings)
    rankings = np.argsort(-sim_matrix, axis=1)
    ranks = []
    for q_idx in range(len(query_embeddings)):
        if q_idx not in ground_truth:
            continue
        correct_idx = ground_truth[q_idx]
        rank = int(np.where(rankings[q_idx] == correct_idx)[0][0])
        ranks.append(rank)
    return np.array(ranks)
```

---

### Task 3: Write and run tests for `retrieval_utils`

**Files:**
- Create: `tests/test_retrieval_utils.py`

- [ ] **Step 1: Create tests directory and write tests**

```bash
mkdir -p tests && touch tests/__init__.py
```

```python
# tests/test_retrieval_utils.py
import numpy as np
import pytest
import sys
sys.path.insert(0, "scripts")
from retrieval_utils import compute_metrics, rank_analysis


def test_perfect_retrieval():
    """Identity matrix: each query matches exactly its own corpus item."""
    q = np.eye(4, dtype=np.float32)
    c = np.eye(4, dtype=np.float32)
    gt = {0: 0, 1: 1, 2: 2, 3: 3}
    m = compute_metrics(q, c, gt)
    assert m["MRR"] == pytest.approx(1.0)
    assert m["MAP"] == pytest.approx(1.0)
    assert m["R-Precision"] == pytest.approx(1.0)
    assert m["Recall@1"] == pytest.approx(1.0)
    assert m["n_queries"] == 4


def test_worst_retrieval():
    """Query matches corpus item ranked last (rank 2 out of 3)."""
    q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    c = np.array([
        [0.99, 0.0, 0.0],  # rank 0 — wrong
        [0.98, 0.0, 0.0],  # rank 1 — wrong
        [0.0,  1.0, 0.0],  # rank 2 — correct but least similar
    ], dtype=np.float32)
    gt = {0: 2}
    m = compute_metrics(q, c, gt, ks=(1, 3))
    assert m["Recall@1"] == pytest.approx(0.0)
    assert m["Recall@3"] == pytest.approx(1.0)
    assert m["MRR"] == pytest.approx(1.0 / 3)


def test_recall_at_k_boundary():
    """Correct item at rank k is NOT within top-k; rank k-1 IS."""
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    c = np.array([
        [0.95, 0.0],  # rank 0
        [0.90, 0.0],  # rank 1
        [0.85, 0.0],  # rank 2
        [0.80, 0.0],  # rank 3
        [0.75, 0.0],  # rank 4 — correct
    ], dtype=np.float32)
    gt = {0: 4}
    m = compute_metrics(q, c, gt, ks=(1, 4, 5))
    assert m["Recall@1"] == pytest.approx(0.0)
    assert m["Recall@4"] == pytest.approx(0.0)   # rank 4 not in top-4 (0-3)
    assert m["Recall@5"] == pytest.approx(1.0)   # rank 4 IS in top-5


def test_rank_analysis():
    q = np.eye(3, dtype=np.float32)
    c = np.eye(3, dtype=np.float32)
    gt = {0: 0, 1: 1, 2: 2}
    ranks = rank_analysis(q, c, gt)
    assert list(ranks) == [0, 0, 0]


def test_skips_missing_queries():
    """Queries not in ground_truth are silently skipped."""
    q = np.eye(3, dtype=np.float32)
    c = np.eye(3, dtype=np.float32)
    gt = {0: 0, 2: 2}  # query 1 missing
    m = compute_metrics(q, c, gt)
    assert m["n_queries"] == 2
    assert m["MRR"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests**

```bash
cd /Users/liorlivyatan/LocalProjects/Thesis/work_morables
python -m pytest tests/test_retrieval_utils.py -v
```
Expected: 5 passed.

- [ ] **Step 3: Commit**

```bash
git add scripts/retrieval_utils.py tests/
git commit -m "feat: add retrieval_utils.py with compute_metrics and rank_analysis"
```

---

### Task 4: Update `requirements.txt`

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Replace contents**

```
datasets
sentence-transformers
matplotlib
pandas
numpy
scikit-learn
tqdm
google-generativeai
InstructorEmbedding
einops
torch
transformers
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add InstructorEmbedding, einops, torch, transformers to requirements"
```

---

## Chunk 2: Adapters — all 8 adapter classes

### Task 5: Scaffold `05_model_comparison.py` — base + ST + E5 + E5Instruct adapters

**Files:**
- Create: `scripts/05_model_comparison.py`

- [ ] **Step 1: Create the file**

```python
# scripts/05_model_comparison.py
"""
Multi-model retrieval comparison: moral -> fable (clean corpus).
18 runs across 10 models with instruction variants.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from retrieval_utils import compute_metrics, rank_analysis

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Instruction strings
TASK_GENERAL  = "Given a text, retrieve the most relevant passage that answers the query"
TASK_SPECIFIC = "Given a moral principle or lesson, retrieve the fable that illustrates it"
TART_INSTRUCTION = "retrieve a fable that illustrates the following moral"


def detect_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# =====================================================================
# Adapters — all return L2-normalized float32 numpy arrays (N, D)
# =====================================================================

class BaseAdapter:
    def set_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def encode_queries(self, texts, batch_size=32):
        raise NotImplementedError

    def encode_corpus(self, texts, batch_size=32):
        raise NotImplementedError


class SentenceTransformerAdapter(BaseAdapter):
    """Wraps sentence-transformers. Optional query_prompt prepended to queries only."""
    def __init__(self, model_name, device="cpu", query_prompt=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prompt = query_prompt

    def set_kwargs(self, query_prompt=None):
        self.query_prompt = query_prompt

    def _encode(self, texts, batch_size):
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)

    def encode_queries(self, texts, batch_size=32):
        if self.query_prompt:
            texts = [self.query_prompt + t for t in texts]
        return self._encode(texts, batch_size)

    def encode_corpus(self, texts, batch_size=32):
        return self._encode(texts, batch_size)


class E5Adapter(BaseAdapter):
    """E5 models require 'query: ' / 'passage: ' prefixes."""
    def __init__(self, model_name, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def _encode(self, texts, batch_size):
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)

    def encode_queries(self, texts, batch_size=32):
        return self._encode(["query: " + t for t in texts], batch_size)

    def encode_corpus(self, texts, batch_size=32):
        return self._encode(["passage: " + t for t in texts], batch_size)


class E5InstructAdapter(BaseAdapter):
    """
    multilingual-e5-large-instruct.
    With task: f'Instruct: {task}\\nQuery: {query}'  <- space after colon.
    Without task: plain encoding. Corpus always plain.
    """
    def __init__(self, model_name, device="cpu", task=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.task = task

    def set_kwargs(self, task=None):
        self.task = task

    def _encode(self, texts, batch_size):
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)

    def encode_queries(self, texts, batch_size=32):
        if self.task:
            texts = [f"Instruct: {self.task}\nQuery: {t}" for t in texts]
        return self._encode(texts, batch_size)

    def encode_corpus(self, texts, batch_size=32):
        return self._encode(texts, batch_size)
```

- [ ] **Step 2: Verify file parses**

```bash
python -c "import ast, pathlib; ast.parse(pathlib.Path('scripts/05_model_comparison.py').read_text()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/05_model_comparison.py
git commit -m "feat: scaffold 05_model_comparison with BaseAdapter, SentenceTransformerAdapter, E5Adapter, E5InstructAdapter"
```

---

### Task 6: Add `HFMeanPoolAdapter` and `TARTAdapter`

Append to `scripts/05_model_comparison.py` after `E5InstructAdapter`.

- [ ] **Step 1: Append to the file**

```python
class HFMeanPoolAdapter(BaseAdapter):
    """
    HuggingFace AutoModel with mean pooling + L2 normalization.
    Used by: contriever.
    """
    def __init__(self, model_name, device="cpu"):
        from transformers import AutoTokenizer, AutoModel
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.train(False)  # set inference mode (equivalent to model.eval())

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def _encode(self, texts, batch_size):
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="encoding"):
            batch = texts[i: i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**encoded)
            embs = self._mean_pool(out.last_hidden_state, encoded["attention_mask"])
            embs = F.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)

    def encode_queries(self, texts, batch_size=32):
        return self._encode(texts, batch_size)

    def encode_corpus(self, texts, batch_size=32):
        return self._encode(texts, batch_size)


class TARTAdapter(HFMeanPoolAdapter):
    """
    TART uses '[SEP]' as delimiter between instruction and query.
    instruction=None: plain encoding (no [SEP]).
    """
    def __init__(self, model_name, device="cpu", instruction=None):
        super().__init__(model_name, device)
        self.instruction = instruction

    def set_kwargs(self, instruction=None):
        self.instruction = instruction

    def encode_queries(self, texts, batch_size=32):
        if self.instruction:
            texts = [f"{self.instruction} [SEP] {t}" for t in texts]
        return self._encode(texts, batch_size)
```

- [ ] **Step 2: Commit**

```bash
git add scripts/05_model_comparison.py
git commit -m "feat: add HFMeanPoolAdapter and TARTAdapter"
```

---

### Task 7: Add `InstructorAdapter`, `JinaAdapter`, `QwenAdapter`

Append to `scripts/05_model_comparison.py` after `TARTAdapter`.

- [ ] **Step 1: Append to the file**

```python
class InstructorAdapter(BaseAdapter):
    """
    hkunlp/instructor-base — list-of-pairs API: [[instruction, text], ...]
    Empty string instruction behaves as a plain encoder.
    """
    def __init__(self, model_name, device="cpu", query_instr="", corpus_instr=""):
        from InstructorEmbedding import INSTRUCTOR
        self.model = INSTRUCTOR(model_name)
        self.query_instr = query_instr
        self.corpus_instr = corpus_instr

    def set_kwargs(self, query_instr="", corpus_instr=""):
        self.query_instr = query_instr
        self.corpus_instr = corpus_instr

    def _encode_pairs(self, instr, texts, batch_size):
        pairs = [[instr, t] for t in texts]
        embs = self.model.encode(pairs, batch_size=batch_size, show_progress_bar=True)
        embs = embs.astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-9)

    def encode_queries(self, texts, batch_size=32):
        return self._encode_pairs(self.query_instr, texts, batch_size)

    def encode_corpus(self, texts, batch_size=32):
        return self._encode_pairs(self.corpus_instr, texts, batch_size)


class JinaAdapter(BaseAdapter):
    """
    jinaai/jina-embeddings-v3.
    use_task=True:  task='retrieval.query' / 'retrieval.passage'
    use_task=False: no task kwarg
    """
    def __init__(self, model_name, device="cpu", use_task=True):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.use_task = use_task

    def set_kwargs(self, use_task=True):
        self.use_task = use_task

    def _encode(self, texts, batch_size, task=None):
        kwargs = dict(
            batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        )
        if task is not None:
            kwargs["task"] = task
        return self.model.encode(texts, **kwargs).astype(np.float32)

    def encode_queries(self, texts, batch_size=32):
        return self._encode(texts, batch_size, task="retrieval.query" if self.use_task else None)

    def encode_corpus(self, texts, batch_size=32):
        return self._encode(texts, batch_size, task="retrieval.passage" if self.use_task else None)


class QwenAdapter(BaseAdapter):
    """
    Qwen3-Embedding — causal LLM, requires last-token pooling (NOT mean pool).
    With task: f'Instruct: {task}\\nQuery:{query}'  <- NO space before query text.
    """
    def __init__(self, model_name, device="cpu", task=None):
        from transformers import AutoTokenizer, AutoModel
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.train(False)  # inference mode
        self.task = task

    def set_kwargs(self, task=None):
        self.task = task

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

    def _encode(self, texts, batch_size):
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="encoding"):
            batch = texts[i: i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**encoded)
            embs = self._last_token_pool(out.last_hidden_state, encoded["attention_mask"])
            embs = F.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)

    def encode_queries(self, texts, batch_size=16):
        if self.task:
            # NOTE: no space between "Query:" and query text — Qwen3 specific
            texts = [f"Instruct: {self.task}\nQuery:{t}" for t in texts]
        return self._encode(texts, batch_size)

    def encode_corpus(self, texts, batch_size=16):
        return self._encode(texts, batch_size)
```

- [ ] **Step 2: Commit**

```bash
git add scripts/05_model_comparison.py
git commit -m "feat: add InstructorAdapter, JinaAdapter, QwenAdapter"
```

---

### Task 8: Add MODEL_REGISTRY

Append to `scripts/05_model_comparison.py` after `QwenAdapter`.

- [ ] **Step 1: Append the registry**

```python
# =====================================================================
# Registry — (run_key, model_name, adapter_cls, adapter_kwargs, batch_size)
# Variants of the same model_name are grouped so the model loads once.
# =====================================================================
MODEL_REGISTRY = [
    ("contriever__plain",
     "facebook/contriever", HFMeanPoolAdapter, {}, 64),

    ("tart__no_instr",
     "orionweller/tart-dual-contriever-msmarco", TARTAdapter, {"instruction": None}, 64),
    ("tart__specific",
     "orionweller/tart-dual-contriever-msmarco", TARTAdapter, {"instruction": TART_INSTRUCTION}, 64),

    ("instructor__no_instr",
     "hkunlp/instructor-base", InstructorAdapter, {"query_instr": "", "corpus_instr": ""}, 32),
    ("instructor__general",
     "hkunlp/instructor-base", InstructorAdapter,
     {"query_instr": "Represent the text for retrieval:",
      "corpus_instr": "Represent the passage:"}, 32),
    ("instructor__specific",
     "hkunlp/instructor-base", InstructorAdapter,
     {"query_instr": "Represent the moral for retrieving a fable:",
      "corpus_instr": "Represent the fable:"}, 32),

    ("bge-base__plain",
     "BAAI/bge-base-en-v1.5", SentenceTransformerAdapter,
     {"query_prompt": "Represent this sentence for searching relevant passages: "}, 64),

    ("e5-base__plain",
     "intfloat/e5-base-v2", E5Adapter, {}, 64),

    ("multilingual-e5__plain",
     "intfloat/multilingual-e5-large", E5Adapter, {}, 32),

    ("multilingual-e5-instruct__no_instr",
     "intfloat/multilingual-e5-large-instruct", E5InstructAdapter, {"task": None}, 32),
    ("multilingual-e5-instruct__general",
     "intfloat/multilingual-e5-large-instruct", E5InstructAdapter, {"task": TASK_GENERAL}, 32),
    ("multilingual-e5-instruct__specific",
     "intfloat/multilingual-e5-large-instruct", E5InstructAdapter, {"task": TASK_SPECIFIC}, 32),

    ("bge-m3__plain",
     "BAAI/bge-m3", SentenceTransformerAdapter, {}, 32),

    ("jina-v3__no_instr",
     "jinaai/jina-embeddings-v3", JinaAdapter, {"use_task": False}, 32),
    ("jina-v3__retrieval",
     "jinaai/jina-embeddings-v3", JinaAdapter, {"use_task": True}, 32),

    ("qwen3__no_instr",
     "Qwen/Qwen3-Embedding-0.6B", QwenAdapter, {"task": None}, 16),
    ("qwen3__general",
     "Qwen/Qwen3-Embedding-0.6B", QwenAdapter, {"task": TASK_GENERAL}, 16),
    ("qwen3__specific",
     "Qwen/Qwen3-Embedding-0.6B", QwenAdapter, {"task": TASK_SPECIFIC}, 16),
]
```

- [ ] **Step 2: Verify registry has 18 entries**

```bash
python -c "
import ast, pathlib
tree = ast.parse(pathlib.Path('scripts/05_model_comparison.py').read_text())
print('Parse OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/05_model_comparison.py
git commit -m "feat: add MODEL_REGISTRY with 18 runs across 10 models"
```

---

## Chunk 3: Main loop, data loading, and execution

### Task 9: Add data loading and main loop

Append to `scripts/05_model_comparison.py`.

- [ ] **Step 1: Append the main block**

```python
# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    # Load data
    with open(DATA_DIR / "fables_corpus.json") as f:
        fables_corpus = json.load(f)
    with open(DATA_DIR / "morals_corpus.json") as f:
        morals_corpus = json.load(f)
    with open(DATA_DIR / "qrels_fable_to_moral.json") as f:
        qrels_f2m = json.load(f)

    fable_texts = [fb["text"] for fb in fables_corpus]
    moral_texts = [m["text"]  for m in morals_corpus]

    # Ground truth: file is fable_idx->moral_idx; invert for moral->fable task
    gt_f2m = {}
    for qrel in qrels_f2m:
        fable_idx = int(qrel["query_id"].split("_")[1])
        moral_idx = int(qrel["doc_id"].split("_")[1])
        gt_f2m[fable_idx] = moral_idx
    gt_m2f = {v: k for k, v in gt_f2m.items()}

    print(f"Queries (morals): {len(moral_texts)}  |  Corpus (fables): {len(fable_texts)}")
    print(f"Ground truth entries: {len(gt_m2f)}")

    device = detect_device()
    print(f"Device: {device}\n")

    all_results = []
    loaded_models = {}  # model_name -> adapter (reused across variants)

    for i, (run_key, model_name, adapter_cls, kwargs, batch_size) in enumerate(MODEL_REGISTRY):
        print(f"\n{'='*70}")
        print(f"Run {i+1}/{len(MODEL_REGISTRY)}: {run_key}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        if model_name not in loaded_models:
            print(f"Loading {model_name}...")
            try:
                loaded_models[model_name] = adapter_cls(model_name, device=device)
            except Exception as e:
                if device == "mps":
                    warnings.warn(f"MPS load failed: {e}. Retrying on CPU.")
                    loaded_models[model_name] = adapter_cls(model_name, device="cpu")
                else:
                    raise

        adapter = loaded_models[model_name]
        adapter.set_kwargs(**kwargs)

        t0 = time.time()
        try:
            print("Encoding queries (morals)...")
            query_embs = adapter.encode_queries(moral_texts, batch_size=batch_size)
            print("Encoding corpus (fables)...")
            corpus_embs = adapter.encode_corpus(fable_texts, batch_size=batch_size)
        except Exception as e:
            if device == "mps":
                warnings.warn(f"MPS encode failed for {run_key}: {e}. Retrying on CPU.")
                loaded_models[model_name] = adapter_cls(model_name, device="cpu")
                adapter = loaded_models[model_name]
                adapter.set_kwargs(**kwargs)
                query_embs = adapter.encode_queries(moral_texts, batch_size=batch_size)
                corpus_embs = adapter.encode_corpus(fable_texts, batch_size=batch_size)
            else:
                raise
        elapsed = time.time() - t0

        metrics = compute_metrics(query_embs, corpus_embs, gt_m2f, ks=(1, 5, 10, 50))
        metrics["run_key"] = run_key
        metrics["model"] = model_name
        metrics["encoding_time_s"] = round(elapsed, 2)
        all_results.append(metrics)

        print(f"  MRR:       {metrics['MRR']:.4f}")
        print(f"  MAP:       {metrics['MAP']:.4f}")
        print(f"  Recall@1:  {metrics['Recall@1']:.4f}")
        print(f"  Recall@5:  {metrics['Recall@5']:.4f}")
        print(f"  Recall@10: {metrics['Recall@10']:.4f}")
        print(f"  Recall@50: {metrics['Recall@50']:.4f}")
        print(f"  Time:      {elapsed:.1f}s")

        # Save incrementally so results survive a crash mid-run
        out_path = RESULTS_DIR / "model_comparison_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # Evict model when all its variants are done
        next_model = MODEL_REGISTRY[i + 1][1] if i + 1 < len(MODEL_REGISTRY) else None
        if next_model != model_name:
            del loaded_models[model_name]
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

    # Summary table sorted by MRR
    sorted_results = sorted(all_results, key=lambda r: r["MRR"], reverse=True)
    print(f"\n{'='*95}")
    print("SUMMARY — sorted by MRR")
    print(f"{'='*95}")
    hdr = f"{'Run Key':<47} {'MRR':>6} {'MAP':>6} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@50':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted_results:
        print(
            f"{r['run_key']:<47} "
            f"{r['MRR']:.4f} {r['MAP']:.4f} "
            f"{r['Recall@1']:.4f} {r['Recall@5']:.4f} "
            f"{r['Recall@10']:.4f} {r['Recall@50']:.4f}"
        )
    print(f"\nResults saved to {out_path}")
```

- [ ] **Step 2: Verify full file parses**

```bash
python -c "import ast, pathlib; ast.parse(pathlib.Path('scripts/05_model_comparison.py').read_text()); print('Parse OK')"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/05_model_comparison.py
git commit -m "feat: add data loading and main loop to 05_model_comparison"
```

---

### Task 10: Install dependencies and run

- [ ] **Step 1: Install dependencies**

```bash
pip install InstructorEmbedding einops
pip install -U sentence-transformers transformers torch
```

- [ ] **Step 2: Verify data files exist**

```bash
python -c "
import json
from pathlib import Path
d = Path('data/processed')
for f in ['fables_corpus.json', 'morals_corpus.json', 'qrels_fable_to_moral.json']:
    data = json.load(open(d / f))
    print(f'{f}: {len(data)} entries')
"
```
Expected: each file has 709 entries.

- [ ] **Step 3: Run all tests one final time**

```bash
python -m pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 4: Run the full experiment**

```bash
python scripts/05_model_comparison.py 2>&1 | tee results/model_comparison_log.txt
```

Results are saved incrementally — if a model crashes, results so far are preserved in `results/model_comparison_results.json`.

- [ ] **Step 5: Verify output**

```bash
python -c "
import json
results = json.load(open('results/model_comparison_results.json'))
print(f'Total runs: {len(results)}')
for r in sorted(results, key=lambda x: x['MRR'], reverse=True):
    print(f\"  {r['run_key']:<47} MRR={r['MRR']:.4f}\")
"
```
Expected: 18 runs, all with valid MRR values.

- [ ] **Step 6: Commit results**

```bash
git add results/model_comparison_results.json results/model_comparison_log.txt
git commit -m "results: add model comparison results (18 runs, 10 models)"
```
