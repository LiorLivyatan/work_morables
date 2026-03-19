# scripts/05_model_comparison.py
"""
Multi-model retrieval comparison: moral -> fable (clean corpus).

Usage:
  python scripts/05_model_comparison.py                     # run all models
  python scripts/05_model_comparison.py --models qwen3      # run all qwen3 variants
  python scripts/05_model_comparison.py --models gritlm bge-gemma2
  python scripts/05_model_comparison.py --run-keys contriever__plain tart__specific

Each invocation creates a timestamped directory under results/runs/:
  results/runs/2026-03-14_15-30-00/
    results.json      ← metrics for every completed run (saved incrementally)
    metadata.json     ← run config: timestamp, device, CLI args, active run keys
    predictions/
      contriever__plain.json
      ...             ← full corpus ranking + correct_rank per query
"""
import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from retrieval_utils import compute_metrics, compute_rankings, rank_analysis

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Add model size column, add instruction/not-instruction column
# TODO: fix 709 -> 678
# TODO: Check the correct morals/fables of ones that returned currect index per query

# ── Instruction strings ────────────────────────────────────────────────────────
TASK_GENERAL  = "Given a text, retrieve the most relevant passage that answers the query"
TASK_SPECIFIC = "Given a moral principle or lesson, retrieve the fable that illustrates it"
TART_INSTRUCTION = "Retrieve a fable that illustrates the following moral"


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


class PromptNameAdapter(BaseAdapter):
    """
    Models using SentenceTransformer prompt_name= API for queries (e.g. drama-1b, Stella).
    query_prompt_name=None: plain encoding.
    query_prompt_name='s2p_query': passes prompt_name to encode().
    """
    def __init__(self, model_name, device="cpu", query_prompt_name=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.query_prompt_name = query_prompt_name

    def set_kwargs(self, query_prompt_name=None):
        self.query_prompt_name = query_prompt_name

    def _encode(self, texts, batch_size, prompt_name=None):
        kwargs = dict(
            batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        )
        if prompt_name is not None:
            kwargs["prompt_name"] = prompt_name
        return self.model.encode(texts, **kwargs).astype(np.float32)

    def encode_queries(self, texts, batch_size=32):
        return self._encode(texts, batch_size, prompt_name=self.query_prompt_name)

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
    multilingual-e5-large-instruct, e5-mistral-7b-instruct, Linq-Embed-Mistral,
    SFR-Embedding-Mistral, gte-Qwen2-*-instruct, llama-embed-nemotron-8b.
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
        self.model.train(False)

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


class InstructorAdapter(BaseAdapter):
    """
    hkunlp/instructor-base and instructor-xl -- list-of-pairs API: [[instruction, text], ...]
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
    Qwen3-Embedding and lychee-embed -- causal LLM, requires last-token pooling (NOT mean pool).
    With task: f'Instruct: {task}\\nQuery:{query}'  <- NO space before query text.
    Without task: plain encoding.
    """
    def __init__(self, model_name, device="cpu", task=None):
        from transformers import AutoTokenizer, AutoModel
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.train(False)
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
            # NOTE: no space between "Query:" and query text — Qwen3/lychee specific
            texts = [f"Instruct: {self.task}\nQuery:{t}" for t in texts]
        return self._encode(texts, batch_size)

    def encode_corpus(self, texts, batch_size=16):
        return self._encode(texts, batch_size)


class NomicAdapter(BaseAdapter):
    """
    nomic-embed-text-v2-moe — uses task-prefix approach.
    use_prefixes=True:  'search_query: ' / 'search_document: ' prefixes.
    use_prefixes=False: plain encoding (no prefix).
    """
    def __init__(self, model_name, device="cpu", use_prefixes=True):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.use_prefixes = use_prefixes

    def set_kwargs(self, use_prefixes=True):
        self.use_prefixes = use_prefixes

    def _encode(self, texts, batch_size):
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)

    def encode_queries(self, texts, batch_size=32):
        if self.use_prefixes:
            texts = ["search_query: " + t for t in texts]
        return self._encode(texts, batch_size)

    def encode_corpus(self, texts, batch_size=32):
        if self.use_prefixes:
            texts = ["search_document: " + t for t in texts]
        return self._encode(texts, batch_size)


class GritLMAdapter(BaseAdapter):
    """
    GritLM-7B — uses gritlm library.
    Queries with instruction: '<|user|>\\n{instruction}\\n<|embed|>\\n'
    Queries without / corpus: '<|embed|>\\n'
    Requires: pip install gritlm
    """
    def __init__(self, model_name, device="cpu", instruction=None):
        from gritlm import GritLM
        # GritLM manages device internally; device kwarg accepted but not forwarded
        self.model = GritLM(model_name, torch_dtype=torch.bfloat16)
        self.instruction = instruction

    def set_kwargs(self, instruction=None):
        self.instruction = instruction

    @staticmethod
    def _wrap(instruction):
        return f"<|user|>\n{instruction}\n<|embed|>\n" if instruction else "<|embed|>\n"

    @staticmethod
    def _normalize(embs):
        embs = embs.astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-9)

    def encode_queries(self, texts, batch_size=8):
        embs = self.model.encode(
            texts, instruction=self._wrap(self.instruction), batch_size=batch_size
        )
        return self._normalize(embs)

    def encode_corpus(self, texts, batch_size=8):
        embs = self.model.encode(texts, instruction=self._wrap(""), batch_size=batch_size)
        return self._normalize(embs)


class BGEGemmaAdapter(BaseAdapter):
    """
    BAAI/bge-multilingual-gemma2 — Gemma-2 based.
    With task: f'<instruct>{task}\\n<query>{query}'
    Without task: plain query text.
    Corpus always plain.
    """
    def __init__(self, model_name, device="cpu", task=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.task = task

    def set_kwargs(self, task=None):
        self.task = task

    def _encode(self, texts, batch_size):
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)

    def encode_queries(self, texts, batch_size=8):
        if self.task:
            texts = [f"<instruct>{self.task}\n<query>{t}" for t in texts]
        return self._encode(texts, batch_size)

    def encode_corpus(self, texts, batch_size=8):
        return self._encode(texts, batch_size)


class NVEmbedAdapter(BaseAdapter):
    """
    nvidia/NV-Embed-v2 — Mistral-7B based, requires trust_remote_code.
    Uses Instruct: {task}\\nQuery: {query} format for queries (same as E5Instruct).
    Corpus always plain.
    """
    def __init__(self, model_name, device="cpu", task=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.task = task

    def set_kwargs(self, task=None):
        self.task = task

    def _encode(self, texts, batch_size):
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)

    def encode_queries(self, texts, batch_size=4):
        if self.task:
            texts = [f"Instruct: {self.task}\nQuery: {t}" for t in texts]
        return self._encode(texts, batch_size)

    def encode_corpus(self, texts, batch_size=4):
        return self._encode(texts, batch_size)


# =====================================================================
# Registry
# (run_key, model_name, adapter_cls, adapter_kwargs, batch_size)
#
# Variants of the same model_name are grouped so the model loads once.
# batch_size is used for both encode_queries and encode_corpus calls.
# =====================================================================
MODEL_REGISTRY = [
    # ── Original models (10 models, 18 runs) ──────────────────────────────────
    ("contriever__plain",
     "facebook/contriever",                          HFMeanPoolAdapter,      {},                                                               64),

    ("tart__no_instr",
     "orionweller/tart-dual-contriever-msmarco",     TARTAdapter,            {"instruction": None},                                            64),
    ("tart__specific",
     "orionweller/tart-dual-contriever-msmarco",     TARTAdapter,            {"instruction": TART_INSTRUCTION},                                64),

    ("instructor-base__no_instr",
     "hkunlp/instructor-base",                       InstructorAdapter,      {"query_instr": "", "corpus_instr": ""},                          32),
    ("instructor-base__general",
     "hkunlp/instructor-base",                       InstructorAdapter,      {"query_instr": "Represent the text for retrieval:",
                                                                               "corpus_instr": "Represent the passage:"},                       32),
    ("instructor-base__specific",
     "hkunlp/instructor-base",                       InstructorAdapter,      {"query_instr": "Represent the moral for retrieving a fable:",
                                                                               "corpus_instr": "Represent the fable:"},                         32),

    ("bge-base__plain",
     "BAAI/bge-base-en-v1.5",                        SentenceTransformerAdapter, {"query_prompt": "Represent this sentence for searching relevant passages: "}, 64),

    ("e5-base__plain",
     "intfloat/e5-base-v2",                          E5Adapter,              {},                                                               64),

    ("multilingual-e5__plain",
     "intfloat/multilingual-e5-large",               E5Adapter,              {},                                                               32),

    ("multilingual-e5-instruct__no_instr",
     "intfloat/multilingual-e5-large-instruct",      E5InstructAdapter,      {"task": None},                                                   32),
    ("multilingual-e5-instruct__general",
     "intfloat/multilingual-e5-large-instruct",      E5InstructAdapter,      {"task": TASK_GENERAL},                                           32),
    ("multilingual-e5-instruct__specific",
     "intfloat/multilingual-e5-large-instruct",      E5InstructAdapter,      {"task": TASK_SPECIFIC},                                          32),

    ("bge-m3__plain",
     "BAAI/bge-m3",                                  SentenceTransformerAdapter, {},                                                           32),

    ("jina-v3__no_instr",
     "jinaai/jina-embeddings-v3",                    JinaAdapter,            {"use_task": False},                                              32),
    ("jina-v3__retrieval",
     "jinaai/jina-embeddings-v3",                    JinaAdapter,            {"use_task": True},                                               32),

    ("qwen3-0.6b__no_instr",
     "Qwen/Qwen3-Embedding-0.6B",                    QwenAdapter,            {"task": None},                                                   16),
    ("qwen3-0.6b__general",
     "Qwen/Qwen3-Embedding-0.6B",                    QwenAdapter,            {"task": TASK_GENERAL},                                           16),
    ("qwen3-0.6b__specific",
     "Qwen/Qwen3-Embedding-0.6B",                    QwenAdapter,            {"task": TASK_SPECIFIC},                                          16),

    # ── New models ─────────────────────────────────────────────────────────────
    # drama-1b: Facebook's dense retrieval model
    ("drama__no_instr",
     "facebook/drama-1b",                            PromptNameAdapter,      {"query_prompt_name": None},                                      32),
    ("drama__prompted",
     "facebook/drama-1b",                            PromptNameAdapter,      {"query_prompt_name": "query"},                                   32),

    # instructor-xl: larger instructor model
    ("instructor-xl__no_instr",
     "hkunlp/instructor-xl",                         InstructorAdapter,      {"query_instr": "", "corpus_instr": ""},                          16),
    ("instructor-xl__general",
     "hkunlp/instructor-xl",                         InstructorAdapter,      {"query_instr": "Represent the text for retrieval:",
                                                                               "corpus_instr": "Represent the passage:"},                       16),
    ("instructor-xl__specific",
     "hkunlp/instructor-xl",                         InstructorAdapter,      {"query_instr": "Represent the moral for retrieving a fable:",
                                                                               "corpus_instr": "Represent the fable:"},                         16),

    # Stella 1.5B: uses prompt_name='s2p_query' for query encoding
    ("stella-1.5b__no_instr",
     "NovaSearch/stella_en_1.5B_v5",                 PromptNameAdapter,      {"query_prompt_name": None},                                      32),
    ("stella-1.5b__prompted",
     "NovaSearch/stella_en_1.5B_v5",                 PromptNameAdapter,      {"query_prompt_name": "s2p_query"},                               32),

    # GTE-Qwen2 1.5B: uses Instruct:/Query: format (E5-instruct style)
    ("gte-qwen2-1.5b__no_instr",
     "Alibaba-NLP/gte-Qwen2-1.5B-instruct",         E5InstructAdapter,      {"task": None},                                                   32),
    ("gte-qwen2-1.5b__general",
     "Alibaba-NLP/gte-Qwen2-1.5B-instruct",         E5InstructAdapter,      {"task": TASK_GENERAL},                                           32),
    ("gte-qwen2-1.5b__specific",
     "Alibaba-NLP/gte-Qwen2-1.5B-instruct",         E5InstructAdapter,      {"task": TASK_SPECIFIC},                                          32),

    # lychee-embed: Qwen3-based, uses Instruct:/Query: (no space) format
    ("lychee__no_instr",
     "vec-ai/lychee-embed",                          QwenAdapter,            {"task": None},                                                   16),
    ("lychee__general",
     "vec-ai/lychee-embed",                          QwenAdapter,            {"task": TASK_GENERAL},                                           16),
    ("lychee__specific",
     "vec-ai/lychee-embed",                          QwenAdapter,            {"task": TASK_SPECIFIC},                                          16),

    # Qwen3-Embedding 4B
    ("qwen3-4b__no_instr",
     "Qwen/Qwen3-Embedding-4B",                      QwenAdapter,            {"task": None},                                                    8),
    ("qwen3-4b__general",
     "Qwen/Qwen3-Embedding-4B",                      QwenAdapter,            {"task": TASK_GENERAL},                                            8),
    ("qwen3-4b__specific",
     "Qwen/Qwen3-Embedding-4B",                      QwenAdapter,            {"task": TASK_SPECIFIC},                                           8),

    # nomic-embed-text-v2-moe: MoE architecture, uses search_query:/search_document: prefixes
    ("nomic-v2-moe__no_instr",
     "nomic-ai/nomic-embed-text-v2-moe",             NomicAdapter,           {"use_prefixes": False},                                          32),
    ("nomic-v2-moe__prefixed",
     "nomic-ai/nomic-embed-text-v2-moe",             NomicAdapter,           {"use_prefixes": True},                                           32),

    # GritLM-7B: generative + embedding model, uses <|user|>/{instr}/<|embed|> tokens
    ("gritlm-7b__no_instr",
     "GritLM/GritLM-7B",                             GritLMAdapter,          {"instruction": None},                                             8),
    ("gritlm-7b__specific",
     "GritLM/GritLM-7B",                             GritLMAdapter,          {"instruction": TASK_SPECIFIC},                                    8),

    # E5-Mistral 7B instruct
    ("e5-mistral-7b__no_instr",
     "intfloat/e5-mistral-7b-instruct",              E5InstructAdapter,      {"task": None},                                                    8),
    ("e5-mistral-7b__general",
     "intfloat/e5-mistral-7b-instruct",              E5InstructAdapter,      {"task": TASK_GENERAL},                                            8),
    ("e5-mistral-7b__specific",
     "intfloat/e5-mistral-7b-instruct",              E5InstructAdapter,      {"task": TASK_SPECIFIC},                                           8),

    # Linq-Embed-Mistral (Mistral-7B fine-tuned for retrieval)
    ("linq-embed-mistral__no_instr",
     "Linq-AI-Research/Linq-Embed-Mistral",          E5InstructAdapter,      {"task": None},                                                    8),
    ("linq-embed-mistral__general",
     "Linq-AI-Research/Linq-Embed-Mistral",          E5InstructAdapter,      {"task": TASK_GENERAL},                                            8),
    ("linq-embed-mistral__specific",
     "Linq-AI-Research/Linq-Embed-Mistral",          E5InstructAdapter,      {"task": TASK_SPECIFIC},                                           8),

    # SFR-Embedding-Mistral (Salesforce, Mistral-7B fine-tuned)
    ("sfr-mistral__no_instr",
     "Salesforce/SFR-Embedding-Mistral",             E5InstructAdapter,      {"task": None},                                                    8),
    ("sfr-mistral__general",
     "Salesforce/SFR-Embedding-Mistral",             E5InstructAdapter,      {"task": TASK_GENERAL},                                            8),
    ("sfr-mistral__specific",
     "Salesforce/SFR-Embedding-Mistral",             E5InstructAdapter,      {"task": TASK_SPECIFIC},                                           8),

    # GTE-Qwen2 7B instruct
    ("gte-qwen2-7b__no_instr",
     "Alibaba-NLP/gte-Qwen2-7B-instruct",            E5InstructAdapter,      {"task": None},                                                    8),
    ("gte-qwen2-7b__general",
     "Alibaba-NLP/gte-Qwen2-7B-instruct",            E5InstructAdapter,      {"task": TASK_GENERAL},                                            8),
    ("gte-qwen2-7b__specific",
     "Alibaba-NLP/gte-Qwen2-7B-instruct",            E5InstructAdapter,      {"task": TASK_SPECIFIC},                                           8),

    # llama-embed-nemotron-8b (NVIDIA, Llama-3.1-8B based)
    ("nemotron-8b__no_instr",
     "nvidia/llama-embed-nemotron-8b",               E5InstructAdapter,      {"task": None},                                                    4),
    ("nemotron-8b__general",
     "nvidia/llama-embed-nemotron-8b",               E5InstructAdapter,      {"task": TASK_GENERAL},                                            4),
    ("nemotron-8b__specific",
     "nvidia/llama-embed-nemotron-8b",               E5InstructAdapter,      {"task": TASK_SPECIFIC},                                           4),

    # bge-multilingual-gemma2 (Gemma-2 based, uses <instruct>/<query> tags)
    ("bge-gemma2__no_instr",
     "BAAI/bge-multilingual-gemma2",                 BGEGemmaAdapter,        {"task": None},                                                    8),
    ("bge-gemma2__general",
     "BAAI/bge-multilingual-gemma2",                 BGEGemmaAdapter,        {"task": TASK_GENERAL},                                            8),
    ("bge-gemma2__specific",
     "BAAI/bge-multilingual-gemma2",                 BGEGemmaAdapter,        {"task": TASK_SPECIFIC},                                           8),

    # NV-Embed-v2 (NVIDIA, Mistral-7B based, requires trust_remote_code)
    ("nv-embed-v2__no_instr",
     "nvidia/NV-Embed-v2",                           NVEmbedAdapter,         {"task": None},                                                    4),
    ("nv-embed-v2__general",
     "nvidia/NV-Embed-v2",                           NVEmbedAdapter,         {"task": TASK_GENERAL},                                            4),
    ("nv-embed-v2__specific",
     "nvidia/NV-Embed-v2",                           NVEmbedAdapter,         {"task": TASK_SPECIFIC},                                           4),
]


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-model retrieval comparison (moral -> fable).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/05_model_comparison.py
  python scripts/05_model_comparison.py --models qwen3 instructor
  python scripts/05_model_comparison.py --run-keys contriever__plain bge-m3__plain
        """,
    )
    parser.add_argument(
        "--models", nargs="+", metavar="PATTERN",
        help="Run only entries whose run_key contains any of these substrings.",
    )
    parser.add_argument(
        "--run-keys", nargs="+", metavar="KEY",
        help="Run only these exact run keys.",
    )
    args = parser.parse_args()

    # Build active registry based on CLI filters
    active_registry = MODEL_REGISTRY
    if args.run_keys:
        key_set = set(args.run_keys)
        active_registry = [r for r in MODEL_REGISTRY if r[0] in key_set]
        unknown = key_set - {r[0] for r in MODEL_REGISTRY}
        if unknown:
            print(f"WARNING: unknown run keys: {unknown}")
    elif args.models:
        patterns = args.models
        active_registry = [r for r in MODEL_REGISTRY if any(p in r[0] for p in patterns)]

    if not active_registry:
        print("No matching entries in MODEL_REGISTRY. Check --models / --run-keys.")
        sys.exit(1)

    print(f"Active runs: {len(active_registry)}")

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

    # ── Create timestamped run directory ─────────────────────────────────────
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RESULTS_DIR / "runs" / run_ts
    preds_dir = run_dir / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(exist_ok=True)
    out_path = run_dir / "results.json"
    print(f"Run directory: {run_dir}\n")

    # ── Write metadata ────────────────────────────────────────────────────────
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "cli_args": {
            "models": args.models,
            "run_keys": args.run_keys,
        },
        "n_queries": len(moral_texts),
        "corpus_size": len(fable_texts),
        "n_runs": len(active_registry),
        "run_keys": [r[0] for r in active_registry],
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    all_results = []
    loaded_models = {}     # model_name -> adapter (reused across variants)
    skipped_models = set() # model names that failed to load

    def _is_mps_error(e):
        """True only for errors genuinely caused by the MPS/Metal device."""
        mps_signals = ("accelerator", "mps", "metal", "device")
        return any(s in str(e).lower() for s in mps_signals)

    for i, (run_key, model_name, adapter_cls, kwargs, batch_size) in enumerate(active_registry):
        print(f"\n{'='*70}")
        print(f"Run {i+1}/{len(active_registry)}: {run_key}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        if model_name in skipped_models:
            print(f"  SKIPPED (model failed to load earlier)")
            all_results.append({"run_key": run_key, "model": model_name, "error": "model failed to load"})
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            continue

        if model_name not in loaded_models:
            print(f"Loading {model_name}...")
            try:
                loaded_models[model_name] = adapter_cls(model_name, device=device)
            except Exception as e:
                if device == "mps" and _is_mps_error(e):
                    warnings.warn(f"MPS load failed: {e}. Retrying on CPU.")
                    loaded_models[model_name] = adapter_cls(model_name, device="cpu")
                else:
                    warnings.warn(f"Skipping {model_name}: {e}")
                    skipped_models.add(model_name)
                    all_results.append({"run_key": run_key, "model": model_name, "error": str(e)})
                    with open(out_path, "w") as f:
                        json.dump(all_results, f, indent=2)
                    continue

        adapter = loaded_models[model_name]
        adapter.set_kwargs(**kwargs)

        t0 = time.time()
        try:
            print("Encoding queries (morals)...")
            query_embs = adapter.encode_queries(moral_texts, batch_size=batch_size)
            print("Encoding corpus (fables)...")
            corpus_embs = adapter.encode_corpus(fable_texts, batch_size=batch_size)
        except Exception as e:
            if device == "mps" and _is_mps_error(e):
                warnings.warn(f"MPS encode failed for {run_key}: {e}. Retrying on CPU.")
                loaded_models[model_name] = adapter_cls(model_name, device="cpu")
                adapter = loaded_models[model_name]
                adapter.set_kwargs(**kwargs)
                query_embs = adapter.encode_queries(moral_texts, batch_size=batch_size)
                corpus_embs = adapter.encode_corpus(fable_texts, batch_size=batch_size)
            else:
                raise
        elapsed = time.time() - t0

        # ── Metrics ──────────────────────────────────────────────────────────
        metrics = compute_metrics(query_embs, corpus_embs, gt_m2f, ks=(1, 5, 10, 50))
        metrics["run_key"] = run_key
        metrics["model"] = model_name
        metrics["encoding_time_s"] = round(elapsed, 2)
        all_results.append(metrics)

        print(f"  MRR:         {metrics['MRR']:.4f}")
        print(f"  NDCG@10:     {metrics['NDCG@10']:.4f}")
        print(f"  Mean Rank:   {metrics['Mean Rank']:.1f}")
        print(f"  Median Rank: {metrics['Median Rank']:.1f}")
        print(f"  Recall@1:    {metrics['Recall@1']:.4f}")
        print(f"  Recall@5:    {metrics['Recall@5']:.4f}")
        print(f"  Recall@10:   {metrics['Recall@10']:.4f}")
        print(f"  Recall@50:   {metrics['Recall@50']:.4f}")
        print(f"  P@1:         {metrics['P@1']:.4f}")
        print(f"  P@10:        {metrics['P@10']:.4f}")
        print(f"  Time:        {elapsed:.1f}s")

        # ── Save predictions (full ranking per query + absolute rank) ────────
        rankings_data = compute_rankings(query_embs, corpus_embs, top_k=len(fable_texts))
        # rank_analysis returns 0-indexed ranks in ascending q_idx order (for queries in GT)
        ranks_arr = rank_analysis(query_embs, corpus_embs, gt_m2f)
        gt_sorted_qidx = sorted(gt_m2f.keys())
        rank_by_qidx = {q: int(r) + 1 for q, r in zip(gt_sorted_qidx, ranks_arr)}  # 1-indexed

        pred_records = []
        for q_idx, ranking in enumerate(rankings_data):
            correct_idx = gt_m2f.get(q_idx)
            pred_records.append({
                "query_idx": q_idx,
                "correct_idx": correct_idx,
                "correct_rank": rank_by_qidx.get(q_idx),  # 1-indexed absolute rank in full corpus
                "top_k_indices": ranking["indices"],
                "top_k_scores": ranking["scores"],
            })
        with open(preds_dir / f"{run_key}.json", "w") as f:
            json.dump({"run_key": run_key, "model": model_name, "top_k": len(fable_texts), "queries": pred_records}, f)

        # ── Incremental save ─────────────────────────────────────────────────
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # ── Evict model when all its variants in active_registry are done ────
        next_model = active_registry[i + 1][1] if i + 1 < len(active_registry) else None
        if next_model != model_name:
            del loaded_models[model_name]
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

    # ── Summary table sorted by MRR ──────────────────────────────────────────
    printable = [r for r in all_results if "error" not in r]
    sorted_results = sorted(printable, key=lambda r: r.get("MRR", 0), reverse=True)
    print(f"\n{'='*105}")
    print("SUMMARY — sorted by MRR")
    print(f"{'='*105}")
    hdr = (
        f"{'Run Key':<47} {'MRR':>6} {'NDCG@10':>8} {'MnRnk':>7} {'MdRnk':>7} "
        f"{'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@50':>6} {'P@1':>6} {'P@10':>6} {'Time':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted_results:
        print(
            f"{r['run_key']:<47} "
            f"{r['MRR']:.4f} "
            f"{r.get('NDCG@10', float('nan')):>8.4f} "
            f"{r.get('Mean Rank', float('nan')):>7.1f} "
            f"{r.get('Median Rank', float('nan')):>7.1f} "
            f"{r['Recall@1']:.4f} "
            f"{r['Recall@5']:.4f} "
            f"{r['Recall@10']:.4f} "
            f"{r['Recall@50']:.4f} "
            f"{r.get('P@1', float('nan')):>6.4f} "
            f"{r.get('P@10', float('nan')):>6.4f} "
            f"{r.get('encoding_time_s', 0):>7.1f}s"
        )
    failed = [r for r in all_results if "error" in r]
    if failed:
        print(f"\nFailed runs ({len(failed)}):")
        for r in failed:
            print(f"  {r['run_key']}: {r['error']}")

    print(f"\nRun directory: {run_dir}")
    print(f"  results.json  — {len(printable)} completed runs")
    print(f"  metadata.json — run config")
    print(f"  predictions/  — {len(list(preds_dir.glob('*.json')))} prediction files")
