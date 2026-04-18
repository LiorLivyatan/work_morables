"""
Retrieval evaluation for fine-tuned SentenceTransformer models.

Public API
----------
evaluate(model, moral_texts, doc_texts, ground_truth, cache_dir=None, force=False)
    -> dict  (MRR, R@1, R@5, R@10, R@50, ...)

Caching
-------
When cache_dir is provided, moral and document embeddings are persisted as .npy
files so re-evaluating a cached model (e.g. with different metrics) skips the
expensive encode step. Pass force=True to re-encode regardless.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.retrieval_utils import compute_metrics


def evaluate(
    model,
    moral_texts: list[str],
    doc_texts: list[str],
    ground_truth: dict[int, int],
    cache_dir: Path | None = None,
    force: bool = False,
    query_prompt: str | None = None,
) -> dict:
    """
    Encode moral queries and fable documents, return full retrieval metrics.

    The doc_texts corpus should always be the full 709 fables — not just the
    test split — to reflect realistic retrieval conditions.

    Args:
        model          fine-tuned (or baseline) SentenceTransformer
        moral_texts    query strings for this evaluation set
        doc_texts      full corpus of document strings (typically all 709 fables)
        ground_truth   {query_local_idx: global_corpus_fable_idx}
        cache_dir      optional directory to persist/load embeddings
        force          re-encode even if embeddings are cached
        query_prompt   optional instruction prefix prepended to each moral before
                       encoding (e.g. Linq-Embed-Mistral's task instruction);
                       only applies to queries, not documents
    """
    moral_embs = _encode(model, moral_texts, cache_dir, "moral_embs.npy", force, prompt=query_prompt)
    doc_embs = _encode(model, doc_texts, cache_dir, "doc_embs.npy", force)
    return compute_metrics(moral_embs, doc_embs, ground_truth)


def _encode(
    model,
    texts: list[str],
    cache_dir: Path | None,
    filename: str,
    force: bool,
    prompt: str | None = None,
) -> np.ndarray:
    cache_path = (cache_dir / filename) if cache_dir else None

    if cache_path and cache_path.exists() and not force:
        print(f"    [cache hit] Loading embeddings ← {cache_path}")
        return np.load(str(cache_path))

    encode_kwargs: dict = {"normalize_embeddings": True, "show_progress_bar": True, "batch_size": 64}
    if prompt:
        encode_kwargs["prompt"] = prompt
    embs = model.encode(texts, **encode_kwargs)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), embs)
        print(f"    [cache saved] Embeddings → {cache_path}")

    return embs
