"""lib/pipeline/paraphrase_filter.py — Post-processing and quality gates for generated text.

Two responsibilities:
1. Word-count enforcement (flag, not truncate).
2. Paraphrase similarity filter — drop rephrases with cosine sim < threshold vs original.

The filter uses BAAI/bge-m3 embeddings (fixed, per spec) cached via lib.embedding_cache.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from lib.pipeline.local_llm import resolve_model_source, sentence_transformer_load_kwargs


# ── Word-count enforcement ────────────────────────────────────────────────────

def check_word_count(
    text: str,
    min_words: int = 5,
    max_words: int = 15,
) -> tuple[bool, int]:
    """
    Return (in_range, word_count). Does NOT modify the text.

    Args:
        text:      The generated sentence.
        min_words: Minimum allowed word count (inclusive).
        max_words: Maximum allowed word count (inclusive).

    Returns:
        (ok, count) — ok is True when min_words ≤ count ≤ max_words.
    """
    count = len(text.split())
    return (min_words <= count <= max_words), count


def check_batch_word_counts(
    texts: list[str],
    ids: list[str],
    min_words: int = 5,
    max_words: int = 15,
) -> list[dict]:
    """
    Check word counts for a batch of texts, returning diagnostics for out-of-range ones.

    Args:
        texts:     List of generated text strings.
        ids:       Corresponding identifiers (same length as texts).
        min_words: Minimum allowed word count.
        max_words: Maximum allowed word count.

    Returns:
        List of dicts describing out-of-range items (empty list if all pass).
        Each dict: {"id": ..., "text": ..., "word_count": ..., "reason": "too_short"|"too_long"}
    """
    violations = []
    for item_id, text in zip(ids, texts):
        ok, count = check_word_count(text, min_words, max_words)
        if not ok:
            violations.append({
                "id": item_id,
                "text": text,
                "word_count": count,
                "reason": "too_short" if count < min_words else "too_long",
            })
    return violations


# ── Paraphrase similarity filter ──────────────────────────────────────────────

_FILTER_MODEL_ID = "BAAI/bge-m3"


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def filter_paraphrases(
    original_text: str,
    paraphrases: list[str],
    cache_dir: Path,
    sim_threshold: float = 0.85,
    model_id: str = _FILTER_MODEL_ID,
) -> tuple[list[str], list[dict]]:
    """
    Filter paraphrases by cosine similarity to the original.

    Embeds `original_text` and each rephrase using `model_id` (BAAI/bge-m3).
    Embeddings are cached via lib.embedding_cache.

    Args:
        original_text:  The source moral statement.
        paraphrases:    List of 3 rephrase strings to evaluate.
        cache_dir:      Directory for the embedding cache.
        sim_threshold:  Minimum cosine similarity to keep a rephrase (default 0.85).
        model_id:       Embedding model ID (should stay BAAI/bge-m3).

    Returns:
        (kept, filtered_log) where:
          - kept:         Rephrases that passed the filter.
          - filtered_log: List of dicts for dropped rephrases:
                          {"text": ..., "similarity": ..., "reason": "low_similarity"}
    """
    from lib.embedding_cache import encode_with_cache

    # Lazy-load BGE-M3 for filtering (no instruction needed for symmetric tasks)
    from sentence_transformers import SentenceTransformer
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_source, is_local_source = resolve_model_source(model_id)
    filter_model = SentenceTransformer(
        model_source,
        device=device,
        **sentence_transformer_load_kwargs(model_id, is_local_source),
    )

    all_texts = [original_text] + paraphrases
    embs = encode_with_cache(
        model=filter_model,
        texts=all_texts,
        model_id=model_id,
        cache_dir=cache_dir,
        query_instruction=None,
        label="paraphrase_filter",
    )

    orig_emb = embs[0]
    kept: list[str] = []
    filtered_log: list[dict] = []

    for i, para in enumerate(paraphrases):
        sim = _cosine_sim(orig_emb, embs[i + 1])
        if sim >= sim_threshold:
            kept.append(para)
        else:
            filtered_log.append({
                "text": para,
                "similarity": round(float(sim), 4),
                "reason": "low_similarity",
            })

    # Spec: if all 3 filtered → fall back to original only
    if not kept:
        kept = []  # caller should use [original_text] as fallback

    return kept, filtered_log


def filter_paraphrase_batch(
    originals: list[str],
    paraphrase_lists: list[list[str]],
    cache_dir: Path,
    sim_threshold: float = 0.85,
    model_id: str = _FILTER_MODEL_ID,
) -> tuple[list[list[str]], list[dict]]:
    """
    Batch version: filter rephrases for N originals in one embedding pass.

    Args:
        originals:        List of N original moral strings.
        paraphrase_lists: List of N lists, each with up to 3 rephrases.
        cache_dir:        Embedding cache directory.
        sim_threshold:    Minimum cosine similarity threshold.
        model_id:         Embedding model ID.

    Returns:
        (kept_lists, all_filtered_log) where:
          - kept_lists:       List of N kept-rephrase lists (may be empty → use original).
          - all_filtered_log: Flat list of all dropped-rephrase diagnostics,
                              augmented with {"original_idx": i}.
    """
    from lib.embedding_cache import encode_with_cache
    from sentence_transformers import SentenceTransformer
    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model_source, is_local_source = resolve_model_source(model_id)
    filter_model = SentenceTransformer(
        model_source,
        device=device,
        **sentence_transformer_load_kwargs(model_id, is_local_source),
    )

    # Build flat list: [orig_0, para_0_0, para_0_1, ..., orig_1, para_1_0, ...]
    # We track slice indices per original.
    flat_texts: list[str] = []
    slices: list[tuple[int, int]] = []  # (start, end) of paraphrases in flat_texts for each orig

    for orig, paras in zip(originals, paraphrase_lists):
        flat_texts.append(orig)
        start = len(flat_texts)
        flat_texts.extend(paras)
        slices.append((start, start + len(paras)))

    embs = encode_with_cache(
        model=filter_model,
        texts=flat_texts,
        model_id=model_id,
        cache_dir=cache_dir,
        query_instruction=None,
        label="paraphrase_filter_batch",
    )

    kept_lists: list[list[str]] = []
    all_filtered_log: list[dict] = []

    orig_pos = 0
    for i, (paras, (start, end)) in enumerate(zip(paraphrase_lists, slices)):
        orig_emb = embs[orig_pos]
        kept: list[str] = []
        for j, para in enumerate(paras):
            sim = _cosine_sim(orig_emb, embs[start + j])
            if sim >= sim_threshold:
                kept.append(para)
            else:
                all_filtered_log.append({
                    "original_idx": i,
                    "text": para,
                    "similarity": round(float(sim), 4),
                    "reason": "low_similarity",
                })
        kept_lists.append(kept)
        orig_pos = end  # next orig is right after its paraphrases

    return kept_lists, all_filtered_log
