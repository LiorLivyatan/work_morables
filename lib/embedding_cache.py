"""
lib/embedding_cache.py — Disk-based embedding cache for retrieval experiments.

Avoids re-encoding the same texts with the same model across runs.
Cache key = MD5(model_id + is_query + all input texts in order).
Embeddings stored as .npy files under cache/embeddings/.

Usage:
    from embedding_cache import encode_with_cache

    embs = encode_with_cache(
        model=model,
        texts=fable_texts,
        model_id="Linq-AI-Research/Linq-Embed-Mistral",
        query_instruction=None,    # None for corpus, string for queries
        cache_dir=CACHE_DIR,
        label="raw fables",        # for logging
    )
"""

import hashlib
from pathlib import Path

import numpy as np


def _cache_key(model_id: str, texts: list[str], query_instruction: str | None) -> str:
    """Compute a stable MD5 hash for a (model, texts, instruction) triple."""
    h = hashlib.md5()
    h.update(model_id.encode())
    h.update((query_instruction or "").encode())
    for t in texts:
        h.update(t.encode())
    return h.hexdigest()


def encode_with_cache(
    model,
    texts: list[str],
    model_id: str,
    cache_dir: Path,
    query_instruction: str | None = None,
    label: str = "",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Encode texts with the model, loading from disk cache if available.

    Args:
        model:             SentenceTransformer model instance.
        texts:             List of strings to encode.
        model_id:          Model identifier string (used in cache key).
        cache_dir:         Directory to store/load .npy cache files.
        query_instruction: If set, prepend "Instruct: {inst}\nQuery: {text}"
                           to each text (E5-instruct format). None = corpus.
        label:             Human-readable label for log output.
        batch_size:        Encoding batch size.

    Returns:
        float32 numpy array of shape (len(texts), embedding_dim), L2-normalized.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(model_id, texts, query_instruction)
    cache_path = cache_dir / f"{key}.npy"
    meta_path  = cache_dir / f"{key}.meta.txt"

    if cache_path.exists():
        embs = np.load(str(cache_path))
        print(f"  [cache hit]  {label or key[:8]}  "
              f"({len(texts)} texts, shape {embs.shape})")
        return embs

    print(f"  [encoding]   {label or key[:8]}  ({len(texts)} texts)...")
    encode_texts = texts
    if query_instruction:
        encode_texts = [f"Instruct: {query_instruction}\nQuery: {t}" for t in texts]

    embs = model.encode(
        encode_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(str(cache_path), embs)

    # Save human-readable metadata for debugging
    with open(meta_path, "w") as f:
        f.write(f"model_id: {model_id}\n")
        f.write(f"label: {label}\n")
        f.write(f"query_instruction: {query_instruction}\n")
        f.write(f"n_texts: {len(texts)}\n")
        f.write(f"shape: {embs.shape}\n")
        f.write(f"first_text: {texts[0][:120]}\n")

    print(f"  [saved]      {cache_path.name}")
    return embs
