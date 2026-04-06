"""lib/pipeline/local_corpus_generator.py — Generate fable summaries using a local HF model.

Stage 1 of the Exp10 matrix pipeline:
  fable text → 1 declarative moral summary (per gen model)

Output: gen_cache/<alias>/corpus_summaries.json + diagnostics.json
Cache key: (alias, prompt_version, fable_id)  — skipped unless --force or prompt_version changes.

The summary prompt is defined in the spec (prompt_version "v1"):
  "Distill the lesson into one declarative sentence of 5-15 words."
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from lib.pipeline.local_llm import generate_batch, load_model, unload_model, strip_thinking_tags
from lib.pipeline.paraphrase_filter import check_batch_word_counts


# ─── Prompt (spec §Prompts / Summarization) ──────────────────────────────────

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are an expert in moral philosophy. Distill the lesson of the following "
    "fable into one declarative sentence of 5 to 15 words. The statement must be "
    "universal and timeless — no character names, no reference to story events. "
    "Output ONLY the moral sentence. No explanation."
)

SUMMARIZATION_USER_TEMPLATE = "Fable: {text}"

# Known degenerate phrases (model-collapse indicators)
_GENERIC_PHRASES = [
    "honesty is the best policy",
    "slow and steady wins the race",
    "do not judge a book by its cover",
    "actions speak louder than words",
    "look before you leap",
]


def _cache_path(gen_cache_dir: Path) -> Path:
    return gen_cache_dir / "corpus_summaries.json"


def _diag_path(gen_cache_dir: Path) -> Path:
    return gen_cache_dir / "diagnostics.json"


def _is_cached(gen_cache_dir: Path, prompt_version: str, fables: list[dict]) -> bool:
    """Return True if a complete, up-to-date cache exists."""
    p = _cache_path(gen_cache_dir)
    if not p.exists():
        return False
    try:
        with open(p) as f:
            data = json.load(f)
        if data.get("status") != "complete":
            return False
        if data.get("prompt_version") != prompt_version:
            return False
        if len(data.get("items", [])) != len(fables):
            return False
        return True
    except Exception:
        return False


def _compute_summary_diagnostics(
    summaries: list[str],
    gen_cache_dir: Path,
    embed_cache_dir: Optional[Path] = None,
) -> dict:
    """
    Compute diagnostic statistics over a list of summaries.

    Returns dict with unique_ratio, duplicate_rate, generic_matches, word_count_violations.
    """
    n = len(summaries)
    if n == 0:
        return {}

    # Unique ratio
    unique_ratio = len(set(summaries)) / n

    # Near-duplicate rate via simple token overlap (fast, no embedding needed here)
    # Full embedding-based dup detection would need the embed model; use Jaccard instead.
    dup_count = 0
    seen_sets: list[set] = []
    for s in summaries:
        tokens = set(s.lower().split())
        is_dup = any(
            len(tokens & prev) / max(len(tokens | prev), 1) > 0.9
            for prev in seen_sets
        )
        if is_dup:
            dup_count += 1
        seen_sets.append(tokens)
    duplicate_rate = dup_count / n

    # Generic phrase detection
    generic_matches = []
    for i, s in enumerate(summaries):
        s_lower = s.lower()
        for phrase in _GENERIC_PHRASES:
            if phrase in s_lower:
                generic_matches.append({"idx": i, "text": s, "matched_phrase": phrase})

    # Word count violations
    ids = [str(i) for i in range(n)]
    wc_violations = check_batch_word_counts(summaries, ids, min_words=5, max_words=15)

    return {
        "n_fables": n,
        "unique_ratio": round(unique_ratio, 4),
        "duplicate_rate": round(duplicate_rate, 4),
        "generic_matches": generic_matches,
        "word_count_violations": wc_violations,
    }


def _write_cache(
    output_path: Path,
    gen_model_alias: str,
    gen_model_id: str,
    prompt_version: str,
    items: list[dict],
    status: str,
) -> None:
    cache_data = {
        "gen_model_alias": gen_model_alias,
        "gen_model_id": gen_model_id,
        "prompt_version": prompt_version,
        "status": status,
        "items": items,
    }
    with open(output_path, "w") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)


def _load_partial_items(output_path: Path, prompt_version: str) -> list[dict]:
    if not output_path.exists():
        return []
    try:
        with open(output_path) as f:
            data = json.load(f)
        if data.get("prompt_version") != prompt_version:
            return []
        return data.get("items", [])
    except Exception:
        return []


def generate_corpus_summaries(
    fables: list[dict],
    gen_model_alias: str,
    gen_model_id: str,
    gen_cache_dir: Path,
    prompt_version: str = "v1",
    max_new_tokens: int = 64,
    device: str = "mps",
    dtype_str: str = "bfloat16",
    force: bool = False,
    batch_size: int = 8,
) -> Path:
    """
    For each fable, generate one declarative moral summary and cache results.

    The model is assumed to already be unloaded from memory by the caller; this
    function loads it, generates all summaries, then unloads it.

    Args:
        fables:          List of fable dicts with keys: doc_id, title, text.
        gen_model_alias: Short alias used in paths (e.g. "Qwen3-8B").
        gen_model_id:    HuggingFace model ID (e.g. "Qwen/Qwen3-8B-Instruct").
        gen_cache_dir:   Directory: <run_dir>/gen_cache/<alias>/
        prompt_version:  Cache-bust key; change to force regeneration.
        max_new_tokens:  Generation budget (default 64 for short moral sentences).
        device:          Torch device string ("mps", "cpu").
        dtype_str:       Torch dtype ("bfloat16").
        force:           Re-generate even if cache exists.

    Returns:
        Path to written corpus_summaries.json.
    """
    gen_cache_dir = Path(gen_cache_dir)
    gen_cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = _cache_path(gen_cache_dir)

    if not force and _is_cached(gen_cache_dir, prompt_version, fables):
        print(f"  [skip] corpus_summaries already cached for {gen_model_alias} (prompt_version={prompt_version})")
        return output_path

    print(f"\n  ── Summarization: {gen_model_alias} ({len(fables)} fables) ──")

    model, tokenizer = load_model(gen_model_id, device=device, dtype_str=dtype_str)

    items = _load_partial_items(output_path, prompt_version)
    done_ids = {item["id"] for item in items}
    summaries_flat: list[str] = [item["summary"] for item in items]

    if items:
        print(f"  [resume] found {len(items)}/{len(fables)} cached summaries for {gen_model_alias}", flush=True)

    # Collect fables that still need generation
    pending = [
        (i, fable) for i, fable in enumerate(fables)
        if f"item_{int(fable['doc_id'].split('_')[1]):03d}" not in done_ids
    ]
    for i, fable in enumerate(fables):
        fable_idx = int(fable["doc_id"].split("_")[1])
        if f"item_{fable_idx:03d}" in done_ids:
            print(f"  [{i+1}/{len(fables)}] {fable.get('title', fable['doc_id'])}: [cached]", flush=True)

    # Process pending fables in batches
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start: batch_start + batch_size]
        user_prompts = [SUMMARIZATION_USER_TEMPLATE.format(text=f["text"]) for _, f in batch]

        raw_outputs = generate_batch(
            model, tokenizer,
            system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
            user_prompts=user_prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        for (orig_i, fable), raw_output in zip(batch, raw_outputs):
            fable_idx = int(fable["doc_id"].split("_")[1])
            item_id = f"item_{fable_idx:03d}"
            summary = strip_thinking_tags(raw_output)
            summaries_flat.append(summary)
            items.append({"id": item_id, "fable_text": fable["text"], "summary": summary})
            done_ids.add(item_id)
            short = summary[:80] + ("..." if len(summary) > 80 else "")
            print(f"  [{orig_i+1}/{len(fables)}] {fable.get('title', item_id)}: {short}", flush=True)

        _write_cache(output_path, gen_model_alias, gen_model_id, prompt_version, items, status="in_progress")

    unload_model(model, tokenizer)

    # Finalize corpus_summaries.json
    _write_cache(output_path, gen_model_alias, gen_model_id, prompt_version, items, status="complete")

    # Compute + write diagnostics
    diag = _compute_summary_diagnostics(summaries_flat, gen_cache_dir)
    diag["gen_model_alias"] = gen_model_alias
    diag["prompt_version"] = prompt_version
    with open(_diag_path(gen_cache_dir), "w") as f:
        json.dump(diag, f, indent=2)

    print(f"\n  Saved {len(items)} summaries to {output_path}")
    print(f"  Diagnostics: unique_ratio={diag['unique_ratio']:.3f}  "
          f"dup_rate={diag['duplicate_rate']:.3f}  "
          f"generic={len(diag['generic_matches'])}  "
          f"wc_violations={len(diag['word_count_violations'])}")
    return output_path


def load_corpus_summaries(gen_cache_dir: Path) -> list[dict]:
    """
    Load the items list from a gen_cache corpus_summaries.json.

    Returns:
        List of dicts: {id, fable_text, summary}
    """
    p = _cache_path(Path(gen_cache_dir))
    if not p.exists():
        raise FileNotFoundError(f"corpus_summaries.json not found at {p}")
    with open(p) as f:
        data = json.load(f)
    if data.get("status") != "complete":
        raise RuntimeError(f"corpus_summaries.json at {p} is still in progress")
    return data["items"]
