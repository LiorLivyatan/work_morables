"""lib/pipeline/local_query_paraphraser.py — Generate moral paraphrases using a local HF model.

Stage 1 (query side) of the Exp10 matrix pipeline:
  moral text → 3 rephrasings (per gen model)

Post-processing:
  - Word count enforcement (5–15 words, flagged in diagnostics.json)
  - Similarity filter via BAAI/bge-m3 (sim < 0.85 → dropped, logged)
  - If all 3 filtered → fallback to [original] (no rephrases kept)

Output: gen_cache/<alias>/query_paraphrases.json + diagnostic data merged into diagnostics.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from lib.pipeline.local_llm import generate_batch, load_model, unload_model, strip_thinking_tags
from lib.pipeline.paraphrase_filter import check_batch_word_counts


# ─── Prompt (spec §Prompts / Paraphrase) ─────────────────────────────────────

PARAPHRASE_SYSTEM_PROMPT = (
    "You are given a moral statement. Write exactly 3 different rephrasings using "
    "different words while preserving the exact same meaning. Each must be abstract "
    "and universal, 5 to 15 words. "
    "Output ONLY the 3 rephrasings, one per line. No numbers, no labels."
)

PARAPHRASE_USER_TEMPLATE = "Moral: {text}"


# ─── Output paths ─────────────────────────────────────────────────────────────

def _output_path(gen_cache_dir: Path) -> Path:
    return gen_cache_dir / "query_paraphrases.json"


def _diag_path(gen_cache_dir: Path) -> Path:
    return gen_cache_dir / "diagnostics.json"


def _is_cached(gen_cache_dir: Path, prompt_version: str, morals: list) -> bool:
    p = _output_path(gen_cache_dir)
    if not p.exists():
        return False
    try:
        with open(p) as f:
            data = json.load(f)
        if data.get("status") != "complete":
            return False
        if data.get("prompt_version") != prompt_version:
            return False
        if len(data.get("items", [])) != len(morals):
            return False
        return True
    except Exception:
        return False


def _parse_paraphrases(raw: str) -> list[str]:
    """
    Parse the model's raw line-separated output into a list of up to 3 paraphrases.
    Strips empty lines, leading numbers/bullets, and surrounding whitespace.
    """
    import re
    lines = [line.strip() for line in raw.strip().splitlines()]
    cleaned = []
    for line in lines:
        if not line:
            continue
        # Remove leading numbering like "1.", "1)", "-", "*"
        line = re.sub(r"^[\d]+[.)]\s*|^[-*]\s*", "", line)
        if line:
            cleaned.append(line)
    return cleaned[:3]  # at most 3


def _write_partial_output(
    out_path: Path,
    gen_model_alias: str,
    gen_model_id: str,
    prompt_version: str,
    sim_threshold: float,
    items: list[dict],
    status: str,
) -> None:
    output_data = {
        "gen_model_alias": gen_model_alias,
        "gen_model_id": gen_model_id,
        "prompt_version": prompt_version,
        "sim_threshold": sim_threshold,
        "status": status,
        "items": items,
    }
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def _load_partial_items(out_path: Path, prompt_version: str) -> list[dict]:
    if not out_path.exists():
        return []
    try:
        with open(out_path) as f:
            data = json.load(f)
        if data.get("prompt_version") != prompt_version:
            return []
        return data.get("items", [])
    except Exception:
        return []


def generate_query_paraphrases(
    moral_entries: list[tuple[int, int]],
    morals: list[dict],
    gen_model_alias: str,
    gen_model_id: str,
    gen_cache_dir: Path,
    embed_cache_dir: Path | None = None,
    prompt_version: str = "v1",
    max_new_tokens: int = 128,
    device: str = "mps",
    dtype_str: str = "bfloat16",
    sim_threshold: float = 0.85,
    force: bool = False,
    batch_size: int = 8,
) -> Path:
    """
    For each moral, generate 3 paraphrases, filter by similarity, and cache results.

    The model is loaded here, generates all paraphrases, then unloaded.
    BGE-M3 is loaded separately inside filter_paraphrase_batch; it is also
    unloaded after filtering (managed inside paraphrase_filter).

    Args:
        moral_entries:   List of (moral_idx, fable_idx) from the pipeline.
        morals:          Full morals list (lib.data.load_morals()).
        gen_model_alias: Short alias (e.g. "Qwen3-8B").
        gen_model_id:    HuggingFace model ID.
        gen_cache_dir:   Directory: <run_dir>/gen_cache/<alias>/
        embed_cache_dir: Directory for BGE-M3 embedding cache used in filtering.
        prompt_version:  Cache-bust key.
        max_new_tokens:  Token budget for paraphrase generation (default 128).
        device:          Torch device.
        dtype_str:       Torch dtype.
        sim_threshold:   Cosine similarity threshold for paraphrase filter (default 0.85).
        force:           Re-generate even if cache exists.

    Returns:
        Path to written query_paraphrases.json.
    """
    gen_cache_dir = Path(gen_cache_dir)
    gen_cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = _output_path(gen_cache_dir)

    if not force and _is_cached(gen_cache_dir, prompt_version, moral_entries):
        print(f"  [skip] query_paraphrases already cached for {gen_model_alias} (prompt_version={prompt_version})")
        return out_path

    print(f"\n  ── Paraphrasing: {gen_model_alias} ({len(moral_entries)} morals) ──")

    # Step A: Generate raw paraphrases with the local LLM
    model, tokenizer = load_model(gen_model_id, device=device, dtype_str=dtype_str)

    raw_items = _load_partial_items(out_path, prompt_version)
    done_moral_ids = {item["moral_idx"] for item in raw_items}
    original_texts: list[str] = [item["original_moral"] for item in raw_items]
    raw_paras_list: list[list[str]] = [item["raw_paraphrases"] for item in raw_items]

    if raw_items:
        print(f"  [resume] found {len(raw_items)}/{len(moral_entries)} cached paraphrase prompts for {gen_model_alias}", flush=True)

    # Collect morals that still need generation
    pending = [
        (i, moral_idx, fable_idx) for i, (moral_idx, fable_idx) in enumerate(moral_entries)
        if moral_idx not in done_moral_ids
    ]
    for i, (moral_idx, _) in enumerate(moral_entries):
        if moral_idx in done_moral_ids:
            print(f"  [{i+1}/{len(moral_entries)}] moral_{moral_idx:03d}: [cached]", flush=True)

    # Process pending morals in batches
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start: batch_start + batch_size]
        user_prompts = [
            PARAPHRASE_USER_TEMPLATE.format(text=morals[moral_idx]["text"])
            for _, moral_idx, _ in batch
        ]

        raw_outputs = generate_batch(
            model, tokenizer,
            system_prompt=PARAPHRASE_SYSTEM_PROMPT,
            user_prompts=user_prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        for (orig_i, moral_idx, fable_idx), raw_output in zip(batch, raw_outputs):
            moral_text = morals[moral_idx]["text"]
            paras = _parse_paraphrases(strip_thinking_tags(raw_output))
            raw_items.append({
                "moral_idx": moral_idx,
                "fable_idx": fable_idx,
                "original_moral": moral_text,
                "raw_paraphrases": paras,
            })
            original_texts.append(moral_text)
            raw_paras_list.append(paras)
            done_moral_ids.add(moral_idx)
            short = ", ".join(f'"{p[:40]}"' for p in paras)
            print(f"  [{orig_i+1}/{len(moral_entries)}] moral_{moral_idx:03d}: {short}", flush=True)

        _write_partial_output(
            out_path, gen_model_alias, gen_model_id, prompt_version,
            sim_threshold, raw_items, status="in_progress",
        )

    unload_model(model, tokenizer)

    # Step B: Word-count check on raw paraphrases (diagnostics only)
    all_para_texts = [p for paras in raw_paras_list for p in paras]
    all_para_ids = [
        f"moral_{moral_entries[i][0]:03d}_r{j}"
        for i, paras in enumerate(raw_paras_list)
        for j in range(len(paras))
    ]
    wc_violations = check_batch_word_counts(all_para_texts, all_para_ids)

    # Assemble output — raw_paraphrases are used directly in retrieval, no filter needed.
    items: list[dict] = []
    for raw_item in raw_items:
        items.append({
            "moral_idx": raw_item["moral_idx"],
            "fable_idx": raw_item["fable_idx"],
            "original_moral": raw_item["original_moral"],
            "raw_paraphrases": raw_item["raw_paraphrases"],
        })

    _write_partial_output(
        out_path,
        gen_model_alias,
        gen_model_id,
        prompt_version,
        sim_threshold,
        items,
        status="complete",
    )

    # Merge diagnostics into diagnostics.json
    diag_path = _diag_path(gen_cache_dir)
    diag: dict = {}
    if diag_path.exists():
        with open(diag_path) as f:
            diag = json.load(f)

    diag["paraphrase_word_count_violations"] = wc_violations
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)

    print(f"\n  Saved {len(items)} paraphrase entries to {out_path}")
    print(f"  WC_violations={len(wc_violations)}", flush=True)
    return out_path


def load_query_paraphrases(gen_cache_dir: Path) -> list[dict]:
    """
    Load the items list from a gen_cache query_paraphrases.json.

    Returns:
        List of dicts: {moral_idx, fable_idx, original_moral, raw_paraphrases}
    """
    p = _output_path(Path(gen_cache_dir))
    if not p.exists():
        raise FileNotFoundError(f"query_paraphrases.json not found at {p}")
    with open(p) as f:
        data = json.load(f)
    if data.get("status") != "complete":
        raise RuntimeError(f"query_paraphrases.json at {p} is still in progress")
    return data["items"]
