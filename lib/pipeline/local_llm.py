"""lib/pipeline/local_llm.py — Load a HuggingFace chat model to MPS, run generation, unload.

Supports models that use chat templates (Qwen3, Gemma, Phi-3.5, etc.).
Designed for one-model-at-a-time usage to avoid MPS OOM.

Usage:
    model, tokenizer = load_model("Qwen/Qwen3-8B-Instruct")
    output = generate(model, tokenizer, system_prompt, user_prompt)
    unload_model(model, tokenizer)
"""
from __future__ import annotations

import gc
import re
from pathlib import Path


def ensure_transformers_dynamic_module_cache() -> str:
    """
    Point transformers' dynamic module cache at a writable project-local path.
    """
    cache_dir = str(Path(__file__).resolve().parent.parent.parent / ".hf_modules_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    import transformers.dynamic_module_utils as dynamic_module_utils
    import transformers.utils as transformers_utils

    dynamic_module_utils.HF_MODULES_CACHE = cache_dir
    transformers_utils.HF_MODULES_CACHE = cache_dir
    return cache_dir


def resolve_model_source(model_id: str) -> tuple[str, bool]:
    """
    Return a local snapshot path when the model is already cached.

    Returns:
        (source, is_local_source)
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        model_source = Path(snapshot_download(model_id, local_files_only=True))
        return str(model_source), True
    except LocalEntryNotFoundError:
        return model_id, False


def sentence_transformer_load_kwargs(model_id: str, is_local_source: bool) -> dict:
    """
    Extra kwargs for SentenceTransformer loaders.

    Some embedding repos, such as Nomic, rely on custom model code referenced
    through `auto_map`. Those need `trust_remote_code=True` so transformers can
    instantiate the architecture from the cached Hub code files.
    """
    kwargs = {"local_files_only": is_local_source}
    if model_id.startswith("nomic-ai/"):
        ensure_transformers_dynamic_module_cache()
        kwargs["trust_remote_code"] = True
        kwargs["model_kwargs"] = {"trust_remote_code": True}
    return kwargs


def load_model(
    model_id: str,
    device: str = "mps",
    dtype_str: str = "bfloat16",
):
    """
    Load a HuggingFace causal-LM model and its tokenizer onto `device`.

    Args:
        model_id:   HuggingFace model repo ID (e.g. "Qwen/Qwen3-8B-Instruct").
        device:     Target device string ("mps", "cpu", "cuda").
        dtype_str:  Torch dtype string ("bfloat16", "float16", "float32").

    Returns:
        (model, tokenizer) tuple — both moved to device.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    # Auto-detect best available device if not explicitly set
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"  [local_llm] Auto-detected device: {device}", flush=True)
    elif device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
        print(f"  [local_llm] MPS not available — falling back to CPU", flush=True)
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print(f"  [local_llm] CUDA not available — falling back to CPU", flush=True)

    # Prefer a fully local snapshot when one is already cached. This avoids
    # extra Hub metadata lookups that can fail in offline environments.
    model_source, is_local_source = resolve_model_source(model_id)

    print(f"  [local_llm] Loading {model_id!r} on {device} ({dtype_str}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_source),
        local_files_only=is_local_source,
    )
    # low_cpu_mem_usage=True uses meta-device initialisation, reducing the CPU RAM
    # peak during loading before we move to MPS. Without this flag, transformers
    # allocates full float32 tensors on CPU even when torch_dtype=bfloat16.
    model = AutoModelForCausalLM.from_pretrained(
        str(model_source),
        local_files_only=is_local_source,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print(f"  [local_llm] Loaded {model_id!r}  params={sum(p.numel() for p in model.parameters())/1e9:.1f}B", flush=True)
    return model, tokenizer


def _build_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Apply chat template with Qwen3 thinking mode disabled."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def generate(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> str:
    """
    Run one chat-template generation. Delegates to generate_batch internally.

    Returns:
        Stripped generated text (only the new tokens, not the prompt).
    """
    return generate_batch(
        model, tokenizer,
        system_prompt=system_prompt,
        user_prompts=[user_prompt],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )[0]


def generate_batch(
    model,
    tokenizer,
    system_prompt: str,
    user_prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> list[str]:
    """
    Run chat-template generation for a batch of user prompts in one GPU call.

    All prompts share the same system_prompt. Results are independent — each
    item in the batch is processed separately by the model; they just travel
    to the GPU together, keeping more cores busy simultaneously.

    Args:
        model:          Loaded HF model (already on device).
        tokenizer:      Matching tokenizer.
        system_prompt:  System-role text (same for all items in the batch).
        user_prompts:   List of user-role texts, one per item.
        max_new_tokens: Max new tokens to generate per item.
        temperature:    Sampling temperature (unused when do_sample=False).
        do_sample:      Whether to use sampling (False → greedy).

    Returns:
        List of stripped generated texts, one per input prompt (same order).
    """
    import torch

    prompt_texts = [_build_prompt(tokenizer, system_prompt, up) for up in user_prompts]

    # Tokenize with left-padding so all sequences end at the same position,
    # which is required for correct batch decoding.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(model.device)
    tokenizer.padding_side = original_padding_side

    # With left-padding all sequences are right-aligned, so generated tokens
    # always start at the padded input length (same index for every item).
    padded_input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens for each item in the batch
    results = []
    for i in range(len(user_prompts)):
        new_tokens = output_ids[i, padded_input_len:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append(raw.strip())
    return results


def unload_model(model, tokenizer) -> None:
    """
    Move model to CPU then delete; run garbage collection to free MPS memory.
    """
    import torch

    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    del tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("  [local_llm] Model unloaded and cache cleared.", flush=True)


def strip_thinking_tags(text: str) -> str:
    """
    Remove <think>...</think> blocks emitted by reasoning-mode models (e.g. Qwen3).
    Returns the remaining text stripped of leading/trailing whitespace.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()
