"""
exp_13_gemma4_gpu_summarization — Generate fable summaries using Gemma 4 on GPU.

Generates a one-sentence moral summary for each of 709 fables using each
configured Gemma 4 model × prompt variant combination. Output is a single
unified JSON with structure: list[{fable_alias, fable_text, summaries: {model_alias: {variant: text}}}].

This is the GPU-server counterpart of exp_09 (MLX/Mac). Uses bfloat16 + transformers
for full-precision generation and supports Gemma 4's thinking mode (enable_thinking=True).

Usage
-----
    # All models, all variants:
    ./run.sh experiments/13_gemma4_gpu_summarization/generate.py

    # Specific models/variants:
    ./run.sh experiments/13_gemma4_gpu_summarization/generate.py --models gemma4-E2B gemma4-26B-A4B
    ./run.sh experiments/13_gemma4_gpu_summarization/generate.py --variants cot_proverb direct_moral

    # Quick smoke test (5 fables, not saved):
    ./run.sh experiments/13_gemma4_gpu_summarization/generate.py --sample 5

    # Resume interrupted run:
    ./run.sh experiments/13_gemma4_gpu_summarization/generate.py --resume results/2026-xx-xx_summaries.json

    # Remote GPU:
    ./run.sh experiments/13_gemma4_gpu_summarization/generate.py --remote --gpu 2
"""
import argparse
import gc
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

EXP_DIR = Path(__file__).parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify

CONFIG_PATH = EXP_DIR / "config.yaml"
RESULTS_DIR = EXP_DIR / "results"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_base_data():
    from lib.data import load_fables, load_morals, load_qrels_moral_to_fable
    fables        = load_fables()
    morals        = load_morals()
    gt_m2f        = load_qrels_moral_to_fable()
    fable_to_moral = {
        fable_idx: morals[moral_idx]["text"]
        for moral_idx, fable_idx in gt_m2f.items()
    }
    return fables, fable_to_moral


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_processor(model_cfg: dict):
    """Load Gemma 4 model + tokenizer for text-only generation.
    Uses AutoTokenizer (not AutoProcessor) to avoid torchvision dependency from
    Gemma4VideoProcessor. Falls back to AutoModelForCausalLM if
    AutoModelForMultimodalLM is not available. Supports load_in_4bit for large models."""
    from transformers import AutoTokenizer
    try:
        from transformers import AutoModelForMultimodalLM
        model_class = AutoModelForMultimodalLM
    except ImportError:
        from transformers import AutoModelForCausalLM
        model_class = AutoModelForCausalLM

    dtype_str = model_cfg.get("torch_dtype", "bfloat16")
    torch_dtype = getattr(torch, dtype_str) if isinstance(dtype_str, str) else dtype_str

    load_kwargs: dict = {"device_map": "auto", "torch_dtype": torch_dtype}
    if model_cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model     = model_class.from_pretrained(model_cfg["id"], **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["id"])
    return model, tokenizer


# ── Summary generation ────────────────────────────────────────────────────────

# Patterns that indicate a quality-check / meta note, NOT the actual answer
_META_RE = re.compile(
    r"self-correction:|must be\s+\*?only|"
    r"ensure\s+it\s+(is|sounds|captures|conveys)|"
    r"quality\s+check|final\s+polish|final\s+check|verify\s+that",
    re.IGNORECASE,
)


def _strip_preamble(line: str) -> str:
    """Strip leading parenthetical blocks, label prefixes, and quotation marks."""
    line = re.sub(r"^\([^)]*\)\s*", "", line).strip()
    line = re.sub(
        r"^(?:Proverb|Moral|Answer|Lesson|Final proverb|Final moral|Summary|Principle)"
        r"\s*:?\s*\**\s*",
        "", line, flags=re.IGNORECASE,
    ).strip()
    line = re.sub(r'^["\']+|["\']+$', "", line).strip()
    return line


def _postprocess(text: str, variant_cfg: dict) -> str:
    """Strip thinking blocks and extract the final answer.

    Gemma 4 thinking output has varied formats. The model often reasons in
    numbered steps and appends a quality-check line after the actual answer:
      1. **Analyze:** ...
      2. **Extract:** ...
      3. **State as proverb:** "One good turn deserves another."
      Must be *only* the proverb. (Self-Correction: ...)

    Strategy: work backwards through lines; skip quality-check lines; extract
    the content of the last numbered step that isn't itself a meta note.
    """
    enable_thinking = variant_cfg.get("enable_thinking", False)

    # Strip residual special tokens that appear when skip_special_tokens=False
    text = re.sub(r"<(bos|eos|pad|unk|sep|cls)>|</?(s|pad)>", "", text).strip()
    text = re.sub(r"<\|[^|>]+\|>|<turn\|>|\|turn>", "", text).strip()

    # Strip tag-based thinking blocks (older transformers)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Gemma 4 thinking format: <|channel>thought\n[reasoning]\n<channel|>[answer]
    # Also present on 26B/31B standard variants (empty thinking token in template).
    # Split on <channel|> and keep only the answer part.
    if "<channel|>" in text:
        text = text.split("<channel|>")[-1].strip()

    if enable_thinking:

        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        for line in reversed(lines):
            if len(line.split()) > 80:
                # Long reasoning prose — skip
                continue

            is_numbered = bool(re.match(r"^\d+\.\s+\*\*", line))
            has_meta    = bool(_META_RE.search(line))

            if has_meta and not is_numbered:
                # Pure quality-check line — try to salvage embedded answer
                parts = re.split(r"\)\s*", line)
                for part in reversed(parts):
                    part = part.strip().lstrip("(").strip()
                    if part and not _META_RE.search(part) and len(part.split()) >= 3:
                        return _strip_preamble(part)
                continue  # nothing salvageable, skip

            if has_meta and is_numbered:
                # Numbered quality-check step (e.g., "4. **Final check:** Must be…")
                continue

            if is_numbered:
                # Numbered reasoning step — extract its content
                content = re.sub(r"^\d+\.?\s*\*\*[^*]+\*\*:?\s*", "", line).strip()
                content = _strip_preamble(content)
                if content and len(content.split()) >= 3:
                    return content
                continue

            # Plain line — use it
            if len(line.split()) >= 3:
                return _strip_preamble(line)

        # Absolute fallback
        if lines:
            return _strip_preamble(lines[-1])
        return text

    elif variant_cfg["id"] in ("cot_proverb", "conceptual_abstract"):
        # Non-thinking CoT: safety net in case model leaked reasoning
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if lines:
            text = lines[-1]

    return text


def generate_batch(
    model,
    tokenizer,
    fable_text: str,
    variants: list,
    max_new_tokens: int,
    max_new_tokens_thinking: int,
) -> dict:
    """Generate summaries for all variants of one fable in two batched generate() calls.

    Splits variants into standard (non-thinking) and thinking groups so each group
    can use its own max_new_tokens. Within each group all prompts are tokenized
    together with left-padding and run through model.generate() once, which keeps
    the GPU busy across the full batch rather than idling between individual calls.

    Returns {variant_id: {"text": ..., "input_tokens": ..., "output_tokens": ...}}
    """
    results: dict = {}

    # Ensure left-padding so all sequences align at the right (generation side)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    standard = [v for v in variants if not v.get("enable_thinking", False)]
    thinking  = [v for v in variants if v.get("enable_thinking", False)]

    for group, n_tokens in [(standard, max_new_tokens), (thinking, max_new_tokens_thinking)]:
        if not group:
            continue

        enable_thinking = group[0].get("enable_thinking", False)

        # Build one prompt string per variant (no tokenization yet)
        prompt_strings = []
        for vcfg in group:
            messages = [
                {"role": "system", "content": vcfg["system"].strip()},
                {"role": "user",   "content": f"Fable: {fable_text}"},
            ]
            try:
                s = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                s = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            prompt_strings.append(s)

        # Batch-tokenize with left-padding → one forward pass for the whole group
        inputs = tokenizer(
            prompt_strings,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                do_sample=False,
            )

        # Decode: input length is uniform across the batch (left-padded)
        n_input = inputs["input_ids"].shape[1]
        for i, vcfg in enumerate(group):
            new_ids = output_ids[i][n_input:]
            raw  = tokenizer.decode(new_ids, skip_special_tokens=False).strip()
            text = _postprocess(raw, vcfg)
            results[vcfg["id"]] = {
                "text":          text,
                "raw":           raw,
                "input_tokens":  n_input,
                "output_tokens": int(new_ids.shape[0]),
            }

    return results


# ── Corpus I/O ─────────────────────────────────────────────────────────────────

def _fable_key(fable: dict) -> str:
    return fable.get("alias", fable["doc_id"])


def _save(corpus_by_key: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(list(corpus_by_key.values()), f, indent=2, ensure_ascii=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",   nargs="+", help="Model aliases to run (default: all)")
    parser.add_argument("--variants", nargs="+", help="Prompt variant IDs to run (default: all)")
    parser.add_argument("--sample",   type=int,  help="Only process first N fables (smoke test, no save)")
    parser.add_argument("--resume",   type=str,  help="Path to existing summaries JSON to resume from")
    parser.add_argument("--output",   type=str,  help="Override output JSON path")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models_cfg   = config["models"]
    variants_cfg = config["prompt_variants"]

    if args.models:
        sel = {m.lower() for m in args.models}
        models_cfg = [m for m in models_cfg if m["alias"].lower() in sel]
    if args.variants:
        sel = {v.lower() for v in args.variants}
        variants_cfg = [v for v in variants_cfg if v["id"].lower() in sel]

    fables, fable_to_moral = load_base_data()
    if args.sample:
        fables = fables[:args.sample]

    # Load existing corpus if resuming
    corpus_by_key: dict[str, dict] = {}
    if args.resume and Path(args.resume).exists():
        with open(args.resume) as f:
            for item in json.load(f):
                corpus_by_key[item["fable_alias"]] = item
        # Migrate legacy string entries to {"text": ..., "raw": ...} schema.
        # For entries generated before this schema change, raw == text (best we can do).
        n_migrated = 0
        for item in corpus_by_key.values():
            for model_dict in item.get("summaries", {}).values():
                for vid, val in list(model_dict.items()):
                    if isinstance(val, str):
                        model_dict[vid] = {"text": val, "raw": val}
                        n_migrated += 1
        print(f"  Resuming from {args.resume} ({len(corpus_by_key)} fables loaded"
              f"{f', {n_migrated} entries migrated to text/raw schema' if n_migrated else ''})")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"{ts}_summaries.json"
    is_sample   = bool(args.sample)

    n_total = len(models_cfg) * len(fables) * len(variants_cfg)
    print(f"\n[exp_13_gemma4_gpu_summarization]")
    print(f"  models={len(models_cfg)}  fables={len(fables)}  variants={len(variants_cfg)}")
    print(f"  total generations: {n_total}")
    if not is_sample:
        print(f"  output: {output_path}")

    notify.send(
        f"🦙 exp_13 generate starting\n"
        f"models: {', '.join(m['alias'] for m in models_cfg)}\n"
        f"variants: {', '.join(v['id'] for v in variants_cfg)}\n"
        f"fables: {len(fables)}"
    )

    for model_cfg in models_cfg:
        alias = model_cfg["alias"]
        print(f"\n{'─'*60}")
        print(f"  Loading {alias} ({model_cfg['id']}) …")

        try:
            model, processor = load_model_and_processor(model_cfg)
        except Exception as e:
            print(f"  ✗ Failed to load {alias}: {e}")
            notify.send(f"exp_13 ✗ load failed: {alias}\n{str(e)[:200]}")
            continue

        max_tokens          = model_cfg.get("max_new_tokens", 256)
        max_tokens_thinking = model_cfg.get("max_new_tokens_thinking", 4096)

        for i, fable in enumerate(fables):
            key = _fable_key(fable)

            # Initialise corpus entry if not resuming
            if key not in corpus_by_key:
                fable_idx = int(fable["doc_id"].split("_")[1])
                corpus_by_key[key] = {
                    "fable_id":           f"item_{fable_idx:03d}",
                    "fable_alias":        key,
                    "fable_text":         fable["text"],
                    "ground_truth_moral": fable_to_moral.get(fable_idx, ""),
                    "summaries":          {},
                }

            item = corpus_by_key[key]
            item["summaries"].setdefault(alias, {})

            pending = [v for v in variants_cfg if v["id"] not in item["summaries"][alias]]
            if not pending:
                continue

            if is_sample:
                print(f"\n  [{i+1}/{len(fables)}] {fable.get('title', key)}")
            else:
                print(f"  [{i+1:>3}/{len(fables)}] {fable.get('title', key)[:60]}")

            try:
                batch = generate_batch(
                    model, processor, fable["text"],
                    pending, max_tokens, max_tokens_thinking,
                )
                for vid, result in batch.items():
                    item["summaries"][alias][vid] = {"text": result["text"], "raw": result["raw"]}
                    preview = result["text"][:80] + ("…" if len(result["text"]) > 80 else "")
                    print(f"    {vid}: {preview}"
                          f"  [{result['input_tokens']}in/{result['output_tokens']}out]")
            except Exception as e:
                print(f"    ✗ {alias}|batch: {e}")
                err = f"[ERROR: {str(e)[:100]}]"
                for vcfg in pending:
                    item["summaries"][alias][vcfg["id"]] = {"text": err, "raw": err}

            # Checkpoint every 20 fables
            if not is_sample and (i + 1) % 20 == 0:
                _save(corpus_by_key, output_path)
                print(f"  [checkpoint] {len(corpus_by_key)} fables → {output_path}")

        notify.send(f"exp_13 ✓ {alias} done ({len(fables)} fables)")

        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if not is_sample:
        _save(corpus_by_key, output_path)
        n_done = len(corpus_by_key)
        print(f"\n  Done! {n_done} fables → {output_path}")
        notify.send(f"✅ exp_13 generation done\n{n_done} fables saved")
    else:
        print(f"\n  Sample run complete (not saved).")


if __name__ == "__main__":
    main()
