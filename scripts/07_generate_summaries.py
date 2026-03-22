"""
07_generate_summaries.py — Part 1: Generate LLM moral summaries for all fables.

For each Qwen model × prompt style combination, generates a one-sentence moral
summary for every fable and caches the results to disk.

Run this BEFORE 07_llm_experiments.py (Approach B).

Usage:
  # Quick test — generate 3 samples per combo, print and exit
  python scripts/07_generate_summaries.py --sample 3

  # Full generation for all models and prompts (~60 min)
  python scripts/07_generate_summaries.py

  # Full generation for specific models only
  python scripts/07_generate_summaries.py --models qwen2.5-3b qwen2.5-7b

  # Full generation for specific prompts only
  python scripts/07_generate_summaries.py --prompts direct few_shot
"""
# TODO: Add seeds
# TODO: add a table of different prompt/model/seed with a few examples
# Dataset / Modeling (summary/instructions/finetuning/indexing) / Models (to choose... in the future)
import argparse
import json
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

DATA_DIR        = Path(__file__).parent.parent / "data" / "processed"
SUMMARIES_CACHE = Path(__file__).parent.parent / "results" / "summaries"
SUMMARIES_CACHE.mkdir(parents=True, exist_ok=True)

# ── Qwen models ────────────────────────────────────────────────────────────────
QWEN_MODELS = {
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-2b":   "Qwen/Qwen3.5-2B",
    "qwen3.5-4b":   "Qwen/Qwen3.5-4B",
    "qwen3.5-9b":   "Qwen/Qwen3.5-9B",
}

# ── Prompt templates ───────────────────────────────────────────────────────────
SUMMARY_PROMPTS = {
    "direct": (
        "What is the moral of this fable? Write it as a single short sentence. "
        "Do NOT start with 'The moral is' or 'The lesson is' — state it directly.\n\n"
        "Fable: {fable}\n\n"
        "Moral:"
    ),
    "detailed": (
        "Read this fable and identify the deeper lesson or moral principle it teaches. "
        "Be specific about what virtue, vice, or life wisdom the story illustrates. "
        "Write one concise sentence. Do NOT use phrases like 'The moral is' or "
        "'This fable teaches' — state the principle directly as a standalone truth.\n\n"
        "Fable: {fable}\n\n"
        "Moral:"
    ),
    "cot": (
        "Given a fable, think step by step, then return ONLY the moral on the last line, without any prefix or suffix.\n\n"
        "Think through:\n"
        "1. What happens in the story?\n"
        "2. What does the main character do, and what results from it?\n"
        "3. What does the story seem to be warning against or encouraging?\n\n"
        "Fable: {fable}\n\n"
        "Moral:"
    ),
    "few_shot": (
        "Here are two examples of fables and their morals:\n\n"
        "Example 1:\n"
        "Fable: A hare laughed at a tortoise for being slow. The tortoise challenged the hare "
        "to a race. The hare, confident in his speed, stopped to rest midway. The tortoise "
        "plodded on steadily and won.\n"
        "Moral: Consistent effort overcomes natural talent when the talented grow complacent.\n\n"
        "Example 2:\n"
        "Fable: A crow held a piece of cheese in her beak. A fox told her she must have a "
        "beautiful singing voice. When she opened her beak to sing, the cheese fell to the fox.\n"
        "Moral: Vanity makes us easy prey for flatterers.\n\n"
        "Now write the moral for this fable. Follow the exact format above — "
        "a short standalone phrase, no 'The moral is...' prefix.\n\n"
        "Fable: {fable}\n\n"
        "Moral:"
    ),
}

# ── Data ───────────────────────────────────────────────────────────────────────
with open(DATA_DIR / "fables_corpus.json") as f:
    fable_texts = [x["text"] for x in json.load(f)]

print(f"Loaded {len(fable_texts)} fables")


# ── Model loading / generation ─────────────────────────────────────────────────

def load_model(model_id: str):
    from mlx_lm import load

    print(f"  Loading {model_id}...")
    model, tokenizer = load(model_id)
    return tokenizer, model


def generate_one(tokenizer, model, fable: str, prompt_template: str) -> str:
    from mlx_lm import generate

    prompt = prompt_template.format(fable=fable)
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    response = generate(model, tokenizer, prompt=text, max_tokens=80, verbose=False)

    # Strip <think> blocks
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Extract MORAL: line from CoT output
    for line in response.split("\n"):
        if line.strip().upper().startswith("MORAL:"):
            response = line.split(":", 1)[1].strip()
            break

    # Take first non-empty line
    lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
    return lines[0] if lines else response


def unload_model(model, tokenizer):
    import mlx.core as mx
    del model, tokenizer
    mx.clear_cache()


# ── Main generation logic ──────────────────────────────────────────────────────

def run_generation(model_labels, prompt_labels, sample_size, print_samples):
    for model_label in model_labels:
        model_id = QWEN_MODELS[model_label]
        print(f"\n{'─'*60}")
        print(f"Model: {model_label}  ({model_id})")
        print(f"{'─'*60}")

        tokenizer, model = load_model(model_id)

        for prompt_label in prompt_labels:
            cache_file = SUMMARIES_CACHE / model_label / f"{prompt_label}.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Check cache
            if cache_file.exists():
                with open(cache_file) as f:
                    cached = json.load(f)
                if len(cached) == len(fable_texts):
                    print(f"\n  [{prompt_label}] Already cached ({len(cached)} summaries) — skipping")
                    if print_samples:
                        _print_samples(cached, model_label, prompt_label)
                    continue

            indices = list(range(sample_size if sample_size else len(fable_texts)))
            summaries: dict[str, str] = {}

            print(f"\n  [{prompt_label}] Generating {len(indices)} summaries...")
            for i in tqdm(indices, desc=f"  {model_label}/{prompt_label}"):
                summaries[str(i)] = generate_one(
                    tokenizer, model, fable_texts[i],
                    SUMMARY_PROMPTS[prompt_label],
                )

            if not sample_size:
                # Full run — save to cache
                with open(cache_file, "w") as f:
                    json.dump(summaries, f, indent=2, ensure_ascii=False)
                print(f"  Cached {len(summaries)} summaries → {cache_file}")

            if print_samples or sample_size:
                _print_samples(summaries, model_label, prompt_label)

        unload_model(model, tokenizer)


def _print_samples(summaries: dict, model_label: str, prompt_label: str):
    print(f"\n  ── Sample summaries ({model_label} / {prompt_label}) ──")
    for i in range(min(3, len(summaries))):
        s = summaries.get(str(i), "")
        fable_preview = fable_texts[i][:80].replace("\n", " ")
        print(f"  Fable [{i}]: {fable_preview}...")
        print(f"  Summary:   {s}")
        print()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and cache LLM moral summaries for all fables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(QWEN_MODELS.keys()),
        default=list(QWEN_MODELS.keys()),
        help="Which Qwen models to run (default: all four).",
    )
    parser.add_argument(
        "--prompts", nargs="+", choices=list(SUMMARY_PROMPTS.keys()),
        default=list(SUMMARY_PROMPTS.keys()),
        help="Which prompt styles to run (default: all four).",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Generate only N summaries per combo and print them (does NOT save to cache).",
    )
    args = parser.parse_args()

    print_samples = args.sample is not None

    if args.sample:
        print(f"\nSample mode: generating {args.sample} summaries per combo (not cached)\n")
    else:
        total = len(args.models) * len(args.prompts)
        print(f"\nFull generation: {len(args.models)} models × {len(args.prompts)} prompts = {total} combos\n")

    run_generation(args.models, args.prompts, args.sample, print_samples)

    if not args.sample:
        print("\n✓ All summaries generated and cached.")
        print(f"  Location: {SUMMARIES_CACHE}")
        print("\nNext step:")
        print("  python scripts/07_llm_experiments.py --approach B")
