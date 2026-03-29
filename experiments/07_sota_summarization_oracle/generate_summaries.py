"""
generate_summaries.py — Generate golden moral summaries using Gemini 3.1 Pro.

Calls Gemini 3.1 Pro with 3 distinct prompt styles for each of 709 fables,
producing a self-contained golden_summaries.json corpus.

Prompt variants:
  - direct_moral:          Direct one-sentence moral extraction
  - narrative_distillation: Summarize the narrative, then distill the lesson
  - conceptual_abstract:   Chain-of-thought reasoning to extract abstract concept

Usage:
  python experiments/07_sota_summarization_oracle/generate_summaries.py
  python experiments/07_sota_summarization_oracle/generate_summaries.py --variants direct_moral
  python experiments/07_sota_summarization_oracle/generate_summaries.py --sample 10  # quick test
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "lib"))
from data import load_fables, load_morals, load_qrels_moral_to_fable

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
RUNS_DIR = RESULTS_DIR / "generation_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load .env ────────────────────────────────────────────────────────────────
_env = ROOT_DIR / ".env"
if _env.exists():
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.split("#")[0].strip()
                os.environ.setdefault(k.strip(), v)

# ── Gemini model config ──────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "gemini-3.1-pro-preview"

# ── System prompts ───────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "direct_moral": (
        "You are an expert in fables, parables, and moral philosophy. "
        "When given a fable, extract its moral lesson as a single sentence. "
        "Be as concise as possible."
    ),
    "narrative_distillation": (
        "You are an expert literary analyst. When given a fable, first mentally "
        "summarize what happens in the story, then distill the core lesson into "
        "a single sentence. Focus on what the story illustrates about human nature "
        "or behavior. Be as concise as possible."
    ),
    "conceptual_abstract": (
        "You are a moral philosopher. When given a fable, reason step by step:\n"
        "1. What is the central conflict or situation?\n"
        "2. What does the outcome reveal about human nature?\n"
        "3. What abstract principle does this illustrate?\n\n"
        "After reasoning, output ONLY the abstract moral principle as a single "
        "sentence on the last line. Do NOT include your reasoning in the output. "
        "Be as concise as possible."
    ),
    "proverb": (
        "You are an expert in fables and proverbs. "
        "When given a fable, state its moral as a proverb or maxim. "
        "Be as concise as possible."
    ),
    "universal_truth": (
        "You are an expert in fables and moral philosophy. "
        "When given a fable, state the moral as a universal truth about human nature. "
        "Use a declarative statement, not advice. "
        "Be as concise as possible."
    ),
    "few_shot_proverb": (
        "You are an expert in fables. When given a fable, state its moral as a short proverb.\n\n"
        "Examples:\n"
        "Fable about a tortoise beating a hare in a race -> Slow and steady wins the race.\n"
        "Fable about a fox calling unreachable grapes sour -> It is easy to despise what you cannot get.\n"
        "Fable about a boy who cried wolf and was not believed -> A liar will not be believed, even when telling the truth.\n\n"
        "Now state the moral of the following fable as a proverb. Be as concise as possible."
    ),
    "cot_proverb": (
        "You are an expert in fables and proverbs. When given a fable, reason step by step:\n"
        "1. What is the central conflict or situation?\n"
        "2. What does the outcome reveal about human nature?\n"
        "3. What abstract principle does this illustrate?\n\n"
        "After reasoning, state the moral as a proverb or maxim. "
        "Output ONLY the proverb on the last line. Do NOT include your reasoning in the output. "
        "Be as concise as possible."
    ),
    "cot_few_shot_proverb": (
        "You are an expert in fables and proverbs. When given a fable, reason step by step:\n"
        "1. What is the central conflict or situation?\n"
        "2. What does the outcome reveal about human nature?\n"
        "3. What abstract principle does this illustrate?\n\n"
        "After reasoning, state the moral as a short proverb. "
        "Output ONLY the proverb on the last line. Do NOT include your reasoning in the output.\n\n"
        "Examples of the style expected:\n"
        "Fable about a tortoise beating a hare in a race -> Slow and steady wins the race.\n"
        "Fable about a fox calling unreachable grapes sour -> It is easy to despise what you cannot get.\n"
        "Fable about a boy who cried wolf and was not believed -> A liar will not be believed, even when telling the truth.\n\n"
        "Be as concise as possible."
    ),
}

USER_PROMPT_TEMPLATE = "Fable: {fable}"


# ── Gemini API ───────────────────────────────────────────────────────────────

def create_client():
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment or .env")
    return genai.Client(api_key=api_key)


def generate_summary(client, fable_text: str, variant: str, model_id: str,
                     max_retries: int = 5) -> dict:
    """
    Call Gemini 3.1 Pro with exponential backoff on rate limits.

    Returns dict with keys: text, input_tokens, output_tokens, total_tokens.
    Raw model output is returned as-is — no post-processing or cleanup.
    """
    system_prompt = SYSTEM_PROMPTS[variant]
    user_prompt = USER_PROMPT_TEMPLATE.format(fable=fable_text)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=user_prompt,
                config={
                    "system_instruction": system_prompt,
                },
            )
            text = response.text.strip()

            # Extract token usage from response (includes thinking tokens)
            usage = response.usage_metadata
            token_info = {
                "input_tokens": usage.prompt_token_count if usage else 0,
                "output_tokens": usage.candidates_token_count if usage else 0,
                "thinking_tokens": (usage.thoughts_token_count or 0) if usage else 0,
                "total_tokens": usage.total_token_count if usage else 0,
            }

            return {"text": text, **token_info}

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str or "quota" in err_str:
                wait = 2 ** attempt + 1
                print(f"    Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"    Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return {"text": f"[ERROR: {e}]",
                            "input_tokens": 0, "output_tokens": 0,
                            "thinking_tokens": 0, "total_tokens": 0}
                time.sleep(2)

    return {"text": "[ERROR: max retries exceeded]",
            "input_tokens": 0, "output_tokens": 0,
            "thinking_tokens": 0, "total_tokens": 0}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate golden summaries with Gemini 3.1 Pro")
    parser.add_argument(
        "--variants", nargs="+",
        choices=list(SYSTEM_PROMPTS.keys()),
        default=list(SYSTEM_PROMPTS.keys()),
        help="Which prompt variants to run (default: all).",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Only process first N fables (for quick testing).",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing golden_summaries.json, skipping already-generated variants.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_ID,
        help=f"Gemini model ID (default: {DEFAULT_MODEL_ID}).",
    )
    args = parser.parse_args()
    model_id = args.model

    # Load data
    fables = load_fables()
    morals = load_morals()
    gt_m2f = load_qrels_moral_to_fable()
    fable_to_moral = {fable_idx: morals[moral_idx]["text"]
                      for moral_idx, fable_idx in gt_m2f.items()}

    if args.sample:
        fables = fables[:args.sample]

    # Create timestamped run directory
    sample_tag = f"_sample{args.sample}" if args.sample else ""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / f"{ts}{sample_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "golden_summaries.json"

    print(f"\n07_sota_summarization_oracle — Golden Corpus Generation")
    print(f"  Model: {model_id}")
    print(f"  Fables: {len(fables)}")
    print(f"  Variants: {args.variants}")
    print(f"  Delay: {args.delay}s between calls")
    print(f"  Run dir: {run_dir}")

    # Resume from existing file if requested
    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing_list = json.load(f)
        existing = {item["id"]: item for item in existing_list}
        print(f"  Resuming: {len(existing)} fables already in corpus")

    # Initialize Gemini client
    client = create_client()

    # Generate summaries
    corpus = []
    total_input = 0
    total_output = 0
    total_thinking = 0
    for i, fable in enumerate(fables):
        fable_idx = int(fable["doc_id"].split("_")[1])
        item_id = f"item_{fable_idx:03d}"

        # Check if we can reuse existing data
        if item_id in existing:
            item = existing[item_id]
            # Check which variants still need generation
            missing_variants = [v for v in args.variants
                                if v not in item.get("summaries", {})]
            if not missing_variants:
                corpus.append(item)
                continue
        else:
            item = {
                "id": item_id,
                "original_fable_id": fable.get("alias", fable["doc_id"]),
                "fable_text": fable["text"],
                "ground_truth_moral": fable_to_moral.get(fable_idx, ""),
                "summaries": {},
                "token_usage": {},
                "metadata": {
                    "source": fable.get("alias", "unknown").split("_")[0],
                    "word_count_fable": len(fable["text"].split()),
                    "model": model_id,
                },
            }
            missing_variants = args.variants

        print(f"\n  [{i + 1}/{len(fables)}] {fable['title']} ({item_id})")

        for variant in missing_variants:
            result = generate_summary(client, fable["text"], variant, model_id)
            item["summaries"][variant] = result["text"]
            item.setdefault("token_usage", {})[variant] = {
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "thinking_tokens": result["thinking_tokens"],
                "total_tokens": result["total_tokens"],
            }
            total_input += result["input_tokens"]
            total_output += result["output_tokens"]
            total_thinking += result["thinking_tokens"]
            print(f"    {variant}: {result['text'][:80]}{'...' if len(result['text']) > 80 else ''}"
                  f"  [{result['input_tokens']}in/{result['thinking_tokens']}think/{result['output_tokens']}out]")
            time.sleep(args.delay)

        corpus.append(item)

        # Save incrementally every 50 fables
        if (i + 1) % 50 == 0:
            with open(output_path, "w") as f:
                json.dump(corpus, f, indent=2, ensure_ascii=False)
            print(f"  [checkpoint] Saved {len(corpus)} fables to {output_path}")

    # Final save
    with open(output_path, "w") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    print(f"\n  Done! Saved {len(corpus)} fables to {output_path}")

    # Quick stats per variant
    for variant in args.variants:
        summaries = [item["summaries"].get(variant, "") for item in corpus]
        avg_words = sum(len(s.split()) for s in summaries) / len(summaries)
        errors = sum(1 for s in summaries if s.startswith("[ERROR"))
        v_input = sum(item.get("token_usage", {}).get(variant, {}).get("input_tokens", 0)
                      for item in corpus)
        v_output = sum(item.get("token_usage", {}).get(variant, {}).get("output_tokens", 0)
                       for item in corpus)
        print(f"  {variant}: avg {avg_words:.1f} words, {errors} errors, "
              f"{v_input:,} input tokens, {v_output:,} output tokens")

    # Total token usage summary
    print(f"\n  Total tokens: {total_input:,} input + {total_thinking:,} thinking + "
          f"{total_output:,} output = {total_input + total_thinking + total_output:,} total")

    # Save token usage summary alongside the corpus
    usage_summary = {
        "model": model_id,
        "n_fables": len(corpus),
        "variants": args.variants,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_thinking_tokens": total_thinking,
        "total_tokens": total_input + total_thinking + total_output,
    }
    with open(run_dir / "token_usage.json", "w") as f:
        json.dump(usage_summary, f, indent=2)

    print(f"  Run saved to {run_dir}")


if __name__ == "__main__":
    main()
