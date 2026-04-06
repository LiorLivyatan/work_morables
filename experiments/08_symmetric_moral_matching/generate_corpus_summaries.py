"""
generate_corpus_summaries.py — Generate style-matched moral summaries for fable corpus.

Calls Gemini with 2 prompt styles designed to match ground-truth moral style:
  - ground_truth_style:     Short aphorism (5-15 words), few-shot grounded
  - declarative_universal:  Declarative universal truth (5-15 words), few-shot grounded

Output: results/generation_runs/<timestamp>_sample<N>/corpus_summaries.json

Usage:
  python experiments/08_symmetric_moral_matching/generate_corpus_summaries.py
  python experiments/08_symmetric_moral_matching/generate_corpus_summaries.py --sample 10
  python experiments/08_symmetric_moral_matching/generate_corpus_summaries.py --variants ground_truth_style
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
DEFAULT_MODEL_ID = "gemini-3-flash-preview"

# ── System prompts ───────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "ground_truth_style": (
        "You are an expert in fables. When given a fable, state its moral as a concise "
        "aphorism of 5 to 15 words. Use no character names. Be abstract and universal.\n\n"
        "Examples of the exact style required:\n"
        "- Appearances are deceptive.\n"
        "- Vices are their own punishment.\n"
        "- An ounce of prevention is worth a pound of cure.\n"
        "- Gratitude is the sign of noble souls.\n"
        "- Misfortune tests the sincerity of friends.\n\n"
        "Output ONLY the moral. No explanation, no narrative description."
    ),
    "declarative_universal": (
        "You are an expert in moral philosophy. When given a fable, distill its lesson "
        "into one declarative sentence of 5 to 15 words. The statement must be universal "
        "and timeless — no character names, no reference to the story's events. State an "
        "observation about human nature or behavior.\n\n"
        "Examples of the exact style required:\n"
        "- Those who envy others invite their own misfortune.\n"
        "- Necessity drives men to find solutions.\n"
        "- He who is content with little needs nothing more.\n\n"
        "Output ONLY the moral sentence. No explanation."
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
    """Call Gemini with exponential backoff. Returns {text, input_tokens, output_tokens,
    thinking_tokens, total_tokens}."""
    system_prompt = SYSTEM_PROMPTS[variant]
    user_prompt = USER_PROMPT_TEMPLATE.format(fable=fable_text)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=user_prompt,
                config={"system_instruction": system_prompt},
            )
            text = response.text.strip()
            usage = response.usage_metadata
            return {
                "text": text,
                "input_tokens": usage.prompt_token_count if usage else 0,
                "output_tokens": usage.candidates_token_count if usage else 0,
                "thinking_tokens": (usage.thoughts_token_count or 0) if usage else 0,
                "total_tokens": usage.total_token_count if usage else 0,
            }
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str or "quota" in err_str:
                wait = 2 ** attempt + 1
                print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"    Error on attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    return {"text": f"[ERROR: {e}]", "input_tokens": 0,
                            "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}
                time.sleep(2)

    return {"text": "[ERROR: max retries exceeded]", "input_tokens": 0,
            "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate style-matched corpus summaries")
    parser.add_argument("--variants", nargs="+", choices=list(SYSTEM_PROMPTS.keys()),
                        default=list(SYSTEM_PROMPTS.keys()))
    parser.add_argument("--sample", type=int, default=None,
                        help="Only process first N fables (default: all).")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API calls in seconds (default: 0.5).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    args = parser.parse_args()

    fables = load_fables()
    morals = load_morals()
    gt_m2f = load_qrels_moral_to_fable()
    fable_to_moral = {fable_idx: morals[moral_idx]["text"]
                      for moral_idx, fable_idx in gt_m2f.items()}

    if args.sample:
        fables = fables[:args.sample]

    range_tag = f"_sample{args.sample}" if args.sample else ""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / f"{ts}{range_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "corpus_summaries.json"

    print(f"\n08_symmetric_moral_matching — Corpus Summary Generation")
    print(f"  Model: {args.model}")
    print(f"  Fables: {len(fables)}")
    print(f"  Variants: {args.variants}")
    print(f"  Run dir: {run_dir}")

    client = create_client()
    corpus = []
    total_input = total_output = total_thinking = 0

    for i, fable in enumerate(fables):
        fable_idx = int(fable["doc_id"].split("_")[1])
        item_id = f"item_{fable_idx:03d}"
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
                "model": args.model,
            },
        }

        print(f"\n  [{i+1}/{len(fables)}] {fable['title']} ({item_id})")

        for variant in args.variants:
            result = generate_summary(client, fable["text"], variant, args.model)
            item["summaries"][variant] = result["text"]
            item["token_usage"][variant] = {
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "thinking_tokens": result["thinking_tokens"],
                "total_tokens": result["total_tokens"],
            }
            total_input += result["input_tokens"]
            total_output += result["output_tokens"]
            total_thinking += result["thinking_tokens"]
            print(f"    {variant}: {result['text'][:80]}"
                  f"{'...' if len(result['text']) > 80 else ''}"
                  f"  [{result['input_tokens']}in/{result['thinking_tokens']}think/{result['output_tokens']}out]")
            time.sleep(args.delay)

        corpus.append(item)

    with open(output_path, "w") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    print(f"\n  Done! Saved {len(corpus)} items to {output_path}")
    for variant in args.variants:
        summaries = [item["summaries"].get(variant, "") for item in corpus]
        avg_words = sum(len(s.split()) for s in summaries) / max(len(summaries), 1)
        errors = sum(1 for s in summaries if s.startswith("[ERROR"))
        print(f"  {variant}: avg {avg_words:.1f} words, {errors} errors")

    print(f"\n  Total tokens: {total_input:,} input + {total_thinking:,} thinking + "
          f"{total_output:,} output")

    usage_summary = {
        "model": args.model,
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
