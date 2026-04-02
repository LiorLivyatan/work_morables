"""
generate_query_expansions.py — Generate paraphrases for moral query expansion.

For each of the first N fables, finds its ground-truth moral and generates 3
paraphrase variants:
  - moral_rephrase:   Same meaning, different wording (≤15 words)
  - moral_elaborate:  Broader principle (≤20 words, abstract only)
  - moral_abstract:   Most concise form (≤10 words)

Output: results/generation_runs/<timestamp>_sample<N>/query_expansions.json
        (saved in the most recent run_dir, alongside corpus_summaries.json)

Usage:
  python experiments/08_symmetric_moral_matching/generate_query_expansions.py
  python experiments/08_symmetric_moral_matching/generate_query_expansions.py --sample 10
  python experiments/08_symmetric_moral_matching/generate_query_expansions.py --run-dir path/to/run_dir
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
PARAPHRASE_PROMPTS = {
    "moral_rephrase": (
        "You are an expert in rephrasing moral statements. "
        "Given a moral from a fable, rephrase it using different words while preserving "
        "the exact same meaning. Output a single sentence of at most 15 words. "
        "Do not use character names or narrative description. Abstract and universal only."
    ),
    "moral_elaborate": (
        "You are an expert in moral philosophy. "
        "Given a moral from a fable, broaden it slightly to express the same principle "
        "in a wider context. Output a single sentence of at most 20 words. "
        "Keep it abstract and universal — no character names, no narrative examples."
    ),
    "moral_abstract": (
        "You are an expert in distilling principles to their essence. "
        "Given a moral from a fable, strip it to its most concise and abstract form. "
        "Output a single sentence of at most 10 words."
    ),
}

USER_PROMPT_TEMPLATE = "Moral: {moral}"


# ── Gemini API ───────────────────────────────────────────────────────────────

def create_client():
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment or .env")
    return genai.Client(api_key=api_key)


def generate_paraphrase(client, moral_text: str, variant: str, model_id: str,
                        max_retries: int = 5) -> dict:
    """Call Gemini with exponential backoff. Returns {text, input_tokens, output_tokens,
    thinking_tokens, total_tokens}."""
    system_prompt = PARAPHRASE_PROMPTS[variant]
    user_prompt = USER_PROMPT_TEMPLATE.format(moral=moral_text)

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


def find_latest_run_dir() -> Path:
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"No run dirs at {RUNS_DIR}. Run generate_corpus_summaries.py first.")
    run_dirs = sorted(RUNS_DIR.iterdir())
    if not run_dirs:
        raise FileNotFoundError("No run dirs found.")
    return run_dirs[-1]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate moral query paraphrases")
    parser.add_argument("--sample", type=int, default=None,
                        help="Only process morals for first N fables.")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Run dir to save into (default: latest).")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    args = parser.parse_args()

    fables = load_fables()
    morals = load_morals()
    gt_m2f = load_qrels_moral_to_fable()  # {moral_idx: fable_idx}

    # Determine which fable indices to include
    n_fables = args.sample if args.sample else len(fables)
    target_fable_indices = set(range(n_fables))

    # Find morals pointing to those fables, ordered by moral index
    moral_entries = sorted(
        [(moral_idx, fable_idx) for moral_idx, fable_idx in gt_m2f.items()
         if fable_idx in target_fable_indices],
        key=lambda x: x[0]
    )

    run_dir = args.run_dir or find_latest_run_dir()
    output_path = run_dir / "query_expansions.json"

    print(f"\n08_symmetric_moral_matching — Query Expansion Generation")
    print(f"  Model: {args.model}")
    print(f"  Morals to expand: {len(moral_entries)}")
    print(f"  Run dir: {run_dir}")

    client = create_client()
    expansions = []
    total_input = total_output = total_thinking = 0

    for i, (moral_idx, fable_idx) in enumerate(moral_entries):
        moral_text = morals[moral_idx]["text"]
        item_id = f"moral_{moral_idx:03d}"

        item = {
            "id": item_id,
            "moral_idx": moral_idx,
            "fable_idx": fable_idx,
            "original_moral": moral_text,
            "paraphrases": {},
            "token_usage": {},
        }

        print(f"\n  [{i+1}/{len(moral_entries)}] {item_id}: {moral_text[:60]}")

        for variant in PARAPHRASE_PROMPTS:
            result = generate_paraphrase(client, moral_text, variant, args.model)
            item["paraphrases"][variant] = result["text"]
            item["token_usage"][variant] = {
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "thinking_tokens": result["thinking_tokens"],
                "total_tokens": result["total_tokens"],
            }
            total_input += result["input_tokens"]
            total_output += result["output_tokens"]
            total_thinking += result["thinking_tokens"]
            print(f"    {variant}: {result['text'][:70]}"
                  f"{'...' if len(result['text']) > 70 else ''}")
            time.sleep(args.delay)

        expansions.append(item)

    with open(output_path, "w") as f:
        json.dump(expansions, f, indent=2, ensure_ascii=False)

    print(f"\n  Done! Saved {len(expansions)} moral expansions to {output_path}")
    print(f"  Total tokens: {total_input:,} input + {total_thinking:,} thinking + "
          f"{total_output:,} output")


if __name__ == "__main__":
    main()
