"""lib/pipeline/query_expander.py — Generate paraphrase expansions for moral queries."""
import json
import time
from pathlib import Path

from lib.pipeline import llm_client as _llm


_OUTPUT_FILE = "query_expansions.json"
_TOKEN_FILE = "query_expansion_token_usage.json"


def generate_query_expansions(
    client,
    moral_entries: list[tuple[int, int]],
    morals: list[dict],
    variants: list[dict],
    model_id: str,
    run_dir: Path,
    delay: float = 0.5,
    force: bool = False,
) -> Path:
    """
    Generate query expansion paraphrases for each moral x variant.

    Args:
        client:        Gemini client from llm_client.create_client()
        moral_entries: List of (moral_idx, fable_idx) tuples to process
        morals:        Full morals list from lib.data.load_morals()
        variants:      Resolved variant dicts each with: name, system_prompt, user_prompt_template
        model_id:      Gemini model identifier string
        run_dir:       Directory to write query_expansions.json
        delay:         Seconds to sleep between API calls
        force:         Re-generate even if output already exists

    Returns:
        Path to written query_expansions.json
    """
    run_dir = Path(run_dir)
    output_path = run_dir / _OUTPUT_FILE

    if output_path.exists() and not force:
        print(f"  [skip] {output_path.name} already exists (use force=True to regenerate)")
        return output_path

    print(f"\n  Query expansion: {len(moral_entries)} morals x {len(variants)} variants  |  model: {model_id}")

    expansions = []
    total_input = total_output = total_thinking = 0

    for i, (moral_idx, fable_idx) in enumerate(moral_entries):
        moral_text = morals[moral_idx]["text"]
        item = {
            "id": f"moral_{moral_idx:03d}",
            "moral_idx": moral_idx,
            "fable_idx": fable_idx,
            "original_moral": moral_text,
            "paraphrases": {},
            "token_usage": {},
        }

        print(f"\n  [{i+1}/{len(moral_entries)}] moral_{moral_idx:03d}: {moral_text[:60]}")

        for variant in variants:
            user_prompt = variant["user_prompt_template"].format(text=moral_text)
            result = _llm.call(client, model_id, variant["system_prompt"], user_prompt)
            item["paraphrases"][variant["name"]] = result["text"]
            item["token_usage"][variant["name"]] = {
                k: result[k]
                for k in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens")
            }
            total_input += result["input_tokens"]
            total_output += result["output_tokens"]
            total_thinking += result["thinking_tokens"]
            print(f"    {variant['name']}: {result['text'][:70]}")
            time.sleep(delay)

        expansions.append(item)

    with open(output_path, "w") as f:
        json.dump(expansions, f, indent=2, ensure_ascii=False)

    token_summary = {
        "model": model_id,
        "n_morals": len(expansions),
        "variants": [v["name"] for v in variants],
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_thinking_tokens": total_thinking,
        "total_tokens": total_input + total_thinking + total_output,
    }
    with open(run_dir / _TOKEN_FILE, "w") as f:
        json.dump(token_summary, f, indent=2)

    print(f"\n  Saved {len(expansions)} expansions to {output_path}")
    return output_path
