"""lib/pipeline/corpus_generator.py — Generate style-matched summaries for fable corpus."""
import json
import time
from pathlib import Path

from lib.pipeline import llm_client as _llm


_OUTPUT_FILE = "corpus_summaries.json"
_TOKEN_FILE = "token_usage.json"


def generate_corpus_summaries(
    client,
    fables: list[dict],
    variants: list[dict],
    model_id: str,
    run_dir: Path,
    delay: float = 0.5,
    force: bool = False,
) -> Path:
    """
    Generate corpus summaries for each fable × variant combination.

    Args:
        client:   Gemini client from llm_client.create_client()
        fables:   List of fable dicts with keys: doc_id, title, text
        variants: Resolved variant dicts each with: name, system_prompt, user_prompt_template
        model_id: Gemini model identifier string
        run_dir:  Directory to write corpus_summaries.json and token_usage.json
        delay:    Seconds to sleep between API calls (rate limiting)
        force:    Re-generate even if output already exists

    Returns:
        Path to written corpus_summaries.json
    """
    run_dir = Path(run_dir)
    output_path = run_dir / _OUTPUT_FILE

    if output_path.exists() and not force:
        print(f"  [skip] {output_path.name} already exists (use force=True to regenerate)")
        return output_path

    print(f"\n  Corpus generation: {len(fables)} fables x {len(variants)} variants  |  model: {model_id}")

    corpus = []
    total_input = total_output = total_thinking = 0

    for i, fable in enumerate(fables):
        fable_idx = int(fable["doc_id"].split("_")[1])
        item = {
            "id": f"item_{fable_idx:03d}",
            "original_fable_id": fable.get("alias", fable["doc_id"]),
            "fable_text": fable["text"],
            "summaries": {},
            "token_usage": {},
            "metadata": {
                "source": fable.get("alias", "unknown").split("_")[0],
                "word_count_fable": len(fable["text"].split()),
                "model": model_id,
            },
        }

        print(f"\n  [{i+1}/{len(fables)}] {fable.get('title', item['id'])}")

        for variant in variants:
            user_prompt = variant["user_prompt_template"].format(text=fable["text"])
            result = _llm.call(client, model_id, variant["system_prompt"], user_prompt)
            item["summaries"][variant["name"]] = result["text"]
            item["token_usage"][variant["name"]] = {
                k: result[k]
                for k in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens")
            }
            total_input += result["input_tokens"]
            total_output += result["output_tokens"]
            total_thinking += result["thinking_tokens"]
            short = result["text"][:80] + ("..." if len(result["text"]) > 80 else "")
            print(f"    {variant['name']}: {short}  [{result['input_tokens']}in/{result['output_tokens']}out]")
            time.sleep(delay)

        corpus.append(item)

    with open(output_path, "w") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    token_summary = {
        "model": model_id,
        "n_fables": len(corpus),
        "variants": [v["name"] for v in variants],
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_thinking_tokens": total_thinking,
        "total_tokens": total_input + total_thinking + total_output,
    }
    with open(run_dir / _TOKEN_FILE, "w") as f:
        json.dump(token_summary, f, indent=2)

    print(f"\n  Saved {len(corpus)} items to {output_path}")
    return output_path
