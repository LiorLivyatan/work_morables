"""
generate_summaries.py — Generate golden moral summaries using Gemma 4 (MLX).

Runs Gemma 4 instruct models locally via mlx_lm for each of 709 fables ×
prompt variants, producing a golden_summaries.json corpus compatible with run.py.

Usage:
  # Quick smoke test — 3 fables, no save
  python experiments/09_gemma4_summarization/generate_summaries.py --sample 3

  # Single model, all variants
  python experiments/09_gemma4_summarization/generate_summaries.py --models gemma4-e2b

  # All models (runs sequentially, unloads between)
  python experiments/09_gemma4_summarization/generate_summaries.py

  # Resume interrupted run
  python experiments/09_gemma4_summarization/generate_summaries.py --models gemma4-e4b --resume
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "lib"))
from data import load_fables, load_morals, load_qrels_moral_to_fable  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"
RUNS_DIR = RESULTS_DIR / "generation_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── Models ───────────────────────────────────────────────────────────────────

GEMMA_MODELS = {
    "gemma4-e2b": "mlx-community/gemma-4-e2b-it-4bit",
    "gemma4-e4b": "mlx-community/gemma-4-e4b-it-4bit",
    "gemma4-31b": "mlx-community/gemma-4-31b-it-4bit",
}

# ── Prompt variants ───────────────────────────────────────────────────────────

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
}

USER_PROMPT_TEMPLATE = "Fable: {fable}"


# ── Pure functions (tested) ───────────────────────────────────────────────────

def postprocess_summary(text: str, variant: str) -> str:
    """Strip artefacts and extract the relevant line from model output.

    For direct_moral: returns the first non-empty line (model answers directly).
    For narrative_distillation / conceptual_abstract: returns the last non-empty
    line — both prompts ask the model to reason/summarize first, then give the
    lesson, so the moral is at the end.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown bold markers (e.g. **Summary:** -> Summary:)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return text
    return lines[0] if variant == "direct_moral" else lines[-1]


def build_corpus_item(fable_idx: int, fable: dict, ground_truth_moral: str,
                      summaries: dict, model_id: str) -> dict:
    """Build a corpus item matching Exp 07's golden_summaries.json schema."""
    alias = fable.get("alias", fable["doc_id"])
    return {
        "id": f"item_{fable_idx:03d}",
        "original_fable_id": alias,
        "fable_text": fable["text"],
        "ground_truth_moral": ground_truth_moral,
        "summaries": summaries,
        "metadata": {
            "source": alias.split("_")[0],
            "word_count_fable": len(fable["text"].split()),
            "model": model_id,
        },
    }


# ── Model I/O ─────────────────────────────────────────────────────────────────

def load_model(model_id: str):
    from mlx_lm import load
    print(f"  Loading {model_id}...")
    model, tokenizer = load(model_id)
    return model, tokenizer


def unload_model(model, tokenizer):
    import mlx.core as mx
    del model, tokenizer
    mx.clear_cache()


def generate_summary(model, tokenizer, fable_text: str, variant: str) -> str:
    """Generate a one-sentence moral summary for a single fable."""
    from mlx_lm import generate

    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[variant]},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(fable=fable_text)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=150, verbose=False)
    return postprocess_summary(raw, variant)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_generation(model_labels: list[str], variant_labels: list[str],
                   sample_size: int | None, resume: bool) -> None:
    fables = load_fables()
    morals = load_morals()
    gt_m2f = load_qrels_moral_to_fable()
    fable_to_moral = {
        fable_idx: morals[moral_idx]["text"]
        for moral_idx, fable_idx in gt_m2f.items()
    }

    fables_to_run = fables[:sample_size] if sample_size else fables

    for model_label in model_labels:
        model_id = GEMMA_MODELS[model_label]
        print(f"\n{'─' * 60}")
        print(f"Model: {model_label}  ({model_id})")
        print(f"Fables: {len(fables_to_run)}  |  Variants: {variant_labels}")
        print(f"{'─' * 60}")

        # Determine output path
        if sample_size:
            run_dir = None
            output_path = None
        elif resume:
            # Find the most recent existing run dir for this model to continue it
            existing = sorted(
                [d for d in RUNS_DIR.iterdir() if d.name.endswith(f"_{model_label}")],
                reverse=True,
            )
            if existing and (existing[0] / "golden_summaries.json").exists():
                run_dir = existing[0]
                print(f"  Resuming from: {run_dir.name}")
            else:
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                run_dir = RUNS_DIR / f"{ts}_{model_label}"
                run_dir.mkdir(parents=True, exist_ok=True)
            output_path = run_dir / "golden_summaries.json"
        else:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = RUNS_DIR / f"{ts}_{model_label}"
            run_dir.mkdir(parents=True, exist_ok=True)
            output_path = run_dir / "golden_summaries.json"

        # Resume: load existing corpus
        corpus_by_id: dict[str, dict] = {}
        if run_dir and resume and output_path.exists():
            with open(output_path) as f:
                for item in json.load(f):
                    corpus_by_id[item["id"]] = item
            print(f"  Resuming: {len(corpus_by_id)} items already generated")

        model, tokenizer = load_model(model_id)

        corpus: list[dict] = []
        for i, fable in enumerate(tqdm(fables_to_run, desc=f"  {model_label}")):
            fable_idx = int(fable["doc_id"].split("_")[1])
            item_id = f"item_{fable_idx:03d}"
            ground_truth = fable_to_moral.get(fable_idx, "")

            # Reuse existing item if resuming and all variants present
            if item_id in corpus_by_id:
                existing_item = corpus_by_id[item_id]
                missing = [v for v in variant_labels if v not in existing_item.get("summaries", {})]
                if not missing:
                    corpus.append(existing_item)
                    continue
                summaries = dict(existing_item.get("summaries", {}))
                variants_to_run = missing
            else:
                summaries = {}
                variants_to_run = variant_labels

            for variant in variants_to_run:
                summary = generate_summary(model, tokenizer, fable["text"], variant)
                summaries[variant] = summary

            item = build_corpus_item(fable_idx, fable, ground_truth, summaries, model_id)
            corpus.append(item)

            # Print sample output in sample mode
            if sample_size:
                print(f"\n  [{i + 1}] {fable.get('title', item_id)}")
                for v in variant_labels:
                    print(f"    {v}: {summaries.get(v, '')[:100]}")

            # Incremental checkpoint every 50 fables
            if output_path and (i + 1) % 50 == 0:
                with open(output_path, "w") as f:
                    json.dump(corpus, f, indent=2, ensure_ascii=False)
                print(f"  [checkpoint] {len(corpus)} items saved")

        unload_model(model, tokenizer)

        # Final save
        if output_path:
            with open(output_path, "w") as f:
                json.dump(corpus, f, indent=2, ensure_ascii=False)
            print(f"\n  Done — {len(corpus)} items saved to {output_path}")

            # Per-variant summary
            for v in variant_labels:
                texts = [item["summaries"].get(v, "") for item in corpus]
                avg_words = sum(len(t.split()) for t in texts) / max(len(texts), 1)
                errors = sum(1 for t in texts if not t or t.startswith("[ERROR"))
                print(f"  {v}: avg {avg_words:.1f} words, {errors} errors")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate golden moral summaries using Gemma 4 (MLX).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(GEMMA_MODELS.keys()),
        default=list(GEMMA_MODELS.keys()),
        help="Which Gemma 4 models to run (default: all three).",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=list(SYSTEM_PROMPTS.keys()),
        default=list(SYSTEM_PROMPTS.keys()),
        help="Which prompt variants to run (default: all three).",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Run on first N fables only and print output (no save).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume the most recent run for each model, skipping already-generated items.",
    )
    args = parser.parse_args()

    if args.sample:
        print(f"\nSample mode: {args.sample} fables per model (not saved)\n")
    else:
        total = len(args.models) * len(args.variants) * 709
        print(f"\nFull run: {len(args.models)} models × {len(args.variants)} variants × 709 fables = {total:,} calls\n")

    run_generation(args.models, args.variants, args.sample, args.resume)
