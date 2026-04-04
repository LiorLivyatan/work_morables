# Exp 09: Gemma 4 Summarization Oracle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recreate Exp 07's summarization oracle using Gemma 4 models running locally via MLX, producing a `golden_summaries.json` compatible with Exp 07's `run.py`.

**Architecture:** `generate_summaries.py` uses `mlx_lm` (same pattern as Exp 03) to run Gemma 4 instruct models on each fable × prompt variant. Output matches Exp 07's JSON schema exactly so `run.py` (copied from Exp 07) needs zero logic changes.

**Tech Stack:** Python 3.13, `mlx-lm`, `mlx.core`, existing `lib/data.py` helpers.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `experiments/09_gemma4_summarization/generate_summaries.py` | Create | MLX-LM inference loop, JSON output |
| `experiments/09_gemma4_summarization/run.py` | Create (copy+adapt) | Retrieval evaluation (Exp 07 logic) |
| `experiments/09_gemma4_summarization/tests/test_generate.py` | Create | Unit tests for pure functions |
| `experiments/09_gemma4_summarization/README.md` | Create | Experiment documentation |
| `experiments/09_gemma4_summarization/results/` | Create (dirs) | Output storage |

---

### Task 1: Create experiment scaffold and failing tests

**Files:**
- Create: `experiments/09_gemma4_summarization/tests/test_generate.py`
- Create: `experiments/09_gemma4_summarization/results/generation_runs/.gitkeep`
- Create: `experiments/09_gemma4_summarization/results/runs/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p experiments/09_gemma4_summarization/tests
mkdir -p experiments/09_gemma4_summarization/results/generation_runs
mkdir -p experiments/09_gemma4_summarization/results/runs
touch experiments/09_gemma4_summarization/results/generation_runs/.gitkeep
touch experiments/09_gemma4_summarization/results/runs/.gitkeep
```

- [ ] **Step 2: Write failing tests**

Create `experiments/09_gemma4_summarization/tests/test_generate.py`:

```python
"""Tests for generate_summaries.py pure functions.

Run from repo root:
    pytest experiments/09_gemma4_summarization/tests/test_generate.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import generate_summaries as gs


# ── postprocess_summary ──────────────────────────────────────────────────────

def test_postprocess_direct_moral_returns_first_line():
    raw = "Slow and steady wins the race.\nSome extra text."
    assert gs.postprocess_summary(raw, "direct_moral") == "Slow and steady wins the race."


def test_postprocess_conceptual_abstract_returns_last_line():
    raw = "1. The hare was overconfident.\n2. The tortoise persisted.\nPersistence overcomes arrogance."
    assert gs.postprocess_summary(raw, "conceptual_abstract") == "Persistence overcomes arrogance."


def test_postprocess_strips_think_blocks():
    raw = "<think>some reasoning</think>Vanity blinds us to real danger."
    assert gs.postprocess_summary(raw, "direct_moral") == "Vanity blinds us to real danger."


def test_postprocess_handles_empty_lines():
    raw = "\n\nThe strong should not mock the weak.\n\n"
    assert gs.postprocess_summary(raw, "narrative_distillation") == "The strong should not mock the weak."


def test_postprocess_multiline_think_block():
    raw = "<think>\nstep 1\nstep 2\n</think>\nKindness is repaid in kind."
    assert gs.postprocess_summary(raw, "direct_moral") == "Kindness is repaid in kind."


# ── build_corpus_item ────────────────────────────────────────────────────────

def test_build_corpus_item_schema():
    fable = {"doc_id": "fable_001", "alias": "aesop_001", "text": "A fox and a crow..."}
    summaries = {"direct_moral": "Flattery is dangerous.", "narrative_distillation": "Watch for flatterers."}
    item = gs.build_corpus_item(1, fable, "Do not trust flatterers.", summaries, "mlx-community/gemma-4-e2b-it-4bit")

    assert item["id"] == "item_001"
    assert item["original_fable_id"] == "aesop_001"
    assert item["fable_text"] == "A fox and a crow..."
    assert item["ground_truth_moral"] == "Do not trust flatterers."
    assert item["summaries"] == summaries
    assert item["metadata"]["model"] == "mlx-community/gemma-4-e2b-it-4bit"
    assert item["metadata"]["source"] == "aesop"
    assert item["metadata"]["word_count_fable"] == 5


def test_build_corpus_item_id_zero_padded():
    fable = {"doc_id": "fable_007", "text": "Short fable."}
    item = gs.build_corpus_item(7, fable, "moral", {}, "model-id")
    assert item["id"] == "item_007"


def test_build_corpus_item_fallback_when_no_alias():
    fable = {"doc_id": "fable_042", "text": "Another fable."}
    item = gs.build_corpus_item(42, fable, "moral", {}, "model-id")
    assert item["original_fable_id"] == "fable_042"
    assert item["metadata"]["source"] == "fable"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest experiments/09_gemma4_summarization/tests/test_generate.py -v
```

Expected: `ModuleNotFoundError: No module named 'generate_summaries'` (file doesn't exist yet)

- [ ] **Step 4: Commit scaffold**

```bash
git add experiments/09_gemma4_summarization/
git commit -m "exp09: add tests and scaffold for Gemma 4 summarization experiment"
```

---

### Task 2: Implement generate_summaries.py

**Files:**
- Create: `experiments/09_gemma4_summarization/generate_summaries.py`

- [ ] **Step 1: Write the implementation**

Create `experiments/09_gemma4_summarization/generate_summaries.py`:

```python
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

    For conceptual_abstract (CoT): returns the last non-empty line (the principle).
    For other variants: returns the first non-empty line.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return text
    return lines[-1] if variant == "conceptual_abstract" else lines[0]


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
            run_dir = None  # sample mode: no save
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
                existing = corpus_by_id[item_id]
                missing = [v for v in variant_labels if v not in existing.get("summaries", {})]
                if not missing:
                    corpus.append(existing)
                    continue
                # Generate only missing variants
                summaries = dict(existing.get("summaries", {}))
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
```

- [ ] **Step 2: Run the tests**

```bash
pytest experiments/09_gemma4_summarization/tests/test_generate.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add experiments/09_gemma4_summarization/generate_summaries.py
git commit -m "exp09: implement generate_summaries.py with MLX-LM Gemma 4 backend"
```

---

### Task 3: Create run.py (adapted from Exp 07)

**Files:**
- Create: `experiments/09_gemma4_summarization/run.py`

- [ ] **Step 1: Copy and adapt run.py from Exp 07**

Create `experiments/09_gemma4_summarization/run.py` — identical to `experiments/07_sota_summarization_oracle/run.py` with these two changes only:

1. Line ~28: Change `RESULTS_DIR = Path(__file__).parent / "results"` (already the same pattern — no change needed since it uses `Path(__file__).parent`)
2. Line ~128: Change the print header from `"07_sota_summarization_oracle"` to `"09_gemma4_summarization"`.

Full file content (copy of Exp 07 with header string updated):

```python
"""
run.py — Evaluate Gemma 4 golden summaries for moral-to-fable retrieval.

Loads golden_summaries.json (generated by generate_summaries.py) and evaluates
retrieval using Linq-Embed-Mistral on multiple configurations:

  Config A: Summary only        — embed {summary} as corpus
  Config B: Fable + Summary     — embed "{fable}\\n\\nMoral summary: {summary}"
  Config C: Fable + Instruction — embed "Fable: {fable}" (prefix baseline)

Usage:
  python experiments/09_gemma4_summarization/run.py
  python experiments/09_gemma4_summarization/run.py --configs A B
  python experiments/09_gemma4_summarization/run.py --variants direct_moral
  python experiments/09_gemma4_summarization/run.py --summaries-path path/to/golden_summaries.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "lib"))
from data import load_moral_to_fable_retrieval_data
from retrieval_utils import compute_metrics

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"
GENERATION_RUNS_DIR = RESULTS_DIR / "generation_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── Embedding model (best baseline) ─────────────────────────────────────────
EMBED_MODEL_ID = "Linq-AI-Research/Linq-Embed-Mistral"
QUERY_INSTRUCTION = "Given a text, retrieve the most relevant passage that answers the query"

# ── Retrieval configs ────────────────────────────────────────────────────────
CONFIGS = {
    "A": {
        "name": "Summary only",
        "template": "{summary}",
    },
    "B": {
        "name": "Fable + Summary",
        "template": "{fable}\n\nMoral summary: {summary}",
    },
    "C": {
        "name": "Fable + Instruction",
        "template": "Fable: {fable}",
        "no_summary": True,
    },
}


def find_latest_summaries() -> Path:
    if not GENERATION_RUNS_DIR.exists():
        raise FileNotFoundError(
            f"No generation runs found at {GENERATION_RUNS_DIR}.\n"
            f"Run generate_summaries.py first."
        )
    run_dirs = sorted(GENERATION_RUNS_DIR.iterdir())
    if not run_dirs:
        raise FileNotFoundError("No generation runs found. Run generate_summaries.py first.")
    return run_dirs[-1] / "golden_summaries.json"


def load_golden_summaries(summaries_path: Path | None = None) -> list[dict]:
    path = summaries_path or find_latest_summaries()
    if not path.exists():
        raise FileNotFoundError(f"Golden summaries not found at {path}")
    print(f"  Loading summaries from: {path}")
    with open(path) as f:
        return json.load(f)


def build_corpus_texts(
    golden: list[dict],
    fable_texts: list[str],
    config_key: str,
    variant: str,
) -> list[str]:
    config = CONFIGS[config_key]
    template = config["template"]

    summary_lookup = {}
    for item in golden:
        fable_idx = int(item["id"].split("_")[1])
        summary_lookup[fable_idx] = item["summaries"].get(variant, "")

    corpus = []
    for i, fable_text in enumerate(fable_texts):
        summary = summary_lookup.get(i, "")
        text = template.format(fable=fable_text, summary=summary)
        corpus.append(text)

    return corpus


def run_experiment(config_keys: list[str], variants: list[str],
                   summaries_path: Path | None = None):
    fable_texts, moral_texts, ground_truth = load_moral_to_fable_retrieval_data()
    golden = load_golden_summaries(summaries_path)

    available_variants = set()
    for item in golden:
        available_variants.update(item.get("summaries", {}).keys())

    variants = [v for v in variants if v in available_variants]
    if not variants:
        print("ERROR: No matching variants found in golden_summaries.json")
        print(f"  Available: {available_variants}")
        return []

    print(f"\n09_gemma4_summarization — Retrieval Evaluation")
    print(f"  {len(fable_texts)} fables, {len(moral_texts)} moral queries")
    print(f"  Configs: {config_keys}")
    print(f"  Variants: {variants}")

    print(f"\n  Loading {EMBED_MODEL_ID}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL_ID)

    def encode(texts: list[str], is_query: bool = False, batch_size: int = 32):
        if is_query:
            texts = [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {t}" for t in texts]
        return model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)

    print("\n  Encoding moral queries...")
    moral_embs = encode(moral_texts, is_query=True)

    print("\n  Encoding raw fables (baseline)...")
    t0 = time.time()
    baseline_embs = encode(fable_texts)
    baseline_time = time.time() - t0
    baseline_metrics = compute_metrics(moral_embs, baseline_embs, ground_truth)
    print(f"  Baseline: MRR={baseline_metrics['MRR']:.4f}  "
          f"R@1={baseline_metrics['Recall@1']:.3f}  "
          f"R@10={baseline_metrics['Recall@10']:.3f}")

    all_results = [{
        "config": "baseline",
        "variant": "none",
        "description": "Raw fable text (Linq-Embed-Mistral)",
        "encode_time_s": round(baseline_time, 1),
        **baseline_metrics,
    }]

    for config_key in config_keys:
        config = CONFIGS[config_key]

        if config.get("no_summary"):
            print(f"\n  {'═' * 60}")
            print(f"  Config {config_key}: {config['name']}")

            corpus_texts = build_corpus_texts(golden, fable_texts, config_key, "")
            t0 = time.time()
            corpus_embs = encode(corpus_texts)
            encode_time = time.time() - t0

            metrics = compute_metrics(moral_embs, corpus_embs, ground_truth)
            delta = metrics["MRR"] - baseline_metrics["MRR"]
            print(f"    MRR={metrics['MRR']:.4f}  R@1={metrics['Recall@1']:.3f}  "
                  f"R@10={metrics['Recall@10']:.3f}  "
                  f"(vs baseline: {'+' if delta >= 0 else ''}{delta:.4f})")

            all_results.append({
                "config": config_key,
                "variant": "none",
                "description": config["name"],
                "encode_time_s": round(encode_time, 1),
                **metrics,
            })
            continue

        for variant in variants:
            print(f"\n  {'═' * 60}")
            print(f"  Config {config_key}: {config['name']} | Variant: {variant}")

            corpus_texts = build_corpus_texts(golden, fable_texts, config_key, variant)
            sample = corpus_texts[0]
            print(f"    Sample: {sample[:120]}{'...' if len(sample) > 120 else ''}")

            t0 = time.time()
            corpus_embs = encode(corpus_texts)
            encode_time = time.time() - t0

            metrics = compute_metrics(moral_embs, corpus_embs, ground_truth)
            delta = metrics["MRR"] - baseline_metrics["MRR"]
            print(f"    MRR={metrics['MRR']:.4f}  R@1={metrics['Recall@1']:.3f}  "
                  f"R@10={metrics['Recall@10']:.3f}  "
                  f"(vs baseline: {'+' if delta >= 0 else ''}{delta:.4f})")

            all_results.append({
                "config": config_key,
                "variant": variant,
                "description": f"{config['name']} ({variant})",
                "encode_time_s": round(encode_time, 1),
                **metrics,
            })

    return all_results


def make_run_dir() -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d = RUNS_DIR / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma 4 golden summaries for moral-to-fable retrieval"
    )
    parser.add_argument(
        "--configs", nargs="+", choices=list(CONFIGS.keys()),
        default=list(CONFIGS.keys()),
    )
    parser.add_argument(
        "--variants", nargs="+",
        default=["direct_moral", "narrative_distillation", "conceptual_abstract"],
    )
    parser.add_argument(
        "--summaries-path", type=Path, default=None,
        help="Path to golden_summaries.json (default: latest generation run).",
    )
    args = parser.parse_args()

    results = run_experiment(args.configs, args.variants, args.summaries_path)

    if not results:
        sys.exit(1)

    run_dir = make_run_dir()
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'═' * 85}")
    print(f"  SUMMARY")
    print(f"{'═' * 85}")
    print(f"  {'Config':<12} {'Variant':<25} {'MRR':>8} {'R@1':>8} {'R@10':>8} {'vs Base':>8}")
    print(f"  {'─' * 75}")

    baseline_mrr = results[0]["MRR"] if results else 0
    for r in sorted(results, key=lambda x: x["MRR"], reverse=True):
        delta = r["MRR"] - baseline_mrr
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}" if r["config"] != "baseline" else "—"
        print(f"  {r['config']:<12} {r['variant']:<25} {r['MRR']:>8.4f} "
              f"{r['Recall@1']:>7.1%} {r['Recall@10']:>7.1%} {delta_str:>8}")

    print(f"\n  Results saved to {run_dir}")
```

- [ ] **Step 2: Commit**

```bash
git add experiments/09_gemma4_summarization/run.py
git commit -m "exp09: add run.py (adapted from Exp 07, path changes only)"
```

---

### Task 4: README

**Files:**
- Create: `experiments/09_gemma4_summarization/README.md`

- [ ] **Step 1: Write README**

Create `experiments/09_gemma4_summarization/README.md`:

```markdown
# Experiment 09: Gemma 4 Summarization Oracle

## Hypothesis
Gemma 4 (local, MLX, open-weights) can match or exceed Gemini API quality for
moral summarization, with larger models producing better retrieval results.

## Method
1. Generate moral summaries for all 709 fables using Gemma 4 instruct models locally.
2. Evaluate retrieval with Linq-Embed-Mistral on 3 corpus configurations.

## Models
| Key | HuggingFace ID | Disk |
|-----|----------------|------|
| gemma4-e2b | `mlx-community/gemma-4-e2b-it-4bit` | 3.6 GB |
| gemma4-e4b | `mlx-community/gemma-4-e4b-it-4bit` | 5.2 GB |
| gemma4-31b | `mlx-community/gemma-4-31b-it-4bit` | 18.4 GB |

## How to Run

```bash
# Step 1: Quick smoke test (3 fables, no save)
python experiments/09_gemma4_summarization/generate_summaries.py --sample 3 --models gemma4-e2b

# Step 2: Full generation for a single model
python experiments/09_gemma4_summarization/generate_summaries.py --models gemma4-e2b

# Step 3: Evaluate retrieval
python experiments/09_gemma4_summarization/run.py

# Resume interrupted run
python experiments/09_gemma4_summarization/generate_summaries.py --models gemma4-e4b --resume
```

## Baselines
- Exp 02: Raw fable MRR = 0.210
- Exp 03: Qwen3.5-9b MRR = 0.215

## Results
*Pending — run experiment to populate.*
```

- [ ] **Step 2: Commit**

```bash
git add experiments/09_gemma4_summarization/README.md
git commit -m "exp09: add README"
```

---

### Task 5: Smoke test

- [ ] **Step 1: Run unit tests (final check)**

```bash
pytest experiments/09_gemma4_summarization/tests/test_generate.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 2: Run smoke test with 3 fables**

```bash
python experiments/09_gemma4_summarization/generate_summaries.py --sample 3 --models gemma4-e2b
```

Expected output (approximately):
```
Sample mode: 3 fables per model (not saved)

────────────────────────────────────────────────────────────
Model: gemma4-e2b  (mlx-community/gemma-4-e2b-it-4bit)
Fables: 3  |  Variants: ['direct_moral', 'narrative_distillation', 'conceptual_abstract']
────────────────────────────────────────────────────────────
  Loading mlx-community/gemma-4-e2b-it-4bit...
  100%|████████████| 3/3 [...]

  [1] <fable title>
    direct_moral: <one-sentence moral>
    narrative_distillation: <one-sentence moral>
    conceptual_abstract: <one-sentence moral>
  ...
```

No errors, summaries are coherent single sentences.

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "exp09: Gemma 4 summarization oracle — implementation complete"
```
