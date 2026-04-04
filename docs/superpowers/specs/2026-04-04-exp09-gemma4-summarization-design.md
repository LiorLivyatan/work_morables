# Exp 09: Gemma 4 Summarization Oracle — Design Spec

## Hypothesis

Gemma 4 (local, open-weights) can serve as a high-quality moral summarizer for fables, matching or exceeding Gemini API quality while running entirely offline. Comparing E2B, E4B, and 31B model sizes reveals how model capacity affects summary quality for retrieval.

## Method

1. Use Gemma 4 (via `mlx_lm`, Apple Silicon) to generate moral summaries for all 709 fables.
2. Save summaries in a `golden_summaries.json` corpus compatible with Exp 07's `run.py`.
3. Evaluate retrieval with Linq-Embed-Mistral on the same 3 corpus configurations as Exp 07.

### Models

| Key | HuggingFace ID | Disk | Role |
|-----|----------------|------|------|
| `gemma4-e2b` | `mlx-community/gemma-4-e2b-it-4bit` | 3.6 GB | quick smoke test |
| `gemma4-e4b` | `mlx-community/gemma-4-e4b-it-4bit` | 5.2 GB | main run |
| `gemma4-31b` | `mlx-community/gemma-4-31b-it-4bit` | 18.4 GB | large model |

### Prompt Variants (same as Exp 07)

| Key | Style |
|-----|-------|
| `direct_moral` | Direct one-sentence extraction |
| `narrative_distillation` | Summarize narrative → distill lesson |
| `conceptual_abstract` | Chain-of-thought → abstract principle (last line only) |

### Retrieval Configs (same as Exp 07)

| Config | Corpus text |
|--------|------------|
| A | `{summary}` only |
| B | `{fable}\n\nMoral summary: {summary}` |
| C | `Fable: {fable}` (no summary — baseline) |

## Directory Structure

```
experiments/09_gemma4_summarization/
├── generate_summaries.py        # MLX-LM backend (Exp 03 pattern)
├── run.py                       # Copy of Exp 07 run.py (path changes only)
├── tests/
│   └── test_generate.py         # Unit tests for pure functions
├── README.md
└── results/
    ├── generation_runs/         # <timestamp>_<model>/golden_summaries.json
    └── runs/                    # retrieval evaluation results
```

## Output Schema (matches Exp 07)

```json
[
  {
    "id": "item_000",
    "original_fable_id": "aesop_001",
    "fable_text": "...",
    "ground_truth_moral": "...",
    "summaries": {
      "direct_moral": "...",
      "narrative_distillation": "...",
      "conceptual_abstract": "..."
    },
    "metadata": {
      "source": "aesop",
      "word_count_fable": 133,
      "model": "mlx-community/gemma-4-e2b-it-4bit"
    }
  }
]
```

## Implementation Notes

- Use `mlx_lm.load()` / `mlx_lm.generate()` (Exp 03 pattern), NOT `mlx_vlm`
- System prompts via `apply_chat_template` with `{"role": "system", ...}` message
- Post-processing: strip `<think>` blocks; for `conceptual_abstract`, take last non-empty line
- Each model gets its own timestamped run dir (preserves comparisons)
- Incremental save every 50 fables; `--resume` flag to continue interrupted runs
- `--sample N` for quick testing (prints only, no save)
- One model loaded at a time, unloaded and cache-cleared between models

## Baselines

- Exp 02: Raw fable MRR = 0.210
- Exp 03: Qwen3.5-9b MRR = 0.215
- Exp 07: Gemini API (pending full results)

## Success Criteria

MRR > 0.215 for any Gemma 4 variant indicates improvement over local Qwen baseline.
MRR > 0.30 would be a significant breakthrough.
