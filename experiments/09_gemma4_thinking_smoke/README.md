# Experiment 09: Gemma 4 Summarization Oracle

## Hypothesis

Gemma 4 (local, MLX, open-weights) can match or exceed Gemini API quality for
moral summarization, with larger models producing better retrieval results.
Comparing E2B → E4B → 31B reveals how model capacity affects the abstraction gap.

## Method

1. Use Gemma 4 instruct models (via `mlx_lm`, Apple Silicon) to generate moral summaries for all 709 fables.
2. Save as a `golden_summaries.json` corpus with 3 prompt variants per fable.
3. Evaluate retrieval with Linq-Embed-Mistral on 3 corpus configurations.

## Models

| Key | HuggingFace ID | Disk |
|-----|----------------|------|
| `gemma4-e2b` | `mlx-community/gemma-4-e2b-it-4bit` | 3.6 GB |
| `gemma4-e4b` | `mlx-community/gemma-4-e4b-it-4bit` | 5.2 GB |
| `gemma4-31b` | `mlx-community/gemma-4-31b-it-4bit` | 18.4 GB |

## Prompt Variants

| Variant | Style |
|---------|-------|
| `direct_moral` | Direct one-sentence extraction |
| `narrative_distillation` | Summarize narrative → distill core lesson |
| `conceptual_abstract` | Chain-of-thought → abstract principle (last line) |

## How to Run

```bash
# Step 1: Quick smoke test (3 fables, no save)
python experiments/09_gemma4_summarization/generate_summaries.py --sample 3 --models gemma4-e2b

# Step 2: Full generation for a single model
python experiments/09_gemma4_summarization/generate_summaries.py --models gemma4-e2b

# Step 3: Evaluate retrieval
python experiments/09_gemma4_summarization/run.py

# Evaluate a specific summaries file
python experiments/09_gemma4_summarization/run.py \
  --summaries-path experiments/09_gemma4_summarization/results/generation_runs/<run>/golden_summaries.json

# Resume an interrupted generation run
python experiments/09_gemma4_summarization/generate_summaries.py --models gemma4-e4b --resume
```

## Baselines

- Exp 02: Raw fable MRR = 0.210
- Exp 03: Qwen3.5-9b MRR = 0.215
- Exp 07: Gemini API (pending)

## Results

*Pending — run experiment to populate.*
