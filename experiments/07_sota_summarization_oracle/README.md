# Experiment 07: SOTA Summarization Oracle (PAR-11)

## Hypothesis
Current summaries (Qwen3.5-9b) are lossy. A SOTA model (Gemini 3.1 Pro) will produce "perfect" moral distillations. If MRR remains low, the bottleneck is the embedding model's latent space; if MRR jumps significantly, the bottleneck was summary quality.

## Method
1. Use Gemini 3.1 Pro to generate moral summaries for all 709 fables
2. Save as a "Golden Summary" corpus with 3 summary variants per fable
3. Run retrieval using Linq-Embed-Mistral on three configurations:
   - **(A) Summary only** — embed only the generated summary
   - **(B) Fable + Summary** — concatenate fable text with generated summary
   - **(C) Fable + Instruction** — prepend a retrieval instruction to the fable

### Summary Variants (3 Gemini prompts)
| Variant | Prompt Style | Goal |
|---------|-------------|------|
| `direct_moral` | Direct extraction | "State the moral of this fable in one sentence" |
| `narrative_distillation` | Summarize & extract | Distill the narrative into its core lesson |
| `conceptual_abstract` | Chain-of-thought | Reason step-by-step to extract the abstract moral concept |

### Retrieval Configurations (per variant)
| Config | Corpus text | Expected effect |
|--------|------------|-----------------|
| A: Summary only | `{summary}` | Tests if summary alone is sufficient for retrieval |
| B: Fable + Summary | `{fable}\n\nMoral summary: {summary}` | Enriches fable with moral signal |
| C: Fable + Instruction | `Fable: {fable}` | Baseline with prefix instruction |

## How to run
```bash
# Step 1: Generate golden summaries (requires GEMINI_API_KEY in .env)
python experiments/07_sota_summarization_oracle/generate_summaries.py

# Step 2: Run retrieval evaluation
python experiments/07_sota_summarization_oracle/run.py
```

## Baselines
- 02_model_comparison: Raw Fable MRR = 0.210
- 03_llm_summarisation: Qwen3.5-9b MRR = 0.215

## Success Criteria
MRR > 0.30 would indicate a significant breakthrough in representation-led retrieval.

## Key results
*Pending — run experiment to populate.*
