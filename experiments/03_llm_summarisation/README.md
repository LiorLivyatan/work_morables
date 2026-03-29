# Experiment 03: LLM Summarisation (Approach B)

## What it tests
Whether using an LLM to generate a one-sentence moral summary for each fable, then embedding the summary instead of the raw fable, improves retrieval.

## Method
1. Use local Qwen3.5 LLM (0.8b, 2b, 4b, 9b) to generate moral summaries per fable
2. Test 4 prompt styles: direct, detailed, cot (chain-of-thought), few_shot
3. Embed summaries with Linq-Embed-Mistral
4. Compare against raw fable embedding baseline

Also includes **Approach C (Oracle)**: embed fable + ground-truth moral concatenated as upper bound.

## How to run
```bash
# Generate summaries
python experiments/03_llm_summarisation/generate_summaries.py

# Run retrieval experiment
python experiments/03_llm_summarisation/run.py
```

## Key results

| Model | Prompt | MRR | R@1 | R@10 |
|-------|--------|-----|-----|------|
| Baseline (raw fable) | — | 0.210 | 14.1% | 36.4% |
| qwen3.5-9b | cot | 0.215 | 14.1% | 36.1% |
| qwen3.5-4b | detailed | 0.178 | 10.6% | 32.6% |
| **Oracle (fable+moral)** | — | **0.893** | **82.7%** | **98.6%** |

## Key findings
- **Summarisation does not beat the baseline.** Best variant (9b+CoT) merely ties it
- Model size matters enormously: 0.8b is catastrophically bad, 9b barely matches raw fable
- Few-shot prompting consistently underperforms
- **Oracle upper bound (MRR=0.893)** proves the embedding space CAN represent moral meaning — the gap is 4.3x
