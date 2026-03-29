# Approach B: LLM Summarisation → Embed & Retrieve

**Task:** Moral → Fable retrieval (709 fable-moral pairs, MORABLES dataset)

**Method:** Use a local Qwen3.5 LLM to generate a one-sentence moral summary for each fable, then embed the summary instead of the raw fable text. Embedding model: `Linq-Embed-Mistral`.

---

## Results

| Model | Prompt | MRR | R@1 | R@5 | R@10 |
|-------|--------|-----|-----|-----|------|
| **Baseline** (raw fable) | — | **0.210** | **14.1%** | **28.2%** | **36.4%** |
| qwen3.5-9b | cot | 0.215 | 14.1% | 27.9% | 36.1% |
| qwen3.5-9b | direct | 0.204 | 12.7% | 27.2% | 36.0% |
| qwen3.5-9b | detailed | 0.196 | 12.1% | 26.4% | 33.7% |
| qwen3.5-9b | few_shot | 0.183 | 10.9% | 24.1% | 31.9% |
| qwen3.5-4b | detailed | 0.178 | 10.6% | 23.7% | 32.6% |
| qwen3.5-4b | direct | 0.171 | 10.0% | 23.1% | 30.8% |
| qwen3.5-4b | cot | 0.167 | 10.3% | 21.7% | 28.6% |
| qwen3.5-4b | few_shot | 0.141 | 8.2% | 18.9% | 23.4% |
| qwen3.5-2b | direct | 0.126 | 7.8% | 15.7% | 21.7% |
| qwen3.5-2b | cot | 0.114 | 6.8% | 15.2% | 20.3% |
| qwen3.5-2b | detailed | 0.111 | 6.4% | 13.8% | 20.6% |
| qwen3.5-2b | few_shot | 0.080 | 4.4% | 11.1% | 14.3% |
| qwen3.5-0.8b | detailed | 0.100 | 5.6% | 12.8% | 17.6% |
| qwen3.5-0.8b | cot | 0.094 | 5.1% | 12.4% | 16.8% |
| qwen3.5-0.8b | direct | 0.088 | 4.7% | 11.3% | 15.0% |
| qwen3.5-0.8b | few_shot | 0.066 | 3.8% | 7.9% | 11.7% |

---

## Key Findings

1. **Summarisation does not beat the baseline.** Embedding raw fable text directly matches or outperforms LLM-generated summaries for all model sizes except 9b+CoT, which ties the baseline (R@1=14.1%).

2. **Model size matters enormously.** R@1 scales from ~5% (0.8b) → ~8% (2b) → ~10% (4b) → ~14% (9b). Smaller models introduce noise that hurts retrieval.

3. **Few-shot prompting consistently underperforms** across all model sizes — the in-context examples appear to constrain the output in unhelpful ways.

4. **The bottleneck is the embedding model**, not the representation format. Linq-Embed-Mistral already captures enough fable content to match what a 9b LLM distills.

---

## Interpretation

Approach B tests whether LLM distillation helps the embedding model find the moral signal. The answer is: only if the LLM is large enough (≥9b) to produce accurate summaries. This suggests the difficulty of moral retrieval lies in the embedding space itself — not in surface clutter from the fable narrative.

**Next step:** Approach C (embed fable + ground-truth moral concatenated) will establish the theoretical upper bound.
