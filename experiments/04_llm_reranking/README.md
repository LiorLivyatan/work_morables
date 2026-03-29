# Experiment 04: LLM CoT Reranking (Approach A)

## What it tests
Whether using a large language model (Gemini) with chain-of-thought reasoning to rerank top-K embedding retrieval candidates improves results.

## Method
1. Stage 1: Retrieve top-K candidates using embedding model (Linq-Embed-Mistral or BGE)
2. Stage 2: Send each (moral, candidate fable) pair to Gemini with a CoT prompt asking it to evaluate relevance
3. Rerank based on LLM scores

## How to run
```bash
python experiments/04_llm_reranking/run.py
```

## Key results

| Setup | Queries | Baseline MRR | Reranked MRR |
|-------|---------|-------------|-------------|
| Gemini flash-lite, top-20 | 100 | 0.248 | 0.257 |
| Gemini 2.5-flash, top-20 | 5 | 0.201 | 0.201 |

## Key findings
- **Inconclusive** — only tested on small samples (5-100 queries), never at full scale (709)
- Marginal improvement at best (0.248 -> 0.257 on 100 queries)
- Expensive: requires API calls for each query x candidate pair
- Needs full-scale evaluation to draw conclusions
