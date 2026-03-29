# Moral Retrieval with Fables — Project Overview

_Meeting material: March 2026_

---

## What Is This Project?

We are building a **retrieval benchmark** that tests whether embedding models can capture _abstract moral meaning_ — going beyond surface-level word matching to evaluate genuine analogical moral reasoning.

### The Core Question

> Can an AI model, given a moral lesson like _"Appearances are deceptive"_, find the fable that best illustrates it — even though the fable never uses those words?

This is hard. A fable about a wolf in sheep's clothing shares almost no vocabulary with the moral "appearances are deceptive". Yet a human instantly recognises the connection. Can embeddings do the same?

---

## The Dataset: MORABLES

**709 fable–moral pairs** from the Western literary tradition (Aesop, Perry, Gibbs, Abstemius).

Each entry has:

- `story` — the full fable text (~117 words on average)
- `moral` — the distilled lesson (~10 words on average)
- `distractors` — carefully crafted wrong morals (similar characters, partial story, injected adjectives)

**Key stat:** Lexical overlap (content word IoU) between fable and moral = **0.011** — extremely low. You cannot match fable to moral by keyword. The task requires semantic understanding.

![EDA Overview](fig1_eda_overview.png)

**Dataset splits:**
| Config | Description |
|--------|-------------|
| `fables_only` | 709 clean fable–moral pairs |
| `mcqa` | Multiple choice with 4 distractors per fable |
| `adversarial` | 18 variants with character swaps, trait injection |
| `binary` | Binary relevance judgements |

**Sources:**

- Gibbs: 50.8%
- Aesop: 16.8%
- Perry: 12.6%
- Abstemius: 11.7%

---

## Retrieval Tasks

**Task 1 — Moral → Fable** _(main task)_
Given a moral, retrieve the correct fable from a corpus of 709.

**Task 2 — Fable → Moral**
Given a fable, retrieve the correct moral.

**Task 3 — Augmented corpus**
Same as Task 1, but the corpus also contains 2,313 distractor morals — making it much harder.

**Evaluation metrics:** MRR, R@1, R@5, R@10, NDCG@10, Mean/Median Rank

---

## What Has Been Done So Far

### Phase 0 — Setup & Data Preparation ✅

- Git repo initialised
- MORABLES dataset downloaded and explored
- EDA: story length distributions, source breakdown, lexical overlap analysis
- Data structured into clean retrieval format (corpus JSON + qrels)

### Phase 1 — Baseline Retrieval ✅

#### 1.1–1.3 Sentence-BERT Baselines

Two models tested: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`

| Model  | Task        | MRR   | R@1  |
| ------ | ----------- | ----- | ---- |
| MiniLM | Moral→Fable | 0.079 | 4.4% |
| MPNet  | Moral→Fable | 0.078 | 3.2% |

**Finding:** Near-random performance. 68% of correct fables rank below position 50. These models do not capture moral abstraction.

#### 1.4 Stronger Embedding Models

Tested: BGE-large-en-v1.5, E5-large-v2, multilingual-E5-large, **Linq-Embed-Mistral**

| Model                  | MRR       | R@1       | R@5       | R@10      |
| ---------------------- | --------- | --------- | --------- | --------- |
| E5-large-v2            | 0.102     | 5.6%      | 13.5%     | 17.8%     |
| multilingual-E5        | 0.095     | 5.6%      | 12.4%     | 15.2%     |
| **Linq-Embed-Mistral** | **0.210** | **14.1%** | **28.2%** | **36.4%** |

Linq-Embed-Mistral is a large Mistral-7B-based embedding model — it dramatically outperforms the others. **This is now our best baseline.**

![Model Comparison — Moral → Fable Retrieval](fig2_model_comparison.png)

---

## Experiments: LLM-Enhanced Retrieval

### Approach B — LLM Summarisation → Embed ✅

**Idea:** Instead of embedding the raw fable (long, narrative), use a local LLM to first generate a one-sentence moral summary, then embed that summary.

**Hypothesis:** LLM distillation removes the surface noise and makes the embedding space better suited for moral-level matching.

**Setup:** 4 Qwen3.5 model sizes × 4 prompt styles = 16 combinations. Embedding model: Linq-Embed-Mistral.

| Model                | Best Prompt | MRR   | R@1       |
| -------------------- | ----------- | ----- | --------- |
| Baseline (raw fable) | —           | 0.210 | **14.1%** |
| qwen3.5-0.8b         | detailed    | 0.100 | 5.6%      |
| qwen3.5-2b           | direct      | 0.126 | 7.8%      |
| qwen3.5-4b           | detailed    | 0.178 | 10.6%     |
| qwen3.5-9b           | cot         | 0.215 | 14.1%     |

**Finding:** Summarisation doesn't beat the raw baseline — only 9b+CoT ties it. Smaller models introduce noise. **Few-shot prompting consistently hurts across all sizes.**

**Interpretation:** The bottleneck is the embedding model's ability to understand moral abstraction, not the representation format. Linq-Embed-Mistral already extracts enough signal from raw fable text.

![Approach B — LLM Summarisation Results](fig3_approach_b.png)

### Approach C — Upper Bound (Fable + Ground-Truth Moral) ✅

**Idea:** Concatenate the fable with its _ground-truth_ moral before embedding. This tests the theoretical upper bound: what if we always had a perfect summary?

| Variant                        | MRR       | R@1       | R@5       | R@10      |
| ------------------------------ | --------- | --------- | --------- | --------- |
| Raw fable (baseline)           | 0.210     | 14.1%     | 28.2%     | 36.4%     |
| Fable + prefix tag             | 0.210     | 13.3%     | 28.5%     | 35.4%     |
| **Fable + ground-truth moral** | **0.893** | **82.7%** | **97.6%** | **98.6%** |

**Finding:** Staggering jump — R@1 goes from 14% to 83% when the moral is explicit. The embedding model _can_ represent moral meaning perfectly — the gap is entirely in surfacing it from narrative text. **This strongly motivates the thesis.**

![Approach C — Upper Bound](fig4_approach_c.png)

### Approach A — Gemini Reranking 🔄 _(ready to run)_

**Idea:** Use the embedding model to retrieve a shortlist of top-K candidate fables, then use a Gemini LLM with reasoning to rerank them by moral relevance.

**Setup:**

- Embedding model: Linq-Embed-Mistral (best baseline)
- Reranker: Gemini 3.1 Flash Lite
- Top-K candidates: 20

**Why this might help:** The embedding model does a good first-pass retrieval. Gemini can then apply deeper reasoning about which fable best _embodies_ the moral principle — going beyond cosine similarity.

### Approach A — Results (100-query sample, seed=42) ✅

|                             | MRR   | R@1   |
| --------------------------- | ----- | ----- |
| Embedding baseline (top-20) | 0.248 | 20.0% |
| After Gemini reranking      | 0.258 | 20.0% |

**Per-query breakdown (100 queries):**
| Outcome | Count |
|---------|-------|
| Correct fable not in top-20 (unreachable) | 57 |
| Already rank 1, stayed rank 1 | 12 |
| Gemini improved the rank | 16 |
| Gemini worsened the rank | 13 |
| Unchanged (in top-20, not rank 1) | 14 |

**Findings:**

- MRR improved marginally (+1%) but R@1 unchanged
- 57% of queries are unreachable — correct fable outside top-20 shortlist
- Among the 43 reachable queries: Gemini helped 16, hurt 13 — net roughly neutral
- `gemini-3.1-flash-lite-preview` appears too weak for nuanced moral reasoning at this scale

![Approach A — Gemini Reranking vs Embedding Baseline (100 queries)](fig5_approach_a.png)

**Breakdown of what happened per query (100 queries, random sample):**

| Outcome                                       | Count | Notes                              |
| --------------------------------------------- | ----- | ---------------------------------- |
| Correct fable **not in top-20** (unreachable) | 57    | Gemini never sees the right answer |
| Already rank 1, stayed rank 1                 | 12    | Embedding was already correct      |
| Gemini **improved** the rank                  | 16    | e.g. rank 4 → 1, rank 10 → 1       |
| Gemini **worsened** the rank                  | 13    | e.g. rank 1 → 2, rank 1 → 6        |
| Unchanged (in top-20, not rank 1)             | 14    | No movement                        |

**Open questions from Approach A:**

- Would a stronger model (gemini-3-flash, accepting ~50s/query) produce a clear signal?
- Should we increase top-K (top-30/40/50) to make more queries reachable before reranking?
- Can a better prompt improve Gemini's hit rate on the reachable queries?

---

## Open Questions & Discussion Points

### 1. What's the ceiling for Approach A?

Gemini can only rerank what the embedding model includes in the shortlist. If the correct fable isn't in the top-K, reranking can't help.

| Top-K shortlist | Correct fable included | Max possible R@1 |
| --------------- | ---------------------- | ---------------- |
| Top-10          | 36.4% (258/709)        | 36.4%            |
| **Top-20**      | **43.0% (305/709)**    | **43.0%**        |
| Top-30          | 48.0% (340/709)        | 48.0%            |
| Top-40          | 52.5% (372/709)        | 52.5%            |
| Top-50          | 55.9% (396/709)        | 55.9%            |
| Top-100         | 66.9% (474/709)        | 66.9%            |

**Discussion:** Should we increase K? Larger K = more chances to include the right answer, but also a harder reranking problem (more noise for Gemini to sift through). Also: larger prompts = slower & more expensive API calls.

### 2. Why does few-shot hurt in Approach B?

In every model size, few-shot prompting produced the worst results. Possible explanations:

- The example morals in the prompt are from famous fables (tortoise/hare, crow/fox) — the model may be biased toward generating morals similar to those examples
- The format constraint ("short standalone phrase") may cause the model to oversimplify

### 3. Why is there a 14% → 83% gap (Approach B vs C)?

Even the best LLM summarisation (9b) only reaches 14.1% R@1, while having the true moral gives 82.7%. This means LLMs at this scale don't reliably distill the precise moral from a fable. The remaining gap is a research opportunity:

- Can fine-tuned models close it?
- Is the task inherently ambiguous (fables can have multiple valid morals)?

### 4. Cross-parable analogy retrieval (not yet done)

Given one fable, can we retrieve _other_ fables that share the same moral but have completely different surface content? This is the hardest version of the task and most directly tests analogical reasoning.

### 5. Adversarial robustness (not yet done)

MORABLES includes adversarial variants: character swaps, trait injections, tautologies. Do our best models remain robust when surface features change? This could reveal _what_ the embeddings are actually latching onto.

### 6. Fine-tuning direction

Contrastive fine-tuning using MORABLES pairs (correct moral as positive, distractors as hard negatives) could explicitly teach an embedding model what "moral similarity" means. How much of the 14% → 83% gap can fine-tuning close?

---

## Summary: Where We Stand

| Method                      | MRR   | R@1   | Notes                          |
| --------------------------- | ----- | ----- | ------------------------------ |
| Sentence-BERT (baseline)    | ~0.08 | ~4%   | Near-random                    |
| Linq-Embed-Mistral          | 0.210 | 14.1% | Best off-the-shelf model       |
| LLM Summarisation (9b+CoT)  | 0.215 | 14.1% | Ties baseline, doesn't beat it |
| Gemini Reranking            | TBD   | TBD   | Running next                   |
| Oracle (fable + true moral) | 0.893 | 82.7% | Theoretical upper bound        |

**The gap between 14% and 83% is the thesis.**

---

## Next Steps

1. **Run Approach A (full 709 queries)** — Gemini 3.1 Flash Lite reranking, ~18 min
2. **Explore larger top-K** — Try top-30/40/50 shortlists for reranking
3. **Adversarial robustness** — Test best model on MORABLES ADV split
4. **Cross-parable analogy retrieval** — New subtask
5. **Contrastive fine-tuning** — Train on MORABLES pairs with hard negatives
6. **Write thesis proposal** — One-pager for advisor circulation
