# Experiment 07: Gemini 3 Flash Summarization Oracle
_March 2026 — MORABLES Moral-to-Fable Retrieval_

---

## Background & Motivation

**The core finding so far:** The best off-the-shelf embedding model (Linq-Embed-Mistral) achieves MRR = 0.210 on raw fable text, while the oracle (concatenating the fable with its *ground-truth* moral) reaches MRR = 0.893. That's a 4× gap.

**Exp 03** (Qwen local LLMs) showed that LLM-generated moral summaries added only marginally (best: MRR = 0.215). The question was:

> *Is the bottleneck the **quality of the summaries**, or the **embedding model's latent space**?*

If a state-of-the-art API model producing near-perfect summaries also fails, the bottleneck is the embedding space itself. If it succeeds, the bottleneck was summary quality and we need better generators.

---

## Setup

**Summary generator:** Gemini 3 Flash (`gemini-3-flash-preview`, API) — a fast, capable API model used to test whether high-quality LLM summaries can bridge the moral-to-fable retrieval gap.

**Retrieval model:** Linq-Embed-Mistral — identical to all other experiments. Only the corpus documents change.

**Dataset:** All 709 MORABLES fable-moral pairs.

---

## Configurations

| Config | Corpus document | What it tests |
|--------|----------------|--------------|
| **A — Summary only** | `{generated summary}` | Can the summary alone match moral queries? |
| **B — Fable + Summary** | `{fable text}\n\nMoral summary: {summary}` | Does the summary enrich fable retrieval? |
| **C — Fable + prefix** | `Fable: {fable text}` | Control — does a text prefix alone help? |

---

## Prompts (4 variants)

### `direct_moral`
> *"State the moral of this fable in one sentence."*

Short, direct extraction. No reasoning chain. Produces concise morals.

---

### `proverb`
> *"Distill this fable into a proverb-style maxim."*

Encourages aphoristic, memorable phrasing. Slightly more abstract than direct_moral.

---

### `cot_proverb`
> *Step-by-step reasoning: (1) central conflict → (2) what does the outcome reveal? → (3) what abstract principle? Then output the moral as a proverb on the last line.*

Chain-of-thought that concludes with a proverb. The reasoning is hidden — only the final line is used.

---

### `conceptual_abstract`
> *Same 3-step reasoning as cot_proverb, but framed as a moral philosopher extracting an abstract universal principle rather than a folk proverb.*

Tends to produce more general, transferable principles.

---

## Results

![Results tables](../results/exp07_gemini_oracle_results.png)

### Full Numbers

| Config / Prompt | MRR | R@1 | R@10 | vs Baseline |
|----------------|-----|-----|------|-------------|
| **cot_proverb + fable** | **0.360** | 25.2% | 57.5% | **+0.150** |
| conceptual_abstract (summary only) | 0.355 | **26.5%** | 55.4% | +0.146 |
| direct_moral + fable | 0.352 | 23.7% | 57.5% | +0.142 |
| proverb + fable | 0.351 | 24.0% | 57.3% | +0.141 |
| conceptual_abstract + fable | 0.349 | 23.6% | **57.7%** | +0.139 |
| direct_moral (summary only) | 0.337 | 23.0% | 55.4% | +0.127 |
| cot_proverb (summary only) | 0.305 | 21.0% | 49.2% | +0.095 |
| proverb (summary only) | 0.278 | 18.2% | 46.1% | +0.068 |
| **Baseline (raw fable)** | **0.210** | **14.0%** | **36.4%** | — |

### Summary Exactness

How often does the generated summary exactly equal the ground-truth moral?

| Variant | Exact matches | Total | Match rate |
|---------|--------------|-------|-----------|
| **cot_proverb** | **34** | 709 | **4.8%** |
| proverb | 18 | 709 | 2.5% |
| direct_moral | 15 | 709 | 2.1% |
| conceptual_abstract | 10 | 709 | 1.4% |

*Exact match = case-insensitive, punctuation-normalised string equality.*

---

## Key Findings

### 1. The bottleneck is **both** summary quality and embedding space

Gemini summaries **do** produce a significant jump — from MRR 0.210 to 0.360 (+71% relative). This rules out pure embedding-space bottleneck. Better summaries clearly help.

But MRR 0.360 is still far from the oracle (0.893). The embedding model still cannot perfectly bridge abstract morals to narrative fables even with excellent summaries. Both are bottlenecks.

### 2. Fable + summary consistently beats summary only

In every variant, Config B (fable + summary) outperforms Config A (summary only) — with one notable exception: `conceptual_abstract`. This suggests Gemini's abstract philosophical summaries are so well-aligned with moral queries that they work best when used alone, without fable text diluting the signal.

### 3. `cot_proverb` is the best prompt overall

The chain-of-thought + proverb format produces the highest MRR (0.360) and is also the most "exact" — 34 generated summaries are verbatim identical to the ground-truth moral (4.8%). The structured reasoning appears to help Gemini zero in on the intended moral formulation.

### 4. Exact match rate is low but revealing

Even the best prompt (`cot_proverb`) only exactly matches the ground-truth moral 4.8% of the time. This means retrieval improvement is not coming from memorising morals — it comes from generating semantically aligned summaries. The embedding model is learning to bridge *approximate* moral language to the query.

### 5. Config C adds nothing

Adding just a "Fable:" prefix (Config C) produces exactly the same MRR as raw fables (0.210). The gain in Config B is entirely attributable to the LLM summary content.

---

## Implications for Next Steps

| Question | Answer |
|---------|--------|
| Is summary quality a bottleneck? | Yes — Gemini beats Qwen 9B significantly |
| Is the embedding space a bottleneck? | Yes — even Gemini can't reach the oracle |
| Which prompt is best? | `cot_proverb` (use in Exp 09 with Gemma 4) |
| What's next? | HyDE (query expansion), contrastive fine-tuning |

---

## Reproducibility

```bash
# Requires GEMINI_API_KEY in .env
python experiments/07_sota_summarization_oracle/generate_summaries.py

# Evaluate
python experiments/07_sota_summarization_oracle/run.py
```

Results stored in `experiments/07_sota_summarization_oracle/results/generation_runs/full_709/`.
