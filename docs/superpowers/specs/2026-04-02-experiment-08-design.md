# Experiment 08 — Symmetric Moral Matching (Design Spec)

**Date:** 2026-04-02
**Goal:** Improve R@1 over experiment 07's best R@1 result (26.52%) by eliminating the asymmetry between short moral queries and long fable documents.

---

## Problem

Experiment 07 enriched the fable corpus with Gemini-generated summaries. The best **R@1** result was `conceptual_abstract__summary_only` at R@1=26.52% (MRR=0.355). The query side (morals) was never enriched in exp 07. Even with a summary appended, the corpus is still "fable-flavored" — the embedding model does cross-domain matching every single time.

---

## Core Idea

Convert the matching problem from **moral vs. narrative** into **moral vs. moral**:

- **Corpus:** Use Gemini to rewrite each fable as a single moral-style sentence (no fable text). The corpus document now has the same length and style as the query.
- **Query:** Expand each ground-truth moral into 3 paraphrases using Gemini. At retrieval, take the max cosine score across all 4 query variants (original + 3 paraphrases). Note: max-score aggregation is susceptible to generic paraphrases lifting wrong documents — results should be validated per-query, not just in aggregate.
- **Fusion:** Apply Reciprocal Rank Fusion (RRF, k=60) across ranked lists (argsort of score matrices per query) for all query×corpus combinations.

**Key constraint:** Gemini never sees the ground-truth morals when generating corpus summaries. It only sees the fable text. This remains an oracle experiment.

---

## Scope

**Pilot run: first 10 fables only.** Success on the pilot justifies scaling to all 709.

**Step 0 (before any new generation):** Re-run exp 07's R@1-best config (`conceptual_abstract__summary_only`) restricted to the 10-fable subset to establish the formal baseline threshold. The existing exp 07 results cover all 709 fables and cannot be used directly.

**Success criterion:** R@1 on the 10-fable subset beats `conceptual_abstract__summary_only` evaluated on the same 10 fables.

---

## Components

### 1. New Corpus Variants (2 new Gemini prompts)

Both prompts target the style of ground-truth morals: 5–15 words, declarative, abstract, no character names. Unlike `cot_proverb` (which uses CoT reasoning and outputs a proverb-style sentence), these variants include **few-shot examples drawn directly from the ground-truth morals corpus** to force output style closer to the dataset distribution.

| Variant | Prompt goal |
|---------|------------|
| `ground_truth_style` | Generate a moral as a concise aphorism (5–15 words) with 3 few-shot examples from the ground-truth morals (e.g., *"Appearances are deceptive."*, *"Vices are their own punishment."*). No character names. |
| `declarative_universal` | Distill the moral into one declarative sentence (5–15 words). Universal, timeless, no narrative description. Few-shot examples included to anchor style. |

**LLM cost:** 2 × 10 = 20 Gemini calls.

### 2. Query Expansion (3 paraphrases per moral)

All paraphrases must remain **under 20 words and abstract** — no narrative examples, no character names — to preserve the symmetry with short corpus documents.

| Variant | Style |
|---------|-------|
| `moral_rephrase` | Same meaning, different wording (≤15 words) |
| `moral_elaborate` | Slightly broader principle (≤20 words, abstract only, no examples) |
| `moral_abstract` | Most concise form of the principle (≤10 words) |

At retrieval: compute cosine similarity for all 4 query vectors (original + 3), take **max score** per corpus document.

**LLM cost:** 3 × 10 = 30 Gemini calls.

### 3. Retrieval Configurations

| Config | Corpus | Query |
|--------|--------|-------|
| Baseline (exp 07 R@1-best, re-run on 50) | conceptual_abstract summary only | original moral |
| A | ground_truth_style summary only | original moral |
| B | declarative_universal summary only | original moral |
| A+expand | ground_truth_style summary only | max over 4 paraphrases |
| B+expand | declarative_universal summary only | max over 4 paraphrases |
| RRF-all | — | RRF over ranked lists of all 5 configs above |

Note: exp 07 embeddings are already cached — including the baseline config in RRF costs no additional Gemini calls.

### 4. Score Fusion

Apply RRF (k=60) over **ranked lists** (per-query argsort of cosine score matrices), not over raw scores. All 5 new configurations are included in the fusion pool. k=60 is a standard default and should be treated as a hyperparameter if results are borderline.

---

## Evaluation

- **Primary metric:** R@1
- **Secondary metric:** MRR
- **Corpus size:** 10 fables, 10 moral queries (indices 0–9)
- **Embedding model:** Linq-Embed-Mistral (same as exp 07, for fair comparison)

---

## File Structure

```
experiments/08_symmetric_moral_matching/
├── README.md
├── generate_corpus_summaries.py   # Generate 2 new fable→moral-style summaries
├── generate_query_expansions.py   # Generate 3 paraphrases per moral query
├── run.py                         # Step 0 baseline + retrieval + RRF evaluation
└── results/
    └── generation_runs/
        └── <timestamp>_sample50/
            ├── corpus_summaries.json
            ├── query_expansions.json
            └── retrieval_results.json
```

---

## LLM Cost Summary

| Step | Calls |
|------|-------|
| Step 0: baseline (uses cached exp 07 embeddings) | 0 |
| Corpus variants (2 × 10) | 20 |
| Query expansion (3 × 10) | 30 |
| **Total new calls** | **~50** |

Model: `gemini-3-flash-preview` (same as exp 07).

---

## Next Steps After Pilot

If R@1 on 10 beats exp 07 R@1-best on same 10: scale to 50, then all 709.
Scaling cost: ~3,545 new calls (2 × 709 corpus + 3 × 709 query). Exp 07 embeddings for all 709 are already cached.
