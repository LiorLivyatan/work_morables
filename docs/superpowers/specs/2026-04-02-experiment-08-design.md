# Experiment 08 — Symmetric Moral Matching (Design Spec)

**Date:** 2026-04-02
**Goal:** Improve R@1 over experiment 07's best result (25.2%) by eliminating the asymmetry between short moral queries and long fable documents.

---

## Problem

Experiment 07 enriched the fable corpus with Gemini-generated summaries. The best result was `cot_proverb__fable_plus_summary` at MRR=0.360, R@1=25.2%. The core remaining bottleneck: the query (a ~10-word moral) is still being matched against a paragraph-length document. Even with a summary appended, the corpus is "fable-flavored."

---

## Core Idea

Convert the matching problem from **moral vs. narrative** into **moral vs. moral**:

- **Corpus:** Use Gemini to rewrite each fable as a single moral-style sentence (no fable text). The corpus document now has the same length and style as the query.
- **Query:** Expand each ground-truth moral into 3 paraphrases using Gemini. At retrieval, take the max cosine score across all 4 query variants (original + 3 paraphrases).
- **Fusion:** Apply Reciprocal Rank Fusion (RRF) across all query×corpus combinations.

**Key constraint:** Gemini never sees the ground-truth morals when generating corpus summaries. It only sees the fable text. This remains an oracle experiment.

---

## Scope

**Pilot run: first 50 fables only.** Success on the pilot justifies scaling to all 709.

**Success criterion:** R@1 on the 50-fable subset beats exp 07's best config (`cot_proverb__fable_plus_summary`) evaluated on the same 50 fables.

---

## Components

### 1. New Corpus Variants (2 new Gemini prompts)

Both prompts target the style of ground-truth morals: 5–15 words, declarative, abstract, no character names.

| Variant | Prompt goal |
|---------|------------|
| `ground_truth_style` | Generate a moral as a concise aphorism (5–15 words), matching the style of classic fable morals like *"Appearances are deceptive."* No character names. |
| `declarative_universal` | Distill the moral into one declarative sentence (5–15 words). Universal, timeless, no narrative description. |

**LLM cost:** 2 × 50 = 100 Gemini calls.

### 2. Query Expansion (3 paraphrases per moral)

| Variant | Style |
|---------|-------|
| `moral_rephrase` | Same meaning, different wording |
| `moral_elaborate` | Expand the principle with slightly more context |
| `moral_abstract` | Strip to the most abstract principle |

At retrieval: compute cosine similarity for all 4 query vectors (original + 3), take **max score** per corpus document.

**LLM cost:** 3 × 50 = 150 Gemini calls.

### 3. Retrieval Configurations

| Config | Corpus | Query |
|--------|--------|-------|
| Baseline (exp 07 best) | fable + cot_proverb summary | original moral |
| A | ground_truth_style summary only | original moral |
| B | declarative_universal summary only | original moral |
| A+expand | ground_truth_style summary only | max over 4 paraphrases |
| B+expand | declarative_universal summary only | max over 4 paraphrases |
| RRF-all | — | RRF fusion of all score matrices |

### 4. Score Fusion

Apply RRF (k=60) across all score matrices from the configurations above. This is the primary submission configuration.

---

## Evaluation

- **Primary metric:** R@1
- **Secondary metric:** MRR
- **Corpus size:** 50 fables, 50 moral queries
- **Embedding model:** Linq-Embed-Mistral (same as exp 07, for fair comparison)

---

## File Structure

```
experiments/08_symmetric_moral_matching/
├── README.md
├── generate_corpus_summaries.py   # Generate 2 new fable→moral-style summaries
├── generate_query_expansions.py   # Generate 3 paraphrases per moral query
├── run.py                         # Retrieval + RRF evaluation
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
| Corpus variants (2 × 50) | 100 |
| Query expansion (3 × 50) | 150 |
| **Total** | **~250** |

Model: `gemini-3-flash-preview` (same as exp 07).

---

## Next Steps After Pilot

If R@1 on 50 beats exp 07 baseline on same 50: scale to all 709 (total ~3,500 additional Gemini calls).
