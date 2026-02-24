# Thesis Tasks: Moral Retrieval with Parables

## Overview

**Core idea:** Build a retrieval benchmark that tests whether embedding models capture abstract moral meaning — going beyond surface-level lexical matching to evaluate analogical moral reasoning.

**Dataset foundation:** MORABLES (709 fable-moral pairs from Western literary tradition, with carefully crafted distractors).

**Key retrieval tasks to explore:**
1. Moral → Parable retrieval
2. Parable → Moral retrieval
3. Cross-parable analogy retrieval (same moral, different surface domain)
4. Cross-lingual moral retrieval (future)

---

## Phase 0: Setup & Data Preparation

- [x] **0.1 — Initialize git repository** ✓
- [x] **0.2 — Download and explore the MORABLES dataset** ✓
  - Configs: `fables_only` (709 pairs), `mcqa` (with distractors), `adversarial` (18 variants), `binary`
  - Schema: alias, title, story, moral, note, alternative_moral
  - Distractor classes: ground_truth, similar_characters, based_on_adjectives, injected_adjectives, partial_story
- [x] **0.3 — Exploratory data analysis** ✓ (see `results/eda_overview.png`)
  - 709 pairs, 678 unique morals, 27 morals shared across multiple fables
  - Story length: mean 117 words, moral length: mean 10 words
  - **Lexical overlap (content words) IoU: 0.011** — extremely low, confirms lexical matching fails
  - Sources: Gibbs 50.8%, Aesop 16.8%, Perry 12.6%, Abstemius 11.7%
- [x] **0.4 — Structure the data for retrieval** ✓
  - Clean corpus: 709 fables, 709 morals, 1:1 qrels
  - Augmented corpus: 2803 morals (490 correct + 2313 distractors)

---

## Phase 1: Baseline Retrieval Experiments

- [x] **1.1–1.3 — Sentence-BERT baseline retrieval** ✓ (see `results/baseline_results.json`)

  **Results (first 2 models):**

  | Model | Task | MRR | R-Prec | R@1 | R@5 | R@10 |
  |-------|------|-----|--------|-----|-----|------|
  | all-MiniLM-L6-v2 | fable→moral (clean) | 0.063 | 0.034 | 0.034 | 0.075 | 0.117 |
  | all-MiniLM-L6-v2 | moral→fable (clean) | 0.079 | 0.044 | 0.044 | 0.095 | 0.145 |
  | all-MiniLM-L6-v2 | fable→moral (augmented) | 0.026 | 0.012 | 0.012 | 0.035 | 0.045 |
  | all-mpnet-base-v2 | fable→moral (clean) | 0.065 | 0.027 | 0.027 | 0.083 | 0.127 |
  | all-mpnet-base-v2 | moral→fable (clean) | 0.078 | 0.032 | 0.032 | 0.103 | 0.162 |
  | all-mpnet-base-v2 | fable→moral (augmented) | 0.026 | 0.012 | 0.012 | 0.028 | 0.047 |

  **Key findings:**
  - Off-the-shelf Sentence-BERT is near-random for moral retrieval (~3% R@1)
  - Moral→fable is slightly easier than fable→moral
  - Augmented corpus (with distractors) makes it even harder
  - 68% of correct morals rank below position 50 — embeddings don't capture moral abstraction
  - **This confirms the benchmark is meaningful and the task is hard**

- [ ] **1.4 — Try additional embedding models**
  Candidates (at minimum):
  - `sentence-transformers/all-MiniLM-L6-v2` (fast baseline)
  - `sentence-transformers/all-mpnet-base-v2` (stronger)
  - `BAAI/bge-large-en-v1.5`
  - `intfloat/e5-large-v2`
  - An LLM-based embedder (e.g., `nomic-embed-text` or OpenAI `text-embedding-3-large`)

- [ ] **1.5 — Document baseline results**
  Create a results table comparing models across tasks and metrics.

---

## Phase 2: Deeper Analysis & Harder Evaluation

- [ ] **2.1 — Cross-parable analogy retrieval**
  Given one fable, retrieve other fables that share an analogous moral but have different surface content (different characters, settings).
  This requires grouping fables by moral similarity first.

- [ ] **2.2 — Adversarial robustness evaluation**
  Use the MORABLES ADV variant (character swaps, trait injection, tautologies) — do embeddings remain robust when surface features change?

- [ ] **2.3 — Qualitative error analysis**
  For the best baseline model, manually inspect the top failures:
  - Which fable-moral pairs are hardest?
  - Are failures due to lexical confounders, cultural knowledge, or genuine abstraction difficulty?

---

## Phase 3: Improvement with Fine-tuning

- [ ] **3.1 — Contrastive fine-tuning setup**
  Fine-tune a Sentence-BERT model using contrastive learning:
  - Positive pairs: (fable, correct moral).
  - Hard negatives: MORABLES distractors.
  - Loss: e.g., MultipleNegativesRankingLoss or TripletLoss.

- [ ] **3.2 — Train/eval split design**
  Decide on a proper split (e.g., 80/20 or cross-validation) — ensure no fable leaks between train and eval.

- [ ] **3.3 — Run fine-tuning experiments**
  Compare fine-tuned models against Phase 1 baselines.

- [ ] **3.4 — Reranker experiments**
  Use a cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) on top of retrieval results. Measure precision lift.

---

## Phase 4: Thesis Proposal (One-pager)

- [ ] **4.1 — Write thesis proposal document**
  Following the template structure:
  - **Introduction & motivation:** Why moral retrieval matters, gap in existing benchmarks.
  - **Task definition:** Formal definition of the retrieval tasks.
  - **Data:** MORABLES description, size, examples, pie chart of composition.
  - **Related work:** MORABLES paper, ePiC, Moral Stories, STORAL, UniMoral, sentence-BERT, retrieval benchmarks.
  - **Research questions table:**
    - RQ1 (Main): Can embedding models capture abstract moral meaning in a retrieval setting?
    - RQ2: Which distractor types exploit which embedding weaknesses?
    - RQ3: Does contrastive fine-tuning on moral narratives improve moral retrieval?
  - **Timeline with milestones.**
  - **Resources and needs.**

- [ ] **4.2 — Get advisor feedback & iterate**
  Once the one-pager is ready, Kai will circulate it for broader feedback.

---

## Phase 5: Expansion (Future)

- [ ] **5.1 — Survey existing translations of fables**
  Look for multilingual fable collections (Aesop translations exist in many languages).

- [ ] **5.2 — Cross-lingual retrieval experiments**
  Fables in multiple languages, morals in a pivot language (English/Hebrew). Test multilingual encoders.

- [ ] **5.3 — Expand dataset**
  Add fables from non-Western traditions, modern parables, etc.

---

## Suggested Immediate Priority (Next 1-2 Weeks)

| Priority | Task | Why |
|----------|------|-----|
| 1 | 0.1 — Git repo setup | Foundation for all work |
| 2 | 0.2 — Download MORABLES | Need the data |
| 3 | 0.3 — Exploratory analysis + charts | Professor asked for the pie chart of sizes/languages |
| 4 | 0.4 — Structure data for retrieval | Core formatting step |
| 5 | 1.1 — Sentence-BERT baseline (parable→moral) | First real result |
| 6 | 1.2 — Moral→parable baseline | Second result |
| 7 | 1.3 — Distractor analysis | Interesting finding for the proposal |
| 8 | 4.1 — Draft thesis proposal one-pager | Kai wants to circulate this |
