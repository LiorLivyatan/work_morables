# ft_03: Hard-Negative Fine-Tuning — Architecture & Experiment Plan

## Context

Our task is **moral-to-fable retrieval**: given a short moral statement (e.g. *"Gratitude is the sign of noble souls"*), retrieve the correct fable from a 709-fable corpus.

**ft_02 baseline results (Linq-Embed-Mistral, 5-fold GroupKFold CV):**

| Setting | MRR |
|---|---|
| Baseline Linq (zero-shot) — raw | 0.2097 |
| ft_02 — raw (in-batch negatives only) | 0.3121 |
| ft_02 — fable+summary (cross-eval) | 0.4102 |

ft_02 uses **in-batch negatives only**: for each moral, every other fable in the same training batch is treated as a negative. These are easy negatives — a random fable about a fox is obviously not the answer to a moral about gratitude. ft_03 introduces **hard negatives** to force deeper semantic understanding.

---

## Dataset Resources

### Core retrieval dataset
- **709 (moral, fable) pairs** — each moral maps to exactly one fable
- **27 morals appear in 2–4 fables** — same moral text, multiple correct fables (see §Multi-Positive Masking)

### MCQA dataset (`data/raw/mcqa.json`)
From the MORABLES benchmark (Marcuzzo et al., EMNLP 2025). 709 entries, one per fable. Each entry contains the story, its true moral, and **4 adversarial distractor morals**:

| Type | Description | Example (for *"Gratitude is the sign of noble souls"*) |
|---|---|---|
| `similar_characters` | Moral from a different story with similar character types | *"Never trust a known deceiver."* |
| `based_on_adjectives` | Moral derived from descriptive adjectives in the text | *"Bravery and compassion heal wounds."* |
| `injected_adjectives` | True moral modified with misleading adjectives | *"The true leader proves himself by his brave qualities."* |
| `partial_story` | Moral derived from only part of the narrative | *"Compassion can bridge the gap between the strongest and the weakest."* |

These distractors are **human-verified** and specifically designed to fool language models.

---

## Training Triplet Definition

For each fable `F` with true moral `M` and a chosen distractor `Di`:

```
query (anchor)     M    — the true moral
positive           F+   — the correct fable (raw text or fable+summary)
hard negative      F-   — a different fable, mined via Di (see §Hard Negative Mining)
soft negatives          — all other fables in the batch (in-batch negatives)
```

### Hard Negative Mining

For each fable `F` and each distractor `Di`:

1. Encode all 709 true morals with the **zero-shot Linq model**
2. Encode distractor `Di`
3. Retrieve the fable `G` from the corpus whose true moral has highest cosine similarity to `Di`
4. `G` is the hard negative for the triple `(M, F, G)`

**Intuition**: `G` is a fable the current model would likely retrieve if the query were `Di` instead of `M`. Training on `(M, F, G)` teaches the model to prefer the correct fable `F` over this confusable alternative.

This produces up to **4 hard negatives per fable × 709 fables = 2,836 triples**. After GroupKFold splitting (~567 training fables per fold), each fold has ~2,268 hard negative triples.

---

## Loss Function

We use **InfoNCE** (= cross-entropy over cosine similarities), the same framework used by Gecko and IdioLink. This loss pushes the query `M` closer to its correct fable `F+`, and simultaneously farther from both the hard negative `F-` and all other in-batch fables.

For a batch of `B` triples `(M_i, F_i+, F_i-)`:

```
L = (1/B) Σ_i  -log [ exp(sim(M_i, F_i+) / τ) / Z_i ]

Z_i = Σ_{j: F_j+ ∉ gold(M_i)}  exp(sim(M_i, F_j+) / τ)    ← in-batch negatives (masked)
    +                            exp(sim(M_i, F_i-) / τ)    ← hard negative
```

Where `sim(·,·)` is cosine similarity and `τ` is temperature.

**Relation to Gecko (Eq. 3) and IdioLink**: The denominator structure is identical — in-batch negatives plus an explicit hard negative. The one difference is that Gecko and IdioLink also include **same-tower negatives** (query×query similarities in the denominator). We omit these because our task is **asymmetric**: queries are always morals and documents are always fables. There is no meaningful signal in pushing moral embeddings away from each other.

### Multi-Positive Masking

27 morals appear in 2–4 fables — the same moral text is the true moral of multiple fables. Without masking, when `(M, F1)` and `(M, F2)` appear in the same batch, `F2` is treated as a wrong answer for `M`. This is a **false negative** — the most damaging error in contrastive training, as it actively trains the model away from a correct retrieval.

Fix: maintain `gold(M)` = all fable indices whose true moral equals `M`. Any `F_j+` in `gold(M_i)` is excluded from the denominator during loss computation.

### Key Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Hard negatives per anchor (αH) | **1** | Gecko uses 1; IdioLink found gradient instability at αH > 2 |
| Temperature τ | sweep {0.05, 0.07, 0.1} | 0.05 is Gecko/IdioLink default; sweep to find optimum |
| Batch size | 4 (+ grad accumulation ×4, effective 16) | GPU memory constraint for 7B model |
| Same-tower negatives | **no** | Task is asymmetric (moral→fable); not applicable |
| Curriculum learning | **no** | Linq is pre-trained; hard negatives from epoch 1 |

---

## Ablation Plan: Distractor Type

We do not know a priori which distractor type produces the most useful hard negatives. We run a **distractor-type ablation** — 5 runs, all other hyperparameters fixed:

| Run | Distractors used to mine F- | Hypothesis |
|---|---|---|
| ft_03a | `similar_characters` only | Broadest — may be too easy |
| ft_03b | `based_on_adjectives` only | References actual fable content |
| ft_03c | `injected_adjectives` only | Closest to true moral — likely hardest |
| ft_03d | `partial_story` only | Partially correct signal |
| ft_03e | All 4 types (one per anchor, random) | Combined baseline |

This directly answers: *which distractor type is most informative for retrieval training?* — a natural contribution for the thesis.

---

## Implementation Roadmap

### Step 1 — Mine hard negatives (`ft_03/mine_negatives.py`)
- Encode 709 true morals + 4×709 distractors with zero-shot Linq
- Retrieve top-1 fable per distractor
- Output: `ft_03/data/hard_negatives.json`

```json
{
  "fable_alias": "aesop_section_1_5",
  "moral": "Gratitude is the sign of noble souls.",
  "distractor_type": "injected_adjectives",
  "distractor_text": "The true leader proves himself by his brave qualities.",
  "hard_neg_alias": "aesop_section_2_14",
  "hard_neg_moral": "..."
}
```

### Step 2 — Training script (`ft_03/train.py`)
- Load mined triples, apply GroupKFold splits
- Build multi-positive mask from shared-moral lookup
- InfoNCE loss with hard negative in denominator
- Sweep τ ∈ {0.05, 0.07, 0.1}
- Evaluate with same MRR pipeline as ft_02

### Step 3 — Ablation runs
- 5 distractor-type runs × 3 τ values × 5 folds = 75 training jobs
- Parallelizable across GPUs

---

## Future Data Directions

The current 709-fable / 2,836-triple dataset is intentional as a first grounded experiment. Two directions planned later:

1. **Additional moral-fable corpora** — other fable collections (e.g. La Fontaine) expand the training set. The pipeline above generalizes directly.
2. **LLM-generated distractors** — a teammate is exploring synthetic distractor generation, which multiplies the hard negative pool without relying solely on MCQA. This plugs directly into the same mining + training pipeline.

---

## Related Work

| Paper | Relevance |
|---|---|
| **Gecko** (Google, 2024) | InfoNCE loss structure, hard negative mining via retrieval, αH=1 default |
| **IdioLink** (2025) | Applied Gecko-style InfoNCE to idiom retrieval; used same-tower negatives (omitted here); found gradient instability at αH > 2 |
| **MORABLES benchmark** (Marcuzzo et al., EMNLP 2025) | Source of MCQA distractors; their work tests LLM comprehension, ours builds the retrieval complement |
