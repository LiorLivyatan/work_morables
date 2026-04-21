# ft_03: Hard-Negative Fine-Tuning — Proposal

## Executive Summary

We have a dataset of 709 Aesop fables, each paired with its true moral. The task is **moral-to-fable retrieval**: given a short moral statement (e.g. *"Gratitude is the sign of noble souls"*), find the correct fable from the 709-fable corpus.

Our current approach (ft_02) fine-tunes Linq-Embed-Mistral (7B) using contrastive learning: for each moral, we show the model its correct fable as a positive and treat every other fable in the same training batch as a negative. These are called **in-batch negatives** — and they're easy. A random fable about a fox and a crow is obviously not the answer to a moral about gratitude. The model learns, but not as much as it could.

The next step is to add **hard negatives**: wrong fables that *look like* they could be the right answer. The model has to work harder to distinguish them, which forces deeper semantic understanding. This proposal describes how to do that using the MORABLES MCQA benchmark dataset.

---

## Background: The Gecko Fine-Tuning Framework

Gecko (Google, 2024) defines the training set as triplets:

```
D = (task_instruction, query q, positive d+, hard_negative d-)
```

The loss function (InfoNCE / cross-entropy) pushes `q` closer to `d+` and farther from both `d-` and all other in-batch items:

```
L = -log [ exp(sim(q, d+)/τ) / (Σ exp(sim(q, dj+)/τ) + exp(sim(q, d-)/τ)) ]
```

Where `τ` is temperature and the denominator sums over all positives and negatives in the batch. Crucially, hard negatives appear explicitly in the denominator — they get a dedicated penalty, not just a share of the in-batch signal.

**Key insight from Gecko**: In-batch negatives alone are insufficient. Explicit hard negatives substantially improve performance by forcing the model to make genuinely difficult distinctions.

---

## Our Dataset Resources

### 1. Core retrieval dataset
- **709 (moral, fable) pairs** — each moral maps to exactly one fable
- **27 morals appear in 2–4 fables** — same moral text, multiple correct fables (already handled by GroupKFold splits)
- Source: Aesop's Fables (MORABLES dataset)

### 2. MCQA dataset (`data/raw/mcqa.json`)
From the MORABLES benchmark (Marcuzzo et al., EMNLP 2025):
- **709 entries**, one per fable
- Each entry: `story`, `true_moral`, and 4 **adversarial distractor morals**
- Distractor types (one of each per fable):
  - `similar_characters` — moral from a different story with similar characters
  - `based_on_adjectives` — moral derived from descriptive adjectives in the text
  - `injected_adjectives` — true moral modified with misleading adjectives
  - `partial_story` — moral derived from only part of the narrative

These distractors are **human-verified** and specifically designed to fool models — making them ideal hard negatives.

---

## Proposed Training Architecture

### Triple definition

For each fable `F` with true moral `M`:

```
query          q  = M  (true moral)
positive       d+ = F  (correct fable, raw or fable+summary)
hard negative  d- = G  (fable whose true moral is semantically closest to a distractor of F)
soft negatives    = all other (q, d+) pairs in the batch (in-batch negatives)
```

### How to mine `d-` (the hard negative fable)

For each fable `F`, we have 4 distractor morals `D1..D4`. Each `Di` is a moral that *looks like* it could belong to `F` — but doesn't. The fable that `Di` actually describes is the hard negative for `F`.

**Mining procedure:**
1. Encode all 709 true morals with the baseline Linq model
2. For each distractor `Di` of fable `F`, encode `Di` and retrieve the top-1 fable from the corpus by cosine similarity
3. That retrieved fable `G` is the hard negative for `(M, F)` — it's a fable whose real moral is semantically similar to what `F`'s distractor suggests

This gives up to **4 hard negatives per fable × 709 fables = 2,836 hard negative triples**.

### Alternative: direct distractor-as-query approach

Use the distractors themselves as queries with "no correct answer" — train the model to assign LOW similarity between `Di` and `F`. This requires a modified loss that includes explicit repulsion terms.

---

## Training Configuration (proposed)

| Parameter | Value |
|---|---|
| Base model | Linq-Embed-Mistral (7B) |
| Adapter | LoRA r=64, α=128 |
| Loss | MultipleNegativesRankingLoss with hard negatives |
| Query instruction | `"Instruct: Given a moral statement, retrieve the fable that best conveys this moral.\nQuery: "` |
| Document format | `"{fable}\n\nMoral summary: {cot_proverb_summary}"` (fable+summary mode) |
| Hard negatives per anchor | 1–4 (mined from MCQA distractors) |
| Batch size | 4 (+ gradient accumulation steps=4) |
| Splits | 5-fold CV, GroupKFold (no moral text leakage) |

---

## Expected Benefit

Current ft_02 results as baseline:

| Setting | MRR |
|---|---|
| Baseline Linq — raw | 0.2097 |
| ft_02 — raw (in-batch negatives only) | 0.3121 |
| ft_02 — fable+summary (cross-eval, no retraining) | 0.4102 |

With hard negatives, we expect ft_03 to outperform ft_02 on both corpora. The improvement should be most visible on the `fable+summary` corpus where thematic similarity between fables matters most.

---

## Open Questions

1. **Which distractor types are most useful?** `injected_adjectives` subtly modifies the true moral — arguably the hardest. `similar_characters` is broader. Worth ablating by distractor type.

2. **How many hard negatives per anchor?** 1 vs. 2 vs. 4. More is not always better — too many hard negatives can destabilize training.

3. **Should distractors themselves be queries?** Training on `(Di, G, F)` where `G` is Di's nearest fable and `F` is the hard negative uses the distractor as a query — a form of data augmentation that multiplies the dataset size by 4.

4. **Loss function modification?** SentenceTransformers' `MultipleNegativesRankingLoss` supports hard negatives natively via the `(anchor, positive, negative)` triplet format — no custom loss needed.

---

## Connection to Related Work

- **MORABLES benchmark** (Marcuzzo et al., 2025) — our MCQA distractors come from this paper. Our work is the retrieval complement: they test LLMs on moral comprehension; we build a retrieval system that grounds morals in specific fables.
- **Gecko** (Google, 2024) — the Gecko framework (FRet task definition, hard negative mining) directly inspired this approach.
- **IdioLink** — prior work by this group used the same Gecko-inspired triplet structure for idiom retrieval, establishing the pattern of (literal, idiomatic, cross-type) hard negatives.
