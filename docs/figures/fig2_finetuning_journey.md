# Fig 2 — Fine-Tuning Journey

![Fine-Tuning Journey](fig2_finetuning_journey.png)

## What this shows

MRR@10 progression across all fine-tuning experiments, from sanity check to hyperparameter search.  
Error bars show ±1 std across 5 GroupKFold splits (where available).

---

## Experiment Descriptions

---

### ft_00 — Single-Sample Overfit (Sanity Check)

**Model:** BGE-base-en-v1.5 (110M) — full fine-tune, no LoRA  
**Result:** MRR = 0.974 — deliberate overfit, not a generalization result  
**Purpose:** Verify the pipeline end-to-end before investing GPU time.

#### Method

**Loss:** `MultipleNegativesRankingLoss` (sentence-transformers built-in)  
**Negatives:** In-batch only — every other fable in the same mini-batch is treated as a negative.  
**Training set:** All 709 pairs. Evaluated on the same 709 pairs (overfit by design).

```
L = -log [ exp(sim(q, d+) / τ) / Σ_j exp(sim(q, dj) / τ) ]
```

> **Example:** Query = *"Gratitude is the sign of noble souls."* → positive = the Lion and the Mouse fable.  
> All 63 other fables in the batch are negatives. The model is forced to score the correct fable highest.

---

### ft_01 — BGE-base 5-Fold Cross-Validation

**Model:** BGE-base-en-v1.5 (110M) — full fine-tune, no LoRA  
**Splits:** 5-fold **GroupKFold** — morals that appear in multiple fables are never split across train/test  
**Corpus:** Raw fables  
**Result:** 0.122 ± 0.022 (vs BGE zero-shot ≈ 0.069 → **+0.053**)

#### Method

**Loss:** `MultipleNegativesRankingLoss` — same as ft_00, temperature τ = 0.05  
**Negatives:** In-batch only. Batch size = 64, so each sample has 63 in-batch negatives.  
**Fine-tuning:** All 110M parameters updated (no LoRA).  
**Why GroupKFold matters:** 27 morals each appear in 2–4 fables. If standard KFold puts fable A and fable B (same moral) in different folds, the model sees the moral during training — that leaks. GroupKFold forces all fables sharing a moral into the same fold.

> **Example:** Moral = *"Appearances are deceptive."* — this moral is shared by 3 different Aesop fables. GroupKFold keeps all 3 in the same fold so none leaks into training when the others are being tested.

---

### ft_02 — Linq + LoRA, Standard InfoNCE (5-Fold CV)

**Model:** Linq-Embed-Mistral (7B) + **LoRA** (r=64, α=128, dropout=0.05, targets: q/k/v/o_proj — ~0.76% of params)  
**Loss:** `MultipleNegativesRankingLoss` — same structure as ft_01  
**Negatives:** In-batch only. Batch size = 4 (7B model, GPU memory constraint), grad accumulation = 4 steps → effective batch = 16.  

| Corpus | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| Raw fables | 0.210 | 0.318 ± 0.034 | **+0.108** |
| Fable + summary | 0.360 | 0.416 ± 0.033 | **+0.056** |

#### Method

Same InfoNCE loss as ft_01 but with LoRA — only ~50M of 7B parameters are updated. LoRA injects trainable low-rank matrices (A, B) into the attention projections: the actual weight update is `W + BA` where B is (d × r) and A is (r × d), r=64.

```
L = -log [ exp(sim(q, d+) / τ) / Σ_j exp(sim(q, dj) / τ) ]
```

**No multi-positive masking here.** If two training items share the same moral (27 morals do), the other item's positive fable is silently treated as a hard negative — this causes a subtle training signal corruption that ft_03 fixes.

> **Example (batch of 4):**  
> q1 = *"Gratitude is the sign of noble souls."* → d1+ = Lion and Mouse fable  
> q2 = *"Slow and steady wins the race."* → d2+ = Tortoise and Hare fable  
> q3 = *"Appearances are deceptive."* → d3+ = Fox and Grapes fable  
> q4 = *"Appearances are deceptive."* → d4+ = Wolf in Sheep's Clothing fable ← same moral as q3!  
> For q3, d4+ is treated as a negative even though it's a correct answer. ft_03 fixes this.

---

### ft_03 — Linq + LoRA, Moral-Spreading InfoNCE (5-Fold CV)

**Model:** Linq-Embed-Mistral (7B) + LoRA (r=64, same as ft_02)  
**Result (basic mode):** 0.438 ± 0.043 (vs ft_02: **+0.022**)

#### Method — Basic Mode (Type 1 + Type 2)

Custom `InfoNCELoss` with **two types of negatives per query**:

**Type 1 — In-batch fable negatives** (same as ft_02):
```
push moral_i away from all other fables in the batch
```

**Type 2 — In-batch moral negatives** (new):
```
push moral_i away from all other morals in the batch
```

Both are computed from a single forward pass — no extra data needed. The full logit matrix is:

```
all_logits = [ fable_sim (B×B) | moral_sim (B×B) ]   shape: (B, 2B)
target_i   = i   (correct fable is column i in the left half)

L = CrossEntropy(all_logits, targets)
```

**Multi-positive masking:** When two items in the batch share the same moral text (e.g., q3 and q4 above), their respective positive fables are masked to −∞ in each other's row — they are excluded from the denominator instead of being penalised as negatives.

> **Example:**  
> q3 = *"Appearances are deceptive."* — positive = Fox and Grapes  
> q4 = *"Appearances are deceptive."* — positive = Wolf in Sheep's Clothing  
> **ft_02** would penalise q3 for ranking Wolf in Sheep's Clothing highly.  
> **ft_03** masks that cell to −∞ so it has zero gradient contribution.  
> Additionally, q3 is pushed away from q4 (Type 2), keeping the two morals separated in embedding space even though they're identical strings — preventing the model from collapsing them into one point.

---

### ft_03 — Hard Negative Variants (Types 1 + 2 + 3)

Three **Type 3** hard negative strategies were tested, all using the same mining pipeline:

#### Mining pipeline
1. The MCQA dataset provides 4 LLM-generated distractor morals per fable (wrong answers in a multiple-choice quiz).
2. Each distractor is encoded with zero-shot Linq.
3. The most similar **true** moral in the dataset (by cosine similarity) is found.
4. The fable that teaches **that** true moral becomes the hard negative for training.

The distractor types tested:

| Type | What the distractor looks like | Hard negative |
|---|---|---|
| `based_on_adjectives` | Uses adjectives from the fable text | Fable whose true moral is most similar to those adjectives |
| `injected_adjectives` | Adjectives injected into a plausible-sounding moral | Same mining |
| `partial_story` | Moral describing only part of the story | Same mining |

**Stop-gradient on hard negative:** The hard negative fable is encoded with `torch.no_grad()` — its embeddings are detached. Only the query (moral) encoder receives gradient through Type 3.

> **Concrete example for `partial_story`:**  
> Fable: *"The Lion and the Mouse"* — true moral: *"Gratitude is the sign of noble souls."*  
> Partial-story distractor: *"Compassion can bridge the gap between the strongest and the weakest."*  
> This distractor is encoded → most similar true moral found: *"Kindness is never wasted."*  
> The fable teaching *"Kindness is never wasted."* becomes the Type-3 hard negative for this training step.  
> The model must score *"The Lion and the Mouse"* higher than the *"Kindness is never wasted."* fable — a genuinely hard discrimination.

**Results:**

| Hard neg strategy | Mean MRR | Notes |
|---|---|---|
| None (basic) | **0.438 ± 0.043** | Best overall |
| injected_adjectives | 0.432 ± 0.048 | Small regression |
| partial_story | 0.438 ± 0.045 | Ties basic, higher variance |
| based_on_adjectives | 0.356 | **Collapsed on fold 0** — embedding degeneration |

---

### ft_04 — Linq + LoRA + STORAL Augmentation (Fold 0 only)

**Added data:** 1,675 (moral, modern-story) pairs from STORAL (Guan et al., NAACL 2022)  
**Loss:** Same as ft_03 basic (Type 1 + Type 2)  
**Result:** MRR = 0.420 (fold 0) vs ft_03 fold 0 = 0.438 → **−0.018**

#### Method

STORAL pairs are appended to the training set before batching. No special treatment — they appear alongside MORABLES pairs in the same batches and contribute to the same InfoNCE loss.

> **Example STORAL pair:**  
> Moral: *"Hard work always pays off in the end."*  
> Story: *"Maria stayed late every night for a month studying for her exam..."* (modern short story)  
> This is stylistically very different from *"The Tortoise, though slow, pressed on steadily..."* (Aesop).  
> The model must reconcile two very different "story" styles for the same type of moral.

**Why it hurts:** At 1,675 STORAL + 570 MORABLES, ~75% of each batch is out-of-domain. The in-batch negatives are increasingly STORAL-vs-STORAL, which teaches domain-internal discrimination rather than the Aesop-specific signal needed for retrieval. See Fig 4 for the mixing ratio analysis.

---

### ft_05 — BGE-base STORAL Mixing Ratio Sweep (Fold 0 proxy)

**Model:** BGE-base-en-v1.5 (110M) — full FT, no LoRA  
**Loss:** Custom `InfoNCELoss` (Type 1 + Type 2), τ = 0.05  
**Purpose:** Cheap proxy experiment to find the optimal STORAL ratio before committing to Linq runs.  
**Result:** Sweet spot at ~200 pairs (+0.010 MRR). Full STORAL (1,675) matches zero augmentation.

> See Fig 4 for the full curve.

---

### ft_06 — Linq + LoRA, Random Hyperparameter Search (Fold 0)

**Model:** Linq-Embed-Mistral (7B) + LoRA  
**Loss:** Same as ft_03 basic (Type 1 + Type 2 InfoNCE)  
**Evaluation:** Fold 0 only — best config will go to full 5-fold CV  
**Search space:** r ∈ {32, 64, 128} · τ ∈ {0.02, 0.05, 0.07, 0.10} · lr ∈ {5e-5, 1e-4, 2e-4}  
**Best so far:** r=32, τ=0.02, lr=1e-4 → MRR = 0.439 (6/10 trials complete)

#### Method — what changes across trials

Only three things vary per trial; everything else (architecture, loss structure, data, folds) is identical to ft_03 basic:

| Hyperparameter | Effect |
|---|---|
| LoRA rank `r` | Controls adapter expressivity. Higher r = more params, more overfit risk on 570 training pairs. |
| Temperature `τ` | Controls sharpness of the softmax. Lower τ = harder gradient signal, model must be more certain about positives. |
| Learning rate | Controls step size. Too high → instability; too low → doesn't converge within ~4 epochs (early stopping window). |

> **Why τ=0.02 wins:** At τ=0.05 (ft_03 default), two morals with cosine similarity 0.85 produce logits ~17 and ~16 — a soft push apart. At τ=0.02, those same morals produce logits ~42.5 and ~40 — a much steeper gradient. The model is forced to make sharper distinctions between semantically similar morals like *"Slow and steady wins the race."* vs. *"Patience always pays off."*

---

## Reference Lines

| Line | Value | Meaning |
|---|---|---|
| Linq zero-shot | 0.360 | Best zero-shot model on fable+summary |
| Current best (ft_03) | 0.438 | Best mean 5-fold MRR |
| Oracle ceiling | 0.893 | Fable concatenated with its own ground-truth moral |
