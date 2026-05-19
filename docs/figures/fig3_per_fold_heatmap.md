# Fig 3 — Per-Fold MRR Heatmap

![Per-Fold Heatmap](fig3_per_fold_heatmap.png)

## What this shows

Per-fold MRR@10 for every fine-tuning variant that used full 5-fold GroupKFold cross-validation.  
The **Mean** column (right, yellow border) is the primary reported metric.

---

## GroupKFold Splits

The 709 fable-moral pairs are split into 5 folds where pairs sharing the same moral stay in the same fold.  
This is critical: 27 morals appear in 2–4 fables each, so standard KFold would leak them across splits.

Each fold has ~142 test pairs and ~567 training pairs.

---

## Row Descriptions

| Row | Model | Loss | Corpus |
|---|---|---|---|
| ft_01: BGE 5-fold | BGE-base-en-v1.5 (110M) | Standard InfoNCE | Raw fables |
| ft_02: Linq LoRA (raw) | Linq-Embed-Mistral (7B) + LoRA r=64 | Standard InfoNCE | Raw fables |
| ft_02: Linq LoRA (f+s) | Linq-Embed-Mistral (7B) + LoRA r=64 | Standard InfoNCE | Fable+summary |
| ft_03: Moral-neg InfoNCE | Linq-Embed-Mistral (7B) + LoRA r=64 | Moral-spreading InfoNCE | Fable+summary |
| ft_03 + Hard neg (injected) | Same as above | + in-batch injected-adjective negatives | Fable+summary |
| ft_03 + Hard neg (partial story) | Same as above | + partial-story hard negatives | Fable+summary |
| ft_03 + Hard neg (adj only) | Same as above | + adjective-only hard negatives | Fable+summary |

---

## Notable Cells

- **⚠ Fold 0, adj-only hard neg (0.008):** Training collapsed — the adjective-only distractor strategy caused the embedding space to degenerate on this fold. The model learned to ignore morals entirely. Other folds recovered because the fold-0 training set happened to have more ambiguous moral clusters.
- **Best mean (ft_03 basic and partial_story):** Both achieve mean MRR = 0.438, but partial_story hard negatives show higher std (0.045 vs 0.043), suggesting slightly less stable training.
- **Fold 1 consistently best:** Fold 1 reliably produces MRR > 0.490 across all Linq variants. This fold likely has a more discriminative moral distribution.

---

## Loss Functions — Quick Reference

For full method details, formulas, and worked examples see **[fig2_finetuning_journey.md](fig2_finetuning_journey.md)**.

### ft_01 / ft_02 — Standard InfoNCE (`MultipleNegativesRankingLoss`)

```
L = -log [ exp(sim(q, d+) / τ) / Σ_j exp(sim(q, dj) / τ) ]
```

- **q** = moral embedding, **d+** = correct fable embedding, **dj** = all other fables in the batch
- Negatives are purely in-batch — no mining needed
- **No multi-positive masking** — when two morals in a batch share a fable, the other's positive is silently treated as a hard negative (a flaw ft_03 fixes)

> Example: batch of 4 pairs → each moral is pushed away from the 3 other fables in the batch.

---

### ft_03 basic — Moral-Spreading InfoNCE (Type 1 + Type 2)

Adds a second negative type on top of ft_02:

```
all_logits = [ fable_sim (B×B) | moral_sim (B×B) ]    shape: (B, 2B)
L = CrossEntropy(all_logits, targets)
```

- **Type 1** (left half): moral_i vs all fables in batch — same as ft_02
- **Type 2** (right half): moral_i vs all other morals in batch — new
- **Multi-positive masking**: shared-moral pairs are masked to −∞ in the denominator, not penalised

> Example: two queries *"Appearances are deceptive."* are in the same batch.  
> Type 2 pushes their embeddings apart. Multi-positive masking stops the model from treating each other's correct fable as wrong.

---

### ft_03 hard neg variants — Type 1 + Type 2 + Type 3

Adds one explicit hard negative per training pair (mined offline):

```
all_logits = [ fable_sim (B×B) | hard_neg_sim (B×1) | moral_sim (B×B) ]
```

- **Type 3** (middle column): moral_i vs one mined hard-negative fable
- Hard negatives are mined using MCQA distractors → Linq similarity → closest true moral → that fable
- Stop-gradient on the hard negative embedding (saves ~33% VRAM)

| Strategy | How distractor is constructed |
|---|---|
| `based_on_adjectives` | Adjectives extracted from fable text → synthetic moral |
| `injected_adjectives` | Adjectives injected into a plausible moral template |
| `partial_story` | LLM summarises only part of the story into a moral |

> Example: Fable = *"The Lion and the Mouse"* · true moral = *"Gratitude is the sign of noble souls."*  
> Partial-story distractor = *"Compassion can bridge the gap between the strongest and the weakest."*  
> → mined hard negative = the fable teaching *"Kindness is never wasted."*
