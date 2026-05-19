# Fig 4 — STORAL Augmentation Mixing Ratio Sweep (ft_05)

![STORAL Mixing Ratio](fig4_storal_mixing_ratio.png)

## What this shows

How many STORAL pairs to add before performance starts degrading.  
This was a **proxy experiment on BGE-base** (110M, fast) before committing GPU hours to the full Linq (7B) run.

---

## Setup & Method

**Model:** BGE-base-en-v1.5 (110M parameters, no LoRA — full fine-tune)  
**Corpus:** Fable+summary  
**Evaluation:** Fold 0 test set only (142 queries)  
**STORAL dataset:** Guan et al., NAACL 2022 — 1,675 clean (moral, story) pairs from modern moral tales  

**Loss:** Custom `InfoNCELoss` (Type 1 + Type 2), τ = 0.05 — same structure as ft_03 basic.  
STORAL pairs are concatenated with MORABLES pairs before batching; no special treatment or weighting.

```
Training set at ratio=500: 570 MORABLES pairs + 500 STORAL pairs → 1,070 total
Each mini-batch (size 32) contains a mix of Aesop and modern-story pairs.
```

> **Example STORAL pair:**  
> Moral: *"Persistence in the face of failure leads to eventual success."*  
> Story: *"After failing his driving test four times, David signed up for extra lessons..."* (modern narrative)  
>  
> Contrast with a typical MORABLES pair:  
> Moral: *"Slow and steady wins the race."*  
> Fable: *"A Hare one day ridiculed the short feet and slow pace of the Tortoise..."* (Aesop, terse, allegorical)  
>  
> Both land in the same batch → the model must reconcile two very different "story" styles as valid positives for similar moral themes.

---

## Results

| STORAL pairs added | MRR@10 | Δ vs no augment |
|---|---|---|
| 0 (MORABLES only) | 0.2774 | — |
| **200** | **0.2872** | **+0.010** ← best |
| 500 | 0.2777 | +0.000 |
| 1,000 | 0.2740 | −0.003 |
| 1,675 (all) | 0.2740 | −0.003 |

---

## Interpretation

**More STORAL = worse performance.** Two reasons:
1. **Domain mismatch:** STORAL stories are modern moral fables — very different stylistically from Aesop's fables (terse, ancient, often allegorical). The model wastes capacity fitting an out-of-domain distribution.
2. **Signal dilution:** With 1,675 STORAL pairs added to 570 MORABLES training pairs, ~75% of each batch is out-of-domain. The contrastive signal degrades.

**Sweet spot: ~200 pairs.** At this scale, STORAL adds useful moral-diversity signal without overwhelming the in-domain examples.

**However:** ft_04 tested the full Linq+LoRA pipeline with STORAL (fold 0 only) and found MRR = 0.420 vs ft_03's 0.438 — a net loss. The ft_05 proxy result (+0.010 for BGE) doesn't reliably transfer to Linq because:
- Linq's 7B parameters are harder to push into the STORAL distribution
- The moral-spreading InfoNCE loss (ft_03) already handles the key failure mode that STORAL was meant to fix

**Conclusion:** Data augmentation via STORAL is not a viable path. Hard negatives (ft_03) are a stronger technique.
