# MORABLES — Fine-Tuning & Model Evaluation Progress
**Date:** April 27, 2026  
**Task:** Moral-to-Fable Retrieval — Given a moral statement, retrieve the fable that best conveys it  
**Dataset:** 709 Aesop fables, 709 moral queries  
**Metric:** MRR@10 (Mean Reciprocal Rank)

---

## Baselines

| Setup | MRR | Notes |
|---|---|---|
| Random retrieval | ~0.002 | Lower bound |
| Linq-Embed-Mistral (raw fables) | 0.210 | Best zero-shot model from 20+ evaluated |
| Linq-Embed-Mistral (fable + LLM summary) | **0.360** | Best corpus representation |
| Oracle ceiling | 0.893 | Fable concatenated with its own ground-truth moral |

The **fable+summary** corpus appends a Gemini-generated conceptual summary to each raw fable, consistently boosting retrieval across all experiments.

---

## Fine-Tuning Experiments

### ft_02 — Linq + LoRA, Standard Contrastive Loss (5-fold CV)
**Model:** Linq-Embed-Mistral (7B) · **LoRA:** r=64, ~0.76% params trained  
**Loss:** Standard in-batch InfoNCE · **Splits:** GroupKFold (prevents shared-moral leakage)

| Corpus | Baseline MRR | Fine-tuned MRR | Δ |
|---|---|---|---|
| Raw fables | 0.210 | 0.318 ± 0.034 | **+0.108** |
| Fable + summary | 0.360 | 0.416 ± 0.033 | **+0.056** |

> Key insight: GroupKFold is critical — 27 morals appear in 2–4 fables. Standard KFold leaks these across train/test, inflating results.

---

### ft_03 — Linq + LoRA, InfoNCE with Moral Negatives (5-fold CV)
**Model:** Linq-Embed-Mistral (7B) · **LoRA:** r=64  
**Loss:** Custom InfoNCE with two negative types:
- **Type 1:** In-batch fable negatives (same as ft_02)
- **Type 2:** In-batch moral–moral negatives *(new)* — prevents embedding collapse when different morals map to the same fable

| Fold | ft_02 | ft_03 | Δ |
|---|---|---|---|
| 0 | 0.424 | 0.438 | +0.014 |
| 1 | 0.446 | 0.509 | +0.063 |
| 2 | 0.450 | 0.438 | −0.012 |
| 3 | 0.396 | 0.431 | +0.035 |
| 4 | 0.363 | 0.374 | +0.011 |
| **Mean** | **0.416** | **0.438 ± 0.043** | **+0.022** |

> Key insight: Early stopping fires at epoch ~4 every fold; best model always at epoch 2. A 7B model overfits quickly on only 570 training pairs — `BestAdapterCallback` captures the epoch-2 peak before trainer cleanup.

---

### ft_04 — Linq + LoRA + STORAL Augmentation (fold 0 only)
**Added:** 1,675 clean (moral, story) pairs from STORAL dataset (Guan et al. NAACL 2022)  
**Result (fold 0):** MRR = 0.420 vs ft_03 fold 0 = 0.438 → **augmentation hurts**

> STORAL stories are modern moral tales (out-of-domain vs. Aesop fables). The distributional mismatch outweighs the additional training signal.

---

### ft_05 — BGE-base STORAL Mixing Ratio Sweep (fold 0)
Fast proxy experiment on BGE-base-en-v1.5 (110M, no LoRA) to find optimal augmentation size before committing GPU hours to Linq.

| STORAL pairs added | MRR (fold 0) |
|---|---|
| 0 (MORABLES only) | 0.277 |
| **200** | **0.287** ← best |
| 500 | 0.278 |
| 1,000 | 0.274 |
| 1,675 (all) | 0.274 |

> More STORAL = worse. Sweet spot ~200 pairs (+0.01 MRR). Hard negatives (ft_03) are a stronger technique than data augmentation for this task.

---

### ft_06 — Linq + LoRA, Random Hyperparameter Search (fold 0)
**Search space:** LoRA rank r ∈ {32, 64, 128} · temperature τ ∈ {0.02, 0.05, 0.07, 0.10} · lr ∈ {5e-5, 1e-4, 2e-4}  
**10 random trials** sampled, evaluated on fold 0. Best config → full 5-fold run.

#### Results so far (7/10 trials complete):

| Rank | Config | MRR (fold 0) |
|---|---|---|
| 1 | r=32, τ=0.02, lr=1e-4 | **0.439** |
| 2 | r=64, τ=0.02, lr=2e-4 | 0.435 |
| 3 | r=64, τ=0.05, lr=2e-4 | 0.428 |
| 4 | r=64, τ=0.05, lr=5e-5 | 0.422 |
| 5 | r=64, τ=0.05, lr=1e-4 | 0.427 |
| 6 | r=32, τ=0.07, lr=1e-4 | 0.417 |
| 7 | r=32, τ=0.05, lr=1e-4 | running... |

> **Emerging pattern:** Low temperature (τ=0.02) consistently outperforms — tighter contrastive signal forces harder discrimination between semantically similar morals. r=32 sufficient; larger rank (r=128) untested but likely to overfit.

---

## Fine-Tuning Journey — MRR Progression

```
0.21  ████████░░░░░░░░░░░░░░░░  Linq zero-shot (raw)
0.32  █████████████░░░░░░░░░░░  ft_02: LoRA fine-tune (raw)
0.36  ██████████████░░░░░░░░░░  Linq zero-shot (fable+summary)
0.42  █████████████████░░░░░░░  ft_02: LoRA fine-tune (fable+summary)
0.44  █████████████████░░░░░░░  ft_03: InfoNCE + moral negatives ← current best
0.44  █████████████████░░░░░░░  ft_06: best trial so far (fold 0 only)
0.89  ████████████████████████  Oracle ceiling
```

---

## Exp_11 — Zero-Shot Baseline: New Embedding Models

Testing whether newer models beat Linq zero-shot before committing to fine-tuning.  
**Corpus:** fable+summary · **Queries:** all 709 morals · **No fine-tuning**

| Model | Size | MRR | vs Linq (0.360) |
|---|---|---|---|
| Linq-Embed-Mistral *(reference)* | 7B | 0.360 | — |
| Nomic-Embed-v2-MoE | 475M active | 0.221 | −0.139 |
| Llama-Embed-Nemotron-8B | 7.5B | 0.086 | ❌ broken (bidirectional attn needs custom code) |
| NV-Embed-v2 | 7B | *running...* | |
| GTE-Qwen2-7B, BGE-en-ICL, Qwen3-8B, KaLM-12B, GritLM-7B | — | *pending* | |

---

## Next Steps

1. **ft_06 complete** → identify best hyperparams → full 5-fold run on GPU server
2. **ft_07** (planned): Best config from ft_06 + hard negatives (mine semantically similar fables)
3. **Exp_11** complete → any model beating Linq 0.360 zero-shot becomes a fine-tuning candidate
4. **EmbeddingGemma-300M** — pending HF access approval (gated repo)
