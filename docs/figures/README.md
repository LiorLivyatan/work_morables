# MORABLES — Experiment Figures

**Task:** Moral-to-Fable Retrieval — given a moral statement, retrieve the Aesop fable that best conveys it.  
**Dataset:** 709 Aesop fables, 709 moral queries.  
**Primary metric:** MRR@10 (Mean Reciprocal Rank at cutoff 10).  
**Oracle ceiling:** 0.893 · **Random baseline:** 0.002

---

## Figures

| Figure | File | What it shows |
|---|---|---|
| [Fig 1](fig1_zero_shot_baselines.md) | `fig1_zero_shot_baselines.png` | Zero-shot baselines: early models (Exp 01/02) and large models (Exp 11) |
| [Fig 2](fig2_finetuning_journey.md) | `fig2_finetuning_journey.png` | MRR progression across all fine-tuning experiments (ft_00 → ft_06) |
| [Fig 3](fig3_per_fold_heatmap.md) | `fig3_per_fold_heatmap.png` | Per-fold MRR heatmap for all 5-fold CV variants |
| [Fig 4](fig4_storal_mixing_ratio.md) | `fig4_storal_mixing_ratio.png` | STORAL augmentation mixing ratio sweep (ft_05, BGE-base proxy) |
| [Fig 5](fig5_ft06_hyperparam_search.md) | `fig5_ft06_hyperparam_search.png` | Random hyperparameter search results (ft_06, 6/10 trials) |

---

## Summary Table — Best Results Per Experiment

| Experiment | Model | Corpus | Mean MRR ± Std | vs Linq ZS |
|---|---|---|---|---|
| Zero-shot | Linq-Embed-Mistral (7B) | raw | 0.210 | — |
| Zero-shot | Linq-Embed-Mistral (7B) | fable+summary | 0.360 | reference |
| **ft_01** | BGE-base (110M) + full FT | raw | 0.122 ± 0.022 | — |
| **ft_02** | Linq (7B) + LoRA r=64 | raw | 0.318 ± 0.034 | +0.108 raw |
| **ft_02** | Linq (7B) + LoRA r=64 | fable+summary | 0.416 ± 0.033 | +0.056 |
| **ft_03** | Linq (7B) + LoRA r=64 + moral-neg | fable+summary | **0.438 ± 0.043** | **+0.078** |
| **ft_04** | Linq + LoRA + STORAL (fold 0) | fable+summary | 0.420 (fold 0) | +0.060 |
| **ft_06** | Linq + LoRA r=32, τ=0.02 (fold 0) | fable+summary | 0.439 (fold 0) | best so far |
| Oracle | Fable + ground-truth moral | — | 0.893 | upper bound |
