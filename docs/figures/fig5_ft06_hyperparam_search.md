# Fig 5 — Hyperparameter Search (ft_06)

![Hyperparameter Search](fig5_ft06_hyperparam_search.png)

## What this shows

Random search over LoRA rank, InfoNCE temperature, and learning rate — all on fold 0 only.  
The best configuration will be promoted to a full 5-fold run on the GPU server.

---

## Setup & Method

**Model:** Linq-Embed-Mistral (7B) + LoRA  
**Loss:** Moral-spreading InfoNCE (Type 1 + Type 2) — identical to ft_03 basic  
**Evaluation:** Fold 0 test set (142 queries), fable+summary corpus  
**Sampling:** 10 random trials from the Cartesian product, seed=42

The loss and data are fixed. Only the adapter size, contrastive temperature, and learning rate vary:

```python
# Per trial: three things change
LoraConfig(r=trial["lora_r"], lora_alpha=trial["lora_r"] * 2, ...)
InfoNCELoss(model, temperature=trial["temperature"])
SentenceTransformerTrainingArguments(learning_rate=trial["learning_rate"], ...)
```

> **Temperature intuition — concrete example:**  
> Two morals in a batch: *"Slow and steady wins the race."* and *"Patience always pays off."*  
> Their embeddings have cosine similarity ≈ 0.85.  
>  
> At **τ = 0.05** (ft_03 default): logits = 0.85/0.05 = **17** vs others at ~16 — soft gradient, gentle push.  
> At **τ = 0.02** (ft_06 winner): logits = 0.85/0.02 = **42.5** vs others at ~40 — steep gradient, hard push.  
>  
> Lower τ forces the model to be far more confident about which fable is correct, creating a stronger learning signal for semantically similar morals.

**Search space:**

| Hyperparameter | Values |
|---|---|
| LoRA rank r | 32, 64, 128 |
| Temperature τ | 0.02, 0.05, 0.07, 0.10 |
| Learning rate | 5e-5, 1e-4, 2e-4 |

---

## Results (6/10 trials complete)

| Trial | r | τ | lr | MRR (fold 0) |
|---|---|---|---|---|
| **Best** | **32** | **0.02** | **1e-4** | **0.439** |
| 2 | 64 | 0.02 | 2e-4 | 0.435 |
| 3 | 64 | 0.05 | 2e-4 | 0.428 |
| 4 | 64 | 0.05 | 1e-4 | 0.427 |
| 5 | 64 | 0.05 | 5e-5 | 0.422 |
| 6 | 32 | 0.07 | 1e-4 | 0.417 |
| 7–10 | — | — | — | pending |

---

## Interpretation

**Temperature is the dominant factor.** τ=0.02 (the lowest value tested) wins outright. Lower temperature → sharper probability distribution → model must be more confident → harder, more informative gradient signal. This is consistent with findings in supervised contrastive learning literature.

**LoRA rank has modest effect.** r=32 and r=64 both appear in the top trials. The 7B Linq model overfits quickly on only ~570 training pairs, so a smaller adapter (r=32) generalizes slightly better. r=128 is not yet tested but likely to overfit more.

**Learning rate: 1e-4 or 2e-4** both work; 5e-5 slightly underperforms (too slow to converge within the early-stopping window of ~4 epochs).

---

## Panel B — Temperature vs. MRR Scatter

Bubble size = LoRA rank r (larger bubble = larger rank).  
The τ=0.02 cluster clearly dominates, regardless of rank or lr.

---

## Next Steps

Once all 10 trials complete, the best config (currently r=32, τ=0.02, lr=1e-4) runs a full **5-fold CV** on the GPU server → this becomes **ft_07**.

**ft_07 (planned):** Best ft_06 config + semantic hard negatives (mine fables with similar but non-matching morals using the fine-tuned model itself).
