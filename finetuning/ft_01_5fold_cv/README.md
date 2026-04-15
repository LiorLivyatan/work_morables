# ft_01_5fold_cv — 5-Fold Cross-Validation on MORABLES

## Purpose

Evaluate how well a fine-tuned bi-encoder generalises within the MORABLES dataset.
Each fold trains on ~567 pairs and evaluates on ~142 held-out morals, retrieving
from the full 709-fable corpus. Final MRR = mean ± std across 5 folds.

Two document representation variants are compared:
- `raw` — fable text only
- `fable_plus_summary` — fable + Gemini cot_proverb summary (best from Exp 07)

## How to Run

```bash
# Step 1 — generate splits (one-time, commit the output)
python finetuning/ft_01_5fold_cv/prepare_data.py

# Step 2 — run all 5 folds
python finetuning/ft_01_5fold_cv/train.py
python finetuning/ft_01_5fold_cv/train.py --doc_mode fable_plus_summary

# Run a single fold for quick iteration
python finetuning/ft_01_5fold_cv/train.py --fold 0

# Re-train even if cached
python finetuning/ft_01_5fold_cv/train.py --force
```

## Resuming an Interrupted Run

Training checkpoints are saved after every epoch to `cache/checkpoints/<doc_mode>/fold_N/`.
If a run is interrupted (Ctrl-C, crash, or manual kill), **just re-run the same command without `--force`**:

```bash
# Resumes automatically from the last completed epoch
python finetuning/ft_01_5fold_cv/train.py
```

The trainer detects the latest checkpoint via `get_last_checkpoint()` and passes it to
`trainer.train(resume_from_checkpoint=...)`. No data is lost.

> **Note (April 2026):** The `raw` doc_mode run was paused mid-way through fold 2
> (epoch ~8/30) to pivot to fine-tuning Linq-Embed-Mistral. Fold 0 is complete
> (MRR=0.1118). Resume with the command above when needed.

## Caching

| Path | Contents | Committed? |
|------|----------|------------|
| `cache/splits/folds.json` | Fold indices (seed=42, reproducible) | **Yes** |
| `cache/models/<doc_mode>/fold_N/` | Trained model per fold | No (large) |
| `cache/embeddings/<doc_mode>/fold_N/` | Corpus + query embeddings (.npy) | No (large) |
| `results/*.json` | Per-fold metrics + summary | Yes |

The splits file is committed so all collaborators use identical folds. Model weights
and embeddings are cached locally for re-evaluation but not pushed.

## Expected Results

| doc_mode | Expected MRR | Notes |
|----------|-------------|-------|
| raw | 0.28–0.35 | Fine-tuning from exp07 plan estimate |
| fable_plus_summary | 0.35–0.45 | Richer document representation may help |

Baseline (no fine-tuning): MRR ≈ 0.21 (raw) / 0.36 (fable+summary, from Exp 07).
