# ft_02_linq_5fold_cv — Linq-Embed-Mistral LoRA Fine-tuning

## Purpose

Fine-tune the strongest model from the retrieval pipeline (Linq-Embed-Mistral, 7B)
on MORABLES using LoRA adapters, then evaluate with 5-fold cross-validation.

Uses the **identical splits** as `ft_01_5fold_cv` for direct comparison:
- ft_01: bge-base-en-v1.5 (110M), full FT → MRR ≈ 0.11
- ft_02: Linq-Embed-Mistral (7B), LoRA FT → target MRR > 0.21 (baseline)

Key design choices:
- **LoRA (r=64)**: trains ~1% of parameters, strong regularization for 567-pair dataset
- **Query instruction**: task-specific prefix prepended to moral queries only
- **Early stopping (patience=3)**: halts when per-epoch MRR plateaus, saves best epoch
- **Shared splits**: `ft_01_5fold_cv/cache/splits/folds.json` — do not regenerate

## How to Run

```bash
# Run all 5 folds (raw fables):
python finetuning/ft_02_linq_5fold_cv/train.py

# Run with fable+summary documents:
python finetuning/ft_02_linq_5fold_cv/train.py --doc_mode fable_plus_summary

# Run a single fold for quick iteration:
python finetuning/ft_02_linq_5fold_cv/train.py --fold 0

# Re-train even if cached:
python finetuning/ft_02_linq_5fold_cv/train.py --force

# Disable wandb:
python finetuning/ft_02_linq_5fold_cv/train.py --no-wandb

# Standalone evaluation (requires cached models):
python finetuning/ft_02_linq_5fold_cv/evaluate.py
```

## Resuming an Interrupted Run

Checkpoints saved after every epoch to `cache/checkpoints/<doc_mode>/fold_N/`.
Just re-run **without** `--force` to resume automatically:

```bash
python finetuning/ft_02_linq_5fold_cv/train.py
```

## Caching

| Path | Contents | Committed? |
|------|----------|------------|
| `../ft_01_5fold_cv/cache/splits/folds.json` | Fold indices (shared) | **Yes** |
| `cache/models/<doc_mode>/fold_N/` | Merged model (LoRA baked in) | No (large) |
| `cache/checkpoints/<doc_mode>/fold_N/` | Per-epoch trainer state | No (large) |
| `cache/embeddings/<doc_mode>/fold_N/` | Corpus + query embeddings | No (large) |
| `results/*.json` | Per-fold metrics | Yes |

## Hardware Notes

Linq-Embed-Mistral (7B) loaded in bfloat16 ≈ 14 GB. With LoRA adapters and
optimizer state, total memory usage ≈ 16–20 GB. Tested on M4 Pro 64 GB.
Training one fold ≈ 20–40 min depending on early stopping.
