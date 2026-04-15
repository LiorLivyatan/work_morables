# ft_00_overfit — Upper Bound Check

## Purpose

Train on all 709 MORABLES pairs and evaluate on the same set. This is intentional
data leakage — the model is expected to memorise the pairs, giving MRR → ~1.0.

**Why?** Before running cross-validation (ft_01), this verifies that:
1. The training pipeline is correctly wired (loss, data format, optimizer)
2. The model *can* learn the moral-fable mapping given enough signal
3. We have a meaningful upper bound to benchmark CV results against

If MRR doesn't approach 1.0 here, something is broken in the pipeline.

## How to Run

```bash
# Raw fables (default)
python finetuning/ft_00_overfit/train.py

# Fable + Gemini summary
python finetuning/ft_00_overfit/train.py --doc_mode fable_plus_summary

# Re-train even if a cached model exists
python finetuning/ft_00_overfit/train.py --force
```

## Caching

| Path | Contents | Committed? |
|------|----------|------------|
| `cache/models/<doc_mode>/` | Trained SentenceTransformer weights | No (large) |
| `cache/embeddings/<doc_mode>/` | Final corpus + query embeddings (.npy) | No (large) |
| `results/*.json` | MRR + full metrics per run | Yes |

On re-run without `--force`, the cached model and embeddings are loaded directly —
no retraining needed.

## Expected Results

| doc_mode | Baseline MRR | Post-train MRR |
|----------|-------------|----------------|
| raw | ~0.08 | ~0.90–1.0 |
| fable_plus_summary | ~0.35 | ~0.95–1.0 |

The baseline MRR for `fable_plus_summary` is higher because the encoded documents
already contain semantic content closer to the moral queries (from the Gemini summary).
