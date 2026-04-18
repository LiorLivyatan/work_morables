# Fine-Tuning Experiments

Contrastive fine-tuning of bi-encoder models on the MORABLES moral-to-fable
retrieval task. Each sub-directory is a self-contained experiment.

## Experiments

| Dir | Stage | Description | Status |
|-----|-------|-------------|--------|
| `ft_00_overfit/` | 0 | Train=test upper bound (sanity check) | Ready |
| `ft_01_5fold_cv/` | 1 | 5-fold CV on raw MORABLES pairs | Ready |

## Shared Library (`lib/`)

| Module | Responsibility |
|--------|---------------|
| `lib/data.py` | Load MORABLES pairs; `build_doc_text()` for document representation |
| `lib/trainer.py` | `train_model()` — SentenceTransformer fine-tuning with caching |
| `lib/eval.py` | `evaluate()` — encode + compute MRR/R@k with embedding caching |

## Adding a New Experiment

1. Copy an existing experiment dir as a template
2. Update `config.yaml` for your specific setup
3. Write `train.py` using the shared `lib/` — keep it a thin orchestrator
4. Add a `README.md` documenting purpose, how to run, and expected results
5. Add an entry to this table

## Baseline Reference

| Setup | MRR | Source |
|-------|-----|--------|
| Off-the-shelf bge-base (no FT) | ~0.08 | Exp 01 |
| Best zero-shot (Linq-Embed-Mistral) | 0.210 | Exp 02 |
| + Gemini fable+summary (no FT) | 0.360 | Exp 07 |
| Oracle upper bound | 0.893 | Exp 03 |
