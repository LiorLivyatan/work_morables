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

## GPU Memory: How 7B/8B Models Fit on a 24 GB Card

Large LLM-based embedding models (Linq-Embed-Mistral 7B, Qwen3-Embedding 8B) are trained with
three stacked techniques that keep peak VRAM well under 24 GB:

| Technique | What it does | Memory saved |
|-----------|-------------|-------------|
| **LoRA** (r=64, α=128) | Only trains low-rank adapters on 4 attention projections (`q/k/v/o_proj`). ~7B params frozen → ~100M trainable. Optimizer state shrinks ~70×. | ~8–10 GB |
| **Gradient checkpointing** | Discards intermediate activations during forward pass; recomputes them during backprop. ~30% slower but roughly halves activation memory. | ~3–4 GB |
| **Small batch + accumulation** | `batch_size=4`, `gradient_accumulation_steps=4` → effective batch 16, but only 4 samples live in VRAM at once. | ~2–3 GB |

**Rough VRAM budget for Linq 7B:**

| Item | ~VRAM |
|------|-------|
| Model weights (bf16) | 14 GB |
| LoRA adapter gradients + optimizer | 3–4 GB |
| Activations (batch=4, seq=512, checkpointed) | 3–4 GB |
| **Total** | **~20–21 GB** |

BGE-base (110M params) uses no LoRA — full fine-tune fits in ~2 GB, so it runs at `batch_size=32`.

## Baseline Reference

| Setup | MRR | Source |
|-------|-----|--------|
| Off-the-shelf bge-base (no FT) | ~0.08 | Exp 01 |
| Best zero-shot (Linq-Embed-Mistral) | 0.210 | Exp 02 |
| + Gemini fable+summary (no FT) | 0.360 | Exp 07 |
| Oracle upper bound | 0.893 | Exp 03 |
