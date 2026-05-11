# Option B — Sequential Pre-training

Stage 1: fine-tune Linq+LoRA on IdioLink (440 triplets).
Stage 2: continue fine-tuning on MORABLES (5-fold CV, ft_09 settings).

**Status:** Not started

**Key design decision:** merge LoRA after Stage 1 before Stage 2? See `../README.md`.

**Expected files:**
- `pretrain.py` — Stage 1: IdioLink fine-tuning, saves merged model to disk
- `finetune.py` — Stage 2: MORABLES 5-fold CV from Stage 1 checkpoint
- `config_pretrain.yaml` — Stage 1 hyperparams
- `config_finetune.yaml` — Stage 2 hyperparams (mirrors ft_09)
- `results/` — Stage 2 per-fold eval JSON + summary

See `../README.md` for full rationale and design notes.
