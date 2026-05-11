# Option A — Zero-Shot Transfer

Train Linq+LoRA on IdioLink training data only (440 triplets). Evaluate on MORABLES 5-fold CV.

**Status:** Not started

**Expected files:**
- `train.py` — loads IdioLink triplets, trains with InfoNCE, saves model
- `eval.py` — runs the saved model through MORABLES 5-fold evaluation
- `config.yaml` — model/LoRA/training hyperparams
- `results/` — per-fold eval JSON + summary

See `../README.md` for full rationale and design notes.
