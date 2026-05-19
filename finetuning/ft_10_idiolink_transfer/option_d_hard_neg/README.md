# Option D — IdioLink Curriculum: Explicit Hard Negatives

Adapts IdioLink's same-PIE hard negative principle to MORABLES: near-duplicate morals
(sim > 0.85) become explicit hard negatives in the loss rather than masked-out pairs
(as in ft_09). No IdioLink text is used.

**Status:** Not started

**Contrast with ft_09:**
- ft_09: near-duplicate pairs → `-inf` mask (excluded from loss)
- Option D: near-duplicate pairs → active hard negative term (contributes to loss)

**Expected files:**
- `train.py` — modified MaskedInfoNCELoss with hard-negative weighting
- `config.yaml` — mirrors ft_09 config
- `results/` — 5-fold eval JSON + summary

See `../README.md` for full rationale and design notes.
