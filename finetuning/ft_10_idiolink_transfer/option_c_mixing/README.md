# Option C — Data Mixing

Mix IdioLink triplets into MORABLES training batches at varying ratios. Single-stage,
5-fold CV. Inspired by ft_05 (STORAL mixing sweep) but with richer IdioLink negatives.

**Status:** Not started

**Ratios to sweep (fold 0 first):** 50, 110, 220, 440 pairs
**Expected sweet spot:** well below 50% based on ft_05 STORAL findings.

**Expected files:**
- `train.py` — accepts `--ratio N` arg; mixes N IdioLink triplets per fold
- `config.yaml` — base hyperparams (mirrors ft_09)
- `results/` — per-ratio result JSONs

See `../README.md` for full rationale and design notes.
