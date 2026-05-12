# 08_concept_steering — Activation-level concept suppression for moral→fable retrieval

**Spec:** `docs/superpowers/specs/2026-05-09-concept-steering-retrieval-design.md`
**Plan:** `docs/superpowers/plans/2026-05-09-concept-steering-v0.md`

## What this is

A v0 white-box experiment that asks: if we suppress a specific concept in the
fable representation of a vanilla embedding model, does retrieval change in a
way that is concept-specific?

Pipeline:

1. **Discovery** — Fisher's exact + BH-FDR on metadata tags from
   `data/enriched/fable_elements.json` finds tag values whose presence is
   statistically overrepresented in retrieval failures.
2. **Vector build** — for 3 targets + 1 difficulty-matched placebo, CAA
   matched-pair concept vectors built from hidden states at 5 layers (with
   `mean_diff` byproduct as a sanity check).
3. **Intervention** — for each (concept × layer × α), `h ← h − α·v_C` is
   injected on the residual stream and fables re-encoded.
4. **Validation** — paired bootstrap CI of the specificity gap
   `S = ΔMRR_target − ΔMRR_control`, plus shuffled-tag null controls at
   candidate cells.

Sign convention: `α > 0` suppresses the concept, `α < 0` amplifies.

## Run order

All scripts run via `./run.sh` (project policy in CLAUDE.md). Both `--remote`
and `--gpu N` work transparently. Telegram notifications are sent at start and
end of each stage when `TG_BOT_TOKEN` and `TG_CHAT_ID` env vars are set.

1. **Baseline**

   ```bash
   ./run.sh analysis/08_concept_steering/run_baseline.py --remote --gpu 2
   ```

   Writes `results/ranks_baseline.json`. Caches per-text embeddings.

2. **Discovery** (no GPU needed)

   ```bash
   ./run.sh analysis/08_concept_steering/run_discovery.py
   ```

   Writes `results/discovery_report.json`. Updates `concepts.targets` and
   `concepts.placebo` in `config.yaml`.

3. **Intervention sweep + nulls + summary + plot** (overnight)

   ```bash
   ./run.sh analysis/08_concept_steering/run_intervention.py --remote --gpu 2
   ./run.sh pull
   ```

   Outputs:
   - `results/specificity_summary.png` — the headline figure
   - `results/specificity_summary.json` — full numerical results
   - `results/stage2_decision.json` — four-condition go/no-go
   - `results/concept_vectors/*.npy` — saved concept directions
   - `results/concept_vectors/*.meta.json` — matched-pair quality logs
   - `results/ranks_intervened/*.json` — per-cell rankings + rr

## Smoke run

`config_smoke.yaml` runs 1 target + 1 placebo across 2 layers and 3 α values
into `results_smoke/`. Useful for end-to-end verification:

```bash
./run.sh analysis/08_concept_steering/run_baseline.py     --config analysis/08_concept_steering/config_smoke.yaml --remote --gpu 2
./run.sh analysis/08_concept_steering/run_discovery.py    --config analysis/08_concept_steering/config_smoke.yaml
./run.sh analysis/08_concept_steering/run_intervention.py --config analysis/08_concept_steering/config_smoke.yaml --remote --gpu 2
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| All target curves flat (S ≈ 0 across α) | Layer choice misses where the concept lives | Inspect `concept_vectors/<name>.meta.json` `cos_caa_meandiff_per_layer` — a small magnitude suggests the concept is weakly captured at that layer |
| Pooled cosine ≈ 1.0 yet MRR moves | Intervention only changing magnitude; under cosine retrieval this is a bug | Verify hook code in `lib/model.py:encode_with_intervention` is actually mutating the residual stream |
| Placebo also drops | Placebo not difficulty-matched, or random direction at that layer also drops | Inspect `discovery_report.json` for an alternative placebo with the same baseline MRR |
| OOM | bfloat16 + 7B + batch_size 8 too tight | Drop `model.batch_size` to 4 in config |
| `_get_layer_module` raises | Architecture differs from Mistral/Llama | Extend `_get_layer_module` in `lib/model.py` for the new layer-list path |

## Architecture

- `lib/model.py` is the **only** file that imports `transformers` or registers
  hooks. Pooling kind and layer count are detected at load time and logged.
- `lib/{retrieval, discovery, vectors, eval, nulls, plotting, io}.py` are pure
  Python over numpy/pandas — no torch.
- `run_*.py` are thin CLI wrappers that read the config and call into `lib/`.

Adding a new vector method (e.g. LEACE): one function in `lib/vectors.py` and
one entry under `vectors.methods.sanity_byproducts` in the config.
