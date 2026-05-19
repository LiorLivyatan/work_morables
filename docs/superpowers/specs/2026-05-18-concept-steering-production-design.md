# Concept Steering — Production Run Design

**Date:** 2026-05-18
**Status:** Approved (brainstorm)
**Author:** Asif Amar
**Builds on:** `2026-05-09-concept-steering-retrieval-design.md`
**Predecessor run:** smoke (recal, 36 cells) — `analysis/08_concept_steering/results_smoke/`

## 1. Research claim

> For a concept X, scaling the CAA direction `v̂_X` into fable embeddings at the right layer causally and **specifically** changes retrieval of X-tagged morals — relative to (a) untagged morals and (b) directions built from shuffled-tag controls.

This is a **causal** claim. The primary statistic is the per-concept **specificity gap** `S = MRR_lift(tagged) − MRR_lift(untagged)`, with cluster-aware bootstrap 95% CIs (see §7).

### Sign convention (locked)

- CAA direction: `v = mean(h_pos − h_neg)` over matched pairs (`lib/vectors.py:93`)
- Intervention: `hs ← hs − α·v̂` at the output of block `layer_idx` (`lib/intervene.py:216`), where `v̂ = v / ‖v‖` is unit-normalised
- α has no a-priori "enhance" / "suppress" semantics — the empirical effect direction is read from the data. The smoke shows α=+15 → S=+0.20 and α=−15 → S=−0.17 for fox at layer 28, i.e. the α sweep is informative on both sides
- `passing_lift` (§7) tests `|S|_ci_lo > 0` (CI lower bound of effect *magnitude* excludes zero) so the test does not assume a sign direction

This locks in `intervene.py`'s current behaviour and removes the README's "α > 0 suppresses" claim, which is unsupported by the data. The README will be updated as part of the implementation (§10.5).

## 2. Shape of the experiment

**Exploratory landscape**, not a confirmatory single-concept run. The goal is a **(concept × layer × α)** map showing where in the model each concept is encoded and what α-range produces the cleanest causal lift. Per-concept rigor is light by design (uniform null protocol via the pipeline's existing `candidate_only` mode); breadth comes from 8 concepts spanning three abstraction levels.

A follow-up confirmatory run can pre-register the strongest 1–2 concepts from this landscape with deeper null controls.

## 3. Data — clustered corpora (mandatory)

Production uses **deduplicated, multi-target clustered data**, not the per-fable processed data the smoke ran on:

- `data/clustered/morals_unique_corpus.json` — 668 unique morals (vs. 709 raw, 41 dedup'd)
- `data/clustered/fables_corpus.json` — 709 fables (same set as processed; deduplication only affects morals)
- `data/clustered/qrels_moral_to_fable_clustered.json` — 1085 (moral, fable) pairs (avg 1.62 relevant fables per moral)
- `data/enriched/fable_elements.json` — unchanged (tags are per-fable)

**Why clustered:** semantically-identical morals map to multiple fables (e.g. "Appearances are deceptive" → 4 fables). The retrieval task is to find *any* of the relevant fables in the cluster, not a single canonical one. Per-concept tagged-moral counts go **up** (more statistical power), not down.

### Multi-target semantics

A moral is **X-tagged** iff at least one of its relevant fables is X-tagged.
MRR / Recall use **best rank among the relevant set** (standard multi-target IR).

Caveat: smoke results (recal, layer-28 fox peak at α=+15) were on per-fable processed data; production landscape is **not directly comparable** to the smoke. The smoke remains a valid end-to-end sanity check; production stands on its own.

## 4. Concept slate (8 concepts)

Selected for **decorrelation within level** and **diversity across levels**. Tagged-fable counts from `fable_elements.json`; tagged-moral counts on clustered morals (per §3 multi-target semantics).

| Field | Concept | n_tagged_fables | n_tagged_morals | Level |
|---|---|---|---|---|
| characters | fox | 67 | 95 | concrete entity |
| characters | lion | 55 | 95 | concrete entity |
| characters | wolf | 49 | 73 | concrete entity |
| setting | forest | 113 | 155 | concrete location |
| setting | field | 148 | 183 | concrete location |
| moral_category | deception | 87 | 115 | abstract theme |
| moral_category | justice | 66 | 81 | abstract theme |
| moral_category | pride | 47 | 66 | abstract theme |

**Why these:** smoke evidence for fox at layer 28 (independent of discovery, *on processed data*); `deception` was closest to discovery significance (p=0.071); minimum n_tagged_morals = 66 (well above `min_matched_pairs: 15`). FDR significance from `run_discovery.py` is **not** a gate — discovery was conservative over 14 tags.

**Placebo strategy:** the pipeline's built-in **shuffled-tag CAA** and **random-direction** null controls run *per concept on candidate cells*; no separate placebo concept is needed.

## 5. Sweep grid

- **Layers (6):** `[4, 12, 20, 24, 28, 31]`
  - Layer 4: shallow negative control (expected near-null)
  - Layer 12: mid (smoke: null for fox)
  - Layer 20, 24, 28: dense in the deep range where smoke fired
  - Layer 31: final hidden layer (directly feeds the mean-pooled output)
- **Alphas (10):** `[-20, -10, -5, -3, 0, +3, +5, +10, +15, +20]`
  - Calibrated from recal smoke: +3 already significant, +15 peak, +30 over-steers → cap at ±20
  - Symmetric for direction-of-effect
  - α=0 is a no-op short-circuit in the pipeline
- **Cells per concept:** 6 × 10 = 60 (≈54 real after α=0 short-circuit)
- **Total cells:** 8 × 60 = 480 (≈432 real)

## 6. Compute & time budget

- Realistic pace from recal smoke: **~50–60 sec/cell** on RTX 3090 (each cell does *two* encodes of 709 fables — baseline + intervened — see `lib/intervene.py:203,229`). The earlier 30s estimate was optimistic.
- **Sweep:** 432 real cells × ~55 s ≈ **6.5 hours**
- **Nulls (candidate-only):**
  - 50 shuffled-tag perms × 2 encodes + 5 random-direction seeds × 2 encodes per passing cell
  - Estimate 1–3 passing cells per concept → 8–24 candidate cells across the run
  - 24 × 55 perms × ~55 s ≈ +1.5–2 hours
- **Total: ~8–10 hours** end-to-end.

Optimisation deferred (would cut ~40% off the sweep, see §10.3): `encode_with_intervention` currently recomputes `base_embs` per cell even though baseline embeddings already exist in `ranks_baseline.json` cache. Not blocking — overnight run fits even without it. Flagged in §12.

Overnight on GPU 0 or 2.

## 7. Success criteria — exploratory reporting

The binary Stage-2 gate is **fully disabled** when `concepts.placebo: []` (see §10.4). Per (concept, layer, α) cell, compute three flags:

- `passing_lift`: `min(S_ci_lo, −S_ci_hi) > 0` — i.e. CI of the specificity-gap *magnitude* excludes zero (sign-agnostic; see §1)
- `passing_null` (**primary causal null = shuffled-tag CAA**): when the shuffled-tag null ran, `S_median > null_hi` OR `S_median < null_lo`. The random-direction null is reported as a *secondary diagnostic* but is not part of the strict-pass definition
- `passing_strict`: both of the above

**Cluster-aware bootstrap:** CIs are computed over **moral_unique IDs** (668 unique morals). Because clustering already deduplicated identical morals into one ID each, resampling at the moral_unique level *is* cluster-level resampling for the moral side. We additionally log the per-moral relevant-set size; any cell whose `passing_strict` flips when restricted to singleton-cluster morals (`relevant_fables == 1`) is flagged in the report as cluster-fanout-sensitive.

Per-concept summary row: best (layer, α), peak |S_median|, CI, three flags, null status. No experiment-level go/no-go gate — every concept's verdict stands alone.

## 8. Headline outputs

Two figures + one table, all generated post-hoc from `specificity_summary.json` via a new `analysis/08_concept_steering/report.py`. The existing pipeline already produces all data needed.

- **Figure 1 — Landscape:** 2×4 grid of 8 small heatmaps. Each cell: rows = layers (4→31, low→high), columns = α (-20→+20), color = S_median, asterisks marking passing cells.
- **Figure 2 — Dose-response:** per-concept line plot of S_median vs α, one line per layer (6 lines/panel). Shows the *shape* of the effect.
- **Table:** Markdown, one row per concept. Columns: concept, n_tagged, best (layer, α), peak S_median, CI, passing_lift, passing_null, passing_strict.

The existing `specificity_summary.png` (concept × layer heatmap from the smoke pipeline) is kept as a backup but is not the headline.

## 9. Operational plan

### Storage (CLAUDE.md mandate)

```yaml
output:
  results_dir: analysis/08_concept_steering/results
  cache_dir: analysis/08_concept_steering/cache
  save_intermediate_embeddings: false

model_output_dir: /data/lior/08_concept_steering/models
```

- Final models / artefacts → `/data/lior/...` (long-term)
- Hidden-state cache, CAA vectors, per-cell rank JSONs → `analysis/08_concept_steering/results/` and `cache/` (physical disk, fast I/O during run)
- `results_smoke/` is kept untouched as the smoke artefact

### Pre-launch checks (mandatory)

```bash
./run.sh status                                       # GPU availability
ssh "$GPU_USER@$GPU_HOST" "df -h ~ && df -h /data/lior"   # disk
```

Abort if physical disk < 30 GB free.

### Launch

```bash
./run.sh analysis/08_concept_steering/run_intervention.py \
    --config analysis/08_concept_steering/config.yaml \
    --remote --gpu <free GPU>
```

- Telegram notifications fire on start, per-concept milestones, and completion (existing pipeline behavior).
- Monitor with `./run.sh watch --gpu N` if needed.
- Pause: `./run.sh pause --gpu N` (resumes from per-cell checkpoint on rerun).

### Post-run

1. `./run.sh pull` to sync results JSONs locally.
2. Run `analysis/08_concept_steering/report.py --results results/ --out reports/` to generate Figure 1, Figure 2, and the summary table.
3. Commit results JSON files + figures + table to git.
4. Decide on confirmatory follow-up based on landscape (which concepts to deep-test, which layers).

## 10. Changes to existing code

### 10.1 Multi-target qrels refactor (required before launch)

The current code path assumes one ground-truth fable per moral. Clustered qrels are multi-target. Every call site touched:

**Library:**
- `lib/data.py`
  - `_qrels_to_map(qrels)` → `dict[str, list[str]]` (preserve all relevant fables; filter `relevance==0`)
  - `Corpus.gt_fable_idx: list[int]` → `Corpus.gt_fable_idxs: list[list[int]]` (one inner list per moral; never empty)
  - `load_corpus` builds `gt_fable_idxs` via dict→list lookup
- `lib/retrieval.py`
  - `mrr_at_k(rankings, gt, k)`: accept `gt` as list of int-lists (or `np.ndarray[object]`); per query, `rr = 1/(best_position+1)` where `best_position = min({i : rankings[q,i] ∈ gt_set})`
  - `group_mrr(...)` unchanged signature, threads new structure
- `lib/eval.py`
  - `reciprocal_rank_per_query(rankings, gt, k=10)` same multi-target update; preserve `mrr_at_10` label (rank capped at k=10)
- `lib/intervene.py:46-49`
  - `ranks_int[i]` from `np.where(rankings[i] == gt_indices[i])` → multi-target `np.where(np.isin(rankings[i], gt_set))`; record *best* rank per moral
- `lib/vectors.py`
  - No semantic change. `pos_baseline_mrr` and `neg_baseline_mrr` (logged in matched-pair quality) come from `mrr_at_k` so they inherit the multi-target semantics. Verify in test.

**Run scripts (all three currently hardcode `data/processed/qrels_moral_to_fable.json` — must be replaced with `cfg["data"]["qrels_path"]`):**
- `run_baseline.py:36` — qrels_path from config
- `run_baseline.py:60` — `gt = np.array(corpus.gt_fable_idx)` → object array
- `run_baseline.py:72-75` — `ranks_baseline.json` schema: `gt_fable_idx: int` → `gt_fable_idxs: list[int]`; `gt_rank: int` becomes "best rank over relevant set"
- `run_discovery.py:41` — qrels_path from config
- `run_discovery.py:36` — `failed_doc_ids` reads `gt_fable_doc_ids` (list); failure definition: a moral fails iff *best-relevant rank > 1*
- `run_discovery.py:60` — gt object array
- `run_intervention.py:192` — qrels_path from config
- `run_intervention.py:91, 152, 248, 295` — `target_mask = any(set(gt_idxs) & pos_indices_set for gt_idxs in ...)` (any-of-relevant tagged)
- `run_intervention.py:250-253` — `rr_baseline` reconstruction from new schema; assertion that α=0 cell reproduces baseline exactly

**Pre-launch corpus assertions (added to each run script's startup):**
```python
assert len(corpus.moral_ids) == 668, f"expected 668 unique morals, got {len(corpus.moral_ids)}"
assert sum(len(g) for g in corpus.gt_fable_idxs) == 1085, "expected 1085 multi-target qrel pairs"
assert all(len(g) >= 1 for g in corpus.gt_fable_idxs), "every moral must have ≥1 relevant fable"
```

**Tests** (`analysis/08_concept_steering/tests/test_multi_target_qrels.py`):
- Fixture: 3 morals × 5 fables, gt sizes [1, 2, 3]
- `_qrels_to_map`: returns multi-target dict
- `mrr_at_k` with k=3: hand-computed best-rank MRR matches
- `reciprocal_rank_per_query`: per-query best-rank reciprocal matches
- `load_corpus` end-to-end on the fixture: schema is `list[list[int]]`

### 10.2 Config

- `data.morals_path`: `data/clustered/morals_unique_corpus.json`
- `data.fables_path`: `data/clustered/fables_corpus.json` (unchanged content; explicit for clarity)
- Add `data.qrels_path: data/clustered/qrels_moral_to_fable_clustered.json` (replace the hardcoded `data/processed/qrels_moral_to_fable.json` in `run_baseline.py`, `run_discovery.py`, `run_intervention.py`)
- Populate `concepts.targets` (the 8 above) and `concepts.placebo: []`
- Update `intervention.alphas` to `[-20, -10, -5, -3, 0, +3, +5, +10, +15, +20]`
- `vectors.layers: [4, 12, 20, 24, 28, 31]`
- Add `model_output_dir: /data/lior/08_concept_steering/models`

### 10.3 Reporting

- **New: `analysis/08_concept_steering/report.py`** — reads `specificity_summary.json`, produces Figure 1 (landscape heatmap grid), Figure 2 (dose-response panels), and a Markdown summary table. Computes the three `passing_*` flags from raw S CIs and null envelopes.

### 10.4 Exploratory mode — fully disable Stage-2 gate

When `concepts.placebo: []`, `run_intervention.py` **must not** call `stage2_go_no_go(...)`. Instead, write:

```json
{"mode": "exploratory", "go": null, "reasons": ["exploratory landscape; per-concept verdicts in specificity_summary.json"], "targets_passing": []}
```

to `stage2_decision.json`. The four-condition gate (designed for `target == 1, placebo == 1`) is meaningless here and its passing logic is sign-mismatched against the empirical data (existing gate uses `S_ci_hi < 0` which only fires for negative effects; smoke shows positive effects). Disabling cleanly is safer than "graceful degradation."

`run_intervention.py` change: guard the call site with `if cfg["concepts"].get("placebo"):` and skip the gate when empty. Per-concept `passing_lift` / `passing_null` / `passing_strict` flags are computed directly in `specificity_summary.json` (§7).

### 10.5 Sign-convention documentation

- `README.md`: remove the "α > 0 suppresses" line; replace with the locked convention from §1
- `lib/plotting.py`: axis label "(suppress →)" → "α"  (no implied direction)
- `lib/intervene.py:216`: add a 2-line comment pinning the convention and pointing to spec §1

## 11. Acceptance criteria — what "done" means

This experiment is considered done only when ALL of the following hold:

1. **Real steering applied.**
   - hook activation log present in `results/run_log.json`
   - α=0 short-circuit reproduces baseline rankings exactly (`np.allclose(α0_rr, baseline_rr)`)
   - **Pooled-cosine non-trivial-intervention assertion**: at least one α≠0 cell in the landscape has `pooled_cosine_mean < 0.999` (else the hook isn't measurably changing embeddings). If this fails the run is **aborted** with a clear error.

2. **Clustered data used end-to-end.**
   - `lib/data.py` returns `gt_fable_idxs: list[list[int]]` and tests in `tests/test_multi_target_qrels.py` pass
   - Pre-launch corpus assertions (§10.1) hold: `len(corpus.moral_ids)==668`, `sum(len(g) for g in corpus.gt_fable_idxs)==1085`
   - Baseline MRR on clustered data is logged and **strictly higher** than baseline MRR on processed data (the multi-target rank-of-best is bounded below by the single-target rank). Magnitude expected ≥ +0.02.

3. **Full sweep completed.** All 432 real cells written to `results/ranks_intervened/`. 8 concept-vector files + 8 metadata files in `results/concept_vectors/`. **Hard fail** if any concept's matched-pair count `< 15` (would silently skip in current code; assert and abort).

4. **Null controls ran on all candidate cells.** For every cell with `passing_lift=True`, the shuffled-tag CAA null (50 perms) **and** random-direction null (5 seeds) executed; `null_lo`/`null_hi` populated. If zero cells `passing_lift`, this is logged but not a failure.

5. **Reporting artefacts produced and committed.**
   - `results/specificity_summary.json` populated for all 8 × 6 × 10 cells
   - `reports/figure1_landscape.png` (8 heatmaps), `reports/figure2_dose_response.png` (8 panels), `reports/summary_table.md`
   - All three committed to git

6. **Clear outcomes documented.** `analysis/08_concept_steering/RESULTS.md` includes, for each concept:
   - Verdict: `passing_strict` cell count, peak (layer, α), peak |S|, CI, primary (shuffled-tag) null status
   - Depth pattern: which layers show signal? matches abstraction level?
   - **Limitations subsection**: concept-tag correlations (fox↔deception known), bootstrap honesty (no FDR over cells), cluster-fanout-sensitive cells flagged, "what the α sign means" tied to spec §1
   - Next confirmatory run (if any)

7. **Abstract-concept lexical-confound diagnostic** (deception, justice, pride): for each abstract concept's CAA direction at the peak layer, log cosine similarity to a simple lexical baseline (mean embedding of fables tagged with that concept minus mean embedding of all fables). If the CAA direction is ≥ 0.95 cosine to this lexical baseline at the peak layer, flag the concept in RESULTS.md as "lexically dominated."

8. **Bootstrap honesty**: cluster-fanout sensitivity (per §7) is reported; any cell whose verdict flips when restricted to singleton-cluster morals is flagged.

## 12. Open questions deferred to writing-plans

- Whether `run_intervention.py`'s Stage-2 decision logic needs an explicit `mode: exploratory` switch or whether `placebo: []` is enough to bypass it cleanly.
- Whether the cell-level flag computation (`passing_lift`, `passing_null`, `passing_strict`) is added to `specificity_summary.json` at run time or computed in `report.py` from raw CIs and nulls.
- Whether `report.py` is one script or two (figures vs. table).
- Whether to also regenerate `run_discovery.py`'s output on clustered qrels (recommended, since the baseline failure rate may shift and concept p-values with it — but discovery is not gating the slate, so this is informational only).

## 13. Out of scope

- Confirmatory follow-up run with pre-registered targets (post-landscape decision).
- Comparison to non-steering interventions (e.g. fine-tuning, prompting). Concept-steering only.
- Other models. Linq-Embed-Mistral only.
- Concepts with n_tagged < 20 (statistical power too low for clean CIs).
- Multi-concept simultaneous steering (e.g. `α₁·v_fox + α₂·v_deception`). Single-concept only.
