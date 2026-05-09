# Concept Steering for Moral→Fable Retrieval — Design

**Date:** 2026-05-09
**Owner:** Asif Amar
**Status:** Draft, awaiting user review
**Target experiment dir:** `analysis/08_concept_steering/`

## 1. Motivation

Vanilla and fine-tuned embedding models on MORABLES leave a large gap between achieved MRR (~0.21 vanilla, ~0.44 ft07) and the oracle ceiling (~0.89). The information needed to retrieve the right fable is present in the embedding space — the model just fails to project it the right way for many queries. We want to test a white-box hypothesis: **specific concepts in the fable representation are over-anchoring retrieval, and selectively suppressing them changes retrieval behavior in predictable, concept-specific ways.**

If true, this gives us (a) a diagnostic tool for *why* retrieval fails on certain fables, and (b) the foundation for follow-up work that uses concept-aware interventions to *fix* retrieval failures.

## 2. Research question

> If we suppress the internal representation of a specific concept (discovered from data, not hand-picked) inside an encoder used for retrieval, does retrieval behavior change in a way that is **specific to fables that involve that concept**, while leaving other fables unaffected?

## 3. Pipeline

```
Step 1: BASELINE        → encode 709 morals + 709 fables with vanilla
                          Linq-Embed-Mistral; compute moral→fable rankings.

Step 2: DISCOVERY       → split queries into success/failure (rank-1 hit vs miss);
                          for each metadata tag value, compute failure-overrep
                          via Fisher's exact test (FDR-corrected); rank concepts.

Step 3: VECTOR BUILD    → for 4 chosen concepts, extract hidden states at 5 layers
                          for all 709 fables, build CAA matched-pair v_C plus a
                          plain mean-difference v_C as a comparison byproduct.

Step 4: INTERVENTION    → for each (concept, layer, α), re-encode 709 fables with
                          a forward hook injecting h ← h − α·v_C at the target
                          layer; recompute moral→fable rankings.

Step 5: SPECIFICITY     → for each (concept, layer, α), report ΔMRR on C-tagged
                          fables vs ΔMRR on non-C-tagged fables; produce one
                          summary figure with target/control/placebo curves.
```

The pipeline ends at v0 success/failure. A follow-up experiment (out of scope here) will run the *confusion-shift* (Stage 2) test on whichever concepts pass Stage 1.

## 4. Decisions and rationale

| Decision | Choice | Why |
|---|---|---|
| Definition of "problematic concept" | Statistically overrepresented in failures vs successes (Fisher's exact, FDR-corrected) | Data-driven; falsifiable; avoids hand-picking "fox" |
| Model | Vanilla `Linq-AI-Research/Linq-Embed-Mistral` | Most failures = strongest discovery signal; no fine-tuning confound; standard Mistral architecture for activation editing |
| Concept granularity | Metadata tags from `data/enriched/fable_elements.json` for discovery; raw fable text contrasts for vector construction | Metadata gives interpretable, statistically clean discovery; text contrasts give denser, more robust vector directions |
| Proof criterion | Stage 1 = specificity contrast (this v0). Stage 2 = confusion-shift (follow-up). | Two-stage proof. Stage 1 = "knob is concept-specific." Stage 2 = "knob explains the failure." Stage 1 must pass before Stage 2 is interpretable. |
| Intervention side | Document side (fables) only | Metadata tags live on fables; cleanest single-side test; morals are too short (avg 12 words) to carry much concept signal in activations |
| Layer | Coarse sweep at 5 layers: 4, 12, 20, 28, last | Avoids arbitrary single-layer pick (A) without paying for full 32-layer sweep (B); 5 layers is enough to see early/mid/late behavior |
| Concept vector method | CAA matched-pairs (primary) + mean-difference (byproduct) | Matched pairs control nuisance variables (length, setting, fable_type); mean-diff is free comparison and a sanity check |
| Concept count | 3 problematic (one per metadata field: `characters`, `character_roles`, `moral_category`) + 1 placebo (low failure-rate concept) | Tests method generalization across field types; placebo is the strongest possible specificity control |
| Min sample size per concept | ≥15 tagged fables | Below this, matched-pair direction is too noisy. Verified: fox=67, trickster=64, deception=87 — all viable |
| α-sweep range | {-2.0, -1.0, -0.5, -0.25, 0, +0.25, +0.5, +1.0, +2.0} | α=0 is the no-op control; negative α suppresses, positive amplifies; range covers small (likely sweet spot) through aggressive |

## 5. Components

All under `analysis/08_concept_steering/`.

### 5.1 `config.yaml`

Single source of truth for the experiment. Everything tunable lives here. Code reads it; nothing hard-coded.

```yaml
model:
  hf_id: Linq-AI-Research/Linq-Embed-Mistral
  pooling: auto                  # auto | last_token | mean — verified at load
  device: cuda
  dtype: bfloat16

data:
  morals_path: data/processed/morals_corpus.json
  fables_path: data/processed/fables_corpus.json
  metadata_path: data/enriched/fable_elements.json

discovery:
  failure_definition: rank_gt_1   # rank_gt_1 | rank_gt_5
  metadata_fields:                # which fields to scan for problematic tags
    - characters
    - character_roles
    - moral_category
    - setting
    - fable_type
  min_tagged_fables: 15
  fdr_alpha: 0.05

concepts:
  # Filled after Step 2 by the discovery script. Example placeholder:
  targets:
    - {field: characters,        value: fox}
    - {field: character_roles,   value: trickster}
    - {field: moral_category,    value: deception}
  placebo:
    - {field: moral_category,    value: friendship}   # low failure-rate concept

vectors:
  layers: [4, 12, 20, 28, -1]    # -1 = last layer
  methods: [caa_matched, mean_diff]
  matching:                       # for caa_matched
    fields: [setting, fable_type]
    length_tolerance: 0.20        # ±20% token length

intervention:
  alphas: [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]
  hook_position: residual_stream  # where in the block to apply h ← h − α·v
  renormalize: true               # L2-renormalize the final pooled embedding

eval:
  metrics: [mrr_at_10, recall_at_1, recall_at_5, recall_at_10]
  group_by:                       # specificity contrast groups
    - target_tagged                # fables WITH the concept
    - target_untagged              # fables WITHOUT the concept
  specificity_thresholds:
    target_min_drop:   0.03        # |ΔMRR_target| at the chosen α must reach this
    control_max_drift: 0.015       # |ΔMRR_control| at the same α must stay within this

output:
  results_dir: analysis/08_concept_steering/results
  cache_dir:   analysis/08_concept_steering/cache
```

### 5.2 Module layout

```
analysis/08_concept_steering/
├── config.yaml
├── README.md                     ← what this experiment is, how to run
├── lib/
│   ├── __init__.py
│   ├── model.py                  ← load Linq, expose hooked forward, return hidden states
│   ├── discovery.py              ← Step 2 — Fisher's exact + FDR per tag
│   ├── vectors.py                ← Step 3 — CAA matched pairs + mean-diff
│   ├── intervene.py              ← Step 4 — apply hook, re-encode, return new embeddings
│   ├── eval.py                   ← Step 5 — group MRR, ΔMRR, specificity contrast
│   └── plotting.py               ← single summary figure builder
├── run_baseline.py               ← Step 1 entry point
├── run_discovery.py              ← Step 2 entry point
├── run_intervention.py           ← Steps 3+4+5 in one script (sweeps concept × layer × α)
└── results/
    ├── ranks_baseline.json
    ├── discovery_report.json     ← all tags ranked by failure overrep, with p-values
    ├── concept_vectors/
    │   └── {concept}_{layer}_{method}.npy
    ├── ranks_intervened/
    │   └── {concept}_{layer}_{alpha}.json
    └── specificity_summary.json + specificity_summary.png
```

**Why this split.** Each `run_*.py` is a thin CLI that reads the config, calls into `lib/`, and writes a JSON. The library functions are pure (config in, artifact out) and unit-testable. Adding a new concept-vector method (e.g. LEACE, logistic-probe) means adding one function in `vectors.py` and one entry under `vectors.methods` in the config — no other code change.

### 5.3 Hooked forward (the critical piece)

`lib/model.py` exposes a single function:

```python
def encode_with_intervention(
    model, tokenizer, texts: list[str],
    *,
    layer_idx: int | None = None,
    direction: torch.Tensor | None = None,   # shape: (hidden_dim,)
    alpha: float = 0.0,
    pooling: Literal["last_token", "mean"] = "last_token",
    renormalize: bool = True,
) -> torch.Tensor:                            # shape: (n_texts, hidden_dim)
    """
    Run the model forward; if direction is given, register a forward hook on
    transformer block `layer_idx` that subtracts `alpha * direction` from the
    residual stream at every token position. Returns pooled, optionally
    L2-normalized sentence embeddings.

    Setting alpha=0 OR direction=None means no intervention (used for baseline
    and for layer-wise hidden-state extraction during vector building).
    """
```

This is the ONLY place in the codebase that touches model internals. Every other module calls it. If the chosen layer numbering is wrong or the pooling is misidentified, the bug is in one place.

## 6. Evaluation — the v0 success criterion (Stage 1)

For each (concept C, layer L, α ≠ 0):

- `MRR_target = MRR@10 over morals whose ground-truth fable IS tagged C`
- `MRR_control = MRR@10 over morals whose ground-truth fable is NOT tagged C`

Define `ΔMRR_target(α) = MRR_target(α) − MRR_target(α=0)`, similarly `ΔMRR_control`.

**Pass criterion (per concept, per layer):**

1. There exists at least one α ≠ 0 with `ΔMRR_target ≤ −0.03` (target moves down meaningfully).
2. At that same α, `|ΔMRR_control| ≤ 0.015` (control stays flat within noise).
3. The placebo concept does NOT satisfy criterion 1 at any (layer, α). i.e. subtracting a non-problematic concept does not selectively damage its tagged group.

Thresholds are stored in the config (`eval.specificity_thresholds`) and are conservative based on standard MRR noise floor at n≈100. They can be tightened if v0 results are too lenient.

**Headline figure:** one plot, 4 columns (3 targets + 1 placebo) × 5 rows (one per layer). Each subplot has α on the x-axis and two lines: target-tagged MRR and control MRR. We expect the 3 target columns to show a separating fan; the placebo column should show two flat overlapping lines.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| AI-generated metadata is noisy (per `data/enriched/README.md`) — failure-rate stats are biased | Spot-check 30 random tagged fables per chosen concept before running intervention; flag any concept with <80% precision and re-pick |
| Mean pooling vs last-token pooling identification wrong | `lib/model.py` reads pooling from the loaded SentenceTransformer modules.json, doesn't hard-code; logged at startup |
| Layer indexing off-by-one between HuggingFace and SentenceTransformer wrappers | `run_baseline.py` at startup logs hidden_states tensor shapes for a single example so we can verify before any 5-layer sweep |
| n=15–87 per concept is small for matched pairs | Report bootstrap CI on the concept vector cosine norm; if direction is too noisy, report and skip |
| L2-renormalization after intervention may absorb the effect (since cosine retrieval cares about direction, not magnitude) | Run a subtraction with α large enough that direction shifts (not just magnitude); ablate by reporting both with/without renormalize |
| The specificity contrast moves but in the wrong direction (target stays up, control drops) | Reporting both directions explicitly; not assuming the sign |
| GPU disk pressure | Concept vectors are tiny (~16 KB each). Re-encoded fable embeddings are ~22 MB per (concept, layer, α). Total <2 GB for full sweep. No `model_output_dir` issue since no model saving. |

## 8. Out of scope

- Stage 2 confusion-shift evaluation (follow-up after Stage 1 passes).
- Query-side intervention.
- Fine-tuned model (ft07).
- Other concept-vector methods (LEACE, logistic probes, SAE features).
- Generation-side validation (different model class).
- ROME / MEMIT (designed for autoregressive factual edits, not encoder retrieval).

These are explicitly listed in `README.md` as planned follow-ups, with config knobs already shaped to accept them (e.g. `vectors.methods` is a list, `intervention.hook_position` is a string).

## 9. What we'll know after v0

- A ranked list of problematic concepts with statistical evidence (`discovery_report.json`).
- For each chosen concept, per-layer evidence of whether the intervention is concept-specific (specificity figure).
- A reusable hooked-forward implementation that any future intervention experiment can build on.
- A clear go/no-go for Stage 2 (confusion-shift): if Stage 1 passes for ≥2 of 3 targets and the placebo behaves correctly, Stage 2 is justified.
