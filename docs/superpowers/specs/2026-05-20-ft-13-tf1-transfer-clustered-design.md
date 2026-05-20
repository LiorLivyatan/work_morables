# ft_13 — TF1-synthetic transfer (clustered) — design spec

**Branch:** `tf1-iou-diagnostic`
**Date:** 2026-05-20
**Related:**
- Predecessor template: [`finetuning/ft_12_storal_transfer_clustered/`](../../../finetuning/ft_12_storal_transfer_clustered/)
- Source corpus build: [`experiments/11_tf1_diagnostic/`](../../../experiments/11_tf1_diagnostic/) (ft_12 was for STORAL; ft_13 is its sibling for TF1-synthetic)
- Earlier TF1 evaluation: [`experiments/11_tf1_diagnostic/REPORT.md`](../../../experiments/11_tf1_diagnostic/REPORT.md)

## Goal

Fine-tune embedding models on TF1-synthetic moral→fable pairs (exact-cluster mode only), then evaluate transfer to clustered MORABLES using the same metrics, rankings persistence, and comprehensive CSV format as `ft_12_storal_transfer_clustered`. The experiment answers: *does training on synthetic Llama-3.1-8B-generated fables (with a fixed 100-moral pool) transfer to literary moral→fable retrieval?*

This is the TF1 analogue of ft_12. ft_12 used STORAL (~1500 unique morals) as the external corpus; ft_13 uses TF1 (100 unique morals × 10 fables each). The two experiments are designed to be directly comparable.

## Non-goals

- Not generating new TF1 fables — use the cached `samples.jsonl` from the diagnostic run.
- Not using TF1's near-paraphrase clusters (model-generated, not human-annotated). Exact-mode only.
- Not mining explicit hard negatives in this experiment. In-batch InfoNCE only. Hard-neg mining is parked for a follow-up (ft_14).
- Not adding the multi-positive loss handling — `InfoNCELoss` from `finetuning/lib/losses.py` already supports it via integer labels.

## Inputs

### Source training corpus — TF1-synthetic

Two parallel corpora coexist (built by the same `build_tf1_corpus.py` script with `--selection`):

| Variant | Location | Selection rule | Purpose |
|---|---|---|---|
| Random (existing) | `data/external/tf1_synthetic/processed/` + `clustered/` | `random.sample(10)` per moral | Baseline; kept for ablation. |
| Low-IoU-clean (new, **primary for ft_13**) | `data/external/tf1_synthetic_low_iou/processed/` + `clustered/` | Anti-leakage filter + ascending `iou_no_stop`, take 10 lowest per moral | Trains on the MORABLES-like regime where moral words don't appear in the fable. |

Each corpus contains, for exact-cluster mode:
- `processed/morals_corpus.json` — 100 morals (`moral_tf1_000`…`moral_tf1_099`)
- `processed/fables_corpus.json` — 1000 fables (`fable_tf1_00000`…`fable_tf1_00999`)
- `processed/qrels_moral_to_fable.json` — 1000 rows (1 moral → 10 fables)
- `clustered/cluster_mapping_exact.json` — 100 single-moral clusters (each moral is its own cluster)
- `clustered/morals_unique_corpus_exact.json` — 100 cluster reps
- `clustered/qrels_moral_to_fable_clustered_exact.json` — 1000 rows

For ft_13, **only the low-IoU corpus is used in training**; the random corpus is referenced for the ablation row only.

### Evaluation target — clustered MORABLES (read-only)

Identical files as ft_12 uses:
- `data/clustered/morals_unique_corpus.json` (668 unique morals as queries)
- `data/clustered/fables_corpus.json` (709 fables as documents)
- `data/clustered/qrels_moral_to_fable_clustered.json` (multi-label qrels)

Eval doc_configs use the existing `experiments/20_final_zero_shot/summary_inputs/{generator}/golden_summaries.json` files (no new summaries needed; ft_12 has fully validated this pipeline).

## Architecture — ft_13 = ft_12 with three surgical replacements

ft_12's `train.py` is the template. The vast majority of code is reused verbatim, including:
- argparse interface (`--model`, `--models`, `--size`, `--sizes`, `--eval_doc_configs`, `--summary_generator`, `--skip_train`, `--force`, `--continue_on_error`, `--no-wandb`)
- Trainer delegation via `train_model()` (`finetuning/lib/trainer.py`)
- Loss invocation: `InfoNCELoss(model, temperature=τ)`
- Group-aware validation split logic
- Eval pipeline (`load_clustered()` → `load_qrels()` → `evaluate_and_rank()` → `compute_multilabel_metrics_from_matrix()`)
- Ranking persistence (full `ranked_fable_ids` + `scores` per query)
- Comprehensive CSV append (36 columns)
- Telegram start/per-run/queue-end notifications
- Per-run JSON dump under `results/`
- Resume-from-tracking-CSV runner pattern (`run_remaining_from_tracking.py`)

Three things change:

### Change 1 — source data loader

Replace `load_storal()` with `load_tf1_synthetic_exact()`:

```python
def load_tf1_synthetic_exact(size_cfg: dict, seed: int, source_dir: Path) -> tuple[list[dict], dict]:
    """Load TF1 exact-cluster pairs from data/external/tf1_synthetic_low_iou/."""
    morals = load_json(source_dir / "processed" / "morals_corpus.json")
    fables = load_json(source_dir / "processed" / "fables_corpus.json")
    qrels = load_json(source_dir / "processed" / "qrels_moral_to_fable.json")

    moral_by_id = {m["doc_id"]: m["text"] for m in morals}
    fable_by_id = {f["doc_id"]: f["text"] for f in fables}
    pairs = [
        {
            "moral": moral_by_id[q["query_id"]],
            "story": fable_by_id[q["doc_id"]],
            "moral_id": q["query_id"],
            "fable_id": q["doc_id"],
        }
        for q in qrels
    ]
    pairs = _subsample_morals(pairs, size_cfg, seed)
    stats = {
        "raw_total": len(qrels),
        "selected_rows": len(pairs),
        "selected_morals": len(set(p["moral_id"] for p in pairs)),
        "selection_strategy": source_dir.name,
        "size_config": size_cfg,
    }
    return pairs, stats
```

Key difference from `load_storal`: sizes sample **morals** then take all their fables (preserving the 1:10 structure), not individual rows.

```python
def _subsample_morals(pairs: list[dict], size_cfg: dict, seed: int) -> list[dict]:
    """Sample size_cfg['n_morals'] distinct morals; keep all their fables."""
    n_morals = size_cfg.get("n_morals")
    if n_morals is None:
        return pairs  # sfull = all 100 morals
    by_moral: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_moral[p["moral_id"]].append(p)
    moral_ids = sorted(by_moral.keys())
    rng = random.Random(seed)
    chosen = rng.sample(moral_ids, n_morals)
    return [p for mid in chosen for p in by_moral[mid]]
```

### Change 2 — dataset assembly with integer cluster labels

`make_tf1_dataset()` is structurally identical to `make_storal_dataset()`, but labels come directly from `moral_id` (in exact-cluster mode `moral_id` IS the cluster id):

```python
def make_tf1_dataset(pairs: list[dict], instruction: str):
    from datasets import Dataset

    moral_to_label: dict[str, int] = {}
    labels: list[int] = []
    for p in pairs:
        if p["moral_id"] not in moral_to_label:
            moral_to_label[p["moral_id"]] = len(moral_to_label)
        labels.append(moral_to_label[p["moral_id"]])

    return Dataset.from_dict({
        "anchor":   [f"{instruction}{p['moral']}" for p in pairs],
        "positive": [p["story"] for p in pairs],
        "label":    labels,
    })
```

The label assignment is structurally simpler than ft_12's (no text-normalization step) because TF1 morals are already deduplicated by construction; `moral_id` IS the canonical identity.

### Change 3 — validation split by moral_id

`split_tf1_groups()` mirrors `split_storal_groups()` but keys on `moral_id` directly (no `normalize_moral` step needed). With 100 morals and a 10% validation ratio, we get 90 train morals / 10 validation morals at sfull, scaling proportionally at smaller sizes.

```python
def split_tf1_groups(pairs: list[dict], seed: int, validation_ratio: float) -> tuple[list[dict], list[dict]]:
    by_moral: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_moral[p["moral_id"]].append(p)
    moral_ids = list(by_moral.keys())
    rng = random.Random(seed)
    rng.shuffle(moral_ids)
    n_val = max(1, round(len(moral_ids) * validation_ratio))
    val_morals = set(moral_ids[:n_val])
    train_rows = [p for mid in moral_ids[n_val:] for p in by_moral[mid]]
    val_rows = [p for mid in moral_ids[:n_val] for p in by_moral[mid]]
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows
```

## New: extending `build_tf1_corpus.py` with `--selection`

The build script gets one new argument and one new function:

```python
parser.add_argument(
    "--selection", choices=["random", "low_iou_clean"], default="random",
    help="How to pick 10 fables per moral. random = current uniform behavior. "
         "low_iou_clean = filter out moral-restating fables, then take 10 with "
         "lowest iou_no_stop.",
)
```

`low_iou_clean` selection algorithm:

```python
LEAKAGE_PATTERNS = [
    re.compile(r"the\s+(moral|lesson|teaching|takeaway)\s+(of|is|here\s+is)", re.IGNORECASE),
    re.compile(r"this\s+(story|fable|tale)\s+teaches\s+(us|that)", re.IGNORECASE),
    re.compile(r"^\s*moral\s*:", re.IGNORECASE | re.MULTILINE),
]

def has_explicit_moral(fable_text: str, moral_text: str) -> bool:
    """True if the fable contains an explicit restatement of the moral."""
    for pattern in LEAKAGE_PATTERNS:
        if pattern.search(fable_text):
            return True
    # Also catch: moral text appears as a near-verbatim substring (>=70% content-word overlap)
    moral_words = set(WORD_RE.findall(moral_text.lower())) - STOP_WORDS
    if len(moral_words) < 2:
        return False
    for sent in re.split(r"[.!?]", fable_text):
        sent_words = set(WORD_RE.findall(sent.lower())) - STOP_WORDS
        if moral_words and len(moral_words & sent_words) / len(moral_words) >= 0.70:
            return True
    return False


def select_low_iou_clean(grouped: dict[str, list[dict]], n: int) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for moral, rows in grouped.items():
        clean = [r for r in rows if not has_explicit_moral(r["fable"], r["moral"])]
        clean.sort(key=lambda r: r["iou_no_stop"])
        if len(clean) < n:
            raise ValueError(
                f"After leakage filter, only {len(clean)} fables remain for "
                f"{moral!r}; need {n}. Consider re-streaming more TF1 rows."
            )
        out[moral] = clean[:n]
    return out
```

The script's dispatch:

```python
if args.selection == "random":
    sampled = sample_n_per_moral(grouped, n=args.n, seed=args.seed)
elif args.selection == "low_iou_clean":
    sampled = select_low_iou_clean(grouped, n=args.n)
```

The `--out` argument defaults stay backward-compatible:
- `--selection random` → default out = `data/external/tf1_synthetic/`
- `--selection low_iou_clean` → default out = `data/external/tf1_synthetic_low_iou/`

The README that `build_tf1_corpus.py` writes records the selection strategy.

## Negative-leakage guarantees (concise)

| Risk | Status |
|---|---|
| **R1** — implicit (in-batch) same-moral collision | **Handled automatically** by `InfoNCELoss` (label mask in `losses.py:62-65`). Since `label = moral_id`, any two rows sharing the moral get masked out of each other's negative set in every batch. |
| **R2** — explicit hard-negative pool same-moral collision | **Not applicable** — ft_13 uses in-batch negatives only. R2 becomes relevant only if/when a follow-up experiment mines hard negatives; see "Open questions" below. |

## Configuration

### Sizes

| Key | n_morals | n_fables (= n_morals × 10) | Rationale |
|---|---|---|---|
| `s200` | 20 | 200 | Matches ft_12 sweep start; also a probable sweet-spot per ft_12 results |
| `s500` | 50 | 500 | Mid-sweep |
| `sfull` | 100 (all) | 1000 | Maximum from our cache |

`s1000` is **deliberately omitted** — at TF1's 100-moral ceiling, `s1000 == sfull` (100 morals × 10 fables = 1000 rows). Including both would be redundant.

### Models (two waves)

Picked from ft_12's 15-model run. The CSV shows a clear three-tier ranking; ft_13 covers all three tiers in the first wave plus extends to the second wave for budget reasons.

**First wave (must run):**
| alias | model_name | size class | ft_12 best MAP@10 |
|---|---|---|---|
| `bge` | `BAAI/bge-base-en-v1.5` | 110M | 0.2949 |
| `all_minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 22M | 0.2622 |
| `all_mpnet` | `sentence-transformers/all-mpnet-base-v2` | 110M | 0.3228 |
| `qwen3_0_6b` | `Qwen/Qwen3-Embedding-0.6B` | 600M | 0.3366 |
| `qwen3_4b` | `Qwen/Qwen3-Embedding-4B` | 4B | 0.4304 |
| `linq` | `Linq-AI-Research/Linq-Embed-Mistral` | 7B (LoRA) | 0.4618 |

**Second wave (run if first wave shows ft_13 is worth scaling):**
| alias | model_name | rationale |
|---|---|---|
| `qwen3` | `Qwen/Qwen3-Embedding-8B` | Top-3 in ft_12 (0.4467 MAP@10). User's TODO listed this conditionally as "if smaller runs look useful". |
| `sfr` | `Salesforce/SFR-Embedding-Mistral` | 2nd in ft_12 (0.4537); independent second opinion from a non-Qwen non-Linq 7B-class model. |

**`multilingual_e5_large_instruct` is excluded** — ft_12 shows it produces MAP@10 = 0.011 across all configurations, indicating a model-loading or inference bug rather than poor performance.

### Eval doc_configs

ft_12's CSV shows summary-augmented configs beat raw by +0.10 to +0.18 MAP@10 across every model. ft_13 must evaluate at least this set to avoid measuring an artificially low ceiling:

**Smoke test:** `raw` only.

**Real runs:** at minimum `raw, fable_cot_proverb, fable_direct_moral, conceptual_abstract_fable` with **`gemini`** generator (the best in ft_12 by ~0.09 MAP@10). Full ft_12 doc_config list available for ablation in the `--eval_doc_configs all` mode.

### Hyperparameters

Inherit from ft_12 unchanged unless flagged:
- `seed: 42`
- `epochs: 10`
- `temperature: 0.05`
- `early_stopping_patience: 3`
- `early_stopping_metric: ndcg@10`
- `validation_ratio: 0.10`
- `exact_moral_masking: true`
- `ks: [1, 5, 10, 15, 50, 100, 200, 300]`
- Per-model `batch_size`, `learning_rate`, `LoRA` settings: copy verbatim from ft_12's `config.yaml`

### Storage

- Trained models: `/data/lior/ft_13_tf1_transfer_clustered/models/{model_alias}/{size_name}`
- Per-run JSON: `finetuning/ft_13_tf1_transfer_clustered/results/{timestamp}_{model_alias}_{size}.json`
- Rankings JSON: `finetuning/ft_13_tf1_transfer_clustered/results/rankings/{model_alias}__{size}__{doc_config}__{generator}.json`
- Comprehensive CSV: `finetuning/ft_13_tf1_transfer_clustered/results/ft13_comprehensive_results.csv`

CSV header matches ft_12 exactly except `experiment` column holds `"ft_13_tf1_transfer_clustered"` and one column rename: `n_storal_selected` → `n_tf1_rows_selected`. Two columns added: `n_tf1_morals_selected` and `selection_strategy`. (The corpus_template, summary_generator, all metric columns, etc. stay byte-identical to enable cross-experiment analysis.)

## Smoke test (mandatory first run)

Before any wave-1 model runs:

```bash
./run.sh finetuning/ft_13_tf1_transfer_clustered/train.py \
    --model bge --size s200 --eval_doc_configs raw
```

Expected outcome:
- TF1 corpus loaded from `data/external/tf1_synthetic_low_iou/processed/` (so the low-IoU build must complete first)
- 20 morals split 18 train / 2 val → 180 train rows, 20 val rows (10 fables per moral)
- One row in `ft13_comprehensive_results.csv` with `eval_doc_config=raw`, `summary_generator=none`, non-empty MAP@10
- One ranking file at `results/rankings/bge__s200__raw__none.json` containing 668 entries (one per MORABLES moral query) with full ranked fable IDs
- Telegram start + end notifications fired
- Model saved to `/data/lior/ft_13_tf1_transfer_clustered/models/bge/s200`

If any of these are missing, the smoke test has failed and we fix before proceeding.

## Reproduction commands

```bash
# Prerequisites — produce the low-IoU corpus (one-time, ~1 minute):
./run.sh experiments/11_tf1_diagnostic/build_tf1_corpus.py \
    --selection low_iou_clean
./run.sh experiments/11_tf1_diagnostic/cluster_tf1_morals.py \
    --mode exact --in data/external/tf1_synthetic_low_iou

# Smoke test:
./run.sh finetuning/ft_13_tf1_transfer_clustered/train.py \
    --model bge --size s200 --eval_doc_configs raw

# Wave 1 — first-wave models on all sizes, full eval suite:
./run.sh finetuning/ft_13_tf1_transfer_clustered/train.py \
    --models bge all_minilm all_mpnet qwen3_0_6b qwen3_4b linq \
    --sizes s200 s500 sfull \
    --eval_doc_configs raw fable_cot_proverb fable_direct_moral conceptual_abstract_fable \
    --summary_generator gemini \
    --remote --gpu 2

# Wave 2 (optional):
./run.sh finetuning/ft_13_tf1_transfer_clustered/train.py \
    --models qwen3 sfr --sizes s500 sfull \
    --eval_doc_configs all --summary_generator gemini \
    --remote --gpu 3

# Optional ablation — same training command, but against the random-selection corpus:
./run.sh finetuning/ft_13_tf1_transfer_clustered/train.py \
    --model linq --size sfull --eval_doc_configs raw \
    --tf1_corpus_dir data/external/tf1_synthetic
```

## Risks and open questions (parking lot)

### Risks

- **100-moral training pool is structurally narrow.** Even at `sfull`, ft_13 trains on fewer unique morals than ft_12's smallest config (s200 had ~180 unique morals after dedup). The transfer ceiling may be lower than ft_12's 0.46 MAP@10. **Reporting honestly:** the headline number is whatever we get, framed as "what does training on a 100-moral synthetic corpus buy us."
- **Low-IoU filter may exclude high-quality fables.** Some fables with above-median IoU may still be perfectly good narratives; filtering them out reduces the available pool. We accept this for the distribution-match payoff. The random corpus stays available for the ablation row.

### Open questions (for team discussion)

1. **Hard-negative follow-up.** Should we plan a ft_14 that adds mined hard negatives over the same TF1 corpus, with the two-layer exclusion (positives + cluster siblings)? Likely yes if ft_13 results are encouraging, but defer the spec until ft_13 numbers are in.
2. **Strategy C (diversity-aware selection).** Currently deferred. If ft_13 results are encouraging but transfer plateaus, the diversity-aware variant is the obvious next ablation.
3. **Wave-2 budget.** Wave 2 adds two large models (qwen3 7B-ish, sfr 7B). Should we wait for wave-1 sfull-vs-s200 comparison before committing GPU time? Suggestion: yes.
4. **Cross-corpus comparison.** Should the comprehensive CSV include a "random" vs "low_iou_clean" run side-by-side for the same `(model, size, doc_config)`, or is that a separate ablation table? Recommended: include both rows in the same CSV with `selection_strategy` as the distinguishing column.

## Out of scope

- Modifying ft_12 (the template) in any way.
- Touching `data/clustered/` (MORABLES) — read-only here.
- Removing the random-selection TF1 corpus — kept for ablation.
- Implementing the diversity-aware (strategy C) selection.
- Implementing hard-negative mining.
- Adding new models that ft_12 hasn't characterized (other than Qwen3-8B if budget allows).
- Modifying `finetuning/lib/losses.py` or `finetuning/lib/trainer.py`.
