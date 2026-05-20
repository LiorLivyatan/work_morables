# ft_13 — TF1-synthetic transfer (clustered) implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `finetuning/ft_13_tf1_transfer_clustered/` (a TF1-synthetic analog of `ft_12_storal_transfer_clustered`) plus extend `experiments/11_tf1_diagnostic/build_tf1_corpus.py` with an anti-leakage low-IoU fable-selection strategy. Smoke-test BGE on the smallest size to validate the full pipeline.

**Architecture:** Reuses ft_12's training loop, eval pipeline, ranking persistence, and CSV format verbatim. Three surgical replacements: source-data loader (TF1 instead of STORAL), per-size sub-sampling (samples morals × fables-per-moral instead of rows), and `InfoNCELoss` label assignment direct from `moral_id`. A new low-IoU corpus is built at `data/external/tf1_synthetic_low_iou/` via an extended `build_tf1_corpus.py` and clustered via existing `cluster_tf1_morals.py --mode exact`.

**Tech Stack:** Python 3.13, `uv`, `sentence-transformers`, `datasets`, `pytest`, `scipy`, `numpy`. Existing helpers: `finetuning/lib/losses.py::InfoNCELoss`, `finetuning/lib/trainer.py::train_model`, `lib/retrieval_utils.py::compute_multilabel_metrics_from_matrix`.

**Spec:** [`docs/superpowers/specs/2026-05-20-ft-13-tf1-transfer-clustered-design.md`](../specs/2026-05-20-ft-13-tf1-transfer-clustered-design.md)

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `experiments/11_tf1_diagnostic/build_tf1_corpus.py` | Modify | Add `has_explicit_moral`, `select_low_iou_clean`, wire `--selection` flag into `run_build` |
| `tests/experiments/test_build_tf1_corpus.py` | Modify | Add unit tests for the two new helpers + integration test for `--selection low_iou_clean` |
| `data/external/tf1_synthetic_low_iou/` | Generate | New corpus at N=20, low-IoU + anti-leakage selection (built by the modified script) |
| `finetuning/ft_13_tf1_transfer_clustered/__init__.py` | Create | Empty package marker |
| `finetuning/ft_13_tf1_transfer_clustered/config.yaml` | Create | Sizes, models, doc_configs, hyperparams (copy ft_12 with adjustments) |
| `finetuning/ft_13_tf1_transfer_clustered/README.md` | Create | One-screen description |
| `finetuning/ft_13_tf1_transfer_clustered/train.py` | Create | Main training script (copy ft_12 with three surgical changes) |
| `tests/finetuning/test_ft_13_data_loaders.py` | Create | Unit tests for `_subsample_morals`, `split_tf1_groups`, `make_tf1_dataset`, `load_tf1_synthetic_exact` |

---

### Task 1: Add `has_explicit_moral` helper to build_tf1_corpus.py

**Files:**
- Modify: `experiments/11_tf1_diagnostic/build_tf1_corpus.py`
- Modify: `tests/experiments/test_build_tf1_corpus.py`

- [ ] **Step 1: Add the failing tests for `has_explicit_moral`**

Append to `tests/experiments/test_build_tf1_corpus.py`:

```python
has_explicit_moral = build.has_explicit_moral


def test_has_explicit_moral_detects_phrase_the_moral_of():
    assert has_explicit_moral("...and the moral of the story is to never lie.", "honesty wins")


def test_has_explicit_moral_detects_phrase_this_fable_teaches():
    assert has_explicit_moral("They learned much. This fable teaches us that kindness wins.", "be kind")


def test_has_explicit_moral_detects_phrase_lesson_here_is():
    assert has_explicit_moral("In the end, the lesson here is patience.", "patience pays")


def test_has_explicit_moral_detects_explicit_moral_label():
    assert has_explicit_moral("A short story.\nMoral: always tell the truth.\n", "honesty wins")


def test_has_explicit_moral_detects_high_overlap_sentence():
    # Sentence contains 3 of 3 content words from the moral.
    fable = "After the storm they learned that timely help earns lasting loyalty."
    moral = "timely help earns lasting loyalty"
    assert has_explicit_moral(fable, moral)


def test_has_explicit_moral_false_when_clean():
    fable = ("In a quiet glade lived a young deer who often wandered far from home. "
             "One day a wise owl told him a strange riddle, and after much thought "
             "the deer understood that he had been chasing illusions all along.")
    moral = "patience pays"
    assert not has_explicit_moral(fable, moral)


def test_has_explicit_moral_safe_for_short_morals():
    # 1-content-word moral should not trigger overlap rule (else everything matches)
    assert not has_explicit_moral("A quiet story with nothing notable.", "patience")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/experiments/test_build_tf1_corpus.py::test_has_explicit_moral_detects_phrase_the_moral_of -v
```

Expected: `AttributeError: module 'build_tf1_corpus' has no attribute 'has_explicit_moral'`.

- [ ] **Step 3: Implement `has_explicit_moral`**

Add to `experiments/11_tf1_diagnostic/build_tf1_corpus.py` right after the existing constants block at the top of the file (where `STOP_WORDS` would already be defined if you've followed the existing structure). If `STOP_WORDS` and `WORD_RE` are not already module-level constants in this file, copy them from `experiments/11_tf1_diagnostic/check_iou.py`. Then add:

```python
LEAKAGE_PATTERNS = [
    re.compile(r"the\s+(moral|lesson|teaching|takeaway)\s+(of|is|here\s+is)", re.IGNORECASE),
    re.compile(r"this\s+(story|fable|tale)\s+teaches\s+(us|that)", re.IGNORECASE),
    re.compile(r"^\s*moral\s*:", re.IGNORECASE | re.MULTILINE),
]

# Reuse the tokenization established in check_iou.py
WORD_RE = re.compile(r"\b\w+\b")
STOP_WORDS = {
    "the","a","an","is","was","were","are","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "to","of","in","for","on","with","at","by","from","as","into","through",
    "during","before","after","and","but","or","nor","not","so","if","than",
    "that","this","it","its","his","her","their","who","which","what","when",
    "where","how","all","each","every","both","few","more","most","other","some",
    "such","no","only","own","same","he","she","they","them","him","we","you",
    "i","me","my","your","our",
}


def has_explicit_moral(fable_text: str, moral_text: str) -> bool:
    """True if the fable contains an explicit restatement of the moral.

    Two layers:
    1. Regex patterns catching common LLM closing phrases like
       "The moral of the story is...".
    2. Per-sentence content-word overlap of at least 70% with the moral
       (catches near-verbatim moral restatements without the regex tells).

    Morals with fewer than 2 content words are exempt from layer 2 to
    avoid spurious matches on tiny morals.
    """
    if not fable_text or not moral_text:
        return False
    for pattern in LEAKAGE_PATTERNS:
        if pattern.search(fable_text):
            return True
    moral_content = set(WORD_RE.findall(moral_text.lower())) - STOP_WORDS
    if len(moral_content) < 2:
        return False
    for sentence in re.split(r"[.!?]", fable_text):
        sent_content = set(WORD_RE.findall(sentence.lower())) - STOP_WORDS
        if moral_content and len(moral_content & sent_content) / len(moral_content) >= 0.70:
            return True
    return False
```

If the imports `import re` is not already present at the top of `build_tf1_corpus.py`, add it.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/experiments/test_build_tf1_corpus.py -v
```

Expected: all 7 new tests pass; all previously-passing tests continue to pass (final summary line should be `passed in ...s` with no failures).

- [ ] **Step 5: Commit**

```bash
git add experiments/11_tf1_diagnostic/build_tf1_corpus.py tests/experiments/test_build_tf1_corpus.py
git commit -m "feat(tf1_synthetic): add has_explicit_moral leakage detector for fable selection"
```

---

### Task 2: Add `select_low_iou_clean` + wire `--selection` flag

**Files:**
- Modify: `experiments/11_tf1_diagnostic/build_tf1_corpus.py`
- Modify: `tests/experiments/test_build_tf1_corpus.py`

- [ ] **Step 1: Add failing tests for `select_low_iou_clean` and the new arg dispatch**

Append to `tests/experiments/test_build_tf1_corpus.py`:

```python
select_low_iou_clean = build.select_low_iou_clean


def _row_with_iou(idx, moral, fable, iou):
    return {
        "idx": idx, "chunk": 0, "prompt_hash": f"h{idx}",
        "moral": moral, "fable": fable,
        "iou_no_stop": iou,
    }


def test_select_low_iou_clean_takes_n_lowest_iou_per_moral():
    grouped = {
        "moral_a": [
            _row_with_iou(0, "moral_a", "clean fable 1", iou=0.05),
            _row_with_iou(1, "moral_a", "clean fable 2", iou=0.02),
            _row_with_iou(2, "moral_a", "clean fable 3", iou=0.03),
            _row_with_iou(3, "moral_a", "clean fable 4", iou=0.04),
        ],
    }
    out = select_low_iou_clean(grouped, n=2)
    assert len(out["moral_a"]) == 2
    assert [r["iou_no_stop"] for r in out["moral_a"]] == [0.02, 0.03]


def test_select_low_iou_clean_filters_leaky_fables_before_sorting():
    # Leaky fable has lower iou but should be excluded.
    grouped = {
        "honesty wins": [
            _row_with_iou(0, "honesty wins", "The moral of the story is honesty.", iou=0.01),
            _row_with_iou(1, "honesty wins", "Clean narrative about birds.", iou=0.02),
            _row_with_iou(2, "honesty wins", "Another clean fable.", iou=0.05),
        ],
    }
    out = select_low_iou_clean(grouped, n=2)
    fables = [r["fable"] for r in out["honesty wins"]]
    assert "The moral of the story is honesty." not in fables
    assert fables == ["Clean narrative about birds.", "Another clean fable."]


def test_select_low_iou_clean_raises_when_not_enough_clean():
    grouped = {
        "m": [
            _row_with_iou(0, "m", "Moral: be kind.", iou=0.01),  # leaky
            _row_with_iou(1, "m", "Clean.", iou=0.02),
        ],
    }
    with pytest.raises(ValueError, match="only 1 fables remain"):
        select_low_iou_clean(grouped, n=2)


def test_run_build_with_selection_low_iou_clean_uses_clean_fables(tmp_path):
    samples = [
        {"idx": 0, "chunk": 0, "prompt_hash": "h0", "moral": "be kind",
         "fable": "A clean story about a fox.", "iou_no_stop": 0.02},
        {"idx": 1, "chunk": 0, "prompt_hash": "h1", "moral": "be kind",
         "fable": "Another clean story.", "iou_no_stop": 0.03},
        {"idx": 2, "chunk": 0, "prompt_hash": "h2", "moral": "be kind",
         "fable": "The moral of the story is to be kind.", "iou_no_stop": 0.01},
    ]
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text("\n".join(json.dumps(r) for r in samples))

    out_dir = tmp_path / "out"
    build.run_build(
        samples_path=samples_path, n=2, seed=42,
        out_dir=out_dir, expected_unique_morals=1,
        selection="low_iou_clean",
    )
    fables = json.loads((out_dir / "processed" / "fables_corpus.json").read_text())
    fable_texts = [f["text"] for f in fables]
    assert "The moral of the story is to be kind." not in fable_texts
    assert len(fables) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/experiments/test_build_tf1_corpus.py -v
```

Expected: 4 new tests fail; existing tests still pass.

- [ ] **Step 3: Implement `select_low_iou_clean` and extend `run_build`**

Add to `experiments/11_tf1_diagnostic/build_tf1_corpus.py` (place after `sample_n_per_moral` definition):

```python
def select_low_iou_clean(grouped: dict[str, list[dict]], n: int) -> dict[str, list[dict]]:
    """Pick the n fables with lowest iou_no_stop per moral, after filtering
    out fables that explicitly restate the moral.

    Each row must contain an `iou_no_stop` float field (produced by
    experiments/11_tf1_diagnostic/check_iou.py during sample dumping).
    """
    out: dict[str, list[dict]] = {}
    for moral, rows in grouped.items():
        clean = [r for r in rows if not has_explicit_moral(r["fable"], r["moral"])]
        if len(clean) < n:
            raise ValueError(
                f"After leakage filter, only {len(clean)} fables remain for "
                f"{moral!r}; need {n}. Consider re-streaming more TF1 rows."
            )
        clean.sort(key=lambda r: r["iou_no_stop"])
        out[moral] = clean[:n]
    return out
```

Now extend `run_build`. Find the existing signature in `build_tf1_corpus.py` and modify the signature + dispatch. Replace the existing definition with:

```python
def run_build(
    samples_path: Path,
    n: int,
    seed: int,
    out_dir: Path,
    expected_unique_morals: int = 100,
    selection: str = "random",
) -> dict:
    rows = _read_samples(samples_path)
    unique_morals = first_seen_order(rows)
    assert len(unique_morals) == expected_unique_morals, (
        f"expected {expected_unique_morals} unique morals, got {len(unique_morals)}"
    )

    grouped = group_by_moral(rows)
    moral_ids = assign_moral_ids(unique_morals)
    if selection == "random":
        sampled = sample_n_per_moral(grouped, n=n, seed=seed)
    elif selection == "low_iou_clean":
        sampled = select_low_iou_clean(grouped, n=n)
    else:
        raise ValueError(f"unknown selection {selection!r}; expected 'random' or 'low_iou_clean'")

    morals_corpus = build_morals_corpus(unique_morals, moral_ids)
    fables_corpus = build_fables_corpus(sampled, unique_morals, moral_ids, n=n)
    qrels_mtf = build_qrels_moral_to_fable(fables_corpus)
    qrels_ftm = build_qrels_fable_to_moral(fables_corpus)

    hashes = [f["prompt_hash"] for f in fables_corpus]
    assert len(set(hashes)) == len(hashes), "duplicate prompt_hash in sampled fables"

    processed = out_dir / "processed"
    _write_json(processed / "morals_corpus.json", morals_corpus)
    _write_json(processed / "fables_corpus.json", fables_corpus)
    _write_json(processed / "qrels_moral_to_fable.json", qrels_mtf)
    _write_json(processed / "qrels_fable_to_moral.json", qrels_ftm)
    _write_readme(out_dir, n=n, seed=seed, source=samples_path,
                  n_morals=len(unique_morals), selection=selection)

    return {
        "n_morals": len(unique_morals),
        "n_fables": len(fables_corpus),
        "selection": selection,
        "out_dir": str(out_dir),
    }
```

Extend `_write_readme` to accept and surface `selection`:

```python
def _write_readme(out_dir: Path, n: int, seed: int, source: Path,
                  n_morals: int, selection: str = "random") -> None:
    rel_source = source.relative_to(ROOT) if source.is_relative_to(ROOT) else source
    readme = f"""# TF1-EN-3M synthetic — MORABLES-shaped derivative

Source: https://huggingface.co/datasets/klusai/ds-tf1-en-3m (MIT)
Paper: https://arxiv.org/abs/2504.20605

Built from {rel_source}
via experiments/11_tf1_diagnostic/check_iou.py (--n 50000 --chunks 10).

## Snapshot (this build)

- Selection strategy: {selection}
- N per moral: {n}
- Total morals: {n_morals}
- Total fables: {n * n_morals}
- Seed: {seed}
- Built: {datetime.now().isoformat(timespec='seconds')}

## Build commands

./run.sh experiments/11_tf1_diagnostic/build_tf1_corpus.py --selection {selection} --n {n}
./run.sh experiments/11_tf1_diagnostic/cluster_tf1_morals.py --mode exact --in {out_dir}

See experiments/11_tf1_diagnostic/REPORT.md for the analysis that motivated this derivative.
"""
    (out_dir / "README.md").write_text(readme)
```

Finally extend `main()` to accept `--selection` and auto-pick `--out` when not provided:

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument(
        "--selection", choices=["random", "low_iou_clean"], default="random",
        help="How to pick n fables per moral. random = uniform sample. "
             "low_iou_clean = filter explicit-moral restatements, then take "
             "the n lowest by iou_no_stop.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output corpus dir. Defaults to data/external/tf1_synthetic/ "
             "for --selection random, or data/external/tf1_synthetic_low_iou/ "
             "for --selection low_iou_clean.",
    )
    parser.add_argument(
        "--expected-unique-morals", type=int, default=100,
        help="invariant from the diagnostic; lower for testing on small samples",
    )
    args = parser.parse_args()

    samples_path = args.source or _latest_samples_path()
    if args.out is not None:
        out_dir = args.out
    elif args.selection == "low_iou_clean":
        out_dir = ROOT / "data" / "external" / "tf1_synthetic_low_iou"
    else:
        out_dir = DEFAULT_OUT

    print(f"Reading samples from {samples_path}")
    print(f"Selection: {args.selection}  Output dir: {out_dir}")
    result = run_build(
        samples_path=samples_path,
        n=args.n,
        seed=args.seed,
        out_dir=out_dir,
        expected_unique_morals=args.expected_unique_morals,
        selection=args.selection,
    )
    print(f"Wrote {result['n_fables']} fables across {result['n_morals']} morals to {result['out_dir']}")
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
uv run pytest tests/experiments/test_build_tf1_corpus.py -v
```

Expected: all tests pass, including the 4 new ones from Step 1 and all existing ones.

- [ ] **Step 5: Commit**

```bash
git add experiments/11_tf1_diagnostic/build_tf1_corpus.py tests/experiments/test_build_tf1_corpus.py
git commit -m "feat(tf1_synthetic): add --selection low_iou_clean to build_tf1_corpus"
```

---

### Task 3: Generate the low-IoU corpus at N=20 and cluster it

**Files:**
- None modified directly; this is an execution task that produces files under `data/external/tf1_synthetic_low_iou/`.

- [ ] **Step 1: Verify the diagnostic samples.jsonl is available**

```bash
ls -la experiments/11_tf1_diagnostic/results/runs/*/samples.jsonl | tail -1
```

Expected: a path printed. If missing, run `./run.sh experiments/11_tf1_diagnostic/check_iou.py --n 50000 --chunks 10` first.

- [ ] **Step 2: Run the low-IoU corpus build at N=20**

```bash
./run.sh experiments/11_tf1_diagnostic/build_tf1_corpus.py --selection low_iou_clean --n 20
```

Expected last line: `Wrote 2000 fables across 100 morals to /Users/.../data/external/tf1_synthetic_low_iou`

If any moral has fewer than 20 clean fables, the script will fail with `ValueError`. If that happens, escalate — we would need to either re-stream more TF1 rows (raise --n in `check_iou.py`) or reduce N for the affected morals.

- [ ] **Step 3: Verify the processed/ outputs**

```bash
python3 -c "
import json
from pathlib import Path
p = Path('data/external/tf1_synthetic_low_iou/processed')
m = json.loads((p / 'morals_corpus.json').read_text())
f = json.loads((p / 'fables_corpus.json').read_text())
print('morals:', len(m))
print('fables:', len(f))
print('first moral:', m[0])
print('first fable doc_id:', f[0]['doc_id'])
"
```

Expected:
```
morals: 100
fables: 2000
first moral: {'doc_id': 'moral_tf1_000', 'text': '<some moral>'}
first fable doc_id: fable_tf1_00000
```

- [ ] **Step 4: Build the exact-mode clustered files for the new corpus**

```bash
./run.sh experiments/11_tf1_diagnostic/cluster_tf1_morals.py --mode exact --in data/external/tf1_synthetic_low_iou
```

Expected last line: `Wrote clustered outputs (mode=exact) to /Users/.../data/external/tf1_synthetic_low_iou/clustered`.

- [ ] **Step 5: Verify the clustered outputs**

```bash
python3 -c "
import json
from pathlib import Path
p = Path('data/external/tf1_synthetic_low_iou/clustered')
cm = json.loads((p / 'cluster_mapping_exact.json').read_text())
print('clusters:', len(cm))
print('n_morals per cluster (min, max):', min(c['n_morals'] for c in cm), max(c['n_morals'] for c in cm))
print('n_fables per cluster (min, max):', min(c['n_fables'] for c in cm), max(c['n_fables'] for c in cm))
print('total fables across clusters:', sum(c['n_fables'] for c in cm))
"
```

Expected:
```
clusters: 100
n_morals per cluster (min, max): 1 1
n_fables per cluster (min, max): 20 20
total fables across clusters: 2000
```

- [ ] **Step 6: Commit the new corpus**

```bash
git add data/external/tf1_synthetic_low_iou/
git commit -m "data(tf1_synthetic): low-IoU clean corpus at N=20 (2000 fables, 100 morals)"
```

---

### Task 4: ft_13 directory scaffolding (init, README, config.yaml)

**Files:**
- Create: `finetuning/ft_13_tf1_transfer_clustered/__init__.py`
- Create: `finetuning/ft_13_tf1_transfer_clustered/README.md`
- Create: `finetuning/ft_13_tf1_transfer_clustered/config.yaml`

- [ ] **Step 1: Create the package marker**

Create empty `finetuning/ft_13_tf1_transfer_clustered/__init__.py`. Contents: empty file.

- [ ] **Step 2: Create the README**

Create `finetuning/ft_13_tf1_transfer_clustered/README.md`:

```markdown
# ft_13_tf1_transfer_clustered

TF1-synthetic transfer experiment evaluated on clustered MORABLES.

## What it does

Fine-tunes embedding models on TF1-synthetic moral->fable pairs from the
low-IoU+clean corpus at `data/external/tf1_synthetic_low_iou/`. Trains
with `InfoNCELoss` + multi-positive masking (label = moral_id, exact
clusters only). Evaluates each trained model on clustered MORABLES
(`data/clustered/`) across multiple doc_configs, persists full per-query
rankings, and appends a comprehensive CSV.

This is the TF1 analog of ft_12_storal_transfer_clustered. The two
experiments share the same eval pipeline, ranking format, and CSV
schema (modulo three TF1-specific columns) so cross-experiment
analysis is straightforward.

## Sizes

- s200: 20 morals x 10 fables = 200 rows
- s500: 50 morals x 10 fables = 500 rows
- sfull: 100 morals x 10 fables = 1000 rows
- sfull_n20: 100 morals x 20 fables = 2000 rows (isolates per-moral redundancy)

## Reproduction

Smoke test:
    ./run.sh finetuning/ft_13_tf1_transfer_clustered/train.py \
        --model bge --size s200 --eval_doc_configs raw

Wave 1:
    ./run.sh finetuning/ft_13_tf1_transfer_clustered/train.py \
        --models bge all_minilm all_mpnet qwen3_0_6b qwen3_4b linq \
        --sizes s200 s500 sfull sfull_n20 \
        --eval_doc_configs raw fable_cot_proverb fable_direct_moral conceptual_abstract_fable \
        --summary_generator gemini --remote --gpu 2

## Outputs

- Per-run JSON:    results/<timestamp>_<model>_<size>.json
- Rankings:        results/rankings/<model>__<size>__<doc_config>__<generator>.json
- Comprehensive:   results/ft13_comprehensive_results.csv
- Trained models:  /data/lior/ft_13_tf1_transfer_clustered/models/<model>/<size>
```

- [ ] **Step 3: Create config.yaml**

Copy `finetuning/ft_12_storal_transfer_clustered/config.yaml` to `finetuning/ft_13_tf1_transfer_clustered/config.yaml` exactly, then apply these surgical changes:

```bash
cp finetuning/ft_12_storal_transfer_clustered/config.yaml \
   finetuning/ft_13_tf1_transfer_clustered/config.yaml
```

Then edit `finetuning/ft_13_tf1_transfer_clustered/config.yaml`:

(a) Replace the **entire `dataset_sizes:` block** with:

```yaml
dataset_sizes:
  s200:
    n_morals: 20
    n_fables_per_moral: 10
    description: "20 TF1 morals x 10 fables each (smallest, probable sweet-spot)."
  s500:
    n_morals: 50
    n_fables_per_moral: 10
    description: "50 TF1 morals x 10 fables each (mid-sweep)."
  sfull:
    n_morals: null
    n_fables_per_moral: 10
    description: "All 100 TF1 morals x 10 fables each."
  sfull_n20:
    n_morals: null
    n_fables_per_moral: 20
    description: "All 100 TF1 morals x 20 fables each (isolates per-moral redundancy)."
```

(b) In every `model_output_dir:` line, replace `ft_12_storal_transfer_clustered` with `ft_13_tf1_transfer_clustered`. Use a single sed:

```bash
sed -i.bak \
  's|ft_12_storal_transfer_clustered|ft_13_tf1_transfer_clustered|g' \
  finetuning/ft_13_tf1_transfer_clustered/config.yaml
rm finetuning/ft_13_tf1_transfer_clustered/config.yaml.bak
```

(c) At the top of the file, immediately under the existing `seed:` etc., add a `source:` section pointing at the new corpus:

```yaml
source:
  tf1_corpus_dir: data/external/tf1_synthetic_low_iou
```

(d) Remove `multilingual_e5_large_instruct` from the `models:` mapping (ft_12 results show MAP@10 = 0.011 — broken). Open the file, find the `multilingual_e5_large_instruct:` entry under `models:`, and delete its YAML block.

- [ ] **Step 4: Verify the config parses and has the right shape**

```bash
python3 -c "
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path('finetuning/ft_13_tf1_transfer_clustered/config.yaml').read_text())
print('sizes:', sorted(cfg['dataset_sizes']))
print('models:', sorted(cfg['models']))
print('source:', cfg['source'])
print('eval doc_configs (count):', len(cfg['doc_configs']))
print('mei in models:', 'multilingual_e5_large_instruct' in cfg['models'])
print('first model output_dir:', list(cfg['models'].values())[0]['model_output_dir'])
"
```

Expected:
```
sizes: ['s200', 's500', 'sfull', 'sfull_n20']
models: ['all_minilm', 'all_mpnet', 'bge', 'bge_large', 'bge_m3', 'contriever', 'e5_base', 'e5_large', 'linq', 'multilingual_e5_large', 'qwen3', 'qwen3_0_6b', 'qwen3_4b', 'sfr']
source: {'tf1_corpus_dir': 'data/external/tf1_synthetic_low_iou'}
eval doc_configs (count): 13
mei in models: False
first model output_dir: /data/lior/ft_13_tf1_transfer_clustered/models/<something>
```

- [ ] **Step 5: Commit**

```bash
git add finetuning/ft_13_tf1_transfer_clustered/__init__.py \
        finetuning/ft_13_tf1_transfer_clustered/README.md \
        finetuning/ft_13_tf1_transfer_clustered/config.yaml
git commit -m "feat(ft_13): scaffolding (README, __init__, config.yaml derived from ft_12)"
```

---

### Task 5: TDD `_subsample_morals` (size-aware row sampler)

**Files:**
- Create: `tests/finetuning/test_ft_13_data_loaders.py`
- (Implementation will land in `finetuning/ft_13_tf1_transfer_clustered/train.py` in the next task.)

- [ ] **Step 1: Write the failing test, including the importlib loader**

Create `tests/finetuning/test_ft_13_data_loaders.py`:

```python
import sys
import json
import importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PATH = (
    Path(__file__).parent.parent.parent
    / "finetuning" / "ft_13_tf1_transfer_clustered" / "train.py"
)
train = _load("ft_13_train", _PATH)


def _pair(idx, moral_id="moral_tf1_000", fable_id_suffix=0):
    return {
        "moral": f"moral text {moral_id}",
        "story": f"fable text {fable_id_suffix}",
        "moral_id": moral_id,
        "fable_id": f"fable_tf1_{fable_id_suffix:05d}",
    }


def test_subsample_morals_keeps_all_when_unrestricted():
    pairs = [_pair(i, "moral_tf1_000", i) for i in range(10)]
    out = train._subsample_morals(pairs, {"n_morals": None, "n_fables_per_moral": 10}, seed=42)
    assert len(out) == 10


def test_subsample_morals_caps_fables_per_moral():
    pairs = [_pair(i, "moral_tf1_000", i) for i in range(20)]
    out = train._subsample_morals(pairs, {"n_morals": None, "n_fables_per_moral": 5}, seed=42)
    assert len(out) == 5
    # First 5 by file order (insertion preserved)
    assert [p["fable_id"] for p in out] == [f"fable_tf1_{i:05d}" for i in range(5)]


def test_subsample_morals_samples_morals_then_takes_all_fables():
    pairs = []
    for moral_idx in range(4):
        for fable_idx in range(10):
            pairs.append(_pair(moral_idx * 10 + fable_idx,
                               moral_id=f"moral_tf1_{moral_idx:03d}",
                               fable_id_suffix=moral_idx * 10 + fable_idx))
    out = train._subsample_morals(pairs, {"n_morals": 2, "n_fables_per_moral": 10}, seed=42)
    morals = {p["moral_id"] for p in out}
    assert len(morals) == 2
    assert len(out) == 20  # 2 morals x 10 fables


def test_subsample_morals_is_seed_stable():
    pairs = []
    for moral_idx in range(10):
        for fable_idx in range(10):
            pairs.append(_pair(moral_idx * 10 + fable_idx,
                               moral_id=f"moral_tf1_{moral_idx:03d}",
                               fable_id_suffix=moral_idx * 10 + fable_idx))
    a = train._subsample_morals(pairs, {"n_morals": 3, "n_fables_per_moral": 5}, seed=42)
    b = train._subsample_morals(pairs, {"n_morals": 3, "n_fables_per_moral": 5}, seed=42)
    assert [(p["moral_id"], p["fable_id"]) for p in a] == [(p["moral_id"], p["fable_id"]) for p in b]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/finetuning/test_ft_13_data_loaders.py -v
```

Expected: `FileNotFoundError` for `train.py` (it doesn't exist yet).

- [ ] **Step 3: Create the minimal train.py with just `_subsample_morals`**

Create `finetuning/ft_13_tf1_transfer_clustered/train.py`:

```python
"""ft_13 — TF1-synthetic transfer evaluated on clustered MORABLES.

See docs/superpowers/specs/2026-05-20-ft-13-tf1-transfer-clustered-design.md
for the full design. This file mirrors ft_12_storal_transfer_clustered/train.py
with surgical replacements in the data layer (TF1 source instead of STORAL,
size-as-morals-and-fables-per-moral instead of size-as-row-count, label=moral_id).
"""
from __future__ import annotations

import random
from collections import defaultdict


def _subsample_morals(pairs: list[dict], size_cfg: dict, seed: int) -> list[dict]:
    """Sample n_morals distinct morals (or keep all), then take the first
    n_fables_per_moral fables for each.

    Sub-sampling is deterministic for a given (pairs, size_cfg, seed). Because
    the low-IoU corpus stores fables per moral in ascending-IoU order, taking
    the first K is equivalent to taking the K lowest-IoU fables for that moral.
    """
    by_moral: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_moral[p["moral_id"]].append(p)
    moral_ids = sorted(by_moral.keys())

    n_morals = size_cfg.get("n_morals")
    if n_morals is not None and n_morals < len(moral_ids):
        rng = random.Random(seed)
        moral_ids = rng.sample(moral_ids, n_morals)

    n_fables = size_cfg["n_fables_per_moral"]
    return [p for mid in moral_ids for p in by_moral[mid][:n_fables]]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/finetuning/test_ft_13_data_loaders.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add finetuning/ft_13_tf1_transfer_clustered/train.py tests/finetuning/test_ft_13_data_loaders.py
git commit -m "feat(ft_13): add _subsample_morals helper with tests"
```

---

### Task 6: TDD `split_tf1_groups` + `make_tf1_dataset`

**Files:**
- Modify: `finetuning/ft_13_tf1_transfer_clustered/train.py`
- Modify: `tests/finetuning/test_ft_13_data_loaders.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/finetuning/test_ft_13_data_loaders.py`:

```python
def test_split_tf1_groups_separates_by_moral_id(tmp_path):
    pairs = []
    for moral_idx in range(10):
        for fable_idx in range(10):
            pairs.append(_pair(moral_idx * 10 + fable_idx,
                               moral_id=f"moral_tf1_{moral_idx:03d}",
                               fable_id_suffix=moral_idx * 10 + fable_idx))
    train_rows, val_rows = train.split_tf1_groups(pairs, seed=42, validation_ratio=0.20)
    train_morals = {p["moral_id"] for p in train_rows}
    val_morals = {p["moral_id"] for p in val_rows}
    assert train_morals.isdisjoint(val_morals)
    assert len(val_morals) == 2  # 20% of 10 morals
    assert len(train_morals) == 8
    assert len(train_rows) == 80
    assert len(val_rows) == 20


def test_split_tf1_groups_seed_stable():
    pairs = [_pair(i, f"moral_tf1_{i//10:03d}", i) for i in range(100)]
    a = train.split_tf1_groups(pairs, seed=42, validation_ratio=0.10)
    b = train.split_tf1_groups(pairs, seed=42, validation_ratio=0.10)
    assert [p["fable_id"] for p in a[0]] == [p["fable_id"] for p in b[0]]
    assert [p["fable_id"] for p in a[1]] == [p["fable_id"] for p in b[1]]


def test_make_tf1_dataset_assigns_integer_labels_per_moral():
    pairs = [
        _pair(0, "moral_tf1_000", 0),
        _pair(1, "moral_tf1_000", 1),
        _pair(2, "moral_tf1_001", 2),
    ]
    ds = train.make_tf1_dataset(pairs, instruction="Q: ")
    assert ds["anchor"] == ["Q: moral text moral_tf1_000", "Q: moral text moral_tf1_000", "Q: moral text moral_tf1_001"]
    assert ds["positive"] == ["fable text 0", "fable text 1", "fable text 2"]
    # Same moral_id -> same int label
    assert ds["label"][0] == ds["label"][1]
    # Different moral_id -> different int label
    assert ds["label"][0] != ds["label"][2]


def test_make_tf1_dataset_empty_instruction_passes_moral_verbatim():
    pairs = [_pair(0, "moral_tf1_000", 0)]
    ds = train.make_tf1_dataset(pairs, instruction="")
    assert ds["anchor"] == ["moral text moral_tf1_000"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/finetuning/test_ft_13_data_loaders.py -v
```

Expected: 4 new tests fail with `AttributeError`.

- [ ] **Step 3: Implement both functions**

Append to `finetuning/ft_13_tf1_transfer_clustered/train.py`:

```python
def split_tf1_groups(
    pairs: list[dict], seed: int, validation_ratio: float
) -> tuple[list[dict], list[dict]]:
    """Group-aware train/val split: morals (not rows) are sampled into the
    validation set; all fables of a given moral go together. Prevents within-
    moral leakage.
    """
    by_moral: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_moral[p["moral_id"]].append(p)
    moral_ids = sorted(by_moral.keys())
    rng = random.Random(seed)
    rng.shuffle(moral_ids)
    n_val = max(1, round(len(moral_ids) * validation_ratio))
    val_morals = set(moral_ids[:n_val])
    train_rows = [p for mid in moral_ids[n_val:] for p in by_moral[mid]]
    val_rows = [p for mid in moral_ids[:n_val] for p in by_moral[mid]]
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def make_tf1_dataset(pairs: list[dict], instruction: str):
    """Build the 3-column training Dataset: anchor=instruction+moral,
    positive=fable, label=integer-per-moral (for InfoNCELoss multi-positive
    masking). In exact-cluster mode, label = moral_id integer index.
    """
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

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/finetuning/test_ft_13_data_loaders.py -v
```

Expected: all 8 tests in this file pass.

- [ ] **Step 5: Commit**

```bash
git add finetuning/ft_13_tf1_transfer_clustered/train.py tests/finetuning/test_ft_13_data_loaders.py
git commit -m "feat(ft_13): add split_tf1_groups and make_tf1_dataset"
```

---

### Task 7: TDD `load_tf1_synthetic_exact`

**Files:**
- Modify: `finetuning/ft_13_tf1_transfer_clustered/train.py`
- Modify: `tests/finetuning/test_ft_13_data_loaders.py`

- [ ] **Step 1: Add the failing integration test**

Append to `tests/finetuning/test_ft_13_data_loaders.py`:

```python
def test_load_tf1_synthetic_exact_from_tmp_dir(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    morals = [
        {"doc_id": "moral_tf1_000", "text": "be kind"},
        {"doc_id": "moral_tf1_001", "text": "be brave"},
    ]
    fables = [
        {"doc_id": "fable_tf1_00000", "text": "Kind story 0",
         "moral_id": "moral_tf1_000", "source_idx": 0, "source_chunk": 0, "prompt_hash": "h0"},
        {"doc_id": "fable_tf1_00001", "text": "Kind story 1",
         "moral_id": "moral_tf1_000", "source_idx": 1, "source_chunk": 0, "prompt_hash": "h1"},
        {"doc_id": "fable_tf1_00002", "text": "Brave story 0",
         "moral_id": "moral_tf1_001", "source_idx": 2, "source_chunk": 0, "prompt_hash": "h2"},
    ]
    qrels = [
        {"query_id": "moral_tf1_000", "doc_id": "fable_tf1_00000", "relevance": 1},
        {"query_id": "moral_tf1_000", "doc_id": "fable_tf1_00001", "relevance": 1},
        {"query_id": "moral_tf1_001", "doc_id": "fable_tf1_00002", "relevance": 1},
    ]
    (processed / "morals_corpus.json").write_text(json.dumps(morals))
    (processed / "fables_corpus.json").write_text(json.dumps(fables))
    (processed / "qrels_moral_to_fable.json").write_text(json.dumps(qrels))

    pairs, stats = train.load_tf1_synthetic_exact(
        size_cfg={"n_morals": None, "n_fables_per_moral": 10},
        seed=42,
        source_dir=tmp_path,
    )
    assert len(pairs) == 3
    # Pair schema
    sample = pairs[0]
    assert set(sample.keys()) >= {"moral", "story", "moral_id", "fable_id"}
    # Stats keys
    assert stats["raw_total"] == 3
    assert stats["selected_rows"] == 3
    assert stats["selected_morals"] == 2
    assert stats["selection_strategy"] == "tmp"  # last segment of tmp_path stays
    assert stats["size_config"]["n_fables_per_moral"] == 10


def test_load_tf1_synthetic_exact_size_subsamples(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    morals = [{"doc_id": f"moral_tf1_{i:03d}", "text": f"moral {i}"} for i in range(5)]
    fables = []
    qrels = []
    for moral_idx in range(5):
        for fable_idx in range(4):
            fid = f"fable_tf1_{moral_idx * 4 + fable_idx:05d}"
            fables.append({
                "doc_id": fid, "text": f"story {moral_idx}/{fable_idx}",
                "moral_id": f"moral_tf1_{moral_idx:03d}",
                "source_idx": moral_idx * 4 + fable_idx,
                "source_chunk": 0, "prompt_hash": f"h{moral_idx}{fable_idx}",
            })
            qrels.append({
                "query_id": f"moral_tf1_{moral_idx:03d}",
                "doc_id": fid, "relevance": 1,
            })
    (processed / "morals_corpus.json").write_text(json.dumps(morals))
    (processed / "fables_corpus.json").write_text(json.dumps(fables))
    (processed / "qrels_moral_to_fable.json").write_text(json.dumps(qrels))

    pairs, stats = train.load_tf1_synthetic_exact(
        size_cfg={"n_morals": 2, "n_fables_per_moral": 3},
        seed=42, source_dir=tmp_path,
    )
    assert len(pairs) == 6  # 2 morals * 3 fables
    assert stats["selected_morals"] == 2
    assert stats["selected_rows"] == 6
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/finetuning/test_ft_13_data_loaders.py -v
```

Expected: 2 new tests fail with `AttributeError`.

- [ ] **Step 3: Implement `load_tf1_synthetic_exact`**

Append to `finetuning/ft_13_tf1_transfer_clustered/train.py`:

```python
import json  # add to top of file if not already imported
from pathlib import Path  # add if not already imported


def load_tf1_synthetic_exact(
    size_cfg: dict, seed: int, source_dir: Path
) -> tuple[list[dict], dict]:
    """Load TF1 exact-cluster pairs from source_dir/processed/, then sub-sample
    per the size config.

    Returns (pairs, stats):
        pairs: list of dicts with keys {moral, story, moral_id, fable_id}
        stats: {raw_total, selected_rows, selected_morals, selection_strategy,
                size_config}
    """
    morals = json.loads((source_dir / "processed" / "morals_corpus.json").read_text())
    fables = json.loads((source_dir / "processed" / "fables_corpus.json").read_text())
    qrels = json.loads((source_dir / "processed" / "qrels_moral_to_fable.json").read_text())

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
    raw_total = len(pairs)
    pairs = _subsample_morals(pairs, size_cfg, seed)
    selected_morals = len({p["moral_id"] for p in pairs})

    stats = {
        "raw_total": raw_total,
        "selected_rows": len(pairs),
        "selected_morals": selected_morals,
        "selection_strategy": source_dir.name,
        "size_config": dict(size_cfg),
    }
    return pairs, stats
```

Move the `import json` and `from pathlib import Path` lines to the top of the file with the other imports (where `import random` and `from collections import defaultdict` already are).

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/finetuning/test_ft_13_data_loaders.py -v
```

Expected: all 10 tests in this file pass.

- [ ] **Step 5: Commit**

```bash
git add finetuning/ft_13_tf1_transfer_clustered/train.py tests/finetuning/test_ft_13_data_loaders.py
git commit -m "feat(ft_13): add load_tf1_synthetic_exact loader"
```

---

### Task 8: Wire `main()` and `run_one()` — copy ft_12 with three surgical changes

**Files:**
- Modify: `finetuning/ft_13_tf1_transfer_clustered/train.py`

This task is the largest single edit: we copy the entire body of `ft_12_storal_transfer_clustered/train.py` (everything except the data-layer functions we already wrote in Tasks 5–7) and apply three surgical replacements.

- [ ] **Step 1: Copy the ft_12 train.py body verbatim**

Open `finetuning/ft_12_storal_transfer_clustered/train.py`. Copy everything from the top-of-file imports through the end of `main()` and `if __name__ == "__main__":`, **but skip** these three functions which we'll replace:

- `load_storal` (lines roughly 134–168)
- `split_storal_groups` (lines roughly 171–199)
- `make_storal_dataset` (lines roughly 213–223) — and the helper `assign_exact_labels` (lines roughly 202–210)
- `normalize_moral` (lines roughly 127–131) — TF1 doesn't need it; moral_id is the canonical identity.

Append the rest after the helpers from Tasks 5–7 in `finetuning/ft_13_tf1_transfer_clustered/train.py`, **then** apply changes (a)–(d) below.

(a) **Update the module docstring** at the top to read:
```
"""ft_13 — TF1-synthetic transfer evaluated on clustered MORABLES.

See docs/superpowers/specs/2026-05-20-ft-13-tf1-transfer-clustered-design.md.
Mirrors ft_12_storal_transfer_clustered with TF1-specific data loader,
size-config interpretation (n_morals + n_fables_per_moral), and
label=moral_id assignment for InfoNCELoss multi-positive masking.
"""
```

(b) **Replace `EXPERIMENT_NAME` and path constants**:

```python
EXPERIMENT_NAME = "ft_13_tf1_transfer_clustered"
EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
CONFIG_PATH = EXP_DIR / "config.yaml"
RESULTS_DIR = EXP_DIR / "results"
RANKINGS_DIR = RESULTS_DIR / "rankings"
COMPREHENSIVE_CSV = RESULTS_DIR / "ft13_comprehensive_results.csv"

# Read TF1 corpus dir from the config later in main(); no hard-coded STORAL_PATH.

# Clustered MORABLES (read-only target)
MORALS_PATH = ROOT / "data" / "clustered" / "morals_unique_corpus.json"
FABLES_PATH = ROOT / "data" / "clustered" / "fables_corpus.json"
QRELS_PATH = ROOT / "data" / "clustered" / "qrels_moral_to_fable_clustered.json"
```

(c) **Replace `CSV_FIELDNAMES`** to:
- Rename `n_storal_selected` → `n_tf1_rows_selected`
- Insert `n_tf1_morals_selected` and `selection_strategy` after it

The full updated list, in order:
```python
CSV_FIELDNAMES = [
    "timestamp", "experiment", "model_alias", "model_id", "size",
    "n_tf1_rows_selected", "n_tf1_morals_selected", "selection_strategy",
    "train_rows", "validation_rows", "unique_normalized_morals",
    "exact_duplicate_moral_groups", "exact_duplicate_moral_rows",
    "eval_doc_config", "summary_generator", "corpus_template", "rankings_path",
    "MAP@10", "MRR@10", "NDCG@10",
    "Recall@5", "Recall@10", "Recall@15", "Recall@50", "Recall@100",
    "Recall@200", "Recall@300",
    "Hit@1", "Hit@5", "Hit@10", "Hit@100",
    "Mean_Rank", "Median_Rank", "n_queries", "seed",
    "early_stopping_metric", "model_dir", "error",
]
```

(d) **In `run_one` (or equivalent — the function that loads source data and assembles datasets)**, replace the STORAL call with TF1:

Find the line in ft_12:
```python
storal_pairs, storal_stats = load_storal(size_cfg, seed)
```

Replace it with:
```python
tf1_corpus_dir = ROOT / config["source"]["tf1_corpus_dir"]
tf1_pairs, tf1_stats = load_tf1_synthetic_exact(size_cfg, seed, tf1_corpus_dir)
```

Then replace any references to `storal_pairs`, `storal_stats`, `split_storal_groups`, `make_storal_dataset` with their TF1 counterparts: `tf1_pairs`, `tf1_stats`, `split_tf1_groups`, `make_tf1_dataset`.

In the result dict at the end of `run_one`, rename:
```python
"storal_stats": storal_stats,
```
to:
```python
"tf1_stats": tf1_stats,
```

(e) **Update `append_comprehensive_csv`** to populate the new/renamed columns and keep STORAL-specific ones as constants. Inside the `rows.append({...})` block for the success path, replace:
```python
"n_storal_selected": result["storal_stats"]["selected"],
...
"unique_normalized_morals": result["storal_stats"]["unique_normalized_morals"],
"exact_duplicate_moral_groups": result["storal_stats"]["exact_duplicate_moral_groups"],
"exact_duplicate_moral_rows": result["storal_stats"]["exact_duplicate_moral_rows"],
```
with:
```python
"n_tf1_rows_selected": result["tf1_stats"]["selected_rows"],
"n_tf1_morals_selected": result["tf1_stats"]["selected_morals"],
"selection_strategy": result["tf1_stats"]["selection_strategy"],
...
"unique_normalized_morals": result["tf1_stats"]["selected_morals"],   # same as morals for TF1
"exact_duplicate_moral_groups": 0,
"exact_duplicate_moral_rows": 0,
```

(f) **Update Telegram messages and the description string** in `argparse`:
- `description="ft_13 TF1 transfer -> clustered MORABLES"`
- All `"ft_12_storal_transfer_clustered"` literals → `EXPERIMENT_NAME` (= `"ft_13_tf1_transfer_clustered"`)
- The Telegram start message: `f"{EXPERIMENT_NAME} starting\n..."`
- Per-run done message: `f"ft_13 run done\n..."`
- Per-run failed message: `f"ft_13 run failed\n..."`
- Queue summary message: `f"ft_13 queue done\n..."`

(g) **Add `--tf1_corpus_dir` override** in `argparse`:
```python
parser.add_argument(
    "--tf1_corpus_dir", type=Path, default=None,
    help="Override the TF1 corpus dir from config (for ablation against the "
         "random-selection corpus). Defaults to config['source']['tf1_corpus_dir'].",
)
```

In `main()`, after reading config:
```python
if args.tf1_corpus_dir is not None:
    config["source"]["tf1_corpus_dir"] = str(args.tf1_corpus_dir)
```

- [ ] **Step 2: Run a quick syntax + import check**

```bash
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
import importlib.util
spec = importlib.util.spec_from_file_location(
    'ft_13_train',
    'finetuning/ft_13_tf1_transfer_clustered/train.py',
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('module loaded successfully')
print('has main:', hasattr(mod, 'main'))
print('has run_one:', hasattr(mod, 'run_one'))
print('CSV fields:', len(mod.CSV_FIELDNAMES))
"
```

Expected: `module loaded successfully`, `has main: True`, `has run_one: True`, `CSV fields: 37`.

- [ ] **Step 3: Run all data-layer tests again to confirm no regression**

```bash
uv run pytest tests/finetuning/test_ft_13_data_loaders.py -v
```

Expected: all 10 tests still pass.

- [ ] **Step 4: Verify argparse runs `--help` without error**

```bash
./run.sh finetuning/ft_13_tf1_transfer_clustered/train.py --help
```

Expected: argparse `--help` text prints all flags including `--tf1_corpus_dir`. No tracebacks.

- [ ] **Step 5: Commit**

```bash
git add finetuning/ft_13_tf1_transfer_clustered/train.py
git commit -m "feat(ft_13): wire main/run_one by adapting ft_12 with TF1 surgical changes"
```

---

### Task 9: Smoke test execution (BGE on s200, raw)

**Files:** None modified; produces files under `finetuning/ft_13_tf1_transfer_clustered/results/` and `/data/lior/`.

- [ ] **Step 1: Verify prerequisites**

```bash
ls data/external/tf1_synthetic_low_iou/processed/ data/external/tf1_synthetic_low_iou/clustered/
```

Expected to see the 4 processed JSONs and the 5 clustered exact JSONs from Task 3.

```bash
ls data/clustered/morals_unique_corpus.json data/clustered/fables_corpus.json data/clustered/qrels_moral_to_fable_clustered.json
```

Expected: all three exist (read-only target).

- [ ] **Step 2: Run the smoke test**

```bash
./run.sh finetuning/ft_13_tf1_transfer_clustered/train.py \
    --model bge --size s200 --eval_doc_configs raw
```

Expected console output (key lines):
- `ft_13_tf1_transfer_clustered starting ...` (Telegram message also fired)
- `[load] TF1 source: data/external/tf1_synthetic_low_iou` (or equivalent — depends on log lines inherited from ft_12)
- Training progress for `bge` on `s200` (180 train rows, 20 val rows)
- Final eval line per doc_config: `[raw/none] MRR@10=... MAP@10=... Hit@10=... Recall@100=...`
- `Results -> finetuning/ft_13_tf1_transfer_clustered/results/<timestamp>_bge_s200.json`
- `ft_13 run done ...` (Telegram)

If training fails, **escalate** — do not silently retry. Inspect the traceback for missing keys (the most likely cause is a config or stats key the ft_12 code references that we didn't populate; fix in train.py, commit, and re-run).

- [ ] **Step 3: Verify the CSV row was appended**

```bash
python3 -c "
from pathlib import Path
import csv
p = Path('finetuning/ft_13_tf1_transfer_clustered/results/ft13_comprehensive_results.csv')
assert p.exists(), 'CSV not created'
with p.open() as f:
    rows = list(csv.DictReader(f))
print('rows:', len(rows))
print('cols:', len(rows[0]))
print('experiment:', rows[0]['experiment'])
print('model_alias:', rows[0]['model_alias'])
print('size:', rows[0]['size'])
print('eval_doc_config:', rows[0]['eval_doc_config'])
print('MAP@10:', rows[0]['MAP@10'])
print('selection_strategy:', rows[0]['selection_strategy'])
print('n_tf1_rows_selected:', rows[0]['n_tf1_rows_selected'])
print('n_tf1_morals_selected:', rows[0]['n_tf1_morals_selected'])
"
```

Expected:
- `rows: 1`
- `cols: 37`
- `experiment: ft_13_tf1_transfer_clustered`
- `model_alias: bge`
- `size: s200`
- `eval_doc_config: raw`
- `MAP@10: <some float, non-empty>`
- `selection_strategy: tf1_synthetic_low_iou`
- `n_tf1_rows_selected: 200`
- `n_tf1_morals_selected: 20`

- [ ] **Step 4: Verify the ranking file was written**

```bash
python3 -c "
from pathlib import Path
import json
p = Path('finetuning/ft_13_tf1_transfer_clustered/results/rankings/bge__s200__raw__none.json')
assert p.exists(), f'Missing ranking file at {p}'
data = json.loads(p.read_text())
print('rows (= MORABLES queries):', len(data))
print('first query_id:', data[0]['query_id'])
print('ranked_fable_ids (top 3):', data[0]['ranked_fable_ids'][:3])
print('all fables ranked:', len(data[0]['ranked_fable_ids']))
print('scores (top 3):', data[0]['scores'][:3])
"
```

Expected:
- `rows (= MORABLES queries): 668`
- `first query_id: moral_unique_0000` (or similar)
- `ranked_fable_ids (top 3): ['fable_XXXX', 'fable_XXXX', 'fable_XXXX']`
- `all fables ranked: 709`
- `scores (top 3): [<float>, <float>, <float>]` with monotonically decreasing values

- [ ] **Step 5: Verify the model was saved**

```bash
ls /data/lior/ft_13_tf1_transfer_clustered/models/bge/s200/ 2>/dev/null || ssh "$GPU_USER@$GPU_HOST" "ls /data/lior/ft_13_tf1_transfer_clustered/models/bge/s200/"
```

Expected: at least `config_sentence_transformers.json`, `modules.json`, and a `1_Pooling/` subdir (the typical sentence-transformers save layout).

- [ ] **Step 6: Commit if needed**

If the smoke test produced any source-code fixes (e.g., a missing config key surfaced), commit them. The CSV row and ranking file are written under `finetuning/ft_13_tf1_transfer_clustered/results/`; verify whether this dir is gitignored by running `git check-ignore -v finetuning/ft_13_tf1_transfer_clustered/results/ft13_comprehensive_results.csv`. If not gitignored, commit the CSV + ranking + per-run JSON as smoke-test artifacts:

```bash
git status finetuning/ft_13_tf1_transfer_clustered/
git add finetuning/ft_13_tf1_transfer_clustered/results/
git commit -m "data(ft_13): smoke-test artifacts (bge/s200/raw)"
```

If the results dir IS gitignored (per the existing `experiments/*/results/runs/` pattern), nothing to commit — the run is logged in `/data/lior/` and the smoke-test pass criteria are met locally.

---

### Task 10: Push the branch

**Files:** None modified.

- [ ] **Step 1: Verify all commits land cleanly**

```bash
git log --oneline aa68976..HEAD | head -30
git status
```

Expected: clean working tree, 9–10 new commits beyond the spec/plan, all messages start with `feat(ft_13)`, `feat(tf1_synthetic)`, or `data(...)`.

- [ ] **Step 2: Push with the large-buffer flag**

The branch already exists upstream; we just push the new commits. Use the larger `http.postBuffer` because earlier commits in this branch include the 879 KB embedded-figure HTML:

```bash
git -c http.postBuffer=524288000 push origin tf1-iou-diagnostic
```

Expected: `<sha>..<sha> tf1-iou-diagnostic -> tf1-iou-diagnostic` (no errors).

- [ ] **Step 3: Verify upstream is in sync**

```bash
git ls-remote origin tf1-iou-diagnostic
git log -1 --format="%H" origin/tf1-iou-diagnostic
```

Expected: both SHAs match.

---

## Self-review notes

- **Spec coverage:** every spec requirement maps to at least one task:
  - Spec §"Inputs / Source training corpus / new low-IoU build" → Tasks 1–3
  - Spec §"Architecture / Change 1 — source data loader" → Task 7
  - Spec §"Architecture / Change 2 — dataset assembly" → Task 6
  - Spec §"Architecture / Change 3 — validation split" → Task 6
  - Spec §"New: extending build_tf1_corpus.py with --selection" → Tasks 1–2
  - Spec §"Negative-leakage guarantees" → label assignment in Task 6 (the multi-positive mask in `InfoNCELoss` already exists); no separate task because no new code is needed
  - Spec §"Configuration / Sizes" + "Models" + "Eval doc_configs" → Task 4 (config.yaml)
  - Spec §"Storage" → Task 4 (config.yaml `model_output_dir`)
  - Spec §"Smoke test" → Task 9
- **Type consistency:** `pair = {moral, story, moral_id, fable_id}` used identically in Tasks 5–7 and the train script. `size_cfg = {n_morals, n_fables_per_moral}` used identically. CSV field names enumerated explicitly in Task 8 step (c).
- **No placeholders:** every code step contains the full code.
- **Estimated wall time:** Tasks 1–3 ≈ 30 min (corpus build is seconds). Tasks 4–8 ≈ 60 min. Task 9 smoke test is the GPU-bound piece — depends on GPU availability and model size (BGE-base on s200 should fit comfortably in 20 min).
