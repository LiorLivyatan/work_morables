# TF1-synthetic corpus build implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a MORABLES-shaped derivative of TF1-EN-3M at `data/external/tf1_synthetic/{processed,clustered}/`, plus two scripts under `experiments/11_tf1_diagnostic/` that produce it from the existing 50K sample dump.

**Architecture:** Two scripts, each split into pure helpers (unit-tested) and a thin `main()` orchestrator. `build_tf1_corpus.py` consumes the diagnostic's `samples.jsonl` cache and emits the four `processed/` files. `cluster_tf1_morals.py` consumes `processed/` and a sentence-transformer embedder to emit the five `clustered/` files. No re-streaming; the cache caps N at ~500/moral.

**Tech Stack:** Python 3.13, `numpy`, `scipy.cluster.hierarchy` for agglomerative clustering, `sentence-transformers` (BAAI/bge-large-en-v1.5) for moral embeddings, `pytest` for tests. All runs invoked via `./run.sh`.

**Spec:** [`docs/superpowers/specs/2026-05-06-tf1-synthetic-corpus-design.md`](../specs/2026-05-06-tf1-synthetic-corpus-design.md)

---

## File structure

| File | Responsibility |
|---|---|
| `data/external/.gitignore` (modify) | Add `tf1_synthetic/raw/` exclusion |
| `data/external/tf1_synthetic/` (new) | Corpus output dir (created at runtime) |
| `experiments/11_tf1_diagnostic/build_tf1_corpus.py` (new) | Pure helpers + main; reads `samples.jsonl`, writes 4 `processed/` files |
| `experiments/11_tf1_diagnostic/cluster_tf1_morals.py` (new) | Pure helpers + main; reads `processed/`, embeds, clusters, writes 5 `clustered/` files |
| `tests/experiments/__init__.py` (new) | Marker file |
| `tests/experiments/test_build_tf1_corpus.py` (new) | Unit + integration tests for build helpers and main |
| `tests/experiments/test_cluster_tf1_morals.py` (new) | Unit + integration tests for cluster helpers and main |

---

### Task 1: Setup — gitignore and test package marker

**Files:**
- Modify: `data/external/.gitignore`
- Create: `tests/experiments/__init__.py`

- [ ] **Step 1: Add gitignore entry for the future raw dir**

Edit `data/external/.gitignore` to add the second line:

```
# Raw external dataset files are large — keep processed outputs only
storal/raw/
tf1_synthetic/raw/
```

- [ ] **Step 2: Create the tests/experiments package marker**

```python
# tests/experiments/__init__.py
```
(empty file)

- [ ] **Step 3: Commit**

```bash
git add data/external/.gitignore tests/experiments/__init__.py
git commit -m "chore(tf1_synthetic): gitignore raw/ and add tests/experiments package"
```

---

### Task 2: TDD build_tf1_corpus pure helpers (group, first-seen, ids, sample)

**Files:**
- Create: `experiments/11_tf1_diagnostic/build_tf1_corpus.py`
- Create: `tests/experiments/test_build_tf1_corpus.py`

- [ ] **Step 1: Write the failing tests for helpers**

```python
# tests/experiments/test_build_tf1_corpus.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from experiments._11_tf1_diagnostic.build_tf1_corpus import (
    group_by_moral,
    first_seen_order,
    assign_moral_ids,
    sample_n_per_moral,
)


def _row(idx, moral, fable="...", chunk=0, prompt_hash="h"):
    return {"idx": idx, "chunk": chunk, "prompt_hash": prompt_hash, "moral": moral, "fable": fable}


def test_group_by_moral_lowercases_and_strips():
    rows = [_row(0, " Greed Leads To Downfall "), _row(1, "greed leads to downfall")]
    groups = group_by_moral(rows)
    assert list(groups.keys()) == ["greed leads to downfall"]
    assert len(groups["greed leads to downfall"]) == 2


def test_first_seen_order_preserves_insertion():
    rows = [_row(0, "B"), _row(1, "A"), _row(2, "B"), _row(3, "C")]
    assert first_seen_order(rows) == ["b", "a", "c"]


def test_assign_moral_ids_zero_padded():
    ids = assign_moral_ids(["a", "b", "c"])
    assert ids == {"a": "moral_tf1_000", "b": "moral_tf1_001", "c": "moral_tf1_002"}


def test_sample_n_per_moral_is_deterministic_with_same_seed():
    grouped = {"m": [_row(i, "m") for i in range(20)]}
    out_a = sample_n_per_moral(grouped, n=5, seed=42)
    out_b = sample_n_per_moral(grouped, n=5, seed=42)
    assert [r["idx"] for r in out_a["m"]] == [r["idx"] for r in out_b["m"]]


def test_sample_n_per_moral_differs_with_different_seed():
    grouped = {"m": [_row(i, "m") for i in range(100)]}
    out_a = sample_n_per_moral(grouped, n=10, seed=1)
    out_b = sample_n_per_moral(grouped, n=10, seed=2)
    assert [r["idx"] for r in out_a["m"]] != [r["idx"] for r in out_b["m"]]


def test_sample_n_per_moral_errors_when_not_enough_rows():
    grouped = {"m": [_row(i, "m") for i in range(3)]}
    with pytest.raises(ValueError, match="only 3"):
        sample_n_per_moral(grouped, n=10, seed=42)
```

Note: import path uses `experiments._11_tf1_diagnostic` because Python identifiers can't start with a digit. Create a symlink or use `importlib`:

Actually — the cleaner solution is to import via `importlib` because `experiments/11_tf1_diagnostic/` starts with a digit. Update the test imports:

```python
import importlib.util
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

_BUILD_PATH = Path(__file__).parent.parent.parent / "experiments" / "11_tf1_diagnostic" / "build_tf1_corpus.py"
build = _load_module("build_tf1_corpus", _BUILD_PATH)
group_by_moral = build.group_by_moral
first_seen_order = build.first_seen_order
assign_moral_ids = build.assign_moral_ids
sample_n_per_moral = build.sample_n_per_moral
```

Use that loader at the top of the test file, then drop the broken `from experiments._11_...` import. Replace the original import block with the loader block.

- [ ] **Step 2: Run tests to verify they fail**

```bash
./run.sh -m pytest tests/experiments/test_build_tf1_corpus.py -v 2>&1 | tail -20
```

Expected: ModuleNotFoundError / FileNotFoundError for `build_tf1_corpus.py`.

Note: `run.sh` accepts pytest invocation. If `-m` isn't supported, run directly:
```bash
uv run pytest tests/experiments/test_build_tf1_corpus.py -v
```

- [ ] **Step 3: Implement the helpers**

Create `experiments/11_tf1_diagnostic/build_tf1_corpus.py` with:

```python
"""
Build a MORABLES-shaped derivative of TF1-EN-3M under
data/external/tf1_synthetic/processed/ from the diagnostic's samples.jsonl
cache. See docs/superpowers/specs/2026-05-06-tf1-synthetic-corpus-design.md.
"""
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DEFAULT_RUNS_DIR = ROOT / "experiments" / "11_tf1_diagnostic" / "results" / "runs"
DEFAULT_OUT = ROOT / "data" / "external" / "tf1_synthetic"


def group_by_moral(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        key = r["moral"].lower().strip()
        out.setdefault(key, []).append(r)
    return out


def first_seen_order(rows: list[dict]) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for r in rows:
        key = r["moral"].lower().strip()
        if key not in seen:
            seen.add(key)
            order.append(key)
    return order


def assign_moral_ids(unique_morals: list[str]) -> dict[str, str]:
    return {m: f"moral_tf1_{i:03d}" for i, m in enumerate(unique_morals)}


def sample_n_per_moral(grouped: dict[str, list[dict]], n: int, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {}
    for moral, rows in grouped.items():
        if len(rows) < n:
            raise ValueError(
                f"Moral has only {len(rows)} cached rows, need {n}: {moral!r}"
            )
        sorted_rows = sorted(rows, key=lambda r: r["idx"])
        out[moral] = rng.sample(sorted_rows, n)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/experiments/test_build_tf1_corpus.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/11_tf1_diagnostic/build_tf1_corpus.py tests/experiments/test_build_tf1_corpus.py
git commit -m "feat(tf1_synthetic): add build_tf1_corpus helpers (group/order/ids/sample)"
```

---

### Task 3: TDD build_tf1_corpus output assemblers (4 file builders)

**Files:**
- Modify: `experiments/11_tf1_diagnostic/build_tf1_corpus.py`
- Modify: `tests/experiments/test_build_tf1_corpus.py`

- [ ] **Step 1: Write the failing tests for output builders**

Append to `tests/experiments/test_build_tf1_corpus.py`:

```python
build_morals_corpus = build.build_morals_corpus
build_fables_corpus = build.build_fables_corpus
build_qrels_moral_to_fable = build.build_qrels_moral_to_fable
build_qrels_fable_to_moral = build.build_qrels_fable_to_moral


def test_build_morals_corpus_shape():
    unique = ["greed leads to downfall", "honesty wins"]
    ids = {"greed leads to downfall": "moral_tf1_000", "honesty wins": "moral_tf1_001"}
    out = build_morals_corpus(unique, ids)
    assert out == [
        {"doc_id": "moral_tf1_000", "text": "greed leads to downfall"},
        {"doc_id": "moral_tf1_001", "text": "honesty wins"},
    ]


def test_build_fables_corpus_assigns_globally_unique_ids():
    sampled = {
        "a": [_row(10, "a", fable="F0", chunk=1, prompt_hash="h0"),
              _row(11, "a", fable="F1", chunk=1, prompt_hash="h1")],
        "b": [_row(20, "b", fable="G0", chunk=2, prompt_hash="h2"),
              _row(21, "b", fable="G1", chunk=2, prompt_hash="h3")],
    }
    unique = ["a", "b"]
    ids = {"a": "moral_tf1_000", "b": "moral_tf1_001"}
    out = build_fables_corpus(sampled, unique, ids, n=2)
    assert [f["doc_id"] for f in out] == [
        "fable_tf1_00000", "fable_tf1_00001",
        "fable_tf1_00002", "fable_tf1_00003",
    ]
    assert out[0]["moral_id"] == "moral_tf1_000"
    assert out[2]["moral_id"] == "moral_tf1_001"
    assert out[0]["source_idx"] == 10
    assert out[0]["prompt_hash"] == "h0"


def test_build_qrels_moral_to_fable_pair_per_row():
    fables = [
        {"doc_id": "fable_tf1_00000", "moral_id": "moral_tf1_000"},
        {"doc_id": "fable_tf1_00001", "moral_id": "moral_tf1_000"},
    ]
    qrels = build_qrels_moral_to_fable(fables)
    assert qrels == [
        {"query_id": "moral_tf1_000", "doc_id": "fable_tf1_00000", "relevance": 1},
        {"query_id": "moral_tf1_000", "doc_id": "fable_tf1_00001", "relevance": 1},
    ]


def test_build_qrels_fable_to_moral_is_inverse():
    fables = [
        {"doc_id": "fable_tf1_00000", "moral_id": "moral_tf1_000"},
        {"doc_id": "fable_tf1_00001", "moral_id": "moral_tf1_001"},
    ]
    qrels = build_qrels_fable_to_moral(fables)
    assert qrels == [
        {"query_id": "fable_tf1_00000", "doc_id": "moral_tf1_000", "relevance": 1},
        {"query_id": "fable_tf1_00001", "doc_id": "moral_tf1_001", "relevance": 1},
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/experiments/test_build_tf1_corpus.py -v
```

Expected: 4 new tests FAIL with AttributeError (functions don't exist).

- [ ] **Step 3: Implement the builders**

Append to `experiments/11_tf1_diagnostic/build_tf1_corpus.py`:

```python
def build_morals_corpus(unique_morals: list[str], moral_ids: dict[str, str]) -> list[dict]:
    return [{"doc_id": moral_ids[m], "text": m} for m in unique_morals]


def build_fables_corpus(
    sampled: dict[str, list[dict]],
    unique_morals: list[str],
    moral_ids: dict[str, str],
    n: int,
) -> list[dict]:
    out: list[dict] = []
    for moral_idx, moral_text in enumerate(unique_morals):
        for i, row in enumerate(sampled[moral_text]):
            fable_id = f"fable_tf1_{moral_idx * n + i:05d}"
            out.append({
                "doc_id": fable_id,
                "text": row["fable"],
                "moral_id": moral_ids[moral_text],
                "source_idx": row["idx"],
                "source_chunk": row["chunk"],
                "prompt_hash": row["prompt_hash"],
            })
    return out


def build_qrels_moral_to_fable(fables_corpus: list[dict]) -> list[dict]:
    return [
        {"query_id": f["moral_id"], "doc_id": f["doc_id"], "relevance": 1}
        for f in fables_corpus
    ]


def build_qrels_fable_to_moral(fables_corpus: list[dict]) -> list[dict]:
    return [
        {"query_id": f["doc_id"], "doc_id": f["moral_id"], "relevance": 1}
        for f in fables_corpus
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/experiments/test_build_tf1_corpus.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/11_tf1_diagnostic/build_tf1_corpus.py tests/experiments/test_build_tf1_corpus.py
git commit -m "feat(tf1_synthetic): add corpus + qrels assemblers for build script"
```

---

### Task 4: Wire build_tf1_corpus.main + end-to-end integration test

**Files:**
- Modify: `experiments/11_tf1_diagnostic/build_tf1_corpus.py`
- Modify: `tests/experiments/test_build_tf1_corpus.py`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/experiments/test_build_tf1_corpus.py`:

```python
run_build = build.run_build


def test_run_build_writes_all_four_files(tmp_path):
    samples = [
        {"idx": 0, "chunk": 0, "prompt_hash": "h0", "moral": "A", "fable": "fa0"},
        {"idx": 1, "chunk": 0, "prompt_hash": "h1", "moral": "A", "fable": "fa1"},
        {"idx": 2, "chunk": 0, "prompt_hash": "h2", "moral": "A", "fable": "fa2"},
        {"idx": 3, "chunk": 0, "prompt_hash": "h3", "moral": "B", "fable": "fb0"},
        {"idx": 4, "chunk": 0, "prompt_hash": "h4", "moral": "B", "fable": "fb1"},
        {"idx": 5, "chunk": 0, "prompt_hash": "h5", "moral": "B", "fable": "fb2"},
    ]
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text("\n".join(json.dumps(r) for r in samples))

    out_dir = tmp_path / "out"
    run_build(samples_path=samples_path, n=2, seed=42, out_dir=out_dir, expected_unique_morals=2)

    processed = out_dir / "processed"
    morals = json.loads((processed / "morals_corpus.json").read_text())
    fables = json.loads((processed / "fables_corpus.json").read_text())
    qmf = json.loads((processed / "qrels_moral_to_fable.json").read_text())
    qfm = json.loads((processed / "qrels_fable_to_moral.json").read_text())

    assert len(morals) == 2
    assert len(fables) == 4
    assert len(qmf) == 4 and len(qfm) == 4
    assert (out_dir / "README.md").exists()

    # Cross-file consistency
    fable_ids = {f["doc_id"] for f in fables}
    assert {q["doc_id"] for q in qmf} == fable_ids
    assert {q["query_id"] for q in qfm} == fable_ids


def test_run_build_fails_on_wrong_unique_moral_count(tmp_path):
    samples = [{"idx": i, "chunk": 0, "prompt_hash": str(i), "moral": "Only", "fable": "f"}
               for i in range(5)]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in samples))
    with pytest.raises(AssertionError, match="expected 99"):
        run_build(samples_path=p, n=2, seed=42, out_dir=tmp_path / "o", expected_unique_morals=99)


def test_run_build_fails_on_duplicate_prompt_hash(tmp_path):
    samples = [{"idx": i, "chunk": 0, "prompt_hash": "DUP", "moral": "M", "fable": "f"}
               for i in range(5)]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in samples))
    with pytest.raises(AssertionError, match="duplicate prompt_hash"):
        run_build(samples_path=p, n=2, seed=42, out_dir=tmp_path / "o", expected_unique_morals=1)
```

- [ ] **Step 2: Run integration test to verify it fails**

```bash
uv run pytest tests/experiments/test_build_tf1_corpus.py::test_run_build_writes_all_four_files -v
```

Expected: AttributeError (run_build not defined).

- [ ] **Step 3: Implement run_build + main**

Append to `experiments/11_tf1_diagnostic/build_tf1_corpus.py`:

```python
def _latest_samples_path() -> Path:
    runs = sorted(DEFAULT_RUNS_DIR.glob("*/samples.jsonl"))
    if not runs:
        raise FileNotFoundError(
            f"No samples.jsonl found under {DEFAULT_RUNS_DIR}. "
            "Run check_iou.py first."
        )
    return runs[-1]


def _read_samples(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _write_readme(out_dir: Path, n: int, seed: int, source: Path, n_morals: int) -> None:
    readme = f"""# TF1-EN-3M synthetic — MORABLES-shaped derivative

Source: https://huggingface.co/datasets/klusai/ds-tf1-en-3m (MIT)
Paper: https://arxiv.org/abs/2504.20605

Built from {source.relative_to(ROOT) if source.is_relative_to(ROOT) else source}
via experiments/11_tf1_diagnostic/check_iou.py (--n 50000 --chunks 10).

## Build commands

./run.sh experiments/11_tf1_diagnostic/build_tf1_corpus.py --n <N> --seed <S>
./run.sh experiments/11_tf1_diagnostic/cluster_tf1_morals.py --threshold <T>

## Snapshot (this build)

- N per moral: {n}
- Total morals: {n_morals}
- Total fables: {n * n_morals}
- Seed: {seed}
- Built: {datetime.now().isoformat(timespec='seconds')}

See experiments/11_tf1_diagnostic/REPORT.md for the analysis that motivated this derivative.
"""
    (out_dir / "README.md").write_text(readme)


def run_build(
    samples_path: Path,
    n: int,
    seed: int,
    out_dir: Path,
    expected_unique_morals: int = 100,
) -> dict:
    rows = _read_samples(samples_path)
    unique_morals = first_seen_order(rows)
    assert len(unique_morals) == expected_unique_morals, (
        f"expected {expected_unique_morals} unique morals, got {len(unique_morals)}"
    )

    grouped = group_by_moral(rows)
    moral_ids = assign_moral_ids(unique_morals)
    sampled = sample_n_per_moral(grouped, n=n, seed=seed)

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
    _write_readme(out_dir, n=n, seed=seed, source=samples_path, n_morals=len(unique_morals))

    return {
        "n_morals": len(unique_morals),
        "n_fables": len(fables_corpus),
        "out_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--expected-unique-morals", type=int, default=100,
        help="invariant from the diagnostic; set lower for testing on small samples",
    )
    args = parser.parse_args()

    samples_path = args.source or _latest_samples_path()
    print(f"Reading samples from {samples_path}")
    result = run_build(
        samples_path=samples_path,
        n=args.n,
        seed=args.seed,
        out_dir=args.out,
        expected_unique_morals=args.expected_unique_morals,
    )
    print(f"Wrote {result['n_fables']} fables across {result['n_morals']} morals to {result['out_dir']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
uv run pytest tests/experiments/test_build_tf1_corpus.py -v
```

Expected: all 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/11_tf1_diagnostic/build_tf1_corpus.py tests/experiments/test_build_tf1_corpus.py
git commit -m "feat(tf1_synthetic): wire build_tf1_corpus.main with end-to-end run_build"
```

---

### Task 5: Execute build on real data + sanity check

**Files:**
- (none modified; just running and inspecting)

- [ ] **Step 1: Verify the latest samples.jsonl exists**

```bash
ls -la experiments/11_tf1_diagnostic/results/runs/*/samples.jsonl | tail -3
```

Expected: at least one path printed. If not, abort and run `./run.sh experiments/11_tf1_diagnostic/check_iou.py --n 50000 --chunks 10` first.

- [ ] **Step 2: Run the build script with defaults (N=10)**

```bash
./run.sh experiments/11_tf1_diagnostic/build_tf1_corpus.py
```

Expected output (last line):
```
Wrote 1000 fables across 100 morals to /Users/.../data/external/tf1_synthetic
```

- [ ] **Step 3: Inspect the four output files**

```bash
python3 -c "
import json
from pathlib import Path
p = Path('data/external/tf1_synthetic/processed')
m = json.loads((p / 'morals_corpus.json').read_text())
f = json.loads((p / 'fables_corpus.json').read_text())
qmf = json.loads((p / 'qrels_moral_to_fable.json').read_text())
qfm = json.loads((p / 'qrels_fable_to_moral.json').read_text())
print('morals:', len(m), 'first:', m[0])
print('fables:', len(f), 'first doc_id:', f[0]['doc_id'])
print('qrels m->f rows:', len(qmf), 'unique queries:', len({r['query_id'] for r in qmf}))
print('qrels f->m rows:', len(qfm), 'unique queries:', len({r['query_id'] for r in qfm}))
"
```

Expected:
```
morals: 100    first: {'doc_id': 'moral_tf1_000', 'text': 'a kind heart finds friends'}  # or whichever moral is first
fables: 1000    first doc_id: fable_tf1_00000
qrels m→f: 1000  unique queries: 100
qrels f→m: 1000  unique queries: 1000
```

- [ ] **Step 4: Verify README was written**

```bash
cat data/external/tf1_synthetic/README.md
```

Expected: full README with N=10, seed=42, timestamp, etc.

- [ ] **Step 5: Commit the new corpus**

```bash
git add data/external/tf1_synthetic/
git commit -m "data(tf1_synthetic): processed corpus at N=10 (1000 fables, 100 morals)"
```

---

### Task 6: TDD cluster_tf1_morals pure helpers (cluster, classify, canonical)

**Files:**
- Create: `experiments/11_tf1_diagnostic/cluster_tf1_morals.py`
- Create: `tests/experiments/test_cluster_tf1_morals.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/experiments/test_cluster_tf1_morals.py
import json
import sys
import importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PATH = Path(__file__).parent.parent.parent / "experiments" / "11_tf1_diagnostic" / "cluster_tf1_morals.py"
cl = _load("cluster_tf1_morals", _PATH)


def test_agglomerative_clusters_merges_above_threshold():
    # 3 items: 0 and 1 similar, 2 far
    sim = np.array([
        [1.0, 0.95, 0.10],
        [0.95, 1.0, 0.05],
        [0.10, 0.05, 1.0],
    ])
    clusters = cl.agglomerative_clusters(sim, threshold=0.80)
    sets = {frozenset(c) for c in clusters}
    assert sets == {frozenset({0, 1}), frozenset({2})}


def test_agglomerative_clusters_all_singletons_below_threshold():
    sim = np.array([[1.0, 0.5], [0.5, 1.0]])
    clusters = cl.agglomerative_clusters(sim, threshold=0.80)
    sets = {frozenset(c) for c in clusters}
    assert sets == {frozenset({0}), frozenset({1})}


def test_classify_cluster_type_singleton():
    sim = np.eye(3)
    assert cl.classify_cluster_type([0], sim) == "singleton"


def test_classify_cluster_type_exact():
    sim = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert cl.classify_cluster_type([0, 1], sim) == "exact"


def test_classify_cluster_type_near():
    sim = np.array([[1.0, 0.85], [0.85, 1.0]])
    assert cl.classify_cluster_type([0, 1], sim) == "near"


def test_pick_canonical_text_picks_highest_count():
    texts = ["alpha", "beta", "gamma"]
    counts = {"alpha": 10, "beta": 50, "gamma": 5}
    assert cl.pick_canonical_text([0, 1, 2], texts, counts) == "beta"


def test_pick_canonical_text_tie_breaks_by_lowest_index():
    texts = ["alpha", "beta", "gamma"]
    counts = {"alpha": 10, "beta": 10, "gamma": 5}
    assert cl.pick_canonical_text([0, 1, 2], texts, counts) == "alpha"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/experiments/test_cluster_tf1_morals.py -v
```

Expected: ModuleNotFoundError (cluster_tf1_morals.py does not exist yet).

- [ ] **Step 3: Implement the helpers**

Create `experiments/11_tf1_diagnostic/cluster_tf1_morals.py`:

```python
"""
Cluster the 100 unique morals from data/external/tf1_synthetic/processed/ into
semantic groups and emit data/external/tf1_synthetic/clustered/, mirroring
data/clustered/ for MORABLES. See the design spec.
"""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).parent.parent.parent
DEFAULT_IN = ROOT / "data" / "external" / "tf1_synthetic"
DEFAULT_INSPECT_ROOT = ROOT / "experiments" / "11_tf1_diagnostic" / "cluster_inspection"
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
EXACT_THRESHOLD = 0.999


def agglomerative_clusters(sim_matrix: np.ndarray, threshold: float) -> list[list[int]]:
    n = sim_matrix.shape[0]
    if n <= 1:
        return [[0]] if n == 1 else []
    dist = np.clip(1.0 - sim_matrix, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="single")
    labels = fcluster(z, t=1.0 - threshold, criterion="distance")
    groups: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(idx)
    return list(groups.values())


def classify_cluster_type(members: list[int], sim_matrix: np.ndarray) -> str:
    if len(members) == 1:
        return "singleton"
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            if sim_matrix[a, b] >= EXACT_THRESHOLD:
                return "exact"
    return "near"


def pick_canonical_text(members: list[int], moral_texts: list[str], counts: dict[str, int]) -> str:
    def key(i: int) -> tuple[int, int]:
        return (-counts.get(moral_texts[i], 0), i)
    best = sorted(members, key=key)[0]
    return moral_texts[best]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/experiments/test_cluster_tf1_morals.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/11_tf1_diagnostic/cluster_tf1_morals.py tests/experiments/test_cluster_tf1_morals.py
git commit -m "feat(tf1_synthetic): add cluster_tf1_morals helpers (cluster/classify/canonical)"
```

---

### Task 7: TDD cluster_tf1_morals output assemblers (5 file builders)

**Files:**
- Modify: `experiments/11_tf1_diagnostic/cluster_tf1_morals.py`
- Modify: `tests/experiments/test_cluster_tf1_morals.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/experiments/test_cluster_tf1_morals.py`:

```python
def test_build_clustered_outputs_basic():
    morals = [
        {"doc_id": "moral_tf1_000", "text": "honesty is best"},
        {"doc_id": "moral_tf1_001", "text": "honesty wins"},
        {"doc_id": "moral_tf1_002", "text": "greed leads to downfall"},
    ]
    qmf_processed = [
        {"query_id": "moral_tf1_000", "doc_id": "fable_tf1_00000", "relevance": 1},
        {"query_id": "moral_tf1_000", "doc_id": "fable_tf1_00001", "relevance": 1},
        {"query_id": "moral_tf1_001", "doc_id": "fable_tf1_00002", "relevance": 1},
        {"query_id": "moral_tf1_002", "doc_id": "fable_tf1_00003", "relevance": 1},
    ]
    sim = np.array([
        [1.0, 0.92, 0.10],
        [0.92, 1.0, 0.15],
        [0.10, 0.15, 1.0],
    ])
    counts = {"honesty is best": 30, "honesty wins": 50, "greed leads to downfall": 100}

    out = cl.build_clustered_outputs(
        morals=morals, qmf=qmf_processed, sim_matrix=sim,
        counts=counts, threshold=0.80,
    )

    # 2 clusters expected: {honesty is best, honesty wins}, {greed}
    assert len(out["cluster_mapping.json"]) == 2
    canonical_texts = {u["text"] for u in out["morals_unique_corpus.json"]}
    # higher-count "honesty wins" should be canonical for the merged cluster
    assert "honesty wins" in canonical_texts and "honesty is best" not in canonical_texts
    assert "greed leads to downfall" in canonical_texts

    # Each fable maps to exactly one cluster moral in the inverse qrels
    qfm = out["qrels_fable_to_moral_clustered.json"]
    fable_ids = {q["query_id"] for q in qfm}
    assert fable_ids == {f["doc_id"] for f in qmf_processed}
    counts_per_fable = Counter(q["query_id"] for q in qfm)
    assert all(c == 1 for c in counts_per_fable.values())


from collections import Counter  # add to imports at top of test file (or just place here for clarity)
```

(Move the `Counter` import to the top of the file when adding.)

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/experiments/test_cluster_tf1_morals.py::test_build_clustered_outputs_basic -v
```

Expected: AttributeError (build_clustered_outputs not defined).

- [ ] **Step 3: Implement build_clustered_outputs**

Append to `experiments/11_tf1_diagnostic/cluster_tf1_morals.py`:

```python
def build_clustered_outputs(
    morals: list[dict],
    qmf: list[dict],
    sim_matrix: np.ndarray,
    counts: dict[str, int],
    threshold: float,
) -> dict[str, list[dict]]:
    moral_texts = [m["text"] for m in morals]
    moral_ids = [m["doc_id"] for m in morals]

    # fable_id grouped by source moral_id (from input qrels)
    fables_for_moral: dict[str, list[str]] = {}
    for row in qmf:
        fables_for_moral.setdefault(row["query_id"], []).append(row["doc_id"])

    clusters = agglomerative_clusters(sim_matrix, threshold=threshold)

    morals_unique: list[dict] = []
    cluster_mapping: list[dict] = []
    moral_to_cluster: list[dict] = []
    qrels_mtf_clustered: list[dict] = []
    qrels_ftm_clustered: list[dict] = []

    for cluster_idx, members in enumerate(sorted(clusters, key=lambda c: (-len(c), c[0]))):
        cluster_id = f"cluster_{cluster_idx:03d}"
        cluster_type = classify_cluster_type(members, sim_matrix)
        canonical_text = pick_canonical_text(members, moral_texts, counts)
        member_moral_ids = [moral_ids[i] for i in members]
        member_texts = [moral_texts[i] for i in members]
        relevant_fable_ids = [
            fid for mid in member_moral_ids for fid in fables_for_moral.get(mid, [])
        ]

        unique_doc_id = f"moral_tf1_unique_{cluster_idx:04d}"
        morals_unique.append({
            "doc_id": unique_doc_id,
            "text": canonical_text,
            "cluster_id": cluster_id,
            "cluster_type": cluster_type,
            "relevant_fable_ids": relevant_fable_ids,
            "cluster_moral_set": member_texts,
        })
        cluster_mapping.append({
            "cluster_id": cluster_id,
            "type": cluster_type,
            "moral_set": member_texts,
            "fables": relevant_fable_ids,
            "n_morals": len(member_texts),
            "n_fables": len(relevant_fable_ids),
        })
        for mid, text in zip(member_moral_ids, member_texts):
            moral_to_cluster.append({
                "query_id": mid,
                "text": text,
                "cluster_id": cluster_id,
            })
        for fid in relevant_fable_ids:
            qrels_mtf_clustered.append({
                "query_id": unique_doc_id, "doc_id": fid, "relevance": 1,
            })
            qrels_ftm_clustered.append({
                "query_id": fid, "doc_id": unique_doc_id, "relevance": 1,
            })

    return {
        "morals_unique_corpus.json": morals_unique,
        "cluster_mapping.json": cluster_mapping,
        "moral_to_cluster.json": moral_to_cluster,
        "qrels_moral_to_fable_clustered.json": qrels_mtf_clustered,
        "qrels_fable_to_moral_clustered.json": qrels_ftm_clustered,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/experiments/test_cluster_tf1_morals.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/11_tf1_diagnostic/cluster_tf1_morals.py tests/experiments/test_cluster_tf1_morals.py
git commit -m "feat(tf1_synthetic): assemble all 5 clustered output files"
```

---

### Task 8: Wire cluster_tf1_morals.main + integration test (with stubbed embedder)

**Files:**
- Modify: `experiments/11_tf1_diagnostic/cluster_tf1_morals.py`
- Modify: `tests/experiments/test_cluster_tf1_morals.py`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/experiments/test_cluster_tf1_morals.py`:

```python
def test_run_cluster_writes_all_files(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    morals = [
        {"doc_id": "moral_tf1_000", "text": "honesty is best"},
        {"doc_id": "moral_tf1_001", "text": "honesty wins"},
        {"doc_id": "moral_tf1_002", "text": "greed leads to downfall"},
    ]
    (processed / "morals_corpus.json").write_text(json.dumps(morals))
    (processed / "qrels_moral_to_fable.json").write_text(json.dumps([
        {"query_id": "moral_tf1_000", "doc_id": "fable_tf1_00000", "relevance": 1},
        {"query_id": "moral_tf1_001", "doc_id": "fable_tf1_00001", "relevance": 1},
        {"query_id": "moral_tf1_002", "doc_id": "fable_tf1_00002", "relevance": 1},
    ]))

    sim = np.array([
        [1.0, 0.92, 0.10],
        [0.92, 1.0, 0.15],
        [0.10, 0.15, 1.0],
    ])
    counts = {"honesty is best": 1, "honesty wins": 2, "greed leads to downfall": 5}

    out_dir = cl.run_cluster(
        in_dir=tmp_path,
        threshold=0.80,
        inspect_thresholds=[0.80, 0.85],
        sim_matrix=sim,
        counts_override=counts,
        inspection_root=tmp_path / "inspect",
    )

    clustered = tmp_path / "clustered"
    for fname in [
        "morals_unique_corpus.json",
        "cluster_mapping.json",
        "moral_to_cluster.json",
        "qrels_moral_to_fable_clustered.json",
        "qrels_fable_to_moral_clustered.json",
    ]:
        assert (clustered / fname).exists()

    # inspection dumps written
    inspection_subdir = next((tmp_path / "inspect").iterdir())
    assert (inspection_subdir / "clusters_at_0.80.json").exists()
    assert (inspection_subdir / "clusters_at_0.85.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/experiments/test_cluster_tf1_morals.py::test_run_cluster_writes_all_files -v
```

Expected: AttributeError (run_cluster not defined).

- [ ] **Step 3: Implement run_cluster + main**

Append to `experiments/11_tf1_diagnostic/cluster_tf1_morals.py`:

```python
def _load_counts_from_samples(default: dict[str, int] | None = None) -> dict[str, int]:
    """
    Compute the per-moral occurrence count from the latest samples.jsonl.
    Falls back to `default` when given (for testing).
    """
    if default is not None:
        return default
    runs_root = ROOT / "experiments" / "11_tf1_diagnostic" / "results" / "runs"
    samples = sorted(runs_root.glob("*/samples.jsonl"))
    if not samples:
        raise FileNotFoundError(f"No samples.jsonl under {runs_root}")
    counts: Counter = Counter()
    with samples[-1].open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            counts[r["moral"].lower().strip()] += 1
    return counts


def _embed_morals(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return emb @ emb.T


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def run_cluster(
    in_dir: Path,
    threshold: float,
    inspect_thresholds: list[float],
    inspection_root: Path,
    sim_matrix: np.ndarray | None = None,
    counts_override: dict[str, int] | None = None,
    model_name: str = DEFAULT_MODEL,
) -> Path:
    morals = json.loads((in_dir / "processed" / "morals_corpus.json").read_text())
    qmf = json.loads((in_dir / "processed" / "qrels_moral_to_fable.json").read_text())
    moral_texts = [m["text"] for m in morals]

    if sim_matrix is None:
        print(f"Embedding {len(moral_texts)} morals with {model_name} ...")
        sim_matrix = _embed_morals(moral_texts, model_name)

    counts = _load_counts_from_samples(counts_override)

    # Side-by-side inspection dumps
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inspect_dir = inspection_root / timestamp
    inspect_dir.mkdir(parents=True, exist_ok=True)
    for t in inspect_thresholds:
        clusters_at_t = agglomerative_clusters(sim_matrix, threshold=t)
        payload = [
            {
                "size": len(c),
                "members": [moral_texts[i] for i in c],
            }
            for c in sorted(clusters_at_t, key=lambda c: -len(c))
        ]
        _write_json(inspect_dir / f"clusters_at_{t:.2f}.json", payload)

    # Canonical clustered outputs
    out = build_clustered_outputs(
        morals=morals, qmf=qmf, sim_matrix=sim_matrix,
        counts=counts, threshold=threshold,
    )
    clustered_dir = in_dir / "clustered"
    for filename, data in out.items():
        _write_json(clustered_dir / filename, data)

    # Append clustering summary to the existing README
    n_clusters = len(out["cluster_mapping.json"])
    types = Counter(c["type"] for c in out["cluster_mapping.json"])
    readme = (in_dir / "README.md").read_text() if (in_dir / "README.md").exists() else ""
    readme += (
        f"\n\n## Clustering (this run)\n\n"
        f"- Threshold: {threshold}\n"
        f"- Model: {model_name}\n"
        f"- Clusters: {n_clusters}  "
        f"(singleton={types.get('singleton', 0)}, "
        f"near={types.get('near', 0)}, "
        f"exact={types.get('exact', 0)})\n"
        f"- Inspection dumps: experiments/11_tf1_diagnostic/cluster_inspection/{timestamp}/\n"
    )
    (in_dir / "README.md").write_text(readme)

    return clustered_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument(
        "--inspect-thresholds", type=str, default="0.80,0.85,0.90",
        help="comma-separated cosine thresholds to dump alongside the canonical run",
    )
    parser.add_argument("--in", dest="in_dir", type=Path, default=DEFAULT_IN)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    inspect = [float(t) for t in args.inspect_thresholds.split(",")]
    out = run_cluster(
        in_dir=args.in_dir,
        threshold=args.threshold,
        inspect_thresholds=inspect,
        inspection_root=DEFAULT_INSPECT_ROOT,
        model_name=args.model,
    )
    print(f"Wrote clustered outputs to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
uv run pytest tests/experiments/test_cluster_tf1_morals.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/11_tf1_diagnostic/cluster_tf1_morals.py tests/experiments/test_cluster_tf1_morals.py
git commit -m "feat(tf1_synthetic): wire cluster_tf1_morals.main with embedder + inspection dumps"
```

---

### Task 9: Execute cluster on real data + sanity check

**Files:**
- (none modified; just running and inspecting)

- [ ] **Step 1: Run the cluster script with defaults (threshold 0.80)**

```bash
./run.sh experiments/11_tf1_diagnostic/cluster_tf1_morals.py
```

Expected: model downloads on first run (~1.3 GB), then prints `Wrote clustered outputs to .../data/external/tf1_synthetic/clustered`.

- [ ] **Step 2: Inspect the clustered outputs**

```bash
python3 -c "
import json
from pathlib import Path
from collections import Counter
p = Path('data/external/tf1_synthetic/clustered')
cm = json.loads((p / 'cluster_mapping.json').read_text())
mu = json.loads((p / 'morals_unique_corpus.json').read_text())
mc = json.loads((p / 'moral_to_cluster.json').read_text())
qmf = json.loads((p / 'qrels_moral_to_fable_clustered.json').read_text())
qfm = json.loads((p / 'qrels_fable_to_moral_clustered.json').read_text())
print(f'Clusters: {len(cm)}')
types = Counter(c['type'] for c in cm)
print(f'  by type: {dict(types)}')
print(f'Unique-moral queries: {len(mu)}')
print(f'moral→cluster rows: {len(mc)} (should be 100)')
print(f'qrels m→f rows: {len(qmf)} (should be 1000)')
print(f'qrels f→m rows: {len(qfm)} (should be 1000, 1 per fable)')
print()
print('Top 5 clusters by size:')
for c in sorted(cm, key=lambda x: -x['n_morals'])[:5]:
    print(f\"  {c['cluster_id']} ({c['type']}, {c['n_morals']} morals, {c['n_fables']} fables)\")
    for m in c['moral_set']:
        print(f'    - {m}')
"
```

Expected: K clusters where K probably between 30 and 70 (depends on threshold). Top clusters should show obvious paraphrase groups (honesty / truth / gratitude / etc.).

- [ ] **Step 3: Inspect the side-by-side threshold dumps**

```bash
ls experiments/11_tf1_diagnostic/cluster_inspection/ | tail -1 | xargs -I {} ls experiments/11_tf1_diagnostic/cluster_inspection/{}
```

Expected: `clusters_at_0.80.json  clusters_at_0.85.json  clusters_at_0.90.json`.

```bash
python3 -c "
import json
from pathlib import Path
latest = sorted(Path('experiments/11_tf1_diagnostic/cluster_inspection').iterdir())[-1]
for f in sorted(latest.glob('*.json')):
    data = json.loads(f.read_text())
    n_merged = sum(1 for c in data if c['size'] > 1)
    print(f'{f.name}: {len(data)} clusters, {n_merged} non-singleton')
"
```

Expected: threshold 0.80 should produce the fewest clusters (most merging); 0.90 the most. If 0.80 merges too aggressively (e.g., < 20 clusters), flag this for human re-pick — but at this stage just print and move on.

- [ ] **Step 4: Verify README was extended with clustering snapshot**

```bash
cat data/external/tf1_synthetic/README.md | tail -20
```

Expected: a "Clustering (this run)" section at the bottom with threshold, model, cluster counts.

- [ ] **Step 5: Commit the clustered corpus**

```bash
git add data/external/tf1_synthetic/clustered/ data/external/tf1_synthetic/README.md
git commit -m "data(tf1_synthetic): clustered corpus at threshold=0.80"
```

---

### Task 10: Verify the full pipeline reproducibly + final sanity

**Files:**
- (none modified)

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/experiments/ -v
```

Expected: 22 tests PASS (13 build + 9 cluster).

- [ ] **Step 2: Verify branch state**

```bash
git status
git log --oneline -8
```

Expected: clean tree; commits in order matching the task sequence (1 chore, multiple feat/data commits).

- [ ] **Step 3: Sanity-confirm the layout is complete**

```bash
find data/external/tf1_synthetic -type f | sort
```

Expected:
```
data/external/tf1_synthetic/README.md
data/external/tf1_synthetic/clustered/cluster_mapping.json
data/external/tf1_synthetic/clustered/moral_to_cluster.json
data/external/tf1_synthetic/clustered/morals_unique_corpus.json
data/external/tf1_synthetic/clustered/qrels_fable_to_moral_clustered.json
data/external/tf1_synthetic/clustered/qrels_moral_to_fable_clustered.json
data/external/tf1_synthetic/processed/fables_corpus.json
data/external/tf1_synthetic/processed/morals_corpus.json
data/external/tf1_synthetic/processed/qrels_fable_to_moral.json
data/external/tf1_synthetic/processed/qrels_moral_to_fable.json
```
