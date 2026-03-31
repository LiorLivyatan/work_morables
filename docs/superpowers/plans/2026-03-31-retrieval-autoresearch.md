# Retrieval Autoresearch Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous experimentation loop (Experiment 08) where Claude Code iteratively edits a single retrieval pipeline file, measures MRR on the fixed 709-fable validation set, and keeps only improvements — starting from the 0.210 MRR embedding baseline.

**Architecture:** Claude Code is the agent — it reads `program.md` + `results.tsv` + `retrieval_pipeline.py`, edits the pipeline, then calls `runner.py` which handles git commit, subprocess execution, MRR parsing, keep/discard logic, and TSV logging. This clean split means Claude never touches git or file I/O directly.

**Tech Stack:** Python 3.11+, sentence-transformers, numpy, sklearn (cosine_similarity), rank_bm25 (optional, for sparse fusion), pytest

---

## File Map

| File | Role | Created/Modified |
|---|---|---|
| `experiments/08_autoresearch/retrieval_pipeline.py` | **Agent-editable** — model, instructions, chunking, fusion, reranking | Create |
| `experiments/08_autoresearch/runner.py` | Read-only orchestrator — git, subprocess, TSV | Create |
| `experiments/08_autoresearch/program.md` | Agent instructions (for Claude Code) | Create |
| `experiments/08_autoresearch/results.tsv` | Auto-generated experiment log | Auto-created by runner |
| `experiments/08_autoresearch/tests/test_runner.py` | Unit tests for runner utilities | Create |
| `experiments/08_autoresearch/results/embedding_cache/` | Per-experiment embedding cache | Auto-created |

**Read-only (never modified):**
- `lib/retrieval_utils.py` — `compute_metrics()`
- `lib/embedding_cache.py` — `encode_with_cache()`
- `lib/data.py` — `load_moral_to_fable_retrieval_data()`

---

## Task 1: Write failing tests for runner.py utilities

**Files:**
- Create: `experiments/08_autoresearch/tests/__init__.py`
- Create: `experiments/08_autoresearch/tests/test_runner.py`

- [ ] **Step 1: Create the tests directory and empty init**

```bash
mkdir -p experiments/08_autoresearch/tests
touch experiments/08_autoresearch/tests/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `experiments/08_autoresearch/tests/test_runner.py`:

```python
"""Tests for runner.py utility functions.

Run from repo root:
    pytest experiments/08_autoresearch/tests/test_runner.py -v
"""
import sys
from pathlib import Path

# runner.py lives one level up from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))

import runner  # noqa: E402  (not yet implemented — tests will fail)


# ── parse_metric ────────────────────────────────────────────────────────────

def test_parse_mrr_basic():
    stdout = "mrr: 0.2100\nr@1: 0.1400\n"
    assert runner.parse_metric(stdout, "mrr") == 0.2100


def test_parse_mrr_missing_returns_none():
    assert runner.parse_metric("nothing here", "mrr") is None


def test_parse_metric_r1_and_r5():
    stdout = "mrr: 0.2100\nr@1: 0.1400\nr@5: 0.3640\nr@10: 0.4200\n"
    assert runner.parse_metric(stdout, "r@1") == 0.1400
    assert runner.parse_metric(stdout, "r@5") == 0.3640
    assert runner.parse_metric(stdout, "r@10") == 0.4200


def test_parse_metric_with_extra_noise():
    stdout = "encoding 709 texts...\nmrr: 0.2456\nsome other output\n"
    assert runner.parse_metric(stdout, "mrr") == 0.2456


# ── get_best_mrr ─────────────────────────────────────────────────────────────

def test_get_best_mrr_no_file_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_TSV", tmp_path / "results.tsv")
    assert runner.get_best_mrr() == 0.0


def test_get_best_mrr_only_considers_keep_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_TSV", tmp_path / "results.tsv")
    runner.append_result("abc1234", 0.2100, 0.14, 0.36, 0.42, "baseline", "baseline")
    runner.append_result("def5678", 0.1900, 0.12, 0.32, 0.38, "discard", "worse attempt")
    assert runner.get_best_mrr() == 0.2100  # discard row ignored


def test_get_best_mrr_returns_max_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_TSV", tmp_path / "results.tsv")
    runner.append_result("abc1234", 0.2100, 0.14, 0.36, 0.42, "baseline", "baseline")
    runner.append_result("def5678", 0.2300, 0.16, 0.38, 0.44, "keep", "better model")
    runner.append_result("ghi9012", 0.2250, 0.15, 0.37, 0.43, "discard", "not better")
    assert runner.get_best_mrr() == 0.2300


# ── append_result ────────────────────────────────────────────────────────────

def test_append_result_creates_file_with_header(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_TSV", tmp_path / "results.tsv")
    runner.append_result("abc1234", 0.2100, 0.1400, 0.3640, 0.4200, "baseline", "Linq baseline")
    content = (tmp_path / "results.tsv").read_text()
    assert "commit" in content        # header written
    assert "abc1234" in content
    assert "0.2100" in content
    assert "baseline" in content
    assert "Linq baseline" in content


def test_append_result_no_duplicate_header(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_TSV", tmp_path / "results.tsv")
    runner.append_result("abc1234", 0.2100, 0.14, 0.36, 0.42, "baseline", "first")
    runner.append_result("def5678", 0.2300, 0.16, 0.38, 0.44, "keep", "second")
    lines = (tmp_path / "results.tsv").read_text().strip().split("\n")
    header_lines = [l for l in lines if l.startswith("commit")]
    assert len(header_lines) == 1     # exactly one header


def test_append_result_tab_separated(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_TSV", tmp_path / "results.tsv")
    runner.append_result("abc1234", 0.2100, 0.14, 0.36, 0.42, "baseline", "test")
    lines = (tmp_path / "results.tsv").read_text().strip().split("\n")
    assert len(lines[1].split("\t")) == 7  # 7 tab-separated columns
```

- [ ] **Step 3: Run tests to confirm they fail (runner.py doesn't exist yet)**

```bash
cd /Users/liorlivyatan/LocalProjects/Thesis/work_morables
pytest experiments/08_autoresearch/tests/test_runner.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'runner'`

---

## Task 2: Implement runner.py and make tests pass

**Files:**
- Create: `experiments/08_autoresearch/runner.py`

- [ ] **Step 1: Create runner.py**

Create `experiments/08_autoresearch/runner.py`:

```python
#!/usr/bin/env python3
"""
runner.py — Retrieval autoresearch loop orchestrator.
READ-ONLY: Never modified by the agent.

Called by Claude Code after editing retrieval_pipeline.py:
    python runner.py "description of this experiment"
    python runner.py --baseline    (first run: logs without committing)

Flow:
  1. Commit retrieval_pipeline.py with the description
  2. Run it (10-min timeout)
  3. Parse MRR from stdout
  4. MRR > best → keep commit, record "keep"
     MRR ≤ best → git reset, record "discard"
     crash/timeout → git reset, record "crash"
  5. Append row to results.tsv
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_TSV = EXPERIMENT_DIR / "results.tsv"
PIPELINE_FILE = EXPERIMENT_DIR / "retrieval_pipeline.py"
REPO_ROOT = EXPERIMENT_DIR.parent.parent
TIMEOUT_SECONDS = 600
TSV_HEADER = ["commit", "MRR", "R@1", "R@5", "R@10", "status", "description"]


# ── Utility functions (unit-tested) ─────────────────────────────────────────

def parse_metric(stdout: str, key: str) -> float | None:
    """Parse 'key: 0.XXXX' from stdout. Returns None if not found."""
    match = re.search(rf"^{re.escape(key)}:\s*([0-9.]+)", stdout, re.MULTILINE)
    return float(match.group(1)) if match else None


def get_best_mrr() -> float:
    """Return highest MRR from kept+baseline rows in results.tsv. 0.0 if no file."""
    if not RESULTS_TSV.exists():
        return 0.0
    with open(RESULTS_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        mrrs = [
            float(row["MRR"])
            for row in reader
            if row.get("status") in ("keep", "baseline")
        ]
    return max(mrrs, default=0.0)


def append_result(
    commit: str, mrr: float, r1: float, r5: float, r10: float,
    status: str, description: str,
) -> None:
    """Append one row to results.tsv, writing header on first call."""
    write_header = not RESULTS_TSV.exists()
    with open(RESULTS_TSV, "a") as f:
        if write_header:
            f.write("\t".join(TSV_HEADER) + "\n")
        f.write(
            f"{commit}\t{mrr:.4f}\t{r1:.4f}\t{r5:.4f}\t{r10:.4f}"
            f"\t{status}\t{description}\n"
        )


# ── Git helpers ──────────────────────────────────────────────────────────────

def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )


def git_commit(description: str) -> str:
    _git("add", str(PIPELINE_FILE))
    _git("commit", "-m", f"exp08: {description}")
    return _git("rev-parse", "--short", "HEAD").stdout.strip()


def git_reset() -> None:
    _git("reset", "--hard", "HEAD~1")


def get_current_commit() -> str:
    return _git("rev-parse", "--short", "HEAD").stdout.strip()


# ── Pipeline execution ───────────────────────────────────────────────────────

def run_pipeline() -> tuple[float, float, float, float]:
    """Run retrieval_pipeline.py. Returns (mrr, r1, r5, r10). Raises on failure."""
    result = subprocess.run(
        [sys.executable, str(PIPELINE_FILE)],
        capture_output=True, text=True,
        timeout=TIMEOUT_SECONDS,
        cwd=EXPERIMENT_DIR,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline exited {result.returncode}:\n{result.stderr[-2000:]}")
    stdout = result.stdout
    mrr = parse_metric(stdout, "mrr")
    if mrr is None:
        raise RuntimeError(f"'mrr: X.XXXX' not found in stdout:\n{stdout[-1000:]}")
    r1  = parse_metric(stdout, "r@1") or 0.0
    r5  = parse_metric(stdout, "r@5") or 0.0
    r10 = parse_metric(stdout, "r@10") or 0.0
    return mrr, r1, r5, r10


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    baseline_mode = "--baseline" in sys.argv
    description_parts = [a for a in sys.argv[1:] if not a.startswith("--")]
    description = " ".join(description_parts) if description_parts else "no description"

    best_mrr = get_best_mrr()
    print(f"\n{'='*60}")
    print(f"Experiment: {description}")
    print(f"Best so far: {best_mrr:.4f} MRR")
    print(f"{'='*60}\n")

    if baseline_mode:
        commit = get_current_commit()
        print(f"Baseline mode — using current commit: {commit}")
    else:
        commit = git_commit(description)
        print(f"Committed: {commit}")

    try:
        mrr, r1, r5, r10 = run_pipeline()
        print(f"\nResult: MRR={mrr:.4f}  R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}")

        if baseline_mode:
            status = "baseline"
            print("BASELINE — logged without keep/discard decision")
        elif mrr > best_mrr:
            status = "keep"
            print(f"KEEP  (+{mrr - best_mrr:.4f} MRR)")
        else:
            status = "discard"
            git_reset()
            print(f"DISCARD  ({mrr:.4f} ≤ {best_mrr:.4f})")

    except (subprocess.TimeoutExpired, RuntimeError) as e:
        mrr = r1 = r5 = r10 = 0.0
        status = "crash"
        if not baseline_mode:
            git_reset()
        print(f"CRASH: {e}")

    append_result(commit, mrr, r1, r5, r10, status, description)
    print(f"\nLogged → results.tsv  [{status}]  {commit}  MRR={mrr:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests — expect all to pass**

```bash
cd /Users/liorlivyatan/LocalProjects/Thesis/work_morables
pytest experiments/08_autoresearch/tests/test_runner.py -v
```

Expected output:
```
tests/test_runner.py::test_parse_mrr_basic PASSED
tests/test_runner.py::test_parse_mrr_missing_returns_none PASSED
tests/test_runner.py::test_parse_metric_r1_and_r5 PASSED
tests/test_runner.py::test_parse_metric_with_extra_noise PASSED
tests/test_runner.py::test_get_best_mrr_no_file_returns_zero PASSED
tests/test_runner.py::test_get_best_mrr_only_considers_keep_rows PASSED
tests/test_runner.py::test_get_best_mrr_returns_max_kept PASSED
tests/test_runner.py::test_append_result_creates_file_with_header PASSED
tests/test_runner.py::test_append_result_no_duplicate_header PASSED
tests/test_runner.py::test_append_result_tab_separated PASSED

10 passed in 0.XXs
```

- [ ] **Step 3: Commit**

```bash
git add experiments/08_autoresearch/runner.py experiments/08_autoresearch/tests/
git commit -m "Add runner.py and passing tests for Exp 08 autoresearch loop"
```

---

## Task 3: Create baseline retrieval_pipeline.py

**Files:**
- Create: `experiments/08_autoresearch/retrieval_pipeline.py`

- [ ] **Step 1: Create the pipeline**

Create `experiments/08_autoresearch/retrieval_pipeline.py`:

```python
"""
retrieval_pipeline.py — Experiment 08: Retrieval Autoresearch
════════════════════════════════════════════════════════════════
AGENT: You may modify ANYTHING in this file.

Contract (do not change):
  • run_pipeline() returns a dict with at minimum key "MRR"
  • Running as __main__ prints:
        mrr: 0.XXXX
        r@1: 0.XXXX
        r@5: 0.XXXX
        r@10: 0.XXXX
    (each on its own line — parsed by runner.py)

Available from lib/ (do not modify lib/):
  from retrieval_utils import compute_metrics
  from embedding_cache  import encode_with_cache
  from data             import load_moral_to_fable_retrieval_data
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))

from sentence_transformers import SentenceTransformer  # noqa: E402
from data import load_moral_to_fable_retrieval_data    # noqa: E402
from embedding_cache import encode_with_cache          # noqa: E402
from retrieval_utils import compute_metrics            # noqa: E402

# ── Configuration (agent modifies these) ─────────────────────────────────────

MODEL_ID = "Linq-AI-Research/Linq-Embed-Mistral"
QUERY_INSTRUCTION = "Given a text, retrieve the most relevant passage that answers the query"
CORPUS_INSTRUCTION = None   # None = no instruction prefix for corpus texts

CHUNKING = "full"           # "full" | "sentences" | "sliding_N_stride_S"
CHUNK_AGG = "max"           # "max" | "mean"  (used only when CHUNKING != "full")
SPARSE_WEIGHT = 0.0         # 0.0 = pure dense; blend BM25 scores when > 0
RERANKER_ID = None          # None | cross-encoder model id/path
RERANK_TOP_K = 50           # candidate pool size fed to reranker

CACHE_DIR = Path(__file__).parent / "results" / "embedding_cache"

# ── Query rewriting (agent modifies this function) ────────────────────────────

def rewrite_query(moral: str) -> str:
    """
    Optional query rewriting. Return moral unchanged to skip.
    Examples the agent may try:
      - f"This moral means: {moral}"
      - f"Fable lesson: {moral}"
      - moral.lower().strip(".")
    """
    return moral


# ── Pipeline (agent may extend run_pipeline) ─────────────────────────────────

def run_pipeline() -> dict:
    """
    Load data, encode, retrieve, evaluate. Returns metrics dict.
    Must include key 'MRR'.
    """
    fable_texts, moral_texts, ground_truth = load_moral_to_fable_retrieval_data()

    model = SentenceTransformer(MODEL_ID, device="mps")

    query_texts = [rewrite_query(m) for m in moral_texts]

    query_embs = encode_with_cache(
        model, query_texts, MODEL_ID, CACHE_DIR,
        query_instruction=QUERY_INSTRUCTION,
        label="morals (queries)",
    )
    corpus_embs = encode_with_cache(
        model, fable_texts, MODEL_ID, CACHE_DIR,
        query_instruction=CORPUS_INSTRUCTION,
        label="fables (corpus)",
    )

    metrics = compute_metrics(query_embs, corpus_embs, ground_truth)
    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    metrics = run_pipeline()
    print(f"mrr: {metrics['MRR']:.4f}")
    print(f"r@1: {metrics['Recall@1']:.4f}")
    print(f"r@5: {metrics['Recall@5']:.4f}")
    print(f"r@10: {metrics['Recall@10']:.4f}")
```

- [ ] **Step 2: Run it standalone to verify baseline output format**

```bash
cd /Users/liorlivyatan/LocalProjects/Thesis/work_morables
python experiments/08_autoresearch/retrieval_pipeline.py
```

Expected output (last 4 lines):
```
mrr: 0.2100
r@1: 0.1400
r@5: 0.3640
r@10: 0.4480
```

(Exact values may differ slightly from the cached Exp 02 run — that's fine.
The key is that all 4 lines appear in `key: 0.XXXX` format.)

- [ ] **Step 3: Commit**

```bash
git add experiments/08_autoresearch/retrieval_pipeline.py
git commit -m "Add baseline retrieval_pipeline.py for Exp 08 autoresearch"
```

---

## Task 4: Create program.md — agent instructions

**Files:**
- Create: `experiments/08_autoresearch/program.md`

- [ ] **Step 1: Create program.md**

Create `experiments/08_autoresearch/program.md`:

```markdown
# Retrieval Autoresearch — Agent Instructions

## Your Task
Improve MRR on the MORABLES moral-to-fable retrieval benchmark.
Starting baseline: ~0.210 MRR (Linq-Embed-Mistral, raw embedding).
Oracle ceiling: 0.893 MRR (fable+moral concatenation).

## The Loop

Each iteration:
1. Read `results.tsv` to see what's been tried and the current best MRR
2. Read `retrieval_pipeline.py` to understand the current state
3. Propose ONE focused change to `retrieval_pipeline.py` (edit it with your Edit tool)
4. Run: `python experiments/08_autoresearch/runner.py "description of change"`
5. Read the output — runner.py handles git keep/discard and logs the result
6. Repeat from step 1

You are the agent. runner.py is your tool. Never touch runner.py.

## What You May Modify

Only `experiments/08_autoresearch/retrieval_pipeline.py`.

Inside it, you may:
- Change `MODEL_ID` to any locally available sentence-transformers model
- Change `QUERY_INSTRUCTION` and/or `CORPUS_INSTRUCTION`
- Change `CHUNKING`, `CHUNK_AGG` for chunking strategies
- Set `SPARSE_WEIGHT > 0` to blend BM25 (requires: `pip install rank_bm25` — OK to add)
- Set `RERANKER_ID` to a cross-encoder model id for reranking
- Modify `rewrite_query()` for query reformulation (no API calls)
- Add any logic inside `run_pipeline()` (score normalization, ensembles, etc.)

## What You Must Not Change

- `runner.py` — never touch it
- `lib/` — never touch retrieval_utils.py, embedding_cache.py, data.py
- The `run_pipeline() -> dict` return contract (must include key "MRR")
- The stdout format: `mrr: 0.XXXX`, `r@1: 0.XXXX`, etc. (parsed by runner)
- Do NOT make API calls inside `retrieval_pipeline.py` (local models only)
- Do NOT install packages other than `rank_bm25` (already in environment or simple add)

## Exploration Priority

Work through these axes roughly in order — earlier axes have higher impact/cost ratio:

1. **Query instructions** — Try 5-10 different phrasings. Short, abstract instructions
   often outperform verbose ones (see Exp 05 finding: specificity hurts). Examples:
   - `"Retrieve a fable that teaches this lesson"`
   - `"Find the story behind this moral"`
   - `"moral lesson"` (ultra-short)
   - `""` (empty instruction — try it)
   - `None` (no instruction at all)

2. **Corpus instructions** — Try applying instructions to fables too, not just queries.
   Asymmetric instructions (query has instruction, corpus doesn't, or vice versa) sometimes
   help with instruction-tuned models.

3. **Embedding model swap** — Try these (all available via sentence-transformers):
   - `"intfloat/e5-large-v2"` with instruction prefix `"query: "` / `"passage: "`
   - `"BAAI/bge-large-en-v1.5"` with `"Represent this sentence: "`
   - `"thenlper/gte-large"`
   - `"nomic-ai/nomic-embed-text-v1"` (requires `trust_remote_code=True`)
   - `"intfloat/multilingual-e5-large-instruct"`
   Models are cached by sentence-transformers in `~/.cache/torch/sentence_transformers/`.

4. **Query rewriting** — Reformulate morals to sound more like narrative content.
   No API calls: use templates. Examples:
   - `f"A story about: {moral}"`
   - `f"The lesson '{moral}' can be learned from a fable about"`
   - Add filler words that match fable vocabulary ("once upon a time", "a tale of")

5. **Sparse fusion (BM25)** — Add keyword matching alongside dense retrieval.
   Install: `pip install rank_bm25`
   Then add BM25 scoring to `run_pipeline()` and blend with cosine similarity.
   Try `SPARSE_WEIGHT` values: 0.1, 0.2, 0.3, 0.5.

6. **Reranking** — Use a cross-encoder on the top-50 candidates from dense retrieval.
   Set `RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"` as a starting point.
   This runs slower (~2-3 min extra) but can significantly improve precision.

## Research Heuristics

- **One change at a time**: Isolate the effect. If you change model AND instruction together
  and it improves, you don't know which helped.
- **Combine near-misses**: Two discarded experiments that each got close may combine well.
- **Don't repeat failures**: If an exact config was tried and discarded, don't retry it.
- **Simplest first**: A 1-line instruction change before a full model swap.
- **If stuck (5 consecutive discards)**: Try something radical — different model family,
  corpus-side instruction, or a query rewriting approach you haven't tried.
- **Lower is not always better for instructions**: Verbose instructions can hurt.
  Counter-intuitively, `None` (no instruction) sometimes beats a well-crafted one.

## Reference: What's Been Tried

| Exp | Best config | MRR |
|-----|-------------|-----|
| 02  | Linq-Embed-Mistral + generic instruction | 0.210 |
| 05  | Qwen3-Embedding + task-specific instructions | 0.183 (worse!) |
| 06  | Sentence chunking | 0.151 (worse!) |

Key learnings:
- Task-specific instructions HURT on this dataset (Exp 05)
- Sentence-level chunking HURTS — morals need full fable context (Exp 06)
- Linq-Embed-Mistral outperformed 20+ other models in Exp 02

## Never Stop
Run the loop indefinitely until manually interrupted (Ctrl-C or end of session).
When one session ends, the next session can resume by reading results.tsv.
```

- [ ] **Step 2: Commit**

```bash
git add experiments/08_autoresearch/program.md
git commit -m "Add program.md agent instructions for Exp 08 autoresearch"
```

---

## Task 5: Establish baseline in results.tsv

- [ ] **Step 1: Run runner.py in baseline mode**

This logs the starting point without creating a new commit (the pipeline file is already committed).

```bash
cd /Users/liorlivyatan/LocalProjects/Thesis/work_morables
python experiments/08_autoresearch/runner.py --baseline "Linq-Embed-Mistral generic instruction"
```

Expected output:
```
============================================================
Experiment: Linq-Embed-Mistral generic instruction
Best so far: 0.0000 MRR
============================================================

Baseline mode — using current commit: XXXXXXX
  [cache hit]  morals (queries)  (709 texts, shape (709, 768))
  [cache hit]  fables (corpus)   (709 texts, shape (709, 768))

Result: MRR=0.2100  R@1=0.1400  R@5=0.3640  R@10=0.4480
BASELINE — logged without keep/discard decision

Logged → results.tsv  [baseline]  XXXXXXX  MRR=0.2100
```

- [ ] **Step 2: Verify results.tsv was created correctly**

```bash
cat experiments/08_autoresearch/results.tsv
```

Expected:
```
commit	MRR	R@1	R@5	R@10	status	description
XXXXXXX	0.2100	0.1400	0.3640	0.4480	baseline	Linq-Embed-Mistral generic instruction
```

- [ ] **Step 3: Commit results.tsv and add it to .gitignore note**

`results.tsv` should be committed so the baseline is preserved in git history.

```bash
git add experiments/08_autoresearch/results.tsv
git commit -m "exp08: establish MRR=0.2100 baseline (Linq-Embed-Mistral)"
```

---

## Task 6: Smoke-test the full loop with one experiment

This verifies the full keep/discard cycle works end-to-end before handing off to the autonomous loop.

- [ ] **Step 1: Edit retrieval_pipeline.py to try a shorter instruction**

Edit line with `QUERY_INSTRUCTION` in `experiments/08_autoresearch/retrieval_pipeline.py`:

```python
# Change from:
QUERY_INSTRUCTION = "Given a text, retrieve the most relevant passage that answers the query"
# To:
QUERY_INSTRUCTION = "Retrieve a fable that teaches this lesson"
```

- [ ] **Step 2: Run runner.py and observe keep/discard outcome**

```bash
python experiments/08_autoresearch/runner.py "shorter instruction: retrieve fable that teaches lesson"
```

Expected: runner commits, runs pipeline (~60-90s on MPS with cache hits), prints result and keep/discard decision, appends to results.tsv.

- [ ] **Step 3: Verify results.tsv has the second row**

```bash
cat experiments/08_autoresearch/results.tsv
```

Expected (2 rows after header):
```
commit	MRR	R@1	R@5	R@10	status	description
XXXXXXX	0.2100	0.1400	0.3640	0.4480	baseline	Linq-Embed-Mistral generic instruction
YYYYYYY	0.XXXX	0.XXXX	0.XXXX	0.XXXX	keep/discard	shorter instruction: retrieve fable that teaches lesson
```

- [ ] **Step 4: If the smoke-test discarded, reset retrieval_pipeline.py to baseline state**

Runner already handles git reset on discard. Verify the pipeline is back to baseline:
```bash
grep "QUERY_INSTRUCTION" experiments/08_autoresearch/retrieval_pipeline.py
```

Expected: shows the original instruction text (not the modified one) if discarded.

- [ ] **Step 5: Commit results.tsv update**

```bash
git add experiments/08_autoresearch/results.tsv
git commit -m "exp08: smoke-test complete, loop verified end-to-end"
```

---

## Loop is Ready

After Task 6 passes, the autoresearch loop is operational. To run it:

1. Open a Claude Code session in this repo
2. Tell Claude: *"Run the Exp 08 retrieval autoresearch loop. Read `experiments/08_autoresearch/program.md` for instructions."*
3. Claude reads program.md + results.tsv + retrieval_pipeline.py, proposes experiments, and calls `python experiments/08_autoresearch/runner.py "description"` after each edit

Claude Code will iterate through the exploration axes in program.md, tracking all results in results.tsv. When the session ends, start a new session — Claude re-reads results.tsv to resume exactly where it left off.
