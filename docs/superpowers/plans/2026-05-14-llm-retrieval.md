# LLM Retrieval Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `llm_retrieval/` — an experiment that tests whether frontier LLMs can rank relevant fables from the full 709-fable corpus given a moral, serving as an oracle baseline against fine-tuned bi-encoders.

**Architecture:** Async-within-model execution (asyncio + semaphore per model), sequential across models. Each (model × variant) run saves results incrementally to a per-run CSV; a unified CSV accumulates aggregate metrics across all runs. Agno provides a unified multi-provider LLM client with Pydantic-enforced structured output.

**Tech Stack:** Python 3.13, `agno`, `openai`, `anthropic`, `google-generativeai`, `pydantic`, `asyncio`, `pyyaml`, `csv`

---

## File Map

| File | Responsibility |
|------|---------------|
| `llm_retrieval/lib/corpus.py` | Build the `[fable_id] text` prompt block from the fables corpus |
| `llm_retrieval/lib/prompt.py` | Render (system, user) prompt pair from a variant config + inputs |
| `llm_retrieval/lib/eval.py` | Per-query metrics (RR, R@k) and aggregate metrics (MRR, R@1/5/10, NDCG@10) |
| `llm_retrieval/lib/results.py` | Incremental per-run CSV append + unified CSV summary merge |
| `llm_retrieval/lib/providers.py` | Agno model factory: maps provider name → Agno model instance |
| `llm_retrieval/config.yaml` | All models, prompt variants, top_k, test_sizes |
| `llm_retrieval/run.py` | Main entry point: argparse, async orchestration, Telegram notifications |
| `llm_retrieval/README.md` | Usage, results table, design notes |
| `tests/llm_retrieval/test_corpus.py` | Unit tests for corpus block building |
| `tests/llm_retrieval/test_prompt.py` | Unit tests for prompt rendering |
| `tests/llm_retrieval/test_eval.py` | Unit tests for metric computation |
| `tests/llm_retrieval/test_results.py` | Unit tests for CSV writing and unified merge |

---

## Task 1: Project scaffolding and dependencies

**Files:**
- Create: `llm_retrieval/__init__.py`
- Create: `llm_retrieval/lib/__init__.py`
- Create: `llm_retrieval/results/runs/.gitkeep`
- Create: `tests/llm_retrieval/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `agno` and provider SDK deps to `pyproject.toml`**

Open `pyproject.toml` and add to the `dependencies` list:

```toml
dependencies = [
    # ... existing deps ...
    "agno",
    "openai",
    "anthropic",
]
```

(`google-generativeai` is already present. `openai` and `anthropic` are used by both native calls and Agno internals.)

- [ ] **Step 2: Install the new deps**

```bash
uv sync
```

Expected: resolves without errors. `agno` and `openai` and `anthropic` appear in `uv.lock`.

- [ ] **Step 3: Create directory skeleton**

```bash
mkdir -p llm_retrieval/lib llm_retrieval/results/runs tests/llm_retrieval
touch llm_retrieval/__init__.py llm_retrieval/lib/__init__.py
touch tests/llm_retrieval/__init__.py
touch llm_retrieval/results/runs/.gitkeep
```

- [ ] **Step 4: Commit scaffold**

```bash
git add pyproject.toml uv.lock llm_retrieval/ tests/llm_retrieval/
git commit -m "feat(llm_retrieval): scaffold directory and add agno/openai/anthropic deps"
```

---

## Task 2: `lib/corpus.py` — build corpus block

**Files:**
- Create: `llm_retrieval/lib/corpus.py`
- Create: `tests/llm_retrieval/test_corpus.py`

- [ ] **Step 1: Write the failing test**

Create `tests/llm_retrieval/test_corpus.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_retrieval.lib.corpus import build_corpus_block


def test_format_single_fable():
    fables = [{"doc_id": "fable_0001", "text": "A fox lost its tail."}]
    block = build_corpus_block(fables)
    assert block == "[fable_0001] A fox lost its tail."


def test_format_multiple_fables():
    fables = [
        {"doc_id": "fable_0001", "text": "A fox."},
        {"doc_id": "fable_0002", "text": "A crow."},
    ]
    block = build_corpus_block(fables)
    lines = block.split("\n\n")
    assert len(lines) == 2
    assert lines[0] == "[fable_0001] A fox."
    assert lines[1] == "[fable_0002] A crow."


def test_empty_corpus():
    assert build_corpus_block([]) == ""


def test_preserves_order():
    fables = [{"doc_id": f"fable_{i:04d}", "text": f"Story {i}."} for i in range(5)]
    block = build_corpus_block(fables)
    lines = block.split("\n\n")
    for i, line in enumerate(lines):
        assert line.startswith(f"[fable_{i:04d}]")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/llm_retrieval/test_corpus.py -v
```

Expected: `ImportError: cannot import name 'build_corpus_block'`

- [ ] **Step 3: Implement `llm_retrieval/lib/corpus.py`**

```python
def build_corpus_block(fables: list[dict]) -> str:
    if not fables:
        return ""
    return "\n\n".join(f"[{f['doc_id']}] {f['text']}" for f in fables)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/llm_retrieval/test_corpus.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add llm_retrieval/lib/corpus.py tests/llm_retrieval/test_corpus.py
git commit -m "feat(llm_retrieval): corpus block builder"
```

---

## Task 3: `lib/prompt.py` — render prompt variants

**Files:**
- Create: `llm_retrieval/lib/prompt.py`
- Create: `tests/llm_retrieval/test_prompt.py`

- [ ] **Step 1: Write the failing test**

Create `tests/llm_retrieval/test_prompt.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_retrieval.lib.prompt import render_prompt


VARIANT = {
    "label": "minimal",
    "system": "You are a retrieval system.",
    "user_template": "Moral: {moral}\n\nCorpus:\n{corpus}\n\nReturn the {top_k} most relevant fable IDs.",
}


def test_renders_system_unchanged():
    system, _ = render_prompt("Be kind.", "corpus text", VARIANT, top_k=10)
    assert system == "You are a retrieval system."


def test_substitutes_moral():
    _, user = render_prompt("Be kind.", "corpus text", VARIANT, top_k=10)
    assert "Be kind." in user


def test_substitutes_corpus():
    _, user = render_prompt("Be kind.", "[fable_0001] A story.", VARIANT, top_k=10)
    assert "[fable_0001] A story." in user


def test_substitutes_top_k():
    _, user = render_prompt("Be kind.", "corpus", VARIANT, top_k=5)
    assert "5" in user


def test_missing_placeholder_raises():
    bad_variant = {
        "label": "bad",
        "system": "sys",
        "user_template": "Moral: {moral}",  # missing {corpus} and {top_k}
    }
    import pytest
    with pytest.raises(KeyError):
        render_prompt("Be kind.", "corpus", bad_variant, top_k=10)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/llm_retrieval/test_prompt.py -v
```

Expected: `ImportError: cannot import name 'render_prompt'`

- [ ] **Step 3: Implement `llm_retrieval/lib/prompt.py`**

```python
def render_prompt(
    moral: str,
    corpus_block: str,
    variant: dict,
    top_k: int,
) -> tuple[str, str]:
    system = variant["system"]
    user = variant["user_template"].format(
        moral=moral,
        corpus=corpus_block,
        top_k=top_k,
    )
    return system, user
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/llm_retrieval/test_prompt.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add llm_retrieval/lib/prompt.py tests/llm_retrieval/test_prompt.py
git commit -m "feat(llm_retrieval): prompt variant renderer"
```

---

## Task 4: `lib/eval.py` — per-query and aggregate metrics

**Files:**
- Create: `llm_retrieval/lib/eval.py`
- Create: `tests/llm_retrieval/test_eval.py`

- [ ] **Step 1: Write the failing test**

Create `tests/llm_retrieval/test_eval.py`:

```python
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_retrieval.lib.eval import (
    reciprocal_rank,
    recall_at_k,
    rank_of,
    compute_row_metrics,
    aggregate_metrics,
)


def test_reciprocal_rank_first():
    assert reciprocal_rank(["fable_0001", "fable_0002"], "fable_0001") == pytest.approx(1.0)


def test_reciprocal_rank_second():
    assert reciprocal_rank(["fable_0002", "fable_0001"], "fable_0001") == pytest.approx(0.5)


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(["fable_0002", "fable_0003"], "fable_0001") == pytest.approx(0.0)


def test_recall_at_k_hit():
    assert recall_at_k(["fable_0001", "fable_0002"], "fable_0001", k=1) == 1.0


def test_recall_at_k_miss():
    assert recall_at_k(["fable_0002", "fable_0003"], "fable_0001", k=2) == 0.0


def test_recall_at_k_outside_window():
    ids = ["fable_0002", "fable_0003", "fable_0001"]
    assert recall_at_k(ids, "fable_0001", k=2) == 0.0
    assert recall_at_k(ids, "fable_0001", k=3) == 1.0


def test_rank_of_found():
    assert rank_of(["fable_0001", "fable_0002"], "fable_0001") == 1
    assert rank_of(["fable_0002", "fable_0001"], "fable_0001") == 2


def test_rank_of_not_found():
    assert rank_of(["fable_0002"], "fable_0001") is None


def test_compute_row_metrics_perfect():
    row = compute_row_metrics(
        moral_id="moral_0001",
        moral_text="Be kind.",
        relevant_fable="fable_0001",
        ranked_ids=["fable_0001", "fable_0002"],
        latency_s=0.5,
    )
    assert row["moral_id"] == "moral_0001"
    assert row["relevant_fable"] == "fable_0001"
    assert row["reciprocal_rank"] == pytest.approx(1.0)
    assert row["r_at_1"] == 1.0
    assert row["r_at_5"] == 1.0
    assert row["r_at_10"] == 1.0
    assert row["rank"] == 1
    assert row["latency_s"] == pytest.approx(0.5)


def test_compute_row_metrics_miss():
    row = compute_row_metrics(
        moral_id="moral_0001",
        moral_text="Be kind.",
        relevant_fable="fable_0001",
        ranked_ids=["fable_0002", "fable_0003"],
        latency_s=1.0,
    )
    assert row["reciprocal_rank"] == pytest.approx(0.0)
    assert row["r_at_1"] == 0.0
    assert row["rank"] is None


def test_aggregate_metrics_perfect():
    rows = [
        {"reciprocal_rank": 1.0, "r_at_1": 1.0, "r_at_5": 1.0, "r_at_10": 1.0, "rank": 1},
        {"reciprocal_rank": 1.0, "r_at_1": 1.0, "r_at_5": 1.0, "r_at_10": 1.0, "rank": 1},
    ]
    agg = aggregate_metrics(rows)
    assert agg["MRR@10"] == pytest.approx(1.0)
    assert agg["R@1"] == pytest.approx(1.0)
    assert agg["R@10"] == pytest.approx(1.0)
    assert agg["Mean_Rank"] == pytest.approx(1.0)
    assert agg["n_queries"] == 2


def test_aggregate_metrics_with_miss():
    rows = [
        {"reciprocal_rank": 1.0, "r_at_1": 1.0, "r_at_5": 1.0, "r_at_10": 1.0, "rank": 1},
        {"reciprocal_rank": 0.0, "r_at_1": 0.0, "r_at_5": 0.0, "r_at_10": 0.0, "rank": None},
    ]
    agg = aggregate_metrics(rows)
    assert agg["MRR@10"] == pytest.approx(0.5)
    assert agg["R@1"] == pytest.approx(0.5)
    assert agg["Mean_Rank"] == pytest.approx(1.0)  # only the hit row counts
    assert agg["Median_Rank"] == pytest.approx(1.0)
    assert agg["n_queries"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/llm_retrieval/test_eval.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `llm_retrieval/lib/eval.py`**

```python
import math
import statistics
from typing import Optional


def reciprocal_rank(ranked_ids: list[str], relevant_id: str) -> float:
    try:
        return 1.0 / (ranked_ids.index(relevant_id) + 1)
    except ValueError:
        return 0.0


def recall_at_k(ranked_ids: list[str], relevant_id: str, k: int) -> float:
    return 1.0 if relevant_id in ranked_ids[:k] else 0.0


def rank_of(ranked_ids: list[str], relevant_id: str) -> Optional[int]:
    try:
        return ranked_ids.index(relevant_id) + 1
    except ValueError:
        return None


def ndcg_at_k(ranked_ids: list[str], relevant_id: str, k: int) -> float:
    rank = rank_of(ranked_ids[:k], relevant_id)
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)  # ideal DCG = 1/log2(2) = 1.0


def compute_row_metrics(
    moral_id: str,
    moral_text: str,
    relevant_fable: str,
    ranked_ids: list[str],
    latency_s: float,
) -> dict:
    rr = reciprocal_rank(ranked_ids, relevant_fable)
    return {
        "moral_id": moral_id,
        "moral_text": moral_text,
        "relevant_fable": relevant_fable,
        "ranked_ids": "|".join(ranked_ids),
        "reciprocal_rank": rr,
        "r_at_1": recall_at_k(ranked_ids, relevant_fable, 1),
        "r_at_5": recall_at_k(ranked_ids, relevant_fable, 5),
        "r_at_10": recall_at_k(ranked_ids, relevant_fable, 10),
        "ndcg_at_10": ndcg_at_k(ranked_ids, relevant_fable, 10),
        "rank": rank_of(ranked_ids, relevant_fable),
        "latency_s": round(latency_s, 3),
    }


def aggregate_metrics(rows: list[dict]) -> dict:
    found_ranks = [r["rank"] for r in rows if r["rank"] is not None]
    return {
        "MRR@10": statistics.mean(r["reciprocal_rank"] for r in rows),
        "R@1":    statistics.mean(r["r_at_1"]  for r in rows),
        "R@5":    statistics.mean(r["r_at_5"]  for r in rows),
        "R@10":   statistics.mean(r["r_at_10"] for r in rows),
        "NDCG@10": statistics.mean(r["ndcg_at_10"] for r in rows),
        "Mean_Rank":   statistics.mean(found_ranks)   if found_ranks else None,
        "Median_Rank": statistics.median(found_ranks) if found_ranks else None,
        "n_queries": len(rows),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/llm_retrieval/test_eval.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add llm_retrieval/lib/eval.py tests/llm_retrieval/test_eval.py
git commit -m "feat(llm_retrieval): per-query and aggregate metric computation"
```

---

## Task 5: `lib/results.py` — incremental CSV writing and unified merge

**Files:**
- Create: `llm_retrieval/lib/results.py`
- Create: `tests/llm_retrieval/test_results.py`

- [ ] **Step 1: Write the failing test**

Create `tests/llm_retrieval/test_results.py`:

```python
import csv
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_retrieval.lib.results import (
    get_run_path,
    append_query_row,
    count_completed_rows,
    load_completed_rows,
    append_summary_row,
    RUN_FIELDNAMES,
    UNIFIED_FIELDNAMES,
)


def test_get_run_path_format():
    base = Path("/tmp/results")
    path = get_run_path(base, "GPT-4o", "minimal", "2026-05-14")
    assert path == Path("/tmp/results/runs/2026-05-14_GPT-4o_minimal.csv")


def test_append_and_count(tmp_path):
    path = tmp_path / "run.csv"
    row = {f: "x" for f in RUN_FIELDNAMES}
    append_query_row(path, row)
    append_query_row(path, row)
    assert count_completed_rows(path) == 2


def test_count_nonexistent_file(tmp_path):
    assert count_completed_rows(tmp_path / "missing.csv") == 0


def test_load_completed_rows(tmp_path):
    path = tmp_path / "run.csv"
    row = {f: "val" for f in RUN_FIELDNAMES}
    append_query_row(path, row)
    rows = load_completed_rows(path)
    assert len(rows) == 1
    assert rows[0]["moral_id"] == "val"


def test_load_completed_rows_casts_numerics(tmp_path):
    """CSV round-trip must preserve numeric types — needed for aggregate_metrics after resume."""
    path = tmp_path / "run.csv"
    row = {f: "" for f in RUN_FIELDNAMES}
    row.update({
        "moral_id": "moral_0001",
        "relevant_fable": "fable_0001",
        "ranked_ids": "fable_0001|fable_0002",
        "reciprocal_rank": "1.0",
        "r_at_1": "1.0",
        "r_at_5": "1.0",
        "r_at_10": "1.0",
        "ndcg_at_10": "1.0",
        "rank": "1",
        "latency_s": "0.5",
    })
    append_query_row(path, row)
    rows = load_completed_rows(path)
    assert isinstance(rows[0]["reciprocal_rank"], float)
    assert isinstance(rows[0]["r_at_1"], float)
    assert isinstance(rows[0]["rank"], int)
    assert isinstance(rows[0]["latency_s"], float)


def test_load_completed_rows_null_rank(tmp_path):
    path = tmp_path / "run.csv"
    row = {f: "" for f in RUN_FIELDNAMES}
    row["rank"] = ""
    append_query_row(path, row)
    rows = load_completed_rows(path)
    assert rows[0]["rank"] is None


def test_append_summary_row(tmp_path):
    unified = tmp_path / "unified.csv"
    summary = {f: "x" for f in UNIFIED_FIELDNAMES}
    append_summary_row(unified, summary)
    append_summary_row(unified, summary)
    with open(unified) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/llm_retrieval/test_results.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `llm_retrieval/lib/results.py`**

```python
import csv
from pathlib import Path
from typing import Optional


RUN_FIELDNAMES = [
    "moral_id", "moral_text", "relevant_fable",
    "ranked_ids", "reciprocal_rank",
    "r_at_1", "r_at_5", "r_at_10", "ndcg_at_10",
    "rank", "latency_s",
]

UNIFIED_FIELDNAMES = [
    "run_date", "model_alias", "model_id", "provider", "variant_label",
    "n_queries", "MRR@10", "R@1", "R@5", "R@10", "NDCG@10",
    "Mean_Rank", "Median_Rank", "avg_latency_s",
]

_FLOAT_FIELDS = {"reciprocal_rank", "r_at_1", "r_at_5", "r_at_10", "ndcg_at_10", "latency_s"}
_INT_FIELDS   = {"rank"}


def _cast_row(row: dict) -> dict:
    result = dict(row)
    for field in _FLOAT_FIELDS:
        if field in result and result[field] not in ("", None):
            result[field] = float(result[field])
    for field in _INT_FIELDS:
        if field in result:
            result[field] = int(result[field]) if result[field] not in ("", "None", None) else None
    return result


def get_run_path(results_dir: Path, model_alias: str, variant_label: str, run_date: str) -> Path:
    filename = f"{run_date}_{model_alias}_{variant_label}.csv"
    return results_dir / "runs" / filename


def append_query_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RUN_FIELDNAMES})


def count_completed_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in csv.DictReader(f))


def load_completed_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [_cast_row(row) for row in csv.DictReader(f)]


def append_summary_row(unified_path: Path, summary: dict) -> None:
    unified_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not unified_path.exists()
    with open(unified_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: summary.get(k, "") for k in UNIFIED_FIELDNAMES})
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/llm_retrieval/test_results.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add llm_retrieval/lib/results.py tests/llm_retrieval/test_results.py
git commit -m "feat(llm_retrieval): incremental CSV writing and unified summary merge"
```

---

## Task 6: `lib/providers.py` — Agno model factory

**Files:**
- Create: `llm_retrieval/lib/providers.py`

No unit tests for this module — it wraps external APIs. Verified in the smoke test (Task 9).

- [ ] **Step 1: Create `llm_retrieval/lib/providers.py`**

```python
import os


def make_agno_model(model_cfg: dict):
    provider = model_cfg["provider"]
    model_id = model_cfg["id"]

    if provider == "openai":
        from agno.models.openai import OpenAIChat
        return OpenAIChat(id=model_id)

    if provider == "anthropic":
        from agno.models.anthropic import Claude
        return Claude(id=model_id)

    if provider == "google":
        from agno.models.google import Gemini
        return Gemini(id=model_id)

    if provider == "openrouter":
        from agno.models.openai.like import OpenAILike
        return OpenAILike(
            id=model_id,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    if provider == "together":
        from agno.models.openai.like import OpenAILike
        return OpenAILike(
            id=model_id,
            base_url="https://api.together.xyz/v1",
            api_key=os.environ["TOGETHER_API_KEY"],
        )

    raise ValueError(f"Unknown provider: {provider!r}")
```

- [ ] **Step 2: Verify import works**

```bash
python -c "from llm_retrieval.lib.providers import make_agno_model; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add llm_retrieval/lib/providers.py
git commit -m "feat(llm_retrieval): Agno model factory for openai/anthropic/google/openrouter/together"
```

---

## Task 7: `config.yaml` — full model and variant configuration

**Files:**
- Create: `llm_retrieval/config.yaml`

- [ ] **Step 1: Create `llm_retrieval/config.yaml`**

```yaml
top_k: 10
test_sizes: [50, 100, 200]  # --test N accepts any integer; these are documented suggestions

# ── Prompt variants ──────────────────────────────────────────────────────────
prompt_variants:

  - label: minimal
    system: "You are a retrieval system. Return only the requested JSON."
    user_template: |
      Moral: {moral}

      Corpus:
      {corpus}

      Return a JSON array of the {top_k} fable IDs most relevant to this moral, ranked by relevance. IDs are in the format fable_XXXX.

  - label: detailed
    system: "You are an expert in literature and moral philosophy. Your task is to identify which fables best illustrate a given moral lesson."
    user_template: |
      A fable 'conveys' a moral when the story's lesson, theme, or implicit message aligns with the stated moral.

      Moral: {moral}

      Below is a corpus of fables. Read them and identify which best convey the given moral.

      Corpus:
      {corpus}

      Return a JSON array of the {top_k} fable IDs that best convey this moral, ranked from most to least relevant. IDs are in the format fable_XXXX.

  - label: cot
    system: "You are an expert in literature and moral philosophy."
    user_template: |
      Consider the following moral:

      {moral}

      Below is a corpus of fables. Think step by step:
      1. What is the core teaching of this moral?
      2. What story elements would best convey this teaching?
      3. Which fables in the corpus match these elements most closely?

      Corpus:
      {corpus}

      After reasoning, return a JSON array of the {top_k} fable IDs that best convey this moral, ranked from most to least relevant. IDs are in the format fable_XXXX.

# ── Models ───────────────────────────────────────────────────────────────────
models:

  # OpenAI
  - alias: GPT-4o-mini
    id: gpt-4o-mini
    provider: openai
    concurrency: 20

  - alias: GPT-4o
    id: gpt-4o
    provider: openai
    concurrency: 10

  - alias: GPT-5.5
    id: gpt-5.5
    provider: openai
    concurrency: 5

  # OpenAI OSS (via OpenRouter free tier, 131K ctx — monitor for overflow)
  - alias: GPT-OSS-20B
    id: openai/gpt-oss-20b:free
    provider: openrouter
    concurrency: 10

  - alias: GPT-OSS-120B
    id: openai/gpt-oss-120b:free
    provider: openrouter
    concurrency: 5

  # Anthropic
  - alias: Claude-Haiku
    id: claude-haiku-4-5-20251001
    provider: anthropic
    concurrency: 10

  - alias: Claude-Sonnet
    id: claude-sonnet-4-6
    provider: anthropic
    concurrency: 5

  - alias: Claude-Opus
    id: claude-opus-4-7
    provider: anthropic
    concurrency: 3

  # Google
  - alias: Gemini-2.5-Pro
    id: gemini-2.5-pro
    provider: google
    concurrency: 5

  - alias: Gemini-3-Flash
    id: gemini-3-flash
    provider: google
    concurrency: 10

  - alias: Gemini-3.1-Pro
    id: gemini-3.1-pro
    provider: google
    concurrency: 5

  # OpenRouter — Qwen
  - alias: Qwen3.5-Flash
    id: qwen/qwen3.5-flash
    provider: openrouter
    concurrency: 10

  - alias: Qwen3.6-Plus
    id: qwen/qwen3.6-plus
    provider: openrouter
    concurrency: 5

  - alias: Qwen3.6-27B
    id: qwen/qwen3.6-27b
    provider: openrouter
    concurrency: 5

  # OpenRouter — Meta
  - alias: Llama-4-Scout
    id: meta-llama/Llama-4-Scout-17B-Instruct
    provider: openrouter
    concurrency: 10
```

- [ ] **Step 2: Verify config loads cleanly**

```bash
python -c "
import yaml
from pathlib import Path
cfg = yaml.safe_load((Path('llm_retrieval/config.yaml')).read_text())
print(f'top_k={cfg[\"top_k\"]}')
print(f'variants: {[v[\"label\"] for v in cfg[\"prompt_variants\"]]}')
print(f'models: {[m[\"alias\"] for m in cfg[\"models\"]]}')
"
```

Expected: `top_k=10`, 3 variant labels, 15 model aliases.

- [ ] **Step 3: Commit**

```bash
git add llm_retrieval/config.yaml
git commit -m "feat(llm_retrieval): full model and prompt variant config"
```

---

## Task 8: `run.py` — async main entry point

**Files:**
- Create: `llm_retrieval/run.py`

- [ ] **Step 1: Create `llm_retrieval/run.py`**

```python
"""
llm_retrieval/run.py — LLM oracle baseline for moral-to-fable retrieval.

Given a moral statement and the full 709-fable corpus in context, asks each
configured LLM to rank the top-k most relevant fables. Results are saved
incrementally to per-run CSVs and a unified summary CSV.

Usage
-----
    # Full run, all models, all variants:
    ./run.sh llm_retrieval/run.py

    # Test mode — 100 morals, one model, one variant:
    ./run.sh llm_retrieval/run.py --test 100 --models GPT-4o-mini --variants minimal

    # Skip already-completed runs:
    ./run.sh llm_retrieval/run.py --skip-existing

    # Force re-run (overwrite existing results):
    ./run.sh llm_retrieval/run.py --force
"""
import argparse
import asyncio
import json
import time
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel

EXP_DIR     = Path(__file__).parent
ROOT        = EXP_DIR.parent
RESULTS_DIR = EXP_DIR / "results"
CONFIG_PATH = EXP_DIR / "config.yaml"

FABLES_PATH = ROOT / "data/processed/fables_corpus.json"
MORALS_PATH = ROOT / "data/processed/morals_corpus.json"

import sys
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from llm_retrieval.lib.corpus import build_corpus_block
from llm_retrieval.lib.eval import compute_row_metrics, aggregate_metrics
from llm_retrieval.lib.prompt import render_prompt
from llm_retrieval.lib.providers import make_agno_model
from llm_retrieval.lib.results import (
    get_run_path, append_query_row, count_completed_rows,
    load_completed_rows, append_summary_row,
)


class RankedFables(BaseModel):
    ids: list[str]


def load_queries(n: int | None) -> list[dict]:
    morals = json.loads(MORALS_PATH.read_text())
    if n is not None:
        morals = morals[:n]
    return [
        {
            "moral_id":       m["doc_id"],
            "moral_text":     m["text"],
            "relevant_fable": m["fable_id"],
        }
        for m in morals
        if m.get("fable_id")
    ]


async def run_one_query(
    agent,
    query: dict,
    variant_cfg: dict,
    corpus_block: str,
    top_k: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    _, user = render_prompt(query["moral_text"], corpus_block, variant_cfg, top_k)
    async with semaphore:
        t0 = time.monotonic()
        try:
            response = await agent.arun(user)
            ranked_ids = response.content.ids[:top_k]
        except Exception as exc:
            print(f"  [WARN] {query['moral_id']} failed: {exc}")
            ranked_ids = []
        latency = time.monotonic() - t0
    return compute_row_metrics(
        moral_id=query["moral_id"],
        moral_text=query["moral_text"],
        relevant_fable=query["relevant_fable"],
        ranked_ids=ranked_ids,
        latency_s=latency,
    )


async def run_model_variant(
    model_cfg: dict,
    variant_cfg: dict,
    queries: list[dict],
    corpus_block: str,
    top_k: int,
    run_path: Path,
    run_date: str,
) -> list[dict]:
    from agno.agent import Agent

    already_done = count_completed_rows(run_path)
    queries_to_run = queries[already_done:]

    if not queries_to_run:
        print(f"  ✓ Already complete ({already_done} rows)")
        return load_completed_rows(run_path)

    if already_done:
        print(f"  Resuming from row {already_done}/{len(queries)}")

    model = make_agno_model(model_cfg)
    agent = Agent(
        model=model,
        instructions=variant_cfg["system"],
        response_model=RankedFables,
        add_history_to_messages=False,
        stream=False,
    )
    sem = asyncio.Semaphore(model_cfg.get("concurrency", 5))

    tasks = [
        run_one_query(
            agent=agent,
            query=q,
            variant_cfg=variant_cfg,
            corpus_block=corpus_block,
            top_k=top_k,
            semaphore=sem,
        )
        for q in queries_to_run
    ]

    new_rows = []
    for coro in asyncio.as_completed(tasks):
        row = await coro
        append_query_row(run_path, row)
        new_rows.append(row)

    prior_rows = load_completed_rows(run_path)
    return prior_rows


def main():
    parser = argparse.ArgumentParser(description="LLM oracle retrieval experiment")
    parser.add_argument("--test", type=int, metavar="N", help="Run on first N morals only")
    parser.add_argument("--models", nargs="+", metavar="ALIAS", help="Model aliases to run (default: all)")
    parser.add_argument("--variants", nargs="+", metavar="LABEL", help="Variant labels to run (default: all)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip (model, variant) pairs with a complete run file")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    top_k = cfg["top_k"]
    run_date = date.today().isoformat()

    models = cfg["models"]
    if args.models:
        models = [m for m in models if m["alias"] in args.models]
        missing = set(args.models) - {m["alias"] for m in models}
        if missing:
            print(f"[WARN] Unknown model aliases: {missing}")

    variants = cfg["prompt_variants"]
    if args.variants:
        variants = [v for v in variants if v["label"] in args.variants]

    queries = load_queries(args.test)
    print(f"Loaded {len(queries)} queries")

    fables = json.loads(FABLES_PATH.read_text())
    corpus_block = build_corpus_block(fables)
    print(f"Corpus block: {len(corpus_block):,} chars (~{len(corpus_block)//4:,} tokens)")

    mode_label = f"test-{args.test}" if args.test else "full"
    notify.send(
        f"🔬 llm_retrieval starting\n"
        f"mode: {mode_label} ({len(queries)} morals)\n"
        f"models: {[m['alias'] for m in models]}\n"
        f"variants: {[v['label'] for v in variants]}"
    )

    unified_path = RESULTS_DIR / "unified.csv"

    for model_cfg in models:
        for variant_cfg in variants:
            alias   = model_cfg["alias"]
            label   = variant_cfg["label"]
            run_path = get_run_path(RESULTS_DIR, alias, label, run_date)

            if args.skip_existing and count_completed_rows(run_path) >= len(queries):
                print(f"[SKIP] {alias} × {label} (already complete)")
                continue

            if args.force and run_path.exists():
                run_path.unlink()

            print(f"\n▶ {alias} × {label}")
            t_start = time.monotonic()

            rows = asyncio.run(run_model_variant(
                model_cfg=model_cfg,
                variant_cfg=variant_cfg,
                queries=queries,
                corpus_block=corpus_block,
                top_k=top_k,
                run_path=run_path,
                run_date=run_date,
            ))

            elapsed = time.monotonic() - t_start
            agg = aggregate_metrics(rows)

            summary = {
                "run_date":      run_date,
                "model_alias":   alias,
                "model_id":      model_cfg["id"],
                "provider":      model_cfg["provider"],
                "variant_label": label,
                "n_queries":     agg["n_queries"],
                "MRR@10":        round(agg["MRR@10"], 4),
                "R@1":           round(agg["R@1"],    4),
                "R@5":           round(agg["R@5"],    4),
                "R@10":          round(agg["R@10"],   4),
                "NDCG@10":       round(agg["NDCG@10"], 4),
                "Mean_Rank":     round(agg["Mean_Rank"],   1) if agg["Mean_Rank"]   is not None else "",
                "Median_Rank":   round(agg["Median_Rank"], 1) if agg["Median_Rank"] is not None else "",
                "avg_latency_s": round(elapsed / max(len(rows), 1), 2),
            }
            append_summary_row(unified_path, summary)

            print(f"  MRR@10={summary['MRR@10']}  R@10={summary['R@10']}  ({elapsed:.0f}s)")
            notify.send(
                f"✅ {alias} × {label} done\n"
                f"MRR@10={summary['MRR@10']}  R@10={summary['R@10']}\n"
                f"n={agg['n_queries']}  {elapsed:.0f}s"
            )

    notify.send("🏁 llm_retrieval complete — check results/unified.csv")
    print(f"\nDone. Results in {unified_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses args without errors**

```bash
python llm_retrieval/run.py --help
```

Expected: prints usage with `--test`, `--models`, `--variants`, `--skip-existing`, `--force` options.

- [ ] **Step 3: Commit**

```bash
git add llm_retrieval/run.py
git commit -m "feat(llm_retrieval): async main entry point with incremental saving and Telegram notifications"
```

---

## Task 9: README and full test suite

**Files:**
- Create: `llm_retrieval/README.md`

- [ ] **Step 1: Create `llm_retrieval/README.md`**

```markdown
# LLM Retrieval Experiment

Tests whether frontier LLMs can retrieve the correct fable from a 709-fable corpus given a moral statement — an "oracle baseline" against fine-tuned bi-encoder retrievers.

## How It Works

Each query: the model receives the full corpus (709 fables formatted as `[fable_id] text`) and a moral statement, and must return the 10 most relevant fable IDs as a JSON array.

**Corpus size:** ~120k tokens. Models with 131K context (GPT-OSS-20B/120B) are near capacity — monitor for overflow errors.

## Usage

```bash
# Test mode — smoke test one model, one variant, 100 morals
./run.sh llm_retrieval/run.py --test 100 --models GPT-4o-mini --variants minimal

# Full run — all models, all variants
./run.sh llm_retrieval/run.py

# Skip already-completed runs (safe to re-run after interruption)
./run.sh llm_retrieval/run.py --skip-existing
```

## Required `.env` keys

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
TOGETHER_API_KEY=...
```

## Results

- `results/runs/YYYY-MM-DD_<model>_<variant>.csv` — per-query rows
- `results/unified.csv` — aggregate metrics per (model × variant) run

## Results Table

*(populated after runs)*

| Model | Variant | MRR@10 | R@1 | R@10 | NDCG@10 |
|-------|---------|--------|-----|------|---------|
| — | — | — | — | — | — |
```

- [ ] **Step 2: Run the full test suite**

```bash
python -m pytest tests/llm_retrieval/ -v
```

Expected: all tests pass (corpus, prompt, eval, results — ~27 tests).

- [ ] **Step 3: Commit**

```bash
git add llm_retrieval/README.md
git commit -m "docs(llm_retrieval): README with usage and results table"
```

---

## Task 10: End-to-end smoke test (real API call)

This task requires a valid `OPENAI_API_KEY` in `.env`.

- [ ] **Step 1: Run test mode with GPT-4o-mini, minimal variant, 3 morals**

```bash
./run.sh llm_retrieval/run.py --test 3 --models GPT-4o-mini --variants minimal
```

Expected output:
```
Loaded 3 queries
Corpus block: 434,xxx chars (~108,xxx tokens)
▶ GPT-4o-mini × minimal
  MRR@10=x.xxxx  R@10=x.xxxx  (xxs)
Done. Results in llm_retrieval/results/unified.csv
```

- [ ] **Step 2: Inspect the per-run CSV**

```bash
cat llm_retrieval/results/runs/*.csv
```

Expected: 3 rows, each with `moral_id`, `relevant_fable`, `ranked_ids` (pipe-separated fable IDs), and metric columns.

- [ ] **Step 3: Inspect the unified CSV**

```bash
cat llm_retrieval/results/unified.csv
```

Expected: 1 summary row with `MRR@10`, `R@1`, `R@10`, etc.

- [ ] **Step 4: Test resumability — interrupt and re-run**

```bash
./run.sh llm_retrieval/run.py --test 10 --models GPT-4o-mini --variants minimal
# Ctrl+C after a few seconds, then re-run:
./run.sh llm_retrieval/run.py --test 10 --models GPT-4o-mini --variants minimal --skip-existing
```

Expected: second run resumes from where it left off (prints "Resuming from row N/10").

- [ ] **Step 5: Commit results gitignore and smoke test evidence**

Add `results/runs/*.csv` and `results/unified.csv` to `.gitignore` if not already present (results are large and run-specific). Keep `.gitkeep`.

```bash
echo "llm_retrieval/results/runs/*.csv" >> .gitignore
echo "llm_retrieval/results/unified.csv" >> .gitignore
git add .gitignore
git commit -m "chore(llm_retrieval): gitignore run result CSVs"
```
