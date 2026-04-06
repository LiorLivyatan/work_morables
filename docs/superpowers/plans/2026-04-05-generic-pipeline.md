# Generic Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the exp08 pipeline into `lib/pipeline/` — a fully configurable, reusable framework driven by a YAML config file, so future experiments need only a `config.yaml` and a thin `run_pipeline.py`.

**Architecture:** Shared pipeline modules live in `lib/pipeline/`. Each module owns one step (generate corpus summaries, generate query expansions, retrieval eval). A config loader resolves prompts from inline text, file, or shared key. An orchestrator (`run_experiment()`) ties the steps together, passing the run dir explicitly via `run_manifest.json`.

**Tech Stack:** Python 3.10+, PyYAML, pytest, google-genai, sentence-transformers, numpy, lib/embedding_cache.py, lib/retrieval_utils.py

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `lib/retrieval_utils.py` | Add `compute_metrics_from_matrix()` |
| Create | `lib/pipeline/__init__.py` | `load_config()` + `run_experiment()` orchestrator |
| Create | `lib/pipeline/llm_client.py` | Gemini client creation + call with retry/backoff |
| Create | `lib/pipeline/corpus_generator.py` | Step 1: fable × variant → corpus_summaries.json |
| Create | `lib/pipeline/query_expander.py` | Step 2: moral × variant → query_expansions.json |
| Create | `lib/pipeline/retrieval_eval.py` | Step 3: embed + score + fuse + metrics → retrieval_results.json |
| Create | `lib/pipeline/run_utils.py` | Run dir creation, .env loading, manifest I/O |
| Create | `lib/pipeline/prompts.py` | Shared prompt strings referenced by key |
| Create | `lib/pipeline/default_config.yaml` | Base defaults; experiments override only what differs |
| Create | `experiments/08_symmetric_moral_matching/config.yaml` | Exp08 config (prompts inline, 10 fables) |
| Create | `experiments/08_symmetric_moral_matching/run_pipeline.py` | Thin entry point for exp08 |
| Create | `tests/pipeline/__init__.py` | Make tests/pipeline a package |
| Create | `tests/pipeline/test_run_utils.py` | Tests for run_utils |
| Create | `tests/pipeline/test_llm_client.py` | Tests for llm_client |
| Create | `tests/pipeline/test_config_loader.py` | Tests for load_config |
| Create | `tests/pipeline/test_corpus_generator.py` | Tests for corpus_generator |
| Create | `tests/pipeline/test_query_expander.py` | Tests for query_expander |
| Create | `tests/pipeline/test_retrieval_eval.py` | Tests for retrieval_eval |

---

## Task 1: Add `compute_metrics_from_matrix` to `lib/retrieval_utils.py`

The current `lib/retrieval_utils.compute_metrics()` takes embeddings and computes cosine similarity internally. Fusion outputs a pre-computed score matrix, so we need a sibling function that accepts the matrix directly. Both must return identical metric keys.

**Files:**
- Modify: `lib/retrieval_utils.py`
- Modify: `tests/test_retrieval_utils.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_retrieval_utils.py`:

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.retrieval_utils import compute_metrics_from_matrix


def test_compute_metrics_from_matrix_perfect():
    """Identity score matrix: every query hits rank 0."""
    matrix = np.eye(4, dtype=np.float32)
    gt = {0: 0, 1: 1, 2: 2, 3: 3}
    m = compute_metrics_from_matrix(matrix, gt)
    assert m["MRR"] == pytest.approx(1.0)
    assert m["Recall@1"] == pytest.approx(1.0)
    assert m["n_queries"] == 4


def test_compute_metrics_from_matrix_matches_compute_metrics():
    """Results must be identical to compute_metrics() on normalized embeddings."""
    rng = np.random.default_rng(42)
    q = rng.random((5, 8)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    c = rng.random((10, 8)).astype(np.float32)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    gt = {0: 0, 1: 3, 2: 7, 3: 1, 4: 9}

    from sklearn.metrics.pairwise import cosine_similarity
    matrix = cosine_similarity(q, c).astype(np.float32)

    m_emb = compute_metrics(q, c, gt)
    m_mat = compute_metrics_from_matrix(matrix, gt)

    for key in ["MRR", "MAP", "R-Precision", "Recall@1", "Recall@5", "n_queries"]:
        assert m_emb[key] == pytest.approx(m_mat[key], abs=1e-5), f"Mismatch on {key}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/asifamar/Desktop/Master/NLP-morables
python -m pytest tests/test_retrieval_utils.py::test_compute_metrics_from_matrix_perfect -v
```
Expected: `ImportError: cannot import name 'compute_metrics_from_matrix'`

- [ ] **Step 3: Implement `compute_metrics_from_matrix` in `lib/retrieval_utils.py`**

Add after the existing `compute_metrics` function (after line 67):

```python
def compute_metrics_from_matrix(
    score_matrix: np.ndarray,
    ground_truth: dict,
    ks=(1, 5, 10, 50),
) -> dict:
    """
    Compute retrieval metrics from a pre-computed score matrix.

    Identical metric keys to compute_metrics(). Use this for fused score matrices
    where embeddings are not available directly.

    Args:
        score_matrix: (N_queries, N_docs) float array, higher = more relevant
        ground_truth: dict mapping query_idx (int) -> correct corpus_idx (int)
        ks: tuple of k values for Recall@k, P@k, NDCG@k

    Returns:
        dict with same keys as compute_metrics()
    """
    rankings = np.argsort(-score_matrix, axis=1)

    reciprocal_ranks = []
    recall_at_k = {k: [] for k in ks}
    precision_at_k = {k: [] for k in ks}
    ndcg_at_k = {k: [] for k in ks}
    r_precisions = []

    for q_idx, correct_idx in ground_truth.items():
        ranked = rankings[q_idx]
        rank = int(np.where(ranked == correct_idx)[0][0])

        reciprocal_ranks.append(1.0 / (rank + 1))
        for k in ks:
            hit = 1.0 if rank < k else 0.0
            recall_at_k[k].append(hit)
            precision_at_k[k].append(hit / k)
            ndcg_at_k[k].append(1.0 / np.log2(rank + 2) if rank < k else 0.0)
        r_precisions.append(1.0 if rank == 0 else 0.0)

    ranks_1indexed = [1.0 / rr for rr in reciprocal_ranks]
    results = {
        "MRR": float(np.mean(reciprocal_ranks)),
        "MAP": float(np.mean(reciprocal_ranks)),
        "R-Precision": float(np.mean(r_precisions)),
        "Mean Rank": float(np.mean(ranks_1indexed)),
        "Median Rank": float(np.median(ranks_1indexed)),
        "n_queries": len(reciprocal_ranks),
    }
    for k in ks:
        results[f"Recall@{k}"] = float(np.mean(recall_at_k[k]))
        results[f"P@{k}"] = float(np.mean(precision_at_k[k]))
        results[f"NDCG@{k}"] = float(np.mean(ndcg_at_k[k]))
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_retrieval_utils.py -v
```
Expected: All tests PASS including the 2 new ones.

- [ ] **Step 5: Commit**

```bash
git add lib/retrieval_utils.py tests/test_retrieval_utils.py
git commit -m "feat(lib): add compute_metrics_from_matrix for pre-computed score matrices"
```

---

## Task 2: `lib/pipeline/run_utils.py`

Run directory management, `.env` loading, and `run_manifest.json` I/O.

**Files:**
- Create: `lib/pipeline/__init__.py` (empty for now)
- Create: `lib/pipeline/run_utils.py`
- Create: `tests/pipeline/__init__.py`
- Create: `tests/pipeline/test_run_utils.py`

- [ ] **Step 1: Create empty package files**

Create `lib/pipeline/__init__.py` (empty):
```python
```

Create `tests/pipeline/__init__.py` (empty):
```python
```

- [ ] **Step 2: Write the failing tests**

Create `tests/pipeline/test_run_utils.py`:

```python
"""Tests for lib/pipeline/run_utils.py"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline.run_utils import (
    load_env,
    make_run_dir,
    find_latest_run_dir,
    write_manifest,
    read_manifest,
)


def test_make_run_dir_creates_directory(tmp_path):
    run_dir = make_run_dir(tmp_path, tag="sample10")
    assert run_dir.exists()
    assert run_dir.is_dir()
    assert "sample10" in run_dir.name


def test_make_run_dir_timestamp_format(tmp_path):
    run_dir = make_run_dir(tmp_path)
    # Name starts with YYYY-MM-DD
    import re
    assert re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", run_dir.name)


def test_make_run_dir_no_tag(tmp_path):
    run_dir = make_run_dir(tmp_path)
    assert run_dir.exists()


def test_find_latest_run_dir(tmp_path):
    (tmp_path / "2026-04-01_run").mkdir()
    (tmp_path / "2026-04-02_run").mkdir()
    (tmp_path / "2026-04-03_run").mkdir()
    latest = find_latest_run_dir(tmp_path)
    assert latest.name == "2026-04-03_run"


def test_find_latest_run_dir_missing_base(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_latest_run_dir(tmp_path / "nonexistent")


def test_write_and_read_manifest_roundtrip(tmp_path):
    cfg = {"n_fables": 10, "model": "gemini-3-flash"}
    write_manifest(tmp_path, "generate_corpus_summaries", cfg)
    manifest = read_manifest(tmp_path)
    assert "generate_corpus_summaries" in manifest["steps_completed"]
    assert manifest["config_snapshot"]["n_fables"] == 10


def test_write_manifest_appends_steps(tmp_path):
    write_manifest(tmp_path, "step_one", {})
    write_manifest(tmp_path, "step_two", {})
    manifest = read_manifest(tmp_path)
    assert manifest["steps_completed"] == ["step_one", "step_two"]


def test_write_manifest_no_duplicate_steps(tmp_path):
    write_manifest(tmp_path, "step_one", {})
    write_manifest(tmp_path, "step_one", {})
    manifest = read_manifest(tmp_path)
    assert manifest["steps_completed"].count("step_one") == 1


def test_read_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path)


def test_load_env(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_KEY_XYZ=hello_world\n# comment\nANOTHER=value # inline comment\n")
    os.environ.pop("TEST_KEY_XYZ", None)
    os.environ.pop("ANOTHER", None)
    load_env(tmp_path)
    assert os.environ["TEST_KEY_XYZ"] == "hello_world"
    assert os.environ["ANOTHER"] == "value"


def test_load_env_does_not_override_existing(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_KEY=new_value\n")
    os.environ["EXISTING_KEY"] = "original"
    load_env(tmp_path)
    assert os.environ["EXISTING_KEY"] == "original"
    del os.environ["EXISTING_KEY"]


def test_load_env_missing_file_is_noop(tmp_path):
    load_env(tmp_path / "nonexistent")  # should not raise
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
python -m pytest tests/pipeline/test_run_utils.py -v
```
Expected: `ModuleNotFoundError: No module named 'lib.pipeline.run_utils'`

- [ ] **Step 4: Implement `lib/pipeline/run_utils.py`**

Create `lib/pipeline/run_utils.py`:

```python
"""lib/pipeline/run_utils.py — Run directory management, .env loading, manifest I/O."""
import json
import os
from datetime import datetime
from pathlib import Path


def load_env(root_dir: Path) -> None:
    """Load KEY=VALUE pairs from root_dir/.env into os.environ (skips existing keys)."""
    env_path = Path(root_dir) / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.split("#")[0].strip()
            os.environ.setdefault(k.strip(), v)


def make_run_dir(base_dir: Path, tag: str = "") -> Path:
    """Create and return a timestamped run directory: base_dir/YYYY-MM-DD_HH-MM-SS[_tag]/"""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{ts}_{tag}" if tag else ts
    run_dir = Path(base_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_latest_run_dir(base_dir: Path) -> Path:
    """Return the lexicographically last subdirectory in base_dir."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    dirs = sorted(d for d in base_dir.iterdir() if d.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No subdirectories in {base_dir}")
    return dirs[-1]


def write_manifest(run_dir: Path, step: str, config_snapshot: dict) -> None:
    """Write/update run_manifest.json, recording that `step` completed."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "run_dir": str(run_dir),
            "steps_completed": [],
            "config_snapshot": config_snapshot,
        }

    if step not in manifest["steps_completed"]:
        manifest["steps_completed"].append(step)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def read_manifest(run_dir: Path) -> dict:
    """Read and return run_manifest.json from run_dir."""
    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")
    with open(manifest_path) as f:
        return json.load(f)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/pipeline/test_run_utils.py -v
```
Expected: All 11 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add lib/pipeline/__init__.py lib/pipeline/run_utils.py tests/pipeline/__init__.py tests/pipeline/test_run_utils.py
git commit -m "feat(pipeline): add run_utils — run dir management and manifest I/O"
```

---

## Task 3: `lib/pipeline/llm_client.py`

Gemini API wrapper with retry/backoff. Consolidates the duplicated logic in both exp08 generation scripts.

**Files:**
- Create: `lib/pipeline/llm_client.py`
- Create: `tests/pipeline/test_llm_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_llm_client.py`:

```python
"""Tests for lib/pipeline/llm_client.py"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline.llm_client import call


def _mock_response(text: str, prompt_tokens=10, candidate_tokens=5):
    resp = MagicMock()
    resp.text = text
    usage = MagicMock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = candidate_tokens
    usage.thoughts_token_count = 0
    usage.total_token_count = prompt_tokens + candidate_tokens
    resp.usage_metadata = usage
    return resp


def test_call_returns_expected_keys():
    client = MagicMock()
    client.models.generate_content.return_value = _mock_response("Honesty is the best policy.")
    result = call(client, "gemini-flash", "sys prompt", "user prompt")
    assert set(result.keys()) == {"text", "input_tokens", "output_tokens", "thinking_tokens", "total_tokens"}


def test_call_returns_stripped_text():
    client = MagicMock()
    client.models.generate_content.return_value = _mock_response("  Honesty wins.  ")
    result = call(client, "gemini-flash", "sys", "user")
    assert result["text"] == "Honesty wins."


def test_call_returns_token_counts():
    client = MagicMock()
    client.models.generate_content.return_value = _mock_response("ok", prompt_tokens=20, candidate_tokens=8)
    result = call(client, "gemini-flash", "sys", "user")
    assert result["input_tokens"] == 20
    assert result["output_tokens"] == 8
    assert result["total_tokens"] == 28


def test_call_retries_on_rate_limit():
    client = MagicMock()
    rate_error = Exception("Error 429: rate limit exceeded")
    client.models.generate_content.side_effect = [
        rate_error,
        rate_error,
        _mock_response("Success after retry"),
    ]
    with patch("time.sleep"):
        result = call(client, "gemini-flash", "sys", "user", max_retries=5)
    assert result["text"] == "Success after retry"
    assert client.models.generate_content.call_count == 3


def test_call_returns_error_after_max_retries():
    client = MagicMock()
    client.models.generate_content.side_effect = Exception("429: quota exceeded")
    with patch("time.sleep"):
        result = call(client, "gemini-flash", "sys", "user", max_retries=3)
    assert result["text"].startswith("[ERROR")
    assert result["input_tokens"] == 0


def test_call_passes_system_instruction():
    client = MagicMock()
    client.models.generate_content.return_value = _mock_response("ok")
    call(client, "gemini-flash", "Be concise.", "Tell me a moral.")
    _, kwargs = client.models.generate_content.call_args
    assert kwargs["config"]["system_instruction"] == "Be concise."
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/pipeline/test_llm_client.py -v
```
Expected: `ModuleNotFoundError: No module named 'lib.pipeline.llm_client'`

- [ ] **Step 3: Implement `lib/pipeline/llm_client.py`**

Create `lib/pipeline/llm_client.py`:

```python
"""lib/pipeline/llm_client.py — Gemini API client with retry/backoff."""
import os
import time
from typing import Optional


def create_client(api_key: Optional[str] = None):
    """Create and return a Gemini genai.Client from env or explicit key."""
    from google import genai
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=key)


def call(
    client,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 5,
) -> dict:
    """
    Call Gemini with exponential backoff on rate-limit errors.

    Returns:
        dict with keys: text, input_tokens, output_tokens, thinking_tokens, total_tokens
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=user_prompt,
                config={"system_instruction": system_prompt},
            )
            text = response.text.strip()
            usage = response.usage_metadata
            return {
                "text": text,
                "input_tokens": usage.prompt_token_count if usage else 0,
                "output_tokens": usage.candidates_token_count if usage else 0,
                "thinking_tokens": (usage.thoughts_token_count or 0) if usage else 0,
                "total_tokens": usage.total_token_count if usage else 0,
            }
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str or "quota" in err_str:
                wait = 2 ** attempt + 1
                print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"    Error on attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    return {"text": f"[ERROR: {e}]", "input_tokens": 0,
                            "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}
                time.sleep(2)

    return {"text": "[ERROR: max retries exceeded]", "input_tokens": 0,
            "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/pipeline/test_llm_client.py -v
```
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/pipeline/llm_client.py tests/pipeline/test_llm_client.py
git commit -m "feat(pipeline): add llm_client — Gemini API wrapper with retry/backoff"
```

---

## Task 4: `lib/pipeline/prompts.py` and `lib/pipeline/default_config.yaml`

The shared prompt library and the base config that all experiments inherit from.

**Files:**
- Create: `lib/pipeline/prompts.py`
- Create: `lib/pipeline/default_config.yaml`

- [ ] **Step 1: Create `lib/pipeline/prompts.py`**

```python
"""lib/pipeline/prompts.py — Shared system prompts referenced by prompt_key in configs."""

PROMPTS: dict[str, str] = {
    "ground_truth_style": (
        "You are an expert in fables. When given a fable, state its moral as a concise "
        "aphorism of 5 to 15 words. Use no character names. Be abstract and universal.\n\n"
        "Examples of the exact style required:\n"
        "- Appearances are deceptive.\n"
        "- Vices are their own punishment.\n"
        "- An ounce of prevention is worth a pound of cure.\n"
        "- Gratitude is the sign of noble souls.\n"
        "- Misfortune tests the sincerity of friends.\n\n"
        "Output ONLY the moral. No explanation, no narrative description."
    ),
    "declarative_universal": (
        "You are an expert in moral philosophy. When given a fable, distill its lesson "
        "into one declarative sentence of 5 to 15 words. The statement must be universal "
        "and timeless — no character names, no reference to the story's events. State an "
        "observation about human nature or behavior.\n\n"
        "Examples of the exact style required:\n"
        "- Those who envy others invite their own misfortune.\n"
        "- Necessity drives men to find solutions.\n"
        "- He who is content with little needs nothing more.\n\n"
        "Output ONLY the moral sentence. No explanation."
    ),
    "moral_rephrase": (
        "You are an expert in rephrasing moral statements. "
        "Given a moral from a fable, rephrase it using different words while preserving "
        "the exact same meaning. Output a single sentence of at most 15 words. "
        "Do not use character names or narrative description. Abstract and universal only."
    ),
    "moral_elaborate": (
        "You are an expert in moral philosophy. "
        "Given a moral from a fable, broaden it slightly to express the same principle "
        "in a wider context. Output a single sentence of at most 20 words. "
        "Keep it abstract and universal — no character names, no narrative examples."
    ),
    "moral_abstract": (
        "You are an expert in distilling principles to their essence. "
        "Given a moral from a fable, strip it to its most concise and abstract form. "
        "Output a single sentence of at most 10 words."
    ),
}
```

- [ ] **Step 2: Create `lib/pipeline/default_config.yaml`**

```yaml
# lib/pipeline/default_config.yaml
# Base defaults for all experiments. Each experiment's config.yaml overrides only what differs.

corpus_generation_model: gemini-3-flash-preview
query_expansion_model: gemini-3-flash-preview
embed_model: Linq-AI-Research/Linq-Embed-Mistral
embed_query_instruction: "Given a text, retrieve the most relevant passage that answers the query"

n_fables: null  # null = all fables in dataset

steps:
  generate_corpus_summaries: true
  generate_query_expansions: true
  run_retrieval_eval: true

# Variants are fully defined per-experiment (no defaults here)
corpus_variants: []
query_expansion_variants: []
retrieval_configs: []

baseline: null  # set per-experiment if comparison is needed

cache_dir: null  # null = <run_dir>/embedding_cache/
api_delay_seconds: 0.5
```

- [ ] **Step 3: Verify files exist (no tests needed for static data)**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from lib.pipeline.prompts import PROMPTS
import yaml
with open('lib/pipeline/default_config.yaml') as f:
    cfg = yaml.safe_load(f)
print('prompts keys:', list(PROMPTS.keys()))
print('config keys:', list(cfg.keys()))
assert 'ground_truth_style' in PROMPTS
assert 'corpus_variants' in cfg
print('OK')
"
```
Expected: Prints keys and `OK`.

- [ ] **Step 4: Commit**

```bash
git add lib/pipeline/prompts.py lib/pipeline/default_config.yaml
git commit -m "feat(pipeline): add shared prompts library and default config"
```

---

## Task 5: Config Loader in `lib/pipeline/__init__.py`

Loads default config, merges experiment override, resolves all variant prompts (inline / file / key).

**Files:**
- Modify: `lib/pipeline/__init__.py`
- Create: `tests/pipeline/test_config_loader.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_config_loader.py`:

```python
"""Tests for lib/pipeline load_config()"""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline import load_config


def _write_config(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f)
    return p


def test_load_config_inline_prompt(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "n_fables": 5,
        "corpus_variants": [{"name": "my_style", "prompt": "State the moral briefly."}],
    })
    cfg = load_config(cfg_path)
    assert cfg["corpus_variants"][0]["system_prompt"] == "State the moral briefly."


def test_load_config_prompt_file(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "custom.txt").write_text("Custom prompt from file.\n")
    cfg_path = _write_config(tmp_path, {
        "corpus_variants": [{"name": "custom", "prompt_file": "prompts/custom.txt"}],
    })
    cfg = load_config(cfg_path)
    assert cfg["corpus_variants"][0]["system_prompt"] == "Custom prompt from file."


def test_load_config_prompt_key(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "corpus_variants": [{"name": "gt", "prompt_key": "ground_truth_style"}],
    })
    cfg = load_config(cfg_path)
    assert "aphorism" in cfg["corpus_variants"][0]["system_prompt"]


def test_load_config_missing_prompt_raises(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "corpus_variants": [{"name": "orphan"}],
    })
    with pytest.raises(ValueError, match="must have one of"):
        load_config(cfg_path)


def test_load_config_unknown_prompt_key_raises(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "corpus_variants": [{"name": "x", "prompt_key": "nonexistent_key"}],
    })
    with pytest.raises(ValueError, match="Unknown prompt_key"):
        load_config(cfg_path)


def test_load_config_default_user_prompt_template(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "corpus_variants": [{"name": "s", "prompt": "Summarize."}],
        "query_expansion_variants": [{"name": "r", "prompt": "Rephrase."}],
    })
    cfg = load_config(cfg_path)
    assert cfg["corpus_variants"][0]["user_prompt_template"] == "Fable: {text}"
    assert cfg["query_expansion_variants"][0]["user_prompt_template"] == "Moral: {text}"


def test_load_config_custom_user_prompt_template(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "corpus_variants": [{
            "name": "s",
            "prompt": "Summarize.",
            "user_prompt_template": "Story: {text}",
        }],
    })
    cfg = load_config(cfg_path)
    assert cfg["corpus_variants"][0]["user_prompt_template"] == "Story: {text}"


def test_load_config_merges_scalars_over_defaults(tmp_path):
    cfg_path = _write_config(tmp_path, {"n_fables": 42})
    cfg = load_config(cfg_path)
    assert cfg["n_fables"] == 42
    assert cfg["embed_model"] == "Linq-AI-Research/Linq-Embed-Mistral"


def test_load_config_steps_deep_merge(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "steps": {"generate_corpus_summaries": False},
    })
    cfg = load_config(cfg_path)
    assert cfg["steps"]["generate_corpus_summaries"] is False
    assert cfg["steps"]["generate_query_expansions"] is True
    assert cfg["steps"]["run_retrieval_eval"] is True


def test_load_config_list_replaces_not_extends(tmp_path):
    cfg_path = _write_config(tmp_path, {
        "corpus_variants": [{"name": "only_one", "prompt": "Single variant."}],
    })
    cfg = load_config(cfg_path)
    assert len(cfg["corpus_variants"]) == 1
    assert cfg["corpus_variants"][0]["name"] == "only_one"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/pipeline/test_config_loader.py -v
```
Expected: `ImportError: cannot import name 'load_config' from 'lib.pipeline'`

- [ ] **Step 3: Implement `load_config` in `lib/pipeline/__init__.py`**

Replace the empty `lib/pipeline/__init__.py` with:

```python
"""lib/pipeline — Generic experiment pipeline for NLP-morables."""
import copy
from pathlib import Path
from typing import Optional

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"
_CORPUS_USER_TEMPLATE = "Fable: {text}"
_QUERY_USER_TEMPLATE = "Moral: {text}"


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base. Dicts recurse; lists and scalars replace."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _resolve_prompt(variant_cfg: dict, experiment_dir: Path) -> str:
    """Resolve a variant's system prompt: inline > file > key."""
    if "prompt" in variant_cfg:
        return variant_cfg["prompt"].strip()
    if "prompt_file" in variant_cfg:
        path = Path(experiment_dir) / variant_cfg["prompt_file"]
        return path.read_text().strip()
    if "prompt_key" in variant_cfg:
        from lib.pipeline.prompts import PROMPTS
        key = variant_cfg["prompt_key"]
        if key not in PROMPTS:
            raise ValueError(
                f"Unknown prompt_key: {key!r}. Available: {sorted(PROMPTS)}"
            )
        return PROMPTS[key]
    raise ValueError(
        f"Variant {variant_cfg.get('name', '?')!r} must have one of: "
        "prompt, prompt_file, prompt_key"
    )


def load_config(config_path: Path, experiment_dir: Optional[Path] = None) -> dict:
    """
    Load experiment config, merge over defaults, resolve all variant prompts.

    Args:
        config_path:     Path to experiment's config.yaml
        experiment_dir:  Base dir for prompt_file resolution (default: config_path.parent)

    Returns:
        Fully resolved config dict. Each variant entry has 'system_prompt' and
        'user_prompt_template' keys added.
    """
    experiment_dir = Path(experiment_dir or Path(config_path).parent)

    with open(_DEFAULT_CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}

    with open(config_path) as f:
        override = yaml.safe_load(f) or {}

    config = _deep_merge(config, override)

    for variant in config.get("corpus_variants", []):
        variant["system_prompt"] = _resolve_prompt(variant, experiment_dir)
        variant.setdefault("user_prompt_template", _CORPUS_USER_TEMPLATE)

    for variant in config.get("query_expansion_variants", []):
        variant["system_prompt"] = _resolve_prompt(variant, experiment_dir)
        variant.setdefault("user_prompt_template", _QUERY_USER_TEMPLATE)

    return config
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/pipeline/test_config_loader.py -v
```
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/pipeline/__init__.py tests/pipeline/test_config_loader.py
git commit -m "feat(pipeline): add load_config with inline/file/key prompt resolution"
```

---

## Task 6: `lib/pipeline/corpus_generator.py`

Generates style-matched summaries for each fable × corpus variant.

**Files:**
- Create: `lib/pipeline/corpus_generator.py`
- Create: `tests/pipeline/test_corpus_generator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_corpus_generator.py`:

```python
"""Tests for lib/pipeline/corpus_generator.py"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline.corpus_generator import generate_corpus_summaries


def _make_fables(n: int) -> list[dict]:
    return [
        {"doc_id": f"fable_{i:03d}", "title": f"Fable {i}", "text": f"Once upon a time {i}."}
        for i in range(n)
    ]


def _make_variants(names: list[str]) -> list[dict]:
    return [
        {"name": n, "system_prompt": f"Sys prompt for {n}.", "user_prompt_template": "Fable: {text}"}
        for n in names
    ]


def _mock_client(text: str = "Honesty is best.") -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.text = text
    usage = MagicMock()
    usage.prompt_token_count = 10
    usage.candidates_token_count = 5
    usage.thoughts_token_count = 0
    usage.total_token_count = 15
    resp.usage_metadata = usage
    client.models.generate_content.return_value = resp
    return client


def test_generate_corpus_summaries_creates_output_files(tmp_path):
    with patch("time.sleep"):
        generate_corpus_summaries(
            client=_mock_client(),
            fables=_make_fables(2),
            variants=_make_variants(["style_a"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    assert (tmp_path / "corpus_summaries.json").exists()
    assert (tmp_path / "token_usage.json").exists()


def test_generate_corpus_summaries_json_schema(tmp_path):
    with patch("time.sleep"):
        generate_corpus_summaries(
            client=_mock_client("Virtue triumphs."),
            fables=_make_fables(2),
            variants=_make_variants(["style_a", "style_b"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    with open(tmp_path / "corpus_summaries.json") as f:
        data = json.load(f)
    assert len(data) == 2
    item = data[0]
    assert "id" in item
    assert "summaries" in item
    assert "style_a" in item["summaries"]
    assert "style_b" in item["summaries"]
    assert item["summaries"]["style_a"] == "Virtue triumphs."


def test_generate_corpus_summaries_idempotent(tmp_path):
    """Second call skips if output already exists (force=False)."""
    with patch("time.sleep"):
        generate_corpus_summaries(
            client=_mock_client(),
            fables=_make_fables(3),
            variants=_make_variants(["s"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    call_count_after_first = _mock_client().models.generate_content.call_count

    new_client = _mock_client()
    with patch("time.sleep"):
        generate_corpus_summaries(
            client=new_client,
            fables=_make_fables(3),
            variants=_make_variants(["s"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
            force=False,
        )
    assert new_client.models.generate_content.call_count == 0


def test_generate_corpus_summaries_force_reruns(tmp_path):
    """force=True re-generates even if output exists."""
    with patch("time.sleep"):
        generate_corpus_summaries(
            client=_mock_client(),
            fables=_make_fables(2),
            variants=_make_variants(["s"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    new_client = _mock_client()
    with patch("time.sleep"):
        generate_corpus_summaries(
            client=new_client,
            fables=_make_fables(2),
            variants=_make_variants(["s"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
            force=True,
        )
    assert new_client.models.generate_content.call_count == 2  # 2 fables × 1 variant


def test_generate_corpus_summaries_token_usage_json(tmp_path):
    with patch("time.sleep"):
        generate_corpus_summaries(
            client=_mock_client(),
            fables=_make_fables(3),
            variants=_make_variants(["s"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    with open(tmp_path / "token_usage.json") as f:
        usage = json.load(f)
    assert usage["n_fables"] == 3
    assert usage["model"] == "gemini-flash"
    assert usage["total_input_tokens"] == 30  # 3 fables × 10 tokens
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/pipeline/test_corpus_generator.py -v
```
Expected: `ModuleNotFoundError: No module named 'lib.pipeline.corpus_generator'`

- [ ] **Step 3: Implement `lib/pipeline/corpus_generator.py`**

Create `lib/pipeline/corpus_generator.py`:

```python
"""lib/pipeline/corpus_generator.py — Generate style-matched summaries for fable corpus."""
import json
import time
from pathlib import Path

from lib.pipeline import llm_client as _llm


_OUTPUT_FILE = "corpus_summaries.json"
_TOKEN_FILE = "token_usage.json"


def generate_corpus_summaries(
    client,
    fables: list[dict],
    variants: list[dict],
    model_id: str,
    run_dir: Path,
    delay: float = 0.5,
    force: bool = False,
) -> Path:
    """
    Generate corpus summaries for each fable × variant combination.

    Args:
        client:   Gemini client from llm_client.create_client()
        fables:   List of fable dicts with keys: doc_id, title, text
        variants: Resolved variant dicts each with: name, system_prompt, user_prompt_template
        model_id: Gemini model identifier string
        run_dir:  Directory to write corpus_summaries.json and token_usage.json
        delay:    Seconds to sleep between API calls (rate limiting)
        force:    Re-generate even if output already exists

    Returns:
        Path to written corpus_summaries.json
    """
    run_dir = Path(run_dir)
    output_path = run_dir / _OUTPUT_FILE

    if output_path.exists() and not force:
        print(f"  [skip] {output_path.name} already exists (use force=True to regenerate)")
        return output_path

    print(f"\n  Corpus generation: {len(fables)} fables × {len(variants)} variants  |  model: {model_id}")

    corpus = []
    total_input = total_output = total_thinking = 0

    for i, fable in enumerate(fables):
        fable_idx = int(fable["doc_id"].split("_")[1])
        item = {
            "id": f"item_{fable_idx:03d}",
            "original_fable_id": fable.get("alias", fable["doc_id"]),
            "fable_text": fable["text"],
            "summaries": {},
            "token_usage": {},
            "metadata": {
                "source": fable.get("alias", "unknown").split("_")[0],
                "word_count_fable": len(fable["text"].split()),
                "model": model_id,
            },
        }

        print(f"\n  [{i+1}/{len(fables)}] {fable.get('title', item['id'])}")

        for variant in variants:
            user_prompt = variant["user_prompt_template"].format(text=fable["text"])
            result = _llm.call(client, model_id, variant["system_prompt"], user_prompt)
            item["summaries"][variant["name"]] = result["text"]
            item["token_usage"][variant["name"]] = {
                k: result[k]
                for k in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens")
            }
            total_input += result["input_tokens"]
            total_output += result["output_tokens"]
            total_thinking += result["thinking_tokens"]
            short = result["text"][:80] + ("..." if len(result["text"]) > 80 else "")
            print(f"    {variant['name']}: {short}  [{result['input_tokens']}in/{result['output_tokens']}out]")
            time.sleep(delay)

        corpus.append(item)

    with open(output_path, "w") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    token_summary = {
        "model": model_id,
        "n_fables": len(corpus),
        "variants": [v["name"] for v in variants],
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_thinking_tokens": total_thinking,
        "total_tokens": total_input + total_thinking + total_output,
    }
    with open(run_dir / _TOKEN_FILE, "w") as f:
        json.dump(token_summary, f, indent=2)

    print(f"\n  Saved {len(corpus)} items to {output_path}")
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/pipeline/test_corpus_generator.py -v
```
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/pipeline/corpus_generator.py tests/pipeline/test_corpus_generator.py
git commit -m "feat(pipeline): add corpus_generator — fable × variant summary generation"
```

---

## Task 7: `lib/pipeline/query_expander.py`

Generates query expansion paraphrases for each moral × expansion variant.

**Files:**
- Create: `lib/pipeline/query_expander.py`
- Create: `tests/pipeline/test_query_expander.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_query_expander.py`:

```python
"""Tests for lib/pipeline/query_expander.py"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline.query_expander import generate_query_expansions


def _mock_client(text: str = "Honesty is best.") -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.text = text
    usage = MagicMock()
    usage.prompt_token_count = 8
    usage.candidates_token_count = 4
    usage.thoughts_token_count = 0
    usage.total_token_count = 12
    resp.usage_metadata = usage
    client.models.generate_content.return_value = resp
    return client


def _make_morals(indices: list[int]) -> list[dict]:
    return [{"doc_id": f"moral_{i:03d}", "text": f"Moral text {i}."} for i in range(max(indices) + 1)]


def _make_variants(names: list[str]) -> list[dict]:
    return [
        {"name": n, "system_prompt": f"Rephrase: {n}", "user_prompt_template": "Moral: {text}"}
        for n in names
    ]


def test_generate_query_expansions_creates_output(tmp_path):
    moral_entries = [(0, 0), (1, 1)]
    morals = _make_morals([0, 1])
    with patch("time.sleep"):
        generate_query_expansions(
            client=_mock_client(),
            moral_entries=moral_entries,
            morals=morals,
            variants=_make_variants(["rephrase"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    assert (tmp_path / "query_expansions.json").exists()
    assert (tmp_path / "query_expansion_token_usage.json").exists()


def test_generate_query_expansions_json_schema(tmp_path):
    moral_entries = [(2, 0), (5, 1)]
    morals = _make_morals([2, 5])
    with patch("time.sleep"):
        generate_query_expansions(
            client=_mock_client("Virtue wins."),
            moral_entries=moral_entries,
            morals=morals,
            variants=_make_variants(["rephrase", "abstract"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    with open(tmp_path / "query_expansions.json") as f:
        data = json.load(f)
    assert len(data) == 2
    item = data[0]
    assert item["moral_idx"] == 2
    assert item["fable_idx"] == 0
    assert "rephrase" in item["paraphrases"]
    assert "abstract" in item["paraphrases"]
    assert item["paraphrases"]["rephrase"] == "Virtue wins."


def test_generate_query_expansions_idempotent(tmp_path):
    moral_entries = [(0, 0)]
    morals = _make_morals([0])
    with patch("time.sleep"):
        generate_query_expansions(
            client=_mock_client(),
            moral_entries=moral_entries,
            morals=morals,
            variants=_make_variants(["r"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    new_client = _mock_client()
    with patch("time.sleep"):
        generate_query_expansions(
            client=new_client,
            moral_entries=moral_entries,
            morals=morals,
            variants=_make_variants(["r"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
            force=False,
        )
    assert new_client.models.generate_content.call_count == 0


def test_generate_query_expansions_token_usage(tmp_path):
    moral_entries = [(0, 0), (1, 1), (2, 2)]
    morals = _make_morals([0, 1, 2])
    with patch("time.sleep"):
        generate_query_expansions(
            client=_mock_client(),
            moral_entries=moral_entries,
            morals=morals,
            variants=_make_variants(["r"]),
            model_id="gemini-flash",
            run_dir=tmp_path,
        )
    with open(tmp_path / "query_expansion_token_usage.json") as f:
        usage = json.load(f)
    assert usage["n_morals"] == 3
    assert usage["total_input_tokens"] == 24  # 3 morals × 8 tokens
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/pipeline/test_query_expander.py -v
```
Expected: `ModuleNotFoundError: No module named 'lib.pipeline.query_expander'`

- [ ] **Step 3: Implement `lib/pipeline/query_expander.py`**

Create `lib/pipeline/query_expander.py`:

```python
"""lib/pipeline/query_expander.py — Generate paraphrase expansions for moral queries."""
import json
import time
from pathlib import Path

from lib.pipeline import llm_client as _llm


_OUTPUT_FILE = "query_expansions.json"
_TOKEN_FILE = "query_expansion_token_usage.json"


def generate_query_expansions(
    client,
    moral_entries: list[tuple[int, int]],
    morals: list[dict],
    variants: list[dict],
    model_id: str,
    run_dir: Path,
    delay: float = 0.5,
    force: bool = False,
) -> Path:
    """
    Generate query expansion paraphrases for each moral × variant.

    Args:
        client:        Gemini client from llm_client.create_client()
        moral_entries: List of (moral_idx, fable_idx) tuples to process
        morals:        Full morals list from lib.data.load_morals()
        variants:      Resolved variant dicts each with: name, system_prompt, user_prompt_template
        model_id:      Gemini model identifier string
        run_dir:       Directory to write query_expansions.json
        delay:         Seconds to sleep between API calls
        force:         Re-generate even if output already exists

    Returns:
        Path to written query_expansions.json
    """
    run_dir = Path(run_dir)
    output_path = run_dir / _OUTPUT_FILE

    if output_path.exists() and not force:
        print(f"  [skip] {output_path.name} already exists (use force=True to regenerate)")
        return output_path

    print(f"\n  Query expansion: {len(moral_entries)} morals × {len(variants)} variants  |  model: {model_id}")

    expansions = []
    total_input = total_output = total_thinking = 0

    for i, (moral_idx, fable_idx) in enumerate(moral_entries):
        moral_text = morals[moral_idx]["text"]
        item = {
            "id": f"moral_{moral_idx:03d}",
            "moral_idx": moral_idx,
            "fable_idx": fable_idx,
            "original_moral": moral_text,
            "paraphrases": {},
            "token_usage": {},
        }

        print(f"\n  [{i+1}/{len(moral_entries)}] moral_{moral_idx:03d}: {moral_text[:60]}")

        for variant in variants:
            user_prompt = variant["user_prompt_template"].format(text=moral_text)
            result = _llm.call(client, model_id, variant["system_prompt"], user_prompt)
            item["paraphrases"][variant["name"]] = result["text"]
            item["token_usage"][variant["name"]] = {
                k: result[k]
                for k in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens")
            }
            total_input += result["input_tokens"]
            total_output += result["output_tokens"]
            total_thinking += result["thinking_tokens"]
            print(f"    {variant['name']}: {result['text'][:70]}")
            time.sleep(delay)

        expansions.append(item)

    with open(output_path, "w") as f:
        json.dump(expansions, f, indent=2, ensure_ascii=False)

    token_summary = {
        "model": model_id,
        "n_morals": len(expansions),
        "variants": [v["name"] for v in variants],
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_thinking_tokens": total_thinking,
        "total_tokens": total_input + total_thinking + total_output,
    }
    with open(run_dir / _TOKEN_FILE, "w") as f:
        json.dump(token_summary, f, indent=2)

    print(f"\n  Saved {len(expansions)} expansions to {output_path}")
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/pipeline/test_query_expander.py -v
```
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/pipeline/query_expander.py tests/pipeline/test_query_expander.py
git commit -m "feat(pipeline): add query_expander — moral × variant paraphrase generation"
```

---

## Task 8: `lib/pipeline/retrieval_eval.py`

Embedding + scoring + fusion + metrics for all retrieval configs. Uses `encode_with_cache` and `compute_metrics_from_matrix`.

**Files:**
- Create: `lib/pipeline/retrieval_eval.py`
- Create: `tests/pipeline/test_retrieval_eval.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_retrieval_eval.py`:

```python
"""Tests for lib/pipeline/retrieval_eval.py"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline.retrieval_eval import run_retrieval_eval


def _write_corpus_summaries(run_dir: Path, n: int, variant_names: list[str]):
    data = []
    for i in range(n):
        data.append({
            "id": f"item_{i:03d}",
            "summaries": {v: f"Summary {i} for {v}" for v in variant_names},
        })
    with open(run_dir / "corpus_summaries.json", "w") as f:
        json.dump(data, f)


def _write_query_expansions(run_dir: Path, moral_indices: list[int], variant_names: list[str]):
    data = []
    for mi in moral_indices:
        data.append({
            "moral_idx": mi,
            "paraphrases": {v: f"Paraphrase {mi} for {v}" for v in variant_names},
        })
    with open(run_dir / "query_expansions.json", "w") as f:
        json.dump(data, f)


def _identity_config(variant: str = "style_a") -> dict:
    return {
        "embed_model": "test-model",
        "embed_query_instruction": "retrieve",
        "n_fables": 3,
        "retrieval_configs": [
            {"name": "base", "corpus_variant": variant, "use_expansion": False}
        ],
        "baseline": None,
        "cache_dir": None,
    }


def _mock_model(n_texts: int, dim: int = 4):
    """Return a mock SentenceTransformer that returns identity-ish embeddings."""
    model = MagicMock()
    def encode_side_effect(texts, **kwargs):
        embs = np.zeros((len(texts), dim), dtype=np.float32)
        for i in range(min(len(texts), dim)):
            embs[i, i] = 1.0
        return embs
    model.encode.side_effect = encode_side_effect
    return model


def test_run_retrieval_eval_raises_on_missing_variant(tmp_path):
    _write_corpus_summaries(tmp_path, 3, ["style_a"])
    config = _identity_config(variant="nonexistent_variant")
    fable_texts = [f"fable {i}" for i in range(3)]
    moral_texts = [f"moral {i}" for i in range(3)]
    gt = {0: 0, 1: 1, 2: 2}
    moral_indices = [0, 1, 2]

    with patch("lib.pipeline.retrieval_eval._load_model", return_value=(_mock_model(3), "cpu")):
        with patch("lib.pipeline.retrieval_eval.encode_with_cache") as mock_enc:
            mock_enc.return_value = np.eye(3, 4, dtype=np.float32)
            with pytest.raises(ValueError, match="nonexistent_variant"):
                run_retrieval_eval(tmp_path, config, fable_texts, moral_texts, gt, moral_indices)


def test_run_retrieval_eval_produces_results_json(tmp_path):
    _write_corpus_summaries(tmp_path, 3, ["style_a"])
    config = _identity_config("style_a")
    fable_texts = [f"fable {i}" for i in range(3)]
    moral_texts = [f"moral {i}" for i in range(3)]
    gt = {0: 0, 1: 1, 2: 2}
    moral_indices = [0, 1, 2]

    with patch("lib.pipeline.retrieval_eval._load_model", return_value=(_mock_model(3), "cpu")):
        call_count = [0]
        def mock_encode(model, texts, model_id, cache_dir, query_instruction=None, label="", **kw):
            idx = call_count[0]
            call_count[0] += 1
            n = len(texts)
            embs = np.zeros((n, 4), dtype=np.float32)
            for i in range(min(n, 4)):
                embs[i, i] = 1.0
            return embs

        with patch("lib.pipeline.retrieval_eval.encode_with_cache", side_effect=mock_encode):
            results = run_retrieval_eval(tmp_path, config, fable_texts, moral_texts, gt, moral_indices)

    assert "base" in results
    assert "Recall@1" in results["base"]
    assert "MRR" in results["base"]
    assert (tmp_path / "retrieval_results.json").exists()


def test_run_retrieval_eval_idempotent(tmp_path):
    _write_corpus_summaries(tmp_path, 2, ["style_a"])
    config = _identity_config("style_a")
    config["n_fables"] = 2
    result_data = {"base": {"Recall@1": 1.0, "MRR": 1.0, "n_queries": 2}}
    with open(tmp_path / "retrieval_results.json", "w") as f:
        json.dump(result_data, f)

    with patch("lib.pipeline.retrieval_eval._load_model") as mock_load:
        run_retrieval_eval(
            tmp_path, config,
            ["f0", "f1"], ["m0", "m1"], {0: 0, 1: 1}, [0, 1],
            force=False,
        )
        mock_load.assert_not_called()


def test_run_retrieval_eval_raises_when_expansion_file_missing(tmp_path):
    _write_corpus_summaries(tmp_path, 2, ["style_a"])
    config = {
        "embed_model": "test-model",
        "embed_query_instruction": "retrieve",
        "n_fables": 2,
        "retrieval_configs": [
            {"name": "exp", "corpus_variant": "style_a", "use_expansion": True,
             "expansion_variants": ["rephrase"]}
        ],
        "baseline": None,
        "cache_dir": None,
    }
    with patch("lib.pipeline.retrieval_eval._load_model", return_value=(_mock_model(2), "cpu")):
        with patch("lib.pipeline.retrieval_eval.encode_with_cache",
                   return_value=np.eye(2, 4, dtype=np.float32)):
            with pytest.raises(FileNotFoundError, match="query_expansions"):
                run_retrieval_eval(tmp_path, config, ["f0", "f1"], ["m0", "m1"],
                                   {0: 0, 1: 1}, [0, 1])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/pipeline/test_retrieval_eval.py -v
```
Expected: `ModuleNotFoundError: No module named 'lib.pipeline.retrieval_eval'`

- [ ] **Step 3: Implement `lib/pipeline/retrieval_eval.py`**

Create `lib/pipeline/retrieval_eval.py`:

```python
"""lib/pipeline/retrieval_eval.py — Embedding-based retrieval evaluation."""
import json
from pathlib import Path
from typing import Optional

import numpy as np

from lib.embedding_cache import encode_with_cache
from lib.retrieval_utils import compute_metrics_from_matrix


_OUTPUT_FILE = "retrieval_results.json"


def _load_model(model_id: str, device: Optional[str] = None):
    import torch
    from sentence_transformers import SentenceTransformer
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Loading {model_id} on {device}...")
    return SentenceTransformer(model_id, device=device), device


def _rrf(score_matrices: list[np.ndarray], k: int = 60) -> np.ndarray:
    n_queries, n_docs = score_matrices[0].shape
    fused = np.zeros((n_queries, n_docs), dtype=np.float64)
    for scores in score_matrices:
        ranked = np.argsort(-scores, axis=1)
        rank_matrix = np.empty_like(ranked)
        for q in range(n_queries):
            rank_matrix[q, ranked[q]] = np.arange(1, n_docs + 1)
        fused += 1.0 / (k + rank_matrix)
    return fused.astype(np.float32)


def _max_score(score_matrices: list[np.ndarray]) -> np.ndarray:
    return np.maximum.reduce(score_matrices)


def run_retrieval_eval(
    run_dir: Path,
    config: dict,
    fable_texts: list[str],
    moral_texts: list[str],
    ground_truth: dict,
    moral_indices: list[int],
    force: bool = False,
) -> dict:
    """
    Run retrieval evaluation for all configs in config['retrieval_configs'].

    Args:
        run_dir:       Pipeline run directory containing corpus_summaries.json
        config:        Resolved pipeline config dict
        fable_texts:   Raw fable texts (length = n_fables)
        moral_texts:   Moral query texts
        ground_truth:  {contiguous_query_idx: fable_idx}
        moral_indices: Original moral indices (for expansion lookup)
        force:         Re-run even if retrieval_results.json exists

    Returns:
        Dict mapping config name → metrics dict
    """
    run_dir = Path(run_dir)
    output_path = run_dir / _OUTPUT_FILE

    if output_path.exists() and not force:
        print(f"  [skip] {output_path.name} exists (use force=True to re-evaluate)")
        with open(output_path) as f:
            return json.load(f)

    embed_model_id = config["embed_model"]
    query_instruction = config.get("embed_query_instruction")
    retrieval_configs = config["retrieval_configs"]
    n_fables = config.get("n_fables") or len(fable_texts)
    cache_dir = (
        Path(config["cache_dir"]) if config.get("cache_dir")
        else run_dir / "embedding_cache"
    )

    # Load corpus summaries
    with open(run_dir / "corpus_summaries.json") as f:
        corpus_data = json.load(f)

    corpus_lookup = {
        int(item["id"].split("_")[1]): item["summaries"]
        for item in corpus_data
    }

    # Validate all referenced corpus variants exist
    available_variants: set[str] = set()
    if corpus_data:
        available_variants = set(corpus_data[0]["summaries"].keys())

    for rc in retrieval_configs:
        if "corpus_variant" in rc:
            cv = rc["corpus_variant"]
            if cv not in available_variants:
                raise ValueError(
                    f"Retrieval config {rc['name']!r} references corpus_variant {cv!r}, "
                    f"but corpus_summaries.json only has: {sorted(available_variants)}"
                )

    # Load query expansions if needed
    expansion_lookup: dict[int, dict] = {}
    uses_expansion = any(rc.get("use_expansion") for rc in retrieval_configs)
    if uses_expansion:
        exp_path = run_dir / "query_expansions.json"
        if not exp_path.exists():
            raise FileNotFoundError(
                f"Retrieval config requires expansion but {exp_path} not found. "
                "Run generate_query_expansions step first."
            )
        with open(exp_path) as f:
            expansion_lookup = {item["moral_idx"]: item["paraphrases"] for item in json.load(f)}

    model, _ = _load_model(embed_model_id)

    # Encode moral queries once
    moral_embs = encode_with_cache(
        model=model, texts=moral_texts, model_id=embed_model_id,
        cache_dir=cache_dir, query_instruction=query_instruction, label="moral queries",
    )

    # Lazy corpus embedding cache
    _corpus_embs: dict[str, np.ndarray] = {}

    def get_corpus_embs(variant_name: str) -> np.ndarray:
        if variant_name not in _corpus_embs:
            texts = [corpus_lookup.get(i, {}).get(variant_name, "") for i in range(n_fables)]
            _corpus_embs[variant_name] = encode_with_cache(
                model=model, texts=texts, model_id=embed_model_id,
                cache_dir=cache_dir, query_instruction=None, label=f"corpus:{variant_name}",
            )
        return _corpus_embs[variant_name]

    # Lazy expansion embedding cache
    _expansion_embs: dict[str, np.ndarray] = {}

    def get_expansion_embs(variant_name: str) -> np.ndarray:
        if variant_name not in _expansion_embs:
            texts = [
                expansion_lookup.get(moral_indices[q], {}).get(variant_name, moral_texts[q])
                for q in range(len(moral_texts))
            ]
            _expansion_embs[variant_name] = encode_with_cache(
                model=model, texts=texts, model_id=embed_model_id,
                cache_dir=cache_dir, query_instruction=query_instruction,
                label=f"expansion:{variant_name}",
            )
        return _expansion_embs[variant_name]

    all_results: dict[str, dict] = {}
    score_matrices: dict[str, np.ndarray] = {}

    # Optional baseline
    baseline_cfg = config.get("baseline")
    if baseline_cfg and baseline_cfg.get("path"):
        baseline_path = Path(baseline_cfg["path"])
        if not baseline_path.is_absolute():
            root = Path(__file__).parent.parent.parent
            baseline_path = root / baseline_cfg["path"]
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        bv = baseline_cfg["variant"]
        b_lookup = {int(item["id"].split("_")[1]): item["summaries"].get(bv, "") for item in baseline_data}
        b_texts = [b_lookup.get(i, "") for i in range(n_fables)]
        b_embs = encode_with_cache(
            model=model, texts=b_texts, model_id=embed_model_id,
            cache_dir=cache_dir, query_instruction=None, label=f"baseline:{bv}",
        )
        b_matrix = moral_embs @ b_embs.T
        b_metrics = compute_metrics_from_matrix(b_matrix, ground_truth)
        all_results["baseline"] = b_metrics
        print(f"\n  baseline: R@1={b_metrics['Recall@1']:.3f}  MRR={b_metrics['MRR']:.4f}")

    for rc in retrieval_configs:
        name = rc["name"]

        if "fusion" in rc:
            sources = rc.get("source_configs", [])
            matrices = [score_matrices[s] for s in sources if s in score_matrices]
            if not matrices:
                print(f"  [skip] {name}: no source score matrices ready")
                continue
            score_matrix = _rrf(matrices, k=rc.get("k", 60))
        else:
            corpus_embs = get_corpus_embs(rc["corpus_variant"])
            score_matrix = moral_embs @ corpus_embs.T
            if rc.get("use_expansion"):
                matrices = [score_matrix]
                for ev_name in rc.get("expansion_variants", []):
                    exp_embs = get_expansion_embs(ev_name)
                    matrices.append(exp_embs @ corpus_embs.T)
                score_matrix = _max_score(matrices)

        score_matrices[name] = score_matrix
        metrics = compute_metrics_from_matrix(score_matrix, ground_truth)
        all_results[name] = metrics

        b_r1 = all_results.get("baseline", {}).get("Recall@1")
        delta_str = ""
        if b_r1 is not None:
            delta = metrics["Recall@1"] - b_r1
            delta_str = f"  (vs baseline: {'+' if delta >= 0 else ''}{delta:.3f})"
        print(f"\n  {name}: R@1={metrics['Recall@1']:.3f}  MRR={metrics['MRR']:.4f}{delta_str}")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {output_path}")
    return all_results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/pipeline/test_retrieval_eval.py -v
```
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/pipeline/retrieval_eval.py tests/pipeline/test_retrieval_eval.py
git commit -m "feat(pipeline): add retrieval_eval — embed, fuse, and score retrieval configs"
```

---

## Task 9: `run_experiment()` orchestrator in `lib/pipeline/__init__.py`

Ties all steps together: loads config, builds data subset, runs enabled steps, writes manifest.

**Files:**
- Modify: `lib/pipeline/__init__.py`

- [ ] **Step 1: Add `run_experiment()` to `lib/pipeline/__init__.py`**

Append to the existing `lib/pipeline/__init__.py` (after `load_config`):

```python
def run_experiment(
    config_path: Path,
    run_dir: Optional[Path] = None,
    force: bool = False,
) -> None:
    """
    Run the full pipeline for an experiment.

    Args:
        config_path: Path to experiment's config.yaml
        run_dir:     Existing run dir to continue (default: create new timestamped dir)
        force:       Re-run steps even if their output already exists
    """
    import sys as _sys
    _ROOT = Path(__file__).parent.parent.parent
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))

    from lib.pipeline import (
        corpus_generator,
        query_expander,
        retrieval_eval,
        llm_client as _lc,
    )
    from lib.pipeline.run_utils import load_env, make_run_dir, write_manifest
    from lib.data import load_fables, load_morals, load_qrels_moral_to_fable

    config_path = Path(config_path)
    experiment_dir = config_path.parent

    load_env(_ROOT)
    config = load_config(config_path, experiment_dir)

    n_fables = config.get("n_fables")
    tag = f"sample{n_fables}" if n_fables else "full"

    if run_dir is None:
        base = experiment_dir / "results" / "pipeline_runs"
        run_dir = make_run_dir(base, tag)
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Pipeline: {experiment_dir.name}")
    print(f"  Run dir:  {run_dir}")
    print(f"  n_fables: {n_fables or 'all'}")
    print(f"{'=' * 60}")

    # Load data subset
    fables = load_fables()
    morals = load_morals()
    gt_m2f = load_qrels_moral_to_fable()

    if n_fables:
        fables = fables[:n_fables]
    else:
        n_fables = len(fables)

    target_fable_indices = set(range(n_fables))
    moral_entries = sorted(
        [(m_idx, f_idx) for m_idx, f_idx in gt_m2f.items()
         if f_idx in target_fable_indices],
        key=lambda x: x[0],
    )

    fable_texts = [f["text"] for f in fables]
    moral_texts = [morals[m_idx]["text"] for m_idx, _ in moral_entries]
    moral_indices = [m_idx for m_idx, _ in moral_entries]
    ground_truth = {i: f_idx for i, (_, f_idx) in enumerate(moral_entries)}

    steps = config.get("steps", {})

    # Step 1: generate corpus summaries
    if steps.get("generate_corpus_summaries", True) and config.get("corpus_variants"):
        print("\n── Step 1: Generate corpus summaries ──────────────────")
        client = _lc.create_client()
        corpus_generator.generate_corpus_summaries(
            client=client,
            fables=fables,
            variants=config["corpus_variants"],
            model_id=config["corpus_generation_model"],
            run_dir=run_dir,
            delay=config.get("api_delay_seconds", 0.5),
            force=force,
        )
        write_manifest(run_dir, "generate_corpus_summaries", config)

    # Step 2: generate query expansions
    if steps.get("generate_query_expansions", True) and config.get("query_expansion_variants"):
        print("\n── Step 2: Generate query expansions ──────────────────")
        client = _lc.create_client()
        query_expander.generate_query_expansions(
            client=client,
            moral_entries=moral_entries,
            morals=morals,
            variants=config["query_expansion_variants"],
            model_id=config["query_expansion_model"],
            run_dir=run_dir,
            delay=config.get("api_delay_seconds", 0.5),
            force=force,
        )
        write_manifest(run_dir, "generate_query_expansions", config)

    # Step 3: retrieval eval
    if steps.get("run_retrieval_eval", True) and config.get("retrieval_configs"):
        print("\n── Step 3: Retrieval evaluation ───────────────────────")
        retrieval_eval.run_retrieval_eval(
            run_dir=run_dir,
            config=config,
            fable_texts=fable_texts,
            moral_texts=moral_texts,
            ground_truth=ground_truth,
            moral_indices=moral_indices,
            force=force,
        )
        write_manifest(run_dir, "run_retrieval_eval", config)

    print(f"\n  Done. Results in {run_dir}")
```

- [ ] **Step 2: Verify the full module imports cleanly**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from lib.pipeline import load_config, run_experiment
print('load_config:', load_config)
print('run_experiment:', run_experiment)
print('OK')
"
```
Expected: Prints both functions and `OK`.

- [ ] **Step 3: Run all pipeline tests to confirm nothing broke**

```bash
python -m pytest tests/pipeline/ -v
```
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add lib/pipeline/__init__.py
git commit -m "feat(pipeline): add run_experiment() orchestrator — ties all steps together"
```

---

## Task 10: Wire exp08 — `config.yaml` + `run_pipeline.py`

Create the exp08 config (all prompts inline, n_fables=10) and a thin entry point.

**Files:**
- Create: `experiments/08_symmetric_moral_matching/config.yaml`
- Create: `experiments/08_symmetric_moral_matching/run_pipeline.py`

- [ ] **Step 1: Create `experiments/08_symmetric_moral_matching/config.yaml`**

```yaml
# experiments/08_symmetric_moral_matching/config.yaml
n_fables: 10

corpus_variants:
  - name: ground_truth_style
    prompt: |
      You are an expert in fables. When given a fable, state its moral as a concise
      aphorism of 5 to 15 words. Use no character names. Be abstract and universal.

      Examples of the exact style required:
      - Appearances are deceptive.
      - Vices are their own punishment.
      - An ounce of prevention is worth a pound of cure.
      - Gratitude is the sign of noble souls.
      - Misfortune tests the sincerity of friends.

      Output ONLY the moral. No explanation, no narrative description.

  - name: declarative_universal
    prompt: |
      You are an expert in moral philosophy. When given a fable, distill its lesson
      into one declarative sentence of 5 to 15 words. The statement must be universal
      and timeless — no character names, no reference to the story's events. State an
      observation about human nature or behavior.

      Examples of the exact style required:
      - Those who envy others invite their own misfortune.
      - Necessity drives men to find solutions.
      - He who is content with little needs nothing more.

      Output ONLY the moral sentence. No explanation.

query_expansion_variants:
  - name: moral_rephrase
    prompt: "Given a moral from a fable, rephrase it using different words while preserving the exact same meaning. Output a single sentence of at most 15 words. Abstract and universal only."

  - name: moral_elaborate
    prompt: "Given a moral from a fable, broaden it slightly to express the same principle in a wider context. Output a single sentence of at most 20 words. Abstract and universal only."

  - name: moral_abstract
    prompt: "Given a moral from a fable, strip it to its most concise and abstract form. Output a single sentence of at most 10 words."

retrieval_configs:
  - name: A
    corpus_variant: ground_truth_style
    use_expansion: false

  - name: B
    corpus_variant: declarative_universal
    use_expansion: false

  - name: A_expand
    corpus_variant: ground_truth_style
    use_expansion: true
    expansion_variants: [moral_rephrase, moral_elaborate, moral_abstract]

  - name: B_expand
    corpus_variant: declarative_universal
    use_expansion: true
    expansion_variants: [moral_rephrase, moral_elaborate, moral_abstract]

  - name: RRF_all
    fusion: rrf
    source_configs: [A, B, A_expand, B_expand]
    k: 60

baseline:
  path: experiments/07_sota_summarization_oracle/results/generation_runs/full_709/golden_summaries.json
  variant: conceptual_abstract
```

- [ ] **Step 2: Create `experiments/08_symmetric_moral_matching/run_pipeline.py`**

```python
"""
run_pipeline.py — Generic pipeline entry point for exp08.

Usage:
  python experiments/08_symmetric_moral_matching/run_pipeline.py
  python experiments/08_symmetric_moral_matching/run_pipeline.py --run-dir path/to/run_dir
  python experiments/08_symmetric_moral_matching/run_pipeline.py --force
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lib.pipeline import run_experiment

parser = argparse.ArgumentParser(description="Run exp08 via generic pipeline")
parser.add_argument("--run-dir", type=Path, default=None,
                    help="Existing run dir to continue (default: create new)")
parser.add_argument("--force", action="store_true",
                    help="Re-run steps even if output already exists")
args = parser.parse_args()

run_experiment(
    config_path=Path(__file__).parent / "config.yaml",
    run_dir=args.run_dir,
    force=args.force,
)
```

- [ ] **Step 3: Smoke-test config loading (no API calls)**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from lib.pipeline import load_config
from pathlib import Path
cfg = load_config(Path('experiments/08_symmetric_moral_matching/config.yaml'))
print('n_fables:', cfg['n_fables'])
print('corpus_variants:', [v['name'] for v in cfg['corpus_variants']])
print('query_expansion_variants:', [v['name'] for v in cfg['query_expansion_variants']])
print('retrieval_configs:', [r['name'] for r in cfg['retrieval_configs']])
print('baseline variant:', cfg['baseline']['variant'])
for v in cfg['corpus_variants']:
    assert 'system_prompt' in v, f'Missing system_prompt for {v[\"name\"]}'
    assert 'user_prompt_template' in v
print('All prompts resolved OK')
"
```
Expected output:
```
n_fables: 10
corpus_variants: ['ground_truth_style', 'declarative_universal']
query_expansion_variants: ['moral_rephrase', 'moral_elaborate', 'moral_abstract']
retrieval_configs: ['A', 'B', 'A_expand', 'B_expand', 'RRF_all']
baseline variant: conceptual_abstract
All prompts resolved OK
```

- [ ] **Step 4: Run full pipeline test suite to confirm all green**

```bash
python -m pytest tests/ -v --ignore=tests/test_exp08_run.py
```
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/08_symmetric_moral_matching/config.yaml experiments/08_symmetric_moral_matching/run_pipeline.py
git commit -m "feat(exp08): add config.yaml and run_pipeline.py — wires exp08 to generic pipeline"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered in task |
|---|---|
| `lib/pipeline/` with all 6 modules | Tasks 2–9 |
| Prompts: inline / file / key | Task 5 (load_config) |
| `user_prompt_template` per variant | Task 5 |
| `default_config.yaml` with base defaults | Task 4 |
| Deep-merge (lists replace, dicts recurse) | Task 5 |
| `run_manifest.json` for step coupling | Task 2 + Task 9 |
| Idempotent steps (skip if output exists) | Tasks 6, 7, 8 |
| `--force` flag | Tasks 6, 7, 8, 9 |
| `encode_with_cache` wired in | Task 8 |
| `compute_metrics_from_matrix` (full metrics) | Task 1 |
| Variant name validation at load time | Task 8 |
| Expansion missing → fail fast | Task 8 |
| `expansion_variants` explicit per retrieval config | Task 8 |
| `source_configs` explicit for RRF | Task 8 |
| Two model fields (corpus + query expansion) | Task 4 + Task 9 |
| Separate `pipeline_runs/` dir | Task 9 |
| Old exp08 scripts untouched | Not touched |
| exp08 `config.yaml` + `run_pipeline.py` | Task 10 |

All spec requirements are covered. No gaps found.

**Placeholder scan:** No TBDs, TODOs, or "similar to Task N" patterns. All code blocks are complete.

**Type consistency check:** `generate_corpus_summaries` takes `variants: list[dict]` where each dict has `name`, `system_prompt`, `user_prompt_template` — consistent with what `load_config` produces. `generate_query_expansions` same. `run_retrieval_eval` reads `rc["corpus_variant"]`, `rc.get("use_expansion")`, `rc.get("expansion_variants", [])`, `rc.get("fusion")`, `rc.get("source_configs", [])`, `rc.get("k", 60)` — all consistent with config schema in Task 4 and exp08 config in Task 10.
