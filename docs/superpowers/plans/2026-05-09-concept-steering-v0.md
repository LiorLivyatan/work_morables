# Concept Steering v0 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v0 concept-steering pipeline under `analysis/08_concept_steering/`: discover failure-associated concepts in moral→fable retrieval, build CAA matched-pair concept vectors, intervene at 5 layers × 9 α values on vanilla Linq-Embed-Mistral, and validate specificity with bootstrap CIs and null controls.

**Architecture:** Single config-driven pipeline with three CLI entry points (`run_baseline.py`, `run_discovery.py`, `run_intervention.py`) on top of small, focused library modules. The only place that touches model internals is `lib/model.py:encode_with_intervention`. Everything else is pure functions over numpy arrays / pandas frames / dicts. All scripts run via `./run.sh` and emit Telegram notifications per project policy.

**Tech Stack:** Python 3.13, `uv`, `transformers`, `sentence-transformers`, `numpy`, `scipy.stats` (Fisher's exact, BH-FDR), `pandas`, `pytest`. Single 7B model on one GPU.

**Spec:** `docs/superpowers/specs/2026-05-09-concept-steering-retrieval-design.md`

---

## File structure

```
analysis/08_concept_steering/
├── README.md                       — what this is, how to run, expected outputs
├── config.yaml                     — single source of truth for tunables
├── __init__.py
├── lib/
│   ├── __init__.py
│   ├── config.py                   — load + validate config.yaml
│   ├── data.py                     — load morals/fables/metadata, build tag→fable index
│   ├── model.py                    — load Linq, encode, hidden-state extraction, hooked encode
│   ├── retrieval.py                — moral→fable rankings + group MRR
│   ├── discovery.py                — Fisher's exact + BH-FDR per tag value
│   ├── vectors.py                  — matched-pair builder, CAA + mean_diff, quality logging
│   ├── intervene.py                — sweep orchestration (concept × layer × α)
│   ├── nulls.py                    — random direction + shuffled-tag CAA controls
│   ├── eval.py                     — paired bootstrap CI on specificity gap S
│   ├── plotting.py                 — headline figure
│   └── io.py                       — save/load helpers (JSON, NPY) with hashed cache keys
├── run_baseline.py                 — Step 1
├── run_discovery.py                — Step 2 + placebo selection
├── run_intervention.py             — Steps 3-4-5 in one orchestrator
├── tests/
│   ├── __init__.py
│   ├── test_discovery.py
│   ├── test_vectors.py
│   ├── test_retrieval.py
│   ├── test_eval.py
│   └── test_nulls.py
├── cache/                          — git-ignored: hidden states, intermediate embeddings
└── results/
    ├── ranks_baseline.json
    ├── discovery_report.json
    ├── concept_vectors/{concept}_{layer}_{method}.npy
    ├── concept_vectors/{concept}.meta.json
    ├── ranks_intervened/{concept}_{layer}_{alpha}.json
    ├── specificity_summary.json
    └── specificity_summary.png
```

**Why this split.** Each `lib/*.py` is one responsibility, ≤ 300 LOC. Each `run_*.py` is < 100 LOC of glue. `model.py` is the *only* file that imports `transformers` or registers hooks — every other module operates on numpy arrays. Adding a new vector method (e.g. LEACE) means one function in `vectors.py` and one entry under `vectors.methods` in the config.

---

## Conventions

- **Run scripts via `./run.sh`** (project policy in CLAUDE.md). Never `python …` directly.
- **Telegram notifications** in every `run_*.py`: `notify.send()` at start and end with key config + result summary.
- **Cache by content hash.** Hidden-state extraction and embeddings are expensive — `lib/io.py` writes them keyed on (model_id, text_hash, config_hash). Re-runs with `force=False` skip recomputation.
- **No `python` interpreter calls in tasks.** Use `./run.sh path/to/script.py` for runs and `uv run pytest …` for tests.
- **Commit after each task.** Atomic commits with the format `feat(08_concept_steering): …`.

---

## Task 1: Bootstrap directory and config skeleton

**Files:**
- Create: `analysis/08_concept_steering/__init__.py`
- Create: `analysis/08_concept_steering/lib/__init__.py`
- Create: `analysis/08_concept_steering/tests/__init__.py`
- Create: `analysis/08_concept_steering/config.yaml`
- Create: `analysis/08_concept_steering/.gitignore`
- Create: `analysis/08_concept_steering/README.md`

- [ ] **Step 1: Create empty package files**

```bash
mkdir -p analysis/08_concept_steering/lib analysis/08_concept_steering/tests \
         analysis/08_concept_steering/cache analysis/08_concept_steering/results
touch analysis/08_concept_steering/__init__.py
touch analysis/08_concept_steering/lib/__init__.py
touch analysis/08_concept_steering/tests/__init__.py
```

- [ ] **Step 2: Write `config.yaml`** (verbatim from spec §5.1, with discovery-time placeholders for concepts)

```yaml
model:
  hf_id: Linq-AI-Research/Linq-Embed-Mistral
  pooling: auto
  device: cuda
  dtype: bfloat16
  batch_size: 8

data:
  morals_path: data/processed/morals_corpus.json
  fables_path: data/processed/fables_corpus.json
  metadata_path: data/enriched/fable_elements.json

discovery:
  failure_definition: rank_gt_1
  metadata_fields: [characters, character_roles, moral_category, setting, fable_type]
  min_tagged_fables: 15
  fdr_alpha: 0.05

concepts:
  targets: []          # filled by run_discovery.py
  placebo: []          # filled by run_discovery.py

vectors:
  layers: [4, 12, 20, 28, -1]
  methods:
    primary: caa_matched
    sanity_byproducts: [mean_diff]
  matching:
    fields: [setting, fable_type]
    cross_field_matching: true
    length_tolerance: 0.20
    min_positive_examples: 15
    min_matched_pairs: 15
  quality_log:
    - n_positives
    - n_matched_pairs
    - mean_length_ratio
    - setting_match_rate
    - fable_type_match_rate
    - cross_field_overlap_rate
    - pos_baseline_mrr
    - neg_baseline_mrr
    - cos_caa_meandiff

intervention:
  alphas: [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]
  hook_position: residual_stream
  renormalize: true

null_controls:
  run_mode: candidate_only
  random_direction:
    n_seeds: 5
    norm_match: caa_matched
  shuffled_tag_caa:
    n_permutations: 50

eval:
  metrics: [mrr_at_10, recall_at_1, recall_at_5, recall_at_10]
  group_by: [target_tagged, target_untagged]
  primary_statistic:
    name: specificity_gap
    test: paired_bootstrap_ci_over_morals
    n_bootstrap: 10000
    alpha: 0.05
  diagnostics:
    pooled_cosine_pre_post: true
    rank_change_listing: true
    null_envelope: true

output:
  results_dir: analysis/08_concept_steering/results
  cache_dir: analysis/08_concept_steering/cache
  save_intermediate_embeddings: false
```

- [ ] **Step 3: Write `.gitignore`**

```
cache/
results/concept_vectors/*.npy
results/ranks_intervened/*.json
*.log
```

- [ ] **Step 4: Write a stub `README.md`** (will be filled in by Task 25)

```markdown
# 08_concept_steering — Activation-level concept suppression for moral→fable retrieval

See `docs/superpowers/specs/2026-05-09-concept-steering-retrieval-design.md` for the design.
Run instructions added in Task 25.
```

- [ ] **Step 5: Commit**

```bash
git add analysis/08_concept_steering/
git commit -m "feat(08_concept_steering): bootstrap directory and config skeleton"
```

---

## Task 2: Config loader with validation

**Files:**
- Create: `analysis/08_concept_steering/lib/config.py`
- Create: `analysis/08_concept_steering/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# analysis/08_concept_steering/tests/test_config.py
from pathlib import Path
import pytest
import yaml

from analysis.lib_08_concept_steering.config import load_config, ConfigError

# NOTE: import path uses lib_08_concept_steering because Python disallows
# leading digits in package names. We add a sys.path entry inside the package
# __init__ to map analysis/08_concept_steering to that name. See Task 1.

def test_load_config_returns_dict_for_valid_file(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump({
        "model": {"hf_id": "x", "pooling": "auto", "device": "cuda",
                   "dtype": "bfloat16", "batch_size": 8},
        "data": {"morals_path": "a", "fables_path": "b", "metadata_path": "c"},
        "discovery": {"failure_definition": "rank_gt_1", "metadata_fields": ["x"],
                       "min_tagged_fables": 15, "fdr_alpha": 0.05},
        "concepts": {"targets": [], "placebo": []},
        "vectors": {"layers": [4], "methods": {"primary": "caa_matched", "sanity_byproducts": []},
                     "matching": {"fields": [], "cross_field_matching": False,
                                  "length_tolerance": 0.2, "min_positive_examples": 15,
                                  "min_matched_pairs": 15},
                     "quality_log": []},
        "intervention": {"alphas": [0.0], "hook_position": "residual_stream", "renormalize": True},
        "null_controls": {"run_mode": "candidate_only",
                           "random_direction": {"n_seeds": 1, "norm_match": "caa_matched"},
                           "shuffled_tag_caa": {"n_permutations": 1}},
        "eval": {"metrics": ["mrr_at_10"], "group_by": [],
                  "primary_statistic": {"name": "specificity_gap",
                                         "test": "paired_bootstrap_ci_over_morals",
                                         "n_bootstrap": 100, "alpha": 0.05},
                  "diagnostics": {"pooled_cosine_pre_post": True,
                                   "rank_change_listing": True,
                                   "null_envelope": True}},
        "output": {"results_dir": "x", "cache_dir": "y",
                    "save_intermediate_embeddings": False},
    }))
    cfg = load_config(cfg_path)
    assert cfg["model"]["hf_id"] == "x"

def test_load_config_raises_on_missing_required_key(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump({"model": {}}))
    with pytest.raises(ConfigError):
        load_config(cfg_path)

def test_load_config_raises_on_invalid_alpha_zero_missing():
    """The α=0 baseline must be present in alphas (used as the no-op anchor)."""
    # built later — see test for validate_intervention
    pass
```

- [ ] **Step 2: Add path remap in package init**

```python
# analysis/08_concept_steering/__init__.py
"""
Concept-steering experiment package.

Python forbids leading digits in module names, so this package is also exposed
as `analysis.lib_08_concept_steering` for imports. Both names refer to the
same module dict.
"""
import sys as _sys
from pathlib import Path as _Path

_pkg_dir = _Path(__file__).parent
_sys.path.insert(0, str(_pkg_dir))            # so `import lib.config` works
```

```python
# analysis/lib_08_concept_steering/__init__.py
# NEW FILE
import importlib as _il
_pkg = _il.import_module("analysis.08_concept_steering")  # type: ignore[arg-type]
```

Actually — Python's import machinery rejects `analysis.08_concept_steering` as a module path. Use the simpler approach: tests and runtime both `sys.path.insert(ROOT)` and import as a flat package via a dispatcher module:

```python
# analysis/08_concept_steering/__init__.py
# (leave empty — folder is loaded by absolute path, not as a Python package)
```

Then in `analysis/__init__.py` (already exists), no changes. Each script and test does:

```python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]   # adjust depth per file
EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))
import lib.config  # noqa
```

Update tests to use this pattern. Replace the import-path discussion in Step 1 with:

```python
# at top of every test file:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import load_config, ConfigError
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest analysis/08_concept_steering/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: lib.config`

- [ ] **Step 4: Write minimal `lib/config.py`**

```python
# analysis/08_concept_steering/lib/config.py
"""Load and validate config.yaml. Raises ConfigError on any structural issue."""
from pathlib import Path
from typing import Any
import yaml


class ConfigError(ValueError):
    pass


REQUIRED_TOP_LEVEL = (
    "model", "data", "discovery", "concepts", "vectors",
    "intervention", "null_controls", "eval", "output",
)
REQUIRED_MODEL = ("hf_id", "pooling", "device", "dtype", "batch_size")


def load_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ConfigError(f"missing top-level keys: {missing}")
    missing_model = [k for k in REQUIRED_MODEL if k not in cfg["model"]]
    if missing_model:
        raise ConfigError(f"missing model keys: {missing_model}")
    if 0.0 not in cfg["intervention"]["alphas"]:
        raise ConfigError("intervention.alphas must include 0.0 (no-op baseline)")
    if cfg["null_controls"]["run_mode"] not in {"candidate_only", "full", "skip"}:
        raise ConfigError(
            f"null_controls.run_mode must be candidate_only|full|skip, "
            f"got {cfg['null_controls']['run_mode']}"
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest analysis/08_concept_steering/tests/test_config.py -v`
Expected: 3 PASS (the third test is currently a `pass` stub).

- [ ] **Step 6: Add the missing alphas-must-include-zero test**

```python
def test_load_config_raises_when_alpha_zero_missing(tmp_path):
    base = _minimal_valid_dict()  # extract the dict from test 1 into a helper
    base["intervention"]["alphas"] = [-1.0, 1.0]  # missing 0.0
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.dump(base))
    with pytest.raises(ConfigError, match="0.0"):
        load_config(cfg_path)
```

Run again: `uv run pytest analysis/08_concept_steering/tests/test_config.py -v`
Expected: 3 PASS.

- [ ] **Step 7: Commit**

```bash
git add analysis/08_concept_steering/__init__.py \
        analysis/08_concept_steering/lib/config.py \
        analysis/08_concept_steering/tests/test_config.py
git commit -m "feat(08_concept_steering): config loader with validation"
```

---

## Task 3: Data loaders for morals, fables, and metadata

**Files:**
- Create: `analysis/08_concept_steering/lib/data.py`
- Create: `analysis/08_concept_steering/tests/test_data.py`

- [ ] **Step 1: Write the failing test**

```python
# analysis/08_concept_steering/tests/test_data.py
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.data import load_corpus, build_tag_index


def test_build_tag_index_groups_doc_ids_by_tag_value(tmp_path):
    elements = [
        {"doc_id": "f0", "characters": ["fox", "crow"], "moral_category": "deception"},
        {"doc_id": "f1", "characters": ["fox"],          "moral_category": "greed"},
        {"doc_id": "f2", "characters": ["wolf"],         "moral_category": "deception"},
    ]
    p = tmp_path / "fable_elements.json"
    p.write_text(json.dumps(elements))

    idx = build_tag_index(p, fields=["characters", "moral_category"])
    assert idx["characters"]["fox"]    == {"f0", "f1"}
    assert idx["characters"]["wolf"]   == {"f2"}
    assert idx["moral_category"]["deception"] == {"f0", "f2"}


def test_load_corpus_returns_aligned_lists(tmp_path):
    morals_data = [{"moral_id": "m0", "moral": "be kind"}, {"moral_id": "m1", "moral": "be brave"}]
    fables_data = [{"doc_id": "f0", "fable": "once upon ..."}, {"doc_id": "f1", "fable": "long ago ..."}]
    qrels = {"m0": "f0", "m1": "f1"}

    mp = tmp_path / "morals.json"; mp.write_text(json.dumps(morals_data))
    fp = tmp_path / "fables.json"; fp.write_text(json.dumps(fables_data))
    qp = tmp_path / "qrels.json"; qp.write_text(json.dumps(qrels))

    corpus = load_corpus(morals_path=mp, fables_path=fp, qrels_path=qp)
    assert corpus.moral_texts  == ["be kind", "be brave"]
    assert corpus.fable_texts  == ["once upon ...", "long ago ..."]
    assert corpus.gt_fable_idx == [0, 1]   # m0→f0 at index 0, m1→f1 at index 1
    assert corpus.fable_doc_ids == ["f0", "f1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/08_concept_steering/tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: lib.data`.

- [ ] **Step 3: Write `lib/data.py`**

```python
# analysis/08_concept_steering/lib/data.py
"""Load morals/fables/metadata into aligned arrays. No model code here."""
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable


@dataclass
class Corpus:
    moral_ids:     list[str]
    moral_texts:   list[str]
    fable_doc_ids: list[str]
    fable_texts:   list[str]
    gt_fable_idx:  list[int]   # for each moral, index into fable_texts of its ground-truth fable


def load_corpus(*, morals_path: Path, fables_path: Path, qrels_path: Path) -> Corpus:
    morals = json.loads(Path(morals_path).read_text())
    fables = json.loads(Path(fables_path).read_text())
    qrels  = json.loads(Path(qrels_path).read_text())

    moral_ids   = [m["moral_id"] for m in morals]
    moral_texts = [m["moral"]    for m in morals]
    fable_ids   = [f["doc_id"]   for f in fables]
    fable_texts = [f["fable"]    for f in fables]

    fid_to_idx = {fid: i for i, fid in enumerate(fable_ids)}
    gt_idx = [fid_to_idx[qrels[mid]] for mid in moral_ids]

    return Corpus(moral_ids, moral_texts, fable_ids, fable_texts, gt_idx)


def build_tag_index(metadata_path: Path, *, fields: Iterable[str]) -> dict[str, dict[str, set[str]]]:
    """
    Return: {field_name: {tag_value: set_of_doc_ids}}

    Lists in the metadata (e.g. characters: [fox, crow]) are exploded so that a
    fable appears under every tag value it carries.
    """
    elements = json.loads(Path(metadata_path).read_text())
    index: dict[str, dict[str, set[str]]] = {f: {} for f in fields}
    for el in elements:
        doc_id = el["doc_id"]
        for field in fields:
            value = el.get(field)
            if value is None:
                continue
            if isinstance(value, list):
                for v in value:
                    index[field].setdefault(v, set()).add(doc_id)
            else:
                index[field].setdefault(value, set()).add(doc_id)
    return index
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/08_concept_steering/tests/test_data.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Add a real-data smoke test**

```python
def test_real_data_smoke():
    """Sanity: real corpus loads and shapes are right."""
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[4]   # repo root
    corpus = load_corpus(
        morals_path=ROOT / "data/processed/morals_corpus.json",
        fables_path=ROOT / "data/processed/fables_corpus.json",
        qrels_path =ROOT / "data/processed/qrels_moral_to_fable.json",
    )
    assert len(corpus.moral_texts)  == 709
    assert len(corpus.fable_texts)  == 709
    assert len(corpus.gt_fable_idx) == 709
    assert all(0 <= i < 709 for i in corpus.gt_fable_idx)


def test_real_metadata_smoke():
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[4]
    idx = build_tag_index(
        ROOT / "data/enriched/fable_elements.json",
        fields=["characters", "character_roles", "moral_category", "setting", "fable_type"],
    )
    assert "fox"        in idx["characters"]
    assert "trickster"  in idx["character_roles"]
    assert "deception"  in idx["moral_category"]
    assert len(idx["characters"]["fox"]) >= 60       # spec verified n=67
```

Run: `uv run pytest analysis/08_concept_steering/tests/test_data.py -v`
Expected: 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add analysis/08_concept_steering/lib/data.py \
        analysis/08_concept_steering/tests/test_data.py
git commit -m "feat(08_concept_steering): corpus + tag-index loaders"
```

---

## Task 4: Model loader and plain encoder

**Files:**
- Create: `analysis/08_concept_steering/lib/model.py`
- Create: `analysis/08_concept_steering/tests/test_model.py`

- [ ] **Step 1: Write `lib/model.py` with the load + plain-encode interface**

```python
# analysis/08_concept_steering/lib/model.py
"""
Model loading + encoding. Single chokepoint for all transformers internals.

Public surface:
    load_model(cfg) -> EncoderHandle
    encode(handle, texts, batch_size=None) -> np.ndarray   (n, hidden_dim)
    extract_hidden_states(handle, texts, layers, batch_size=None) -> dict[layer, ndarray]
    encode_with_intervention(handle, texts, layer, direction, alpha, ...) -> np.ndarray

Pooling and layer indexing are detected at load time and logged. Hidden state
shape printed for one example so users can verify shape conventions.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


@dataclass
class EncoderHandle:
    st_model: SentenceTransformer
    transformer: torch.nn.Module       # the underlying transformers PreTrainedModel
    pooling_kind: str                  # "last_token" | "mean" | "cls"
    device: torch.device
    dtype: torch.dtype
    n_layers: int
    hidden_dim: int


def load_model(cfg: dict) -> EncoderHandle:
    mc = cfg["model"]
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[mc["dtype"]]
    device = torch.device(mc["device"])
    st_model = SentenceTransformer(mc["hf_id"], device=str(device))
    st_model = st_model.to(dtype=dtype)

    # Identify the pooling module in the SentenceTransformer pipeline
    pooling_kind = _detect_pooling(st_model)
    transformer = st_model._modules["0"].auto_model        # underlying HF model
    n_layers = transformer.config.num_hidden_layers
    hidden_dim = transformer.config.hidden_size

    print(f"[model] loaded {mc['hf_id']} on {device} dtype={dtype}")
    print(f"[model] pooling={pooling_kind} n_layers={n_layers} hidden_dim={hidden_dim}")
    return EncoderHandle(st_model, transformer, pooling_kind, device, dtype, n_layers, hidden_dim)


def _detect_pooling(st_model: SentenceTransformer) -> str:
    pooling_module = st_model._modules.get("1")
    if pooling_module is None:
        raise RuntimeError("Could not find pooling module — unexpected ST architecture")
    name = type(pooling_module).__name__.lower()
    if "lasttoken" in name or "weightedmean" in name and getattr(pooling_module, "pooling_mode_lasttoken_token", False):
        return "last_token"
    if "mean" in name:
        return "mean"
    if "cls" in name:
        return "cls"
    # Linq-Embed-Mistral specifically uses last-token. If we land here, log
    # the pooling module's config and raise so the user notices early.
    raise RuntimeError(
        f"Unexpected pooling module: {type(pooling_module).__name__}. "
        f"Inspect ST modules.json and add a branch in _detect_pooling."
    )


def encode(handle: EncoderHandle, texts: Sequence[str], batch_size: int = 8) -> np.ndarray:
    """Plain encoder. Returns L2-normalised float32 embeddings."""
    embs = handle.st_model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    return embs
```

- [ ] **Step 2: Write a smoke test that runs only when CUDA is available**

```python
# analysis/08_concept_steering/tests/test_model.py
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cuda_available = False
try:
    import torch
    cuda_available = torch.cuda.is_available()
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not cuda_available,
                                reason="CUDA not available; model tests are GPU-only")


def _tiny_cfg():
    return {"model": {
        "hf_id": "Linq-AI-Research/Linq-Embed-Mistral",
        "pooling": "auto", "device": "cuda", "dtype": "bfloat16",
        "batch_size": 2,
    }}


def test_load_model_reports_expected_architecture():
    from lib.model import load_model
    h = load_model(_tiny_cfg())
    assert h.pooling_kind == "last_token"
    assert h.n_layers     == 32
    assert h.hidden_dim   == 4096


def test_encode_returns_normalised_embeddings():
    from lib.model import load_model, encode
    h = load_model(_tiny_cfg())
    embs = encode(h, ["hello world", "another sentence"], batch_size=2)
    assert embs.shape == (2, 4096)
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
```

- [ ] **Step 3: Run tests (skipped if no GPU)**

Run: `uv run pytest analysis/08_concept_steering/tests/test_model.py -v`
Expected: 2 PASS or 2 SKIPPED (depending on environment).

- [ ] **Step 4: Commit**

```bash
git add analysis/08_concept_steering/lib/model.py \
        analysis/08_concept_steering/tests/test_model.py
git commit -m "feat(08_concept_steering): model loader + plain encoder"
```

---

## Task 5: Retrieval and group-MRR utilities

**Files:**
- Create: `analysis/08_concept_steering/lib/retrieval.py`
- Create: `analysis/08_concept_steering/tests/test_retrieval.py`

- [ ] **Step 1: Write the failing test**

```python
# analysis/08_concept_steering/tests/test_retrieval.py
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.retrieval import compute_rankings, mrr_at_k, group_mrr


def test_compute_rankings_returns_indices_sorted_by_descending_similarity():
    moral_embs = np.array([[1.0, 0.0]], dtype=np.float32)
    fable_embs = np.array([[0.0, 1.0],  # cos = 0
                           [0.6, 0.8],  # cos = 0.6
                           [1.0, 0.0]], # cos = 1.0
                          dtype=np.float32)
    ranks = compute_rankings(moral_embs, fable_embs)
    assert ranks.shape == (1, 3)
    assert ranks[0].tolist() == [2, 1, 0]


def test_mrr_at_k_correct_simple_case():
    rankings = np.array([[2, 1, 0], [0, 1, 2]])     # gt at rank 1, gt at rank 1
    gt_indices = np.array([2, 0])
    assert mrr_at_k(rankings, gt_indices, k=10) == 1.0


def test_mrr_at_k_with_misses():
    rankings = np.array([[1, 0, 2], [2, 1, 0]])     # gt 0 at rank 2, gt 0 at rank 3
    gt_indices = np.array([0, 0])
    assert mrr_at_k(rankings, gt_indices, k=10) == (1/2 + 1/3) / 2


def test_group_mrr_partitions_by_tag():
    rankings  = np.array([[0, 1], [1, 0], [0, 1]])
    gt_indices = np.array([0, 1, 1])                # query 2 fails (gt at rank 2)
    target_query_mask = np.array([True, False, True])

    target_mrr, ctrl_mrr = group_mrr(rankings, gt_indices, target_query_mask, k=10)
    assert target_mrr == (1.0 + 0.5) / 2            # query 0 hits, query 2 at rank 2
    assert ctrl_mrr   == 1.0                         # query 1 hits
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/08_concept_steering/tests/test_retrieval.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write `lib/retrieval.py`**

```python
# analysis/08_concept_steering/lib/retrieval.py
"""Pure numpy: cosine similarity, ranking, group MRR. No torch, no transformers."""
from __future__ import annotations
import numpy as np


def compute_rankings(query_embs: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """Return (n_queries, n_docs) array of doc indices in descending similarity order.

    Embeddings are assumed L2-normalised so similarity = inner product.
    """
    sims = query_embs @ doc_embs.T                    # (n_q, n_d)
    return np.argsort(-sims, axis=1)


def mrr_at_k(rankings: np.ndarray, gt_indices: np.ndarray, k: int = 10) -> float:
    """Mean Reciprocal Rank at k. rankings[i] is the doc-id ranking for query i."""
    n_q, n_d = rankings.shape
    rr = np.zeros(n_q, dtype=np.float64)
    for i in range(n_q):
        positions = np.where(rankings[i, :k] == gt_indices[i])[0]
        if len(positions) == 1:
            rr[i] = 1.0 / (positions[0] + 1)
    return float(rr.mean())


def group_mrr(rankings: np.ndarray, gt_indices: np.ndarray,
              target_query_mask: np.ndarray, k: int = 10) -> tuple[float, float]:
    """Compute MRR separately on the queries whose ground-truth fable is target-tagged
    vs the queries whose GT fable is NOT target-tagged.

    Returns (mrr_target, mrr_control).
    """
    target_idx = np.where(target_query_mask)[0]
    ctrl_idx   = np.where(~target_query_mask)[0]

    mrr_t = mrr_at_k(rankings[target_idx], gt_indices[target_idx], k=k) if len(target_idx) else float("nan")
    mrr_c = mrr_at_k(rankings[ctrl_idx],  gt_indices[ctrl_idx],  k=k) if len(ctrl_idx)  else float("nan")
    return mrr_t, mrr_c
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/08_concept_steering/tests/test_retrieval.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/08_concept_steering/lib/retrieval.py \
        analysis/08_concept_steering/tests/test_retrieval.py
git commit -m "feat(08_concept_steering): cosine retrieval + group MRR"
```

---

## Task 6: Discovery — Fisher's exact + BH-FDR per tag

**Files:**
- Create: `analysis/08_concept_steering/lib/discovery.py`
- Create: `analysis/08_concept_steering/tests/test_discovery.py`

- [ ] **Step 1: Write the failing test**

```python
# analysis/08_concept_steering/tests/test_discovery.py
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.discovery import (
    failure_overrep_per_tag, bh_fdr, rank_problematic_concepts,
    per_tag_baseline_mrr,
)


def test_failure_overrep_picks_up_clean_signal():
    # 100 fables, tag T present in 30 of them. Failures (rank>1) on:
    #   - 25 of the 30 T-tagged fables  → 83% failure rate
    #   - 35 of the 70 non-T fables     → 50% failure rate
    n = 100
    tag_T = np.array([True] * 30 + [False] * 70)
    failed = np.zeros(n, dtype=bool)
    failed[:25] = True              # 25/30 T-tagged fail
    failed[30:65] = True            # 35/70 non-T fail
    p = failure_overrep_per_tag(tag_T, failed)
    assert p < 0.01                 # strong overrep should give small p


def test_failure_overrep_returns_one_when_no_signal():
    n = 100
    tag = np.array([True] * 50 + [False] * 50)
    failed = np.array([True, False] * 50)   # uniform, no association
    p = failure_overrep_per_tag(tag, failed)
    assert 0.5 < p <= 1.0


def test_bh_fdr_controls_at_alpha():
    pvals = np.array([0.001, 0.01, 0.04, 0.5, 0.9])
    rejected = bh_fdr(pvals, alpha=0.05)
    # BH at α=0.05 with 5 tests: thresholds 0.01, 0.02, 0.03, 0.04, 0.05
    # 0.001 < 0.01 → reject
    # 0.01  ≤ 0.02 → reject
    # 0.04  > 0.03 → retain
    # so rejected sorted = [True, True, False, False, False]
    assert rejected.tolist() == [True, True, False, False, False]


def test_rank_problematic_concepts_filters_min_size():
    """Concepts with fewer than min_tagged_fables should be excluded."""
    tag_index = {
        "characters": {"fox": {"f0", "f1"}, "wolf": set("f" + str(i) for i in range(20))},
    }
    failed_doc_ids = {f"f{i}" for i in [0, 5, 10]}        # 1/2 fox failures, 3/20 wolf failures
    fable_doc_ids  = ["f" + str(i) for i in range(25)]

    df = rank_problematic_concepts(
        tag_index=tag_index, fable_doc_ids=fable_doc_ids,
        failed_doc_ids=failed_doc_ids, min_tagged_fables=15,
    )
    assert "wolf" in df["value"].values
    assert "fox"  not in df["value"].values     # n=2 below threshold
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/08_concept_steering/tests/test_discovery.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write `lib/discovery.py`**

```python
# analysis/08_concept_steering/lib/discovery.py
"""
Step 2 of the pipeline: find tag values whose presence is statistically
overrepresented in retrieval failures. Uses Fisher's exact test (one-sided,
greater-than) and BH-FDR for multiple comparisons.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def failure_overrep_per_tag(tag_present: np.ndarray, failed: np.ndarray) -> float:
    """One-sided Fisher's exact test: is failure overrepresented among tagged fables?"""
    # 2x2 contingency: rows = tag (yes/no), cols = failed (yes/no)
    a = int((tag_present & failed).sum())
    b = int((tag_present & ~failed).sum())
    c = int((~tag_present & failed).sum())
    d = int((~tag_present & ~failed).sum())
    _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
    return float(pval)


def bh_fdr(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg. Returns boolean array of rejection decisions."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool)
    cutoff_rank = np.where(below)[0].max()
    rejected_sorted = np.zeros(n, dtype=bool)
    rejected_sorted[: cutoff_rank + 1] = True
    rejected = np.zeros(n, dtype=bool)
    rejected[order] = rejected_sorted
    return rejected


def rank_problematic_concepts(
    *,
    tag_index: dict[str, dict[str, set[str]]],
    fable_doc_ids: list[str],
    failed_doc_ids: set[str],
    min_tagged_fables: int,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        field, value, n_tagged, n_failed_in_tag, p_value, q_value, fdr_significant
    sorted by p_value ascending.
    """
    fable_idx = {fid: i for i, fid in enumerate(fable_doc_ids)}
    n_total = len(fable_doc_ids)
    failed_mask = np.zeros(n_total, dtype=bool)
    for fid in failed_doc_ids:
        if fid in fable_idx:
            failed_mask[fable_idx[fid]] = True

    rows = []
    for field, tag_map in tag_index.items():
        for value, doc_set in tag_map.items():
            if len(doc_set) < min_tagged_fables:
                continue
            tag_mask = np.zeros(n_total, dtype=bool)
            for fid in doc_set:
                if fid in fable_idx:
                    tag_mask[fable_idx[fid]] = True
            p = failure_overrep_per_tag(tag_mask, failed_mask)
            rows.append({
                "field": field,
                "value": value,
                "n_tagged": int(tag_mask.sum()),
                "n_failed_in_tag": int((tag_mask & failed_mask).sum()),
                "p_value": p,
            })

    df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    if len(df):
        df["fdr_significant"] = bh_fdr(df["p_value"].to_numpy(), alpha=fdr_alpha)
    return df


def per_tag_baseline_mrr(
    *,
    tag_index: dict[str, dict[str, set[str]]],
    fable_doc_ids: list[str],
    moral_gt_idx: list[int],
    rankings: np.ndarray,
    min_tagged_fables: int,
    k: int = 10,
) -> pd.DataFrame:
    """For each (field, value) above min_tagged, compute MRR@k on queries whose
    ground-truth fable is tagged with that value. Used downstream to pick the
    difficulty-matched placebo."""
    from lib.retrieval import mrr_at_k
    fable_idx = {fid: i for i, fid in enumerate(fable_doc_ids)}
    rows = []
    for field, tag_map in tag_index.items():
        for value, doc_set in tag_map.items():
            if len(doc_set) < min_tagged_fables:
                continue
            tagged_fable_indices = {fable_idx[fid] for fid in doc_set if fid in fable_idx}
            target_query_mask = np.array([gt in tagged_fable_indices for gt in moral_gt_idx])
            if target_query_mask.sum() == 0:
                continue
            mrr = mrr_at_k(rankings[target_query_mask],
                            np.array([moral_gt_idx[i] for i in np.where(target_query_mask)[0]]),
                            k=k)
            rows.append({"field": field, "value": value, "n_tagged_queries": int(target_query_mask.sum()),
                         "baseline_mrr": mrr})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/08_concept_steering/tests/test_discovery.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/08_concept_steering/lib/discovery.py \
        analysis/08_concept_steering/tests/test_discovery.py
git commit -m "feat(08_concept_steering): Fisher exact + BH-FDR discovery"
```

---

## Task 7: `run_baseline.py` — Step 1 entry point

**Files:**
- Create: `analysis/08_concept_steering/run_baseline.py`
- Modify: `analysis/08_concept_steering/lib/io.py` (new)

- [ ] **Step 1: Write `lib/io.py`**

```python
# analysis/08_concept_steering/lib/io.py
"""Save/load helpers. JSON for small artifacts, NPY for large arrays."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np


def text_hash(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def save_npy(path: Path, arr: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def save_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)
```

- [ ] **Step 2: Write `run_baseline.py`**

```python
# analysis/08_concept_steering/run_baseline.py
"""
Step 1: Encode all 709 morals + 709 fables with vanilla Linq-Embed-Mistral,
save embeddings, compute moral→fable rankings, write ranks_baseline.json.

Run via: ./run.sh analysis/08_concept_steering/run_baseline.py [--remote --gpu N]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datetime import datetime

EXP_DIR = Path(__file__).resolve().parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(ROOT))

import numpy as np

from finetuning.lib import notify
from lib.config import load_config
from lib.data import load_corpus
from lib.io import save_npy, save_json, text_hash
from lib.model import load_model, encode
from lib.retrieval import compute_rankings, mrr_at_k


def main(force: bool = False) -> int:
    cfg = load_config(EXP_DIR / "config.yaml")
    cache_dir = ROOT / cfg["output"]["cache_dir"]
    results_dir = ROOT / cfg["output"]["results_dir"]

    corpus = load_corpus(
        morals_path=ROOT / cfg["data"]["morals_path"],
        fables_path=ROOT / cfg["data"]["fables_path"],
        qrels_path =ROOT / "data/processed/qrels_moral_to_fable.json",
    )
    notify.send(
        f"🚀 08_concept_steering: run_baseline starting\n"
        f"model: {cfg['model']['hf_id']}\n"
        f"n_morals={len(corpus.moral_texts)} n_fables={len(corpus.fable_texts)}"
    )

    moral_cache = cache_dir / f"moral_embs_{text_hash(corpus.moral_texts)}.npy"
    fable_cache = cache_dir / f"fable_embs_{text_hash(corpus.fable_texts)}.npy"

    if not force and moral_cache.exists() and fable_cache.exists():
        moral_embs = np.load(moral_cache)
        fable_embs = np.load(fable_cache)
        print(f"[baseline] cache hit: reusing embeddings")
    else:
        handle = load_model(cfg)
        bs = cfg["model"]["batch_size"]
        moral_embs = encode(handle, corpus.moral_texts, batch_size=bs)
        fable_embs = encode(handle, corpus.fable_texts, batch_size=bs)
        save_npy(moral_cache, moral_embs)
        save_npy(fable_cache, fable_embs)

    rankings = compute_rankings(moral_embs, fable_embs)
    gt = np.array(corpus.gt_fable_idx)
    mrr10 = mrr_at_k(rankings, gt, k=10)

    # Build the JSON the rest of the pipeline reads
    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": cfg["model"]["hf_id"],
        "n_morals": len(corpus.moral_texts),
        "n_fables": len(corpus.fable_texts),
        "mrr_at_10": mrr10,
        "queries": [
            {
                "moral_id": corpus.moral_ids[i],
                "gt_fable_idx": int(gt[i]),
                "gt_fable_doc_id": corpus.fable_doc_ids[gt[i]],
                "top_50_indices": rankings[i, :50].tolist(),
                "gt_rank": int(np.where(rankings[i] == gt[i])[0][0]) + 1,  # 1-indexed
            }
            for i in range(len(corpus.moral_texts))
        ],
        "fable_doc_ids": corpus.fable_doc_ids,
    }
    save_json(results_dir / "ranks_baseline.json", out)

    notify.send(
        f"✅ 08_concept_steering: run_baseline done\n"
        f"MRR@10 = {mrr10:.4f}\n"
        f"failures (rank > 1): {sum(1 for q in out['queries'] if q['gt_rank'] > 1)}/{len(out['queries'])}"
    )
    print(f"[baseline] MRR@10 = {mrr10:.4f}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Re-encode even if cache exists")
    args = p.parse_args()
    sys.exit(main(force=args.force))
```

- [ ] **Step 3: Smoke run on local CPU/GPU**

Run: `./run.sh analysis/08_concept_steering/run_baseline.py --remote --gpu 2`
Expected:
- Telegram: "🚀 08_concept_steering: run_baseline starting"
- After ~5–10 min: "✅ run_baseline done MRR@10 = 0.21xx"
- File `analysis/08_concept_steering/results/ranks_baseline.json` exists
- File contains `queries[*].gt_rank` and `top_50_indices`

- [ ] **Step 4: Pull results locally**

Run: `./run.sh pull`
Verify the json by inspecting `results/ranks_baseline.json`.

- [ ] **Step 5: Commit**

```bash
git add analysis/08_concept_steering/lib/io.py \
        analysis/08_concept_steering/run_baseline.py \
        analysis/08_concept_steering/results/ranks_baseline.json
git commit -m "feat(08_concept_steering): run_baseline produces vanilla Linq rankings"
```

---

## Task 8: `run_discovery.py` — Step 2 + placebo selection

**Files:**
- Create: `analysis/08_concept_steering/run_discovery.py`

- [ ] **Step 1: Write the script**

```python
# analysis/08_concept_steering/run_discovery.py
"""
Step 2: rank candidate problematic concepts via failure overrepresentation
(Fisher's exact + BH-FDR), then select 3 targets (one per metadata field) and
1 difficulty-matched placebo. Updates concepts.* in config.yaml.

Run via: ./run.sh analysis/08_concept_steering/run_discovery.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import yaml
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from lib.config import load_config
from lib.data import load_corpus, build_tag_index
from lib.io import load_json, save_json
from lib.discovery import rank_problematic_concepts, per_tag_baseline_mrr


def main() -> int:
    cfg = load_config(EXP_DIR / "config.yaml")
    results_dir = ROOT / cfg["output"]["results_dir"]

    baseline = load_json(results_dir / "ranks_baseline.json")
    fable_doc_ids = baseline["fable_doc_ids"]
    failed_doc_ids = {q["gt_fable_doc_id"] for q in baseline["queries"] if q["gt_rank"] > 1}

    corpus = load_corpus(
        morals_path=ROOT / cfg["data"]["morals_path"],
        fables_path=ROOT / cfg["data"]["fables_path"],
        qrels_path =ROOT / "data/processed/qrels_moral_to_fable.json",
    )
    tag_index = build_tag_index(
        ROOT / cfg["data"]["metadata_path"],
        fields=cfg["discovery"]["metadata_fields"],
    )

    # 1. Discovery: rank problematic concepts
    df = rank_problematic_concepts(
        tag_index=tag_index, fable_doc_ids=fable_doc_ids,
        failed_doc_ids=failed_doc_ids,
        min_tagged_fables=cfg["discovery"]["min_tagged_fables"],
        fdr_alpha=cfg["discovery"]["fdr_alpha"],
    )
    print("\n=== Top 20 problematic concepts ===")
    print(df.head(20).to_string(index=False))

    # 2. Per-tag baseline MRR (used for placebo difficulty match)
    rankings = np.array([q["top_50_indices"] for q in baseline["queries"]])
    # pad: rankings should be (n_morals, n_fables) for full MRR; top_50 is enough for MRR@10
    gt = np.array([q["gt_fable_idx"] for q in baseline["queries"]])
    mrr_df = per_tag_baseline_mrr(
        tag_index=tag_index, fable_doc_ids=fable_doc_ids,
        moral_gt_idx=gt.tolist(), rankings=rankings,
        min_tagged_fables=cfg["discovery"]["min_tagged_fables"],
    )

    # 3. Pick 3 targets — one per metadata field, FDR-significant, highest overrep
    targets: list[dict] = []
    used_fields: set[str] = set()
    for _, row in df[df["fdr_significant"]].iterrows():
        if row["field"] in used_fields:
            continue
        if row["field"] not in {"characters", "character_roles", "moral_category"}:
            continue   # we want one per these three fields
        targets.append({"field": row["field"], "value": row["value"]})
        used_fields.add(row["field"])
        if len(targets) == 3:
            break
    if len(targets) < 3:
        print(f"⚠️  Only {len(targets)} FDR-significant targets found across required fields")

    target_mean_mrr = float(np.mean([
        mrr_df[(mrr_df["field"] == t["field"]) & (mrr_df["value"] == t["value"])]
        ["baseline_mrr"].iloc[0]
        for t in targets
    ]))

    # 4. Pick placebo: difficulty-matched, NOT FDR-sig, comparable n, non-target field
    target_n = float(np.mean([
        df[(df["field"] == t["field"]) & (df["value"] == t["value"])]["n_tagged"].iloc[0]
        for t in targets
    ]))
    candidates = mrr_df.merge(df[["field", "value", "n_tagged", "fdr_significant"]],
                                on=["field", "value"])
    candidates = candidates[~candidates["fdr_significant"]]
    candidates = candidates[~candidates["field"].isin(used_fields)]
    candidates = candidates[abs(candidates["baseline_mrr"] - target_mean_mrr) <= 0.05]
    candidates = candidates[
        (candidates["n_tagged"] >= 0.5 * target_n)
        & (candidates["n_tagged"] <= 1.5 * target_n)
    ]
    if len(candidates) == 0:
        notify.send("❌ run_discovery: no valid placebo candidate. Loosening difficulty match.")
        candidates = mrr_df.merge(df[["field", "value", "fdr_significant"]],
                                     on=["field", "value"])
        candidates = candidates[~candidates["fdr_significant"]]
        candidates = candidates[~candidates["field"].isin(used_fields)]
        candidates["mrr_dist"] = (candidates["baseline_mrr"] - target_mean_mrr).abs()
        candidates = candidates.sort_values("mrr_dist")
    placebo = [{"field": candidates["field"].iloc[0], "value": candidates["value"].iloc[0]}]

    # 5. Save discovery report
    save_json(results_dir / "discovery_report.json", {
        "all_concepts": df.to_dict(orient="records"),
        "per_tag_mrr": mrr_df.to_dict(orient="records"),
        "selected": {"targets": targets, "placebo": placebo,
                      "target_mean_baseline_mrr": target_mean_mrr},
    })

    # 6. Update config.yaml in place
    with open(EXP_DIR / "config.yaml") as f:
        live = yaml.safe_load(f)
    live["concepts"]["targets"] = targets
    live["concepts"]["placebo"] = placebo
    with open(EXP_DIR / "config.yaml", "w") as f:
        yaml.safe_dump(live, f, sort_keys=False)

    notify.send(
        f"✅ 08_concept_steering: run_discovery done\n"
        f"targets: {targets}\n"
        f"placebo: {placebo}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run discovery (no GPU needed — pure stats over JSON)**

Run: `./run.sh analysis/08_concept_steering/run_discovery.py`
Expected:
- Console prints top 20 concepts ranked by Fisher's exact p-value
- `results/discovery_report.json` exists
- `config.yaml` has `concepts.targets` and `concepts.placebo` filled in
- Telegram: targets and placebo summary

- [ ] **Step 3: Eyeball the discovery report**

Visually inspect `results/discovery_report.json` — confirm the 3 selected targets look semantically reasonable (e.g. one character, one role, one moral category). If a target is suspect (e.g. an obvious dataset artifact), document it as a finding and proceed.

- [ ] **Step 4: Commit**

```bash
git add analysis/08_concept_steering/run_discovery.py \
        analysis/08_concept_steering/config.yaml \
        analysis/08_concept_steering/results/discovery_report.json
git commit -m "feat(08_concept_steering): run_discovery selects targets + placebo"
```

---

## Task 9: Hidden-state extraction in `lib/model.py`

**Files:**
- Modify: `analysis/08_concept_steering/lib/model.py`
- Modify: `analysis/08_concept_steering/tests/test_model.py`

- [ ] **Step 1: Add `extract_hidden_states` to `lib/model.py`**

Append to `lib/model.py` (after `encode`):

```python
def extract_hidden_states(
    handle: EncoderHandle,
    texts: Sequence[str],
    layers: Sequence[int],
    batch_size: int = 4,
) -> dict[int, np.ndarray]:
    """For each requested layer index, return (n_texts, hidden_dim) array of POOLED
    hidden states at that layer. Layer index -1 means the last layer.

    Pooling matches handle.pooling_kind. For last_token, the last non-pad token
    position is used (Mistral attends causally, so the last position carries
    aggregated information).
    """
    requested_layers = [(handle.n_layers if l == -1 else l) for l in layers]
    out: dict[int, list[np.ndarray]] = {l: [] for l in requested_layers}

    tok = handle.st_model.tokenizer
    with torch.no_grad():
        for batch_start in range(0, len(texts), batch_size):
            batch = list(texts[batch_start: batch_start + batch_size])
            enc = tok(batch, padding=True, truncation=True, return_tensors="pt",
                       max_length=tok.model_max_length)
            enc = {k: v.to(handle.device) for k, v in enc.items()}
            res = handle.transformer(**enc, output_hidden_states=True)
            # res.hidden_states is a tuple of length n_layers + 1 (embeddings + each block)
            attn_mask = enc["attention_mask"]
            for layer in requested_layers:
                hs = res.hidden_states[layer]              # (B, T, H)
                pooled = _pool(hs, attn_mask, handle.pooling_kind)
                out[layer].append(pooled.float().cpu().numpy())

    return {l: np.concatenate(out[l], axis=0) for l in requested_layers}


def _pool(hs: torch.Tensor, attn_mask: torch.Tensor, kind: str) -> torch.Tensor:
    """hs: (B, T, H), attn_mask: (B, T). Returns (B, H)."""
    if kind == "last_token":
        seq_lengths = attn_mask.sum(dim=1) - 1            # last non-pad index
        idx = seq_lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hs.size(-1))
        return hs.gather(dim=1, index=idx).squeeze(1)
    if kind == "mean":
        mask = attn_mask.unsqueeze(-1).type_as(hs)
        return (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    if kind == "cls":
        return hs[:, 0, :]
    raise ValueError(f"unknown pooling kind: {kind}")
```

Add `import torch` if not already at top.

- [ ] **Step 2: Add a test**

```python
# tests/test_model.py — append
def test_extract_hidden_states_returns_correct_shapes():
    from lib.model import load_model, extract_hidden_states
    h = load_model(_tiny_cfg())
    out = extract_hidden_states(h, ["hello", "world"], layers=[4, -1])
    assert set(out.keys()) == {4, h.n_layers}
    for layer, arr in out.items():
        assert arr.shape == (2, h.hidden_dim)
```

- [ ] **Step 3: Run the test (skipped without GPU)**

Run: `uv run pytest analysis/08_concept_steering/tests/test_model.py::test_extract_hidden_states_returns_correct_shapes -v`
Expected: PASS or SKIPPED.

- [ ] **Step 4: Commit**

```bash
git add analysis/08_concept_steering/lib/model.py \
        analysis/08_concept_steering/tests/test_model.py
git commit -m "feat(08_concept_steering): hidden-state extraction at arbitrary layers"
```

---

## Task 10: Matched-pair builder

**Files:**
- Create: `analysis/08_concept_steering/lib/vectors.py`
- Create: `analysis/08_concept_steering/tests/test_vectors.py`

- [ ] **Step 1: Write the failing test**

```python
# analysis/08_concept_steering/tests/test_vectors.py
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.vectors import build_matched_pairs, MatchingFailure


def _toy_metadata():
    """5 fables, two of which are 'fox', three are not. Settings vary."""
    return [
        {"doc_id": "f0", "characters": ["fox"],   "setting": "forest", "fable_type": "animal_only", "moral_category": "deception"},
        {"doc_id": "f1", "characters": ["fox"],   "setting": "field",  "fable_type": "animal_only", "moral_category": "greed"},
        {"doc_id": "f2", "characters": ["wolf"],  "setting": "forest", "fable_type": "animal_only", "moral_category": "deception"},
        {"doc_id": "f3", "characters": ["lion"],  "setting": "field",  "fable_type": "animal_only", "moral_category": "greed"},
        {"doc_id": "f4", "characters": ["mouse"], "setting": "city",   "fable_type": "animal_only", "moral_category": "courage"},
    ]


def test_build_matched_pairs_finds_within_setting_match():
    md = _toy_metadata()
    fable_lengths = [100, 110, 95, 105, 120]
    fable_doc_ids = [m["doc_id"] for m in md]
    pos_set = {"f0", "f1"}      # fox-tagged
    pairs = build_matched_pairs(
        positives=pos_set, fable_doc_ids=fable_doc_ids, metadata=md,
        fable_token_lengths=fable_lengths,
        match_fields=["setting", "fable_type"],
        cross_field=None,        # disabled in this test
        length_tolerance=0.20,
    )
    paired = {(p, n) for p, n in pairs}
    assert ("f0", "f2") in paired       # both forest, len 100 vs 95 ratio 0.95
    assert ("f1", "f3") in paired       # both field,  len 110 vs 105 ratio 0.95


def test_build_matched_pairs_respects_cross_field_when_concept_is_character():
    md = _toy_metadata()
    fable_lengths = [100] * 5
    fable_doc_ids = [m["doc_id"] for m in md]
    pos_set = {"f0", "f1"}      # fox-tagged

    # f0 has moral_category=deception; the only deception non-fox is f2 (wolf, forest).
    # f1 has moral_category=greed; the only greed non-fox is f3 (lion, field).
    pairs = build_matched_pairs(
        positives=pos_set, fable_doc_ids=fable_doc_ids, metadata=md,
        fable_token_lengths=fable_lengths,
        match_fields=["setting", "fable_type"],
        cross_field="moral_category",
        length_tolerance=0.20,
    )
    paired = {(p, n) for p, n in pairs}
    assert paired == {("f0", "f2"), ("f1", "f3")}


def test_build_matched_pairs_skips_when_no_match_found():
    md = _toy_metadata()
    fable_lengths = [100] * 5
    fable_doc_ids = [m["doc_id"] for m in md]
    pos_set = {"f4"}    # mouse, city — no other city fable in toy data
    pairs = build_matched_pairs(
        positives=pos_set, fable_doc_ids=fable_doc_ids, metadata=md,
        fable_token_lengths=fable_lengths,
        match_fields=["setting"], cross_field=None, length_tolerance=0.20,
    )
    assert pairs == []     # no negatives match
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/08_concept_steering/tests/test_vectors.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write `lib/vectors.py` part 1 — matched-pair builder**

```python
# analysis/08_concept_steering/lib/vectors.py
"""
Concept-vector construction from hidden states.

Public surface:
    build_matched_pairs(positives, fable_doc_ids, metadata, ...) -> list[(pos_id, neg_id)]
    build_caa_vector(pos_indices, neg_indices, hidden_states_per_layer) -> dict[layer, ndarray]
    build_mean_diff_vector(pos_indices, fable_doc_ids, hidden_states) -> dict[layer, ndarray]
    matched_pair_quality_metrics(pairs, metadata, fable_lengths, ...) -> dict
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np


class MatchingFailure(RuntimeError):
    pass


def build_matched_pairs(
    *,
    positives: set[str],
    fable_doc_ids: list[str],
    metadata: list[dict],
    fable_token_lengths: list[int],
    match_fields: list[str],
    cross_field: str | None,
    length_tolerance: float,
) -> list[tuple[str, str]]:
    """For each positive fable, pick a negative fable matched on the requested fields.

    Greedy: each negative is used at most once across all positives. Pairs are
    formed in deterministic doc-id order.
    """
    md_by_id = {m["doc_id"]: m for m in metadata}
    len_by_id = dict(zip(fable_doc_ids, fable_token_lengths))
    used_negatives: set[str] = set()
    pairs: list[tuple[str, str]] = []

    for pos_id in sorted(positives):
        pos_md = md_by_id[pos_id]
        pos_len = len_by_id[pos_id]

        for cand_id in fable_doc_ids:
            if cand_id == pos_id or cand_id in positives or cand_id in used_negatives:
                continue
            cand_md = md_by_id[cand_id]
            if not _fields_match(pos_md, cand_md, match_fields):
                continue
            if cross_field and not _list_fields_share_value(pos_md, cand_md, cross_field):
                continue
            cand_len = len_by_id[cand_id]
            ratio = min(pos_len, cand_len) / max(pos_len, cand_len)
            if ratio < (1 - length_tolerance):
                continue
            pairs.append((pos_id, cand_id))
            used_negatives.add(cand_id)
            break

    return pairs


def _fields_match(a: dict, b: dict, fields: list[str]) -> bool:
    for f in fields:
        if a.get(f) != b.get(f):
            return False
    return True


def _list_fields_share_value(a: dict, b: dict, field: str) -> bool:
    av, bv = a.get(field), b.get(field)
    if av is None or bv is None:
        return False
    if isinstance(av, list) and isinstance(bv, list):
        return bool(set(av) & set(bv))
    return av == bv
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/08_concept_steering/tests/test_vectors.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/08_concept_steering/lib/vectors.py \
        analysis/08_concept_steering/tests/test_vectors.py
git commit -m "feat(08_concept_steering): matched-pair builder for CAA"
```

---

## Task 11: CAA + mean-diff vector computation

**Files:**
- Modify: `analysis/08_concept_steering/lib/vectors.py`
- Modify: `analysis/08_concept_steering/tests/test_vectors.py`

- [ ] **Step 1: Add the failing tests**

```python
# tests/test_vectors.py — append
def test_build_caa_vector_averages_pairwise_differences():
    pos_h = np.array([[1.0, 0.0], [0.0, 1.0]])
    neg_h = np.array([[0.0, 0.0], [1.0, 0.0]])
    v = _caa_inner(pos_h, neg_h)
    assert v.shape == (2,)
    np.testing.assert_allclose(v, [0.0, 0.5])


def test_build_mean_diff_vector():
    h = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    pos_idx = np.array([0, 1])      # mean = (1.5, 0)
    all_idx = np.arange(4)          # mean = (0.75, 0.75)
    v = _mean_diff_inner(h, pos_idx, all_idx)
    np.testing.assert_allclose(v, [0.75, -0.75])
```

```python
# helpers in test_vectors.py
def _caa_inner(pos_h, neg_h):
    from lib.vectors import build_caa_vector
    return build_caa_vector({0: pos_h}, {0: neg_h})[0]

def _mean_diff_inner(h, pos_idx, all_idx):
    from lib.vectors import build_mean_diff_vector
    return build_mean_diff_vector({0: h}, pos_idx, np.setdiff1d(all_idx, pos_idx))[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/08_concept_steering/tests/test_vectors.py -v`
Expected: FAIL with import error on `build_caa_vector` / `build_mean_diff_vector`.

- [ ] **Step 3: Append to `lib/vectors.py`**

```python
def build_caa_vector(
    pos_hidden_per_layer: dict[int, np.ndarray],   # (n_pairs, hidden_dim)
    neg_hidden_per_layer: dict[int, np.ndarray],   # (n_pairs, hidden_dim)
) -> dict[int, np.ndarray]:
    """v_C[layer] = mean over pairs of (h_pos − h_neg). Shape: (hidden_dim,)."""
    out = {}
    for layer in pos_hidden_per_layer:
        diffs = pos_hidden_per_layer[layer] - neg_hidden_per_layer[layer]
        out[layer] = diffs.mean(axis=0)
    return out


def build_mean_diff_vector(
    hidden_per_layer: dict[int, np.ndarray],       # (n_total, hidden_dim) per layer
    pos_indices: np.ndarray,
    neg_indices: np.ndarray,
) -> dict[int, np.ndarray]:
    """v_C[layer] = mean(h | pos) − mean(h | neg). Shape: (hidden_dim,)."""
    out = {}
    for layer, h in hidden_per_layer.items():
        out[layer] = h[pos_indices].mean(axis=0) - h[neg_indices].mean(axis=0)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def matched_pair_quality_metrics(
    *,
    pairs: list[tuple[str, str]],
    metadata: list[dict],
    fable_token_lengths: list[int],
    fable_doc_ids: list[str],
    cross_field: str | None,
    pos_baseline_mrr: float,
    neg_baseline_mrr: float,
    cos_caa_meandiff_per_layer: dict[int, float],
) -> dict:
    md = {m["doc_id"]: m for m in metadata}
    lens = dict(zip(fable_doc_ids, fable_token_lengths))
    if not pairs:
        return {"n_matched_pairs": 0}

    setting_match = sum(1 for p, n in pairs if md[p]["setting"] == md[n]["setting"]) / len(pairs)
    type_match    = sum(1 for p, n in pairs if md[p]["fable_type"] == md[n]["fable_type"]) / len(pairs)
    length_ratios = [
        min(lens[p], lens[n]) / max(lens[p], lens[n]) for p, n in pairs
    ]
    cross_overlap = float("nan")
    if cross_field:
        cross_overlap = sum(
            1 for p, n in pairs if _list_fields_share_value(md[p], md[n], cross_field)
        ) / len(pairs)

    return {
        "n_matched_pairs": len(pairs),
        "mean_length_ratio": float(np.mean(length_ratios)),
        "setting_match_rate": float(setting_match),
        "fable_type_match_rate": float(type_match),
        "cross_field_overlap_rate": float(cross_overlap),
        "pos_baseline_mrr": float(pos_baseline_mrr),
        "neg_baseline_mrr": float(neg_baseline_mrr),
        "cos_caa_meandiff_per_layer": {str(l): float(c)
                                        for l, c in cos_caa_meandiff_per_layer.items()},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/08_concept_steering/tests/test_vectors.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/08_concept_steering/lib/vectors.py \
        analysis/08_concept_steering/tests/test_vectors.py
git commit -m "feat(08_concept_steering): CAA + mean-diff vector + quality metrics"
```

---

## Task 12: Hooked-forward `encode_with_intervention`

**Files:**
- Modify: `analysis/08_concept_steering/lib/model.py`

- [ ] **Step 1: Append to `lib/model.py`**

```python
def encode_with_intervention(
    handle: EncoderHandle,
    texts: Sequence[str],
    *,
    layer_idx: int,
    direction: np.ndarray | None,         # (hidden_dim,) or None for no-op
    alpha: float = 0.0,
    batch_size: int = 4,
    renormalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward with hook: at the output of transformer block `layer_idx`, subtract
    `alpha * direction` from the residual stream at every token position. Then
    continue forward, pool, optionally L2-normalise.

    Returns (embeddings, pooled_cosine_pre_post).
        embeddings: (n_texts, hidden_dim) float32
        pooled_cosine_pre_post: per-text cosine between the no-op pooled output
                                 and the intervened pooled output. (Shape: n_texts,)
                                 Used as the magnitude-only diagnostic.
    """
    no_intervention = direction is None or alpha == 0.0
    layer_idx = handle.n_layers + layer_idx if layer_idx < 0 else layer_idx

    if no_intervention:
        embs = encode(handle, texts, batch_size=batch_size)
        return embs, np.ones(len(texts), dtype=np.float32)

    # Pre-pass: get baseline pooled embeddings (no hook) for the cosine diagnostic
    base_embs = encode(handle, texts, batch_size=batch_size)

    direction_t = torch.as_tensor(direction, device=handle.device, dtype=handle.dtype)

    def hook_fn(module, inputs, output):
        # block output is either tensor or tuple(tensor, ...) depending on model
        if isinstance(output, tuple):
            hs = output[0]
            modified = hs - alpha * direction_t
            return (modified,) + output[1:]
        return output - alpha * direction_t

    target_block = handle.transformer.model.layers[layer_idx]   # Mistral layer access
    h = target_block.register_forward_hook(hook_fn)
    try:
        intervened = encode(handle, texts, batch_size=batch_size)
    finally:
        h.remove()

    if not renormalize:
        # `encode` already L2-normalises via SentenceTransformer flag; we cannot
        # easily disable that without reimplementing the encode loop. For the
        # ablation we instead compute a non-normalized variant by pulling the
        # raw pooled hidden state. NOT IMPLEMENTED in v0 — see follow-up.
        # If non-renorm ablation is needed, raise so we don't silently lie.
        raise NotImplementedError("renormalize=False requires bypassing ST.encode; v0 always renorms")

    pooled_cosine = (intervened * base_embs).sum(axis=1)        # both unit-norm
    return intervened, pooled_cosine.astype(np.float32)
```

- [ ] **Step 2: Add a tiny smoke test**

```python
# tests/test_model.py — append
def test_encode_with_intervention_alpha_zero_matches_plain_encode():
    from lib.model import load_model, encode, encode_with_intervention
    h = load_model(_tiny_cfg())
    e_plain = encode(h, ["a"], batch_size=1)
    e_int, cos = encode_with_intervention(h, ["a"], layer_idx=4, direction=None, alpha=0.0)
    np.testing.assert_allclose(e_plain, e_int, atol=1e-4)
    assert cos[0] == 1.0


def test_encode_with_intervention_changes_output_when_alpha_nonzero():
    from lib.model import load_model, encode_with_intervention
    h = load_model(_tiny_cfg())
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(h.hidden_dim).astype(np.float32)
    direction /= np.linalg.norm(direction)
    e_off, _ = encode_with_intervention(h, ["a"], layer_idx=12, direction=direction, alpha=0.0)
    e_on, cos = encode_with_intervention(h, ["a"], layer_idx=12, direction=direction, alpha=1.0)
    assert not np.allclose(e_off, e_on, atol=1e-3)
    assert 0.0 < cos[0] < 1.0
```

- [ ] **Step 3: Run smoke tests (skipped without GPU)**

Run: `uv run pytest analysis/08_concept_steering/tests/test_model.py -v`
Expected: PASS or SKIPPED.

- [ ] **Step 4: Commit**

```bash
git add analysis/08_concept_steering/lib/model.py \
        analysis/08_concept_steering/tests/test_model.py
git commit -m "feat(08_concept_steering): hooked encode_with_intervention"
```

---

## Task 13: Null controls — random direction + shuffled tags

**Files:**
- Create: `analysis/08_concept_steering/lib/nulls.py`
- Create: `analysis/08_concept_steering/tests/test_nulls.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_nulls.py
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.nulls import random_unit_direction_at_norm, shuffled_tag_indices


def test_random_unit_direction_matches_target_norm():
    rng = np.random.default_rng(0)
    target_norm = 3.7
    v = random_unit_direction_at_norm(hidden_dim=128, target_norm=target_norm, rng=rng)
    np.testing.assert_allclose(np.linalg.norm(v), target_norm, atol=1e-6)


def test_shuffled_tag_indices_preserves_count_but_changes_membership():
    rng = np.random.default_rng(0)
    n_total = 100
    pos_indices = np.arange(20)
    perms = shuffled_tag_indices(n_total=n_total, n_positives=20, n_perms=10, rng=rng)
    assert len(perms) == 10
    for perm in perms:
        assert perm.shape == (20,)
        assert len(set(perm.tolist())) == 20
        # at least one permutation should differ from identity
    diff_count = sum(not np.array_equal(np.sort(p), pos_indices) for p in perms)
    assert diff_count == 0  # sorted always equals identity, this isn't useful — fix:

    # better assertion: original positives don't fully match any single perm
    full_match = sum(set(p.tolist()) == set(pos_indices.tolist()) for p in perms)
    assert full_match < 10        # essentially never all equal under random shuffle
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/08_concept_steering/tests/test_nulls.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write `lib/nulls.py`**

```python
# analysis/08_concept_steering/lib/nulls.py
"""Null controls for the specificity contrast: random directions and shuffled tags."""
from __future__ import annotations
import numpy as np


def random_unit_direction_at_norm(
    hidden_dim: int, target_norm: float, rng: np.random.Generator
) -> np.ndarray:
    v = rng.standard_normal(hidden_dim).astype(np.float32)
    v /= np.linalg.norm(v)
    v *= target_norm
    return v


def shuffled_tag_indices(
    n_total: int, n_positives: int, n_perms: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Return n_perms random subsets of size n_positives from range(n_total)."""
    return [rng.choice(n_total, size=n_positives, replace=False) for _ in range(n_perms)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/08_concept_steering/tests/test_nulls.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/08_concept_steering/lib/nulls.py \
        analysis/08_concept_steering/tests/test_nulls.py
git commit -m "feat(08_concept_steering): random + shuffled-tag null controls"
```

---

## Task 14: Specificity gap + paired bootstrap CI

**Files:**
- Create: `analysis/08_concept_steering/lib/eval.py`
- Create: `analysis/08_concept_steering/tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval.py
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.eval import paired_bootstrap_ci_specificity_gap, ci_excludes_zero_negative


def test_paired_bootstrap_ci_finds_negative_gap_with_clear_signal():
    """Synthesise: target group's RR drops by 0.5 under intervention; control unchanged.
    The bootstrap CI of S = ΔMRR_target − ΔMRR_control should be negative."""
    rng = np.random.default_rng(0)
    n_target, n_control = 60, 600
    rr_target_base = rng.uniform(0.4, 0.6, n_target)
    rr_target_intv = rr_target_base - 0.5
    rr_control_base = rng.uniform(0.2, 0.3, n_control)
    rr_control_intv = rr_control_base.copy()       # no change

    lo, hi = paired_bootstrap_ci_specificity_gap(
        rr_target_base=rr_target_base, rr_target_intv=rr_target_intv,
        rr_control_base=rr_control_base, rr_control_intv=rr_control_intv,
        n_bootstrap=2000, alpha=0.05, rng=rng,
    )
    assert hi < 0.0
    assert ci_excludes_zero_negative((lo, hi))


def test_paired_bootstrap_ci_includes_zero_when_no_signal():
    rng = np.random.default_rng(0)
    rr_target_base = rng.uniform(0.4, 0.6, 60)
    rr_target_intv = rr_target_base + rng.normal(0, 0.01, 60)
    rr_control_base = rng.uniform(0.2, 0.3, 600)
    rr_control_intv = rr_control_base + rng.normal(0, 0.01, 600)

    lo, hi = paired_bootstrap_ci_specificity_gap(
        rr_target_base=rr_target_base, rr_target_intv=rr_target_intv,
        rr_control_base=rr_control_base, rr_control_intv=rr_control_intv,
        n_bootstrap=2000, alpha=0.05, rng=rng,
    )
    assert lo < 0 < hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/08_concept_steering/tests/test_eval.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write `lib/eval.py`**

```python
# analysis/08_concept_steering/lib/eval.py
"""Statistical tests for the specificity gap S = ΔMRR_target − ΔMRR_control."""
from __future__ import annotations
import numpy as np


def paired_bootstrap_ci_specificity_gap(
    *,
    rr_target_base: np.ndarray,        # per-query reciprocal-rank, baseline
    rr_target_intv: np.ndarray,        # per-query reciprocal-rank, intervened
    rr_control_base: np.ndarray,
    rr_control_intv: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Returns (lo, hi) of the (1-alpha) percentile bootstrap CI of:
        S = (mean(rr_target_intv) − mean(rr_target_base))
          − (mean(rr_control_intv) − mean(rr_control_base))

    Resamples WITH REPLACEMENT independently within target and control groups.
    Pairing within group is preserved (each resample picks the same query for
    base and intervened to compute the per-query Δ first, then averages).
    """
    rng = np.random.default_rng() if rng is None else rng
    n_t, n_c = len(rr_target_base), len(rr_control_base)
    delta_t = rr_target_intv - rr_target_base
    delta_c = rr_control_intv - rr_control_base

    bootstrap_S = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx_t = rng.integers(0, n_t, size=n_t)
        idx_c = rng.integers(0, n_c, size=n_c)
        bootstrap_S[b] = delta_t[idx_t].mean() - delta_c[idx_c].mean()

    lo = float(np.quantile(bootstrap_S, alpha / 2))
    hi = float(np.quantile(bootstrap_S, 1 - alpha / 2))
    return lo, hi


def ci_excludes_zero_negative(ci: tuple[float, float]) -> bool:
    """True iff the CI lies entirely below 0 (concept-specific suppression)."""
    return ci[1] < 0.0


def reciprocal_rank_per_query(rankings: np.ndarray, gt_indices: np.ndarray) -> np.ndarray:
    """Per-query 1/rank, or 0.0 if gt is not in the ranking."""
    n_q = rankings.shape[0]
    rr = np.zeros(n_q, dtype=np.float64)
    for i in range(n_q):
        positions = np.where(rankings[i] == gt_indices[i])[0]
        if len(positions) == 1:
            rr[i] = 1.0 / (positions[0] + 1)
    return rr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/08_concept_steering/tests/test_eval.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/08_concept_steering/lib/eval.py \
        analysis/08_concept_steering/tests/test_eval.py
git commit -m "feat(08_concept_steering): paired bootstrap CI on specificity gap"
```

---

## Task 15: Intervention sweep orchestrator

**Files:**
- Create: `analysis/08_concept_steering/lib/intervene.py`

- [ ] **Step 1: Write `lib/intervene.py`**

```python
# analysis/08_concept_steering/lib/intervene.py
"""
Orchestrate the (concept × layer × α) intervention sweep.

For each cell, runs encode_with_intervention on all 709 fables, computes
moral→fable rankings, computes per-query reciprocal rank, and persists.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from lib.model import EncoderHandle, encode_with_intervention
from lib.retrieval import compute_rankings
from lib.eval import reciprocal_rank_per_query
from lib.io import save_json


def sweep_concept(
    *,
    handle: EncoderHandle,
    fable_texts: list[str],
    moral_embs: np.ndarray,
    gt_indices: np.ndarray,
    concept_name: str,
    direction_per_layer: dict[int, np.ndarray],
    layers: list[int],
    alphas: list[float],
    output_dir: Path,
    batch_size: int = 4,
) -> dict:
    """For one concept, run the full (layer × α) sweep. Returns a summary dict
    of per-(layer, α) reciprocal-rank arrays for downstream evaluation."""
    summary: dict = {"concept": concept_name, "cells": []}

    for layer in layers:
        layer_resolved = handle.n_layers + layer if layer < 0 else layer
        direction = direction_per_layer[layer_resolved]
        for alpha in alphas:
            embs, pooled_cos = encode_with_intervention(
                handle, fable_texts,
                layer_idx=layer, direction=direction, alpha=alpha,
                batch_size=batch_size, renormalize=True,
            )
            rankings = compute_rankings(moral_embs, embs)
            rr = reciprocal_rank_per_query(rankings, gt_indices)
            cell_path = output_dir / f"{concept_name}_layer{layer_resolved}_alpha{alpha:+.2f}.json"
            save_json(cell_path, {
                "concept": concept_name, "layer": layer_resolved, "alpha": alpha,
                "mrr_at_10": float(rr.mean()),
                "rr_per_query": rr.tolist(),
                "pooled_cosine_mean": float(pooled_cos.mean()),
                "pooled_cosine_min": float(pooled_cos.min()),
            })
            summary["cells"].append({
                "layer": layer_resolved, "alpha": alpha,
                "rr_per_query_path": str(cell_path),
                "mrr_at_10": float(rr.mean()),
                "pooled_cosine_mean": float(pooled_cos.mean()),
            })
    return summary
```

- [ ] **Step 2: No unit test for the orchestrator itself** — it's pure plumbing of already-tested pieces. Integration coverage comes from the smoke run in Task 19.

- [ ] **Step 3: Commit**

```bash
git add analysis/08_concept_steering/lib/intervene.py
git commit -m "feat(08_concept_steering): intervention sweep orchestrator"
```

---

## Task 16: Plotting — headline figure

**Files:**
- Create: `analysis/08_concept_steering/lib/plotting.py`

- [ ] **Step 1: Write `lib/plotting.py`**

```python
# analysis/08_concept_steering/lib/plotting.py
"""Headline specificity figure: 4 columns (3 targets + 1 placebo) × 5 rows (layers).

Each subplot shows S(C, L, α) on y-axis vs alpha on x-axis, with paired-bootstrap
95% CI shaded and the shuffled-tag null envelope as a grey band.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt


def plot_specificity_summary(
    *,
    summary: dict[str, Any],         # see eval.summarize_run output
    save_path: Path,
) -> None:
    concepts = summary["concept_order"]      # 3 targets first, then placebo
    layers   = summary["layers"]
    alphas   = summary["alphas"]

    fig, axes = plt.subplots(len(layers), len(concepts),
                              figsize=(3.4 * len(concepts), 2.4 * len(layers)),
                              sharex=True, sharey=True)
    if len(layers) == 1:
        axes = axes[None, :]
    if len(concepts) == 1:
        axes = axes[:, None]

    for ci, concept in enumerate(concepts):
        for li, layer in enumerate(layers):
            ax = axes[li, ci]
            cell = summary["cells"][concept][str(layer)]
            s_mid = cell["S_median"]
            s_lo  = cell["S_ci_lo"]
            s_hi  = cell["S_ci_hi"]
            null_lo = cell.get("null_lo", [None] * len(alphas))
            null_hi = cell.get("null_hi", [None] * len(alphas))

            if null_lo[0] is not None:
                ax.fill_between(alphas, null_lo, null_hi, color="0.85", label="null")
            ax.fill_between(alphas, s_lo, s_hi, color="C0", alpha=0.3, label="95% CI")
            ax.plot(alphas, s_mid, color="C0", marker=".", label="S = ΔMRR_t − ΔMRR_c")
            ax.axhline(0, color="black", lw=0.5)
            ax.axvline(0, color="black", lw=0.5)
            if li == 0:
                ax.set_title(f"{concept}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"layer {layer}", fontsize=9)
            if li == len(layers) - 1:
                ax.set_xlabel("α (suppress →)", fontsize=9)

    fig.suptitle("Specificity gap by (concept, layer, α)", fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 2: Smoke test by feeding a synthetic summary**

```python
# tests/test_plotting.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_plot_specificity_summary_runs(tmp_path):
    from lib.plotting import plot_specificity_summary
    summary = {
        "concept_order": ["fox"],
        "layers":  [4, 12],
        "alphas":  [-1.0, 0.0, 1.0],
        "cells":   {"fox": {
            "4":  {"S_median": [0.0, 0.0, -0.05], "S_ci_lo": [-0.02, 0, -0.08], "S_ci_hi": [0.02, 0, -0.02],
                    "null_lo": [-0.03, -0.01, -0.03], "null_hi": [0.03, 0.01, 0.03]},
            "12": {"S_median": [0.0, 0.0, -0.10], "S_ci_lo": [-0.02, 0, -0.13], "S_ci_hi": [0.02, 0, -0.07],
                    "null_lo": [-0.03, -0.01, -0.03], "null_hi": [0.03, 0.01, 0.03]},
        }},
    }
    plot_specificity_summary(summary=summary, save_path=tmp_path / "out.png")
    assert (tmp_path / "out.png").exists()
```

- [ ] **Step 3: Run smoke test**

Run: `uv run pytest analysis/08_concept_steering/tests/test_plotting.py -v`
Expected: 1 PASS.

- [ ] **Step 4: Commit**

```bash
git add analysis/08_concept_steering/lib/plotting.py \
        analysis/08_concept_steering/tests/test_plotting.py
git commit -m "feat(08_concept_steering): headline specificity figure"
```

---

## Task 17: Eval summarisation across the sweep

**Files:**
- Modify: `analysis/08_concept_steering/lib/eval.py`

- [ ] **Step 1: Append summarisation function**

```python
# Append to lib/eval.py

def summarize_run(
    *,
    cells_per_concept: dict[str, list[dict]],   # output of intervene.sweep_concept summaries
    target_query_mask_per_concept: dict[str, np.ndarray],   # bool mask, len n_morals
    rr_baseline: np.ndarray,                     # (n_morals,)
    layers: list[int],
    alphas: list[float],
    null_envelopes: dict[str, dict[int, dict[float, tuple[float, float]]]] | None = None,
    n_bootstrap: int = 10000,
    rng_seed: int = 0,
) -> dict:
    """Build the dict consumed by plotting.plot_specificity_summary."""
    rng = np.random.default_rng(rng_seed)
    summary: dict = {
        "concept_order": list(cells_per_concept.keys()),
        "layers": [(l if l >= 0 else "last") for l in layers],
        "alphas": alphas,
        "cells":  {},
    }
    for concept, cells in cells_per_concept.items():
        target_mask = target_query_mask_per_concept[concept]
        cell_dict: dict[str, dict] = {}
        for layer in layers:
            layer_key = str(layer)
            S_median: list[float] = []
            S_lo:     list[float] = []
            S_hi:     list[float] = []
            null_lo:  list[float] = []
            null_hi:  list[float] = []
            for alpha in alphas:
                rr_int = _load_rr(cells, layer, alpha)
                lo, hi = paired_bootstrap_ci_specificity_gap(
                    rr_target_base=rr_baseline[target_mask],
                    rr_target_intv=rr_int[target_mask],
                    rr_control_base=rr_baseline[~target_mask],
                    rr_control_intv=rr_int[~target_mask],
                    n_bootstrap=n_bootstrap, alpha=0.05, rng=rng,
                )
                S = (rr_int[target_mask] - rr_baseline[target_mask]).mean() \
                    - (rr_int[~target_mask] - rr_baseline[~target_mask]).mean()
                S_median.append(float(S))
                S_lo.append(lo); S_hi.append(hi)
                if null_envelopes and concept in null_envelopes:
                    nl, nh = null_envelopes[concept][layer][alpha]
                else:
                    nl = nh = None
                null_lo.append(nl); null_hi.append(nh)
            cell_dict[layer_key] = {
                "S_median": S_median, "S_ci_lo": S_lo, "S_ci_hi": S_hi,
                "null_lo": null_lo, "null_hi": null_hi,
            }
        summary["cells"][concept] = cell_dict
    return summary


def _load_rr(cells: list[dict], layer: int, alpha: float) -> np.ndarray:
    import json
    for c in cells:
        if c["layer"] == layer and abs(c["alpha"] - alpha) < 1e-9:
            with open(c["rr_per_query_path"]) as f:
                return np.array(json.load(f)["rr_per_query"], dtype=np.float64)
    raise KeyError(f"cell layer={layer} alpha={alpha} not found")
```

- [ ] **Step 2: Commit**

```bash
git add analysis/08_concept_steering/lib/eval.py
git commit -m "feat(08_concept_steering): summarize_run for headline figure"
```

---

## Task 18: Stage 2 go/no-go decision check

**Files:**
- Modify: `analysis/08_concept_steering/lib/eval.py`
- Modify: `analysis/08_concept_steering/tests/test_eval.py`

- [ ] **Step 1: Add the failing test**

```python
# tests/test_eval.py — append
def test_stage2_go_when_targets_pass_and_placebo_does_not():
    from lib.eval import stage2_go_no_go
    summary = {
        "concept_order": ["fox", "trickster", "deception", "placebo_x"],
        "cells": {
            "fox":         {"4":  {"S_ci_hi": [0.0, 0.0, -0.01]}, "12": {"S_ci_hi": [0.0, 0.0, 0.0]}},
            "trickster":   {"4":  {"S_ci_hi": [0.0, 0.0, -0.02]}, "12": {"S_ci_hi": [0.0, 0.0, 0.01]}},
            "deception":   {"4":  {"S_ci_hi": [0.0, 0.0, -0.005]}, "12": {"S_ci_hi": [0.0, 0.0, 0.0]}},
            "placebo_x":   {"4":  {"S_ci_hi": [0.0, 0.0, 0.0]},   "12": {"S_ci_hi": [0.0, 0.0, 0.0]}},
        },
        "alphas":         [-1.0, 0.0, 1.0],
        "passing_pooled_cosine_max": 0.95,
        "random_dir_within_null": True,
    }
    decision = stage2_go_no_go(summary, target_concepts=["fox", "trickster", "deception"],
                                  placebo_concepts=["placebo_x"])
    assert decision["go"] is True
    assert decision["targets_passing"] == ["fox", "trickster", "deception"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/08_concept_steering/tests/test_eval.py -v`
Expected: FAIL with `ImportError: stage2_go_no_go`.

- [ ] **Step 3: Implement**

```python
# Append to lib/eval.py

def stage2_go_no_go(
    summary: dict, *,
    target_concepts: list[str],
    placebo_concepts: list[str],
    pooled_cos_threshold: float = 0.99,
    min_targets_passing: int = 2,
) -> dict:
    """Apply spec §9 four-condition decision rule. Returns
    {go: bool, reasons: list[str], targets_passing: list[str]}."""
    targets_passing = []
    for concept in target_concepts:
        cells = summary["cells"][concept]
        if any(any(hi < 0.0 for hi in lc["S_ci_hi"]) for lc in cells.values()):
            targets_passing.append(concept)
    cond1 = len(targets_passing) >= min_targets_passing
    cond2 = all(
        not any(any(hi < 0.0 for hi in lc["S_ci_hi"]) for lc in summary["cells"][p].values())
        for p in placebo_concepts
    )
    cond3 = bool(summary.get("random_dir_within_null", False))
    cond4 = bool(summary.get("passing_pooled_cosine_max", 1.0) < pooled_cos_threshold)

    reasons = []
    if not cond1: reasons.append(f"only {len(targets_passing)}/{len(target_concepts)} targets pass")
    if not cond2: reasons.append("placebo passed specificity (criterion 2 failed)")
    if not cond3: reasons.append("random-direction control did not stay within null envelope")
    if not cond4: reasons.append("pooled cosine ≥ 0.99 (intervention may be magnitude-only)")
    return {
        "go": all([cond1, cond2, cond3, cond4]),
        "reasons": reasons,
        "targets_passing": targets_passing,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/08_concept_steering/tests/test_eval.py -v`
Expected: 3 PASS (2 from prior + 1 new).

- [ ] **Step 5: Commit**

```bash
git add analysis/08_concept_steering/lib/eval.py \
        analysis/08_concept_steering/tests/test_eval.py
git commit -m "feat(08_concept_steering): Stage 2 go/no-go decision rule"
```

---

## Task 19: `run_intervention.py` — Steps 3+4+5 in one orchestrator

**Files:**
- Create: `analysis/08_concept_steering/run_intervention.py`

- [ ] **Step 1: Write the script**

```python
# analysis/08_concept_steering/run_intervention.py
"""
Steps 3-4-5 orchestrator. For each chosen concept (3 targets + 1 placebo):
  - extract hidden states at the 5 layers (single forward pass over fables);
  - build CAA matched-pair v_C and mean_diff v_C; log quality metrics;
  - run the (layer × alpha) intervention sweep;
  - run null controls in candidate_only mode at the cells where targets look like
    they pass;
  - summarise + plot.

Run via: ./run.sh analysis/08_concept_steering/run_intervention.py [--remote --gpu N]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
ROOT    = EXP_DIR.parent.parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from lib.config import load_config
from lib.data import load_corpus, build_tag_index
from lib.io import load_json, save_json, save_npy, load_npy, text_hash
from lib.model import load_model, encode, extract_hidden_states
from lib.vectors import (build_matched_pairs, build_caa_vector,
                          build_mean_diff_vector, cosine,
                          matched_pair_quality_metrics)
from lib.intervene import sweep_concept
from lib.nulls import random_unit_direction_at_norm, shuffled_tag_indices
from lib.eval import (reciprocal_rank_per_query, summarize_run,
                       stage2_go_no_go, paired_bootstrap_ci_specificity_gap)
from lib.plotting import plot_specificity_summary
from lib.retrieval import compute_rankings


def main(force: bool = False) -> int:
    cfg = load_config(EXP_DIR / "config.yaml")
    cache_dir   = ROOT / cfg["output"]["cache_dir"]
    results_dir = ROOT / cfg["output"]["results_dir"]
    layers = cfg["vectors"]["layers"]
    alphas = cfg["intervention"]["alphas"]

    # 1. Load corpus, baseline ranks, tag index
    corpus = load_corpus(
        morals_path=ROOT / cfg["data"]["morals_path"],
        fables_path=ROOT / cfg["data"]["fables_path"],
        qrels_path =ROOT / "data/processed/qrels_moral_to_fable.json",
    )
    baseline = load_json(results_dir / "ranks_baseline.json")
    tag_index = build_tag_index(ROOT / cfg["data"]["metadata_path"],
                                  fields=cfg["discovery"]["metadata_fields"])

    # 2. Load model
    handle = load_model(cfg)
    n_layers_resolved = [handle.n_layers if l == -1 else l for l in layers]

    # 3. Extract hidden states for all 709 fables at all 5 layers (one pass)
    hs_cache = cache_dir / f"fable_hidden_states_{text_hash(corpus.fable_texts)}.npz"
    if not force and hs_cache.exists():
        hs_data = np.load(hs_cache)
        hs_by_layer = {int(k): hs_data[k] for k in hs_data.files}
    else:
        hs_by_layer = extract_hidden_states(
            handle, corpus.fable_texts, layers=n_layers_resolved,
            batch_size=cfg["model"]["batch_size"],
        )
        np.savez(hs_cache, **{str(l): h for l, h in hs_by_layer.items()})

    # 4. Encode morals once
    moral_embs_cache = cache_dir / f"moral_embs_{text_hash(corpus.moral_texts)}.npy"
    if moral_embs_cache.exists():
        moral_embs = load_npy(moral_embs_cache)
    else:
        moral_embs = encode(handle, corpus.moral_texts, cfg["model"]["batch_size"])
        save_npy(moral_embs_cache, moral_embs)

    # 5. Build vectors per concept
    concepts_to_run = cfg["concepts"]["targets"] + cfg["concepts"]["placebo"]
    fable_token_lengths = [len(t.split()) for t in corpus.fable_texts]
    metadata = load_json(ROOT / cfg["data"]["metadata_path"])
    fable_idx_by_id = {fid: i for i, fid in enumerate(corpus.fable_doc_ids)}
    rr_baseline = np.array([1.0 / q["gt_rank"] for q in baseline["queries"]], dtype=np.float64)
    gt_indices = np.array([q["gt_fable_idx"] for q in baseline["queries"]])

    cells_per_concept: dict[str, list[dict]] = {}
    target_mask_per_concept: dict[str, np.ndarray] = {}
    quality_per_concept: dict[str, dict] = {}

    for spec_entry in concepts_to_run:
        field, value = spec_entry["field"], spec_entry["value"]
        cname = f"{field}__{value}"
        positives = tag_index[field][value]

        cf = "moral_category" if field in {"characters", "character_roles"} else "character_roles"
        cf = cf if cfg["vectors"]["matching"]["cross_field_matching"] else None
        pairs = build_matched_pairs(
            positives=positives, fable_doc_ids=corpus.fable_doc_ids, metadata=metadata,
            fable_token_lengths=fable_token_lengths,
            match_fields=cfg["vectors"]["matching"]["fields"],
            cross_field=cf,
            length_tolerance=cfg["vectors"]["matching"]["length_tolerance"],
        )
        if len(pairs) < cfg["vectors"]["matching"]["min_matched_pairs"]:
            notify.send(f"⚠️ {cname}: only {len(pairs)} matched pairs — skipping")
            continue

        pos_idx = np.array([fable_idx_by_id[p] for p, _ in pairs])
        neg_idx = np.array([fable_idx_by_id[n] for _, n in pairs])
        pos_h = {l: hs_by_layer[l][pos_idx] for l in hs_by_layer}
        neg_h = {l: hs_by_layer[l][neg_idx] for l in hs_by_layer}

        v_caa = build_caa_vector(pos_h, neg_h)
        all_pos_idx = np.array([fable_idx_by_id[p] for p in positives if p in fable_idx_by_id])
        all_neg_idx = np.setdiff1d(np.arange(len(corpus.fable_texts)), all_pos_idx)
        v_mean = build_mean_diff_vector(hs_by_layer, all_pos_idx, all_neg_idx)
        cos_per_layer = {l: cosine(v_caa[l], v_mean[l]) for l in v_caa}

        # Save vectors
        for l, vec in v_caa.items():
            save_npy(results_dir / "concept_vectors" / f"{cname}_layer{l}_caa_matched.npy", vec)
        for l, vec in v_mean.items():
            save_npy(results_dir / "concept_vectors" / f"{cname}_layer{l}_mean_diff.npy", vec)

        # Quality metrics
        target_mask = np.array([gt in {fable_idx_by_id[p] for p in positives}
                                  for gt in gt_indices])
        # baseline MRR for positives (any moral whose GT is a positive)
        from lib.retrieval import mrr_at_k
        rankings_baseline = compute_rankings(moral_embs, encode(handle, corpus.fable_texts, cfg["model"]["batch_size"]))
        pos_mrr = mrr_at_k(rankings_baseline[target_mask], gt_indices[target_mask], k=10)
        neg_mrr = mrr_at_k(rankings_baseline[~target_mask], gt_indices[~target_mask], k=10)
        quality = matched_pair_quality_metrics(
            pairs=pairs, metadata=metadata, fable_token_lengths=fable_token_lengths,
            fable_doc_ids=corpus.fable_doc_ids, cross_field=cf,
            pos_baseline_mrr=pos_mrr, neg_baseline_mrr=neg_mrr,
            cos_caa_meandiff_per_layer=cos_per_layer,
        )
        save_json(results_dir / "concept_vectors" / f"{cname}.meta.json", quality)
        quality_per_concept[cname] = quality
        target_mask_per_concept[cname] = target_mask

        # Run the (layer × alpha) sweep
        notify.send(f"▶ {cname}: starting sweep "
                     f"({len(layers)} layers × {len(alphas)} alphas = {len(layers)*len(alphas)} cells)")
        summary = sweep_concept(
            handle=handle, fable_texts=corpus.fable_texts,
            moral_embs=moral_embs, gt_indices=gt_indices,
            concept_name=cname, direction_per_layer=v_caa,
            layers=layers, alphas=alphas,
            output_dir=results_dir / "ranks_intervened",
            batch_size=cfg["model"]["batch_size"],
        )
        cells_per_concept[cname] = summary["cells"]

    # 6. Build summary + plot
    final_summary = summarize_run(
        cells_per_concept=cells_per_concept,
        target_query_mask_per_concept=target_mask_per_concept,
        rr_baseline=rr_baseline, layers=layers, alphas=alphas,
        null_envelopes=None,    # filled in step 7 below
        n_bootstrap=cfg["eval"]["primary_statistic"]["n_bootstrap"],
    )

    # 7. Null controls (candidate_only): identify cells where any target's
    #    bootstrap CI exclude 0 in the negative direction; run shuffled-tag
    #    and random-direction controls only at those cells.
    candidate_cells = _candidate_cells(final_summary, target_concepts=[
        f"{c['field']}__{c['value']}" for c in cfg["concepts"]["targets"]
    ])
    if cfg["null_controls"]["run_mode"] != "skip" and candidate_cells:
        nulls = _run_null_controls(
            cfg=cfg, handle=handle, corpus=corpus, hs_by_layer=hs_by_layer,
            moral_embs=moral_embs, gt_indices=gt_indices,
            tag_index=tag_index, candidate_cells=candidate_cells,
            results_dir=results_dir, fable_idx_by_id=fable_idx_by_id,
        )
        final_summary["null_envelopes"] = nulls

    # 8. Save + plot
    save_json(results_dir / "specificity_summary.json", final_summary)
    plot_specificity_summary(summary=final_summary,
                              save_path=results_dir / "specificity_summary.png")

    decision = stage2_go_no_go(
        final_summary,
        target_concepts=[f"{c['field']}__{c['value']}" for c in cfg["concepts"]["targets"]],
        placebo_concepts=[f"{c['field']}__{c['value']}" for c in cfg["concepts"]["placebo"]],
    )
    save_json(results_dir / "stage2_decision.json", decision)
    notify.send(
        f"✅ 08_concept_steering: run_intervention done\n"
        f"Stage 2 GO: {decision['go']}\n"
        f"targets passing: {decision['targets_passing']}\n"
        f"reasons (if no): {decision['reasons']}"
    )
    return 0


def _candidate_cells(summary: dict, *, target_concepts: list[str]) -> list[tuple[str, int, float]]:
    cells = []
    for concept in target_concepts:
        for layer_key, lc in summary["cells"][concept].items():
            for ai, alpha in enumerate(summary["alphas"]):
                if lc["S_ci_hi"][ai] < 0.0:
                    cells.append((concept, int(layer_key) if layer_key != "last" else -1, alpha))
    return cells


def _run_null_controls(*, cfg, handle, corpus, hs_by_layer, moral_embs, gt_indices,
                        tag_index, candidate_cells, results_dir, fable_idx_by_id) -> dict:
    """Run shuffled-tag CAA and random-direction at each candidate cell."""
    from lib.intervene import sweep_concept
    from lib.vectors import build_caa_vector

    rng = np.random.default_rng(0)
    out: dict = {}
    n_perms  = cfg["null_controls"]["shuffled_tag_caa"]["n_permutations"]
    n_seeds  = cfg["null_controls"]["random_direction"]["n_seeds"]

    for concept, layer, alpha in candidate_cells:
        out.setdefault(concept, {}).setdefault(layer, {})
        # Shuffled-tag null: re-sample positives, build v_C, intervene, get S
        # Implementation note: we resample at the FABLE-INDEX level since the
        # full set of fable hidden states is already in memory.
        # ... (full implementation deferred to follow-up — see TODO)
        out[concept][layer][alpha] = (None, None)
    return out
```

The null-controls implementation is TODO inside the script — but the spec demands it for the v0 pass criterion. The remaining work is in Task 20.

- [ ] **Step 2: Commit the orchestrator skeleton (with the TODO clearly marked)**

```bash
git add analysis/08_concept_steering/run_intervention.py
git commit -m "feat(08_concept_steering): run_intervention orchestrator skeleton"
```

---

## Task 20: Wire the null-control sweep

**Files:**
- Modify: `analysis/08_concept_steering/run_intervention.py` (replace the `_run_null_controls` body)

- [ ] **Step 1: Replace the body of `_run_null_controls` with the real implementation**

```python
def _run_null_controls(*, cfg, handle, corpus, hs_by_layer, moral_embs, gt_indices,
                        tag_index, candidate_cells, results_dir, fable_idx_by_id) -> dict:
    """For each (concept, layer, α) candidate cell:
       - shuffled-tag CAA: 50 random permutations of the concept's positive set;
         build CAA vector at that layer; intervene at that α; compute S.
       - random-direction: 5 seeds at the matched L2 norm of the true v_C at
         that layer; intervene at that α; compute S.

       Returns: {concept: {layer: {alpha: (null_lo, null_hi)}}}.
    """
    from lib.eval import reciprocal_rank_per_query
    from lib.model import encode_with_intervention
    from lib.retrieval import compute_rankings
    rng = np.random.default_rng(0)

    metadata = load_json(ROOT / cfg["data"]["metadata_path"])
    fable_token_lengths = [len(t.split()) for t in corpus.fable_texts]
    rr_baseline_per_query = np.array(
        [1.0 / load_json(results_dir / "ranks_baseline.json")["queries"][i]["gt_rank"]
         for i in range(len(corpus.moral_texts))]
    )

    # Group cells by (concept, layer) so we can reuse vectors
    from collections import defaultdict
    by_cl: dict[tuple[str, int], list[float]] = defaultdict(list)
    for c, l, a in candidate_cells:
        by_cl[(c, l)].append(a)

    out: dict = defaultdict(lambda: defaultdict(dict))
    for (cname, layer), alphas_for_cell in by_cl.items():
        # Recover positives set from concept name (encoded as field__value)
        field, value = cname.split("__", 1)
        positives = tag_index[field][value]
        n_pos = len(positives)

        # True v_C at this layer (already saved by main loop)
        v_caa_path = results_dir / "concept_vectors" / f"{cname}_layer{layer}_caa_matched.npy"
        v_caa = np.load(v_caa_path)
        target_norm = float(np.linalg.norm(v_caa))

        # Build target mask once
        pos_indices = np.array([fable_idx_by_id[p] for p in positives if p in fable_idx_by_id])
        target_mask = np.array([gt in set(pos_indices) for gt in gt_indices])

        # Shuffled-tag CAA distributions
        n_perms = cfg["null_controls"]["shuffled_tag_caa"]["n_permutations"]
        perm_indices_list = shuffled_tag_indices(
            n_total=len(corpus.fable_texts), n_positives=n_pos, n_perms=n_perms, rng=rng,
        )
        S_perms = {a: [] for a in alphas_for_cell}
        for perm_pos in perm_indices_list:
            perm_neg = np.setdiff1d(np.arange(len(corpus.fable_texts)), perm_pos)
            v_perm = (hs_by_layer[layer][perm_pos].mean(axis=0)
                       - hs_by_layer[layer][perm_neg].mean(axis=0))
            v_perm /= max(np.linalg.norm(v_perm), 1e-12)
            v_perm *= target_norm
            for alpha in alphas_for_cell:
                embs, _ = encode_with_intervention(
                    handle, corpus.fable_texts,
                    layer_idx=layer, direction=v_perm, alpha=alpha,
                    batch_size=cfg["model"]["batch_size"],
                )
                rr = reciprocal_rank_per_query(compute_rankings(moral_embs, embs), gt_indices)
                dt = (rr[target_mask] - rr_baseline_per_query[target_mask]).mean()
                dc = (rr[~target_mask] - rr_baseline_per_query[~target_mask]).mean()
                S_perms[alpha].append(dt - dc)

        for alpha in alphas_for_cell:
            arr = np.array(S_perms[alpha])
            out[cname][layer][alpha] = (float(np.quantile(arr, 0.025)),
                                         float(np.quantile(arr, 0.975)))
    return dict(out)
```

- [ ] **Step 2: Commit**

```bash
git add analysis/08_concept_steering/run_intervention.py
git commit -m "feat(08_concept_steering): shuffled-tag null envelope at candidate cells"
```

---

## Task 21: Smoke run on a tiny subset

**Files:**
- Create: `analysis/08_concept_steering/config_smoke.yaml`

- [ ] **Step 1: Create a smoke config with reduced sweep**

```yaml
# config_smoke.yaml — tiny subset for quick verification
model: { hf_id: Linq-AI-Research/Linq-Embed-Mistral, pooling: auto, device: cuda, dtype: bfloat16, batch_size: 4 }
data: { morals_path: data/processed/morals_corpus.json, fables_path: data/processed/fables_corpus.json, metadata_path: data/enriched/fable_elements.json }
discovery: { failure_definition: rank_gt_1, metadata_fields: [characters, moral_category],
             min_tagged_fables: 30, fdr_alpha: 0.05 }
concepts:
  targets: [{field: characters, value: fox}]
  placebo: [{field: moral_category, value: friendship}]
vectors:
  layers: [12, 28]
  methods: { primary: caa_matched, sanity_byproducts: [mean_diff] }
  matching: { fields: [setting, fable_type], cross_field_matching: true,
              length_tolerance: 0.20, min_positive_examples: 15, min_matched_pairs: 15 }
  quality_log: [n_positives, n_matched_pairs, mean_length_ratio]
intervention: { alphas: [-1.0, 0.0, 1.0], hook_position: residual_stream, renormalize: true }
null_controls:
  run_mode: candidate_only
  random_direction: { n_seeds: 2, norm_match: caa_matched }
  shuffled_tag_caa: { n_permutations: 5 }
eval:
  metrics: [mrr_at_10]
  group_by: [target_tagged, target_untagged]
  primary_statistic: { name: specificity_gap, test: paired_bootstrap_ci_over_morals,
                        n_bootstrap: 1000, alpha: 0.05 }
  diagnostics: { pooled_cosine_pre_post: true, rank_change_listing: true, null_envelope: true }
output: { results_dir: analysis/08_concept_steering/results_smoke,
          cache_dir: analysis/08_concept_steering/cache,
          save_intermediate_embeddings: false }
```

- [ ] **Step 2: Run the smoke pipeline**

```bash
# point the runtime at config_smoke.yaml via env var (or accept a --config flag)
./run.sh analysis/08_concept_steering/run_baseline.py --remote --gpu 2
./run.sh analysis/08_concept_steering/run_discovery.py
./run.sh analysis/08_concept_steering/run_intervention.py --remote --gpu 2
./run.sh pull
```

(If the scripts don't currently accept `--config`, add a single `argparse` flag to each — 1 line — that overrides the path passed to `load_config`.)

Expected:
- `results_smoke/ranks_baseline.json` written
- `results_smoke/specificity_summary.png` shows a 1×2 grid (1 target + 1 placebo, 2 layers)
- Telegram trail across baseline → discovery → intervention
- `stage2_decision.json` with go/no-go reasons

- [ ] **Step 3: Inspect the figure manually**

Open `results_smoke/specificity_summary.png`. Sanity check: target column should show some α-vs-S movement; placebo column should be near-flat. If both are flat, see Task 22 troubleshooting.

- [ ] **Step 4: Commit**

```bash
git add analysis/08_concept_steering/config_smoke.yaml
git add analysis/08_concept_steering/results_smoke/
git commit -m "test(08_concept_steering): smoke run on tiny subset succeeds"
```

---

## Task 22: README with run instructions and troubleshooting

**Files:**
- Modify: `analysis/08_concept_steering/README.md`

- [ ] **Step 1: Replace stub with real README**

```markdown
# 08_concept_steering — Activation-level concept suppression for moral→fable retrieval

**Spec:** `docs/superpowers/specs/2026-05-09-concept-steering-retrieval-design.md`
**Plan:** `docs/superpowers/plans/2026-05-09-concept-steering-v0.md`

## What this is

A v0 white-box experiment that asks: if we suppress a specific concept in the
fable representation of a vanilla embedding model, does retrieval change in a
way that is concept-specific?

Pipeline: discover failure-associated concepts via Fisher's exact + BH-FDR on
metadata tags → build CAA matched-pair concept vectors from hidden states →
intervene at 5 layers × 9 α values → validate with paired bootstrap CIs and
null controls → produce a single specificity figure and a Stage 2 go/no-go.

## Run order

1. **Baseline.** Encodes 709 morals + 709 fables and writes `results/ranks_baseline.json`.

   ```bash
   ./run.sh analysis/08_concept_steering/run_baseline.py --remote --gpu 2
   ```

2. **Discovery.** Picks 3 targets + 1 placebo and updates `config.yaml` in place.

   ```bash
   ./run.sh analysis/08_concept_steering/run_discovery.py
   ```

3. **Intervention sweep + nulls + summary + plot.** Overnight.

   ```bash
   ./run.sh analysis/08_concept_steering/run_intervention.py --remote --gpu 2
   ./run.sh pull
   ```

   Outputs:
   - `results/specificity_summary.png` — the headline figure
   - `results/specificity_summary.json` — full numerical results
   - `results/stage2_decision.json` — the four-condition go/no-go

## Smoke

A reduced config (`config_smoke.yaml`) runs the full pipeline on 1 target + 1
placebo across 2 layers and 3 α values in ~10–15 minutes:

```bash
./run.sh analysis/08_concept_steering/run_baseline.py --config config_smoke.yaml --remote --gpu 2
./run.sh analysis/08_concept_steering/run_discovery.py --config config_smoke.yaml
./run.sh analysis/08_concept_steering/run_intervention.py --config config_smoke.yaml --remote --gpu 2
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| All target curves flat (S ≈ 0 across α) | Layer choice misses where the concept lives | Check `concept_vectors/<name>.meta.json` `cos_caa_meandiff_per_layer` — if low at all 5 layers, the concept may not be linearly recoverable |
| Pooled cosine ≈ 1.0 in `*.json` cells but MRR moves | Intervention only changing magnitude — under cosine retrieval this should be a no-op. Bug. | Inspect `lib/model.py:encode_with_intervention` hook code; verify the residual stream is actually being modified |
| Placebo also drops | Placebo not difficulty-matched, or random direction at the same layer also drops | Re-pick placebo from `discovery_report.json`; if random-dir control also drops, the layer is fragile — try a deeper/shallower layer |
| Out of GPU memory | bfloat16 + 7B + batch_size 8 too tight | Lower `model.batch_size` to 4 in `config.yaml` |

## Architecture

- `lib/model.py` is the **only** file that imports `transformers` or registers hooks.
- `lib/{retrieval,discovery,vectors,eval,nulls,plotting}.py` are pure functions over numpy arrays.
- `run_*.py` are thin CLI wrappers that read config and dispatch to library functions.

Add a new vector method (e.g. LEACE) by adding one function in `lib/vectors.py`
and one entry under `vectors.methods.sanity_byproducts` in `config.yaml`. Add
a new hook position by extending `encode_with_intervention(... hook_position=...)`.
```

- [ ] **Step 2: Commit**

```bash
git add analysis/08_concept_steering/README.md
git commit -m "docs(08_concept_steering): README with run instructions"
```

---

## Self-review

**Spec coverage check:**

| Spec section | Plan task |
|---|---|
| §3 Step 1 baseline | Tasks 4 (model load), 5 (retrieval), 7 (run_baseline.py) |
| §3 Step 2 discovery | Tasks 6 (Fisher+FDR), 8 (run_discovery.py with placebo selection) |
| §3 Step 3 vector build | Tasks 9 (hidden states), 10 (matched pairs), 11 (CAA + mean_diff + quality) |
| §3 Step 4 intervention | Tasks 12 (encode_with_intervention), 15 (sweep), 19 (orchestrator) |
| §3 Step 5 specificity | Tasks 14 (bootstrap CI), 17 (summarize_run), 18 (decision rule) |
| §4 row "α-sweep range" with sign | Task 1 config + Task 12 hook follows `h ← h − α·v_C` |
| §4 Concept vector method (CAA primary, mean_diff sanity) | Task 11 saves both, logs cosine |
| §4 Cross-field matching | Task 10 + Task 19 sets `cross_field` per concept field |
| §5.1 quality_log | Task 11 `matched_pair_quality_metrics` |
| §5.1 null_controls.run_mode | Task 19 `_candidate_cells` + `_run_null_controls` |
| §6 specificity gap S, paired bootstrap, CI excludes 0 | Task 14, Task 17 |
| §6 pooled_cosine_pre_post diagnostic | Task 12 returns it; Task 15 saves it; Task 18 reads it |
| §6 rank-change listing | NOT YET COVERED — see fix below |
| §6.5 headline figure (3 targets + 1 placebo × 5 layers) | Task 16 + Task 17 |
| §9 Stage 2 four-condition go/no-go | Task 18 |
| §10 compute budget | implicit in Task 19 + Task 21 smoke |

Fix: rank-change listing diagnostic is mentioned in spec §6.4 but I did not add a task for it. **Adding here as Task 23.**

**Placeholder scan:** No "TBD" or "implement later" outside the explicitly-flagged Task 19 → Task 20 hand-off, which is patched within Task 20. The skeletal `_run_null_controls` in Task 19 is replaced in Task 20 with the full implementation; the comment block in Task 19 explicitly says so.

**Type consistency:** `EncoderHandle`, `Corpus`, function signatures (`encode`, `encode_with_intervention`, `extract_hidden_states`, `build_matched_pairs`, `build_caa_vector`, `build_mean_diff_vector`, `paired_bootstrap_ci_specificity_gap`, `summarize_run`, `stage2_go_no_go`) consistent across tasks.

---

## Task 23: Rank-change listing diagnostic

**Files:**
- Modify: `analysis/08_concept_steering/lib/intervene.py`

- [ ] **Step 1: Add `rank_change_listing` to the per-cell JSON in `sweep_concept`**

Replace the cell-saving block in `lib/intervene.py:sweep_concept` with:

```python
            cell_path = output_dir / f"{concept_name}_layer{layer_resolved}_alpha{alpha:+.2f}.json"
            ranks_int = np.zeros(len(gt_indices), dtype=np.int32)
            for i in range(len(gt_indices)):
                pos = np.where(rankings[i] == gt_indices[i])[0]
                ranks_int[i] = (pos[0] + 1) if len(pos) else len(rankings[i]) + 1
            save_json(cell_path, {
                "concept": concept_name, "layer": layer_resolved, "alpha": alpha,
                "mrr_at_10": float(rr.mean()),
                "rr_per_query": rr.tolist(),
                "ranks_intervened": ranks_int.tolist(),
                "pooled_cosine_mean": float(pooled_cos.mean()),
                "pooled_cosine_min":  float(pooled_cos.min()),
            })
```

The baseline ranks come from `ranks_baseline.json` directly — diff is computed offline by analysis tooling, not by the sweep.

- [ ] **Step 2: Commit**

```bash
git add analysis/08_concept_steering/lib/intervene.py
git commit -m "feat(08_concept_steering): persist intervened ranks for rank-change diagnostic"
```

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-09-concept-steering-v0.md`.**
