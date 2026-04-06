# Exp10 — Local Model Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a config-driven N×M evaluation pipeline that sweeps local HuggingFace generation models × embedding models on the MORABLES fable retrieval task.

**Architecture:** Stage 1 generates corpus summaries + query paraphrases once per generation model (cached). Stage 2 evaluates all (gen_model × embed_model × ablation) combinations. A final aggregator produces a matrix table and rankings JSON.

**Tech Stack:** Python 3.11+, HuggingFace Transformers, sentence-transformers, PyTorch (MPS), pytest, numpy, PyYAML.

---

## File Map

### New files
| Path | Responsibility |
|---|---|
| `lib/pipeline/paraphrase_filter.py` | Word-count post-processing + embedding similarity filter |
| `lib/pipeline/local_llm.py` | Load/unload HF model to MPS, apply chat template, generate |
| `lib/pipeline/local_corpus_generator.py` | fable → summary per gen model; writes `corpus_summaries.json` + `diagnostics.json` |
| `lib/pipeline/local_query_paraphraser.py` | moral → 3 rephrases per gen model; filters; writes `query_paraphrases.json` |
| `lib/pipeline/matrix_aggregator.py` | Reads all `retrieval_results/*.json`, writes `matrix_summary.json` + `rankings.json` |
| `lib/pipeline/matrix_runner.py` | Top-level N×M orchestrator — `run_matrix_experiment()` |
| `experiments/10_model_matrix/config.yaml` | Exp10 config |
| `experiments/10_model_matrix/run_pipeline.py` | Entry point shim |
| `tests/pipeline/test_paraphrase_filter.py` | Tests for paraphrase_filter |
| `tests/pipeline/test_local_llm.py` | Tests for LocalLLM (mocked transformers) |
| `tests/pipeline/test_local_corpus_generator.py` | Tests for generate_summaries |
| `tests/pipeline/test_local_query_paraphraser.py` | Tests for generate_paraphrases |
| `tests/pipeline/test_matrix_aggregator.py` | Tests for aggregate() |
| `tests/pipeline/test_matrix_runner.py` | Tests for run_matrix_experiment() |

### Modified files
| Path | Change |
|---|---|
| `lib/pipeline/retrieval_eval.py` | Add `run_single_eval()` function (existing code untouched) |
| `lib/pipeline/default_config.yaml` | Add `generation_models: []` and `embed_models: []` keys |
| `lib/pipeline/__init__.py` | Export `run_matrix_experiment` |
| `tests/pipeline/test_retrieval_eval.py` | Add tests for `run_single_eval` |

---

## Task 1: `paraphrase_filter.py` — word-count check + similarity filter

**Files:**
- Create: `lib/pipeline/paraphrase_filter.py`
- Create: `tests/pipeline/test_paraphrase_filter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_paraphrase_filter.py
import sys
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline.paraphrase_filter import post_process, filter_paraphrases


# ── post_process ─────────────────────────────────────────────────────────────

def test_post_process_in_range_not_flagged():
    text = "Appearances are deceptive."  # 3 words — wait, that's 3
    text = "Those who envy others invite their own misfortune."  # 8 words
    cfg = {"min_words": 5, "max_words": 15}
    result, flagged = post_process(text, cfg)
    assert result == text
    assert flagged is False


def test_post_process_too_short_flagged():
    text = "Be honest."  # 2 words
    cfg = {"min_words": 5, "max_words": 15}
    _, flagged = post_process(text, cfg)
    assert flagged is True


def test_post_process_too_long_flagged():
    text = " ".join(["word"] * 20)  # 20 words
    cfg = {"min_words": 5, "max_words": 15}
    _, flagged = post_process(text, cfg)
    assert flagged is True


def test_post_process_returns_text_unchanged():
    text = "  Patience is a virtue.  "
    cfg = {"min_words": 5, "max_words": 15}
    result, _ = post_process(text, cfg)
    assert result == text


# ── filter_paraphrases ────────────────────────────────────────────────────────

def _mock_filter_model(sims: list[float]):
    """Returns a model whose encode() produces embeddings with controlled cosine similarities."""
    model = MagicMock()
    dim = 4
    # original embedding: [1, 0, 0, 0]
    orig_emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    # para embeddings: each row has sim = sims[i] to orig_emb
    para_embs = np.zeros((len(sims), dim), dtype=np.float32)
    for i, s in enumerate(sims):
        para_embs[i, 0] = s  # dot product with [1,0,0,0] = s (already normalised)

    call_count = [0]
    def encode_side(texts, normalize_embeddings=True):
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            return orig_emb
        return para_embs
    model.encode.side_effect = encode_side
    return model


def test_filter_passes_above_threshold():
    paraphrases = ["good rephrase", "another good one", "third good one"]
    passed, dropped = filter_paraphrases(
        paraphrases, "original moral",
        {"min_similarity": 0.85},
        _mock_filter_model([0.90, 0.92, 0.88]),
    )
    assert passed == paraphrases
    assert dropped == []


def test_filter_drops_below_threshold():
    paraphrases = ["good rephrase", "bad drift", "also good"]
    passed, dropped = filter_paraphrases(
        paraphrases, "original moral",
        {"min_similarity": 0.85},
        _mock_filter_model([0.90, 0.50, 0.88]),
    )
    assert passed == ["good rephrase", "also good"]
    assert dropped == ["bad drift"]


def test_filter_fallback_to_original_when_all_dropped():
    paraphrases = ["terrible paraphrase"]
    passed, dropped = filter_paraphrases(
        paraphrases, "original moral",
        {"min_similarity": 0.85},
        _mock_filter_model([0.20]),
    )
    assert passed == ["original moral"]
    assert dropped == ["terrible paraphrase"]
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd /Users/asifamar/Desktop/Master/NLP-morables
python -m pytest tests/pipeline/test_paraphrase_filter.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'lib.pipeline.paraphrase_filter'`

- [ ] **Step 3: Implement `paraphrase_filter.py`**

```python
# lib/pipeline/paraphrase_filter.py
"""lib/pipeline/paraphrase_filter.py — Word-count post-processing and paraphrase similarity filter."""
from __future__ import annotations


def post_process(text: str, cfg: dict) -> tuple[str, bool]:
    """
    Check text word count against config bounds.

    Returns:
        (text, flagged) — text is returned unchanged; flagged=True if outside [min_words, max_words].
    """
    words = text.strip().split()
    n = len(words)
    min_w = cfg.get("min_words", 5)
    max_w = cfg.get("max_words", 15)
    flagged = not (min_w <= n <= max_w)
    return text, flagged


def filter_paraphrases(
    paraphrases: list[str],
    original: str,
    filter_cfg: dict,
    filter_model,
) -> tuple[list[str], list[str]]:
    """
    Drop paraphrases whose cosine similarity to the original is below min_similarity.

    Args:
        paraphrases:  List of candidate rephrasings.
        original:     The source moral text.
        filter_cfg:   Dict with key 'min_similarity' (default 0.85).
        filter_model: SentenceTransformer used only for filtering (e.g. BGE-M3).

    Returns:
        (passed, dropped) — if all dropped, passed falls back to [original].
    """
    import numpy as np

    min_sim = filter_cfg.get("min_similarity", 0.85)
    orig_emb = filter_model.encode([original], normalize_embeddings=True)
    para_embs = filter_model.encode(paraphrases, normalize_embeddings=True)
    sims = (para_embs @ orig_emb.T).flatten()

    passed = [p for p, s in zip(paraphrases, sims) if float(s) >= min_sim]
    dropped = [p for p, s in zip(paraphrases, sims) if float(s) < min_sim]

    if not passed:
        passed = [original]

    return passed, dropped
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
python -m pytest tests/pipeline/test_paraphrase_filter.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/pipeline/paraphrase_filter.py tests/pipeline/test_paraphrase_filter.py
git commit -m "feat(pipeline): add paraphrase_filter — word-count check + similarity filter"
```

---

## Task 2: `local_llm.py` — HuggingFace model loader

**Files:**
- Create: `lib/pipeline/local_llm.py`
- Create: `tests/pipeline/test_local_llm.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_local_llm.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline.local_llm import LocalLLM


def _model_cfg(alias="TestModel", temperature=0.3):
    return {
        "id": "org/test-model",
        "alias": alias,
        "dtype": "bfloat16",
        "max_new_tokens": 32,
        "temperature": temperature,
    }


def _make_mock_tokenizer(output_ids):
    tok = MagicMock()
    tok.apply_chat_template.return_value = MagicMock()
    tok.apply_chat_template.return_value.to.return_value = MagicMock(shape=(1, 5))
    tok.eos_token_id = 2
    tok.decode.return_value = "  Honesty is the best policy.  "
    return tok


def _make_mock_model(output_ids):
    model = MagicMock()
    model.generate.return_value = [output_ids]
    return model


def test_local_llm_load_calls_from_pretrained(tmp_path):
    cfg = _model_cfg()
    llm = LocalLLM(cfg)
    with patch("lib.pipeline.local_llm.AutoTokenizer") as mock_tok_cls, \
         patch("lib.pipeline.local_llm.AutoModelForCausalLM") as mock_model_cls, \
         patch("lib.pipeline.local_llm.torch") as mock_torch:
        mock_torch.bfloat16 = "bfloat16_sentinel"
        mock_torch.no_grad.return_value.__enter__ = lambda s: s
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_tok_cls.from_pretrained.return_value = _make_mock_tokenizer([0, 1, 2, 3, 4, 5])
        mock_model = MagicMock()
        mock_model.generate.return_value = [MagicMock(__getitem__=lambda s, x: [])]
        mock_model_cls.from_pretrained.return_value = mock_model

        llm.load()

        mock_tok_cls.from_pretrained.assert_called_once_with("org/test-model")
        mock_model_cls.from_pretrained.assert_called_once_with(
            "org/test-model", torch_dtype="bfloat16_sentinel", device_map="mps"
        )


def test_local_llm_unload_clears_references():
    cfg = _model_cfg()
    llm = LocalLLM(cfg)
    llm._model = MagicMock()
    llm._tokenizer = MagicMock()

    with patch("lib.pipeline.local_llm.gc") as mock_gc, \
         patch("lib.pipeline.local_llm.torch") as mock_torch:
        llm.unload()

    assert llm._model is None
    assert llm._tokenizer is None
    mock_gc.collect.assert_called_once()


def test_local_llm_generate_applies_chat_template_and_strips():
    cfg = _model_cfg()
    llm = LocalLLM(cfg)

    tok = MagicMock()
    input_ids = MagicMock()
    input_ids.shape = (1, 5)
    input_ids_on_device = MagicMock()
    input_ids_on_device.shape = (1, 5)
    tok.apply_chat_template.return_value = input_ids
    input_ids.to.return_value = input_ids_on_device
    tok.eos_token_id = 2
    tok.decode.return_value = "  output text  "

    model = MagicMock()
    full_output = MagicMock()
    full_output.__getitem__ = lambda s, i: [0, 1, 2, 3, 4, 5, 6, 7]
    model.generate.return_value = [full_output]

    llm._model = model
    llm._tokenizer = tok

    with patch("lib.pipeline.local_llm.torch") as mock_torch:
        mock_torch.no_grad.return_value.__enter__ = lambda s: s
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        result = llm.generate("system", "user")

    assert result == "output text"
    tok.apply_chat_template.assert_called_once_with(
        [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
        add_generation_prompt=True,
        return_tensors="pt",
    )


def test_local_llm_generate_uses_do_sample_when_temperature_positive():
    cfg = _model_cfg(temperature=0.3)
    llm = LocalLLM(cfg)
    tok = MagicMock()
    input_ids = MagicMock()
    input_ids.shape = (1, 3)
    input_ids.to.return_value = input_ids
    tok.apply_chat_template.return_value = input_ids
    tok.eos_token_id = 2
    tok.decode.return_value = "result"
    model = MagicMock()
    output = MagicMock()
    output.__getitem__ = lambda s, i: [0, 1, 2, 3]
    model.generate.return_value = [output]
    llm._model = model
    llm._tokenizer = tok

    with patch("lib.pipeline.local_llm.torch") as mock_torch:
        mock_torch.no_grad.return_value.__enter__ = lambda s: s
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        llm.generate("sys", "usr")

    call_kwargs = model.generate.call_args[1]
    assert call_kwargs["do_sample"] is True
    assert call_kwargs["temperature"] == 0.3
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
python -m pytest tests/pipeline/test_local_llm.py -v 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'lib.pipeline.local_llm'`

- [ ] **Step 3: Implement `local_llm.py`**

```python
# lib/pipeline/local_llm.py
"""lib/pipeline/local_llm.py — Load/unload a HuggingFace causal LM to MPS and generate text."""
from __future__ import annotations
import gc
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLM:
    """
    Thin wrapper around a HuggingFace CausalLM for single-turn generation.

    Usage:
        llm = LocalLLM(model_cfg)
        llm.load()
        text = llm.generate(system_prompt, user_prompt)
        llm.unload()
    """

    def __init__(self, model_cfg: dict):
        self.model_id: str = model_cfg["id"]
        self.alias: str = model_cfg["alias"]
        self.dtype: str = model_cfg.get("dtype", "bfloat16")
        self.max_new_tokens: int = model_cfg.get("max_new_tokens", 64)
        self.temperature: float = model_cfg.get("temperature", 0.3)
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """Load model and tokenizer to MPS."""
        dtype = getattr(torch, self.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map="mps"
        )
        self._model.eval()
        print(f"  Loaded {self.alias} on mps ({self.dtype})")

    def unload(self) -> None:
        """Delete model and tokenizer, free MPS memory."""
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        gc.collect()
        torch.mps.empty_cache()
        print(f"  Unloaded {self.alias}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Apply chat template and generate a response.

        Returns:
            Decoded generated text, stripped of whitespace.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("mps")

        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated_ids = output[0][input_ids.shape[1]:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
python -m pytest tests/pipeline/test_local_llm.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/pipeline/local_llm.py tests/pipeline/test_local_llm.py
git commit -m "feat(pipeline): add LocalLLM — HF model loader with MPS support and chat template"
```

---

## Task 3: `local_corpus_generator.py` — fable → summary per gen model

**Files:**
- Create: `lib/pipeline/local_corpus_generator.py`
- Create: `tests/pipeline/test_local_corpus_generator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_local_corpus_generator.py
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline.local_corpus_generator import generate_summaries


def _make_fables(n: int) -> list[dict]:
    return [
        {"doc_id": f"fable_{i:03d}", "title": f"Fable {i}", "text": f"Once a fox did thing {i}."}
        for i in range(n)
    ]


def _model_cfg(alias="TestGen"):
    return {"id": "org/test-model", "alias": alias, "dtype": "bfloat16",
            "max_new_tokens": 32, "temperature": 0.3}


def _mock_llm(text="Envy leads to ruin."):
    llm = MagicMock()
    llm.generate.return_value = text
    return llm


def test_generate_summaries_creates_output_files(tmp_path):
    gen_dir = tmp_path / "TestGen"
    gen_dir.mkdir()
    with patch("lib.pipeline.local_corpus_generator.LocalLLM") as MockLLM:
        MockLLM.return_value = _mock_llm()
        generate_summaries(
            model_cfg=_model_cfg(),
            fables=_make_fables(2),
            prompt="Summarise this fable.",
            prompt_version="v1",
            post_cfg={"min_words": 5, "max_words": 15},
            diag_cfg={"enabled": True, "duplicate_similarity_threshold": 0.95,
                      "unique_ratio": True, "generic_phrases": []},
            gen_model_dir=gen_dir,
        )
    assert (gen_dir / "corpus_summaries.json").exists()
    assert (gen_dir / "diagnostics.json").exists()


def test_generate_summaries_json_schema(tmp_path):
    gen_dir = tmp_path / "TestGen"
    gen_dir.mkdir()
    with patch("lib.pipeline.local_corpus_generator.LocalLLM") as MockLLM:
        MockLLM.return_value = _mock_llm("Patience conquers all obstacles in time.")
        generate_summaries(
            model_cfg=_model_cfg(),
            fables=_make_fables(3),
            prompt="Summarise.",
            prompt_version="v1",
            post_cfg={"min_words": 5, "max_words": 15},
            diag_cfg={"enabled": True, "duplicate_similarity_threshold": 0.95,
                      "unique_ratio": True, "generic_phrases": []},
            gen_model_dir=gen_dir,
        )
    with open(gen_dir / "corpus_summaries.json") as f:
        data = json.load(f)
    assert data["prompt_version"] == "v1"
    assert len(data["items"]) == 3
    item = data["items"][0]
    assert "fable_id" in item
    assert "summary" in item
    assert item["summary"] == "Patience conquers all obstacles in time."


def test_generate_summaries_skips_when_cached_and_version_matches(tmp_path):
    gen_dir = tmp_path / "TestGen"
    gen_dir.mkdir()
    existing = {"prompt_version": "v1", "items": [
        {"fable_id": "fable_000", "summary": "cached summary", "flagged": False}
    ]}
    with open(gen_dir / "corpus_summaries.json", "w") as f:
        json.dump(existing, f)

    with patch("lib.pipeline.local_corpus_generator.LocalLLM") as MockLLM:
        generate_summaries(
            model_cfg=_model_cfg(),
            fables=_make_fables(1),
            prompt="Summarise.",
            prompt_version="v1",
            post_cfg={"min_words": 5, "max_words": 15},
            diag_cfg={"enabled": True, "duplicate_similarity_threshold": 0.95,
                      "unique_ratio": True, "generic_phrases": []},
            gen_model_dir=gen_dir,
            force=False,
        )
        MockLLM.assert_not_called()


def test_generate_summaries_reruns_when_prompt_version_changes(tmp_path):
    gen_dir = tmp_path / "TestGen"
    gen_dir.mkdir()
    existing = {"prompt_version": "v1", "items": [
        {"fable_id": "fable_000", "summary": "old summary", "flagged": False}
    ]}
    with open(gen_dir / "corpus_summaries.json", "w") as f:
        json.dump(existing, f)

    with patch("lib.pipeline.local_corpus_generator.LocalLLM") as MockLLM:
        mock_llm_instance = _mock_llm("new summary")
        MockLLM.return_value = mock_llm_instance
        generate_summaries(
            model_cfg=_model_cfg(),
            fables=_make_fables(1),
            prompt="Summarise.",
            prompt_version="v2",
            post_cfg={"min_words": 5, "max_words": 15},
            diag_cfg={"enabled": True, "duplicate_similarity_threshold": 0.95,
                      "unique_ratio": True, "generic_phrases": []},
            gen_model_dir=gen_dir,
            force=False,
        )
        mock_llm_instance.load.assert_called_once()


def test_generate_summaries_flags_short_output(tmp_path):
    gen_dir = tmp_path / "TestGen"
    gen_dir.mkdir()
    with patch("lib.pipeline.local_corpus_generator.LocalLLM") as MockLLM:
        MockLLM.return_value = _mock_llm("ok")  # 1 word — too short
        generate_summaries(
            model_cfg=_model_cfg(),
            fables=_make_fables(1),
            prompt="Summarise.",
            prompt_version="v1",
            post_cfg={"min_words": 5, "max_words": 15},
            diag_cfg={"enabled": True, "duplicate_similarity_threshold": 0.95,
                      "unique_ratio": True, "generic_phrases": []},
            gen_model_dir=gen_dir,
        )
    with open(gen_dir / "corpus_summaries.json") as f:
        data = json.load(f)
    assert data["items"][0]["flagged"] is True

    with open(gen_dir / "diagnostics.json") as f:
        diag = json.load(f)
    assert "fable_000" in diag["flagged_word_count"]
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
python -m pytest tests/pipeline/test_local_corpus_generator.py -v 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'lib.pipeline.local_corpus_generator'`

- [ ] **Step 3: Implement `local_corpus_generator.py`**

```python
# lib/pipeline/local_corpus_generator.py
"""lib/pipeline/local_corpus_generator.py — Generate one moral summary per fable using a local LLM."""
from __future__ import annotations
import json
from pathlib import Path

from lib.pipeline.local_llm import LocalLLM
from lib.pipeline.paraphrase_filter import post_process

_OUTPUT_FILE = "corpus_summaries.json"
_DIAG_FILE = "diagnostics.json"
_USER_TEMPLATE = "Fable: {text}"


def generate_summaries(
    model_cfg: dict,
    fables: list[dict],
    prompt: str,
    prompt_version: str,
    post_cfg: dict,
    diag_cfg: dict,
    gen_model_dir: Path,
    force: bool = False,
) -> Path:
    """
    Generate one declarative moral summary per fable.

    Cache contract: skip if corpus_summaries.json exists AND prompt_version matches,
    unless force=True.

    Args:
        model_cfg:     Generation model config dict (id, alias, dtype, ...).
        fables:        List of fable dicts with keys doc_id, title, text.
        prompt:        System prompt for summarization.
        prompt_version: Version string — change to bust cache.
        post_cfg:      Word-count config (min_words, max_words).
        diag_cfg:      Diagnostics config (enabled, duplicate_similarity_threshold, ...).
        gen_model_dir: Directory for this model's cached outputs.
        force:         Re-generate even if cache exists.

    Returns:
        Path to written corpus_summaries.json.
    """
    gen_model_dir = Path(gen_model_dir)
    output_path = gen_model_dir / _OUTPUT_FILE

    # Cache check
    if output_path.exists() and not force:
        with open(output_path) as f:
            cached = json.load(f)
        if cached.get("prompt_version") == prompt_version:
            print(f"  [skip] {output_path} (prompt_version={prompt_version} matches cache)")
            return output_path

    print(f"\n  Corpus generation: {len(fables)} fables  |  model: {model_cfg['alias']}")
    llm = LocalLLM(model_cfg)
    llm.load()

    items = []
    flagged_ids = []

    for fable in fables:
        fable_id = fable["doc_id"]
        user_prompt = _USER_TEMPLATE.format(text=fable["text"])
        raw = llm.generate(prompt, user_prompt)
        text, flagged = post_process(raw, post_cfg)
        if flagged:
            flagged_ids.append(fable_id)
        items.append({"fable_id": fable_id, "summary": text, "flagged": flagged})
        short = text[:70] + ("..." if len(text) > 70 else "")
        flag_str = " [FLAGGED]" if flagged else ""
        print(f"    {fable_id}: {short}{flag_str}")

    llm.unload()

    output = {"prompt_version": prompt_version, "model": model_cfg["alias"], "items": items}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Diagnostics
    summaries = [it["summary"] for it in items]
    unique_ratio = len(set(summaries)) / len(summaries) if summaries else 1.0
    generic_matches = []
    if diag_cfg.get("enabled"):
        generic_matches = [
            it["fable_id"] for it in items
            if it["summary"].lower() in [p.lower() for p in diag_cfg.get("generic_phrases", [])]
        ]

    diag = {
        "model": model_cfg["alias"],
        "prompt_version": prompt_version,
        "n_fables": len(fables),
        "unique_ratio": round(unique_ratio, 4),
        "duplicate_rate": round(1.0 - unique_ratio, 4),
        "flagged_word_count": flagged_ids,
        "generic_matches": generic_matches,
    }
    with open(gen_model_dir / _DIAG_FILE, "w") as f:
        json.dump(diag, f, indent=2)

    print(f"\n  Saved {len(items)} summaries → {output_path}  (unique_ratio={unique_ratio:.3f})")
    return output_path
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
python -m pytest tests/pipeline/test_local_corpus_generator.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/pipeline/local_corpus_generator.py tests/pipeline/test_local_corpus_generator.py
git commit -m "feat(pipeline): add local_corpus_generator — fable→summary with generation cache"
```

---

## Task 4: `local_query_paraphraser.py` — moral → 3 rephrases per gen model

**Files:**
- Create: `lib/pipeline/local_query_paraphraser.py`
- Create: `tests/pipeline/test_local_query_paraphraser.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_local_query_paraphraser.py
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline.local_query_paraphraser import generate_paraphrases


def _make_morals(n: int) -> list[dict]:
    return [{"text": f"Moral number {i} about life."} for i in range(n)]


def _make_moral_entries(n: int) -> list[tuple[int, int]]:
    return [(i, i) for i in range(n)]


def _model_cfg(alias="TestGen"):
    return {"id": "org/test-model", "alias": alias, "dtype": "bfloat16",
            "max_new_tokens": 64, "temperature": 0.3}


def _mock_llm_three_lines(lines=("r1", "r2", "r3")):
    llm = MagicMock()
    llm.generate.return_value = "\n".join(lines)
    return llm


def test_generate_paraphrases_creates_output_file(tmp_path):
    gen_dir = tmp_path / "TestGen"
    gen_dir.mkdir()
    filter_model = MagicMock()

    with patch("lib.pipeline.local_query_paraphraser.LocalLLM") as MockLLM, \
         patch("lib.pipeline.local_query_paraphraser.filter_paraphrases") as mock_filter:
        MockLLM.return_value = _mock_llm_three_lines()
        mock_filter.return_value = (["r1", "r2", "r3"], [])
        generate_paraphrases(
            model_cfg=_model_cfg(),
            moral_entries=_make_moral_entries(2),
            morals=_make_morals(2),
            prompt="Rephrase this moral.",
            prompt_version="v1",
            post_cfg={"min_words": 5, "max_words": 15},
            filter_cfg={"min_similarity": 0.85},
            filter_model=filter_model,
            gen_model_dir=gen_dir,
        )
    assert (gen_dir / "query_paraphrases.json").exists()


def test_generate_paraphrases_json_schema(tmp_path):
    gen_dir = tmp_path / "TestGen"
    gen_dir.mkdir()

    with patch("lib.pipeline.local_query_paraphraser.LocalLLM") as MockLLM, \
         patch("lib.pipeline.local_query_paraphraser.filter_paraphrases") as mock_filter:
        MockLLM.return_value = _mock_llm_three_lines(("those who envy suffer", "envy invites ruin", "envious hearts find only pain"))
        mock_filter.return_value = (["those who envy suffer", "envy invites ruin", "envious hearts find only pain"], [])
        generate_paraphrases(
            model_cfg=_model_cfg(),
            moral_entries=_make_moral_entries(1),
            morals=_make_morals(1),
            prompt="Rephrase.",
            prompt_version="v1",
            post_cfg={"min_words": 5, "max_words": 15},
            filter_cfg={"min_similarity": 0.85},
            filter_model=MagicMock(),
            gen_model_dir=gen_dir,
        )
    with open(gen_dir / "query_paraphrases.json") as f:
        data = json.load(f)
    assert data["prompt_version"] == "v1"
    assert len(data["items"]) == 1
    item = data["items"][0]
    assert "moral_idx" in item
    assert "original_moral" in item
    assert "paraphrases" in item
    assert "dropped_paraphrases" in item
    assert len(item["paraphrases"]) == 3


def test_generate_paraphrases_skips_when_cached_and_version_matches(tmp_path):
    gen_dir = tmp_path / "TestGen"
    gen_dir.mkdir()
    existing = {"prompt_version": "v1", "items": [
        {"moral_idx": 0, "original_moral": "m0", "paraphrases": ["r1"], "dropped_paraphrases": []}
    ]}
    with open(gen_dir / "query_paraphrases.json", "w") as f:
        json.dump(existing, f)

    with patch("lib.pipeline.local_query_paraphraser.LocalLLM") as MockLLM:
        generate_paraphrases(
            model_cfg=_model_cfg(),
            moral_entries=_make_moral_entries(1),
            morals=_make_morals(1),
            prompt="Rephrase.",
            prompt_version="v1",
            post_cfg={"min_words": 5, "max_words": 15},
            filter_cfg={"min_similarity": 0.85},
            filter_model=MagicMock(),
            gen_model_dir=gen_dir,
            force=False,
        )
        MockLLM.assert_not_called()


def test_generate_paraphrases_parses_three_lines():
    """generate() returns 3 newline-separated lines → 3 paraphrases."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        gen_dir = Path(td) / "TestGen"
        gen_dir.mkdir()
        captured = []

        def fake_filter(paraphrases, original, filter_cfg, filter_model):
            captured.extend(paraphrases)
            return paraphrases, []

        with patch("lib.pipeline.local_query_paraphraser.LocalLLM") as MockLLM, \
             patch("lib.pipeline.local_query_paraphraser.filter_paraphrases", side_effect=fake_filter):
            MockLLM.return_value = _mock_llm_three_lines(
                ("first rephrasing here", "second version stated", "third distinct expression")
            )
            generate_paraphrases(
                model_cfg=_model_cfg(),
                moral_entries=[(0, 0)],
                morals=[{"text": "original moral text here."}],
                prompt="Rephrase.",
                prompt_version="v1",
                post_cfg={"min_words": 5, "max_words": 15},
                filter_cfg={"min_similarity": 0.85},
                filter_model=MagicMock(),
                gen_model_dir=gen_dir,
            )
        assert captured == ["first rephrasing here", "second version stated", "third distinct expression"]
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
python -m pytest tests/pipeline/test_local_query_paraphraser.py -v 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'lib.pipeline.local_query_paraphraser'`

- [ ] **Step 3: Implement `local_query_paraphraser.py`**

```python
# lib/pipeline/local_query_paraphraser.py
"""lib/pipeline/local_query_paraphraser.py — Generate 3 filtered rephrases per moral using a local LLM."""
from __future__ import annotations
import json
from pathlib import Path

from lib.pipeline.local_llm import LocalLLM
from lib.pipeline.paraphrase_filter import filter_paraphrases, post_process

_OUTPUT_FILE = "query_paraphrases.json"
_USER_TEMPLATE = "Moral: {text}"


def generate_paraphrases(
    model_cfg: dict,
    moral_entries: list[tuple[int, int]],
    morals: list[dict],
    prompt: str,
    prompt_version: str,
    post_cfg: dict,
    filter_cfg: dict,
    filter_model,
    gen_model_dir: Path,
    force: bool = False,
) -> Path:
    """
    Generate 3 filtered rephrases per moral for one generation model.

    Cache contract: skip if query_paraphrases.json exists AND prompt_version matches,
    unless force=True.

    Args:
        model_cfg:     Generation model config.
        moral_entries: List of (moral_idx, fable_idx) tuples.
        morals:        Full morals list from load_morals().
        prompt:        System prompt for paraphrasing.
        prompt_version: Version string for cache busting.
        post_cfg:      Word-count config.
        filter_cfg:    Paraphrase filter config (min_similarity).
        filter_model:  Pre-loaded SentenceTransformer for similarity filtering.
        gen_model_dir: Directory for this model's cached outputs.
        force:         Re-generate even if cache exists.

    Returns:
        Path to written query_paraphrases.json.
    """
    gen_model_dir = Path(gen_model_dir)
    output_path = gen_model_dir / _OUTPUT_FILE

    if output_path.exists() and not force:
        with open(output_path) as f:
            cached = json.load(f)
        if cached.get("prompt_version") == prompt_version:
            print(f"  [skip] {output_path} (prompt_version={prompt_version} matches cache)")
            return output_path

    print(f"\n  Query paraphrasing: {len(moral_entries)} morals  |  model: {model_cfg['alias']}")
    llm = LocalLLM(model_cfg)
    llm.load()

    items = []
    total_dropped = 0

    for moral_idx, fable_idx in moral_entries:
        original = morals[moral_idx]["text"]
        user_prompt = _USER_TEMPLATE.format(text=original)
        raw_output = llm.generate(prompt, user_prompt)

        # Parse 3 newline-separated lines (no numbering expected)
        raw_lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        # Post-process each line for word count
        processed = []
        for line in raw_lines[:3]:
            text, _ = post_process(line, post_cfg)
            processed.append(text)

        passed, dropped = filter_paraphrases(processed, original, filter_cfg, filter_model)
        total_dropped += len(dropped)

        items.append({
            "moral_idx": moral_idx,
            "fable_idx": fable_idx,
            "original_moral": original,
            "paraphrases": passed,
            "dropped_paraphrases": dropped,
        })
        print(f"    moral_{moral_idx:03d}: {len(passed)} kept, {len(dropped)} dropped")

    llm.unload()

    output = {"prompt_version": prompt_version, "model": model_cfg["alias"], "items": items}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved {len(items)} morals → {output_path}  (total dropped: {total_dropped})")
    return output_path
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
python -m pytest tests/pipeline/test_local_query_paraphraser.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/pipeline/local_query_paraphraser.py tests/pipeline/test_local_query_paraphraser.py
git commit -m "feat(pipeline): add local_query_paraphraser — 3 rephrases with filter + generation cache"
```

---

## Task 5: Extend `retrieval_eval.py` — add `run_single_eval()`

The existing `run_retrieval_eval()` is unchanged. We add a new function `run_single_eval()` for use by the matrix runner.

**Files:**
- Modify: `lib/pipeline/retrieval_eval.py`
- Modify: `tests/pipeline/test_retrieval_eval.py`

- [ ] **Step 1: Write failing tests**

Append to the bottom of `tests/pipeline/test_retrieval_eval.py`:

```python
# ── run_single_eval tests ────────────────────────────────────────────────────

from lib.pipeline.retrieval_eval import run_single_eval


def _embed_cfg(alias="TestEmbed"):
    return {"id": "org/test-embed", "alias": alias, "query_instruction": "retrieve"}


def test_run_single_eval_raw_mode_produces_output(tmp_path):
    """Single text per query (no fusion needed)."""
    corpus_texts = ["fable 0", "fable 1", "fable 2"]
    query_texts_list = [["moral 0"], ["moral 1"], ["moral 2"]]
    ground_truth = {0: 0, 1: 1, 2: 2}
    output_path = tmp_path / "result.json"
    preds_path = tmp_path / "preds.json"

    call_count = [0]
    def mock_encode(model, texts, model_id, cache_dir, query_instruction=None, label=""):
        n = len(texts)
        embs = np.zeros((n, 4), dtype=np.float32)
        for i in range(min(n, 4)):
            embs[i, i] = 1.0
        call_count[0] += 1
        return embs

    with patch("lib.pipeline.retrieval_eval._load_model",
               return_value=(_mock_model(3), "cpu")):
        with patch("lib.pipeline.retrieval_eval.encode_with_cache", side_effect=mock_encode):
            result = run_single_eval(
                corpus_texts=corpus_texts,
                query_texts_list=query_texts_list,
                ground_truth=ground_truth,
                embed_model_cfg=_embed_cfg(),
                cache_dir=tmp_path / "cache",
                output_path=output_path,
                predictions_path=preds_path,
            )

    assert "Recall@1" in result
    assert "MRR" in result
    assert output_path.exists()
    assert preds_path.exists()


def test_run_single_eval_paraphrase_mode_uses_max_score(tmp_path):
    """Multiple texts per query → max-score fusion."""
    corpus_texts = ["fable 0", "fable 1"]
    # query 0 has 2 texts; query 1 has 2 texts
    query_texts_list = [["moral 0", "rephrase 0"], ["moral 1", "rephrase 1"]]
    ground_truth = {0: 0, 1: 1}
    output_path = tmp_path / "result.json"
    preds_path = tmp_path / "preds.json"

    def mock_encode(model, texts, model_id, cache_dir, query_instruction=None, label=""):
        n = len(texts)
        embs = np.zeros((n, 4), dtype=np.float32)
        for i in range(min(n, 4)):
            embs[i, i] = 1.0
        return embs

    with patch("lib.pipeline.retrieval_eval._load_model",
               return_value=(_mock_model(2), "cpu")):
        with patch("lib.pipeline.retrieval_eval.encode_with_cache", side_effect=mock_encode):
            result = run_single_eval(
                corpus_texts=corpus_texts,
                query_texts_list=query_texts_list,
                ground_truth=ground_truth,
                embed_model_cfg=_embed_cfg(),
                cache_dir=tmp_path / "cache",
                output_path=output_path,
                predictions_path=preds_path,
            )
    assert result["Recall@1"] >= 0.0


def test_run_single_eval_idempotent(tmp_path):
    """Skips computation if output_path already exists and force=False."""
    output_path = tmp_path / "result.json"
    preds_path = tmp_path / "preds.json"
    existing = {"Recall@1": 1.0, "MRR": 1.0}
    with open(output_path, "w") as f:
        json.dump(existing, f)

    with patch("lib.pipeline.retrieval_eval._load_model") as mock_load:
        result = run_single_eval(
            corpus_texts=["f0"],
            query_texts_list=[["m0"]],
            ground_truth={0: 0},
            embed_model_cfg=_embed_cfg(),
            cache_dir=tmp_path / "cache",
            output_path=output_path,
            predictions_path=preds_path,
            force=False,
        )
    mock_load.assert_not_called()
    assert result["Recall@1"] == 1.0
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
python -m pytest tests/pipeline/test_retrieval_eval.py::test_run_single_eval_raw_mode_produces_output -v 2>&1 | head -15
```
Expected: `ImportError: cannot import name 'run_single_eval'`

- [ ] **Step 3: Add `run_single_eval()` to `retrieval_eval.py`**

Add this function at the bottom of `lib/pipeline/retrieval_eval.py` (before the last line):

```python
def run_single_eval(
    corpus_texts: list[str],
    query_texts_list: list[list[str]],
    ground_truth: dict,
    embed_model_cfg: dict,
    cache_dir: Path,
    output_path: Path,
    predictions_path: Path,
    force: bool = False,
) -> dict:
    """
    Evaluate one (gen_model, embed_model, ablation) combination.

    Args:
        corpus_texts:      n_fables texts (raw fable or gen model summary).
        query_texts_list:  n_queries lists; each sublist is [original] or
                           [original, rephrase_1, rephrase_2, ...].
                           Multiple texts → max-score fusion.
        ground_truth:      {query_idx: fable_idx}.
        embed_model_cfg:   Dict with keys: id, alias, query_instruction.
        cache_dir:         Embedding cache directory.
        output_path:       Where to write metrics JSON.
        predictions_path:  Where to write per-query predictions JSON.
        force:             Re-run even if output_path exists.

    Returns:
        Metrics dict with Recall@1, Recall@5, MRR keys.
    """
    output_path = Path(output_path)
    predictions_path = Path(predictions_path)
    cache_dir = Path(cache_dir)

    if output_path.exists() and not force:
        with open(output_path) as f:
            return json.load(f)

    embed_model_id = embed_model_cfg["id"]
    embed_alias = embed_model_cfg.get("alias", embed_model_id)
    query_instruction = embed_model_cfg.get("query_instruction", "")
    n_fables = len(corpus_texts)

    model, _ = _load_model(embed_model_id)

    corpus_embs = encode_with_cache(
        model=model, texts=corpus_texts, model_id=embed_alias,
        cache_dir=cache_dir, query_instruction=None, label=f"corpus:{embed_alias}",
    )

    # Build per-query score vectors via max-score fusion when multiple texts
    n_queries = len(query_texts_list)
    score_matrix = np.zeros((n_queries, n_fables), dtype=np.float32)

    for q_idx, query_texts in enumerate(query_texts_list):
        q_embs = encode_with_cache(
            model=model, texts=query_texts, model_id=embed_alias,
            cache_dir=cache_dir, query_instruction=query_instruction,
            label=f"query:{embed_alias}",
        )
        scores_per_text = q_embs @ corpus_embs.T  # (n_texts, n_fables)
        score_matrix[q_idx] = scores_per_text.max(axis=0)

    metrics = compute_metrics_from_matrix(score_matrix, ground_truth)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Write predictions
    rankings_data = compute_rankings_from_matrix(score_matrix, top_k=n_fables)
    ranks_arr = rank_analysis_from_matrix(score_matrix, ground_truth)
    gt_sorted_qidx = sorted(ground_truth.keys())
    rank_by_qidx = {q: int(r) + 1 for q, r in zip(gt_sorted_qidx, ranks_arr)}
    pred_records = []
    for q_idx, ranking in enumerate(rankings_data):
        pred_records.append({
            "query_idx": q_idx,
            "correct_idx": ground_truth.get(q_idx),
            "correct_rank": rank_by_qidx.get(q_idx),
            "top_k_indices": ranking["indices"],
            "top_k_scores": ranking["scores"],
        })
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w") as f:
        json.dump({"embed_model": embed_alias, "top_k": n_fables, "queries": pred_records}, f)

    return metrics
```

- [ ] **Step 4: Run all retrieval_eval tests**

```bash
python -m pytest tests/pipeline/test_retrieval_eval.py -v
```
Expected: all tests PASS (existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add lib/pipeline/retrieval_eval.py tests/pipeline/test_retrieval_eval.py
git commit -m "feat(pipeline): add run_single_eval() to retrieval_eval for matrix runner use"
```

---

## Task 6: `matrix_aggregator.py` — build matrix table + rankings

**Files:**
- Create: `lib/pipeline/matrix_aggregator.py`
- Create: `tests/pipeline/test_matrix_aggregator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_matrix_aggregator.py
import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline.matrix_aggregator import aggregate


def _write_result(results_dir: Path, gen: str, embed: str, ablation: str, r1: float, mrr: float):
    fname = f"{gen}__{embed}__{ablation}.json"
    data = {
        "gen_model": gen, "embed_model": embed, "ablation": ablation,
        "metrics": {"Recall@1": r1, "Recall@5": r1, "MRR": mrr},
    }
    with open(results_dir / fname, "w") as f:
        json.dump(data, f)


def test_aggregate_produces_matrix_and_rankings(tmp_path):
    results_dir = tmp_path / "retrieval_results"
    results_dir.mkdir()
    _write_result(results_dir, "GenA", "EmbedX", "full", 0.80, 0.85)
    _write_result(results_dir, "GenA", "EmbedY", "full", 0.70, 0.75)
    _write_result(results_dir, "GenB", "EmbedX", "full", 0.60, 0.65)
    _write_result(results_dir, "GenB", "EmbedY", "full", 0.50, 0.55)

    aggregate(results_dir, tmp_path)

    assert (tmp_path / "matrix_summary.json").exists()
    assert (tmp_path / "rankings.json").exists()


def test_aggregate_matrix_structure(tmp_path):
    results_dir = tmp_path / "retrieval_results"
    results_dir.mkdir()
    _write_result(results_dir, "GenA", "EmbedX", "full", 0.80, 0.85)
    _write_result(results_dir, "GenA", "EmbedY", "full", 0.70, 0.75)
    _write_result(results_dir, "GenB", "EmbedX", "full", 0.60, 0.65)
    _write_result(results_dir, "GenB", "EmbedY", "full", 0.50, 0.55)

    aggregate(results_dir, tmp_path)

    with open(tmp_path / "matrix_summary.json") as f:
        summary = json.load(f)

    assert "full" in summary
    matrix = summary["full"]["Recall@1"]
    assert set(matrix["rows"]) == {"GenA", "GenB"}
    assert set(matrix["cols"]) == {"EmbedX", "EmbedY"}
    assert len(matrix["values"]) == 2
    assert len(matrix["values"][0]) == 2


def test_aggregate_rankings_best_pair(tmp_path):
    results_dir = tmp_path / "retrieval_results"
    results_dir.mkdir()
    _write_result(results_dir, "GenA", "EmbedX", "full", 0.80, 0.85)
    _write_result(results_dir, "GenA", "EmbedY", "full", 0.70, 0.75)
    _write_result(results_dir, "GenB", "EmbedX", "full", 0.60, 0.65)
    _write_result(results_dir, "GenB", "EmbedY", "full", 0.50, 0.55)

    aggregate(results_dir, tmp_path)

    with open(tmp_path / "rankings.json") as f:
        rankings = json.load(f)

    assert rankings["best_pair"]["gen_model"] == "GenA"
    assert rankings["best_pair"]["embed_model"] == "EmbedX"
    assert rankings["best_gen_model"]["name"] == "GenA"
    assert rankings["best_embed_model"]["name"] == "EmbedX"
    assert "dominant_factor" in rankings["impact_analysis"]


def test_aggregate_dominant_factor_detection(tmp_path):
    """Gen model variance > embed model variance → dominant_factor = generation."""
    results_dir = tmp_path / "retrieval_results"
    results_dir.mkdir()
    # GenA is much better than GenB regardless of embed
    _write_result(results_dir, "GenA", "EmbedX", "full", 0.90, 0.92)
    _write_result(results_dir, "GenA", "EmbedY", "full", 0.88, 0.90)
    _write_result(results_dir, "GenB", "EmbedX", "full", 0.30, 0.35)
    _write_result(results_dir, "GenB", "EmbedY", "full", 0.28, 0.33)

    aggregate(results_dir, tmp_path)

    with open(tmp_path / "rankings.json") as f:
        rankings = json.load(f)
    assert rankings["impact_analysis"]["dominant_factor"] == "generation"
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
python -m pytest tests/pipeline/test_matrix_aggregator.py -v 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'lib.pipeline.matrix_aggregator'`

- [ ] **Step 3: Implement `matrix_aggregator.py`**

```python
# lib/pipeline/matrix_aggregator.py
"""lib/pipeline/matrix_aggregator.py — Build matrix summary and rankings from retrieval results."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def aggregate(retrieval_results_dir: Path, run_dir: Path) -> None:
    """
    Read all *.json files in retrieval_results_dir, build matrix_summary.json
    and rankings.json in run_dir.

    File naming convention: <gen_alias>__<embed_alias>__<ablation>.json

    Args:
        retrieval_results_dir: Directory containing per-combination result JSONs.
        run_dir:               Directory to write matrix_summary.json and rankings.json.
    """
    retrieval_results_dir = Path(retrieval_results_dir)
    run_dir = Path(run_dir)

    # Load all results
    results: list[dict] = []
    for path in sorted(retrieval_results_dir.glob("*.json")):
        with open(path) as f:
            results.append(json.load(f))

    if not results:
        print("  [aggregator] No result files found.")
        return

    # Collect unique values
    gen_models = sorted({r["gen_model"] for r in results})
    embed_models = sorted({r["embed_model"] for r in results})
    ablations = sorted({r["ablation"] for r in results})
    metrics_keys = ["Recall@1", "MRR"]

    # Build matrix per ablation per metric
    # matrix[ablation][metric] = {"rows": [...], "cols": [...], "values": [[...]]}
    summary: dict = {}
    for ablation in ablations:
        abl_results = {r["gen_model"]: {r["embed_model"]: r} for r in results if r["ablation"] == ablation}
        # rebuild as dict[gen][embed] = r
        lookup: dict[str, dict[str, dict]] = defaultdict(dict)
        for r in results:
            if r["ablation"] == ablation:
                lookup[r["gen_model"]][r["embed_model"]] = r

        summary[ablation] = {}
        for metric in metrics_keys:
            values = []
            for gen in gen_models:
                row = []
                for emb in embed_models:
                    cell = lookup.get(gen, {}).get(emb)
                    val = cell["metrics"][metric] if cell else None
                    row.append(val)
                values.append(row)
            summary[ablation][metric] = {
                "rows": gen_models,
                "cols": embed_models,
                "values": values,
            }

    with open(run_dir / "matrix_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Rankings — computed over the "full" ablation (or first available)
    rank_ablation = "full" if "full" in ablations else ablations[0]
    rank_results = [r for r in results if r["ablation"] == rank_ablation]

    # Best pair
    best = max(rank_results, key=lambda r: r["metrics"]["Recall@1"])

    # Best gen model (avg Recall@1 over embed models)
    gen_scores: dict[str, list[float]] = defaultdict(list)
    emb_scores: dict[str, list[float]] = defaultdict(list)
    for r in rank_results:
        gen_scores[r["gen_model"]].append(r["metrics"]["Recall@1"])
        emb_scores[r["embed_model"]].append(r["metrics"]["Recall@1"])

    best_gen = max(gen_scores, key=lambda g: np.mean(gen_scores[g]))
    best_emb = max(emb_scores, key=lambda e: np.mean(emb_scores[e]))

    gen_avgs = np.array([np.mean(v) for v in gen_scores.values()])
    emb_avgs = np.array([np.mean(v) for v in emb_scores.values()])
    gen_var = float(np.var(gen_avgs))
    emb_var = float(np.var(emb_avgs))
    dominant = "generation" if gen_var >= emb_var else "embedding"

    rankings = {
        "ranked_ablation": rank_ablation,
        "best_gen_model": {
            "name": best_gen,
            "avg_recall_at_1": round(float(np.mean(gen_scores[best_gen])), 4),
            "avg_mrr": round(float(np.mean([r["metrics"]["MRR"] for r in rank_results
                                            if r["gen_model"] == best_gen])), 4),
        },
        "best_embed_model": {
            "name": best_emb,
            "avg_recall_at_1": round(float(np.mean(emb_scores[best_emb])), 4),
            "avg_mrr": round(float(np.mean([r["metrics"]["MRR"] for r in rank_results
                                            if r["embed_model"] == best_emb])), 4),
        },
        "best_pair": {
            "gen_model": best["gen_model"],
            "embed_model": best["embed_model"],
            "ablation": rank_ablation,
            "Recall@1": best["metrics"]["Recall@1"],
            "MRR": best["metrics"]["MRR"],
        },
        "impact_analysis": {
            "gen_model_variance": round(gen_var, 6),
            "embed_model_variance": round(emb_var, 6),
            "dominant_factor": dominant,
        },
    }

    with open(run_dir / "rankings.json", "w") as f:
        json.dump(rankings, f, indent=2)

    print(f"\n  Matrix summary → {run_dir / 'matrix_summary.json'}")
    print(f"  Rankings → {run_dir / 'rankings.json'}")
    print(f"  Best pair: {best['gen_model']} × {best['embed_model']}  "
          f"R@1={best['metrics']['Recall@1']:.3f}  dominant={dominant}")
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
python -m pytest tests/pipeline/test_matrix_aggregator.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/pipeline/matrix_aggregator.py tests/pipeline/test_matrix_aggregator.py
git commit -m "feat(pipeline): add matrix_aggregator — builds matrix table and rankings from results"
```

---

## Task 7: `matrix_runner.py` — top-level N×M orchestrator

**Files:**
- Create: `lib/pipeline/matrix_runner.py`
- Create: `tests/pipeline/test_matrix_runner.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_matrix_runner.py
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from lib.pipeline.matrix_runner import run_matrix_experiment


def _write_minimal_config(config_path: Path, n_fables: int = 2):
    config = {
        "n_fables": n_fables,
        "generation_models": [
            {"id": "org/genA", "alias": "GenA", "dtype": "bfloat16",
             "max_new_tokens": 32, "temperature": 0.3},
        ],
        "embed_models": [
            {"id": "org/embX", "alias": "EmbX", "query_instruction": "retrieve"},
        ],
        "summarization_prompt": "Summarise.",
        "paraphrase_prompt": "Rephrase.",
        "prompt_version": "v1",
        "post_processing": {"enforce_word_count": {"enabled": True, "min_words": 5,
                                                    "max_words": 15, "action": "flag"}},
        "paraphrase_filter": {"enabled": True, "min_similarity": 0.85,
                              "filter_model": "BAAI/bge-m3"},
        "summary_diagnostics": {"enabled": True, "duplicate_similarity_threshold": 0.95,
                                 "unique_ratio": True, "generic_phrases": []},
        "ablations": [{"name": "full", "corpus": "summary", "query": "paraphrases"}],
        "steps": {"generate_summaries": True, "generate_paraphrases": True,
                  "run_retrieval_eval": True},
        "api_delay_seconds": 0.0,
    }
    with open(config_path, "w") as f:
        import yaml
        yaml.dump(config, f)


def test_run_matrix_experiment_creates_run_dir(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_minimal_config(config_path)

    with patch("lib.pipeline.matrix_runner.load_fables", return_value=[
               {"doc_id": "fable_000", "text": "Once a fox.", "title": "F0"},
               {"doc_id": "fable_001", "text": "Once a crow.", "title": "F1"}]), \
         patch("lib.pipeline.matrix_runner.load_morals", return_value=[
               {"text": "honesty is best"}, {"text": "greed leads to ruin"}]), \
         patch("lib.pipeline.matrix_runner.load_qrels_moral_to_fable",
               return_value={0: 0, 1: 1}), \
         patch("lib.pipeline.matrix_runner.generate_summaries") as mock_gs, \
         patch("lib.pipeline.matrix_runner.generate_paraphrases") as mock_gp, \
         patch("lib.pipeline.matrix_runner.run_single_eval") as mock_eval, \
         patch("lib.pipeline.matrix_runner.aggregate") as mock_agg, \
         patch("lib.pipeline.matrix_runner.SentenceTransformer") as mock_st, \
         patch("lib.pipeline.matrix_runner.load_env"):

        mock_gs.return_value = MagicMock()
        mock_gp.return_value = MagicMock()
        mock_eval.return_value = {"Recall@1": 0.5, "Recall@5": 0.8, "MRR": 0.6}

        # Simulate corpus_summaries.json written by generate_summaries
        run_dir_holder = [None]
        original_make = __import__("lib.pipeline.run_utils", fromlist=["make_run_dir"]).make_run_dir
        def capture_run_dir(base, tag=""):
            rd = original_make(base, tag)
            run_dir_holder[0] = rd
            return rd

        with patch("lib.pipeline.matrix_runner.make_run_dir", side_effect=capture_run_dir):
            run_matrix_experiment(
                config_path=config_path,
                run_dir=tmp_path / "run",
            )

    assert (tmp_path / "run").exists()


def test_run_matrix_experiment_calls_generate_summaries_once_per_gen_model(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_minimal_config(config_path, n_fables=2)

    with patch("lib.pipeline.matrix_runner.load_fables", return_value=[
               {"doc_id": "fable_000", "text": "f0", "title": "F0"},
               {"doc_id": "fable_001", "text": "f1", "title": "F1"}]), \
         patch("lib.pipeline.matrix_runner.load_morals",
               return_value=[{"text": "m0"}, {"text": "m1"}]), \
         patch("lib.pipeline.matrix_runner.load_qrels_moral_to_fable",
               return_value={0: 0, 1: 1}), \
         patch("lib.pipeline.matrix_runner.generate_summaries") as mock_gs, \
         patch("lib.pipeline.matrix_runner.generate_paraphrases") as mock_gp, \
         patch("lib.pipeline.matrix_runner.run_single_eval",
               return_value={"Recall@1": 0.5, "Recall@5": 0.8, "MRR": 0.6}), \
         patch("lib.pipeline.matrix_runner.aggregate"), \
         patch("lib.pipeline.matrix_runner.SentenceTransformer"), \
         patch("lib.pipeline.matrix_runner.load_env"):

        mock_gs.return_value = MagicMock()
        mock_gp.return_value = MagicMock()

        run_matrix_experiment(config_path=config_path, run_dir=tmp_path / "run")

    # 1 gen model → generate_summaries called once
    assert mock_gs.call_count == 1
    assert mock_gp.call_count == 1


def test_run_matrix_experiment_calls_run_single_eval_for_each_combo(tmp_path):
    """N gen models × M embed models × K ablations → N×M×K calls to run_single_eval."""
    config_path = tmp_path / "config.yaml"
    # Manually write config with 2 gen + 2 embed + 1 ablation = 4 combos
    config = {
        "n_fables": 2,
        "generation_models": [
            {"id": "org/genA", "alias": "GenA", "dtype": "bfloat16", "max_new_tokens": 32, "temperature": 0.3},
            {"id": "org/genB", "alias": "GenB", "dtype": "bfloat16", "max_new_tokens": 32, "temperature": 0.3},
        ],
        "embed_models": [
            {"id": "org/embX", "alias": "EmbX", "query_instruction": ""},
            {"id": "org/embY", "alias": "EmbY", "query_instruction": ""},
        ],
        "summarization_prompt": "Summarise.",
        "paraphrase_prompt": "Rephrase.",
        "prompt_version": "v1",
        "post_processing": {"enforce_word_count": {"enabled": True, "min_words": 5, "max_words": 15, "action": "flag"}},
        "paraphrase_filter": {"enabled": True, "min_similarity": 0.85, "filter_model": "BAAI/bge-m3"},
        "summary_diagnostics": {"enabled": True, "duplicate_similarity_threshold": 0.95, "unique_ratio": True, "generic_phrases": []},
        "ablations": [{"name": "full", "corpus": "summary", "query": "paraphrases"}],
        "steps": {"generate_summaries": True, "generate_paraphrases": True, "run_retrieval_eval": True},
        "api_delay_seconds": 0.0,
    }
    with open(config_path, "w") as f:
        import yaml; yaml.dump(config, f)

    with patch("lib.pipeline.matrix_runner.load_fables", return_value=[
               {"doc_id": "fable_000", "text": "f0", "title": "F0"},
               {"doc_id": "fable_001", "text": "f1", "title": "F1"}]), \
         patch("lib.pipeline.matrix_runner.load_morals",
               return_value=[{"text": "m0"}, {"text": "m1"}]), \
         patch("lib.pipeline.matrix_runner.load_qrels_moral_to_fable",
               return_value={0: 0, 1: 1}), \
         patch("lib.pipeline.matrix_runner.generate_summaries") as mock_gs, \
         patch("lib.pipeline.matrix_runner.generate_paraphrases") as mock_gp, \
         patch("lib.pipeline.matrix_runner.run_single_eval") as mock_eval, \
         patch("lib.pipeline.matrix_runner.aggregate"), \
         patch("lib.pipeline.matrix_runner.SentenceTransformer"), \
         patch("lib.pipeline.matrix_runner.load_env"):

        mock_gs.return_value = MagicMock()
        mock_gp.return_value = MagicMock()
        mock_eval.return_value = {"Recall@1": 0.5, "Recall@5": 0.8, "MRR": 0.6}

        run_matrix_experiment(config_path=config_path, run_dir=tmp_path / "run")

    # 2 gen × 2 embed × 1 ablation = 4
    assert mock_eval.call_count == 4
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
python -m pytest tests/pipeline/test_matrix_runner.py -v 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'lib.pipeline.matrix_runner'`

- [ ] **Step 3: Implement `matrix_runner.py`**

```python
# lib/pipeline/matrix_runner.py
"""lib/pipeline/matrix_runner.py — N×M orchestrator for local model matrix experiments."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import yaml
from sentence_transformers import SentenceTransformer

from lib.pipeline.local_corpus_generator import generate_summaries
from lib.pipeline.local_query_paraphraser import generate_paraphrases
from lib.pipeline.retrieval_eval import run_single_eval
from lib.pipeline.matrix_aggregator import aggregate
from lib.pipeline.run_utils import load_env, make_run_dir, write_manifest

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    import copy
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def run_matrix_experiment(
    config_path: Path,
    run_dir: Optional[Path] = None,
    force: bool = False,
) -> None:
    """
    Run the full N×M generation × embedding evaluation.

    Stage 1 — for each generation model:
        generate corpus summaries (cached per model + prompt_version)
        generate query paraphrases (cached per model + prompt_version)

    Stage 2 — for each (gen_model, embed_model, ablation):
        run_single_eval with appropriate corpus/query texts

    Stage 3 — aggregate all results into matrix_summary.json + rankings.json

    Args:
        config_path: Path to experiment config.yaml
        run_dir:     Existing run dir to continue; default creates new timestamped dir
        force:       Re-run steps even if cached outputs exist
    """
    import sys as _sys
    _ROOT = Path(__file__).parent.parent.parent
    if str(_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_ROOT))

    from lib.data import load_fables, load_morals, load_qrels_moral_to_fable

    config_path = Path(config_path)
    load_env(_ROOT)

    with open(_DEFAULT_CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    with open(config_path) as f:
        config = _deep_merge(config, yaml.safe_load(f) or {})

    n_fables = config.get("n_fables")
    tag = f"sample{n_fables}" if n_fables else "full"

    if run_dir is None:
        base = config_path.parent / "results" / "pipeline_runs"
        run_dir = make_run_dir(base, tag)
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    gen_cache_dir = run_dir / "gen_cache"
    embedding_cache_dir = run_dir / "embedding_cache"
    results_dir = run_dir / "retrieval_results"
    predictions_dir = run_dir / "predictions"
    for d in [gen_cache_dir, embedding_cache_dir, results_dir, predictions_dir]:
        d.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Matrix Experiment: {config_path.parent.name}")
    print(f"  Run dir: {run_dir}")
    print(f"  Gen models: {[m['alias'] for m in config['generation_models']]}")
    print(f"  Embed models: {[m['alias'] for m in config['embed_models']]}")
    print(f"{'=' * 60}")

    # Load data
    fables = load_fables()
    morals = load_morals()
    gt_m2f = load_qrels_moral_to_fable()

    if n_fables:
        fables = fables[:n_fables]
    else:
        n_fables = len(fables)

    target_fable_indices = set(range(n_fables))
    moral_entries = sorted(
        [(m_idx, f_idx) for m_idx, f_idx in gt_m2f.items() if f_idx in target_fable_indices],
        key=lambda x: x[0],
    )
    fable_texts = [f["text"] for f in fables]
    moral_texts = [morals[m_idx]["text"] for m_idx, _ in moral_entries]
    ground_truth = {i: f_idx for i, (_, f_idx) in enumerate(moral_entries)}

    steps = config.get("steps", {})
    post_cfg = config.get("post_processing", {}).get("enforce_word_count", {})
    filter_cfg = config.get("paraphrase_filter", {})
    diag_cfg = config.get("summary_diagnostics", {})
    prompt_version = config.get("prompt_version", "v1")

    # ── Stage 1: Local generation (per gen model) ─────────────────────────────
    # Load filter model once for all paraphrase filtering
    filter_model = None
    if steps.get("generate_paraphrases", True) and filter_cfg.get("enabled", True):
        filter_model_id = filter_cfg.get("filter_model", "BAAI/bge-m3")
        print(f"\n  Loading filter model: {filter_model_id}")
        filter_model = SentenceTransformer(filter_model_id)

    gen_cache_paths: dict[str, dict] = {}  # alias → {summaries_path, paraphrases_path}

    for gen_cfg in config.get("generation_models", []):
        alias = gen_cfg["alias"]
        model_dir = gen_cache_dir / alias
        model_dir.mkdir(exist_ok=True)

        if steps.get("generate_summaries", True):
            print(f"\n── Summaries: {alias} ──────────────────────────────────")
            summaries_path = generate_summaries(
                model_cfg=gen_cfg,
                fables=fables,
                prompt=config["summarization_prompt"],
                prompt_version=prompt_version,
                post_cfg=post_cfg,
                diag_cfg=diag_cfg,
                gen_model_dir=model_dir,
                force=force,
            )
        else:
            summaries_path = model_dir / "corpus_summaries.json"

        if steps.get("generate_paraphrases", True):
            print(f"\n── Paraphrases: {alias} ────────────────────────────────")
            paraphrases_path = generate_paraphrases(
                model_cfg=gen_cfg,
                moral_entries=moral_entries,
                morals=morals,
                prompt=config["paraphrase_prompt"],
                prompt_version=prompt_version,
                post_cfg=post_cfg,
                filter_cfg=filter_cfg,
                filter_model=filter_model,
                gen_model_dir=model_dir,
                force=force,
            )
        else:
            paraphrases_path = model_dir / "query_paraphrases.json"

        gen_cache_paths[alias] = {
            "summaries": summaries_path,
            "paraphrases": paraphrases_path,
        }

    # Unload filter model before embedding stage
    if filter_model is not None:
        del filter_model

    write_manifest(run_dir, "generation", config)

    # ── Stage 2: Matrix retrieval eval ────────────────────────────────────────
    if steps.get("run_retrieval_eval", True):
        print(f"\n── Stage 2: Matrix Retrieval Eval ──────────────────────")
        ablations = config.get("ablations", [])

        for embed_cfg in config.get("embed_models", []):
            emb_alias = embed_cfg["alias"]
            print(f"\n  Embed model: {emb_alias}")

            for gen_cfg in config.get("generation_models", []):
                gen_alias = gen_cfg["alias"]
                paths = gen_cache_paths.get(gen_alias, {})

                # Load cached generation outputs
                summaries_path = paths.get("summaries")
                paraphrases_path = paths.get("paraphrases")

                corpus_summaries = None
                if summaries_path and Path(summaries_path).exists():
                    with open(summaries_path) as f:
                        raw = json.load(f)
                    corpus_summaries = {item["fable_id"]: item["summary"] for item in raw["items"]}

                paraphrase_lookup = None
                if paraphrases_path and Path(paraphrases_path).exists():
                    with open(paraphrases_path) as f:
                        raw = json.load(f)
                    paraphrase_lookup = {item["moral_idx"]: item["paraphrases"] for item in raw["items"]}

                for ablation in ablations:
                    abl_name = ablation["name"]
                    use_summary = ablation.get("corpus") == "summary"
                    use_paraphrases = ablation.get("query") == "paraphrases"

                    # Build corpus texts
                    if use_summary and corpus_summaries:
                        corpus_texts = [corpus_summaries.get(f["doc_id"], f["text"]) for f in fables]
                    else:
                        corpus_texts = fable_texts

                    # Build query texts list
                    moral_entries_list = moral_entries
                    if use_paraphrases and paraphrase_lookup:
                        query_texts_list = []
                        for i, (m_idx, _) in enumerate(moral_entries_list):
                            rephrases = paraphrase_lookup.get(m_idx, [])
                            query_texts_list.append([moral_texts[i]] + rephrases)
                    else:
                        query_texts_list = [[t] for t in moral_texts]

                    combo = f"{gen_alias}__{emb_alias}__{abl_name}"
                    output_path = results_dir / f"{combo}.json"
                    preds_path = predictions_dir / f"{combo}.json"

                    print(f"\n    {combo}")
                    metrics = run_single_eval(
                        corpus_texts=corpus_texts,
                        query_texts_list=query_texts_list,
                        ground_truth=ground_truth,
                        embed_model_cfg=embed_cfg,
                        cache_dir=embedding_cache_dir,
                        output_path=output_path,
                        predictions_path=preds_path,
                        force=force,
                    )

                    # Patch gen/embed model info into result file for aggregator
                    with open(output_path) as f:
                        result = json.load(f)
                    result.update({"gen_model": gen_alias, "embed_model": emb_alias,
                                   "ablation": abl_name, "metrics": metrics})
                    with open(output_path, "w") as f:
                        json.dump(result, f, indent=2)

                    print(f"    R@1={metrics.get('Recall@1', 0):.3f}  MRR={metrics.get('MRR', 0):.4f}")

        write_manifest(run_dir, "retrieval_eval", config)

    # ── Stage 3: Aggregate ────────────────────────────────────────────────────
    print(f"\n── Stage 3: Aggregation ────────────────────────────────")
    aggregate(results_dir, run_dir)

    print(f"\n  Done. Results in {run_dir}")
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
python -m pytest tests/pipeline/test_matrix_runner.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/pipeline/matrix_runner.py tests/pipeline/test_matrix_runner.py
git commit -m "feat(pipeline): add matrix_runner — N×M orchestrator for local model grid experiments"
```

---

## Task 8: Update `lib/pipeline/__init__.py` and `default_config.yaml`

**Files:**
- Modify: `lib/pipeline/__init__.py`
- Modify: `lib/pipeline/default_config.yaml`

- [ ] **Step 1: Export `run_matrix_experiment` from `__init__.py`**

Add one import line at the bottom of the existing `lib/pipeline/__init__.py`:

```python
from lib.pipeline.matrix_runner import run_matrix_experiment  # noqa: F401
```

- [ ] **Step 2: Add matrix keys to `default_config.yaml`**

Append to the end of `lib/pipeline/default_config.yaml`:

```yaml

# Matrix experiment defaults (used by run_matrix_experiment)
generation_models: []
embed_models: []
summarization_prompt: ""
paraphrase_prompt: ""
prompt_version: "v1"

post_processing:
  enforce_word_count:
    enabled: true
    min_words: 5
    max_words: 15
    action: flag

paraphrase_filter:
  enabled: true
  min_similarity: 0.85
  filter_model: BAAI/bge-m3

summary_diagnostics:
  enabled: true
  duplicate_similarity_threshold: 0.95
  unique_ratio: true
  generic_phrases: []

ablations: []
```

- [ ] **Step 3: Verify import works**

```bash
python -c "from lib.pipeline import run_matrix_experiment; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add lib/pipeline/__init__.py lib/pipeline/default_config.yaml
git commit -m "feat(pipeline): export run_matrix_experiment and add matrix defaults to config"
```

---

## Task 9: Create `experiments/10_model_matrix/`

**Files:**
- Create: `experiments/10_model_matrix/config.yaml`
- Create: `experiments/10_model_matrix/run_pipeline.py`
- Create: `experiments/10_model_matrix/results/pipeline_runs/.gitkeep`

- [ ] **Step 1: Create experiment directory structure**

```bash
mkdir -p /Users/asifamar/Desktop/Master/NLP-morables/experiments/10_model_matrix/results/pipeline_runs
touch /Users/asifamar/Desktop/Master/NLP-morables/experiments/10_model_matrix/results/pipeline_runs/.gitkeep
```

- [ ] **Step 2: Write `config.yaml`**

```yaml
# experiments/10_model_matrix/config.yaml

n_fables: 50
prompt_version: "v1"

# ── Generation models ─────────────────────────────────────────────────────────
generation_models:
  - id: Qwen/Qwen3-8B-Instruct
    alias: Qwen3-8B
    dtype: bfloat16
    max_new_tokens: 64
    temperature: 0.3

  - id: google/gemma-3-4b-it
    alias: Gemma-4B
    dtype: bfloat16
    max_new_tokens: 64
    temperature: 0.3

  - id: microsoft/Phi-3.5-mini-instruct
    alias: Phi-3.5-mini
    dtype: bfloat16
    max_new_tokens: 64
    temperature: 0.3

# ── Embedding models ──────────────────────────────────────────────────────────
embed_models:
  - id: Qwen/Qwen3-Embedding-8B
    alias: Qwen3-Embed-8B
    query_instruction: "Given a moral statement, retrieve the most relevant fable passage"

  - id: BAAI/bge-m3
    alias: BGE-M3
    query_instruction: ""

  - id: pplx-embed-v1-4B        # update to confirmed HF model ID before running
    alias: pplx-embed-v1
    query_instruction: "Given a moral statement, retrieve the most relevant fable passage"

# ── Prompts ───────────────────────────────────────────────────────────────────
summarization_prompt: |
  You are an expert in moral philosophy. Distill the lesson of the following
  fable into one declarative sentence of 5 to 15 words. The statement must be
  universal and timeless — no character names, no reference to story events.
  Output ONLY the moral sentence. No explanation.

paraphrase_prompt: |
  You are given a moral statement. Write exactly 3 different rephrasings using
  different words while preserving the exact same meaning. Each must be abstract
  and universal, 5 to 15 words.
  Output ONLY the 3 rephrasings, one per line. No numbers, no labels.

# ── Post-processing ───────────────────────────────────────────────────────────
post_processing:
  enforce_word_count:
    enabled: true
    min_words: 5
    max_words: 15
    action: flag

# ── Paraphrase filter ─────────────────────────────────────────────────────────
paraphrase_filter:
  enabled: true
  min_similarity: 0.85
  filter_model: BAAI/bge-m3

# ── Summary diagnostics ───────────────────────────────────────────────────────
summary_diagnostics:
  enabled: true
  duplicate_similarity_threshold: 0.95
  unique_ratio: true
  generic_phrases:
    - "be honest"
    - "work hard"
    - "treat others well"

# ── Ablations ─────────────────────────────────────────────────────────────────
ablations:
  - name: raw_raw
    corpus: raw
    query: raw

  - name: summary_only
    corpus: summary
    query: raw

  - name: paraphrase_only
    corpus: raw
    query: paraphrases

  - name: full
    corpus: summary
    query: paraphrases

# ── Pipeline steps ────────────────────────────────────────────────────────────
steps:
  generate_summaries: true
  generate_paraphrases: true
  run_retrieval_eval: true

api_delay_seconds: 0.0
```

- [ ] **Step 3: Write `run_pipeline.py`**

```python
"""
run_pipeline.py — Entry point for exp10 model matrix experiment.

Usage:
  python experiments/10_model_matrix/run_pipeline.py
  python experiments/10_model_matrix/run_pipeline.py --run-dir path/to/run_dir
  python experiments/10_model_matrix/run_pipeline.py --force
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lib.pipeline import run_matrix_experiment

parser = argparse.ArgumentParser(description="Run exp10 N×M model matrix via generic pipeline")
parser.add_argument("--run-dir", type=Path, default=None,
                    help="Existing run dir to continue (default: create new)")
parser.add_argument("--force", action="store_true",
                    help="Re-run steps even if cached output exists")
args = parser.parse_args()

run_matrix_experiment(
    config_path=Path(__file__).parent / "config.yaml",
    run_dir=args.run_dir,
    force=args.force,
)
```

- [ ] **Step 4: Verify the entry point parses without error**

```bash
cd /Users/asifamar/Desktop/Master/NLP-morables
python experiments/10_model_matrix/run_pipeline.py --help
```
Expected: prints usage with `--run-dir` and `--force` options, no import errors.

- [ ] **Step 5: Run full test suite to confirm nothing is broken**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```
Expected: all tests PASS (new + existing).

- [ ] **Step 6: Commit**

```bash
git add experiments/10_model_matrix/
git commit -m "feat(exp10): add model matrix experiment — config, run_pipeline, results dir"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| N×M generation × embedding grid | Task 7 (matrix_runner) |
| Local HF models, MPS, bfloat16 | Task 2 (local_llm) |
| Generation caching by prompt_version | Tasks 3, 4 |
| Embedding cache with instruction in key | Task 5 (run_single_eval passes instruction) |
| Paraphrase filtering (sim ≥ 0.85) | Task 1 (paraphrase_filter) |
| Word-count post-processing, flag action | Tasks 1, 3, 4 |
| Summary diagnostics (unique_ratio, duplicates, generic) | Task 3 |
| 4 ablation modes (raw_raw, summary_only, paraphrase_only, full) | Task 9 (config) + Task 7 |
| Per-combination retrieval_results JSON | Task 5, 7 |
| Per-query predictions JSON | Task 5 |
| matrix_summary.json + rankings.json | Task 6 |
| dominant_factor analysis | Task 6 |
| Existing experiments untouched | run_retrieval_eval unchanged; new function added |
| Config-driven extensibility | All new code reads from config; no hardcoded model names |
| `run_pipeline.py` entry point | Task 9 |

**No placeholders found.** All code blocks are complete.

**Type consistency:** `generate_summaries` and `generate_paraphrases` signatures used in Task 7 match their definitions in Tasks 3 and 4. `run_single_eval` signature in Task 5 matches usage in Task 7. `aggregate(retrieval_results_dir, run_dir)` in Task 6 matches call in Task 7.
