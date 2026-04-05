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
