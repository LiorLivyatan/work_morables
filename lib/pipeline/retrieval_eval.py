"""lib/pipeline/retrieval_eval.py — Embedding-based retrieval evaluation."""
import json
from pathlib import Path
from typing import Optional

import numpy as np

from lib.embedding_cache import encode_with_cache
from lib.retrieval_utils import (
    compute_metrics_from_matrix,
    compute_rankings_from_matrix,
    rank_analysis_from_matrix,
)


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
    preds_dir = run_dir / "predictions"

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
    preds_dir.mkdir(exist_ok=True)

    def _dump_predictions(name: str, score_matrix: np.ndarray):
        rankings_data = compute_rankings_from_matrix(score_matrix, top_k=n_fables)
        ranks_arr = rank_analysis_from_matrix(score_matrix, ground_truth)
        gt_sorted_qidx = sorted(ground_truth.keys())
        rank_by_qidx = {q: int(r) + 1 for q, r in zip(gt_sorted_qidx, ranks_arr)}  # 1-indexed

        pred_records = []
        for q_idx, ranking in enumerate(rankings_data):
            correct_idx = ground_truth.get(q_idx)
            pred_records.append({
                "query_idx": q_idx,
                "correct_idx": correct_idx,
                "correct_rank": rank_by_qidx.get(q_idx),  # 1-indexed absolute rank
                "top_k_indices": ranking["indices"],
                "top_k_scores": ranking["scores"],
            })
        with open(preds_dir / f"{name}.json", "w") as f:
            json.dump({"config": name, "top_k": n_fables, "queries": pred_records}, f)

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
        _dump_predictions("baseline", b_matrix)
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
        _dump_predictions(name, score_matrix)

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
