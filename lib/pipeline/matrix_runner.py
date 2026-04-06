"""lib/pipeline/matrix_runner.py — Orchestrate the N x M matrix experiment (Exp10)."""
import json
import gc
import importlib.util
import socket
from pathlib import Path

from lib.pipeline.local_corpus_generator import generate_corpus_summaries, load_corpus_summaries
from lib.pipeline.local_llm import resolve_model_source, sentence_transformer_load_kwargs
from lib.pipeline.local_query_paraphraser import generate_query_paraphrases, load_query_paraphrases
from lib.pipeline.matrix_aggregator import aggregate_matrix
from lib.data import load_fables, load_morals, load_qrels_moral_to_fable


def _cached_on_hub(model_id: str) -> bool:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot_download(model_id, local_files_only=True, allow_patterns=["config.json"])
        return True
    except LocalEntryNotFoundError:
        return False


def _has_hf_connectivity(timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=timeout):
            return True
    except OSError:
        return False


def _preflight(config: dict) -> None:
    required_models = [gm["id"] for gm in config.get("generation_models", [])]
    required_models.extend(em["id"] for em in config.get("embed_models", []))

    # BGE-M3 is always required during paraphrase filtering, independent of the
    # retrieval embedding matrix configured in config.yaml.
    if config.get("generation_models"):
        required_models.append("BAAI/bge-m3")

    if required_models and importlib.util.find_spec("google.protobuf") is None:
        raise RuntimeError(
            "Missing Python dependency 'protobuf'. Install it with "
            "`.venv/bin/pip install protobuf` and rerun the pipeline."
        )

    cached = sorted({model_id for model_id in required_models if _cached_on_hub(model_id)})
    missing = sorted(set(required_models) - set(cached))
    if cached:
        print(f"  [preflight] Cached locally: {', '.join(cached)}")
    if missing:
        print("  [preflight] Not cached locally; these will be downloaded on first use:")
        for model_id in missing:
            print(f"    - {model_id}")
        if not _has_hf_connectivity():
            raise RuntimeError(
                "Required Hugging Face models are not cached locally and "
                "huggingface.co is unreachable from this environment. "
                "Connect to the internet and download them first, or pre-populate "
                "the local Hugging Face cache before rerunning."
            )


def _run_matrix_retrieval(
    run_dir: Path,
    fables: list[dict],
    morals: list[dict],
    moral_entries: list[tuple[int, int]],
    gen_aliases: list[str],
    embed_models: list[dict],
    device: str = "mps"
):
    """
    Stage 2: Evaluate retrieval for all (gen, embed) combinations.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    from lib.embedding_cache import encode_with_cache

    # Resolve "auto" to a concrete device string
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    from lib.retrieval_utils import rank_analysis_from_matrix, compute_rankings_from_matrix, compute_metrics_from_matrix
    import numpy as np

    # Pre-parse ground truth format suitable for ranking functions
    ground_truth = {q_idx: fable_idx for q_idx, (_, fable_idx) in enumerate(moral_entries)}
    n_queries = len(moral_entries)

    # Output directories
    results_dir = run_dir / "retrieval_results"
    results_dir.mkdir(exist_ok=True)
    preds_dir = run_dir / "predictions"
    preds_dir.mkdir(exist_ok=True)

    # Lazy-load generation outputs (gen_aliases mapped to corpus and queries)
    gen_corpus: dict[str, list[dict]] = {}
    gen_queries: dict[str, list[dict]] = {}

    for ga in gen_aliases:
        gc_dir = run_dir / "gen_cache" / ga
        gen_corpus[ga] = load_corpus_summaries(gc_dir)
        gen_queries[ga] = load_query_paraphrases(gc_dir)

    raw_fable_texts = [f["text"] for f in fables]
    raw_moral_texts = [morals[m_idx]["text"] for m_idx, _ in moral_entries]

    for em_cfg in embed_models:
        em_alias = em_cfg["alias"]
        em_id = em_cfg["id"]
        em_cache = run_dir / "embedding_cache" / em_alias
        em_cache.mkdir(parents=True, exist_ok=True)

        print(f"\n  ── Embedding: {em_alias} ──")
        model_source, is_local_source = resolve_model_source(em_id)
        model = SentenceTransformer(
            model_source,
            device=device,
            **sentence_transformer_load_kwargs(em_id, is_local_source),
        )

        # 1. Embed raw data
        raw_corpus_embs = encode_with_cache(
            model=model, texts=raw_fable_texts, model_id=em_id,
            cache_dir=em_cache, query_instruction=None, label="raw_corpus"
        )
        raw_query_embs = encode_with_cache(
            model=model, texts=raw_moral_texts, model_id=em_id,
            cache_dir=em_cache, query_instruction=em_cfg.get("query_instruction"), label="raw_queries"
        )

        for ga in gen_aliases:
            print(f"    Gen Model: {ga}")
            # Summarized corpus
            summaries = [item["summary"] for item in gen_corpus[ga]]
            summary_corpus_embs = encode_with_cache(
                model=model, texts=summaries, model_id=em_id,
                cache_dir=em_cache, query_instruction=None, label=f"{ga}_summaries"
            )

            # Rephrased queries (+ original) — use all raw paraphrases, encode in one GPU call.
            flat_query_texts: list[str] = []
            query_slices: list[tuple[int, int]] = []
            for item in gen_queries[ga]:
                texts = [item["original_moral"]] + item["raw_paraphrases"]
                start = len(flat_query_texts)
                flat_query_texts.extend(texts)
                query_slices.append((start, start + len(texts)))

            all_query_embs = encode_with_cache(
                model=model, texts=flat_query_texts, model_id=em_id,
                cache_dir=em_cache, query_instruction=em_cfg.get("query_instruction"),
                label=f"{ga}_queries_all"
            )
            para_embs_lists = [
                all_query_embs[s:e] for s, e in query_slices
            ]

            # Compute the 4 matrices
            # 1. raw_raw
            matrix_raw_raw = raw_query_embs @ raw_corpus_embs.T

            # 2. summary_only
            matrix_summary_only = raw_query_embs @ summary_corpus_embs.T

            # 3. paraphrase_only (raw corpus, max-score fusion over [original + rephrases])
            matrix_paraphrase_only = np.zeros((n_queries, len(fables)), dtype=np.float32)
            for q_idx in range(n_queries):
                q_embs = para_embs_lists[q_idx]  # (K_q, D)
                scores = q_embs @ raw_corpus_embs.T  # (K_q, M)
                matrix_paraphrase_only[q_idx] = np.max(scores, axis=0)

            # 4. full (summary corpus, max-score fusion over [original + rephrases])
            matrix_full = np.zeros((n_queries, len(fables)), dtype=np.float32)
            for q_idx in range(n_queries):
                q_embs = para_embs_lists[q_idx]
                scores = q_embs @ summary_corpus_embs.T
                matrix_full[q_idx] = np.max(scores, axis=0)

            # 5. rrf — Reciprocal Rank Fusion over all 4 score matrices (k=60)
            # Combines rankings from raw_raw, summary_only, paraphrase_only, full.
            # More robust than max-score: outlier scores don't dominate.
            def _rrf(matrices: list[np.ndarray], k: int = 60) -> np.ndarray:
                n_q, n_d = matrices[0].shape
                fused = np.zeros((n_q, n_d), dtype=np.float64)
                for mat in matrices:
                    ranks = np.argsort(-mat, axis=1)          # (n_q, n_d) — position of each doc
                    rank_of = np.empty_like(ranks)
                    row_idx = np.arange(n_q)[:, None]
                    rank_of[row_idx, ranks] = np.arange(n_d)  # rank_of[q, doc] = rank position
                    fused += 1.0 / (k + rank_of)
                return fused.astype(np.float32)

            matrix_rrf = _rrf([matrix_raw_raw, matrix_summary_only,
                                matrix_paraphrase_only, matrix_full])

            ablations = {
                "raw_raw": matrix_raw_raw,
                "summary_only": matrix_summary_only,
                "paraphrase_only": matrix_paraphrase_only,
                "full": matrix_full,
                "rrf": matrix_rrf,
            }

            for ablation, mat in ablations.items():
                metrics = compute_metrics_from_matrix(mat, ground_truth)
                result_record = {
                    "gen_model": ga,
                    "embed_model": em_alias,
                    "ablation": ablation,
                    "n_queries": n_queries,
                    "metrics": metrics
                }
                combo = f"{ga}__{em_alias}__{ablation}"
                res_path = results_dir / f"{combo}.json"
                with open(res_path, "w") as f:
                    json.dump(result_record, f, indent=2)

                # Dump predictions
                rankings_data = compute_rankings_from_matrix(mat, top_k=len(fables))
                ranks_arr = rank_analysis_from_matrix(mat, ground_truth)
                gt_sorted_qidx = sorted(ground_truth.keys())
                rank_by_qidx = {q: int(r) + 1 for q, r in zip(gt_sorted_qidx, ranks_arr)}

                pred_records = []
                for q_idx, ranking in enumerate(rankings_data):
                    pred_records.append({
                        "query_idx": q_idx,
                        "moral_text": raw_moral_texts[q_idx],
                        "correct_fable_idx": ground_truth.get(q_idx),
                        "correct_rank": rank_by_qidx.get(q_idx),
                        "top_k_indices": ranking["indices"],
                        "top_k_scores": ranking["scores"],
                    })

                pred_record = {
                    "combo": combo,
                    "queries": pred_records
                }
                with open(preds_dir / f"{combo}.json", "w") as f:
                    json.dump(pred_record, f, indent=2)

        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def run_matrix_experiment(config_path: Path, resume_run_dir: Path | None = None):
    """
    Run the entire NxM matrix pipeline.

    Args:
        config_path:    Path to the experiment config.yaml.
        resume_run_dir: If provided, continue an existing run instead of
                        creating a new timestamped directory.  Generation and
                        retrieval steps that already have output files are
                        skipped automatically.
    """
    from lib.pipeline.run_utils import load_env, make_run_dir, write_manifest
    import yaml

    experiment_dir = Path(config_path).parent
    root_dir = experiment_dir.parent.parent
    load_env(root_dir)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    _preflight(config)

    n_fables = config.get("n_fables")
    tag = f"sample{n_fables}" if n_fables else "full"

    if resume_run_dir is not None:
        run_dir = Path(resume_run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"resume_run_dir does not exist: {run_dir}")
        print(f"\n  [resume] Continuing existing run: {run_dir.name}")
    else:
        base_run_dir = experiment_dir / "results" / "pipeline_runs"
        run_dir = make_run_dir(base_run_dir, tag)

    print(f"\n{'=' * 60}")
    print(f"  Matrix Pipeline Run: {run_dir.name}")
    print(f"{'=' * 60}")

    # Load data
    fables = load_fables()
    morals = load_morals()
    gt_m2f = load_qrels_moral_to_fable()

    if n_fables:
        fables = fables[:n_fables]
    target_fable_indices = set(range(len(fables)))
    moral_entries = sorted(
        [(m_idx, f_idx) for m_idx, f_idx in gt_m2f.items() if f_idx in target_fable_indices],
        key=lambda x: x[0]
    )

    gen_models = config.get("generation_models", [])
    embed_models = config.get("embed_models", [])
    prompt_version = config.get("prompt_version", "v1")
    device = config.get("device", "auto")

    # Stage 1: Generation
    for gm in gen_models:
        alias = gm["alias"]
        gen_cache_dir = run_dir / "gen_cache" / alias

        batch_size = gm.get("batch_size", 8)
        generate_corpus_summaries(
            fables=fables,
            gen_model_alias=alias,
            gen_model_id=gm["id"],
            gen_cache_dir=gen_cache_dir,
            prompt_version=prompt_version,
            force=config.get("force_generation", False),
            batch_size=batch_size,
            device=device,
        )
        generate_query_paraphrases(
            moral_entries=moral_entries,
            morals=morals,
            gen_model_alias=alias,
            gen_model_id=gm["id"],
            gen_cache_dir=gen_cache_dir,
            prompt_version=prompt_version,
            force=config.get("force_generation", False),
            batch_size=batch_size,
            device=device,
        )

    # Stage 2: Retrieval Eval
    _run_matrix_retrieval(
        run_dir=run_dir,
        fables=fables,
        morals=morals,
        moral_entries=moral_entries,
        gen_aliases=[gm["alias"] for gm in gen_models],
        embed_models=embed_models,
        device=device,
    )

    # Stage 3: Aggregation
    aggregate_matrix(run_dir)
    write_manifest(run_dir, "matrix_run_complete", config)
    print(f"\n  Done. Results in {run_dir}")
