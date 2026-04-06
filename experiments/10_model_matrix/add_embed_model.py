"""
add_embed_model.py — Add a single embedding model row to an existing exp10 run.

Skips generation (uses existing gen_cache summaries) and runs only the
retrieval stage for the specified embedding model, then re-aggregates.

Usage:
  python experiments/10_model_matrix/add_embed_model.py \
    --run-dir experiments/10_model_matrix/results/pipeline_runs/2026-04-06_17-32-23_sample50 \
    --embed-alias Linq-Embed-Mistral

The embed model config is read from experiments/10_model_matrix/config.yaml.
If the alias is not found there, pass --embed-id to specify the model id manually.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lib.pipeline.matrix_aggregator import aggregate_matrix
from lib.pipeline.local_corpus_generator import load_corpus_summaries
from lib.pipeline.local_query_paraphraser import load_query_paraphrases
from lib.pipeline.local_llm import resolve_model_source, sentence_transformer_load_kwargs
from lib.data import load_fables, load_morals, load_qrels_moral_to_fable


def main():
    parser = argparse.ArgumentParser(description="Add one embedding model to an existing exp10 run.")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to the existing pipeline run directory.")
    parser.add_argument("--embed-alias", type=str, required=True,
                        help="Alias of the embedding model (must exist in config.yaml).")
    parser.add_argument("--embed-id", type=str, default=None,
                        help="Override: HuggingFace model id (if alias is not in config.yaml).")
    parser.add_argument("--query-instruction", type=str, default=None,
                        help="Optional query instruction prefix for the embedding model.")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Pass trust_remote_code=True when loading the model.")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "config.yaml",
                        help="Path to config.yaml (default: config.yaml next to this script).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (auto/cuda/mps/cpu). Defaults to config value.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"ERROR: run-dir does not exist: {run_dir}")
        sys.exit(1)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Resolve embed model config
    em_cfg = None
    for em in config.get("embed_models", []):
        if em["alias"] == args.embed_alias:
            em_cfg = dict(em)
            break

    if em_cfg is None:
        if args.embed_id is None:
            print(f"ERROR: alias '{args.embed_alias}' not found in config.yaml and --embed-id not given.")
            sys.exit(1)
        em_cfg = {"alias": args.embed_alias, "id": args.embed_id}

    # Apply CLI overrides
    if args.embed_id:
        em_cfg["id"] = args.embed_id
    if args.query_instruction:
        em_cfg["query_instruction"] = args.query_instruction
    if args.trust_remote_code:
        em_cfg["trust_remote_code"] = True

    device = args.device or config.get("device", "auto")

    print(f"\n  Embed model : {em_cfg['alias']} ({em_cfg['id']})")
    print(f"  Run dir     : {run_dir}")
    print(f"  Device      : {device}")

    # Resolve device
    import torch
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"  Resolved device: {device}")

    # Load data (same subset as original run)
    manifest_path = run_dir / "run_manifest.json"
    n_fables = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        n_fables = manifest.get("config_snapshot", {}).get("n_fables")

    fables = load_fables()
    morals = load_morals()
    gt_m2f = load_qrels_moral_to_fable()

    if n_fables:
        fables = fables[:n_fables]
        print(f"  Using first {n_fables} fables (from manifest)")

    target_fable_indices = set(range(len(fables)))
    moral_entries = sorted(
        [(m_idx, f_idx) for m_idx, f_idx in gt_m2f.items() if f_idx in target_fable_indices],
        key=lambda x: x[0]
    )
    ground_truth = {q_idx: fable_idx for q_idx, (_, fable_idx) in enumerate(moral_entries)}
    n_queries = len(moral_entries)

    raw_fable_texts = [f["text"] for f in fables]
    raw_moral_texts = [morals[m_idx]["text"] for m_idx, _ in moral_entries]

    # Discover gen_aliases from existing gen_cache
    gen_cache_root = run_dir / "gen_cache"
    gen_aliases = sorted([d.name for d in gen_cache_root.iterdir() if d.is_dir()])
    print(f"  Gen aliases : {gen_aliases}")

    # Load gen outputs (skip aliases whose generation is still in-progress)
    gen_corpus: dict[str, list[dict]] = {}
    gen_queries: dict[str, list[dict]] = {}
    for ga in gen_aliases:
        gc_dir = gen_cache_root / ga
        try:
            gen_corpus[ga] = load_corpus_summaries(gc_dir)
            gen_queries[ga] = load_query_paraphrases(gc_dir)
        except RuntimeError as e:
            print(f"  [skip] {ga}: {e}")
    gen_aliases = list(gen_corpus.keys())
    print(f"  Completed gen aliases: {gen_aliases}")

    # Output dirs
    results_dir = run_dir / "retrieval_results"
    results_dir.mkdir(exist_ok=True)
    preds_dir = run_dir / "predictions"
    preds_dir.mkdir(exist_ok=True)
    em_cache = run_dir / "embedding_cache" / em_cfg["alias"]
    em_cache.mkdir(parents=True, exist_ok=True)

    # Load embedding model
    from sentence_transformers import SentenceTransformer
    from lib.embedding_cache import encode_with_cache
    from lib.retrieval_utils import rank_analysis_from_matrix, compute_rankings_from_matrix, compute_metrics_from_matrix
    import numpy as np

    print(f"\n  Loading embedding model: {em_cfg['alias']} ...")
    model_source, is_local_source = resolve_model_source(em_cfg["id"])
    model = SentenceTransformer(
        model_source,
        device=device,
        **sentence_transformer_load_kwargs(em_cfg["id"], is_local_source),
    )

    em_alias = em_cfg["alias"]
    em_id = em_cfg["id"]
    query_instruction = em_cfg.get("query_instruction")

    # Embed raw data
    raw_corpus_embs = encode_with_cache(
        model=model, texts=raw_fable_texts, model_id=em_id,
        cache_dir=em_cache, query_instruction=None, label="raw_corpus"
    )
    raw_query_embs = encode_with_cache(
        model=model, texts=raw_moral_texts, model_id=em_id,
        cache_dir=em_cache, query_instruction=query_instruction, label="raw_queries"
    )

    for ga in gen_aliases:
        print(f"\n    Gen Model: {ga}")
        summaries = [item["summary"] for item in gen_corpus[ga]]
        summary_corpus_embs = encode_with_cache(
            model=model, texts=summaries, model_id=em_id,
            cache_dir=em_cache, query_instruction=None, label=f"{ga}_summaries"
        )

        flat_query_texts: list[str] = []
        query_slices: list[tuple[int, int]] = []
        for item in gen_queries[ga]:
            texts = [item["original_moral"]] + item["raw_paraphrases"]
            start = len(flat_query_texts)
            flat_query_texts.extend(texts)
            query_slices.append((start, start + len(texts)))

        all_query_embs = encode_with_cache(
            model=model, texts=flat_query_texts, model_id=em_id,
            cache_dir=em_cache, query_instruction=query_instruction,
            label=f"{ga}_queries_all"
        )
        para_embs_lists = [all_query_embs[s:e] for s, e in query_slices]

        matrix_raw_raw = raw_query_embs @ raw_corpus_embs.T
        matrix_summary_only = raw_query_embs @ summary_corpus_embs.T

        matrix_paraphrase_only = np.zeros((n_queries, len(fables)), dtype=np.float32)
        for q_idx in range(n_queries):
            scores = para_embs_lists[q_idx] @ raw_corpus_embs.T
            matrix_paraphrase_only[q_idx] = np.max(scores, axis=0)

        matrix_full = np.zeros((n_queries, len(fables)), dtype=np.float32)
        for q_idx in range(n_queries):
            scores = para_embs_lists[q_idx] @ summary_corpus_embs.T
            matrix_full[q_idx] = np.max(scores, axis=0)

        def _rrf(matrices, k=60):
            n_q, n_d = matrices[0].shape
            fused = np.zeros((n_q, n_d), dtype=np.float64)
            for mat in matrices:
                ranks = np.argsort(-mat, axis=1)
                rank_of = np.empty_like(ranks)
                row_idx = np.arange(n_q)[:, None]
                rank_of[row_idx, ranks] = np.arange(n_d)
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

        gt_sorted_qidx = sorted(ground_truth.keys())

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
            print(f"      [{ablation}] R@1={metrics.get('Recall@1', '?'):.2f}  saved → {res_path.name}")

            rankings_data = compute_rankings_from_matrix(mat, top_k=len(fables))
            ranks_arr = rank_analysis_from_matrix(mat, ground_truth)
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
            with open(preds_dir / f"{combo}.json", "w") as f:
                json.dump({"combo": combo, "queries": pred_records}, f, indent=2)

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Re-aggregate to update matrix_summary.json
    print("\n  Re-aggregating matrix_summary.json ...")
    aggregate_matrix(run_dir)
    print(f"\n  Done. New results added for: {em_alias}")


if __name__ == "__main__":
    main()
