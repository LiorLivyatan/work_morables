"""
Phase 1 raw instruction sweep for the final zero-shot experiment.

Runs clustered moral queries against the raw fable corpus and evaluates each
model's instruction variants. Saves metrics and the full ranked fable order for
every query.

Usage:
    ./run.sh experiments/20_final_zero_shot/phase_1_raw_instruction_sweep/run.py
    ./run.sh experiments/20_final_zero_shot/phase_1_raw_instruction_sweep/run.py --models all-MiniLM-L6-v2
    ./run.sh experiments/20_final_zero_shot/phase_1_raw_instruction_sweep/run.py --models Linq-Embed-Mistral --remote --gpu 2
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from lib.retrieval_utils import compute_multilabel_metrics_from_matrix

TRACKING_CSV = ROOT / "docs/zero_shot_full_tracking_matrix.csv"
MODEL_INVENTORY_CSV = ROOT / "docs/zero_shot_model_inventory.csv"

CLUSTERED_QUERIES = ROOT / "data/clustered/morals_unique_corpus.json"
CLUSTERED_FABLES = ROOT / "data/clustered/fables_corpus.json"
CLUSTERED_QRELS = ROOT / "data/clustered/qrels_moral_to_fable_clustered.json"

DEFAULT_REMOTE_STORAGE = Path("/data/lior/final_zero_shot/phase_1_raw_instruction_sweep")

GENERAL_TASK = "Given a text, retrieve the most relevant passage that answers the query"
MORAL_TASK = "Given a moral statement, retrieve the fable that best conveys this moral."

GENERAL_PREFIX = f"Instruct: {GENERAL_TASK}\nQuery: "
MORAL_PREFIX = f"Instruct: {MORAL_TASK}\nQuery: "

KS = (1, 5, 10, 15, 50, 100, 200, 300)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def storage_dirs(device: str) -> tuple[Path, Path]:
    storage_root = os.environ.get("FINAL_ZERO_SHOT_STORAGE_ROOT")
    if storage_root:
        base = Path(storage_root)
    elif device == "cuda" and Path("/data/lior").exists():
        base = DEFAULT_REMOTE_STORAGE
    else:
        base = EXP_DIR
    return base / "results", base / "cache"


def load_clustered_data() -> tuple[list[dict], list[dict], dict[int, list[int]]]:
    queries = json.loads(CLUSTERED_QUERIES.read_text())
    fables = json.loads(CLUSTERED_FABLES.read_text())
    qrels = json.loads(CLUSTERED_QRELS.read_text())

    query_idx = {q["doc_id"]: i for i, q in enumerate(queries)}
    fable_idx = {f["doc_id"]: i for i, f in enumerate(fables)}

    relevant: dict[int, list[int]] = defaultdict(list)
    for row in qrels:
        qid = row["query_id"]
        did = row["doc_id"]
        if qid not in query_idx or did not in fable_idx:
            continue
        relevant[query_idx[qid]].append(fable_idx[did])

    return queries, fables, dict(relevant)


def load_model_inventory() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with MODEL_INVENTORY_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["api name"]] = row
    return rows


def load_phase1_runs() -> list[dict[str, str]]:
    runs = []
    with TRACKING_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["corpus_config"] == "raw":
                runs.append(row)
    return runs


def select_runs(
    runs: list[dict[str, str]],
    model_filters: list[str] | None,
    instruction_filters: list[str] | None,
) -> list[dict[str, str]]:
    selected = runs
    if model_filters:
        filters = [m.lower() for m in model_filters]
        selected = [
            r for r in selected
            if any(
                f in r["model_alias"].lower() or f in r["model_id"].lower()
                for f in filters
            )
        ]
    if instruction_filters:
        filters = {i.lower() for i in instruction_filters}
        selected = [r for r in selected if r["instruction_variant"].lower() in filters]
    return selected


def parse_params(params: str) -> float | None:
    if not params:
        return None
    p = params.strip().lower()
    try:
        if p.endswith("b"):
            return float(p[:-1])
        if p.endswith("m"):
            return float(p[:-1]) / 1000.0
        return float(p)
    except ValueError:
        return None


def batch_size_for_model(meta: dict[str, str]) -> int:
    params_b = parse_params(meta.get("Params", ""))
    model_type = meta.get("type", "").lower()
    if params_b is None:
        return 16
    if params_b >= 7 or "llm" in model_type:
        return 4
    if params_b >= 1:
        return 8
    return 32


def load_embedding_model(model_id: str, device: str, meta: dict[str, str]):
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    model_kwargs: dict[str, Any] = {}
    params_b = parse_params(meta.get("Params", ""))

    if device == "cuda" and params_b and params_b >= 1:
        model_kwargs["torch_dtype"] = torch.bfloat16
    if device == "cuda" and params_b and params_b >= 10:
        model_kwargs["device_map"] = "auto"

    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs

    if model_kwargs.get("device_map") == "auto":
        return SentenceTransformer(model_id, **kwargs)
    return SentenceTransformer(model_id, device=device, **kwargs)


def default_format(alias: str, model_id: str) -> dict[str, Any]:
    if model_id in {"intfloat/e5-large-v2", "intfloat/e5-base-v2", "intfloat/multilingual-e5-large"}:
        return {"query_prefix": "query: ", "doc_prefix": "passage: "}
    if model_id == "nomic-ai/nomic-embed-text-v2-moe":
        return {"query_prefix": "search_query: ", "doc_prefix": "search_document: "}
    if model_id == "BAAI/bge-en-icl":
        return {"query_prefix": "Given a moral statement, retrieve the fable that best conveys this moral.\n"}
    if model_id in {"BAAI/bge-large-en-v1.5", "BAAI/bge-base-en-v1.5"}:
        return {"query_prefix": "Represent this sentence for searching relevant passages: "}
    if model_id in {"hkunlp/instructor-xl", "hkunlp/instructor-base"}:
        return {"query_prefix": "Represent the moral statement for retrieving relevant fables: "}
    if model_id == "NovaSearch/stella_en_1.5B_v5":
        return {"query_prompt_name": "s2p_query"}
    if model_id == "facebook/drama-1b":
        return {"query_prompt_name": "query"}
    if model_id == "orionweller/tart-dual-contriever-msmarco":
        return {"query_prefix": "Retrieve a fable that illustrates the following moral [SEP] "}
    if model_id == "GritLM/GritLM-7B":
        return {"query_prefix": f"<|user|>\n{MORAL_TASK}\n<|embed|>\n", "doc_prefix": "<|embed|>\n"}
    return {"query_prefix": ""}


def resolve_format(run_row: dict[str, str]) -> dict[str, Any]:
    variant = run_row["instruction_variant"]
    model_id = run_row["model_id"]
    alias = run_row["model_alias"]

    if variant == "no_instr":
        return {"query_prefix": "", "doc_prefix": ""}
    if variant == "general":
        doc_prefix = "passage: " if model_id in {
            "intfloat/e5-large-v2",
            "intfloat/e5-base-v2",
            "intfloat/multilingual-e5-large",
        } else ""
        return {"query_prefix": GENERAL_PREFIX, "doc_prefix": doc_prefix}
    if variant == "moral_specific":
        doc_prefix = "passage: " if model_id in {
            "intfloat/e5-large-v2",
            "intfloat/e5-base-v2",
            "intfloat/multilingual-e5-large",
        } else ""
        return {"query_prefix": MORAL_PREFIX, "doc_prefix": doc_prefix}
    if variant == "default":
        return default_format(alias, model_id)
    raise ValueError(f"Unknown instruction variant: {variant}")


def encode_texts(
    model,
    texts: list[str],
    cache_path: Path,
    batch_size: int,
    force: bool,
    prompt_name: str | None = None,
) -> np.ndarray:
    if cache_path.exists() and not force:
        print(f"    [cache hit] {cache_path}")
        return np.load(cache_path)

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "normalize_embeddings": True,
        "show_progress_bar": True,
        "convert_to_numpy": True,
    }
    if prompt_name:
        kwargs["prompt_name"] = prompt_name

    embeddings = model.encode(texts, **kwargs).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"    [cache saved] {cache_path}")
    return embeddings


def average_precision_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    hits = 0
    precision_sum = 0.0
    for rank_idx, doc_idx in enumerate(ranked[:k], start=1):
        if int(doc_idx) in relevant:
            hits += 1
            precision_sum += hits / rank_idx
    denom = min(len(relevant), k)
    return precision_sum / denom if denom else 0.0


def map_at_k(score_matrix: np.ndarray, ground_truth: dict[int, list[int]], k: int) -> float:
    rankings = np.argsort(-score_matrix, axis=1)
    aps = []
    for q_idx, relevant_docs in ground_truth.items():
        relevant = {int(d) for d in relevant_docs}
        aps.append(average_precision_at_k(rankings[q_idx], relevant, k))
    return float(np.mean(aps))


def metrics_for_scores(score_matrix: np.ndarray, ground_truth: dict[int, list[int]]) -> dict[str, float]:
    metrics = compute_multilabel_metrics_from_matrix(score_matrix, ground_truth, ks=KS)
    metrics["MAP@10"] = map_at_k(score_matrix, ground_truth, 10)
    return metrics


def metrics_row(run_id: str, run_row: dict[str, str], metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_alias": run_row["model_alias"],
        "model_id": run_row["model_id"],
        "instruction_variant": run_row["instruction_variant"],
        "corpus_config": "raw",
        "MAP@10": round(metrics.get("MAP@10", 0.0), 6),
        "MRR@10": round(metrics.get("MRR", 0.0), 6),
        "NDCG@10": round(metrics.get("NDCG@10", 0.0), 6),
        "Recall@5": round(metrics.get("Recall@5", 0.0), 6),
        "Recall@10": round(metrics.get("Recall@10", 0.0), 6),
        "Recall@15": round(metrics.get("Recall@15", 0.0), 6),
        "Recall@50": round(metrics.get("Recall@50", 0.0), 6),
        "Recall@100": round(metrics.get("Recall@100", 0.0), 6),
        "Recall@200": round(metrics.get("Recall@200", 0.0), 6),
        "Recall@300": round(metrics.get("Recall@300", 0.0), 6),
        "Hit@1": round(metrics.get("Hit@1", 0.0), 6),
        "Hit@5": round(metrics.get("Hit@5", 0.0), 6),
        "Hit@10": round(metrics.get("Hit@10", 0.0), 6),
        "Hit@100": round(metrics.get("Hit@100", 0.0), 6),
        "Mean_Rank": round(metrics.get("Mean Rank", 0.0), 2),
        "Median_Rank": round(metrics.get("Median Rank", 0.0), 1),
        "n_queries": metrics.get("n_queries", 0),
        "error": "",
    }


def save_rankings(
    out_path: Path,
    run_id: str,
    run_row: dict[str, str],
    score_matrix: np.ndarray,
    queries: list[dict],
    fables: list[dict],
    ground_truth: dict[int, list[int]],
) -> None:
    rankings = np.argsort(-score_matrix, axis=1)
    payload = {
        "run_id": run_id,
        "model_alias": run_row["model_alias"],
        "model_id": run_row["model_id"],
        "instruction_variant": run_row["instruction_variant"],
        "corpus_config": "raw",
        "n_queries": len(queries),
        "n_docs": len(fables),
        "queries": [],
    }
    fable_ids = [f["doc_id"] for f in fables]

    for q_idx, query in enumerate(queries):
        ranked_indices = rankings[q_idx].tolist()
        relevant = set(ground_truth.get(q_idx, []))
        relevant_ranks = {
            fable_ids[d_idx]: ranked_indices.index(d_idx) + 1
            for d_idx in relevant
            if d_idx in ranked_indices
        }
        payload["queries"].append({
            "query_idx": q_idx,
            "query_id": query["doc_id"],
            "query_text": query["text"],
            "relevant_fable_ids": [fable_ids[i] for i in sorted(relevant)],
            "relevant_fable_ranks": relevant_ranks,
            "ranked_fable_ids": [fable_ids[i] for i in ranked_indices],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id", "model_alias", "model_id", "instruction_variant", "corpus_config",
        "MAP@10", "MRR@10", "NDCG@10",
        "Recall@5", "Recall@10", "Recall@15", "Recall@50", "Recall@100", "Recall@200", "Recall@300",
        "Hit@1", "Hit@5", "Hit@10", "Hit@100",
        "Mean_Rank", "Median_Rank", "n_queries", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final zero-shot Phase 1 raw instruction sweep")
    parser.add_argument("--models", nargs="+", help="Model aliases or model id substrings to run")
    parser.add_argument("--instructions", nargs="+", help="Instruction variants to run")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings even when cached")
    parser.add_argument("--dry-run", action="store_true", help="Print selected runs without executing")
    args = parser.parse_args()

    queries, fables, ground_truth = load_clustered_data()
    model_inventory = load_model_inventory()
    selected_runs = select_runs(load_phase1_runs(), args.models, args.instructions)

    if not selected_runs:
        raise SystemExit("No Phase 1 runs selected.")

    device = get_device()
    results_dir, cache_dir = storage_dirs(device)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = results_dir / timestamp
    rankings_dir = run_dir / "rankings"
    metrics_rows: list[dict[str, Any]] = []

    print("\n[final_zero_shot: phase_1_raw_instruction_sweep]")
    print(f"  device={device}")
    print(f"  clustered_queries={len(queries)}  fables={len(fables)}  qrels={sum(len(v) for v in ground_truth.values())}")
    print(f"  selected_runs={len(selected_runs)}")
    print(f"  output={run_dir}")

    if args.dry_run:
        for row in selected_runs:
            print(f"  {row['model_alias']} | {row['instruction_variant']} | {row['model_id']}")
        return

    notify.send(
        "final_zero_shot phase 1 starting\n"
        f"device: {device}\n"
        f"runs: {len(selected_runs)}\n"
        f"models: {', '.join(sorted({r['model_alias'] for r in selected_runs}))}"
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    rankings_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps({
            "phase": "phase_1_raw_instruction_sweep",
            "timestamp": timestamp,
            "device": device,
            "selected_models": args.models,
            "selected_instructions": args.instructions,
            "n_runs": len(selected_runs),
            "n_queries": len(queries),
            "n_fables": len(fables),
            "n_qrels": sum(len(v) for v in ground_truth.values()),
            "runs": selected_runs,
        }, indent=2),
        encoding="utf-8",
    )

    query_texts_raw = [q["text"] for q in queries]
    doc_texts_raw = [f["text"] for f in fables]

    runs_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected_runs:
        runs_by_model[row["model_id"]].append(row)

    for model_id, model_runs in runs_by_model.items():
        alias = model_runs[0]["model_alias"]
        meta = model_inventory.get(model_id, {})
        batch_size = batch_size_for_model(meta)
        print(f"\n{'-' * 72}")
        print(f"Loading {alias} ({model_id}) batch_size={batch_size}")

        try:
            model = load_embedding_model(model_id, device, meta)
        except Exception as exc:
            print(f"  load failed: {exc}")
            for row in model_runs:
                run_id = f"{row['model_alias']}__{row['instruction_variant']}__raw"
                error_row = metrics_row(run_id, row, {})
                error_row["error"] = f"load_error: {str(exc)[:200]}"
                metrics_rows.append(error_row)
            write_metrics_csv(run_dir / "metrics.csv", metrics_rows)
            continue

        for row in model_runs:
            run_id = f"{row['model_alias']}__{row['instruction_variant']}__raw"
            fmt = resolve_format(row)
            query_prefix = fmt.get("query_prefix", "")
            doc_prefix = fmt.get("doc_prefix", "")
            prompt_name = fmt.get("query_prompt_name")

            print(f"\n  [{run_id}]")
            print(f"    query_prefix={query_prefix[:80]!r} doc_prefix={doc_prefix[:40]!r} prompt_name={prompt_name!r}")

            query_texts = [f"{query_prefix}{text}" for text in query_texts_raw]
            doc_texts = [f"{doc_prefix}{text}" for text in doc_texts_raw]

            safe_run_id = run_id.replace("/", "__")
            cache_base = cache_dir / alias / row["instruction_variant"]
            query_cache = cache_base / "query_embs.npy"
            doc_cache = cache_dir / alias / f"docs_{doc_prefix or 'plain'}".replace("/", "_") / "doc_embs.npy"

            try:
                query_embs = encode_texts(
                    model,
                    query_texts,
                    query_cache,
                    batch_size=batch_size,
                    force=args.force,
                    prompt_name=prompt_name,
                )
                doc_embs = encode_texts(
                    model,
                    doc_texts,
                    doc_cache,
                    batch_size=batch_size,
                    force=args.force,
                )
                score_matrix = query_embs @ doc_embs.T
                metrics = metrics_for_scores(score_matrix, ground_truth)
                row_out = metrics_row(run_id, row, metrics)
                metrics_rows.append(row_out)
                save_rankings(
                    rankings_dir / f"{safe_run_id}.json",
                    run_id,
                    row,
                    score_matrix,
                    queries,
                    fables,
                    ground_truth,
                )
                print(
                    f"    MRR={row_out['MRR@10']:.4f} MAP@10={row_out['MAP@10']:.4f} "
                    f"Hit@10={row_out['Hit@10']:.4f} Recall@100={row_out['Recall@100']:.4f}"
                )
                notify.send(
                    f"final_zero_shot phase 1\n{run_id}\n"
                    f"MRR={row_out['MRR@10']:.4f} MAP@10={row_out['MAP@10']:.4f}"
                )
            except Exception as exc:
                print(f"    failed: {exc}")
                error_row = metrics_row(run_id, row, {})
                error_row["error"] = f"error: {str(exc)[:200]}"
                metrics_rows.append(error_row)

            write_metrics_csv(run_dir / "metrics.csv", metrics_rows)
            (run_dir / "metrics.json").write_text(json.dumps(metrics_rows, indent=2), encoding="utf-8")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    ok = [r for r in metrics_rows if not r["error"]]
    ok.sort(key=lambda r: r["MRR@10"], reverse=True)
    print("\nTop Phase 1 runs:")
    for row in ok[:15]:
        print(f"  {row['model_alias']:<30} {row['instruction_variant']:<15} MRR={row['MRR@10']:.4f} MAP@10={row['MAP@10']:.4f}")

    notify.send(
        "final_zero_shot phase 1 done\n"
        f"completed: {len(ok)}/{len(metrics_rows)}\n"
        f"output: {run_dir.name}"
    )


if __name__ == "__main__":
    main()
