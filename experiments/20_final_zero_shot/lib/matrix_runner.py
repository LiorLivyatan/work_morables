from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify

_retrieval_spec = importlib.util.spec_from_file_location(
    "morables_retrieval_utils",
    ROOT / "lib/retrieval_utils.py",
)
if _retrieval_spec is None or _retrieval_spec.loader is None:
    raise ImportError("Could not load lib/retrieval_utils.py")
_retrieval_utils = importlib.util.module_from_spec(_retrieval_spec)
_retrieval_spec.loader.exec_module(_retrieval_utils)
compute_multilabel_metrics_from_matrix = _retrieval_utils.compute_multilabel_metrics_from_matrix

TRACKING_CSV = ROOT / "docs/zero_shot_full_tracking_matrix.csv"
MODEL_INVENTORY_CSV = ROOT / "docs/zero_shot_model_inventory.csv"
SUMMARY_SOURCES_JSON = EXP_DIR / "summary_sources.json"

CLUSTERED_QUERIES = ROOT / "data/clustered/morals_unique_corpus.json"
CLUSTERED_FABLES = ROOT / "data/clustered/fables_corpus.json"
CLUSTERED_QRELS = ROOT / "data/clustered/qrels_moral_to_fable_clustered.json"

DEFAULT_REMOTE_STORAGE = Path("/data/lior/final_zero_shot")

GENERAL_TASK = "Given a text, retrieve the most relevant passage that answers the query"
MORAL_TASK = "Given a moral statement, retrieve the fable that best conveys this moral."

GENERAL_PREFIX = f"Instruct: {GENERAL_TASK}\nQuery: "
MORAL_PREFIX = f"Instruct: {MORAL_TASK}\nQuery: "

KS = (1, 5, 10, 15, 50, 100, 200, 300)

CSV_FIELDNAMES = [
    "run_id", "model_alias", "model_id", "instruction_variant",
    "corpus_config", "summary_generator", "summary_generator_model_id",
    "generator_prompt_variant", "MAP@10", "MRR@10", "NDCG@10",
    "Recall@5", "Recall@10", "Recall@15", "Recall@50", "Recall@100",
    "Recall@200", "Recall@300", "Hit@1", "Hit@5", "Hit@10", "Hit@100",
    "Mean_Rank", "Median_Rank", "n_queries", "error",
]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-plan-csv", help="Exact run-plan CSV to execute")
    parser.add_argument("--models", nargs="+", help="Model aliases or model id substrings")
    parser.add_argument("--instructions", nargs="+", help="Instruction variants")
    parser.add_argument("--generators", nargs="+", help="Summary generator aliases")
    parser.add_argument("--generator-models", nargs="+", help="Summary generator model ids")
    parser.add_argument("--prompt-variants", nargs="+", help="Generator prompt variants")
    parser.add_argument("--corpus-configs", nargs="+", help="Corpus config ids")
    parser.add_argument("--limit", type=int, help="Limit selected runs for smoke tests")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings")
    parser.add_argument("--dry-run", action="store_true", help="Print selected runs only")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def storage_dirs(phase_name: str, device: str) -> tuple[Path, Path]:
    storage_root = os.environ.get("FINAL_ZERO_SHOT_STORAGE_ROOT")
    if storage_root:
        base = Path(storage_root) / phase_name
    elif device == "cuda" and Path("/data/lior").exists():
        base = DEFAULT_REMOTE_STORAGE / phase_name
    else:
        base = EXP_DIR / phase_name
    return base / "results", base / "cache"


def load_clustered_data() -> tuple[list[dict], list[dict], dict[int, list[int]]]:
    queries = json.loads(CLUSTERED_QUERIES.read_text())
    fables = json.loads(CLUSTERED_FABLES.read_text())
    qrels = json.loads(CLUSTERED_QRELS.read_text())

    query_idx = {q["doc_id"]: i for i, q in enumerate(queries)}
    fable_idx = {f["doc_id"]: i for i, f in enumerate(fables)}
    relevant: dict[int, list[int]] = defaultdict(list)
    for row in qrels:
        if row["query_id"] in query_idx and row["doc_id"] in fable_idx:
            relevant[query_idx[row["query_id"]]].append(fable_idx[row["doc_id"]])
    return queries, fables, dict(relevant)


def load_model_inventory() -> dict[str, dict[str, str]]:
    with MODEL_INVENTORY_CSV.open(newline="", encoding="utf-8") as f:
        return {row["api name"]: row for row in csv.DictReader(f)}


def load_tracking_rows() -> list[dict[str, str]]:
    with TRACKING_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_run_plan(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _matches_any(value: str, filters: list[str] | None, substring: bool = False) -> bool:
    if not filters:
        return True
    value_l = value.lower()
    filters_l = [f.lower() for f in filters]
    if substring:
        return any(f in value_l for f in filters_l)
    return value_l in set(filters_l)


def select_rows(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    phase_generators: list[str] | None = None,
    phase_corpus_configs: list[str] | None = None,
) -> list[dict[str, str]]:
    if getattr(args, "run_plan_csv", None):
        rows = load_run_plan(args.run_plan_csv)
        phase_generators = None
        phase_corpus_configs = None

    selected = []
    for row in rows:
        if phase_generators is not None and row["summary_generator"] not in phase_generators:
            continue
        if phase_corpus_configs is not None and row["corpus_config"] not in phase_corpus_configs:
            continue
        if args.models and not (
            _matches_any(row["model_alias"], args.models, substring=True)
            or _matches_any(row["model_id"], args.models, substring=True)
        ):
            continue
        if not _matches_any(row["instruction_variant"], args.instructions):
            continue
        if not _matches_any(row["summary_generator"], args.generators):
            continue
        if not _matches_any(row["summary_generator_model_id"], args.generator_models, substring=True):
            continue
        if not _matches_any(row["generator_prompt_variant"], args.prompt_variants):
            continue
        if not _matches_any(row["corpus_config"], args.corpus_configs):
            continue
        selected.append(row)
    return selected[: args.limit] if args.limit else selected


def load_summary_sources() -> dict[str, dict[str, Any]]:
    raw = json.loads(SUMMARY_SOURCES_JSON.read_text())
    sources: dict[str, dict[str, Any]] = {}
    for alias, meta in raw.items():
        path = ROOT / meta["path"]
        summaries = None
        if path.exists():
            summaries = load_summaries(path)
        sources[alias] = {**meta, "path": path, "summaries": summaries}
    return sources


def load_summaries(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text())
    summaries: dict[str, dict[str, str]] = {}
    for item in data:
        alias = item.get("original_fable_id") or item.get("fable_alias") or item.get("alias")
        if alias:
            summaries[alias] = item.get("summaries", {})
    return summaries


def build_documents(row: dict[str, str], fables: list[dict], sources: dict[str, dict[str, Any]]) -> list[str]:
    if row["corpus_config"] == "raw":
        return [f["text"] for f in fables]

    generator = row["summary_generator"]
    source = sources.get(generator)
    if not source or source.get("summaries") is None:
        raise FileNotFoundError(f"No summary source configured/found for generator={generator}")

    template = row["corpus_template"].replace("\\n", "\n")
    docs = []
    for fable in fables:
        summary = source["summaries"].get(fable["alias"], {})
        docs.append(template.format(
            fable=fable["text"],
            direct_moral=summary.get("direct_moral", ""),
            cot_proverb=summary.get("cot_proverb", ""),
            conceptual_abstract=summary.get("conceptual_abstract", ""),
            proverb=summary.get("proverb", ""),
        ))
    return docs


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
    model_id = meta.get("api name", "")
    if model_id in {
        "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "GritLM/GritLM-7B",
        "Salesforce/SFR-Embedding-Mistral",
    }:
        return 1
    if model_id in {
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "NovaSearch/stella_en_1.5B_v5",
    }:
        return 4
    params_b = parse_params(meta.get("Params", ""))
    model_type = meta.get("type", "").lower()
    if params_b is None:
        return 16
    if params_b >= 7 or "llm" in model_type:
        return 4
    if params_b >= 1:
        return 8
    return 32


def patch_rope_theta_compat(model_id: str) -> None:
    if model_id in {
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "NovaSearch/stella_en_1.5B_v5",
    }:
        try:
            from transformers.cache_utils import DynamicCache
        except Exception:
            DynamicCache = None
        if DynamicCache is not None and not hasattr(DynamicCache, "from_legacy_cache"):
            @classmethod
            def from_legacy_cache(cls, past_key_values=None):
                return cls()

            DynamicCache.from_legacy_cache = from_legacy_cache
        if DynamicCache is not None and not hasattr(DynamicCache, "get_usable_length"):
            def get_usable_length(self, new_seq_length, layer_idx=0):
                return 0

            DynamicCache.get_usable_length = get_usable_length
        if DynamicCache is not None and not hasattr(DynamicCache, "to_legacy_cache"):
            def to_legacy_cache(self):
                return None

            DynamicCache.to_legacy_cache = to_legacy_cache

    if model_id in {
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "NovaSearch/stella_en_1.5B_v5",
    }:
        try:
            from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
        except Exception:
            return
        if not hasattr(Qwen2Config, "rope_theta"):
            Qwen2Config.rope_theta = 1_000_000.0

    if model_id in {
        "GritLM/GritLM-7B",
        "Salesforce/SFR-Embedding-Mistral",
    }:
        try:
            from transformers.models.mistral.configuration_mistral import MistralConfig
        except Exception:
            return
        if not hasattr(MistralConfig, "rope_theta"):
            MistralConfig.rope_theta = 10_000.0


def load_embedding_model(model_id: str, device: str, meta: dict[str, str]):
    from sentence_transformers import SentenceTransformer

    patch_rope_theta_compat(model_id)

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


def default_format(model_id: str) -> dict[str, Any]:
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


def resolve_format(row: dict[str, str]) -> dict[str, Any]:
    variant = row["instruction_variant"]
    model_id = row["model_id"]
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
        return default_format(model_id)
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
        aps.append(average_precision_at_k(rankings[q_idx], {int(d) for d in relevant_docs}, k))
    return float(np.mean(aps))


def metrics_for_scores(score_matrix: np.ndarray, ground_truth: dict[int, list[int]]) -> dict[str, float]:
    metrics = compute_multilabel_metrics_from_matrix(score_matrix, ground_truth, ks=KS)
    metrics["MAP@10"] = map_at_k(score_matrix, ground_truth, 10)
    return metrics


def run_id_for(row: dict[str, str]) -> str:
    parts = [row["model_alias"], row["instruction_variant"], row["corpus_config"]]
    if row["summary_generator"]:
        parts.extend([row["summary_generator"], row["generator_prompt_variant"]])
    return "__".join(parts)


def metrics_row(run_id: str, row: dict[str, str], metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_alias": row["model_alias"],
        "model_id": row["model_id"],
        "instruction_variant": row["instruction_variant"],
        "corpus_config": row["corpus_config"],
        "summary_generator": row["summary_generator"],
        "summary_generator_model_id": row["summary_generator_model_id"],
        "generator_prompt_variant": row["generator_prompt_variant"],
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


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def save_compact_rankings(
    out_path: Path,
    run_id: str,
    row: dict[str, str],
    score_matrix: np.ndarray,
    queries: list[dict],
    fables: list[dict],
) -> None:
    rankings = np.argsort(-score_matrix, axis=1)
    fable_ids = [f["doc_id"] for f in fables]
    payload = {
        "schema_version": "ranking_v1",
        "run_id": run_id,
        "model_alias": row["model_alias"],
        "model_id": row["model_id"],
        "instruction_variant": row["instruction_variant"],
        "corpus_config": row["corpus_config"],
        "summary_generator": row["summary_generator"],
        "summary_generator_model_id": row["summary_generator_model_id"],
        "generator_prompt_variant": row["generator_prompt_variant"],
        "n_queries": len(queries),
        "n_docs": len(fables),
        "query_ids": [q["doc_id"] for q in queries],
        "ranked_fable_ids": [[fable_ids[i] for i in ranked] for ranked in rankings.tolist()],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")


def run_matrix_phase(
    phase_name: str,
    phase_description: str,
    args: argparse.Namespace,
    phase_generators: list[str] | None = None,
    phase_corpus_configs: list[str] | None = None,
) -> None:
    queries, fables, ground_truth = load_clustered_data()
    inventory = load_model_inventory()
    selected_rows = select_rows(load_tracking_rows(), args, phase_generators, phase_corpus_configs)
    if not selected_rows:
        raise SystemExit("No runs selected.")

    device = get_device()
    results_dir, cache_dir = storage_dirs(phase_name, device)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = results_dir / timestamp
    rankings_dir = run_dir / "rankings"
    sources = load_summary_sources()

    print(f"\n[final_zero_shot: {phase_name}]")
    print(f"  {phase_description}")
    print(f"  device={device}")
    print(f"  clustered_queries={len(queries)}  fables={len(fables)}  qrels={sum(len(v) for v in ground_truth.values())}")
    print(f"  selected_runs={len(selected_rows)}")
    print(f"  output={run_dir}")
    print("  summary_sources:")
    for alias, meta in sources.items():
        status = "ok" if meta.get("summaries") is not None else "missing"
        print(f"    {alias}: {status} ({meta['path']})")

    if args.dry_run:
        for row in selected_rows:
            print(
                f"  {row['model_alias']} | {row['instruction_variant']} | "
                f"{row['corpus_config']} | {row['summary_generator']} | "
                f"{row['generator_prompt_variant']}"
            )
        return

    notify.send(
        f"final_zero_shot {phase_name} starting\n"
        f"device: {device}\n"
        f"runs: {len(selected_rows)}\n"
        f"models: {', '.join(sorted({r['model_alias'] for r in selected_rows}))}"
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    rankings_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps({
            "phase": phase_name,
            "description": phase_description,
            "timestamp": timestamp,
            "device": device,
            "selected_args": vars(args),
            "n_runs": len(selected_rows),
            "n_queries": len(queries),
            "n_fables": len(fables),
            "n_qrels": sum(len(v) for v in ground_truth.values()),
            "runs": selected_rows,
        }, indent=2),
        encoding="utf-8",
    )

    query_texts_raw = [q["text"] for q in queries]
    metrics_rows: list[dict[str, Any]] = []
    runs_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        runs_by_model[row["model_id"]].append(row)

    for model_id, model_rows in runs_by_model.items():
        alias = model_rows[0]["model_alias"]
        meta = inventory.get(model_id, {})
        batch_size = batch_size_for_model(meta)
        print(f"\n{'-' * 72}")
        print(f"Loading {alias} ({model_id}) batch_size={batch_size}")

        try:
            model = load_embedding_model(model_id, device, meta)
        except Exception as exc:
            print(f"  load failed: {exc}")
            for row in model_rows:
                run_id = run_id_for(row)
                out = metrics_row(run_id, row, {})
                out["error"] = f"load_error: {str(exc)[:200]}"
                metrics_rows.append(out)
            write_metrics_csv(run_dir / "metrics.csv", metrics_rows)
            continue

        for row in model_rows:
            run_id = run_id_for(row)
            safe_run_id = run_id.replace("/", "__")
            fmt = resolve_format(row)
            query_prefix = fmt.get("query_prefix", "")
            doc_prefix = fmt.get("doc_prefix", "")
            prompt_name = fmt.get("query_prompt_name")
            print(f"\n  [{run_id}]")
            print(f"    query_prefix={query_prefix[:80]!r} doc_prefix={doc_prefix[:40]!r} prompt_name={prompt_name!r}")

            try:
                docs_raw = build_documents(row, fables, sources)
                query_texts = [f"{query_prefix}{text}" for text in query_texts_raw]
                doc_texts = [f"{doc_prefix}{text}" for text in docs_raw]

                cache_base = cache_dir / alias / row["instruction_variant"] / row["corpus_config"] / row["summary_generator"]
                if row["generator_prompt_variant"]:
                    cache_base = cache_base / row["generator_prompt_variant"]
                query_cache = cache_dir / alias / row["instruction_variant"] / "queries.npy"
                doc_cache = cache_base / "docs.npy"

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
                out = metrics_row(run_id, row, metrics)
                metrics_rows.append(out)
                save_compact_rankings(
                    rankings_dir / f"{safe_run_id}.json",
                    run_id,
                    row,
                    score_matrix,
                    queries,
                    fables,
                )
                print(
                    f"    MRR={out['MRR@10']:.4f} MAP@10={out['MAP@10']:.4f} "
                    f"Hit@10={out['Hit@10']:.4f} Recall@100={out['Recall@100']:.4f}"
                )
            except Exception as exc:
                print(f"    failed: {exc}")
                out = metrics_row(run_id, row, {})
                out["error"] = f"error: {str(exc)[:200]}"
                metrics_rows.append(out)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

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
    print("\nTop runs:")
    for row in ok[:15]:
        print(
            f"  {row['model_alias']:<30} {row['instruction_variant']:<15} "
            f"{row['corpus_config']:<28} {row['summary_generator']:<12} "
            f"MRR={row['MRR@10']:.4f} MAP@10={row['MAP@10']:.4f}"
        )

    notify.send(
        f"final_zero_shot {phase_name} done\n"
        f"completed: {len(ok)}/{len(metrics_rows)}\n"
        f"output: {run_dir.name}"
    )
