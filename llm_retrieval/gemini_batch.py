"""
Gemini Batch API runner for LLM retrieval.

Uses Google AI File API + Batch API with one JSONL request per moral. This
runner is intentionally minimal-prompt-only for the current LLM retrieval setup.

Usage
-----
./run.sh llm_retrieval/gemini_batch.py --model Gemini-3.1-Flash-Lite-Preview
./run.sh llm_retrieval/gemini_batch.py --test 1 --model Gemini-3.1-Flash-Lite-Preview
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from google import genai
from google.genai import types

ROOT = Path(__file__).parent.parent
EXP_DIR = ROOT / "llm_retrieval"
CONFIG_PATH = EXP_DIR / "config.yaml"
RESULTS_DIR = EXP_DIR / "results"
BATCH_DIR = RESULTS_DIR / "gemini_batches"

FABLES_PATH = ROOT / "data/processed/fables_corpus.json"
MORALS_PATH = ROOT / "data/processed/morals_corpus.json"

sys.path.insert(0, str(ROOT))

from llm_retrieval.lib.corpus import build_corpus_block
from llm_retrieval.lib.eval import aggregate_metrics, compute_row_metrics
from llm_retrieval.lib.prompt import render_prompt
from llm_retrieval.lib.results import append_summary_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini Batch API retrieval runner")
    parser.add_argument("--test", type=int, help="Number of morals to submit; omit for full 709")
    parser.add_argument("--model", default="Gemini-3.1-Flash-Lite-Preview", help="Model alias from config.yaml")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=7200, help="Seconds to wait for batch completion")
    parser.add_argument("--collect-dir", help="Collect outputs for an existing Gemini batch run directory")
    return parser.parse_args()


def load_minimal_variant(cfg: dict) -> dict:
    for variant in cfg["prompt_variants"]:
        if variant["label"] == "minimal":
            return variant
    raise ValueError("No prompt variant labeled 'minimal' found")


def load_model_cfg(cfg: dict, alias: str) -> dict:
    for model in cfg["models"]:
        if model["alias"] == alias:
            if model["provider"] != "google":
                raise ValueError(f"Gemini batch runner supports Google models only, got {model['provider']}")
            return model
    raise ValueError(f"Unknown model alias: {alias}")


def load_queries(n: int | None) -> list[dict]:
    morals = json.loads(MORALS_PATH.read_text())
    rows = []
    for moral in morals:
        if moral.get("fable_id"):
            rows.append(
                {
                    "moral_id": moral["doc_id"],
                    "moral_text": moral["text"],
                    "relevant_fable": moral["fable_id"],
                }
            )
        if n is not None and len(rows) >= n:
            break
    return rows


def build_request_line(query: dict, user: str, system: str) -> str:
    request = {
        "key": query["moral_id"],
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user}],
                }
            ],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                "responseMimeType": "application/json",
            },
        },
    }
    return json.dumps(request) + "\n"


def write_input_jsonl(
    path: Path,
    variant_cfg: dict,
    corpus_block: str,
    queries: list[dict],
    top_k: int,
) -> None:
    with path.open("w") as f:
        for query in queries:
            _, user = render_prompt(query["moral_text"], corpus_block, variant_cfg, top_k)
            f.write(build_request_line(query, user, variant_cfg["system"]))


def job_to_dict(job: Any) -> dict:
    if hasattr(job, "model_dump"):
        return job.model_dump(mode="json")
    if hasattr(job, "to_json_dict"):
        return job.to_json_dict()
    return json.loads(json.dumps(job, default=str))


def state_name(job: Any) -> str:
    state = getattr(job, "state", None)
    return getattr(state, "name", None) or str(state)


def decode_downloaded(content: Any) -> str:
    if isinstance(content, bytes):
        return content.decode("utf-8")
    if hasattr(content, "decode"):
        return content.decode("utf-8")
    return str(content)


def parse_response_text(output_obj: dict) -> str:
    response = output_obj.get("response") or {}
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts if isinstance(part, dict))


def parse_ranked_ids(output_obj: dict) -> list[str]:
    if output_obj.get("error"):
        return []
    text = parse_response_text(output_obj).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        return parsed.get("ids", [])[:10]
    if isinstance(parsed, list):
        return parsed[:10]
    return []


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def collect_existing(out_dir: Path, model_cfg: dict | None = None) -> None:
    client = get_client()
    job_path = out_dir / "batch_job.json"
    metadata = json.loads(job_path.read_text())
    job_name = metadata["name"]
    batch_job = client.batches.get(name=job_name)
    metadata["latest"] = job_to_dict(batch_job)
    job_path.write_text(json.dumps(metadata, indent=2) + "\n")

    if state_name(batch_job) != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch {job_name} is {state_name(batch_job)}; cannot collect")

    result_file_name = batch_job.dest.file_name
    output_path = out_dir / "output.jsonl"
    if output_path.exists():
        output_text = output_path.read_text()
    else:
        output_text = decode_downloaded(client.files.download(file=result_file_name))
        output_path.write_text(output_text)

    output_rows = [json.loads(line) for line in output_text.splitlines() if line.strip()]
    by_id = {row["key"]: row for row in output_rows}
    queries = load_queries(None)
    metric_rows = []
    for query in queries:
        if query["moral_id"] not in by_id:
            continue
        ranked_ids = parse_ranked_ids(by_id[query["moral_id"]])
        metric_rows.append(
            compute_row_metrics(
                moral_id=query["moral_id"],
                moral_text=query["moral_text"],
                relevant_fable=query["relevant_fable"],
                ranked_ids=ranked_ids,
                latency_s=0.0,
            )
        )

    rows_path = out_dir / "rows.csv"
    summary_path = out_dir / "summary.json"
    write_rows_csv(rows_path, metric_rows)
    agg = aggregate_metrics(metric_rows)
    summary = {
        "run_date": date.today().isoformat(),
        "model_alias": metadata.get("model_alias", model_cfg["alias"] if model_cfg else ""),
        "model_id": metadata.get("model_id", model_cfg["id"] if model_cfg else ""),
        "provider": "google_batch",
        "variant_label": "minimal",
        "n_queries": agg["n_queries"],
        "MRR@10": round(agg["MRR@10"], 4),
        "R@1": round(agg["R@1"], 4),
        "R@5": round(agg["R@5"], 4),
        "R@10": round(agg["R@10"], 4),
        "NDCG@10": round(agg["NDCG@10"], 4),
        "Mean_Rank": round(agg["Mean_Rank"], 1) if agg["Mean_Rank"] is not None else "",
        "Median_Rank": round(agg["Median_Rank"], 1) if agg["Median_Rank"] is not None else "",
        "avg_latency_s": 0,
        "batch_name": job_name,
        "result_file_name": result_file_name,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    append_summary_row(ROOT / "llm_retrieval/results/unified.csv", summary)

    print(
        f"n={summary['n_queries']}  "
        f"MRR@10={summary['MRR@10']}  "
        f"R@1={summary['R@1']}  "
        f"R@10={summary['R@10']}  "
        f"NDCG@10={summary['NDCG@10']}"
    )
    print(f"Rows: {rows_path}")
    print(f"Summary: {summary_path}")


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    model_cfg = load_model_cfg(cfg, args.model)

    if args.collect_dir:
        collect_existing(Path(args.collect_dir), model_cfg)
        return

    variant_cfg = load_minimal_variant(cfg)
    top_k = cfg["top_k"]
    queries = load_queries(args.test)
    fables = json.loads(FABLES_PATH.read_text())
    random.seed(42)
    random.shuffle(fables)
    corpus_block = build_corpus_block(fables)

    run_date = date.today().isoformat()
    run_name = f"{run_date}_{model_cfg['alias']}_minimal_batch_n{len(queries)}"
    out_dir = BATCH_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = out_dir / "input.jsonl"
    job_path = out_dir / "batch_job.json"
    write_input_jsonl(input_path, variant_cfg, corpus_block, queries, top_k)
    print(f"Input JSONL: {input_path} ({input_path.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)

    client = get_client()
    uploaded_file = client.files.upload(
        file=str(input_path),
        config=types.UploadFileConfig(display_name=run_name, mime_type="jsonl"),
    )
    print(f"Uploaded file: {uploaded_file.name}", flush=True)

    batch_job = client.batches.create(
        model=model_cfg["id"],
        src=uploaded_file.name,
        config={"display_name": run_name},
    )
    metadata = {
        "run_date": run_date,
        "model_alias": model_cfg["alias"],
        "model_id": model_cfg["id"],
        "name": batch_job.name,
        "uploaded_file": uploaded_file.name,
        "initial": job_to_dict(batch_job),
    }
    job_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Batch job: {batch_job.name} state={state_name(batch_job)}", flush=True)

    terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    deadline = time.monotonic() + args.timeout
    while state_name(batch_job) not in terminal:
        if time.monotonic() >= deadline:
            metadata["latest"] = job_to_dict(batch_job)
            job_path.write_text(json.dumps(metadata, indent=2) + "\n")
            raise TimeoutError(f"Batch {batch_job.name} still {state_name(batch_job)} after {args.timeout}s")
        time.sleep(args.poll_interval)
        batch_job = client.batches.get(name=batch_job.name)
        metadata["latest"] = job_to_dict(batch_job)
        job_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Batch job: {batch_job.name} state={state_name(batch_job)}", flush=True)

    if state_name(batch_job) != "JOB_STATE_SUCCEEDED":
        metadata["latest"] = job_to_dict(batch_job)
        job_path.write_text(json.dumps(metadata, indent=2) + "\n")
        raise RuntimeError(f"Batch ended with state={state_name(batch_job)}; see {job_path}")

    collect_existing(out_dir, model_cfg)


if __name__ == "__main__":
    main()
