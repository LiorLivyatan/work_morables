"""
OpenAI Batch API smoke runner for LLM retrieval.

This submits independent moral-to-fable retrieval requests as a JSONL batch,
waits for completion, downloads the output file, and writes comparable metrics.

Notes
-----
- This runner intentionally supports only the `minimal` prompt variant.
- OpenAI batch input files are capped at 200 MB, so a full 709-query run with
  the full corpus repeated per request may need to be split into multiple files.

Usage
-----
./run.sh llm_retrieval/openai_batch.py --test 1 --model GPT-5.4-Nano
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

ROOT = Path(__file__).parent.parent
EXP_DIR = ROOT / "llm_retrieval"
CONFIG_PATH = EXP_DIR / "config.yaml"
RESULTS_DIR = EXP_DIR / "results"
BATCH_DIR = RESULTS_DIR / "batches"

FABLES_PATH = ROOT / "data/processed/fables_corpus.json"
MORALS_PATH = ROOT / "data/processed/morals_corpus.json"

sys.path.insert(0, str(ROOT))

from llm_retrieval.lib.corpus import build_corpus_block
from llm_retrieval.lib.eval import aggregate_metrics, compute_row_metrics
from llm_retrieval.lib.prompt import render_prompt


RANKED_FABLES_SCHEMA = {
    "name": "ranked_fables",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["ids"],
        "additionalProperties": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI Batch API retrieval smoke runner")
    parser.add_argument("--test", type=int, help="Number of morals to submit; omit for full 709")
    parser.add_argument("--start", type=int, default=0, help="0-based offset into the moral list")
    parser.add_argument("--model", default="GPT-5.4-Nano", help="Model alias from config.yaml")
    parser.add_argument("--max-file-mb", type=float, default=190.0, help="Maximum input JSONL size per batch file")
    parser.add_argument("--max-requests-per-file", type=int, help="Maximum requests per batch input file")
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds to wait for batch completion")
    parser.add_argument("--collect-dir", help="Collect outputs for an existing batch run directory")
    return parser.parse_args()


def load_minimal_variant(cfg: dict) -> dict:
    for variant in cfg["prompt_variants"]:
        if variant["label"] == "minimal":
            return variant
    raise ValueError("No prompt variant labeled 'minimal' found")


def load_model_cfg(cfg: dict, alias: str) -> dict:
    for model in cfg["models"]:
        if model["alias"] == alias:
            if model["provider"] != "openai":
                raise ValueError(f"Batch smoke runner currently supports OpenAI only, got {model['provider']}")
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


def slice_queries(queries: list[dict], start: int, count: int | None) -> list[dict]:
    if start < 0:
        raise ValueError("--start must be non-negative")
    end = None if count is None else start + count
    return queries[start:end]


def build_request_body(model_id: str, system: str, user: str) -> dict:
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": RANKED_FABLES_SCHEMA,
        },
        "max_completion_tokens": 512,
    }


def build_request_line(
    model_cfg: dict,
    variant_cfg: dict,
    corpus_block: str,
    query: dict,
    top_k: int,
) -> str:
    _, user = render_prompt(query["moral_text"], corpus_block, variant_cfg, top_k)
    body = build_request_body(model_cfg["id"], variant_cfg["system"], user)
    request = {
        "custom_id": query["moral_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
    return json.dumps(request) + "\n"


def write_input_jsonl_parts(
    out_dir: Path,
    model_cfg: dict,
    variant_cfg: dict,
    corpus_block: str,
    queries: list[dict],
    top_k: int,
    max_file_mb: float,
    max_requests_per_file: int | None = None,
) -> list[Path]:
    max_bytes = int(max_file_mb * 1024 * 1024)
    parts: list[Path] = []
    current_lines: list[str] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal current_lines, current_bytes
        if not current_lines:
            return
        path = out_dir / f"input_{len(parts):03d}.jsonl"
        path.write_text("".join(current_lines))
        parts.append(path)
        current_lines = []
        current_bytes = 0

    for query in queries:
        line = build_request_line(model_cfg, variant_cfg, corpus_block, query, top_k)
        line_bytes = len(line.encode("utf-8"))
        if line_bytes > max_bytes:
            raise ValueError(
                f"Single request for {query['moral_id']} is {line_bytes} bytes, "
                f"larger than max-file limit {max_bytes}"
            )
        if current_lines and current_bytes + line_bytes > max_bytes:
            flush()
        if max_requests_per_file and len(current_lines) >= max_requests_per_file:
            flush()
        current_lines.append(line)
        current_bytes += line_bytes

    flush()
    return parts


def response_content_to_text(content_response: Any) -> str:
    if hasattr(content_response, "read"):
        data = content_response.read()
    elif hasattr(content_response, "content"):
        data = content_response.content
    elif hasattr(content_response, "text"):
        data = content_response.text
    else:
        data = str(content_response)

    if isinstance(data, bytes):
        return data.decode("utf-8")
    return str(data)


def parse_ranked_ids(output_obj: dict) -> list[str]:
    if output_obj.get("error"):
        return []

    body = output_obj.get("response", {}).get("body", {})
    choices = body.get("choices") or []
    if not choices:
        return []

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    parsed = json.loads(content)
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


def collect_existing(out_dir: Path) -> None:
    client = OpenAI()
    batches_path = out_dir / "batches.json"
    if not batches_path.exists():
        raise FileNotFoundError(f"No batches.json found in {out_dir}")

    batch_records = json.loads(batches_path.read_text())
    output_rows = []
    for record in batch_records:
        batch = client.batches.retrieve(record["batch_id"])
        record["status"] = batch.status
        record["batch"] = batch.model_dump()
        if batch.status != "completed" or not batch.output_file_id:
            continue
        output_path = out_dir / f"output_{record['part']:03d}.jsonl"
        if output_path.exists():
            output_text = output_path.read_text()
        else:
            output_text = response_content_to_text(client.files.content(batch.output_file_id))
            output_path.write_text(output_text)
        output_rows.extend(json.loads(line) for line in output_text.splitlines() if line.strip())

    batches_path.write_text(json.dumps(batch_records, indent=2) + "\n")
    by_id = {row["custom_id"]: row for row in output_rows}

    queries_by_id = {query["moral_id"]: query for query in load_queries(None)}
    metric_rows = []
    for moral_id in sorted(by_id, key=lambda value: int(value.removeprefix("moral_"))):
        query = queries_by_id[moral_id]
        ranked_ids = parse_ranked_ids(by_id[moral_id])
        metric_rows.append(
            compute_row_metrics(
                moral_id=query["moral_id"],
                moral_text=query["moral_text"],
                relevant_fable=query["relevant_fable"],
                ranked_ids=ranked_ids,
                latency_s=0.0,
            )
        )

    if not metric_rows:
        raise RuntimeError(f"No completed output rows found in {out_dir}")

    rows_path = out_dir / "rows.csv"
    summary_path = out_dir / "summary.json"
    write_rows_csv(rows_path, metric_rows)
    summary = {
        "run_date": date.today().isoformat(),
        "provider": "openai_batch",
        "variant_label": "minimal",
        **aggregate_metrics(metric_rows),
        "batch_ids": [record["batch_id"] for record in batch_records],
        "n_completed_batches": sum(1 for record in batch_records if record["status"] == "completed"),
        "n_total_batches": len(batch_records),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Collected n={summary['n_queries']}  "
        f"MRR@10={summary['MRR@10']:.4f}  R@10={summary['R@10']:.4f}"
    )
    print(f"Rows: {rows_path}")
    print(f"Summary: {summary_path}")


def main() -> None:
    args = parse_args()
    if args.collect_dir:
        collect_existing(Path(args.collect_dir))
        return

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    model_cfg = load_model_cfg(cfg, args.model)
    variant_cfg = load_minimal_variant(cfg)
    top_k = cfg["top_k"]

    queries = slice_queries(load_queries(None), args.start, args.test)
    fables = json.loads(FABLES_PATH.read_text())
    random.seed(42)
    random.shuffle(fables)
    corpus_block = build_corpus_block(fables)

    run_date = date.today().isoformat()
    range_label = f"start{args.start}_n{len(queries)}"
    run_name = f"{run_date}_{model_cfg['alias']}_minimal_batch_{range_label}"
    out_dir = BATCH_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = out_dir / "rows.csv"
    summary_path = out_dir / "summary.json"
    batches_path = out_dir / "batches.json"

    input_paths = write_input_jsonl_parts(
        out_dir,
        model_cfg,
        variant_cfg,
        corpus_block,
        queries,
        top_k,
        args.max_file_mb,
        args.max_requests_per_file,
    )
    for input_path in input_paths:
        print(f"Input JSONL: {input_path} ({input_path.stat().st_size / 1024 / 1024:.2f} MB)")

    client = OpenAI()
    batch_records = []
    for part_idx, input_path in enumerate(input_paths):
        uploaded_file = client.files.create(file=input_path.open("rb"), purpose="batch")
        print(f"Uploaded file part={part_idx}: {uploaded_file.id}")

        batch = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "experiment": "llm_retrieval",
                "variant": "minimal",
                "test_n": str(len(queries)),
                "part": str(part_idx),
                "n_parts": str(len(input_paths)),
            },
        )
        batch_records.append(
            {
                "part": part_idx,
                "input_path": str(input_path),
                "input_file_id": uploaded_file.id,
                "batch_id": batch.id,
                "status": batch.status,
                "batch": batch.model_dump(),
            }
        )
        print(f"Batch part={part_idx}: {batch.id} status={batch.status}")

    batches_path.write_text(json.dumps(batch_records, indent=2) + "\n")

    deadline = time.monotonic() + args.timeout
    terminal = {"completed", "failed", "expired", "cancelled"}
    while any(record["status"] not in terminal for record in batch_records):
        time.sleep(args.poll_interval)
        if time.monotonic() >= deadline:
            batches_path.write_text(json.dumps(batch_records, indent=2) + "\n")
            pending = [record["batch_id"] for record in batch_records if record["status"] not in terminal]
            raise TimeoutError(f"Batches still pending after {args.timeout}s: {pending}")
        for record in batch_records:
            if record["status"] in terminal:
                continue
            batch = client.batches.retrieve(record["batch_id"])
            record["status"] = batch.status
            record["batch"] = batch.model_dump()
            print(f"Batch part={record['part']}: {batch.id} status={batch.status} counts={batch.request_counts}")
        batches_path.write_text(json.dumps(batch_records, indent=2) + "\n")

    if any(record["status"] != "completed" for record in batch_records):
        raise RuntimeError(f"One or more batches did not complete; see {batches_path}")

    output_rows = []
    for record in batch_records:
        output_file_id = record["batch"].get("output_file_id")
        if not output_file_id:
            raise RuntimeError(f"Completed batch has no output_file_id; see {batches_path}")
        output_text = response_content_to_text(client.files.content(output_file_id))
        output_path = out_dir / f"output_{record['part']:03d}.jsonl"
        output_path.write_text(output_text)
        output_rows.extend(json.loads(line) for line in output_text.splitlines() if line.strip())
    by_id = {row["custom_id"]: row for row in output_rows}

    metric_rows = []
    for query in queries:
        ranked_ids = parse_ranked_ids(by_id.get(query["moral_id"], {}))
        metric_rows.append(
            compute_row_metrics(
                moral_id=query["moral_id"],
                moral_text=query["moral_text"],
                relevant_fable=query["relevant_fable"],
                ranked_ids=ranked_ids,
                latency_s=0.0,
            )
        )

    write_rows_csv(rows_path, metric_rows)
    summary = {
        "run_date": run_date,
        "model_alias": model_cfg["alias"],
        "model_id": model_cfg["id"],
        "provider": "openai_batch",
        "variant_label": "minimal",
        **aggregate_metrics(metric_rows),
        "batch_ids": [record["batch_id"] for record in batch_records],
        "input_file_ids": [record["input_file_id"] for record in batch_records],
        "output_file_ids": [record["batch"].get("output_file_id") for record in batch_records],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"MRR@10={summary['MRR@10']:.4f}  "
        f"R@1={summary['R@1']:.4f}  "
        f"R@10={summary['R@10']:.4f}"
    )
    print(f"Rows: {rows_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
