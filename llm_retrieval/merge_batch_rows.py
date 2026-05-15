"""
Merge OpenAI batch row CSVs into one retrieval result.

Usage
-----
./run.sh llm_retrieval/merge_batch_rows.py \\
  --inputs path/to/rows_a.csv path/to/rows_b.csv path/to/rows_c.csv \\
  --model-alias GPT-5.4-Nano \\
  --model-id gpt-5.4-nano
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from llm_retrieval.lib.eval import aggregate_metrics
from llm_retrieval.lib.results import append_summary_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge batch row CSVs")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "llm_retrieval/results/batches/merged"))
    return parser.parse_args()


def parse_row(row: dict) -> dict:
    parsed = dict(row)
    for key in ("reciprocal_rank", "r_at_1", "r_at_5", "r_at_10", "ndcg_at_10", "latency_s"):
        parsed[key] = float(parsed[key] or 0)
    parsed["rank"] = int(parsed["rank"]) if parsed.get("rank") else None
    return parsed


def main() -> None:
    args = parse_args()
    rows = []
    seen = set()
    for input_path in args.inputs:
        with Path(input_path).open(newline="") as f:
            for row in csv.DictReader(f):
                moral_id = row["moral_id"]
                if moral_id in seen:
                    raise ValueError(f"Duplicate moral_id across inputs: {moral_id}")
                seen.add(moral_id)
                rows.append(parse_row(row))

    rows.sort(key=lambda row: int(row["moral_id"].removeprefix("moral_")))
    agg = aggregate_metrics(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_date = date.today().isoformat()
    stem = f"{run_date}_{args.model_alias}_minimal_batch_n{len(rows)}"
    rows_path = out_dir / f"{stem}_rows.csv"
    summary_path = out_dir / f"{stem}_summary.json"

    with rows_path.open("w", newline="") as f:
        fieldnames = [
            "moral_id",
            "moral_text",
            "relevant_fable",
            "ranked_ids",
            "reciprocal_rank",
            "r_at_1",
            "r_at_5",
            "r_at_10",
            "ndcg_at_10",
            "rank",
            "latency_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            out_row["rank"] = out_row["rank"] if out_row["rank"] is not None else ""
            writer.writerow(out_row)

    summary = {
        "run_date": run_date,
        "model_alias": args.model_alias,
        "model_id": args.model_id,
        "provider": "openai_batch",
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


if __name__ == "__main__":
    main()
