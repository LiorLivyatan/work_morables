import csv
from pathlib import Path
from typing import Optional


RUN_FIELDNAMES = [
    "moral_id", "moral_text", "relevant_fable",
    "ranked_ids", "reciprocal_rank",
    "r_at_1", "r_at_5", "r_at_10", "ndcg_at_10",
    "rank", "latency_s",
]

UNIFIED_FIELDNAMES = [
    "run_date", "model_alias", "model_id", "provider", "variant_label",
    "n_queries", "MRR@10", "R@1", "R@5", "R@10", "NDCG@10",
    "Mean_Rank", "Median_Rank", "avg_latency_s",
]

_FLOAT_FIELDS = {"reciprocal_rank", "r_at_1", "r_at_5", "r_at_10", "ndcg_at_10", "latency_s"}
_INT_FIELDS   = {"rank"}


def _cast_row(row: dict) -> dict:
    result = dict(row)
    for field in _FLOAT_FIELDS:
        if field in result and result[field] not in ("", None):
            try:
                result[field] = float(result[field])
            except (ValueError, TypeError):
                pass
    for field in _INT_FIELDS:
        if field in result:
            try:
                result[field] = int(result[field]) if result[field] not in ("", "None", None) else None
            except (ValueError, TypeError):
                pass
    return result


def get_run_path(results_dir: Path, model_alias: str, variant_label: str, run_date: str) -> Path:
    filename = f"{run_date}_{model_alias}_{variant_label}.csv"
    return results_dir / "runs" / filename


def append_query_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RUN_FIELDNAMES})


def count_completed_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in csv.DictReader(f))


def load_completed_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [_cast_row(row) for row in csv.DictReader(f)]


def append_summary_row(unified_path: Path, summary: dict) -> None:
    unified_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not unified_path.exists()
    with open(unified_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: summary.get(k, "") for k in UNIFIED_FIELDNAMES})
