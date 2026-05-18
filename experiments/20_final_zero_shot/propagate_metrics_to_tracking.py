from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent

TRACKING_CSV = ROOT / "docs/zero_shot_full_tracking_matrix.csv"
REMOTE_RESULTS = EXP_DIR / "remote_results"

METRIC_COLUMNS = [
    "MAP@10",
    "MRR@10",
    "NDCG@10",
    "Recall@5",
    "Recall@10",
    "Recall@15",
    "Recall@50",
    "Recall@100",
    "Recall@200",
    "Recall@300",
    "Hit@1",
    "Hit@5",
    "Hit@10",
    "Hit@100",
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tracking_run_id(row: dict[str, str]) -> str:
    parts = [row["model_alias"], row["instruction_variant"], row["corpus_config"]]
    if row["corpus_config"] != "raw":
        parts.extend([row["summary_generator"], row["generator_prompt_variant"]])
    return "__".join(parts)


def collect_metrics() -> dict[str, dict[str, str]]:
    metrics_by_run: dict[str, dict[str, str]] = {}
    for path in sorted(REMOTE_RESULTS.glob("*/results/*/metrics.csv")):
        phase_name = path.parts[-4]
        _, rows = read_csv(path)
        for row in rows:
            if row.get("error"):
                continue
            run_id = row["run_id"]
            metrics_by_run[run_id] = {**row, "source_phase": phase_name}
    return metrics_by_run


def main() -> None:
    fieldnames, tracking_rows = read_csv(TRACKING_CSV)
    metrics_by_run = collect_metrics()

    updated = 0
    already_filled = 0
    for row in tracking_rows:
        run_id = tracking_run_id(row)
        metrics = metrics_by_run.get(run_id)
        if not metrics:
            continue
        if any(row.get(col) for col in METRIC_COLUMNS):
            already_filled += 1
        for col in METRIC_COLUMNS:
            row[col] = metrics.get(col, "")
        row["exp_id"] = metrics.get("source_phase", row.get("exp_id", ""))
        updated += 1

    write_csv(TRACKING_CSV, fieldnames, tracking_rows)
    print(f"metrics_sources={len(metrics_by_run)}")
    print(f"tracking_rows={len(tracking_rows)}")
    print(f"updated_rows={updated}")
    print(f"already_filled_rows={already_filled}")


if __name__ == "__main__":
    main()
