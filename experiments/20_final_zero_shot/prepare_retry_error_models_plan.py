from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent

TRACKING = ROOT / "docs/zero_shot_full_tracking_matrix.csv"
REMOTE_RESULTS = EXP_DIR / "remote_results"
OUT_DIR = EXP_DIR / "run_plans"

RETRY_MODELS = [
    "GTE-Qwen2-1.5B",
    "GTE-Qwen2-7B",
    "GritLM-7B",
    "SFR-Embedding-Mistral",
    "Stella-1.5B",
]

RUNNABLE_GENERATORS = {"gemini", "gemma4-E2B", "gemma4-E4B", "gemma4-31B"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_id(row: dict[str, str]) -> str:
    parts = [row["model_alias"], row["instruction_variant"], row["corpus_config"]]
    if row["corpus_config"] != "raw":
        parts.extend([row["summary_generator"], row["generator_prompt_variant"]])
    return "__".join(parts)


def completed_run_ids() -> set[str]:
    done: set[str] = set()
    for path in REMOTE_RESULTS.glob("*/results/*/metrics.csv"):
        _, rows = read_csv(path)
        for row in rows:
            if not row.get("error"):
                done.add(row["run_id"])
    return done


def is_runnable(row: dict[str, str]) -> bool:
    return row["corpus_config"] == "raw" or row["summary_generator"] in RUNNABLE_GENERATORS


def smoke_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = []
    for model in RETRY_MODELS:
        model_rows = [row for row in rows if row["model_alias"] == model and row["corpus_config"] == "raw"]
        if model in {"GritLM-7B", "Stella-1.5B"}:
            default_rows = [row for row in model_rows if row["instruction_variant"] == "default"]
            selected.append(default_rows[0] if default_rows else model_rows[0])
        else:
            no_instr_rows = [row for row in model_rows if row["instruction_variant"] == "no_instr"]
            selected.append(no_instr_rows[0] if no_instr_rows else model_rows[0])
    return selected


def main() -> None:
    fieldnames, tracking = read_csv(TRACKING)
    completed = completed_run_ids()
    model_order = {model: idx for idx, model in enumerate(RETRY_MODELS)}

    full_rows = []
    for row in tracking:
        if row["model_alias"] not in model_order:
            continue
        if run_id(row) in completed:
            continue
        if is_runnable(row):
            full_rows.append(row)

    full_rows.sort(
        key=lambda row: (
            model_order[row["model_alias"]],
            row["summary_generator"],
            row["instruction_variant"],
            row["generator_prompt_variant"],
            row["corpus_config"],
        )
    )
    smoke = smoke_rows(tracking)

    write_csv(OUT_DIR / "gpu0_retry_error_models_smoke.csv", fieldnames, smoke)
    write_csv(OUT_DIR / "gpu0_retry_error_models_full.csv", fieldnames, full_rows)

    print(f"completed_run_ids={len(completed)}")
    print(f"gpu0_retry_error_models_smoke={len(smoke)}")
    print(f"gpu0_retry_error_models_full={len(full_rows)}")
    by_model: dict[str, int] = {}
    by_generator: dict[str, int] = {}
    for row in full_rows:
        by_model[row["model_alias"]] = by_model.get(row["model_alias"], 0) + 1
        by_generator[row["summary_generator"]] = by_generator.get(row["summary_generator"], 0) + 1
    for model in RETRY_MODELS:
        print(f"  {model}: {by_model.get(model, 0)}")
    print(f"  generators: {dict(sorted(by_generator.items()))}")


if __name__ == "__main__":
    main()
