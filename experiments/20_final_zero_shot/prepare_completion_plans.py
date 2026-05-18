from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent

TRACKING = ROOT / "docs/zero_shot_full_tracking_matrix.csv"
REMOTE_RESULTS = EXP_DIR / "remote_results"
OUT_DIR = EXP_DIR / "run_plans"

GPU0_TOP5 = [
    "Qwen3-Embedding-8B",
    "Linq-Embed-Mistral",
    "Qwen3-Embedding-4B",
    "BGE-en-ICL",
    "Instructor-xl",
]

GPU3_NEXT5 = [
    "E5-mistral-7b",
    "Multilingual-E5-large-instruct",
    "Qwen3-Embedding-0.6B",
    "Lychee-Embed",
    "DRAMA-1B",
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


def runnable_missing_rows(
    tracking: list[dict[str, str]],
    completed: set[str],
    models: list[str],
) -> list[dict[str, str]]:
    model_set = set(models)
    selected = []
    for row in tracking:
        if row["model_alias"] not in model_set:
            continue
        if run_id(row) in completed:
            continue
        if row["corpus_config"] == "raw":
            selected.append(row)
            continue
        if row["summary_generator"] in RUNNABLE_GENERATORS:
            selected.append(row)
    order = {model: idx for idx, model in enumerate(models)}
    selected.sort(
        key=lambda row: (
            order[row["model_alias"]],
            row["summary_generator"],
            row["instruction_variant"],
            row["generator_prompt_variant"],
            row["corpus_config"],
        )
    )
    return selected


def main() -> None:
    fieldnames, tracking = read_csv(TRACKING)
    completed = completed_run_ids()

    gpu0_rows = runnable_missing_rows(tracking, completed, GPU0_TOP5)
    gpu3_rows = runnable_missing_rows(tracking, completed, GPU3_NEXT5)

    write_csv(OUT_DIR / "gpu0_complete_top5_missing.csv", fieldnames, gpu0_rows)
    write_csv(OUT_DIR / "gpu3_complete_next5_missing.csv", fieldnames, gpu3_rows)

    print(f"completed_run_ids={len(completed)}")
    print(f"gpu0_complete_top5_missing={len(gpu0_rows)}")
    print(f"gpu3_complete_next5_missing={len(gpu3_rows)}")
    for name, rows in [("gpu0", gpu0_rows), ("gpu3", gpu3_rows)]:
        print(f"\n{name}:")
        by_model: dict[str, int] = {}
        by_generator: dict[str, int] = {}
        for row in rows:
            by_model[row["model_alias"]] = by_model.get(row["model_alias"], 0) + 1
            by_generator[row["summary_generator"]] = by_generator.get(row["summary_generator"], 0) + 1
        for model in sorted(by_model, key=lambda m: (GPU0_TOP5 + GPU3_NEXT5).index(m)):
            print(f"  {model}: {by_model[model]}")
        print(f"  generators: {dict(sorted(by_generator.items()))}")


if __name__ == "__main__":
    main()
