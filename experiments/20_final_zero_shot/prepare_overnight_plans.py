from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent

PHASE1_STAGE = (
    EXP_DIR
    / "phase_1_raw_instruction_sweep/stage_metrics/"
    / "phase_1_raw_instruction_sweep_2026-05-17_21-20-42.csv"
)
TRACKING = ROOT / "docs/zero_shot_full_tracking_matrix.csv"
OUT_DIR = EXP_DIR / "run_plans"

TOP10_MODELS = [
    "Linq-Embed-Mistral",
    "Qwen3-Embedding-8B",
    "BGE-en-ICL",
    "Qwen3-Embedding-4B",
    "Instructor-xl",
    "Lychee-Embed",
    "Qwen3-Embedding-0.6B",
    "Nomic-Embed-v2-MoE",
    "Multilingual-E5-large-instruct",
    "E5-mistral-7b",
]

PHASE3_GENERATORS = {"gemma4-E2B", "gemma4-E4B", "gemma4-31B"}


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


def best_instructions(stage_rows: list[dict[str, str]]) -> dict[str, str]:
    best: dict[str, tuple[float, str]] = {}
    for row in stage_rows:
        if row["stage_status"] != "completed":
            continue
        score = float(row.get("MRR@10") or 0.0)
        model = row["model_alias"]
        if model not in best or score > best[model][0]:
            best[model] = (score, row["instruction_variant"])
    return {model: instr for model, (_, instr) in best.items()}


def main() -> None:
    fieldnames, tracking = read_csv(TRACKING)
    _, stage = read_csv(PHASE1_STAGE)
    best = best_instructions(stage)
    worked = set(best)

    non_raw = [r for r in tracking if r["corpus_config"] != "raw"]

    phase2_best = [
        r for r in non_raw
        if r["summary_generator"] == "gemini"
        and r["model_alias"] in worked
        and r["instruction_variant"] == best[r["model_alias"]]
    ]

    phase3_top10_gemma = [
        r for r in non_raw
        if r["model_alias"] in TOP10_MODELS
        and r["summary_generator"] in PHASE3_GENERATORS
        and r["instruction_variant"] == best[r["model_alias"]]
    ]

    phase2_extra_instr = [
        r for r in non_raw
        if r["summary_generator"] == "gemini"
        and r["model_alias"] in worked
        and r["instruction_variant"] != best[r["model_alias"]]
    ]

    write_csv(OUT_DIR / "gpu0_phase2_gemini_best_instruction.csv", fieldnames, phase2_best)
    write_csv(OUT_DIR / "gpu0_phase3_top10_gemma_generators.csv", fieldnames, phase3_top10_gemma)
    write_csv(OUT_DIR / "gpu3_optional_gemini_extra_instructions.csv", fieldnames, phase2_extra_instr)

    print(f"worked_models={len(worked)}")
    print(f"gpu0_phase2={len(phase2_best)}")
    print(f"gpu0_phase3={len(phase3_top10_gemma)}")
    print(f"gpu3_optional={len(phase2_extra_instr)}")
    print("best_instructions:")
    for model in sorted(best):
        print(f"  {model}: {best[model]}")


if __name__ == "__main__":
    main()
