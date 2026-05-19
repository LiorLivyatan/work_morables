from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR))

from finetuning.lib import notify
from lib.matrix_runner import run_matrix_phase


def args_for(plan: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_plan_csv=str(plan),
        models=None,
        instructions=None,
        generators=None,
        generator_models=None,
        prompt_variants=None,
        corpus_configs=None,
        limit=None,
        force=False,
        dry_run=False,
    )


def main() -> None:
    plan = EXP_DIR / "run_plans/gpu3_complete_rank11_15_missing.csv"
    notify.send("final_zero_shot GPU3 rank 11-15 completion queue starting")
    run_matrix_phase(
        phase_name="phase_6_complete_rank11_15_missing",
        phase_description=(
            "Completion queue: models ranked 11-15 by current best MRR@10, "
            "missing runnable Gemma summary configs not covered by prior sweeps."
        ),
        args=args_for(plan),
    )
    notify.send("final_zero_shot GPU3 rank 11-15 completion queue done")


if __name__ == "__main__":
    main()
