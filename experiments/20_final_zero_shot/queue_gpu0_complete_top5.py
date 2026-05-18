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
    plan = EXP_DIR / "run_plans/gpu0_complete_top5_missing.csv"
    notify.send("final_zero_shot GPU0 completion queue starting\ntop 5 models missing runnable configs")
    run_matrix_phase(
        phase_name="phase_4_complete_top5_missing",
        phase_description=(
            "Completion queue: top 5 retrieval models, missing runnable "
            "Gemma summary configs not covered by prior sweeps."
        ),
        args=args_for(plan),
    )
    notify.send("final_zero_shot GPU0 completion queue done")


if __name__ == "__main__":
    main()
