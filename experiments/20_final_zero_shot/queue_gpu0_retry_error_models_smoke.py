from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR))

from finetuning.lib import notify
from lib.matrix_runner import add_common_args, run_matrix_phase


def args_for(plan: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    if not args.run_plan_csv:
        args.run_plan_csv = str(plan)
    return args


def main() -> None:
    plan = EXP_DIR / "run_plans/gpu0_retry_error_models_smoke.csv"
    notify.send("final_zero_shot GPU0 retry smoke starting\nprevious error models")
    run_matrix_phase(
        phase_name="phase_8_retry_error_models_smoke",
        phase_description=(
            "Smoke retry for previously failed models after loader compatibility "
            "and conservative batch-size changes."
        ),
        args=args_for(plan),
    )
    notify.send("final_zero_shot GPU0 retry smoke done")


if __name__ == "__main__":
    main()
