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
    plan = EXP_DIR / "run_plans/gpu0_retry_error_models_full.csv"
    notify.send("final_zero_shot GPU0 retry full queue starting\nprevious error models")
    run_matrix_phase(
        phase_name="phase_9_retry_error_models_full",
        phase_description=(
            "Full runnable config retry for selected previous error models: "
            "GTE-Qwen2 1.5B/7B, GritLM-7B, SFR-Mistral, Stella-1.5B."
        ),
        args=args_for(plan),
    )
    notify.send("final_zero_shot GPU0 retry full queue done")


if __name__ == "__main__":
    main()
