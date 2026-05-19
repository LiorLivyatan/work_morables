from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR))

from lib.matrix_runner import run_matrix_phase
from finetuning.lib import notify


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
    plan_dir = EXP_DIR / "run_plans"
    notify.send("final_zero_shot GPU0 overnight queue starting\nphase2: 276\nphase3: 360")
    run_matrix_phase(
        phase_name="phase_2_gemini_layout_sweep",
        phase_description="Overnight queue: Gemini summaries, best instruction, all successful Phase 1 models.",
        args=args_for(plan_dir / "gpu0_phase2_gemini_best_instruction.csv"),
    )
    run_matrix_phase(
        phase_name="phase_3_generator_comparison",
        phase_description="Overnight queue: top 10 retrieval models across Gemma summary generators.",
        args=args_for(plan_dir / "gpu0_phase3_top10_gemma_generators.csv"),
    )
    notify.send("final_zero_shot GPU0 overnight queue done")


if __name__ == "__main__":
    main()
