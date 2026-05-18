from __future__ import annotations

import argparse
import runpy
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


def run_gemma_generation(model_alias: str) -> None:
    script = ROOT / "experiments/13_gemma4_gpu_summarization/generate.py"
    output = (
        Path("/data/lior/final_zero_shot/gemma4_generation")
        / f"{model_alias}_summaries.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    old_argv = sys.argv[:]
    sys.argv = [
        str(script),
        "--models",
        model_alias,
        "--variants",
        "cot_proverb",
        "direct_moral",
        "conceptual_abstract",
        "proverb",
        "--output",
        str(output),
    ]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def main() -> None:
    notify.send("final_zero_shot GPU3 optional queue starting\nextra instruction sweep + Gemma E4B/31B generation")
    run_matrix_phase(
        phase_name="phase_2_extra_instruction_sweep",
        phase_description="Overnight optional queue: Gemini summaries with non-best instruction variants.",
        args=args_for(EXP_DIR / "run_plans/gpu3_optional_gemini_extra_instructions.csv"),
    )
    run_gemma_generation("gemma4-E4B")
    run_gemma_generation("gemma4-31B")
    notify.send("final_zero_shot GPU3 optional queue done")


if __name__ == "__main__":
    main()
