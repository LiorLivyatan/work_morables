"""
Phase 3 generator comparison for the final zero-shot experiment.

Default scope:
  - all non-raw rows from docs/zero_shot_full_tracking_matrix.csv
  - all configured summary generators whose summary file is present

Use CLI filters to make this as small or broad as needed. Examples:
  ./run.sh experiments/20_final_zero_shot/phase_3_generator_comparison/run.py --dry-run
  ./run.sh experiments/20_final_zero_shot/phase_3_generator_comparison/run.py --models Linq-Embed-Mistral Qwen3-Embedding-8B --instructions general --dry-run
  ./run.sh experiments/20_final_zero_shot/phase_3_generator_comparison/run.py --generators gemini gemma4-31B --prompt-variants direct_moral conceptual_abstract --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR))

from lib.matrix_runner import add_common_args, run_matrix_phase

NON_RAW_CORPUS_CONFIGS = [
    "fable_direct_moral", "direct_moral_only", "direct_moral_fable",
    "fable_cot_proverb", "cot_proverb_only", "cot_proverb_fable",
    "fable_conceptual_abstract", "conceptual_abstract_only", "conceptual_abstract_fable",
    "fable_proverb", "proverb_only", "proverb_fable",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Final zero-shot Phase 3: generator comparison")
    add_common_args(parser)
    args = parser.parse_args()
    run_matrix_phase(
        phase_name="phase_3_generator_comparison",
        phase_description="Compare summary generators, prompt variants, and layouts using the clustered benchmark.",
        args=args,
        phase_generators=None,
        phase_corpus_configs=NON_RAW_CORPUS_CONFIGS,
    )


if __name__ == "__main__":
    main()
