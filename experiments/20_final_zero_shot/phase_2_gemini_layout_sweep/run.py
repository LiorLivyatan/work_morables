"""
Phase 2 Gemini layout sweep for the final zero-shot experiment.

Default scope:
  - summary generator: gemini
  - corpus configs: every non-raw Gemini config in docs/zero_shot_full_tracking_matrix.csv

Use CLI filters to narrow or expand the run. Examples:
  ./run.sh experiments/20_final_zero_shot/phase_2_gemini_layout_sweep/run.py --dry-run
  ./run.sh experiments/20_final_zero_shot/phase_2_gemini_layout_sweep/run.py --models Linq-Embed-Mistral --instructions general --dry-run
  ./run.sh experiments/20_final_zero_shot/phase_2_gemini_layout_sweep/run.py --models Linq-Embed-Mistral --limit 1
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Final zero-shot Phase 2: Gemini layout sweep")
    add_common_args(parser)
    args = parser.parse_args()
    run_matrix_phase(
        phase_name="phase_2_gemini_layout_sweep",
        phase_description="Gemini summaries across summary-only, fable+summary, and summary+fable layouts.",
        args=args,
        phase_generators=["gemini"],
        phase_corpus_configs=None,
    )


if __name__ == "__main__":
    main()
