"""
run_pipeline.py — Generic pipeline entry point for exp08.

Usage:
  python experiments/08_symmetric_moral_matching/run_pipeline.py
  python experiments/08_symmetric_moral_matching/run_pipeline.py --run-dir path/to/run_dir
  python experiments/08_symmetric_moral_matching/run_pipeline.py --force
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lib.pipeline import run_experiment

parser = argparse.ArgumentParser(description="Run exp08 via generic pipeline")
parser.add_argument("--run-dir", type=Path, default=None,
                    help="Existing run dir to continue (default: create new)")
parser.add_argument("--force", action="store_true",
                    help="Re-run steps even if output already exists")
args = parser.parse_args()

run_experiment(
    config_path=Path(__file__).parent / "config.yaml",
    run_dir=args.run_dir,
    force=args.force,
)
