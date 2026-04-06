import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lib.pipeline import run_matrix_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run experiment 10 model-matrix pipeline."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to the experiment config YAML.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="Path to an existing run directory to continue instead of starting fresh.",
    )
    args = parser.parse_args()
    run_matrix_experiment(
        config_path=args.config_path.resolve(),
        resume_run_dir=args.resume,
    )


if __name__ == "__main__":
    main()
