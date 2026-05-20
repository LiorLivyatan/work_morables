"""
Preset launcher for the remaining ft_11 saved-model evaluations.

This intentionally does not contain remote/GPU routing. Use run.sh for that:

    ./run.sh finetuning/ft_11_clustered/eval_remaining_non_gemini.py --remote --gpu 0

Preview the planned matrix without evaluating:

    ./run.sh finetuning/ft_11_clustered/eval_remaining_non_gemini.py --dry_run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from finetuning.ft_11_clustered import eval_saved_models


DEFAULT_MODELS = [
    "sfr",
    "linq",
    "qwen3",
    "qwen3_4b",
    "qwen3_0_6b",
    "bge",
    "bge_large",
    "bge_m3",
    "e5_large",
    "e5_base",
    "multilingual_e5_large",
    "multilingual_e5_large_instruct",
    "all_mpnet",
    "all_minilm",
    "contriever",
]

NON_RAW_DOC_CONFIGS = [
    "direct_moral_only",
    "conceptual_abstract_only",
    "proverb_only",
    "cot_proverb_only",
    "fable_direct_moral",
    "direct_moral_fable",
    "fable_conceptual_abstract",
    "conceptual_abstract_fable",
    "fable_proverb",
    "proverb_fable",
    "fable_cot_proverb",
    "cot_proverb_fable",
]

NON_GEMINI_GENERATORS = [
    "gemma4-E2B",
    "gemma4-E4B",
    "gemma4-31B",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the already-trained ft_11 raw 5FCV models on the remaining "
            "non-Gemini summary corpora."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model aliases to evaluate. Defaults to all working models, excluding embeddinggemma/stella.",
    )
    parser.add_argument(
        "--eval_doc_configs",
        nargs="+",
        default=NON_RAW_DOC_CONFIGS,
        help="Doc configs to evaluate. Defaults to all non-raw summary layouts.",
    )
    parser.add_argument(
        "--summary_generators",
        nargs="+",
        default=NON_GEMINI_GENERATORS,
        help="Summary generators to evaluate. Defaults to the three Gemma 4 generators.",
    )
    parser.add_argument("--trained_doc_config", default="raw")
    parser.add_argument("--trained_summary_generator")
    parser.add_argument("--fold", type=int, choices=range(5), metavar="0-4")
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Disable the default continue-on-error behavior.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the planned eval matrix and delegated command without running evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_configs = len(args.models) * len(args.eval_doc_configs) * len(args.summary_generators)
    n_fold_evals = n_configs * (1 if args.fold is not None else 5)

    delegated_argv = [
        "eval_saved_models.py",
        "--models",
        *args.models,
        "--trained_doc_config",
        args.trained_doc_config,
        "--eval_doc_configs",
        ",".join(args.eval_doc_configs),
        "--summary_generators",
        ",".join(args.summary_generators),
    ]
    if args.trained_summary_generator:
        delegated_argv.extend(["--trained_summary_generator", args.trained_summary_generator])
    if args.fold is not None:
        delegated_argv.extend(["--fold", str(args.fold)])
    if not args.stop_on_error:
        delegated_argv.append("--continue_on_error")

    plan = (
        "ft_11 remaining non-Gemini saved-model eval\n"
        f"models: {len(args.models)}\n"
        f"eval_doc_configs: {len(args.eval_doc_configs)}\n"
        f"summary_generators: {args.summary_generators}\n"
        f"aggregate configs: {n_configs}\n"
        f"fold evals: {n_fold_evals}\n"
        f"continue_on_error: {not args.stop_on_error}"
    )

    if args.dry_run:
        print(plan)
        print("\nDelegated argv:")
        print(" ".join(delegated_argv))
        return

    notify.send(f"{plan}\nstarting")
    old_argv = sys.argv[:]
    try:
        sys.argv = delegated_argv
        eval_saved_models.main()
    finally:
        sys.argv = old_argv
    notify.send(f"{plan}\ndone")


if __name__ == "__main__":
    main()
