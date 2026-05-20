"""
Run the missing ft_12 STORAL rows from docs/ft_12_storal_full_tracking_matrix.csv.

This script contains no remote/GPU routing. Use run.sh:

    ./run.sh finetuning/ft_12_storal_transfer_clustered/run_remaining_from_tracking.py --remote --gpu 0

Preview only:

    ./run.sh finetuning/ft_12_storal_transfer_clustered/run_remaining_from_tracking.py --dry_run
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import set_seed

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.ft_12_storal_transfer_clustered import train
from finetuning.lib import notify

TRACKING_CSV = ROOT / "docs" / "ft_12_storal_full_tracking_matrix.csv"
RESULTS_DIR = EXP_DIR / "results"

DEFAULT_SKIP_MODELS = ("embeddinggemma", "stella")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only empty rows from the ft_12 STORAL tracking matrix."
    )
    parser.add_argument("--tracking_csv", default=str(TRACKING_CSV))
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional model aliases to include. Defaults to all models in the tracking CSV.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        help="Optional STORAL sizes to include. Defaults to all sizes in the tracking CSV.",
    )
    parser.add_argument(
        "--skip_models",
        nargs="+",
        default=list(DEFAULT_SKIP_MODELS),
        help="Models to skip. Defaults to known problematic embeddinggemma and stella.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        default=True,
        help="Continue after individual model/size/generator failures. Enabled by default.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Disable continue-on-error behavior.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_missing_groups(
    tracking_csv: Path,
    models: set[str] | None,
    sizes: set[str] | None,
    skip_models: set[str],
) -> dict[tuple[str, str, str | None], set[str]]:
    groups: dict[tuple[str, str, str | None], set[str]] = defaultdict(set)
    with tracking_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("MAP@10") or "").strip():
                continue
            model = row["ft_model_key"]
            size = row["storal_size"]
            if model in skip_models:
                continue
            if models is not None and model not in models:
                continue
            if sizes is not None and size not in sizes:
                continue

            doc_config = row["corpus_config"]
            generator = (row.get("summary_generator") or "").strip() or None
            if doc_config == "raw":
                generator = None
            groups[(model, size, generator)].add(doc_config)
    return groups


def write_result_json(result: dict, config: dict, seed: int) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = RESULTS_DIR / f"{ts}_{result['model_alias']}_{result['size']}.json"
    out.write_text(
        json.dumps(
            {
                "experiment": "ft_12_storal_transfer_clustered",
                "seed": seed,
                "early_stopping_metric": config.get("early_stopping_metric", "ndcg@10"),
                "exact_moral_masking": bool(config.get("exact_moral_masking", True)),
                "result": result,
            },
            indent=2,
        )
    )
    return out


def main() -> None:
    args = parse_args()
    continue_on_error = args.continue_on_error and not args.stop_on_error

    with train.CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)

    seed = int(config["seed"])
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    groups = load_missing_groups(
        Path(args.tracking_csv),
        set(args.models) if args.models else None,
        set(args.sizes) if args.sizes else None,
        set(args.skip_models or []),
    )
    ordered = sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or ""))
    total_eval_rows = sum(len(doc_configs) for _, doc_configs in ordered)

    plan = (
        "ft_12 STORAL remaining tracking queue\n"
        f"tracking_csv: {args.tracking_csv}\n"
        f"groups: {len(ordered)}\n"
        f"eval rows: {total_eval_rows}\n"
        f"skip_models: {args.skip_models}\n"
        f"continue_on_error: {continue_on_error}\n"
        f"seed: {seed}"
    )

    print(plan)
    print("\nPlanned groups:")
    for (model, size, generator), doc_configs in ordered:
        print(f"  {model} {size} {generator or 'none'}: {len(doc_configs)} configs")

    if args.dry_run:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    notify.send(f"{plan}\nstarting")

    queue_results = []
    queue_errors = []
    use_wandb = config.get("wandb", {}).get("enabled", False) and not args.no_wandb
    for (model_alias, size_name, summary_generator), doc_configs in ordered:
        doc_config_list = sorted(doc_configs)
        try:
            result = train.run_one(
                model_alias=model_alias,
                size_name=size_name,
                config=config,
                force=args.force,
                skip_train=False,
                eval_doc_configs=doc_config_list,
                summary_generator=summary_generator,
                use_wandb=use_wandb,
            )
            queue_results.append(result)
            out = write_result_json(result, config, seed)
            print(f"  Results -> {out}")
            train.append_comprehensive_csv(result, config)
            notify.send(
                "ft_12 remaining group done\n"
                f"model: {model_alias}  size: {size_name}\n"
                f"generator: {summary_generator or 'none'}  configs: {len(doc_config_list)}"
            )
        except Exception as exc:  # noqa: BLE001 - overnight queues should survive individual failures.
            error = {
                "model_alias": model_alias,
                "size": size_name,
                "summary_generator": summary_generator,
                "doc_configs": doc_config_list,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            queue_errors.append(error)
            train.append_comprehensive_csv(
                {
                    "model_alias": model_alias,
                    "model_config": config["models"].get(model_alias, {}),
                    "size": size_name,
                },
                config,
                error=f"{type(exc).__name__}: {exc}",
            )
            print(
                f"  [error] model={model_alias} size={size_name} "
                f"generator={summary_generator or 'none'}: {type(exc).__name__}: {exc}"
            )
            notify.send(
                "ft_12 remaining group failed\n"
                f"model: {model_alias}  size: {size_name}\n"
                f"generator: {summary_generator or 'none'}\n"
                f"{type(exc).__name__}: {exc}"
            )
            if not continue_on_error:
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = RESULTS_DIR / f"{ts}_remaining_tracking_queue_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "experiment": "ft_12_storal_transfer_clustered_remaining_tracking",
                "seed": seed,
                "tracking_csv": args.tracking_csv,
                "skip_models": args.skip_models,
                "groups_planned": len(ordered),
                "eval_rows_planned": total_eval_rows,
                "completed_groups": len(queue_results),
                "errors": queue_errors,
            },
            indent=2,
        )
    )
    print(f"\n[queue] summary -> {summary_path}")
    notify.send(
        "ft_12 remaining queue done\n"
        f"completed groups: {len(queue_results)}  errors: {len(queue_errors)}\n"
        f"summary: {summary_path}"
    )

    if queue_errors and not continue_on_error:
        raise RuntimeError(f"Queue completed with {len(queue_errors)} error(s)")


if __name__ == "__main__":
    main()
