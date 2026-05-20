"""Prepare first-wave STORAL reranking plan from existing FT12 results.

Run with:
    ./run.sh experiments/21_storal_reranking/prepare_plan.py --top-n 15
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify

CONFIG_PATH = EXP_DIR / "config.yaml"
PLAN_DIR = EXP_DIR / "run_plans"


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def local_rankings_path(remote_or_local: str, rankings_dir: Path) -> Path:
    name = Path(remote_or_local).name
    direct = Path(remote_or_local)
    if direct.exists():
        return direct
    return rankings_dir / name


def parse_float(row: dict, name: str) -> float:
    value = row.get(name, "")
    return float(value) if value not in ("", None) else -1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare top STORAL reranking configs")
    parser.add_argument("--top-n", type=int, help="Number of first-stage configs to select")
    parser.add_argument("--metric", help="Selection metric, default from config.yaml")
    parser.add_argument("--output", default=str(PLAN_DIR / "top_recall100_storal.csv"))
    parser.add_argument("--require-ranking", action="store_true", default=True)
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    metric = args.metric or config["selection_metric"]
    top_n = args.top_n or int(config["default_top_n"])
    results_csv = resolve_path(config["paths"]["ft12_results_csv"])
    rankings_dir = resolve_path(config["paths"]["ft12_rankings_dir"])
    output = resolve_path(args.output)

    notify.send(
        f"storal reranking plan starting\n"
        f"metric: {metric}\n"
        f"top_n: {top_n}"
    )

    latest: dict[tuple[str, str, str, str], dict] = {}
    with results_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            key = (
                row["model_alias"],
                row["size"],
                row["eval_doc_config"],
                row.get("summary_generator") or "none",
            )
            if key not in latest or row.get("timestamp", "") >= latest[key].get("timestamp", ""):
                latest[key] = row

    candidates = []
    for row in latest.values():
        ranking_path = local_rankings_path(row.get("rankings_path", ""), rankings_dir)
        if args.require_ranking and not ranking_path.exists():
            continue
        row = dict(row)
        row["local_rankings_path"] = str(ranking_path)
        row["_selection_score"] = parse_float(row, metric)
        candidates.append(row)

    candidates.sort(
        key=lambda r: (
            r["_selection_score"],
            parse_float(r, "MAP@10"),
            parse_float(r, "NDCG@10"),
            parse_float(r, "MRR@10"),
        ),
        reverse=True,
    )
    selected = candidates[:top_n]

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "selection_metric",
        "selection_score",
        "model_alias",
        "model_id",
        "size",
        "eval_doc_config",
        "summary_generator",
        "MAP@10",
        "MRR@10",
        "NDCG@10",
        "Recall@10",
        "Recall@100",
        "Recall@200",
        "Hit@10",
        "rankings_path",
        "local_rankings_path",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(selected, start=1):
            writer.writerow(
                {
                    "rank": idx,
                    "selection_metric": metric,
                    "selection_score": row["_selection_score"],
                    "model_alias": row["model_alias"],
                    "model_id": row["model_id"],
                    "size": row["size"],
                    "eval_doc_config": row["eval_doc_config"],
                    "summary_generator": row.get("summary_generator") or "none",
                    "MAP@10": row.get("MAP@10", ""),
                    "MRR@10": row.get("MRR@10", ""),
                    "NDCG@10": row.get("NDCG@10", ""),
                    "Recall@10": row.get("Recall@10", ""),
                    "Recall@100": row.get("Recall@100", ""),
                    "Recall@200": row.get("Recall@200", ""),
                    "Hit@10": row.get("Hit@10", ""),
                    "rankings_path": row.get("rankings_path", ""),
                    "local_rankings_path": row["local_rankings_path"],
                }
            )

    print(f"Selected {len(selected)} configs -> {output}")
    for row in selected[: min(10, len(selected))]:
        print(
            f"#{row['rank'] if 'rank' in row else ''} "
            f"{row['model_alias']} {row['size']} {row['eval_doc_config']}/"
            f"{row.get('summary_generator') or 'none'} "
            f"{metric}={row['_selection_score']:.4f} MAP@10={parse_float(row, 'MAP@10'):.4f}"
        )

    notify.send(
        f"storal reranking plan done\n"
        f"selected: {len(selected)}\n"
        f"output: {output.name}"
    )


if __name__ == "__main__":
    main()

