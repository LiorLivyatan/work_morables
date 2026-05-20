"""
Evaluate saved ft_11_clustered fold models on configurable document layouts.

Run with:
    ./run.sh finetuning/ft_11_clustered/eval_saved_models.py --models all --eval_doc_configs all --summary_generators all --remote --gpu 3
"""
import argparse
import gc
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import set_seed

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from finetuning.ft_11_clustered.train import (
    CONFIG_PATH,
    FOLDS_PATH,
    RESULTS_DIR,
    build_doc_texts,
    evaluate_and_rank,
    load_clustered,
    load_json,
    load_qrels,
)

EVAL_RESULTS_DIR = RESULTS_DIR / "saved_model_evals"
EVAL_RANKINGS_DIR = EVAL_RESULTS_DIR / "rankings"


def parse_csv_or_all(value: str | None, options: list[str], default: list[str]) -> list[str]:
    if value is None:
        return default
    values = [part.strip() for part in value.split(",") if part.strip()]
    if "all" in values:
        return options
    unknown = [v for v in values if v not in options]
    if unknown:
        raise KeyError(f"Unknown values {unknown!r}. Options: {options}")
    return values


def parse_models(values: list[str] | None, options: list[str]) -> list[str]:
    if not values:
        return options
    if "all" in values:
        return options
    unknown = [v for v in values if v not in options]
    if unknown:
        raise KeyError(f"Unknown model aliases {unknown!r}. Options: {options}")
    return values


def available_generators() -> list[str]:
    root = ROOT / "experiments" / "20_final_zero_shot" / "summary_inputs"
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "golden_summaries.json").exists())


def metric_summary(metrics: dict) -> str:
    return (
        f"MRR={metrics.get('MRR', 0.0):.4f} "
        f"MAP={metrics.get('MAP', 0.0):.4f} "
        f"NDCG@10={metrics.get('NDCG@10', 0.0):.4f} "
        f"Hit@10={metrics.get('Hit@10', 0.0):.4f} "
        f"Recall@100={metrics.get('Recall@100', 0.0):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved ft_11 clustered fold models")
    parser.add_argument("--models", nargs="+", help="Model aliases to evaluate, or 'all'")
    parser.add_argument("--trained_doc_config", default="raw")
    parser.add_argument("--trained_summary_generator")
    parser.add_argument("--eval_doc_configs", default="raw", help="Comma-separated doc configs, or 'all'")
    parser.add_argument("--summary_generators", default="gemini", help="Comma-separated generators, or 'all'")
    parser.add_argument("--fold", type=int, choices=range(5), metavar="0-4")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    seed = int(config["seed"])
    set_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_aliases = parse_models(args.models, list(config["models"]))
    eval_doc_configs = parse_csv_or_all(args.eval_doc_configs, list(config["doc_configs"]), ["raw"])
    generators = parse_csv_or_all(args.summary_generators, available_generators(), ["gemini"])
    folds = load_json(FOLDS_PATH)
    fold_indices = [args.fold] if args.fold is not None else list(range(len(folds)))

    morals, fables, qrels = load_clustered()
    query_ids_all = [m["doc_id"] for m in morals]
    doc_ids = [f["doc_id"] for f in fables]
    relevant_all = load_qrels(qrels, query_ids_all, doc_ids)

    notify.send(
        f"ft_11 saved-model eval starting\n"
        f"models: {model_aliases}\n"
        f"trained: {args.trained_doc_config}/{args.trained_summary_generator or 'none'}\n"
        f"eval_doc_configs: {eval_doc_configs}\n"
        f"generators: {generators}\n"
        f"folds: {fold_indices}  seed: {seed}"
    )

    print(
        f"\n[ft_11_eval_saved] models={model_aliases} trained={args.trained_doc_config}/"
        f"{args.trained_summary_generator or 'none'} eval_doc_configs={eval_doc_configs} "
        f"generators={generators} folds={fold_indices} seed={seed}"
    )

    results = []
    errors = []
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RANKINGS_DIR.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    for model_alias in model_aliases:
        model_cfg = dict(config["models"][model_alias])
        instruction = model_cfg.get("query_instruction", "")
        model_root = Path(model_cfg["model_output_dir"])
        trained_generator = args.trained_summary_generator or "none"

        for fold_idx in fold_indices:
            model_dir = model_root / args.trained_doc_config / trained_generator / f"fold_{fold_idx}"
            try:
                if not model_dir.exists():
                    raise FileNotFoundError(f"Missing saved model: {model_dir}")

                print(f"\n[load] model={model_alias} fold={fold_idx} <- {model_dir}")
                model = SentenceTransformer(str(model_dir))
                test_indices = list(folds[fold_idx]["test"])
                query_ids = [morals[i]["doc_id"] for i in test_indices]
                query_texts = [f"{instruction}{morals[i]['text']}" for i in test_indices]
                local_relevant = {j: set(relevant_all[i]) for j, i in enumerate(test_indices)}

                for doc_config_name in eval_doc_configs:
                    doc_config = config["doc_configs"][doc_config_name]
                    eval_generators = [None] if doc_config.get("summary_variant") is None else generators
                    for generator in eval_generators:
                        try:
                            doc_texts = build_doc_texts(fables, doc_config, generator)
                            tag = "__".join(
                                [
                                    model_alias,
                                    f"trained_{args.trained_doc_config}",
                                    trained_generator,
                                    f"eval_{doc_config_name}",
                                    generator or "none",
                                    f"fold_{fold_idx}",
                                ]
                            )
                            rankings_path = EVAL_RANKINGS_DIR / f"{tag}.json"
                            metrics = evaluate_and_rank(
                                model,
                                query_texts,
                                doc_texts,
                                local_relevant,
                                query_ids,
                                doc_ids,
                                list(config["ks"]),
                                rankings_path,
                            )
                            result = {
                                "model_alias": model_alias,
                                "fold": fold_idx,
                                "trained_doc_config": args.trained_doc_config,
                                "trained_summary_generator": args.trained_summary_generator,
                                "eval_doc_config": doc_config_name,
                                "eval_summary_generator": generator,
                                "rankings_path": str(rankings_path),
                                "metrics": metrics,
                            }
                            results.append(result)
                            print(f"  [eval] {tag} {metric_summary(metrics)}")
                        except Exception as exc:  # noqa: BLE001 - overnight eval queue should keep running.
                            error = {
                                "model_alias": model_alias,
                                "fold": fold_idx,
                                "eval_doc_config": doc_config_name,
                                "eval_summary_generator": generator,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                            errors.append(error)
                            print(f"  [error] {error['error_type']}: {error['error']}")
                            if not args.continue_on_error:
                                raise

                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:  # noqa: BLE001
                error = {
                    "model_alias": model_alias,
                    "fold": fold_idx,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(error)
                print(f"  [error] model={model_alias} fold={fold_idx}: {type(exc).__name__}: {exc}")
                if not args.continue_on_error:
                    raise

    metric_names = sorted(results[0]["metrics"].keys()) if results else []
    grouped = {}
    for result in results:
        key = (
            result["model_alias"],
            result["trained_doc_config"],
            result["trained_summary_generator"] or "none",
            result["eval_doc_config"],
            result["eval_summary_generator"] or "none",
        )
        grouped.setdefault(key, []).append(result)

    aggregate = []
    for key, rows in sorted(grouped.items()):
        means = {
            metric: float(np.mean([r["metrics"][metric] for r in rows]))
            for metric in metric_names
            if metric in rows[0]["metrics"] and isinstance(rows[0]["metrics"][metric], (int, float))
        }
        aggregate.append(
            {
                "model_alias": key[0],
                "trained_doc_config": key[1],
                "trained_summary_generator": None if key[2] == "none" else key[2],
                "eval_doc_config": key[3],
                "eval_summary_generator": None if key[4] == "none" else key[4],
                "completed_folds": [r["fold"] for r in rows],
                "mean_metrics": means,
            }
        )

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = EVAL_RESULTS_DIR / f"{ts}_saved_model_eval.json"
    out.write_text(
        json.dumps(
            {
                "experiment": "ft_11_clustered_saved_model_eval",
                "seed": seed,
                "models": model_aliases,
                "trained_doc_config": args.trained_doc_config,
                "trained_summary_generator": args.trained_summary_generator,
                "eval_doc_configs": eval_doc_configs,
                "summary_generators": generators,
                "folds": fold_indices,
                "n_results": len(results),
                "n_errors": len(errors),
                "errors": errors,
                "aggregate": aggregate,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nResults -> {out}")
    notify.send(
        f"ft_11 saved-model eval done\n"
        f"results: {len(results)}  errors: {len(errors)}\n"
        f"output: {out.name}"
    )


if __name__ == "__main__":
    main()
