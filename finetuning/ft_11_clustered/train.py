"""
ft_11_clustered - clustered MORABLES fine-tuning.

Run with:
    ./run.sh finetuning/ft_11_clustered/train.py --model sfr --doc_config direct_moral_only --summary_generator gemini --fold 0 --remote --gpu 0
"""
import argparse
import gc
import traceback
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from transformers import TrainerCallback, set_seed

from finetuning.lib import notify
from finetuning.lib.losses import InfoNCELoss
from lib.retrieval_utils import compute_multilabel_metrics_from_matrix

CACHE_DIR = EXP_DIR / "cache"
RESULTS_DIR = EXP_DIR / "results"
RANKINGS_DIR = RESULTS_DIR / "rankings"
CONFIG_PATH = EXP_DIR / "config.yaml"
FOLDS_PATH = EXP_DIR / "data" / "folds.json"

CLUSTERED_DIR = ROOT / "data" / "clustered"
MORALS_PATH = CLUSTERED_DIR / "morals_unique_corpus.json"
FABLES_PATH = CLUSTERED_DIR / "fables_corpus.json"
QRELS_PATH = CLUSTERED_DIR / "qrels_moral_to_fable_clustered.json"
SUMMARY_ROOT = ROOT / "experiments" / "20_final_zero_shot" / "summary_inputs"


class BestAdapterCallback(TrainerCallback):
    """Save the best LoRA adapter state on CPU to avoid GPU reload OOMs."""

    def __init__(self, peft_model, metric_key: str):
        self.peft_model = peft_model
        self.metric_key = metric_key
        self.best_score = -1.0
        self.best_state: dict | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        score = metrics.get(self.metric_key)
        if score is not None and float(score) > self.best_score:
            self.best_score = float(score)
            self.best_state = {
                k: v.detach().cpu().clone()
                for k, v in self.peft_model.state_dict().items()
                if "lora" in k.lower()
            }
            print(f"    [best] {self.metric_key}={float(score):.4f} - adapter saved to CPU")

    def restore(self, peft_model) -> float:
        if self.best_state is None:
            return self.best_score
        device = next(peft_model.parameters()).device
        state = peft_model.state_dict()
        for k, v in self.best_state.items():
            state[k] = v.to(device)
        peft_model.load_state_dict(state)
        print(f"    [best] Restored best adapter ({self.metric_key}={self.best_score:.4f})")
        return self.best_score


def evaluator_metric_key(fold_idx: int, metric: str) -> str:
    """Return the SentenceTransformers evaluator metric key used by HF Trainer."""
    aliases = {
        "mrr": "mrr@10",
        "map": "map@100",
        "ndcg": "ndcg@10",
        "hit": "accuracy@10",
        "hit@10": "accuracy@10",
    }
    metric = aliases.get(metric.lower(), metric.lower())
    allowed = {
        "mrr@10",
        "ndcg@10",
        "map@100",
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "accuracy@1",
        "accuracy@5",
        "accuracy@10",
    }
    if metric not in allowed:
        raise ValueError(f"Unsupported early_stopping_metric={metric!r}; supported: {sorted(allowed)}")
    return f"eval_fold_{fold_idx}_cosine_{metric}"


def load_json(path: Path):
    return json.loads(path.read_text())


def load_clustered() -> tuple[list[dict], list[dict], list[dict]]:
    return load_json(MORALS_PATH), load_json(FABLES_PATH), load_json(QRELS_PATH)


def load_qrels(qrels: list[dict], query_ids: list[str], doc_ids: list[str]) -> dict[int, set[int]]:
    query_to_idx = {qid: i for i, qid in enumerate(query_ids)}
    doc_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    relevant: dict[int, set[int]] = defaultdict(set)
    for row in qrels:
        if int(row.get("relevance", 1)) <= 0:
            continue
        relevant[query_to_idx[row["query_id"]]].add(doc_to_idx[row["doc_id"]])
    return dict(relevant)


def load_summaries(generator: str | None) -> dict[str, dict[str, str]]:
    if not generator:
        return {}
    path = SUMMARY_ROOT / generator / "golden_summaries.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    records = load_json(path)
    return {rec["original_fable_id"]: rec["summaries"] for rec in records}


def build_doc_texts(
    fables: list[dict],
    doc_config: dict,
    summary_generator: str | None,
) -> list[str]:
    summaries_by_alias = load_summaries(summary_generator)
    template = doc_config["template"]
    needs_summary = doc_config.get("summary_variant") is not None

    doc_texts = []
    for fable in fables:
        values = {
            "fable": fable["text"],
            "direct_moral": "",
            "conceptual_abstract": "",
            "proverb": "",
            "cot_proverb": "",
        }
        if needs_summary:
            summaries = summaries_by_alias.get(fable["alias"])
            if summaries is None:
                raise KeyError(f"Missing {summary_generator} summaries for {fable['doc_id']} / {fable['alias']}")
            values.update({k: summaries.get(k, "") for k in values if k != "fable"})
        doc_texts.append(template.format(**values).strip())
    return doc_texts


def make_train_dataset(
    fold: dict,
    morals: list[dict],
    doc_texts: list[str],
    relevant: dict[int, set[int]],
    instruction: str,
    seed: int,
):
    from datasets import Dataset

    rows = []
    for query_idx in fold["train"]:
        for doc_idx in sorted(relevant.get(query_idx, set())):
            rows.append(
                {
                    "anchor": f"{instruction}{morals[query_idx]['text']}",
                    "positive": doc_texts[doc_idx],
                    "label": query_idx,
                }
            )
    rng = random.Random(seed + int(fold["fold"]))
    rng.shuffle(rows)
    return Dataset.from_list(rows)


def build_model(model_cfg: dict):
    from sentence_transformers import SentenceTransformer

    st_kwargs: dict = {}
    if model_cfg.get("trust_remote_code"):
        st_kwargs["trust_remote_code"] = True
    if model_cfg.get("model_kwargs"):
        st_kwargs["model_kwargs"] = dict(model_cfg["model_kwargs"])

    model = SentenceTransformer(model_cfg["model_name"], **st_kwargs)
    if model_cfg.get("max_seq_length"):
        model.max_seq_length = model_cfg["max_seq_length"]

    lora_cfg = model_cfg.get("lora")
    if lora_cfg:
        from peft import LoraConfig, TaskType, get_peft_model

        model[0].auto_model = get_peft_model(
            model[0].auto_model,
            LoraConfig(
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["alpha"],
                target_modules=lora_cfg["target_modules"],
                lora_dropout=lora_cfg.get("dropout", 0.05),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            ),
        )
        model[0].auto_model.print_trainable_parameters()
    return model


def train_model(
    model,
    train_dataset,
    evaluator,
    config: dict,
    model_cfg: dict,
    checkpoint_dir: Path,
    model_dir: Path,
    run_tag: str,
    fold_idx: int,
    force: bool,
    use_wandb: bool,
):
    import shutil
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from transformers import EarlyStoppingCallback
    from transformers.trainer_utils import get_last_checkpoint

    if force and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    checkpoint_to_resume = None
    if checkpoint_dir.exists():
        last_checkpoint = get_last_checkpoint(str(checkpoint_dir))
        if last_checkpoint:
            checkpoint_to_resume = last_checkpoint
            print(f"    [resume] <- {checkpoint_to_resume}")

    steps_per_epoch = max(1, len(train_dataset) // int(model_cfg["batch_size"]))
    monitor_metric = config.get("early_stopping_metric", "ndcg@10")
    best_metric_key = evaluator_metric_key(fold_idx, monitor_metric)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=int(config["epochs"]),
        per_device_train_batch_size=int(model_cfg["batch_size"]),
        gradient_accumulation_steps=int(model_cfg.get("gradient_accumulation_steps", 1)),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
        learning_rate=float(model_cfg["learning_rate"]),
        seed=int(config["seed"]),
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model=best_metric_key,
        greater_is_better=True,
        dataloader_pin_memory=False,
        logging_steps=max(1, steps_per_epoch // 2),
        report_to="wandb" if (use_wandb and wandb.run is not None) else "none",
    )

    callbacks = []
    if config.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(config["early_stopping_patience"])))

    best_adapter_cb = None
    if model_cfg.get("lora"):
        best_adapter_cb = BestAdapterCallback(model[0].auto_model, best_metric_key)
        callbacks.append(best_adapter_cb)

    import os

    if os.getenv("TG_BOT_TOKEN") and os.getenv("TG_CHAT_ID"):
        from finetuning.lib.notify import TelegramCallback

        callbacks.append(TelegramCallback(label=f"ft_11/{run_tag}/fold_{fold_idx}"))

    loss = InfoNCELoss(model, temperature=float(config["temperature"]))
    SentenceTransformerTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        evaluator=evaluator,
        loss=loss,
        callbacks=callbacks or None,
    ).train(resume_from_checkpoint=checkpoint_to_resume)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if best_adapter_cb is not None:
        best_adapter_cb.restore(model[0].auto_model)
    if model_cfg.get("lora"):
        model[0].auto_model = model[0].auto_model.merge_and_unload()

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))
    print(f"    [saved] -> {model_dir}")
    return model


def evaluate_and_rank(
    model,
    query_texts: list[str],
    doc_texts: list[str],
    relevant: dict[int, set[int]],
    query_ids: list[str],
    doc_ids: list[str],
    ks: list[int],
    rankings_path: Path,
) -> dict:
    query_embeddings = model.encode(
        query_texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    doc_embeddings = model.encode(
        doc_texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    score_matrix = np.matmul(query_embeddings, doc_embeddings.T)
    metrics = compute_multilabel_metrics_from_matrix(score_matrix, relevant, ks=tuple(ks))

    rankings = np.argsort(-score_matrix, axis=1)
    rows = []
    for local_idx, ranked in enumerate(rankings):
        rows.append(
            {
                "query_id": query_ids[local_idx],
                "ranked_fable_ids": [doc_ids[int(i)] for i in ranked.tolist()],
                "scores": [round(float(score_matrix[local_idx, int(i)]), 6) for i in ranked.tolist()],
            }
        )
    rankings_path.parent.mkdir(parents=True, exist_ok=True)
    rankings_path.write_text(json.dumps(rows, indent=2))
    return metrics


def run_fold(
    fold: dict,
    morals: list[dict],
    fables: list[dict],
    doc_texts: list[str],
    relevant_all: dict[int, set[int]],
    config: dict,
    model_alias: str,
    model_cfg: dict,
    doc_config_name: str,
    summary_generator: str | None,
    force: bool,
    use_wandb: bool,
) -> dict:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    fold_idx = int(fold["fold"])
    instruction = model_cfg.get("query_instruction", "")
    train_dataset = make_train_dataset(
        fold,
        morals,
        doc_texts,
        relevant_all,
        instruction,
        seed=int(config["seed"]),
    )

    test_indices = list(fold["test"])
    query_ids = [morals[i]["doc_id"] for i in test_indices]
    query_texts = [f"{instruction}{morals[i]['text']}" for i in test_indices]
    doc_ids = [f["doc_id"] for f in fables]
    local_relevant = {j: set(relevant_all[i]) for j, i in enumerate(test_indices)}

    print(
        f"\n  Fold {fold_idx + 1}/{config['n_folds']} "
        f"train_queries={len(fold['train'])} train_pairs={len(train_dataset)} "
        f"test_queries={len(test_indices)} doc_config={doc_config_name}"
    )

    run_tag = "__".join([model_alias, doc_config_name, summary_generator or "none", f"fold_{fold_idx}"])
    model_root = Path(model_cfg["model_output_dir"])
    model_dir = model_root / doc_config_name / (summary_generator or "none") / f"fold_{fold_idx}"
    checkpoint_dir = CACHE_DIR / "checkpoints" / model_alias / doc_config_name / (summary_generator or "none") / f"fold_{fold_idx}"
    rankings_path = RANKINGS_DIR / f"{run_tag}.json"

    if model_dir.exists() and not force:
        print(f"    [cache hit] Loading model <- {model_dir}")
        model = SentenceTransformer(str(model_dir))
    else:
        evaluator = InformationRetrievalEvaluator(
            queries={str(i): query_texts[i] for i in range(len(query_texts))},
            corpus={str(i): doc_texts[i] for i in range(len(doc_texts))},
            relevant_docs={str(i): {str(j) for j in local_relevant[i]} for i in local_relevant},
            mrr_at_k=[10],
            ndcg_at_k=[10],
            accuracy_at_k=[1, 5, 10],
            name=f"fold_{fold_idx}",
        )

        if use_wandb:
            wandb.init(
                project=config["wandb"]["project"],
                name=run_tag,
                group=f"ft_11_clustered/{model_alias}/{doc_config_name}",
                tags=["ft_11_clustered", model_alias, doc_config_name, summary_generator or "none", f"fold_{fold_idx}"],
                config={
                    "model": model_alias,
                    "doc_config": doc_config_name,
                    "summary_generator": summary_generator,
                    "fold": fold_idx,
                    "model_cfg": model_cfg,
                    "common": {k: v for k, v in config.items() if k not in ("models", "doc_configs", "wandb")},
                },
            )

        model = build_model(model_cfg)
        model = train_model(
            model,
            train_dataset,
            evaluator,
            config,
            model_cfg,
            checkpoint_dir,
            model_dir,
            run_tag,
            fold_idx,
            force,
            use_wandb,
        )

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
    print(
        f"  MRR@10={metrics['MRR']:.4f} MAP@10={metrics['MAP']:.4f} "
        f"Hit@10={metrics.get('Hit@10', 0.0):.4f} Recall@100={metrics.get('Recall@100', 0.0):.4f}"
    )

    if use_wandb and wandb.run is not None:
        wandb.log({"fold": fold_idx, **{f"eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}})
        wandb.finish()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold": fold_idx,
        "n_train_pairs": len(train_dataset),
        "n_train_queries": len(fold["train"]),
        "n_test_queries": len(test_indices),
        "rankings_path": str(rankings_path),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ft_11 clustered MORABLES fine-tuning")
    parser.add_argument("--model", default="sfr", help="Model alias from config.yaml")
    parser.add_argument("--models", nargs="+", help="Model aliases to run, or 'all'")
    parser.add_argument("--doc_config", help="Document config from config.yaml")
    parser.add_argument("--summary_generator", help="Summary source directory, e.g. gemini")
    parser.add_argument("--fold", type=int, choices=range(5), metavar="0-4")
    parser.add_argument("--batch_size", type=int, help="Temporary override for per-device train batch size")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue queued folds/models after an error")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    seed = int(config["seed"])
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not FOLDS_PATH.exists():
        raise FileNotFoundError(f"Missing folds: {FOLDS_PATH}. Run prepare_data.py first.")

    if args.models:
        if "all" in args.models:
            model_aliases = list(config["models"])
        else:
            model_aliases = args.models
    else:
        model_aliases = [args.model]
    unknown_models = [alias for alias in model_aliases if alias not in config["models"]]
    if unknown_models:
        raise KeyError(f"Unknown model aliases {unknown_models!r}. Options: {sorted(config['models'])}")

    doc_config_name = args.doc_config or config["default_doc_config"]
    if doc_config_name not in config["doc_configs"]:
        raise KeyError(f"Unknown doc_config {doc_config_name!r}. Options: {sorted(config['doc_configs'])}")

    doc_config = config["doc_configs"][doc_config_name]
    summary_generator = args.summary_generator or config.get("default_summary_generator")
    if doc_config.get("summary_variant") is None:
        summary_generator = None

    morals, fables, qrels = load_clustered()
    query_ids = [m["doc_id"] for m in morals]
    doc_ids = [f["doc_id"] for f in fables]
    relevant_all = load_qrels(qrels, query_ids, doc_ids)
    doc_texts = build_doc_texts(fables, doc_config, summary_generator)
    folds = load_json(FOLDS_PATH)
    fold_indices = [args.fold] if args.fold is not None else list(range(len(folds)))
    use_wandb = config.get("wandb", {}).get("enabled", False) and not args.no_wandb

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    queue_errors = []

    for model_alias in model_aliases:
        model_cfg = dict(config["models"][model_alias])
        if args.batch_size is not None:
            model_cfg["batch_size"] = args.batch_size

        notify.send(
            f"ft_11_clustered starting\n"
            f"model: {model_alias} ({model_cfg['model_name']})\n"
            f"doc_config: {doc_config_name}  generator: {summary_generator or 'none'}\n"
            f"folds: {fold_indices}  batch_size: {model_cfg['batch_size']}  seed: {seed}"
        )

        print(
            f"\n[ft_11_clustered] model={model_alias} doc_config={doc_config_name} "
            f"generator={summary_generator or 'none'} folds={fold_indices} "
            f"queries={len(morals)} fables={len(fables)} qrels={len(qrels)} seed={seed}"
        )

        fold_results = []
        model_errors = []
        for i in fold_indices:
            try:
                fold_results.append(
                    run_fold(
                        folds[i],
                        morals,
                        fables,
                        doc_texts,
                        relevant_all,
                        config,
                        model_alias,
                        model_cfg,
                        doc_config_name,
                        summary_generator,
                        args.force,
                        use_wandb,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - queued overnight runs must survive individual failures.
                error = {
                    "model_alias": model_alias,
                    "fold": i,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                model_errors.append(error)
                queue_errors.append(error)
                print(f"  [error] model={model_alias} fold={i}: {type(exc).__name__}: {exc}")
                notify.send(
                    f"ft_11_clustered fold failed\n"
                    f"model: {model_alias}  fold: {i}\n"
                    f"{type(exc).__name__}: {exc}"
                )
                if not args.continue_on_error:
                    raise

        metric_names = sorted(fold_results[0]["metrics"].keys()) if fold_results else []
        mean_metrics = {
            metric: float(np.mean([r["metrics"][metric] for r in fold_results]))
            for metric in metric_names
            if isinstance(fold_results[0]["metrics"][metric], (int, float))
        }
        std_metrics = {
            metric: float(np.std([r["metrics"][metric] for r in fold_results]))
            for metric in metric_names
            if isinstance(fold_results[0]["metrics"][metric], (int, float))
        }

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = f"fold{args.fold}" if args.fold is not None else "all_folds"
        out = RESULTS_DIR / f"{ts}_{model_alias}_{doc_config_name}_{summary_generator or 'none'}_{suffix}.json"
        out.write_text(
            json.dumps(
                {
                    "experiment": "ft_11_clustered",
                    "seed": seed,
                    "early_stopping_metric": config.get("early_stopping_metric", "ndcg@10"),
                    "continue_on_error": args.continue_on_error,
                    "model_alias": model_alias,
                    "model_config": model_cfg,
                    "doc_config": doc_config_name,
                    "doc_config_details": doc_config,
                    "summary_generator": summary_generator,
                    "folds_run": fold_indices,
                    "completed_folds": [r["fold"] for r in fold_results],
                    "errors": model_errors,
                    "mean_metrics": mean_metrics,
                    "std_metrics": std_metrics,
                    "fold_results": fold_results,
                },
                indent=2,
            )
        )
        print(f"  Results -> {out}")

        notify.send(
            f"ft_11_clustered done\n"
            f"model: {model_alias}  doc_config: {doc_config_name}\n"
            f"completed: {len(fold_results)}/{len(fold_indices)}  errors: {len(model_errors)}\n"
            f"MRR: {mean_metrics.get('MRR', 0.0):.4f}  MAP: {mean_metrics.get('MAP', 0.0):.4f}"
        )

    if queue_errors:
        print(f"\n[queue] completed with {len(queue_errors)} error(s)")
        if not args.continue_on_error:
            raise RuntimeError(f"Queue completed with {len(queue_errors)} error(s)")


if __name__ == "__main__":
    main()
