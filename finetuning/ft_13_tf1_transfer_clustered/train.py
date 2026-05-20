"""ft_13 — TF1-synthetic transfer evaluated on clustered MORABLES.

See docs/superpowers/specs/2026-05-20-ft-13-tf1-transfer-clustered-design.md
for the full design. This file mirrors ft_12_storal_transfer_clustered/train.py
with surgical replacements in the data layer (TF1 source instead of STORAL,
size-as-morals-and-fables-per-moral instead of size-as-row-count, label=moral_id).
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
import wandb
import yaml

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from transformers import TrainerCallback, set_seed

from finetuning.lib import notify
from finetuning.lib.losses import InfoNCELoss
from lib.retrieval_utils import compute_multilabel_metrics_from_matrix

EXPERIMENT_NAME = "ft_13_tf1_transfer_clustered"

CACHE_DIR = EXP_DIR / "cache"
RESULTS_DIR = EXP_DIR / "results"
RANKINGS_DIR = RESULTS_DIR / "rankings"
COMPREHENSIVE_CSV = RESULTS_DIR / "ft13_comprehensive_results.csv"
CONFIG_PATH = EXP_DIR / "config.yaml"

# Clustered MORABLES (read-only target)
MORALS_PATH = ROOT / "data" / "clustered" / "morals_unique_corpus.json"
FABLES_PATH = ROOT / "data" / "clustered" / "fables_corpus.json"
QRELS_PATH = ROOT / "data" / "clustered" / "qrels_moral_to_fable_clustered.json"

SUMMARY_ROOT = ROOT / "experiments" / "20_final_zero_shot" / "summary_inputs"

CSV_FIELDNAMES = [
    "timestamp",
    "experiment",
    "model_alias",
    "model_id",
    "size",
    "n_tf1_rows_selected",
    "n_tf1_morals_selected",
    "selection_strategy",
    "train_rows",
    "validation_rows",
    "unique_normalized_morals",
    "exact_duplicate_moral_groups",
    "exact_duplicate_moral_rows",
    "eval_doc_config",
    "summary_generator",
    "corpus_template",
    "rankings_path",
    "MAP@10",
    "MRR@10",
    "NDCG@10",
    "Recall@5",
    "Recall@10",
    "Recall@15",
    "Recall@50",
    "Recall@100",
    "Recall@200",
    "Recall@300",
    "Hit@1",
    "Hit@5",
    "Hit@10",
    "Hit@100",
    "Mean_Rank",
    "Median_Rank",
    "n_queries",
    "seed",
    "early_stopping_metric",
    "model_dir",
    "error",
]


def _subsample_morals(pairs: list[dict], size_cfg: dict, seed: int) -> list[dict]:
    """Sample n_morals distinct morals (or keep all), then take the first
    n_fables_per_moral fables for each.

    Sub-sampling is deterministic for a given (pairs, size_cfg, seed). Because
    the low-IoU corpus stores fables per moral in ascending-IoU order, taking
    the first K is equivalent to taking the K lowest-IoU fables for that moral.
    """
    by_moral: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_moral[p["moral_id"]].append(p)
    moral_ids = sorted(by_moral.keys())

    n_morals = size_cfg.get("n_morals")
    if n_morals is not None and n_morals < len(moral_ids):
        rng = random.Random(seed)
        moral_ids = rng.sample(moral_ids, n_morals)

    n_fables = size_cfg["n_fables_per_moral"]
    return [p for mid in moral_ids for p in by_moral[mid][:n_fables]]


def split_tf1_groups(
    pairs: list[dict], seed: int, validation_ratio: float
) -> tuple[list[dict], list[dict]]:
    """Group-aware train/val split: morals (not rows) are sampled into the
    validation set; all fables of a given moral go together. Prevents within-
    moral leakage.
    """
    by_moral: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_moral[p["moral_id"]].append(p)
    moral_ids = sorted(by_moral.keys())
    if len(moral_ids) < 2:
        raise ValueError(
            f"split_tf1_groups requires at least 2 distinct morals, got {len(moral_ids)}"
        )
    rng = random.Random(seed)
    rng.shuffle(moral_ids)
    n_val = max(1, round(len(moral_ids) * validation_ratio))
    train_rows = [p for mid in moral_ids[n_val:] for p in by_moral[mid]]
    val_rows = [p for mid in moral_ids[:n_val] for p in by_moral[mid]]
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def make_tf1_dataset(pairs: list[dict], instruction: str):
    """Build the 3-column training Dataset: anchor=instruction+moral,
    positive=fable, label=integer-per-moral (for InfoNCELoss multi-positive
    masking). In exact-cluster mode, label = moral_id integer index.
    """
    from datasets import Dataset

    moral_to_label: dict[str, int] = {}
    labels: list[int] = []
    for p in pairs:
        if p["moral_id"] not in moral_to_label:
            moral_to_label[p["moral_id"]] = len(moral_to_label)
        labels.append(moral_to_label[p["moral_id"]])

    return Dataset.from_dict({
        "anchor":   [f"{instruction}{p['moral']}" for p in pairs],
        "positive": [p["story"] for p in pairs],
        "label":    labels,
    })


def load_tf1_synthetic_exact(
    size_cfg: dict, seed: int, source_dir: Path
) -> tuple[list[dict], dict]:
    """Load TF1 exact-cluster pairs from source_dir/processed/, then sub-sample
    per the size config.

    Returns (pairs, stats):
        pairs: list of dicts with keys {moral, story, moral_id, fable_id}
        stats: {raw_total, selected_rows, selected_morals, selection_strategy,
                size_config}
    """
    morals = json.loads((source_dir / "processed" / "morals_corpus.json").read_text())
    fables = json.loads((source_dir / "processed" / "fables_corpus.json").read_text())
    qrels = json.loads((source_dir / "processed" / "qrels_moral_to_fable.json").read_text())

    moral_by_id = {m["doc_id"]: m["text"] for m in morals}
    fable_by_id = {f["doc_id"]: f["text"] for f in fables}
    pairs = [
        {
            "moral": moral_by_id[q["query_id"]],
            "story": fable_by_id[q["doc_id"]],
            "moral_id": q["query_id"],
            "fable_id": q["doc_id"],
        }
        for q in qrels
    ]
    raw_total = len(pairs)
    pairs = _subsample_morals(pairs, size_cfg, seed)
    selected_morals = len({p["moral_id"] for p in pairs})

    stats = {
        "raw_total": raw_total,
        "selected_rows": len(pairs),
        "selected_morals": selected_morals,
        "selection_strategy": source_dir.name,
        "size_config": dict(size_cfg),
    }
    return pairs, stats


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


def load_json(path: Path):
    return json.loads(path.read_text())


def make_tf1_evaluator(val_pairs: list[dict], instruction: str):
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    moral_to_label: dict[str, int] = {}
    labels: list[int] = []
    for p in val_pairs:
        if p["moral_id"] not in moral_to_label:
            moral_to_label[p["moral_id"]] = len(moral_to_label)
        labels.append(moral_to_label[p["moral_id"]])

    relevant_docs: dict[str, set[str]] = defaultdict(set)
    for i, label in enumerate(labels):
        for j, other_label in enumerate(labels):
            if label == other_label:
                relevant_docs[str(i)].add(str(j))

    return InformationRetrievalEvaluator(
        queries={str(i): f"{instruction}{p['moral']}" for i, p in enumerate(val_pairs)},
        corpus={str(i): p["story"] for i, p in enumerate(val_pairs)},
        relevant_docs=dict(relevant_docs),
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        name="tf1_val",
    )


def evaluator_metric_key(metric: str) -> str:
    aliases = {
        "mrr": "mrr@10",
        "map": "map@100",
        "map@10": "map@10",
        "ndcg": "ndcg@10",
        "hit": "accuracy@10",
        "hit@10": "accuracy@10",
    }
    metric = aliases.get(metric.lower(), metric.lower())
    allowed = {
        "mrr@10",
        "ndcg@10",
        "map@10",
        "map@100",
        "recall@1",
        "recall@5",
        "recall@10",
        "accuracy@1",
        "accuracy@5",
        "accuracy@10",
    }
    if metric not in allowed:
        raise ValueError(f"Unsupported early_stopping_metric={metric!r}; supported: {sorted(allowed)}")
    return f"eval_tf1_val_cosine_{metric}"


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


def build_model(model_cfg: dict, force_cpu: bool = False):
    from sentence_transformers import SentenceTransformer

    st_kwargs: dict = {}
    if model_cfg.get("trust_remote_code"):
        st_kwargs["trust_remote_code"] = True
    if model_cfg.get("model_kwargs"):
        st_kwargs["model_kwargs"] = dict(model_cfg["model_kwargs"])

    model = SentenceTransformer(
        model_cfg["model_name"],
        device="cpu" if force_cpu else None,
        **st_kwargs,
    )
    if model_cfg.get("max_seq_length"):
        model.max_seq_length = int(model_cfg["max_seq_length"])

    lora_cfg = model_cfg.get("lora")
    if lora_cfg:
        from peft import LoraConfig, TaskType, get_peft_model

        model[0].auto_model = get_peft_model(
            model[0].auto_model,
            LoraConfig(
                r=int(lora_cfg["r"]),
                lora_alpha=int(lora_cfg["alpha"]),
                target_modules=lora_cfg["target_modules"],
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
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
    force: bool,
    use_wandb: bool,
):
    import os
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
    best_metric_key = evaluator_metric_key(config.get("early_stopping_metric", "ndcg@10"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=int(config["epochs"]),
        per_device_train_batch_size=int(model_cfg["batch_size"]),
        gradient_accumulation_steps=int(model_cfg.get("gradient_accumulation_steps", 1)),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
        learning_rate=float(model_cfg["learning_rate"]),
        seed=int(config["seed"]),
        use_cpu=bool(model_cfg.get("use_cpu", False)),
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

    if os.getenv("TG_BOT_TOKEN") and os.getenv("TG_CHAT_ID"):
        from finetuning.lib.notify import TelegramCallback

        callbacks.append(TelegramCallback(label=f"{EXPERIMENT_NAME}/{run_tag}"))

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


def evaluate_model(
    model,
    model_alias: str,
    size_name: str,
    model_cfg: dict,
    config: dict,
    eval_doc_configs: list[str],
    summary_generator: str | None,
) -> dict:
    morals, fables, qrels = load_clustered()
    query_ids = [m["doc_id"] for m in morals]
    doc_ids = [f["doc_id"] for f in fables]
    relevant = load_qrels(qrels, query_ids, doc_ids)
    instruction = model_cfg.get("query_instruction", "")
    query_texts = [f"{instruction}{m['text']}" for m in morals]

    results = {}
    for doc_config_name in eval_doc_configs:
        doc_config = config["doc_configs"][doc_config_name]
        generator = None if doc_config.get("summary_variant") is None else summary_generator
        doc_texts = build_doc_texts(fables, doc_config, generator)
        run_tag = "__".join([model_alias, size_name, doc_config_name, generator or "none"])
        rankings_path = RANKINGS_DIR / f"{run_tag}.json"
        metrics = evaluate_and_rank(
            model,
            query_texts,
            doc_texts,
            relevant,
            query_ids,
            doc_ids,
            list(config["ks"]),
            rankings_path,
        )
        results[doc_config_name] = {
            "summary_generator": generator,
            "rankings_path": str(rankings_path),
            "metrics": metrics,
        }
        print(
            f"  [{doc_config_name}/{generator or 'none'}] "
            f"MRR@10={metrics['MRR']:.4f} MAP@10={metrics['MAP']:.4f} "
            f"Hit@10={metrics.get('Hit@10', 0.0):.4f} Recall@100={metrics.get('Recall@100', 0.0):.4f}"
        )
    return results


def append_comprehensive_csv(result: dict, config: dict, error: str = "") -> None:
    COMPREHENSIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    exists = COMPREHENSIVE_CSV.exists()

    rows = []
    if error:
        rows.append(
            {
                "timestamp": timestamp,
                "experiment": EXPERIMENT_NAME,
                "model_alias": result.get("model_alias", ""),
                "model_id": result.get("model_config", {}).get("model_name", ""),
                "size": result.get("size", ""),
                "seed": config.get("seed", ""),
                "early_stopping_metric": config.get("early_stopping_metric", ""),
                "error": error,
            }
        )
    else:
        for doc_config_name, eval_result in result["eval_results"].items():
            metrics = eval_result["metrics"]
            doc_config = config["doc_configs"][doc_config_name]
            rows.append(
                {
                    "timestamp": timestamp,
                    "experiment": EXPERIMENT_NAME,
                    "model_alias": result["model_alias"],
                    "model_id": result["model_config"]["model_name"],
                    "size": result["size"],
                    "n_tf1_rows_selected": result["tf1_stats"]["selected_rows"],
                    "n_tf1_morals_selected": result["tf1_stats"]["selected_morals"],
                    "selection_strategy": result["tf1_stats"]["selection_strategy"],
                    "train_rows": result["train_rows"],
                    "validation_rows": result["validation_rows"],
                    "unique_normalized_morals": result["tf1_stats"]["selected_morals"],
                    "exact_duplicate_moral_groups": 0,
                    "exact_duplicate_moral_rows": 0,
                    "eval_doc_config": doc_config_name,
                    "summary_generator": eval_result["summary_generator"] or "none",
                    "corpus_template": doc_config["template"].replace("\n", "\\n"),
                    "rankings_path": eval_result["rankings_path"],
                    "MAP@10": metrics.get("MAP", ""),
                    "MRR@10": metrics.get("MRR", ""),
                    "NDCG@10": metrics.get("NDCG@10", ""),
                    "Recall@5": metrics.get("Recall@5", ""),
                    "Recall@10": metrics.get("Recall@10", ""),
                    "Recall@15": metrics.get("Recall@15", ""),
                    "Recall@50": metrics.get("Recall@50", ""),
                    "Recall@100": metrics.get("Recall@100", ""),
                    "Recall@200": metrics.get("Recall@200", ""),
                    "Recall@300": metrics.get("Recall@300", ""),
                    "Hit@1": metrics.get("Hit@1", ""),
                    "Hit@5": metrics.get("Hit@5", ""),
                    "Hit@10": metrics.get("Hit@10", ""),
                    "Hit@100": metrics.get("Hit@100", ""),
                    "Mean_Rank": metrics.get("Mean Rank", ""),
                    "Median_Rank": metrics.get("Median Rank", ""),
                    "n_queries": metrics.get("n_queries", ""),
                    "seed": config.get("seed", ""),
                    "early_stopping_metric": config.get("early_stopping_metric", ""),
                    "model_dir": result["model_dir"],
                    "error": "",
                }
            )

    with COMPREHENSIVE_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})


def run_one(
    model_alias: str,
    size_name: str,
    config: dict,
    force: bool,
    skip_train: bool,
    eval_doc_configs: list[str],
    summary_generator: str | None,
    use_wandb: bool,
    force_cpu: bool = False,
) -> dict:
    from sentence_transformers import SentenceTransformer

    model_cfg = dict(config["models"][model_alias])
    size_cfg = config["dataset_sizes"][size_name]
    seed = int(config["seed"])
    instruction = model_cfg.get("query_instruction", "")

    tf1_corpus_dir = ROOT / config["source"]["tf1_corpus_dir"]
    tf1_pairs, tf1_stats = load_tf1_synthetic_exact(size_cfg, seed, tf1_corpus_dir)
    train_pairs, val_pairs = split_tf1_groups(tf1_pairs, seed, float(config["validation_ratio"]))
    train_dataset = make_tf1_dataset(train_pairs, instruction)
    evaluator = make_tf1_evaluator(val_pairs, instruction)

    run_tag = f"{model_alias}_{size_name}"
    model_dir = Path(model_cfg["model_output_dir"]) / size_name
    checkpoint_dir = CACHE_DIR / "checkpoints" / model_alias / size_name

    print(
        f"\n[{EXPERIMENT_NAME}] model={model_alias} size={size_name} "
        f"selected_rows={len(tf1_pairs)} selected_morals={tf1_stats['selected_morals']} "
        f"train={len(train_pairs)} val={len(val_pairs)} seed={seed}"
    )

    if model_dir.exists() and not force:
        print(f"  [cache hit] Loading model <- {model_dir}")
        st_kwargs: dict = {}
        if model_cfg.get("trust_remote_code"):
            st_kwargs["trust_remote_code"] = True
        model = SentenceTransformer(
            str(model_dir),
            device="cpu" if force_cpu else None,
            **st_kwargs,
        )
    elif skip_train:
        raise FileNotFoundError(f"--skip_train requested but saved model does not exist: {model_dir}")
    else:
        if use_wandb:
            wandb.init(
                project=config["wandb"]["project"],
                name=run_tag,
                group=f"{EXPERIMENT_NAME}/{model_alias}",
                tags=[EXPERIMENT_NAME, model_alias, size_name],
                config={
                    "model": model_alias,
                    "size": size_name,
                    "model_cfg": model_cfg,
                    "size_cfg": size_cfg,
                    "tf1_stats": tf1_stats,
                    "common": {k: v for k, v in config.items() if k not in ("models", "doc_configs", "wandb")},
                },
            )

        model = build_model(model_cfg, force_cpu=force_cpu)
        if force_cpu:
            model_cfg["use_cpu"] = True
        model = train_model(
            model,
            train_dataset,
            evaluator,
            config,
            model_cfg,
            checkpoint_dir,
            model_dir,
            run_tag,
            force,
            use_wandb,
        )
        if use_wandb and wandb.run is not None:
            wandb.finish()

    eval_results = evaluate_model(
        model,
        model_alias,
        size_name,
        model_cfg,
        config,
        eval_doc_configs,
        summary_generator,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_alias": model_alias,
        "model_config": model_cfg,
        "size": size_name,
        "size_config": size_cfg,
        "tf1_stats": tf1_stats,
        "train_rows": len(train_pairs),
        "validation_rows": len(val_pairs),
        "model_dir": str(model_dir),
        "eval_doc_configs": eval_doc_configs,
        "summary_generator": summary_generator,
        "eval_results": eval_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ft_13 TF1 transfer -> clustered MORABLES")
    parser.add_argument("--model", default="linq", help="Model alias from config.yaml")
    parser.add_argument("--models", nargs="+", help="Model aliases to run, or 'all'")
    parser.add_argument("--size", default="s500", help="Dataset size key from config.yaml")
    parser.add_argument("--sizes", nargs="+", help="Dataset size keys to run, or 'all'")
    parser.add_argument("--eval_doc_configs", nargs="+", help="MORABLES doc configs to evaluate, or 'all'")
    parser.add_argument("--summary_generator", help="Summary source directory, e.g. gemini")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--skip_train", action="store_true", help="Load saved model and run eval only")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU execution end-to-end (model + trainer). For local "
             "validation when no GPU is available; production runs should "
             "leave this off and use --remote --gpu N.",
    )
    parser.add_argument(
        "--tf1_corpus_dir", type=Path, default=None,
        help="Override the TF1 corpus dir from config (for ablation against the "
             "random-selection corpus). Defaults to config['source']['tf1_corpus_dir'].",
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    if args.tf1_corpus_dir is not None:
        config["source"]["tf1_corpus_dir"] = str(args.tf1_corpus_dir)

    seed = int(config["seed"])
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.models:
        model_aliases = list(config["models"]) if "all" in args.models else args.models
    else:
        model_aliases = [args.model]
    unknown_models = [alias for alias in model_aliases if alias not in config["models"]]
    if unknown_models:
        raise KeyError(f"Unknown model aliases {unknown_models!r}. Options: {sorted(config['models'])}")

    if args.sizes:
        size_names = list(config["dataset_sizes"]) if "all" in args.sizes else args.sizes
    else:
        size_names = [args.size]
    unknown_sizes = [size for size in size_names if size not in config["dataset_sizes"]]
    if unknown_sizes:
        raise KeyError(f"Unknown size keys {unknown_sizes!r}. Options: {sorted(config['dataset_sizes'])}")

    if args.eval_doc_configs:
        eval_doc_configs = list(config["doc_configs"]) if "all" in args.eval_doc_configs else args.eval_doc_configs
    else:
        eval_doc_configs = list(config["default_eval_doc_configs"])
    unknown_doc_configs = [name for name in eval_doc_configs if name not in config["doc_configs"]]
    if unknown_doc_configs:
        raise KeyError(f"Unknown doc configs {unknown_doc_configs!r}. Options: {sorted(config['doc_configs'])}")

    summary_generator = args.summary_generator or config.get("default_summary_generator")
    use_wandb = config.get("wandb", {}).get("enabled", False) and not args.no_wandb

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RANKINGS_DIR.mkdir(parents=True, exist_ok=True)

    queue_errors = []
    queue_results = []
    notify.send(
        f"{EXPERIMENT_NAME} starting\n"
        f"models: {model_aliases}\n"
        f"sizes: {size_names}\n"
        f"eval_doc_configs: {eval_doc_configs}\n"
        f"summary_generator: {summary_generator}  seed: {seed}"
    )

    for model_alias in model_aliases:
        for size_name in size_names:
            try:
                result = run_one(
                    model_alias,
                    size_name,
                    config,
                    args.force,
                    args.skip_train,
                    eval_doc_configs,
                    summary_generator,
                    use_wandb,
                    force_cpu=args.cpu,
                )
                queue_results.append(result)

                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                out = RESULTS_DIR / f"{ts}_{model_alias}_{size_name}.json"
                out.write_text(
                    json.dumps(
                        {
                            "experiment": EXPERIMENT_NAME,
                            "seed": seed,
                            "early_stopping_metric": config.get("early_stopping_metric", "ndcg@10"),
                            "exact_moral_masking": bool(config.get("exact_moral_masking", True)),
                            "result": result,
                        },
                        indent=2,
                    )
                )
                print(f"  Results -> {out}")
                append_comprehensive_csv(result, config)
                best_raw = result["eval_results"].get("raw", {}).get("metrics", {})
                notify.send(
                    f"ft_13 run done\n"
                    f"model: {model_alias}  size: {size_name}\n"
                    f"MRR raw: {best_raw.get('MRR', 0.0):.4f}  MAP raw: {best_raw.get('MAP', 0.0):.4f}"
                )
            except Exception as exc:  # noqa: BLE001 - queued overnight runs must survive individual failures.
                error = {
                    "model_alias": model_alias,
                    "size": size_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                queue_errors.append(error)
                append_comprehensive_csv(
                    {
                        "model_alias": model_alias,
                        "model_config": config["models"].get(model_alias, {}),
                        "size": size_name,
                    },
                    config,
                    error=f"{type(exc).__name__}: {exc}",
                )
                print(f"  [error] model={model_alias} size={size_name}: {type(exc).__name__}: {exc}")
                notify.send(
                    f"ft_13 run failed\n"
                    f"model: {model_alias}  size: {size_name}\n"
                    f"{type(exc).__name__}: {exc}"
                )
                if not args.continue_on_error:
                    raise

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = RESULTS_DIR / f"{ts}_queue_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "seed": seed,
                "models": model_aliases,
                "sizes": size_names,
                "eval_doc_configs": eval_doc_configs,
                "summary_generator": summary_generator,
                "completed": len(queue_results),
                "errors": queue_errors,
                "result_files_written": len(queue_results),
            },
            indent=2,
        )
    )
    print(f"\n[queue] summary -> {summary_path}")
    notify.send(
        f"ft_13 queue done\n"
        f"completed: {len(queue_results)}  errors: {len(queue_errors)}\n"
        f"summary: {summary_path}"
    )

    if queue_errors and not args.continue_on_error:
        raise RuntimeError(f"Queue completed with {len(queue_errors)} error(s)")


if __name__ == "__main__":
    main()
