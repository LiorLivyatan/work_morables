"""
ft_10 Option A — Zero-Shot Transfer: fine-tune each model on IdioLink, eval on MORABLES.

Each of the 10 retrieval models is fine-tuned on the 440 IdioLink training triplets
using MultipleNegativesRankingLoss with one explicit hard negative per sample. Early
stopping is evaluated on the IdioLink validation set (200 queries, 1000-doc corpus).
After training, the model is evaluated on all 12 MORABLES corpus configs and results
are written to option_a_results.csv.

No MORABLES training data is used — this is a pure transfer experiment.

Usage
-----
    # Train + eval all 10 models (sequential, subprocess per model for GPU release):
    ./run.sh finetuning/ft_10_idiolink_transfer/option_a_zero_shot/train.py --remote --gpu 2

    # Single model:
    ./run.sh finetuning/ft_10_idiolink_transfer/option_a_zero_shot/train.py \\
        --model Linq-Embed-Mistral --remote --gpu 2

    # Force re-train even if saved model exists:
    ./run.sh finetuning/ft_10_idiolink_transfer/option_a_zero_shot/train.py \\
        --model Linq-Embed-Mistral --force --remote --gpu 2
"""
import argparse
import csv
import gc
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import yaml

EXP_DIR     = Path(__file__).parent
ROOT        = EXP_DIR.parent.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify
from finetuning.lib.eval import evaluate

CONFIG_PATH  = EXP_DIR / "config.yaml"
RESULTS_DIR  = EXP_DIR / "results"
RESULTS_CSV  = RESULTS_DIR / "option_a_results.csv"
MODELS_DIR   = RESULTS_DIR / "models"

GEMINI_SUMMARIES_PATH = (
    ROOT / "experiments/07_sota_summarization_oracle"
    / "results/generation_runs/full_709/golden_summaries.json"
)

CSV_FIELDNAMES = [
    "model_alias", "model_hf_id",
    "corpus_config", "summary_generator", "summary_variant", "corpus_type",
    "zero_shot_MRR@10", "idiolink_ft_MRR@10", "delta_MRR",
    "zero_shot_R@1",    "idiolink_ft_R@1",
    "zero_shot_R@5",    "idiolink_ft_R@5",
    "zero_shot_NDCG@10","idiolink_ft_NDCG@10",
]

# Corpus configs: (label, summary_generator, summary_variant, corpus_type)
CORPUS_CONFIGS = [
    ("raw",                                            "none",       "—",                     "raw"),
    ("gemini/cot_proverb/fable+summary",               "gemini",     "cot_proverb",           "fable_summary"),
    ("gemini/direct_moral/fable+summary",              "gemini",     "direct_moral",          "fable_summary"),
    ("gemini/conceptual_abstract/fable+summary",       "gemini",     "conceptual_abstract",   "fable_summary"),
    ("gemini/cot_proverb/summary_only",                "gemini",     "cot_proverb",           "summary_only"),
    ("gemini/cot_proverb/summary+fable",               "gemini",     "cot_proverb",           "summary+fable"),
    ("gemma4-E2B/direct_moral/fable+summary",          "gemma4-E2B", "direct_moral",          "fable_summary"),
    ("gemma4-E2B/direct_moral/summary_only",           "gemma4-E2B", "direct_moral",          "summary_only"),
    ("gemma4-E4B/thinking_direct_moral/fable+summary", "gemma4-E4B", "thinking_direct_moral", "fable_summary"),
    ("gemma4-E4B/thinking_direct_moral/summary_only",  "gemma4-E4B", "thinking_direct_moral", "summary_only"),
    ("gemma4-31B/conceptual_abstract/fable+summary",   "gemma4-31B", "conceptual_abstract",   "fable_summary"),
    ("gemma4-31B/conceptual_abstract/summary_only",    "gemma4-31B", "conceptual_abstract",   "summary_only"),
]


# ── IdioLink data loading ──────────────────────────────────────────────────────

def load_idiolink_train(idiolink_dir: Path) -> tuple[list, list, list]:
    """Returns (anchors, positives, hard_negatives) from train_triplets.jsonl."""
    anchors, positives, hard_negatives = [], [], []
    with open(idiolink_dir / "train_triplets.jsonl") as f:
        for line in f:
            t = json.loads(line)
            anchors.append(t["query"])
            positives.append(t["positive"])
            hard_negatives.append(t["negatives"][0])  # first = same-PIE hard negative
    return anchors, positives, hard_negatives


def build_idiolink_val_evaluator(idiolink_dir: Path, name: str = "idiolink_val"):
    """Build an InformationRetrievalEvaluator for the IdioLink validation set."""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    with open(idiolink_dir / "val_indexes.json") as f:
        index_entries = json.load(f)
    corpus = {e["id"]: e["sentence"] for e in index_entries}
    # index by sentence text for fast lookup
    text_to_id = {e["sentence"]: e["id"] for e in index_entries}

    queries, relevant_docs = {}, {}
    with open(idiolink_dir / "val_triplets.jsonl") as f:
        for i, line in enumerate(f):
            t = json.loads(line)
            qid  = f"q_{i}"
            pos_text = t["positive"]
            doc_id = text_to_id.get(pos_text)
            if doc_id is None:
                continue  # positive not in index (shouldn't happen)
            queries[qid]       = t["query"]
            relevant_docs[qid] = {doc_id}

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        ndcg_at_k=[10],
        mrr_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        name=name,
    )


# ── Model building ─────────────────────────────────────────────────────────────

def build_model(model_cfg: dict):
    from sentence_transformers import SentenceTransformer
    hf_id       = model_cfg["hf_id"]
    model_kwargs = model_cfg.get("model_kwargs") or {}
    if "torch_dtype" in model_kwargs:
        model_kwargs = dict(model_kwargs)
        dtype = model_kwargs.pop("torch_dtype")
        model_kwargs["torch_dtype"] = getattr(torch, dtype, dtype) if isinstance(dtype, str) else dtype

    trust_remote = model_kwargs.pop("trust_remote_code", False)
    st_kwargs = {"model_kwargs": model_kwargs} if model_kwargs else {}
    if trust_remote:
        st_kwargs["trust_remote_code"] = True

    model = SentenceTransformer(hf_id, **st_kwargs)
    model.max_seq_length = model_cfg.get("max_seq_length", 512)

    if model_cfg.get("use_lora"):
        from peft import LoraConfig, TaskType, get_peft_model
        lora_cfg = model_cfg["lora"]
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


# ── Training ───────────────────────────────────────────────────────────────────

def train_on_idiolink(
    model_alias:  str,
    model_cfg:    dict,
    global_cfg:   dict,
    idiolink_dir: Path,
    checkpoint_dir: Path,
    use_wandb:    bool,
    force:        bool,
) -> Path:
    """Fine-tune a model on IdioLink. Returns path to saved model."""
    import wandb
    from datasets import Dataset
    from sentence_transformers import losses
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from transformers import EarlyStoppingCallback
    from transformers.trainer_utils import get_last_checkpoint

    # Determine output path
    if model_cfg.get("model_output_dir"):
        model_save_dir = Path(model_cfg["model_output_dir"])
    else:
        model_save_dir = MODELS_DIR / model_alias

    if model_save_dir.exists() and not force:
        print(f"    [cache hit] Saved model found ← {model_save_dir}")
        return model_save_dir

    anchors, positives, hard_negatives = load_idiolink_train(idiolink_dir)
    print(f"    IdioLink train: {len(anchors)} triplets")

    val_evaluator = build_idiolink_val_evaluator(idiolink_dir, name=f"{model_alias}_val")

    if force and checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)

    checkpoint_to_resume = None
    if checkpoint_dir.exists():
        last = get_last_checkpoint(str(checkpoint_dir))
        if last:
            checkpoint_to_resume = last
            print(f"    [resume] ← {checkpoint_to_resume}")

    model = build_model(model_cfg)

    dataset = Dataset.from_dict({
        "anchor":   anchors,
        "positive": positives,
        "negative": hard_negatives,
    })

    loss = losses.MultipleNegativesRankingLoss(model)

    if use_wandb:
        wandb.init(
            project=global_cfg["wandb"]["project"],
            name=f"ft_10a_{model_alias}",
            group="ft_10_option_a",
            tags=["ft_10", "option_a", model_alias],
            config={"model": model_cfg["hf_id"], **{k: v for k, v in global_cfg.items() if k not in ("models", "wandb")}},
        )

    batch_size  = model_cfg["batch_size"]
    grad_accum  = model_cfg.get("gradient_accumulation_steps", 1)
    steps_per_ep = max(1, len(anchors) // batch_size)

    metric_key = f"{model_alias}_val_cosine_ndcg@10"

    callbacks = [EarlyStoppingCallback(early_stopping_patience=global_cfg["early_stopping_patience"])]
    if os.getenv("TG_BOT_TOKEN") and os.getenv("TG_CHAT_ID"):
        from finetuning.lib.notify import TelegramCallback
        callbacks.append(TelegramCallback(label=f"ft_10a/{model_alias}"))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=global_cfg["epochs"],
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
        learning_rate=float(model_cfg["learning_rate"]),
        warmup_steps=global_cfg.get("warmup_steps", 100),
        seed=global_cfg.get("seed", 42),
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_key,
        greater_is_better=True,
        dataloader_pin_memory=False,
        logging_steps=max(1, steps_per_ep // 2),
        report_to="wandb" if (use_wandb and wandb.run is not None) else "none",
    )

    SentenceTransformerTrainer(
        model=model, args=trainer_args,
        train_dataset=dataset, evaluator=val_evaluator,
        loss=loss, callbacks=callbacks,
    ).train(resume_from_checkpoint=checkpoint_to_resume)

    # Merge LoRA into base weights before saving
    if model_cfg.get("use_lora"):
        model[0].auto_model = model[0].auto_model.merge_and_unload()

    model_save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_save_dir))
    print(f"    [saved] → {model_save_dir}")

    if use_wandb and wandb.run is not None:
        wandb.finish()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model_save_dir


# ── MORABLES corpus loading ────────────────────────────────────────────────────

def load_gemini_summaries() -> dict[str, dict[str, str]]:
    """Returns {fable_alias: {variant: summary_text}} for Gemini summaries."""
    with open(GEMINI_SUMMARIES_PATH) as f:
        data = json.load(f)
    return {e["original_fable_id"]: e["summaries"] for e in data}


def load_gemma4_summaries(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    """Returns {fable_alias: {model_alias: {variant: summary_text}}} for Gemma4."""
    with open(path) as f:
        data = json.load(f)
    result = {}
    for entry in data:
        alias = entry.get("fable_alias") or entry.get("original_fable_id", "")
        result[alias] = {
            model: {v: info["text"] for v, info in variants.items()}
            for model, variants in entry.get("summaries", {}).items()
        }
    return result


def build_morables_corpus(
    corpus_config: tuple,
    fables: list[dict],
    gemini_sums: dict,
    gemma4_sums: dict | None,
) -> list[str] | None:
    """
    Build doc_texts for one corpus config. Returns None if required summaries
    are unavailable (e.g. Gemma4 JSON not pulled yet).
    """
    label, gen, variant, corpus_type = corpus_config

    if gen == "none":
        return [f["text"] for f in fables]

    if gen == "gemini":
        docs = []
        for f in fables:
            summary = gemini_sums.get(f["alias"], {}).get(variant, "")
            if corpus_type == "fable_summary":
                docs.append(f"{f['text']}\n\nMoral summary: {summary}")
            elif corpus_type == "summary_only":
                docs.append(summary)
            elif corpus_type == "summary+fable":
                docs.append(f"Moral summary: {summary}\n\n{f['text']}")
        return docs

    if gen.startswith("gemma4-"):
        if gemma4_sums is None:
            return None  # summaries not available locally
        model_alias = gen  # e.g. "gemma4-E2B"
        docs = []
        for f in fables:
            alias_sums = gemma4_sums.get(f["alias"], {})
            model_sums = alias_sums.get(model_alias, {})
            summary    = model_sums.get(variant, "")
            if corpus_type == "fable_summary":
                docs.append(f"{f['text']}\n\nMoral summary: {summary}")
            elif corpus_type == "summary_only":
                docs.append(summary)
        return docs

    raise ValueError(f"Unknown summary generator: {gen!r}")


# ── MORABLES evaluation ────────────────────────────────────────────────────────

def evaluate_morables_all_configs(
    model_alias: str,
    model_save_dir: Path,
    model_cfg: dict,
    gemma4_sums: dict | None,
    force: bool,
) -> dict[str, dict]:
    """Load saved model, evaluate on all 12 corpus configs. Returns {label: metrics}."""
    from sentence_transformers import SentenceTransformer
    from lib.data import load_fables, load_morals, load_qrels_moral_to_fable

    model_kwargs = {}
    raw_kwargs = model_cfg.get("model_kwargs") or {}
    if "torch_dtype" in raw_kwargs:
        dtype = raw_kwargs["torch_dtype"]
        model_kwargs["torch_dtype"] = getattr(torch, dtype, dtype) if isinstance(dtype, str) else dtype
    if raw_kwargs.get("trust_remote_code"):
        model_kwargs["trust_remote_code"] = True

    print(f"\n  Loading fine-tuned model from {model_save_dir} …")
    model = SentenceTransformer(str(model_save_dir), model_kwargs=model_kwargs or None)

    fables       = load_fables()
    morals       = load_morals()
    qrels        = load_qrels_moral_to_fable()
    moral_indices = sorted(qrels.keys())
    moral_texts  = [morals[i]["text"] for i in moral_indices]
    ground_truth = {i: qrels[idx] for i, idx in enumerate(moral_indices)}

    gemini_sums = load_gemini_summaries()

    results = {}
    emb_cache_base = EXP_DIR / "cache" / "embeddings" / model_alias

    for cfg in CORPUS_CONFIGS:
        label, gen, variant, corpus_type = cfg
        print(f"    Evaluating: {label} …")

        doc_texts = build_morables_corpus(cfg, fables, gemini_sums, gemma4_sums)
        if doc_texts is None:
            print(f"    [skip] Gemma4 summaries not available locally")
            results[label] = None
            continue

        # Separate caches: query embeddings depend on model only; doc embeddings
        # depend on model + corpus config.
        query_cache = emb_cache_base / "queries"
        doc_cache   = emb_cache_base / label.replace("/", "_")

        metrics = evaluate(
            model, moral_texts, doc_texts, ground_truth,
            cache_dir=query_cache,
            doc_cache_dir=doc_cache,
            force=force,
        )
        results[label] = metrics
        print(
            f"      MRR={metrics['MRR']:.4f}  "
            f"R@1={metrics['Recall@1']:.4f}  "
            f"R@5={metrics['Recall@5']:.4f}  "
            f"NDCG@10={metrics['NDCG@10']:.4f}"
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ── CSV update ─────────────────────────────────────────────────────────────────

def update_csv(model_alias: str, model_hf_id: str, results: dict[str, dict | None]) -> None:
    """Write idiolink_ft_* columns for this model into the results CSV."""
    with open(RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if row["model_alias"] != model_alias:
            continue
        label   = row["corpus_config"]
        metrics = results.get(label)
        if metrics is None:
            row["idiolink_ft_MRR@10"]  = "n/a (summaries not pulled)"
            row["idiolink_ft_R@1"]     = ""
            row["idiolink_ft_R@5"]     = ""
            row["idiolink_ft_NDCG@10"] = ""
            row["delta_MRR"]           = ""
            continue
        ft_mrr = metrics["MRR"]
        row["idiolink_ft_MRR@10"]  = f"{ft_mrr:.4f}"
        row["idiolink_ft_R@1"]     = f"{metrics['Recall@1']:.4f}"
        row["idiolink_ft_R@5"]     = f"{metrics['Recall@5']:.4f}"
        row["idiolink_ft_NDCG@10"] = f"{metrics['NDCG@10']:.4f}"
        zs = row.get("zero_shot_MRR@10", "")
        if zs:
            row["delta_MRR"] = f"{ft_mrr - float(zs):+.4f}"

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV updated → {RESULTS_CSV}")


# ── Single-model entry point ───────────────────────────────────────────────────

def run_one_model(model_alias: str, config: dict, force: bool, use_wandb: bool) -> None:
    model_cfg    = config["models"][model_alias]
    idiolink_dir = ROOT / config["idiolink_data_dir"]

    gemma4_sums = None
    gemma4_path = config.get("gemma4_summaries_path")
    if gemma4_path:
        p = ROOT / gemma4_path
        if p.exists():
            gemma4_sums = load_gemma4_summaries(p)
        else:
            print(f"  [warn] gemma4_summaries_path set but not found: {p}")

    checkpoint_dir = EXP_DIR / "cache" / "checkpoints" / model_alias

    notify.send(
        f"🚀 ft_10a starting: {model_alias}\n"
        f"model: {model_cfg['hf_id']}\n"
        f"lora: {model_cfg.get('use_lora', False)}  "
        f"bs: {model_cfg['batch_size']}×{model_cfg.get('gradient_accumulation_steps',1)}"
    )

    model_save_dir = train_on_idiolink(
        model_alias, model_cfg, config,
        idiolink_dir, checkpoint_dir, use_wandb, force,
    )

    results = evaluate_morables_all_configs(
        model_alias, model_save_dir, model_cfg, gemma4_sums, force,
    )
    update_csv(model_alias, model_cfg["hf_id"], results)

    evaluated = [k for k, v in results.items() if v is not None]
    skipped   = [k for k, v in results.items() if v is None]
    mrrs      = [f"{results[k]['MRR']:.4f}" for k in evaluated]

    notify.send(
        f"✅ ft_10a done: {model_alias}\n"
        f"configs evaluated: {len(evaluated)}/{len(CORPUS_CONFIGS)}\n"
        f"MRRs: {', '.join(mrrs)}"
        + (f"\nskipped (no Gemma4 summaries): {len(skipped)}" if skipped else "")
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ft_10 Option A: IdioLink → MORABLES transfer")
    parser.add_argument("--model",    type=str, default=None, help="Model alias from config (default: all)")
    parser.add_argument("--force",    action="store_true",    help="Re-train and re-embed even if cached")
    parser.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    use_wandb = config.get("wandb", {}).get("enabled", False) and not args.no_wandb

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_aliases = list(config["models"].keys())
    if args.model:
        if args.model not in config["models"]:
            print(f"Unknown model: {args.model!r}. Available: {model_aliases}")
            sys.exit(1)
        model_aliases = [args.model]

    if len(model_aliases) == 1:
        run_one_model(model_aliases[0], config, args.force, use_wandb)
        return

    # Multiple models: run each in a fresh subprocess so GPU memory is fully
    # released between models (same pattern as ft_09 per-fold subprocesses).
    notify.send(
        f"🚀 ft_10 Option A starting\n"
        f"models: {', '.join(model_aliases)}\n"
        f"corpus configs: {len(CORPUS_CONFIGS)}"
    )

    extra = ["--force"] if args.force else []
    if not use_wandb:
        extra.append("--no-wandb")

    for alias in model_aliases:
        print(f"\n{'='*60}")
        print(f"  [subprocess] {alias}")
        print(f"{'='*60}")
        subprocess.run(
            [sys.executable, __file__, "--model", alias] + extra,
            check=True,
        )

    notify.send(f"✅ ft_10 Option A complete — all {len(model_aliases)} models done")


if __name__ == "__main__":
    main()
