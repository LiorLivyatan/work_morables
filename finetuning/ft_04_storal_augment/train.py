"""
ft_04_storal_augment — InfoNCE fine-tuning with STORAL augmentation.

Identical training setup to ft_03 basic (types 1+2), but every fold's
training set includes all 1,675 clean STORAL (moral, story) pairs on top
of the 570 MORABLES pairs — ~4x more training signal per epoch.

STORAL pairs are always used for training only; evaluation is MORABLES-only
(same 5-fold split as ft_01/02/03), so results are directly comparable.

Usage
-----
    # All folds:
    ./run.sh finetuning/ft_04_storal_augment/train.py

    # Single fold for a quick test:
    ./run.sh finetuning/ft_04_storal_augment/train.py --fold 0

    # Remote GPU:
    ./run.sh finetuning/ft_04_storal_augment/train.py --remote --gpu 2
"""
import argparse
import gc
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from transformers import TrainerCallback

from finetuning.lib import notify
from finetuning.lib.data import load_pairs
from finetuning.lib.eval import evaluate
from finetuning.lib.losses import InfoNCELoss

CACHE_DIR = EXP_DIR / "cache"
RESULTS_DIR = EXP_DIR / "results"
CONFIG_PATH = EXP_DIR / "config.yaml"
SPLITS_PATH = EXP_DIR.parent / "ft_01_5fold_cv" / "cache" / "splits" / "folds.json"
STORAL_PATH = ROOT / "data/external/storal/processed/storal_pairs.json"

_EVAL_KEYS = ("MRR", "Recall@1", "Recall@5", "Recall@10")


class BestAdapterCallback(TrainerCallback):
    """
    Saves LoRA adapter weights to CPU RAM whenever a new best MRR is reached.

    Used instead of load_best_model_at_end=True, which OOMs on 24GB cards because
    it tries to load a checkpoint while the training model is still fully in VRAM.
    After trainer.train() returns, call restore(peft_model) to apply the best state.
    """

    def __init__(self, peft_model, metric_key: str):
        self.peft_model = peft_model
        self.metric_key = metric_key
        self.best_mrr = -1.0
        self.best_state: dict | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        mrr = metrics.get(self.metric_key)
        if mrr is not None and float(mrr) > self.best_mrr:
            self.best_mrr = float(mrr)
            self.best_state = {
                k: v.detach().cpu().clone()
                for k, v in self.peft_model.state_dict().items()
                if "lora" in k.lower()
            }
            print(f"    [best] MRR={mrr:.4f} — adapter state saved to CPU")

    def restore(self, peft_model) -> float:
        """Load the best adapter state back into the model. Returns best MRR."""
        if self.best_state is None:
            return self.best_mrr
        device = next(peft_model.parameters()).device
        state = peft_model.state_dict()
        for k, v in self.best_state.items():
            state[k] = v.to(device)
        peft_model.load_state_dict(state)
        print(f"    [best] Restored best adapter (MRR={self.best_mrr:.4f})")
        return self.best_mrr


def load_storal_pairs() -> list[dict]:
    if not STORAL_PATH.exists():
        raise FileNotFoundError(
            f"Processed STORAL pairs not found: {STORAL_PATH}\n"
            "Run first: ./run.sh finetuning/ft_04_storal_augment/preprocess_storal.py"
        )
    pairs = json.loads(STORAL_PATH.read_text())
    clean = [p for p in pairs if not p["is_duplicate"]]
    print(f"  STORAL: {len(pairs)} total, {len(clean)} clean (non-duplicate) pairs loaded")
    return clean


def build_morables_groups(moral_texts: list[str], train_idx: list[int]) -> list[int]:
    """Group ID per training sample — same moral text → same group (multi-positive masking)."""
    text_to_group: dict[str, int] = {}
    group_ids = []
    for i in train_idx:
        text = moral_texts[i]
        if text not in text_to_group:
            text_to_group[text] = len(text_to_group)
        group_ids.append(text_to_group[text])
    return group_ids


def build_combined_dataset(
    moral_texts: list[str],
    doc_texts: list[str],
    ground_truth: dict[int, int],
    train_idx: list[int],
    storal_pairs: list[dict],
    instruction: str,
    seed: int,
) -> tuple[list[str], list[str], list[int]]:
    """
    Combine MORABLES training pairs + all clean STORAL pairs.

    Group IDs for multi-positive masking:
      - MORABLES: grouped by identical moral text (27 morals map to 2-4 fables each)
      - STORAL: each pair gets a unique group ID (they're independent, no duplicates)
    """
    morables_groups = build_morables_groups(moral_texts, train_idx)
    next_group = max(morables_groups) + 1 if morables_groups else 0

    anchors   = [f"{instruction}{moral_texts[i]}" for i in train_idx]
    positives = [doc_texts[ground_truth[i]] for i in train_idx]
    labels    = list(morables_groups)

    for p in storal_pairs:
        anchors.append(f"{instruction}{p['moral']}")
        positives.append(p["story"])
        labels.append(next_group)
        next_group += 1

    # Shuffle so STORAL and MORABLES pairs are interleaved across batches
    indices = list(range(len(anchors)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    anchors   = [anchors[i]   for i in indices]
    positives = [positives[i] for i in indices]
    labels    = [labels[i]    for i in indices]

    return anchors, positives, labels


def _build_st_model(config: dict):
    from sentence_transformers import SentenceTransformer
    model_kwargs = config.get("model_kwargs") or {}
    model = SentenceTransformer(
        config["model_name"],
        **({"model_kwargs": model_kwargs} if model_kwargs else {}),
    )
    if config.get("max_seq_length"):
        model.max_seq_length = config["max_seq_length"]

    lora_cfg = config.get("lora")
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


def _train(
    model,
    train_dataset,
    evaluator,
    config: dict,
    checkpoint_dir: Path,
    model_cache: Path,
    fold_idx: int,
    force: bool,
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
        last_ckpt = get_last_checkpoint(str(checkpoint_dir))
        if last_ckpt:
            checkpoint_to_resume = last_ckpt
            print(f"    [resume] ← {checkpoint_to_resume}")

    τ = config.get("temperature", 0.05)
    loss = InfoNCELoss(model, temperature=τ)
    lora_cfg = config.get("lora")

    steps_per_epoch = max(1, len(train_dataset) // config["batch_size"])
    best_metric_key = f"eval_fold_{fold_idx}_cosine_mrr@10"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        learning_rate=float(config["learning_rate"]),
        seed=config["seed"],
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        # False because load_best_model_at_end=True OOMs on 24 GB cards: the 7B
        # model occupies ~20.8 GB, leaving only ~63 MB free — not enough to load
        # the adapter checkpoint (~112 MB). BestAdapterCallback handles this instead
        # by saving adapter weights to CPU RAM during training and restoring them
        # after trainer.train() returns (when activations are freed).
        load_best_model_at_end=False,
        metric_for_best_model=best_metric_key,  # still used by EarlyStoppingCallback
        greater_is_better=True,
        dataloader_pin_memory=False,
        logging_steps=max(1, steps_per_epoch // 2),
        report_to="wandb" if wandb.run is not None else "none",
    )

    callbacks = []
    if config.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"]))

    best_adapter_cb = None
    if lora_cfg:
        best_adapter_cb = BestAdapterCallback(model[0].auto_model, best_metric_key)
        callbacks.append(best_adapter_cb)

    import os
    if os.getenv("TG_BOT_TOKEN") and os.getenv("TG_CHAT_ID"):
        from finetuning.lib.notify import TelegramCallback
        callbacks.append(TelegramCallback(label=f"ft_04/fold_{fold_idx}"))

    SentenceTransformerTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        evaluator=evaluator,
        loss=loss,
        callbacks=callbacks or None,
    ).train(resume_from_checkpoint=checkpoint_to_resume)

    # Free training graph and activations first — only then does VRAM have
    # enough headroom (~112 MB) to receive the adapter weights from CPU RAM.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if best_adapter_cb is not None:
        best_adapter_cb.restore(model[0].auto_model)

    if lora_cfg:
        model[0].auto_model = model[0].auto_model.merge_and_unload()

    model_cache.mkdir(parents=True, exist_ok=True)
    model.save(str(model_cache))
    print(f"    [saved] → {model_cache}")
    return model


def run_fold(
    fold: dict,
    moral_texts: list[str],
    doc_texts: list[str],
    ground_truth: dict[int, int],
    storal_pairs: list[dict],
    config: dict,
    force: bool,
    use_wandb: bool,
) -> dict:
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    fold_idx = fold["fold"]
    train_idx, test_idx = fold["train"], fold["test"]
    instruction = config.get("query_instruction", "")
    τ = config.get("temperature", 0.05)

    anchors, positives, labels = build_combined_dataset(
        moral_texts, doc_texts, ground_truth, train_idx, storal_pairs, instruction, config["seed"]
    )
    train_dataset = Dataset.from_dict({
        "anchor": anchors, "positive": positives, "label": labels
    })
    n_morables = len(train_idx)
    n_storal = len(storal_pairs)

    print(
        f"\n  Fold {fold_idx + 1}/5  "
        f"MORABLES={n_morables}  STORAL={n_storal}  total={len(anchors)}  "
        f"test={len(test_idx)}  τ={τ}"
    )

    test_morals = [f"{instruction}{moral_texts[i]}" for i in test_idx]
    test_gt = {j: ground_truth[i] for j, i in enumerate(test_idx)}

    _model_root = Path(config["model_output_dir"]) if config.get("model_output_dir") else CACHE_DIR / "models"
    model_cache = _model_root / config["doc_mode"] / f"fold_{fold_idx}"
    checkpoint_dir = CACHE_DIR / "checkpoints" / config["doc_mode"] / f"fold_{fold_idx}"
    emb_cache = CACHE_DIR / "embeddings" / config["doc_mode"] / f"fold_{fold_idx}"

    if model_cache.exists() and not force:
        print(f"    [cache hit] Loading model ← {model_cache}")
        model = SentenceTransformer(str(model_cache))
    else:
        evaluator = InformationRetrievalEvaluator(
            queries={str(j): text for j, text in enumerate(test_morals)},
            corpus={str(i): text for i, text in enumerate(doc_texts)},
            relevant_docs={str(j): {str(test_gt[j])} for j in range(len(test_morals))},
            mrr_at_k=[10], ndcg_at_k=[10], accuracy_at_k=[1, 5, 10],
            name=f"fold_{fold_idx}",
        )
        if use_wandb:
            wandb.init(
                project=config["wandb"]["project"],
                name=f"fold_{fold_idx}_storal_augment",
                group="ft_04_storal_augment",
                tags=["ft_04_storal_augment", f"fold_{fold_idx}"],
                config={k: v for k, v in config.items() if k != "wandb"},
            )
        model = _build_st_model(config)
        model = _train(model, train_dataset, evaluator, config, checkpoint_dir, model_cache, fold_idx, force)

    metrics = evaluate(model, test_morals, doc_texts, test_gt, cache_dir=emb_cache, force=force)
    print(
        f"  MRR={metrics['MRR']:.4f}  "
        f"R@1={metrics['Recall@1']:.4f}  "
        f"R@5={metrics['Recall@5']:.4f}  "
        f"R@10={metrics['Recall@10']:.4f}"
    )

    # wandb.run is None when the fold was a cache hit (wandb.init was skipped)
    if use_wandb and wandb.run is not None:
        wandb.log({"fold": fold_idx, **{f"eval/{k.lower().replace('@', '_at_')}": metrics[k] for k in _EVAL_KEYS}})
        wandb.finish()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="ft_04 STORAL-augmented InfoNCE fine-tuning")
    parser.add_argument("--fold", type=int, choices=range(5), metavar="0-4",
                        help="Run a single fold instead of all 5")
    parser.add_argument("--doc_mode", choices=["raw", "fable_plus_summary"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-wandb", dest="no_wandb", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    if args.doc_mode:
        config["doc_mode"] = args.doc_mode

    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"Splits not found: {SPLITS_PATH}\nRun ft_01 prepare_data.py first.")

    with open(SPLITS_PATH) as f:
        all_folds = json.load(f)

    storal_pairs = load_storal_pairs()
    use_wandb = config.get("wandb", {}).get("enabled", False) and not args.no_wandb
    τ = config.get("temperature", 0.05)

    print(
        f"\n[ft_04_storal_augment]  model={config['model_name']}  "
        f"τ={τ}  doc_mode={config['doc_mode']}  epochs={config['epochs']}  "
        f"storal_pairs={len(storal_pairs)}"
    )

    notify.send(
        f"🚀 ft_04_storal_augment starting\n"
        f"STORAL pairs: {len(storal_pairs)}  τ: {τ}\n"
        f"doc_mode: {config['doc_mode']}  epochs: {config['epochs']}\n"
        f"folds: {[args.fold] if args.fold is not None else 'all 5'}"
    )

    moral_texts, doc_texts, ground_truth = load_pairs(config["doc_mode"])

    fold_indices = [args.fold] if args.fold is not None else list(range(len(all_folds)))
    fold_metrics = [
        run_fold(
            all_folds[i], moral_texts, doc_texts, ground_truth, storal_pairs,
            config, args.force, use_wandb,
        )
        for i in fold_indices
    ]

    mrr_scores = [m["MRR"] for m in fold_metrics]
    mean_mrr = float(np.mean(mrr_scores))
    std_mrr = float(np.std(mrr_scores))
    print(f"\n  Final MRR: {mean_mrr:.4f} ± {std_mrr:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"fold{args.fold}" if args.fold is not None else "all_folds"
    out = RESULTS_DIR / f"{ts}_storal_augment_tau{str(τ).replace('.', '')}_{config['doc_mode']}_{suffix}.json"
    with open(out, "w") as f:
        json.dump({
            "config": config,
            "storal_pairs_used": len(storal_pairs),
            "temperature": τ,
            "folds_run": fold_indices,
            "mean_mrr": mean_mrr,
            "std_mrr": std_mrr,
            "fold_mrrs": mrr_scores,
            "fold_metrics": fold_metrics,
        }, f, indent=2)
    print(f"  Results → {out}")

    notify.send(
        f"✅ ft_04_storal_augment done\n"
        f"STORAL pairs: {len(storal_pairs)}  τ: {τ}\n"
        f"Final MRR: {mean_mrr:.4f} ± {std_mrr:.4f}"
    )


if __name__ == "__main__":
    main()
