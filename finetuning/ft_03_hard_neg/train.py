"""
ft_03_hard_neg — InfoNCE fine-tuning with moral-spreading and optional hard negatives.

Two modes:
  Basic (types 1+2):  morals are pushed apart from each other in addition to
                      the standard in-batch fable negatives. No mining needed.
  Full  (types 1+2+3): adds an explicit hard negative fable per triplet.
                        Requires mine_negatives.py to have been run first.

Usage
-----
    # Basic mode — types 1+2, all folds:
    ./run.sh finetuning/ft_03_hard_neg/train.py

    # Basic mode — single fold for a quick test:
    ./run.sh finetuning/ft_03_hard_neg/train.py --fold 0

    # Full mode — types 1+2+3, injected_adjectives distractors:
    ./run.sh finetuning/ft_03_hard_neg/train.py --hard_neg --distractor_type injected_adjectives

    # Temperature sweep:
    ./run.sh finetuning/ft_03_hard_neg/train.py --hard_neg --distractor_type injected_adjectives --tau 0.07

    # Remote GPU:
    ./run.sh finetuning/ft_03_hard_neg/train.py --remote --gpu 2
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

CACHE_DIR = EXP_DIR / "cache"
RESULTS_DIR = EXP_DIR / "results"
CONFIG_PATH = EXP_DIR / "config.yaml"
SPLITS_PATH = EXP_DIR.parent / "ft_01_5fold_cv" / "cache" / "splits" / "folds.json"
HARD_NEG_PATH = EXP_DIR / "data" / "hard_negatives.json"

DISTRACTOR_TYPES = ["similar_characters", "based_on_adjectives", "injected_adjectives", "partial_story", "all"]
_EVAL_KEYS = ("MRR", "Recall@1", "Recall@5", "Recall@10")


def build_moral_groups(moral_texts: list[str], train_idx: list[int]) -> list[int]:
    """
    Assign a group ID to each training sample based on its moral text.
    Two samples with identical moral text get the same group ID — used for
    multi-positive masking in the loss.
    """
    text_to_group: dict[str, int] = {}
    group_ids = []
    for i in train_idx:
        text = moral_texts[i]
        if text not in text_to_group:
            text_to_group[text] = len(text_to_group)
        group_ids.append(text_to_group[text])
    return group_ids


def build_basic_dataset(
    moral_texts: list[str],
    doc_texts: list[str],
    ground_truth: dict[int, int],
    train_idx: list[int],
    instruction: str,
) -> tuple[list[str], list[str], list[int]]:
    """Types 1+2: (anchor, positive) pairs with moral group IDs for masking."""
    anchors   = [f"{instruction}{moral_texts[i]}" for i in train_idx]
    positives = [doc_texts[ground_truth[i]] for i in train_idx]
    labels    = build_moral_groups(moral_texts, train_idx)
    return anchors, positives, labels


def build_hard_neg_dataset(
    hard_negs: list[dict],
    distractor_type: str,
    moral_texts: list[str],
    doc_texts: list[str],
    ground_truth: dict[int, int],
    train_idx: list[int],
    instruction: str,
) -> tuple[list[str], list[str], list[str], list[int]]:
    """Types 1+2+3: (anchor, positive, negative) triplets with moral group IDs."""
    moral_idx_to_hard_negs: dict[int, list[dict]] = {}
    for rec in hard_negs:
        moral_idx_to_hard_negs.setdefault(rec["moral_idx"], []).append(rec)

    train_set = set(train_idx)
    group_id_for = {i: g for i, g in zip(train_idx, build_moral_groups(moral_texts, train_idx))}

    anchors, positives, negatives, labels = [], [], [], []
    for moral_idx in train_idx:
        fable_idx  = ground_truth[moral_idx]
        candidates = moral_idx_to_hard_negs.get(moral_idx, [])

        if distractor_type == "all":
            candidates = [random.choice(candidates)] if candidates else []
        else:
            candidates = [r for r in candidates if r["distractor_type"] == distractor_type]

        for rec in candidates:
            hard_neg_idx = rec["hard_neg_fable_idx"]
            if hard_neg_idx == fable_idx:
                continue
            anchors.append(f"{instruction}{moral_texts[moral_idx]}")
            positives.append(doc_texts[fable_idx])
            negatives.append(doc_texts[hard_neg_idx])
            labels.append(group_id_for[moral_idx])

    return anchors, positives, negatives, labels


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
    run_tag: str,
    fold_idx: int,
    force: bool,
):
    import shutil
    from sentence_transformers import SentenceTransformer
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
        callbacks.append(TelegramCallback(label=f"ft_03/{run_tag}/fold_{fold_idx}"))

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
    hard_negs: list[dict] | None,
    config: dict,
    distractor_type: str | None,
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
    use_hard_neg = hard_negs is not None

    # Build dataset
    if use_hard_neg:
        anchors, positives, negatives, labels = build_hard_neg_dataset(
            hard_negs, distractor_type, moral_texts, doc_texts, ground_truth, train_idx, instruction
        )
        train_dataset = Dataset.from_dict({
            "anchor": anchors, "positive": positives, "negative": negatives, "label": labels
        })
        n_triplets = len(anchors)
    else:
        anchors, positives, labels = build_basic_dataset(
            moral_texts, doc_texts, ground_truth, train_idx, instruction
        )
        train_dataset = Dataset.from_dict({
            "anchor": anchors, "positive": positives, "label": labels
        })
        n_triplets = len(anchors)

    mode_str = f"hard_neg={distractor_type}" if use_hard_neg else "basic"
    print(
        f"\n  Fold {fold_idx + 1}/5  "
        f"train={len(train_idx)} morals → {n_triplets} samples  "
        f"test={len(test_idx)}  mode={mode_str}  τ={τ}"
    )

    test_morals = [f"{instruction}{moral_texts[i]}" for i in test_idx]
    test_gt = {j: ground_truth[i] for j, i in enumerate(test_idx)}

    _model_root = Path(config["model_output_dir"]) if config.get("model_output_dir") else CACHE_DIR / "models"
    run_tag = f"{mode_str}_tau{str(τ).replace('.', '')}"
    model_cache = _model_root / config["doc_mode"] / run_tag / f"fold_{fold_idx}"
    checkpoint_dir = CACHE_DIR / "checkpoints" / config["doc_mode"] / run_tag / f"fold_{fold_idx}"
    emb_cache = CACHE_DIR / "embeddings" / config["doc_mode"] / run_tag / f"fold_{fold_idx}"

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
                name=f"fold_{fold_idx}_{run_tag}",
                group=f"ft_03_hard_neg/{run_tag}",
                tags=["ft_03_hard_neg", mode_str, f"fold_{fold_idx}"],
                config={k: v for k, v in config.items() if k != "wandb"},
            )
        model = _build_st_model(config)
        model = _train(model, train_dataset, evaluator, config, checkpoint_dir, model_cache, run_tag, fold_idx, force)

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
    parser = argparse.ArgumentParser(description="ft_03 InfoNCE fine-tuning")
    parser.add_argument("--hard_neg", action="store_true",
                        help="Enable hard negatives (types 1+2+3). Requires mine_negatives.py output.")
    parser.add_argument("--distractor_type", choices=DISTRACTOR_TYPES, default="injected_adjectives",
                        help="Distractor type to use for hard neg mining (only used with --hard_neg)")
    parser.add_argument("--tau", type=float, help="Override temperature τ from config")
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
    if args.tau is not None:
        config["temperature"] = args.tau

    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"Splits not found: {SPLITS_PATH}\nRun ft_01 prepare_data.py first.")

    with open(SPLITS_PATH) as f:
        all_folds = json.load(f)

    hard_negs = None
    if args.hard_neg:
        if not HARD_NEG_PATH.exists():
            raise FileNotFoundError(
                f"Hard negatives not found: {HARD_NEG_PATH}\n"
                "Run first: ./run.sh finetuning/ft_03_hard_neg/mine_negatives.py"
            )
        with open(HARD_NEG_PATH) as f:
            hard_negs = json.load(f)

    use_wandb = config.get("wandb", {}).get("enabled", False) and not args.no_wandb
    τ = config.get("temperature", 0.05)
    mode = f"hard_neg={args.distractor_type}" if args.hard_neg else "basic (types 1+2)"

    print(
        f"\n[ft_03_hard_neg]  model={config['model_name']}  "
        f"mode={mode}  τ={τ}  doc_mode={config['doc_mode']}  epochs={config['epochs']}"
    )

    notify.send(
        f"🚀 ft_03_hard_neg starting\n"
        f"mode: {mode}  τ: {τ}\n"
        f"doc_mode: {config['doc_mode']}  epochs: {config['epochs']}\n"
        f"folds: {[args.fold] if args.fold is not None else 'all 5'}"
    )

    moral_texts, doc_texts, ground_truth = load_pairs(config["doc_mode"])

    fold_indices = [args.fold] if args.fold is not None else list(range(len(all_folds)))
    fold_metrics = [
        run_fold(
            all_folds[i], moral_texts, doc_texts, ground_truth, hard_negs,
            config, args.distractor_type if args.hard_neg else None,
            args.force, use_wandb,
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
    tag = f"hard_neg_{args.distractor_type}" if args.hard_neg else "basic"
    out = RESULTS_DIR / f"{ts}_{tag}_tau{str(τ).replace('.', '')}_{config['doc_mode']}_{suffix}.json"
    with open(out, "w") as f:
        json.dump({
            "config": config,
            "mode": "hard_neg" if args.hard_neg else "basic",
            "distractor_type": args.distractor_type if args.hard_neg else None,
            "temperature": τ,
            "folds_run": fold_indices,
            "mean_mrr": mean_mrr,
            "std_mrr": std_mrr,
            "fold_mrrs": mrr_scores,
            "fold_metrics": fold_metrics,
        }, f, indent=2)
    print(f"  Results → {out}")

    notify.send(
        f"✅ ft_03_hard_neg done\n"
        f"mode: {mode}  τ: {τ}\n"
        f"Final MRR: {mean_mrr:.4f} ± {std_mrr:.4f}"
    )


if __name__ == "__main__":
    main()
