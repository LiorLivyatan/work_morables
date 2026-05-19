"""
ft_06_random_search — Random hyperparameter search for Linq+LoRA on local MPS.

Sweeps random combinations of (LoRA rank r, temperature τ, learning rate) using
MORABLES-only training on fold 0. No STORAL augmentation — ft_05 showed augmentation
provides at most +0.01 MRR, not worth the noise for hyperparameter search.

Best config from this search goes to the GPU server for a full 5-fold run.

Usage
-----
    # Full random search (n_trials from config):
    ./run.sh finetuning/ft_06_random_search/train.py

    # Custom trial count:
    ./run.sh finetuning/ft_06_random_search/train.py --n_trials 6

    # Force re-run (clears cache):
    ./run.sh finetuning/ft_06_random_search/train.py --force
"""
import argparse
import gc
import json
import os
import random
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import torch
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


class BestAdapterCallback(TrainerCallback):
    """
    Saves LoRA adapter weights to CPU RAM whenever MRR improves.
    Avoids load_best_model_at_end=True, which can OOM when the optimizer
    states + model + checkpoint all compete for the same memory pool.
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
            print(f"    [best] MRR={mrr:.4f} — adapter saved to CPU")

    def restore(self, peft_model) -> float:
        if self.best_state is None:
            return self.best_mrr
        device = next(peft_model.parameters()).device
        state = peft_model.state_dict()
        for k, v in self.best_state.items():
            state[k] = v.to(device)
        peft_model.load_state_dict(state)
        print(f"    [best] Restored best adapter (MRR={self.best_mrr:.4f})")
        return self.best_mrr


def sample_trials(search_space: dict, n: int, seed: int) -> list[dict]:
    keys = list(search_space.keys())
    all_combos = list(product(*[search_space[k] for k in keys]))
    rng = random.Random(seed)
    sampled = rng.sample(all_combos, min(n, len(all_combos)))
    return [{k: v for k, v in zip(keys, combo)} for combo in sampled]


def trial_id(trial: dict) -> str:
    r  = trial["lora_r"]
    t  = str(trial["temperature"]).replace(".", "")
    lr = f"{trial['learning_rate']:.0e}".replace("-0", "-")
    return f"r{r}_t{t}_lr{lr}"


def build_dataset(moral_texts, doc_texts, ground_truth, train_idx, instruction, seed):
    from datasets import Dataset

    text_to_group: dict[str, int] = {}
    group_ids = []
    for i in train_idx:
        text = moral_texts[i]
        if text not in text_to_group:
            text_to_group[text] = len(text_to_group)
        group_ids.append(text_to_group[text])

    anchors   = [f"{instruction}{moral_texts[i]}" for i in train_idx]
    positives = [doc_texts[ground_truth[i]] for i in train_idx]

    indices = list(range(len(anchors)))
    random.Random(seed).shuffle(indices)
    return Dataset.from_dict({
        "anchor":   [anchors[i]   for i in indices],
        "positive": [positives[i] for i in indices],
        "label":    [group_ids[i] for i in indices],
    })


def run_trial(
    trial: dict,
    fold: dict,
    moral_texts: list[str],
    doc_texts: list[str],
    ground_truth: dict[int, int],
    config: dict,
    force: bool,
) -> float:
    import shutil
    from peft import LoraConfig, TaskType, get_peft_model
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from transformers import EarlyStoppingCallback
    from transformers.trainer_utils import get_last_checkpoint

    fold_idx   = fold["fold"]
    train_idx  = fold["train"]
    test_idx   = fold["test"]
    instruction = config.get("query_instruction", "")
    tid = trial_id(trial)

    train_dataset = build_dataset(
        moral_texts, doc_texts, ground_truth, train_idx, instruction, config["seed"]
    )
    test_morals = [f"{instruction}{moral_texts[i]}" for i in test_idx]
    test_gt = {j: ground_truth[i] for j, i in enumerate(test_idx)}

    _model_root = Path(config["model_output_dir"]) if config.get("model_output_dir") else CACHE_DIR / "models"
    _ckpt_root  = Path(config["checkpoint_output_dir"]) if config.get("checkpoint_output_dir") else CACHE_DIR / "checkpoints"
    checkpoint_dir = _ckpt_root / tid
    model_cache    = _model_root / tid
    emb_cache      = CACHE_DIR / "embeddings"  / tid

    if force:
        for d in (checkpoint_dir, model_cache):
            if d.exists():
                shutil.rmtree(d)

    if model_cache.exists() and not force:
        print(f"      [cache hit] {tid}")
        model = SentenceTransformer(str(model_cache))
    else:
        raw_kwargs = config.get("model_kwargs") or {}
        model_kwargs = {}
        if "torch_dtype" in raw_kwargs:
            model_kwargs["torch_dtype"] = getattr(torch, raw_kwargs["torch_dtype"])

        model = SentenceTransformer(
            config["model_name"],
            **({"model_kwargs": model_kwargs} if model_kwargs else {}),
        )
        if config.get("max_seq_length"):
            model.max_seq_length = config["max_seq_length"]

        lora_r = trial["lora_r"]
        model[0].auto_model = get_peft_model(
            model[0].auto_model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_r * 2,
                target_modules=config["lora"]["target_modules"],
                lora_dropout=config["lora"].get("dropout", 0.05),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            ),
        )
        model[0].auto_model.print_trainable_parameters()

        τ = trial["temperature"]
        loss = InfoNCELoss(model, temperature=τ)
        best_metric_key = f"eval_fold_{fold_idx}_cosine_mrr@10"
        steps_per_epoch = max(1, len(train_dataset) // config["batch_size"])

        evaluator = InformationRetrievalEvaluator(
            queries={str(j): text for j, text in enumerate(test_morals)},
            corpus={str(i): text for i, text in enumerate(doc_texts)},
            relevant_docs={str(j): {str(test_gt[j])} for j in range(len(test_morals))},
            mrr_at_k=[10], ndcg_at_k=[10], accuracy_at_k=[1, 5, 10],
            name=f"fold_{fold_idx}",
        )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_to_resume = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None

        best_cb = BestAdapterCallback(model[0].auto_model, best_metric_key)

        SentenceTransformerTrainer(
            model=model,
            args=SentenceTransformerTrainingArguments(
                output_dir=str(checkpoint_dir),
                num_train_epochs=config["epochs"],
                per_device_train_batch_size=config["batch_size"],
                gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
                gradient_checkpointing=config.get("gradient_checkpointing", False),
                learning_rate=float(trial["learning_rate"]),
                seed=config["seed"],
                save_strategy="epoch",
                save_total_limit=1,
                eval_strategy="epoch",
                load_best_model_at_end=False,
                metric_for_best_model=best_metric_key,
                greater_is_better=True,
                dataloader_pin_memory=False,
                dataloader_num_workers=0,
                logging_steps=max(1, steps_per_epoch // 2),
                report_to="none",
            ),
            train_dataset=train_dataset,
            evaluator=evaluator,
            loss=loss,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"]),
                best_cb,
            ],
        ).train(resume_from_checkpoint=checkpoint_to_resume)

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        best_cb.restore(model[0].auto_model)
        model[0].auto_model = model[0].auto_model.merge_and_unload()
        model_cache.mkdir(parents=True, exist_ok=True)
        model.save(str(model_cache))
        print(f"      [saved] → {model_cache}")

    metrics = evaluate(model, test_morals, doc_texts, test_gt, cache_dir=emb_cache, force=force)
    mrr = metrics["MRR"]

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mrr


def main() -> None:
    parser = argparse.ArgumentParser(description="ft_06 Linq+LoRA random hyperparameter search")
    parser.add_argument("--n_trials", type=int, help="Override number of random trials")
    parser.add_argument("--start_trial", type=int, default=1, help="1-indexed trial to start from (skip earlier trials)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    with open(SPLITS_PATH) as f:
        all_folds = json.load(f)

    n_trials = args.n_trials if args.n_trials is not None else config["n_trials"]
    trials = sample_trials(config["search_space"], n_trials, config["seed"])
    trials = trials[args.start_trial - 1:]
    fold = all_folds[0]

    moral_texts, doc_texts, ground_truth = load_pairs(config["doc_mode"])

    print(f"\n[ft_06_random_search]  model={config['model_name']}  n_trials={len(trials)}")
    print(f"  Search space: {config['search_space']}")
    print(f"  Sampled {len(trials)} trials:")
    for i, t in enumerate(trials):
        print(f"    {i+1:2d}. {trial_id(t)}")
    print()

    notify.send(
        f"🔍 ft_06_random_search starting\n"
        f"n_trials: {len(trials)}  fold: 0  model: Linq\n"
        f"r={config['search_space']['lora_r']}  "
        f"τ={config['search_space']['temperature']}  "
        f"lr={config['search_space']['learning_rate']}"
    )

    results: list[dict] = []

    for i, trial in enumerate(trials):
        tid = trial_id(trial)
        print(f"  ── trial {i+1}/{len(trials)}: {tid} ──")
        mrr = run_trial(trial, fold, moral_texts, doc_texts, ground_truth, config, args.force)
        results.append({**trial, "trial_id": tid, "mrr": mrr})
        print(f"    → MRR={mrr:.4f}\n")
        notify.send(f"ft_06 trial {i+1}/{len(trials)}: {tid}\nMRR={mrr:.4f}")

    results.sort(key=lambda x: x["mrr"], reverse=True)

    print("\n" + "=" * 58)
    print(f"{'trial_id':>30}  {'r':>4}  {'τ':>6}  {'lr':>7}  {'MRR':>7}")
    print("─" * 58)
    for j, r in enumerate(results):
        marker = " ◀ best" if j == 0 else ""
        print(
            f"  {r['trial_id']:>28}  {r['lora_r']:>4}  "
            f"{r['temperature']:>6}  {r['learning_rate']:>7.0e}  {r['mrr']:.4f}{marker}"
        )
    print("=" * 58)

    best = results[0]
    notify.send(
        f"✅ ft_06_random_search done\n"
        f"Best: {best['trial_id']}\n"
        f"r={best['lora_r']}  τ={best['temperature']}  lr={best['learning_rate']:.0e}\n"
        f"MRR={best['mrr']:.4f}"
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = RESULTS_DIR / f"{ts}_random_search.json"
    with open(out, "w") as f:
        json.dump({
            "config": config,
            "n_trials": len(trials),
            "trials": results,
            "best": best,
        }, f, indent=2)
    print(f"\n  Results → {out}")


if __name__ == "__main__":
    main()
