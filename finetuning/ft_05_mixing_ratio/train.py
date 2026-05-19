"""
ft_05_mixing_ratio — STORAL mixing ratio ablation on BGE-base.

Sweeps over N STORAL pairs added to training (0, 200, 500, 1000, 1675)
using a fast local model (BAAI/bge-base-en-v1.5, 109M params) to find
the optimal augmentation size before committing GPU hours to Linq.

Designed for local M4 Pro runs — no LoRA, no Telegram, full fine-tune.
Results guide the ft_06 Linq run on the server.

Usage
-----
    # Full sweep, all folds (recommended, ~30-60 min):
    ./run.sh finetuning/ft_05_mixing_ratio/train.py

    # Quick single-fold test:
    ./run.sh finetuning/ft_05_mixing_ratio/train.py --fold 0

    # Custom ratios:
    ./run.sh finetuning/ft_05_mixing_ratio/train.py --ratios 0 500 1675
"""
import argparse
import gc
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib.data import load_pairs
from finetuning.lib.eval import evaluate
from finetuning.lib.losses import InfoNCELoss

CACHE_DIR = EXP_DIR / "cache"
RESULTS_DIR = EXP_DIR / "results"
CONFIG_PATH = EXP_DIR / "config.yaml"
SPLITS_PATH = EXP_DIR.parent / "ft_01_5fold_cv" / "cache" / "splits" / "folds.json"
STORAL_PATH = ROOT / "data/external/storal/processed/storal_pairs.json"


def load_storal_pairs() -> list[dict]:
    pairs = json.loads(STORAL_PATH.read_text())
    clean = [p for p in pairs if not p["is_duplicate"]]
    return clean


def sample_storal(all_pairs: list[dict], n: int, seed: int) -> list[dict]:
    if n == 0:
        return []
    if n >= len(all_pairs):
        return all_pairs
    rng = random.Random(seed)
    return rng.sample(all_pairs, n)


def build_moral_groups(moral_texts: list[str], train_idx: list[int]) -> list[int]:
    text_to_group: dict[str, int] = {}
    group_ids = []
    for i in train_idx:
        text = moral_texts[i]
        if text not in text_to_group:
            text_to_group[text] = len(text_to_group)
        group_ids.append(text_to_group[text])
    return group_ids


def build_dataset(
    moral_texts: list[str],
    doc_texts: list[str],
    ground_truth: dict[int, int],
    train_idx: list[int],
    storal_sample: list[dict],
    instruction: str,
    seed: int,
):
    from datasets import Dataset

    morables_groups = build_moral_groups(moral_texts, train_idx)
    next_group = max(morables_groups) + 1 if morables_groups else 0

    anchors   = [f"{instruction}{moral_texts[i]}" for i in train_idx]
    positives = [doc_texts[ground_truth[i]] for i in train_idx]
    labels    = list(morables_groups)

    for p in storal_sample:
        anchors.append(f"{instruction}{p['moral']}")
        positives.append(p["story"])
        labels.append(next_group)
        next_group += 1

    indices = list(range(len(anchors)))
    random.Random(seed).shuffle(indices)
    return Dataset.from_dict({
        "anchor":   [anchors[i]   for i in indices],
        "positive": [positives[i] for i in indices],
        "label":    [labels[i]    for i in indices],
    })


def train_fold(
    fold: dict,
    moral_texts: list[str],
    doc_texts: list[str],
    ground_truth: dict[int, int],
    storal_sample: list[dict],
    config: dict,
    storal_n: int,
    force: bool,
) -> float:
    import shutil
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

    train_dataset = build_dataset(
        moral_texts, doc_texts, ground_truth, train_idx, storal_sample, instruction, config["seed"]
    )

    test_morals = [f"{instruction}{moral_texts[i]}" for i in test_idx]
    test_gt     = {j: ground_truth[i] for j, i in enumerate(test_idx)}

    model_cache    = CACHE_DIR / "models"    / f"ratio_{storal_n}" / f"fold_{fold_idx}"
    checkpoint_dir = CACHE_DIR / "checkpoints" / f"ratio_{storal_n}" / f"fold_{fold_idx}"
    emb_cache      = CACHE_DIR / "embeddings"  / f"ratio_{storal_n}" / f"fold_{fold_idx}"

    if force and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    if force and model_cache.exists():
        shutil.rmtree(model_cache)

    if model_cache.exists() and not force:
        print(f"      [cache] ratio={storal_n} fold={fold_idx}")
        model = SentenceTransformer(str(model_cache))
    else:
        model = SentenceTransformer(config["model_name"])

        τ = config.get("temperature", 0.05)
        loss = InfoNCELoss(model, temperature=τ)

        steps_per_epoch = max(1, len(train_dataset) // config["batch_size"])
        best_metric_key = f"eval_fold_{fold_idx}_cosine_mrr@10"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_to_resume = None
        if checkpoint_dir.exists():
            last_ckpt = get_last_checkpoint(str(checkpoint_dir))
            if last_ckpt:
                checkpoint_to_resume = last_ckpt

        evaluator = InformationRetrievalEvaluator(
            queries={str(j): text for j, text in enumerate(test_morals)},
            corpus={str(i): text for i, text in enumerate(doc_texts)},
            relevant_docs={str(j): {str(test_gt[j])} for j in range(len(test_morals))},
            mrr_at_k=[10], ndcg_at_k=[10], accuracy_at_k=[1, 5, 10],
            name=f"fold_{fold_idx}",
        )

        SentenceTransformerTrainer(
            model=model,
            args=SentenceTransformerTrainingArguments(
                output_dir=str(checkpoint_dir),
                num_train_epochs=config["epochs"],
                per_device_train_batch_size=config["batch_size"],
                learning_rate=float(config["learning_rate"]),
                seed=config["seed"],
                save_strategy="epoch",
                save_total_limit=1,
                eval_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model=best_metric_key,
                greater_is_better=True,
                dataloader_pin_memory=False,
                logging_steps=max(1, steps_per_epoch // 2),
                report_to="none",
            ),
            train_dataset=train_dataset,
            evaluator=evaluator,
            loss=loss,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])],
        ).train(resume_from_checkpoint=checkpoint_to_resume)

        model_cache.mkdir(parents=True, exist_ok=True)
        model.save(str(model_cache))

    metrics = evaluate(model, test_morals, doc_texts, test_gt, cache_dir=emb_cache, force=force)
    mrr = metrics["MRR"]

    del model
    gc.collect()

    return mrr


def main() -> None:
    parser = argparse.ArgumentParser(description="ft_05 STORAL mixing ratio sweep")
    parser.add_argument("--ratios", type=int, nargs="+",
                        help="Override STORAL pair counts to test (e.g. --ratios 0 500 1675)")
    parser.add_argument("--fold", type=int, choices=range(5), metavar="0-4",
                        help="Single fold only (faster for quick checks)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    with open(SPLITS_PATH) as f:
        all_folds = json.load(f)

    all_storal = load_storal_pairs()
    ratios = args.ratios if args.ratios is not None else config["storal_ratios"]
    fold_indices = [args.fold] if args.fold is not None else list(range(len(all_folds)))

    moral_texts, doc_texts, ground_truth = load_pairs(config["doc_mode"])

    print(f"\n[ft_05_mixing_ratio]  model={config['model_name']}  "
          f"ratios={ratios}  folds={fold_indices}")
    print(f"  STORAL available: {len(all_storal)} clean pairs\n")

    summary: dict[int, dict] = {}

    for storal_n in ratios:
        storal_sample = sample_storal(all_storal, storal_n, seed=config["seed"])
        fold_mrrs = []

        print(f"  ── ratio={storal_n} ({len(storal_sample)} STORAL + MORABLES) ──")
        for fi in fold_indices:
            fold = all_folds[fi]
            n_train = len(fold["train"]) + len(storal_sample)
            mrr = train_fold(
                fold, moral_texts, doc_texts, ground_truth,
                storal_sample, config, storal_n, args.force,
            )
            fold_mrrs.append(mrr)
            print(f"    fold {fi}  train={n_train}  MRR={mrr:.4f}")

        mean_mrr = float(np.mean(fold_mrrs))
        std_mrr  = float(np.std(fold_mrrs))
        summary[storal_n] = {"mean_mrr": mean_mrr, "std_mrr": std_mrr, "fold_mrrs": fold_mrrs}
        print(f"  → ratio={storal_n}  mean MRR={mean_mrr:.4f} ± {std_mrr:.4f}\n")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"{'STORAL pairs':>14}  {'MRR@10':>8}  {'± std':>7}")
    print("─" * 50)
    best_n = max(summary, key=lambda n: summary[n]["mean_mrr"])
    for n in ratios:
        s = summary[n]
        marker = " ◀ best" if n == best_n else ""
        print(f"{n:>14}  {s['mean_mrr']:.4f}  ±{s['std_mrr']:.4f}{marker}")
    print("=" * 50)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"fold{args.fold}" if args.fold is not None else "all_folds"
    out = RESULTS_DIR / f"{ts}_mixing_ratio_{suffix}.json"
    with open(out, "w") as f:
        json.dump({
            "config": config,
            "ratios_tested": ratios,
            "folds_run": fold_indices,
            "summary": {str(k): v for k, v in summary.items()},
        }, f, indent=2)
    print(f"\n  Results → {out}")


if __name__ == "__main__":
    main()
