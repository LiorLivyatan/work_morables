"""
ft_10 Option B — Stage 1: IdioLink Pre-training.

Fine-tunes Linq+LoRA on 440 IdioLink triplets using MultipleNegativesRankingLoss
(same loss that outperformed InfoNCE in Option A). After training, the LoRA adapter
is merged into the base weights and the merged model is saved to disk for Stage 2.

No MORABLES data is used here. Stage 2 (finetune.py) loads the merged model and
runs the full MORABLES 5-fold CV with ft_09 settings.

Usage
-----
    ./run.sh finetuning/ft_10_idiolink_transfer/option_b_sequential/pretrain.py --remote --gpu 1

    # Force re-train even if the merged model already exists:
    ./run.sh finetuning/ft_10_idiolink_transfer/option_b_sequential/pretrain.py --force --remote --gpu 1
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml

EXP_DIR     = Path(__file__).parent
ROOT        = EXP_DIR.parent.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib import notify

CONFIG_PATH = EXP_DIR / "config_pretrain.yaml"


def load_idiolink_data(idiolink_dir: Path):
    """Return (anchors, positives, hard_negatives) from train_triplets.jsonl."""
    anchors, positives, negs = [], [], []
    with open(idiolink_dir / "train_triplets.jsonl") as f:
        for line in f:
            t = json.loads(line)
            anchors.append(t["query"])
            positives.append(t["positive"])
            negs.append(t["negatives"][0])
    return anchors, positives, negs


def build_idiolink_val_evaluator(idiolink_dir: Path, name: str = "idiolink_val"):
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    with open(idiolink_dir / "val_indexes.json") as f:
        index_entries = json.load(f)
    corpus     = {e["id"]: e["sentence"] for e in index_entries}
    text_to_id = {e["sentence"]: e["id"] for e in index_entries}

    queries, relevant_docs = {}, {}
    with open(idiolink_dir / "val_triplets.jsonl") as f:
        for i, line in enumerate(f):
            t     = json.loads(line)
            qid   = f"q_{i}"
            doc_id = text_to_id.get(t["positive"])
            if doc_id is None:
                continue
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


def build_model(config: dict):
    from sentence_transformers import SentenceTransformer
    from peft import LoraConfig, TaskType, get_peft_model

    model_kwargs = dict(config.get("model_kwargs") or {})
    if "torch_dtype" in model_kwargs:
        dtype = model_kwargs.pop("torch_dtype")
        model_kwargs["torch_dtype"] = getattr(torch, dtype, dtype) if isinstance(dtype, str) else dtype

    model = SentenceTransformer(config["model_name"], model_kwargs=model_kwargs or None)
    model.max_seq_length = config["max_seq_length"]

    lora_cfg = config["lora"]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="ft_10 Option B — Stage 1: IdioLink pre-training")
    parser.add_argument("--force", action="store_true", help="Re-train even if merged model exists")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    model_output_dir = Path(config["model_output_dir"])
    idiolink_dir     = ROOT / config["idiolink_data_dir"]

    if model_output_dir.exists() and not args.force:
        print(f"[pretrain] Stage 1 model already exists ← {model_output_dir}")
        print("[pretrain] Run with --force to re-train. Exiting.")
        return

    notify.send(
        f"🚀 ft_10b Stage 1 starting\n"
        f"model: {config['model_name']}\n"
        f"lora r={config['lora']['r']}  bs={config['batch_size']}×"
        f"{config['gradient_accumulation_steps']}  lr={config['learning_rate']}"
    )

    anchors, positives, hard_negatives = load_idiolink_data(idiolink_dir)
    print(f"[pretrain] IdioLink train: {len(anchors)} triplets")

    val_evaluator = build_idiolink_val_evaluator(idiolink_dir)

    from datasets import Dataset
    from sentence_transformers import losses
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from transformers import EarlyStoppingCallback
    from transformers.trainer_utils import get_last_checkpoint

    checkpoint_dir = EXP_DIR / "cache" / "checkpoints"
    if args.force and checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)

    checkpoint_to_resume = None
    if checkpoint_dir.exists():
        last = get_last_checkpoint(str(checkpoint_dir))
        if last:
            checkpoint_to_resume = last
            print(f"[pretrain] Resuming from {checkpoint_to_resume}")

    model = build_model(config)

    dataset = Dataset.from_dict({
        "anchor":   anchors,
        "positive": positives,
        "negative": hard_negatives,
    })

    loss = losses.MultipleNegativesRankingLoss(model)

    steps_per_ep = max(1, len(anchors) // config["batch_size"])
    metric_key   = "idiolink_val_cosine_ndcg@10"

    callbacks = [EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])]
    if os.getenv("TG_BOT_TOKEN") and os.getenv("TG_CHAT_ID"):
        from finetuning.lib.notify import TelegramCallback
        callbacks.append(TelegramCallback(label="ft_10b/pretrain"))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        learning_rate=float(config["learning_rate"]),
        warmup_steps=config.get("warmup_steps", 100),
        seed=config.get("seed", 42),
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_key,
        greater_is_better=True,
        dataloader_pin_memory=False,
        logging_steps=max(1, steps_per_ep // 2),
        report_to="none",
    )

    SentenceTransformerTrainer(
        model=model, args=trainer_args,
        train_dataset=dataset, evaluator=val_evaluator,
        loss=loss, callbacks=callbacks,
    ).train(resume_from_checkpoint=checkpoint_to_resume)

    model[0].auto_model = model[0].auto_model.merge_and_unload()
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_output_dir))
    print(f"[pretrain] Merged model saved → {model_output_dir}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    notify.send(
        f"✅ ft_10b Stage 1 done\n"
        f"Merged model → {model_output_dir}\n"
        f"Run Stage 2: finetune.py"
    )


if __name__ == "__main__":
    main()
