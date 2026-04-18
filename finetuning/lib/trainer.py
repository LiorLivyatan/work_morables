"""
SentenceTransformer contrastive fine-tuning wrapper.

Public API
----------
train_model(train_morals, train_docs, config, model_cache, checkpoint_dir,
            evaluator=None, force=False)
    -> SentenceTransformer

Caching & Resume
----------------
After each epoch the HF Trainer saves a full checkpoint to checkpoint_dir:
model weights, optimizer state, LR scheduler state, and RNG state. If training
is interrupted for any reason, the next call automatically resumes from the
latest checkpoint — no data is lost.

Final model weights are saved separately to model_cache in SentenceTransformer
format for fast inference and evaluation.

cache/
├── models/<doc_mode>/[fold_N/]       ← final (or best) model (inference)
├── checkpoints/<doc_mode>/[fold_N/]  ← per-epoch trainer state (resume)
└── embeddings/<doc_mode>/[fold_N/]   ← encoded corpus/query arrays

Mid-training evaluation
-----------------------
Pass an InformationRetrievalEvaluator as `evaluator` to measure retrieval MRR
after each epoch. When provided:
  - Eval metrics stream to wandb alongside training loss
  - load_best_model_at_end=True: the trainer restores the best-epoch model at
    the end of training rather than the final-epoch model

Early stopping
--------------
Set `early_stopping_patience` in config to stop training when MRR stops
improving. Requires an evaluator. Training halts after N consecutive epochs
with no improvement, saving the best-epoch model regardless.

LoRA fine-tuning
----------------
Set a `lora` block in config to use parameter-efficient fine-tuning instead
of updating all weights. Recommended for large models (>1B params) or when
training data is small (reduces overfitting). LoRA adapters are merged into
the base model before saving, so the output is a standard SentenceTransformer
with no runtime PEFT dependency.

    lora:
      r: 64
      alpha: 128
      dropout: 0.05
      target_modules: [q_proj, k_proj, v_proj, o_proj]

Monitoring
----------
If a wandb run is active when train_model() is called, the HuggingFace Trainer
automatically streams training loss and learning rate to that run. wandb
lifecycle (init/finish) is owned by the caller, not this module.
"""
import shutil
from pathlib import Path

import wandb
from sentence_transformers import SentenceTransformer


def train_model(
    train_morals: list[str],
    train_docs: list[str],
    config: dict,
    model_cache: Path,
    checkpoint_dir: Path,
    evaluator=None,
    force: bool = False,
) -> SentenceTransformer:
    """
    Fine-tune a bi-encoder on (moral, fable) pairs with MultipleNegativesRankingLoss.

    Args:
        train_morals    anchor sentences (short moral statements)
        train_docs      positive sentences (fable texts or fable+summary)
        config          dict with keys: model_name, epochs, batch_size, learning_rate,
                        seed; optional keys: lora, early_stopping_patience, model_kwargs
        model_cache     directory for the final SentenceTransformer model (inference)
        checkpoint_dir  directory for per-epoch HF Trainer checkpoints (resume)
        evaluator       optional InformationRetrievalEvaluator; when provided, MRR is
                        measured after every epoch and the best-epoch model is saved
        force           if True, wipe checkpoints + model cache and retrain from scratch
    """
    if model_cache.exists() and not force:
        print(f"    [cache hit] Loading model ← {model_cache}")
        return SentenceTransformer(str(model_cache))

    from datasets import Dataset
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    if force and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print(f"    [force] Cleared checkpoints ← {checkpoint_dir}")

    # Detect a resumable checkpoint
    checkpoint_to_resume = None
    if checkpoint_dir.exists():
        last_ckpt = get_last_checkpoint(str(checkpoint_dir))
        if last_ckpt:
            checkpoint_to_resume = last_ckpt
            print(f"    [resume] Resuming from checkpoint ← {checkpoint_to_resume}")

    lora_cfg = config.get("lora")
    early_stopping_patience = config.get("early_stopping_patience")
    model_kwargs = config.get("model_kwargs") or {}

    lora_tag = f"  LoRA r={lora_cfg['r']}" if lora_cfg else ""
    suffix = " (resuming)" if checkpoint_to_resume else ""
    print(
        f"    Training {config['model_name']} — "
        f"{len(train_morals)} pairs, "
        f"{config['epochs']} epochs, "
        f"batch={config['batch_size']}"
        f"{lora_tag}"
        f"{suffix}"
    )

    st_kwargs = {"model_kwargs": model_kwargs} if model_kwargs else {}
    model = SentenceTransformer(config["model_name"], **st_kwargs)
    if config.get("max_seq_length"):
        model.max_seq_length = config["max_seq_length"]

    # Apply LoRA adapters if configured. Only the adapter weights are trained;
    # the frozen base is kept in memory and merged back before saving.
    if lora_cfg:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model[0].auto_model = get_peft_model(model[0].auto_model, peft_config)
        model[0].auto_model.print_trainable_parameters()

    loss = MultipleNegativesRankingLoss(model)
    train_dataset = Dataset.from_dict({"anchor": train_morals, "positive": train_docs})

    steps_per_epoch = max(1, len(train_morals) // config["batch_size"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # When an evaluator is provided we can track the best epoch and restore it.
    # Without one we just save the final epoch (no eval to compare against).
    has_eval = evaluator is not None
    # InformationRetrievalEvaluator reports e.g. "eval_fold_0_cosine_mrr@10" — not
    # "eval_loss". We must override metric_for_best_model so HF Trainer knows which
    # metric to maximise when load_best_model_at_end=True.
    best_metric = f"eval_{evaluator.name}_cosine_mrr@10" if has_eval else None
    args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        learning_rate=float(config["learning_rate"]),
        seed=config["seed"],
        save_strategy="epoch",
        save_total_limit=2 if has_eval else 1,  # need 2 for load_best_model_at_end
        eval_strategy="epoch" if has_eval else "no",
        load_best_model_at_end=has_eval,
        metric_for_best_model=best_metric,
        greater_is_better=True if has_eval else None,
        dataloader_pin_memory=False,            # MPS doesn't support pin_memory
        logging_steps=max(1, steps_per_epoch // 2),
        report_to="wandb" if wandb.run is not None else "none",
    )

    # EarlyStoppingCallback halts training when MRR stops improving, saving
    # the best-epoch model. Requires an evaluator (has_eval=True).
    callbacks = []
    if early_stopping_patience and has_eval:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    # TelegramCallback — auto-added when TG env vars are set (injected by run.sh
    # for remote runs; silent no-op locally when vars are absent).
    import os
    if os.getenv("TG_BOT_TOKEN") and os.getenv("TG_CHAT_ID"):
        from finetuning.lib.notify import TelegramCallback
        label = "/".join(checkpoint_dir.parts[-3:]) if len(checkpoint_dir.parts) >= 3 else checkpoint_dir.name
        callbacks.append(TelegramCallback(label=label))

    SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        evaluator=evaluator,
        loss=loss,
        callbacks=callbacks or None,
    ).train(resume_from_checkpoint=checkpoint_to_resume)

    # Merge LoRA adapter weights into the base model before saving. The result
    # is a standard SentenceTransformer — no PEFT dependency at inference time.
    if lora_cfg:
        model[0].auto_model = model[0].auto_model.merge_and_unload()
        print("    [LoRA] Adapters merged into base model")

    model_cache.mkdir(parents=True, exist_ok=True)
    model.save(str(model_cache))
    print(f"    [cache saved] {'Best' if has_eval else 'Final'} model → {model_cache}")

    return model
