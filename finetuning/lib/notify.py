"""
Telegram notifications for training scripts.

Reads TG_BOT_TOKEN and TG_CHAT_ID from environment variables (injected by
run.sh for remote runs). Silently no-ops when not configured, so scripts
work identically in local runs without any setup.

Usage
-----
    from finetuning.lib import notify

    # One-off message (start of training, fold completions, etc.)
    notify.send("🚀 ft_01_5fold_cv starting — 5 folds, doc_mode=raw")

    # Epoch-level updates: pass TelegramCallback to train_model() — it is
    # added automatically inside trainer.train_model() when TG is configured.
"""
import os
import urllib.error
import urllib.parse
import urllib.request

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


def send(text: str) -> None:
    """Send a Telegram message. Silent no-op if TG_BOT_TOKEN / TG_CHAT_ID not set."""
    token = os.getenv("TG_BOT_TOKEN")
    chat_id = os.getenv("TG_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=data,
            timeout=5,
        )
    except Exception:
        pass  # never let a notification failure crash training


class TelegramCallback(TrainerCallback):
    """
    HuggingFace TrainerCallback that sends epoch updates to Telegram.

    Automatically added by trainer.train_model() when TG env vars are set.
    Reports training loss and MRR (when an evaluator is present) after each epoch.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.gpu = os.getenv("CUDA_VISIBLE_DEVICES", "local")

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        epoch = int(state.epoch)
        total = int(args.num_train_epochs)

        # Scan log_history (most recent first) for latest loss and MRR
        loss, mrr = None, None
        for entry in reversed(state.log_history):
            if loss is None and "loss" in entry:
                loss = entry["loss"]
            if mrr is None:
                for k, v in entry.items():
                    if "mrr" in k.lower():
                        mrr = v
                        break
            if loss is not None and mrr is not None:
                break

        lines = [f"📊 [GPU {self.gpu}] {self.label} — Epoch {epoch}/{total}"]
        if loss is not None:
            lines.append(f"Loss: {loss:.4f}")
        if mrr is not None:
            lines.append(f"MRR@10: {mrr:.4f}")

        send("\n".join(lines))
