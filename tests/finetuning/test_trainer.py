"""Tests for finetuning/lib/trainer.py"""
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from finetuning.lib.trainer import train_model

_CONFIG = {
    "model_name": "mock-model",
    "epochs": 2,
    "batch_size": 4,
    "learning_rate": 2e-5,
    "seed": 42,
}
_MORALS = ["Honesty is best.", "Pride comes before a fall."]
_DOCS = ["A boy lied about wolves.", "A fox couldn't reach the grapes."]

# Lazy imports inside train_model() are resolved from sys.modules at call time.
# We patch at the source module level so `from X import Y` inside the function
# picks up the mock.
_LAZY_PATCHES = [
    "sentence_transformers.losses.MultipleNegativesRankingLoss",
    "sentence_transformers.trainer.SentenceTransformerTrainer",
    "sentence_transformers.training_args.SentenceTransformerTrainingArguments",
    "datasets.Dataset",
    "transformers.trainer_utils.get_last_checkpoint",
]


def _training_context(mock_model=None, trainer_side_effect=None):
    """Context manager stack for a full training run (no resume)."""
    stack = ExitStack()
    if mock_model is None:
        mock_model = MagicMock()
    mock_trainer = MagicMock()
    if trainer_side_effect:
        mock_trainer.train.side_effect = trainer_side_effect

    stack.enter_context(patch("finetuning.lib.trainer.SentenceTransformer", return_value=mock_model))
    stack.enter_context(patch(_LAZY_PATCHES[0]))
    stack.enter_context(patch(_LAZY_PATCHES[1], return_value=mock_trainer))
    stack.enter_context(patch(_LAZY_PATCHES[2]))
    stack.enter_context(patch(_LAZY_PATCHES[3]))
    stack.enter_context(patch(_LAZY_PATCHES[4], return_value=None))  # no checkpoint to resume
    return stack, mock_model, mock_trainer


def _call(tmp_path, force=False, **kwargs):
    model_cache = tmp_path / "model"
    checkpoint_dir = tmp_path / "checkpoints"
    return model_cache, checkpoint_dir, dict(
        train_morals=_MORALS,
        train_docs=_DOCS,
        config=_CONFIG,
        model_cache=model_cache,
        checkpoint_dir=checkpoint_dir,
        force=force,
        **kwargs,
    )


# ── Cache hit ─────────────────────────────────────────────────────────────────


def test_cache_hit_loads_model_without_training(tmp_path):
    model_cache, checkpoint_dir, kwargs = _call(tmp_path)
    model_cache.mkdir()

    mock_model = MagicMock()
    with patch("finetuning.lib.trainer.SentenceTransformer", return_value=mock_model) as MockST:
        result = train_model(**kwargs)

    MockST.assert_called_once_with(str(model_cache))
    assert result is mock_model


def test_cache_hit_does_not_call_trainer(tmp_path):
    model_cache, checkpoint_dir, kwargs = _call(tmp_path)
    model_cache.mkdir()

    with (
        patch("finetuning.lib.trainer.SentenceTransformer"),
        patch(_LAZY_PATCHES[1]) as MockTrainer,
    ):
        train_model(**kwargs)

    MockTrainer.assert_not_called()


# ── Cache miss (training) ─────────────────────────────────────────────────────


def test_cache_miss_calls_trainer_train(tmp_path):
    _, _, kwargs = _call(tmp_path)
    stack, _, mock_trainer = _training_context()
    with stack:
        train_model(**kwargs)
    mock_trainer.train.assert_called_once()


def test_cache_miss_saves_model_to_cache(tmp_path):
    model_cache, _, kwargs = _call(tmp_path)
    stack, mock_model, _ = _training_context()
    with stack:
        train_model(**kwargs)
    mock_model.save.assert_called_once_with(str(model_cache))


def test_cache_miss_creates_cache_directory(tmp_path):
    _, _, kwargs = _call(tmp_path)
    kwargs["model_cache"] = tmp_path / "nested" / "model"
    stack, _, _ = _training_context()
    with stack:
        train_model(**kwargs)
    assert kwargs["model_cache"].exists()


def test_cache_miss_returns_trained_model(tmp_path):
    _, _, kwargs = _call(tmp_path)
    stack, mock_model, _ = _training_context()
    with stack:
        result = train_model(**kwargs)
    assert result is mock_model


# ── Checkpointing & resume ────────────────────────────────────────────────────


def test_checkpoint_dir_is_created(tmp_path):
    _, checkpoint_dir, kwargs = _call(tmp_path)
    stack, _, _ = _training_context()
    with stack:
        train_model(**kwargs)
    assert checkpoint_dir.exists()


def test_resumes_from_existing_checkpoint(tmp_path):
    """If a checkpoint exists and model_cache does not, trainer.train() is called with it."""
    model_cache, checkpoint_dir, kwargs = _call(tmp_path)
    checkpoint_dir.mkdir(parents=True)
    fake_ckpt = str(checkpoint_dir / "checkpoint-10")

    stack, _, mock_trainer = _training_context()
    with stack:
        with patch(_LAZY_PATCHES[4], return_value=fake_ckpt):
            train_model(**kwargs)

    mock_trainer.train.assert_called_once_with(resume_from_checkpoint=fake_ckpt)


def test_force_clears_checkpoints_before_training(tmp_path):
    _, checkpoint_dir, kwargs = _call(tmp_path, force=True)
    checkpoint_dir.mkdir(parents=True)
    sentinel = checkpoint_dir / "checkpoint-10" / "some_file.bin"
    sentinel.parent.mkdir(parents=True)
    sentinel.touch()

    stack, _, _ = _training_context()
    with stack:
        train_model(**kwargs)

    assert not sentinel.exists()


# ── Force flag ────────────────────────────────────────────────────────────────


def test_force_retrains_even_when_cache_exists(tmp_path):
    model_cache, _, kwargs = _call(tmp_path, force=True)
    model_cache.mkdir()

    stack, _, mock_trainer = _training_context()
    with stack:
        train_model(**kwargs)

    mock_trainer.train.assert_called_once()


# ── Evaluator ────────────────────────────────────────────────────────────────


def test_evaluator_is_passed_to_trainer(tmp_path):
    """When an evaluator is provided it should be forwarded to SentenceTransformerTrainer."""
    _, _, kwargs = _call(tmp_path)
    mock_evaluator = MagicMock()
    mock_evaluator.name = "fold_0"
    kwargs["evaluator"] = mock_evaluator

    stack, _, _ = _training_context()
    with stack:
        with patch(_LAZY_PATCHES[1]) as MockTrainerCls:
            MockTrainerCls.return_value = MagicMock()
            train_model(**kwargs)

    _, trainer_kwargs = MockTrainerCls.call_args
    assert trainer_kwargs.get("evaluator") is mock_evaluator


def test_evaluator_sets_metric_for_best_model(tmp_path):
    """metric_for_best_model must match the evaluator's MRR metric name."""
    _, _, kwargs = _call(tmp_path)
    mock_evaluator = MagicMock()
    mock_evaluator.name = "fold_2"
    kwargs["evaluator"] = mock_evaluator

    stack, _, _ = _training_context()
    with stack:
        with patch(_LAZY_PATCHES[2]) as MockArgs:
            MockArgs.return_value = MagicMock()
            train_model(**kwargs)

    _, args_kwargs = MockArgs.call_args
    assert args_kwargs.get("metric_for_best_model") == "eval_fold_2_cosine_mrr@10"
    assert args_kwargs.get("greater_is_better") is True


def test_no_evaluator_passes_none_to_trainer(tmp_path):
    _, _, kwargs = _call(tmp_path)  # no evaluator key

    stack, _, _ = _training_context()
    with stack:
        with patch(_LAZY_PATCHES[1]) as MockTrainerCls:
            MockTrainerCls.return_value = MagicMock()
            train_model(**kwargs)

    _, trainer_kwargs = MockTrainerCls.call_args
    assert trainer_kwargs.get("evaluator") is None


# ── LoRA ──────────────────────────────────────────────────────────────────────


_LORA_CFG = {
    "r": 8,
    "alpha": 16,
    "target_modules": ["q_proj", "v_proj"],
    "dropout": 0.05,
}


def _peft_modules(mock_peft_model: MagicMock | None = None):
    """Return a sys.modules patch dict that stubs out peft without installing it."""
    if mock_peft_model is None:
        mock_peft_model = MagicMock()
    mock_peft = MagicMock()
    mock_peft.get_peft_model.return_value = mock_peft_model
    return {"peft": mock_peft}, mock_peft, mock_peft_model


def test_lora_get_peft_model_called_when_configured(tmp_path):
    """When config has a 'lora' block, get_peft_model must be called once."""
    _, _, kwargs = _call(tmp_path)
    kwargs["config"] = {**_CONFIG, "lora": _LORA_CFG}

    mock_peft_model = MagicMock()
    mock_peft_model.merge_and_unload.return_value = MagicMock()
    sys_modules, mock_peft, _ = _peft_modules(mock_peft_model)

    stack, _, _ = _training_context()
    with stack:
        with patch.dict("sys.modules", sys_modules):
            train_model(**kwargs)

    mock_peft.get_peft_model.assert_called_once()
    mock_peft_model.merge_and_unload.assert_called_once()


def test_lora_not_applied_when_not_configured(tmp_path):
    """Without a 'lora' key in config, get_peft_model must not be called."""
    _, _, kwargs = _call(tmp_path)  # _CONFIG has no 'lora' key

    sys_modules, mock_peft, _ = _peft_modules()
    stack, _, _ = _training_context()
    with stack:
        with patch.dict("sys.modules", sys_modules):
            train_model(**kwargs)

    mock_peft.get_peft_model.assert_not_called()


# ── Early stopping ────────────────────────────────────────────────────────────


def test_early_stopping_callback_added_when_patience_set(tmp_path):
    """EarlyStoppingCallback must be passed to trainer when patience is configured."""
    _, _, kwargs = _call(tmp_path)
    mock_evaluator = MagicMock()
    mock_evaluator.name = "fold_0"
    kwargs["evaluator"] = mock_evaluator
    kwargs["config"] = {**_CONFIG, "early_stopping_patience": 3}

    stack, _, _ = _training_context()
    with stack:
        with patch("transformers.EarlyStoppingCallback") as MockESC:
            MockESC.return_value = MagicMock()
            with patch(_LAZY_PATCHES[1]) as MockTrainerCls:
                MockTrainerCls.return_value = MagicMock()
                train_model(**kwargs)

    MockESC.assert_called_once_with(early_stopping_patience=3)
    _, trainer_kwargs = MockTrainerCls.call_args
    assert trainer_kwargs.get("callbacks") is not None


def test_early_stopping_not_added_without_evaluator(tmp_path):
    """Early stopping requires an evaluator — must not be added without one."""
    _, _, kwargs = _call(tmp_path)
    kwargs["config"] = {**_CONFIG, "early_stopping_patience": 3}
    # no evaluator

    stack, _, _ = _training_context()
    with stack:
        with patch("transformers.EarlyStoppingCallback") as MockESC:
            with patch(_LAZY_PATCHES[1]) as MockTrainerCls:
                MockTrainerCls.return_value = MagicMock()
                train_model(**kwargs)

    MockESC.assert_not_called()
    _, trainer_kwargs = MockTrainerCls.call_args
    assert not trainer_kwargs.get("callbacks")


# ── Crash safety ──────────────────────────────────────────────────────────────


def test_no_partial_model_cache_on_crash(tmp_path):
    """Model.save() must not be called if trainer.train() raises."""
    _, _, kwargs = _call(tmp_path)

    stack, mock_model, _ = _training_context(
        trainer_side_effect=RuntimeError("simulated crash")
    )
    with stack, pytest.raises(RuntimeError):
        train_model(**kwargs)

    mock_model.save.assert_not_called()
    assert not kwargs["model_cache"].exists()
