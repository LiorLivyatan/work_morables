"""Load and validate config.yaml. Raises ConfigError on any structural issue."""
from pathlib import Path
from typing import Any
import yaml


class ConfigError(ValueError):
    pass


REQUIRED_TOP_LEVEL = (
    "model", "data", "discovery", "concepts", "vectors",
    "intervention", "null_controls", "eval", "output",
)
REQUIRED_MODEL = ("hf_id", "pooling", "device", "dtype", "batch_size")
VALID_RUN_MODES = {"candidate_only", "full", "skip"}


def load_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ConfigError(f"missing top-level keys: {missing}")

    missing_model = [k for k in REQUIRED_MODEL if k not in cfg["model"]]
    if missing_model:
        raise ConfigError(f"missing model keys: {missing_model}")

    if 0.0 not in cfg["intervention"]["alphas"]:
        raise ConfigError("intervention.alphas must include 0.0 (no-op baseline)")

    rm = cfg["null_controls"]["run_mode"]
    if rm not in VALID_RUN_MODES:
        raise ConfigError(
            f"null_controls.run_mode must be one of {sorted(VALID_RUN_MODES)}, got {rm!r}"
        )
