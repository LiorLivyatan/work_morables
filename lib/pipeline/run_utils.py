"""lib/pipeline/run_utils.py — Run directory management, .env loading, manifest I/O."""
import json
import os
from datetime import datetime
from pathlib import Path


def load_env(root_dir: Path) -> None:
    """Load KEY=VALUE pairs from root_dir/.env into os.environ (skips existing keys)."""
    env_path = Path(root_dir) / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.split("#")[0].strip()
            os.environ.setdefault(k.strip(), v)


def make_run_dir(base_dir: Path, tag: str = "") -> Path:
    """Create and return a timestamped run directory: base_dir/YYYY-MM-DD_HH-MM-SS[_tag]/"""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{ts}_{tag}" if tag else ts
    run_dir = Path(base_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_latest_run_dir(base_dir: Path) -> Path:
    """Return the lexicographically last subdirectory in base_dir."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    dirs = sorted(d for d in base_dir.iterdir() if d.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No subdirectories in {base_dir}")
    return dirs[-1]


def write_manifest(run_dir: Path, step: str, config_snapshot: dict) -> None:
    """Write/update run_manifest.json, recording that `step` completed."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "run_dir": str(run_dir),
            "steps_completed": [],
            "config_snapshot": config_snapshot,
        }

    if step not in manifest["steps_completed"]:
        manifest["steps_completed"].append(step)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def read_manifest(run_dir: Path) -> dict:
    """Read and return run_manifest.json from run_dir."""
    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")
    with open(manifest_path) as f:
        return json.load(f)
