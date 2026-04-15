# MORABLES — Claude Code Project Instructions

## Running Scripts (MANDATORY)

All training and evaluation scripts MUST be run via `run.sh`, never with `python` or `uv run python` directly.

```bash
# Local (default)
./run.sh finetuning/ft_01_5fold_cv/train.py

# Remote GPU server (runs in tmux — survives SSH disconnect)
./run.sh finetuning/ft_01_5fold_cv/train.py --remote

# Remote, specific GPU, with script args
./run.sh finetuning/ft_01_5fold_cv/train.py --fold 0 --doc_mode fable_plus_summary --remote --gpu 2
```

`run.sh` handles: loading `.env`, syncing code to server, installing uv + deps on first run, setting `CUDA_VISIBLE_DEVICES`, and running training in a tmux session.

### Server management subcommands

```bash
./run.sh status              # which GPUs are free
./run.sh watch  [--gpu N]   # stream training output (Ctrl+C detaches, training keeps running)
./run.sh pause  [--gpu N]   # interrupt training (saves last epoch checkpoint automatically)
./run.sh pull               # copy result JSON files from server to local
./run.sh pull --models      # also copy trained model weights
```

### Pause & Resume

Training checkpoints are saved after every epoch. To pause:

1. `./run.sh pause --gpu N` — sends interrupt; training stops after the current epoch
2. Re-run the same `./run.sh ... --remote` command to resume automatically from the last checkpoint

Results stay on the server (`~/ParabeLink/finetuning/*/results/`). Run `./run.sh pull` when you need them locally.

## Telegram Notifications (MANDATORY)

Every training script MUST have Telegram notification support. `run.sh --remote` will warn if it's missing.

**Required pattern for every new script:**

```python
from finetuning.lib import notify

def main():
    # 1. Send start notification with key config info
    notify.send(
        f"🚀 <experiment_name> starting\n"
        f"model: {config['model_name']}\n"
        f"<any other relevant config>"
    )
    # ... training ...
```

**Epoch updates are automatic** — `train_model()` in `finetuning/lib/trainer.py` adds `TelegramCallback` automatically when TG env vars are present. No extra code needed for epoch-level loss/MRR updates.

**For scripts that don't use `train_model()`** (e.g. evaluation-only scripts), call `notify.send()` at start and end with a brief summary of results.

`notify.send()` is always safe to call — it silently no-ops when TG env vars are not set (local runs without config).

## Writing New Training Scripts (MANDATORY)

Keep Python scripts **pure** — no remote/local routing logic inside them. They should:

1. Accept their own args via `argparse` (e.g. `--fold`, `--doc_mode`, `--force`)
2. Run identically whether invoked locally or remotely
3. Save results to their own `results/` subfolder (e.g. `finetuning/ft_XX_name/results/`)

`run.sh` owns all execution routing. A new script needs zero changes to support `--remote`.

## Storage Policy (MANDATORY)

**Physical disk** (`~/ParabeLink`): use during active training only. Space fills up fast with large models — always check before starting.

**Long-term storage** (`/data/lior`): 250GB, use for trained models and artifacts not needed daily.

### Check storage before every training run

```bash
./run.sh status                                      # shows GPU availability
ssh server "df -h ~ && df -h /data/lior"             # physical + long-term free space
```

**Do not start training if physical disk has less than 30GB free.**

### What to keep and where

- **Final models**: saved directly to `/data/lior` via `model_output_dir` in `config.yaml` — never accumulate on physical disk
- **Checkpoints**: stay on physical disk (`~/ParabeLink`) during training for fast I/O; delete after training completes. Ask user for permission before deleting.
- **Embeddings**: small, keep them in `cache/embeddings/`
- **Result JSONs**: always keep, commit to git

### model_output_dir (MANDATORY for new experiments)

Every new experiment's `config.yaml` must set `model_output_dir` to avoid filling the physical disk:

```yaml
model_output_dir: /data/lior/<exp_name>/models
```

Train scripts read this with:
```python
_model_root = Path(config["model_output_dir"]) if config.get("model_output_dir") else CACHE_DIR / "models"
model_cache = _model_root / config["doc_mode"] / f"fold_{fold_idx}"
```

If omitted, models fall back to `cache/models/` — fine for local runs, but always set it for remote GPU runs.

### Cleanup commands EXAMPLES

```bash
# Delete checkpoints for an experiment - must ask user for permission first
ssh $GPU_USER@$GPU_HOST "rm -rf ~/ParabeLink/finetuning/<exp>/cache/checkpoints"
```

## GPU Server

- **Check GPU availability**: `./gpu_status.sh`
- **Coordinate usage**: announce which GPU you're taking in the lab spreadsheet before running
- **Default GPU**: 2 (usually free), override with `--gpu N`
- **Physical vs long-term storage**: train on physical (`~/ParabeLink`), move to `/data/lior` when idle for multiple days

## Project Structure

```
finetuning/
  ft_00_overfit/     — sanity check (single-sample overfit)
  ft_01_5fold_cv/    — 5-fold CV, BGE-base
  ft_02_linq_5fold_cv/ — 5-fold CV, LINQ model
  lib/               — shared trainer, data, eval utilities
```

## Environment

- Package manager: `uv` (pyproject.toml + uv.lock)
- Python: 3.13+
- Credentials: in `.env` (gitignored) — `GPU_HOST`, `GPU_USER`, `GPU_PASSWORD`
