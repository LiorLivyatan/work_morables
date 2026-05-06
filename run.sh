#!/usr/bin/env bash
# ============================================================
#  run.sh — local/remote launcher for ParabeLink
#
#  TRAINING
#    ./run.sh <script> [script_args...]              # local (default)
#    ./run.sh <script> [script_args...] --remote     # remote, GPU 2
#    ./run.sh <script> [script_args...] --remote --gpu 3
#
#  SUBCOMMANDS
#    ./run.sh status              # GPU status (which are free)
#    ./run.sh watch  [--gpu N]   # stream training output (Ctrl+C to detach, training keeps going)
#    ./run.sh pause  [--gpu N]   # interrupt training (saves last epoch checkpoint)
#    ./run.sh pull   [--gpu N]   # copy results from server to local
#    ./run.sh pull --models      # also copy trained model weights
#
#  EXAMPLES
#    ./run.sh finetuning/ft_01_5fold_cv/train.py --fold 0 --remote --gpu 2
#    ./run.sh watch --gpu 2
#    ./run.sh pull
# ============================================================

set -euo pipefail

# ── Load .env ────────────────────────────────────────────────
ENV_FILE="$(dirname "$0")/.env"
[[ -f "$ENV_FILE" ]] && set -a && source "$ENV_FILE" && set +a

REMOTE_DIR="${REMOTE_DIR:-~/ParabeLink}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

ssh_run()     { sshpass -p "$GPU_PASSWORD" ssh    $SSH_OPTS      "$GPU_USER@$GPU_HOST" "$@"; }
ssh_run_tty() { sshpass -p "$GPU_PASSWORD" ssh -t $SSH_OPTS      "$GPU_USER@$GPU_HOST" "$@"; }
rsync_to()    { sshpass -p "$GPU_PASSWORD" rsync  -az --progress -e "ssh $SSH_OPTS" "$@"; }
rsync_from()  { sshpass -p "$GPU_PASSWORD" rsync  -az --progress -e "ssh $SSH_OPTS" "$@"; }

# ── Parse subcommand or script ────────────────────────────────
SUBCMD=""
case "${1:-}" in
  status|watch|pause|pull) SUBCMD="$1"; shift ;;
esac

# ── Parse flags ───────────────────────────────────────────────
REMOTE=false
GPU=2
PULL_MODELS=false
SCRIPT=""
PYTHON_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)   REMOTE=true;          shift ;;
    --local)    REMOTE=false;         shift ;;
    --gpu)      GPU="$2";             shift 2 ;;
    --models)
      if [[ "$SUBCMD" == "pull" ]]; then PULL_MODELS=true
      else PYTHON_ARGS+=("--models")
      fi
      shift ;;
    *)
      if [[ -z "$SCRIPT" ]]; then SCRIPT="$1"
      else PYTHON_ARGS+=("$1")
      fi
      shift ;;
  esac
done

GPU_TAG="$(echo "$GPU" | tr ',' '_')"
SESSION="parabelink_gpu${GPU_TAG}"
LOG="/tmp/parabelink_gpu${GPU_TAG}.log"

# ── SUBCOMMANDS ───────────────────────────────────────────────

if [[ "$SUBCMD" == "status" ]]; then
  echo "Connecting to $GPU_HOST..."
  ssh_run "
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,temperature.gpu \
      --format=csv,noheader,nounits | \
      awk -F', ' '{
        used=\$3; free=\$4; total=used+free
        printf \"GPU %s | %-22s | Mem: %5d/%5d MB (%2d%%) | Util: %3d%% | Temp: %s°C\n\", \$1, \$2, used, total, (used/total)*100, \$5, \$6
      }'
    echo ''
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader 2>/dev/null \
      | head -10 || echo 'No processes running'
  "
  exit 0
fi

if [[ "$SUBCMD" == "watch" ]]; then
  echo "[watch] Streaming GPU $GPU training output (Ctrl+C to detach — training keeps running)"
  echo "─────────────────────────────────────────────────────────"
  # tail -f streams the log; Ctrl+C kills only the tail, not the server-side tmux session
  ssh_run "tail -f $LOG 2>/dev/null || echo 'No active training log on GPU $GPU (session: $SESSION)'"
  exit 0
fi

if [[ "$SUBCMD" == "pause" ]]; then
  echo "[pause] Sending interrupt to GPU $GPU training..."
  echo "        Training will stop after the current epoch completes (checkpoint saved)."
  ssh_run "tmux send-keys -t '$SESSION' C-c 2>/dev/null && echo 'Interrupt sent.' || echo 'No active session on GPU $GPU (session: $SESSION)'"
  exit 0
fi

if [[ "$SUBCMD" == "pull" ]]; then
  echo "[pull] Syncing results from server (GPU $GPU)..."
  rsync_from \
    --include='*/' \
    --include='results/***' \
    --include='data/***' \
    --exclude='*' \
    "$GPU_USER@$GPU_HOST:$REMOTE_DIR/finetuning/" \
    "$PROJECT_ROOT/finetuning/"
  if [[ "$PULL_MODELS" == true ]]; then
    echo "[pull] Syncing model weights..."
    rsync_from \
      --include='*/' \
      --include='cache/models/***' \
      --exclude='*' \
      "$GPU_USER@$GPU_HOST:$REMOTE_DIR/finetuning/" \
      "$PROJECT_ROOT/finetuning/"
  fi
  echo "Done. Results are in finetuning/*/results/"
  exit 0
fi

# ── TRAINING ──────────────────────────────────────────────────

if [[ -z "$SCRIPT" ]]; then
  echo "Usage:"
  echo "  ./run.sh <script> [script_args...] [--remote] [--gpu N]"
  echo "  ./run.sh status | watch [--gpu N] | pause [--gpu N] | pull [--models]"
  exit 1
fi

# ── LOCAL ─────────────────────────────────────────────────────
if [[ "$REMOTE" == false ]]; then
  echo "[local] uv run python $SCRIPT ${PYTHON_ARGS[*]:-}"
  uv run python "$SCRIPT" "${PYTHON_ARGS[@]:-}"
  exit $?
fi

# ── REMOTE ────────────────────────────────────────────────────
echo "[remote] GPU $GPU on $GPU_HOST  (session: $SESSION)"
echo ""

# 1. Sync code + data (never syncs results or model cache from local → server)
echo "[1/3] Syncing project to server..."
rsync_to \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='.venv/' \
  --exclude='venv_gen/' \
  --exclude='wandb/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.env' \
  --exclude='finetuning/*/results/' \
  --exclude='experiments/*/results/' \
  --exclude='finetuning/*/cache/models/' \
  --exclude='finetuning/*/cache/checkpoints/' \
  --exclude='finetuning/*/cache/embeddings/' \
  --exclude='.claude/' \
  "$PROJECT_ROOT/" \
  "$GPU_USER@$GPU_HOST:$REMOTE_DIR/"
echo "   Done."

# 2. Ensure uv + Python 3.13 + deps
echo "[2/3] Checking server environment..."
ssh_run "
  set -e
  if ! command -v uv &>/dev/null && [[ ! -f ~/.local/bin/uv ]]; then
    echo '   Installing uv...'
    curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet
  fi
  export PATH=\"\$HOME/.local/bin:\$PATH\"
  cd $REMOTE_DIR
  if [[ ! -d .venv ]]; then
    echo '   Creating venv (first time, may take a minute)...'
    uv sync --quiet 2>&1 | tail -5
  else
    uv sync --quiet 2>&1 | tail -2
  fi
  echo '   Environment ready.'
"

# 3. Launch in tmux, tail log locally
echo "[3/3] Launching on GPU $GPU in tmux session '$SESSION'..."
echo "      Ctrl+C detaches you — training keeps running on the server."
echo "      To reattach:  ./run.sh watch --gpu $GPU"
echo "      To interrupt: ./run.sh pause --gpu $GPU"
echo "─────────────────────────────────────────────────────────"
echo ""

ARGS_STR="${PYTHON_ARGS[*]:-}"
WANDB_KEY="${WANDB_API_KEY:-}"
TG_TOKEN="${TG_BOT_TOKEN:-}"
TG_CHAT="${TG_CHAT_ID:-}"
TRAIN_SCRIPT="/tmp/${SESSION}.sh"

# Verify the script has Telegram notification support
if ! grep -q "notify" "$SCRIPT" 2>/dev/null; then
  echo "⚠️  Warning: $SCRIPT is missing Telegram notification support."
  echo "   Add 'from finetuning.lib import notify' and a notify.send() call."
  echo "   See CLAUDE.md — TG notifications are mandatory for remote runs."
  echo ""
fi

# Write the training + notification script to the server.
# Variables expanded locally ($GPU, $SCRIPT, etc.) are injected at write time.
# Variables escaped with \ ($EXIT, $RESULT, etc.) are expanded on the server at run time.
ssh_run "cat > $TRAIN_SCRIPT" << EOF
#!/usr/bin/env bash
set -o pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export WANDB_API_KEY="${WANDB_KEY}"
export TG_BOT_TOKEN="${TG_TOKEN}"
export TG_CHAT_ID="${TG_CHAT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-/data/lior/hf_cache}"
export PYTHONUNBUFFERED=1
cd ${REMOTE_DIR}

CUDA_VISIBLE_DEVICES=${GPU} uv run python ${SCRIPT} ${ARGS_STR} 2>&1 | tee ${LOG}
EXIT=\${PIPESTATUS[0]}

RESULT=\$(grep -oE 'Final MRR: [0-9.]+' ${LOG} | tail -1)
if [ "\$EXIT" -eq 0 ]; then
  TEXT="✅ [ParabeLink GPU ${GPU}] Done: ${SCRIPT} \${RESULT:+— \$RESULT}"
else
  TEXT="❌ [ParabeLink GPU ${GPU}] Failed: ${SCRIPT}"
fi

curl -s "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
  -d "chat_id=${TG_CHAT}" \
  --data-urlencode "text=\$TEXT" > /dev/null 2>&1 || true

echo "--- DONE ---" >> ${LOG}
EOF

ssh_run "
  chmod +x $TRAIN_SCRIPT
  tmux kill-session -t '$SESSION' 2>/dev/null || true
  > $LOG
  tmux new-session -d -s '$SESSION' '$TRAIN_SCRIPT'
  echo 'Session started. Streaming output...'
  sleep 1
"

# Stream output until Ctrl+C
ssh_run "tail -f $LOG" || true

echo ""
echo "─────────────────────────────────────────────────────────"
echo "Training is still running on the server."
echo "  Watch:  ./run.sh watch --gpu $GPU"
echo "  Pause:  ./run.sh pause --gpu $GPU"
echo "  Pull results when done:  ./run.sh pull"
