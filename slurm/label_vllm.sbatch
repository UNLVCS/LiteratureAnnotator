#!/bin/bash
# =============================================================================
# Whole-paper labeling with an open model served by vLLM on the job's GPUs.
#
# The cluster cannot reach vostok's MinIO/Vault ports directly, only SSH, so the
# job opens an SSH tunnel for MinIO (9000) and Vault (8200), serves the model
# with `vllm serve` on localhost, and runs data_generation/labeler.py against it.
# Papers are read from and results written to MinIO on vostok through the tunnel.
#
# One-time setup: slurm/README.md. Submit from the repo root:
#   sbatch slurm/label_vllm.sbatch                         # all papers in RAW_PREFIX
#   LIMIT=30 sbatch slurm/label_vllm.sbatch                # first 30
#   PMIDS_FILE=pilot30.txt sbatch slurm/label_vllm.sbatch  # a specific list
# Reruns skip papers already labeled by the model, so a killed job can resubmit.
# =============================================================================

#SBATCH --job-name=la_label
#SBATCH --partition=gpu                # UNLV HPC: one node, csgpu-app-002 (2 GPUs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2                   # TP defaults to the GPUs allocated
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/label_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# shellcheck disable=SC1091
source slurm/cluster.env
export HF_HOME VAULT_ROLE_ID VAULT_SECRET_ID_FILE

# Without Vault the config silently falls back to .env.yaml, which has no MinIO keys.
[ -n "${VAULT_ROLE_ID:-}" ] || { echo "VAULT_ROLE_ID is empty in slurm/cluster.env"; exit 1; }
for f in "$VAULT_SECRET_ID_FILE" "$LA_VAULT_CACERT" "$LA_SSH_KEY"; do
  [ -r "$f" ] || { echo "missing or unreadable: $f (see slurm/cluster.env)"; exit 1; }
done
# The model cache must be writable on the compute node (gpt-oss-120b needs ~65 GB).
mkdir -p "$HF_HOME" 2>/dev/null && [ -w "$HF_HOME" ] \
  || { echo "HF_HOME=$HF_HOME is not writable on $(hostname); set HF_HOME in slurm/cluster.env"; exit 1; }
df -h "$HF_HOME" | tail -1

MODEL="${MODEL:-openai/gpt-oss-120b}"
TP="${TP:-${SLURM_GPUS_ON_NODE:-2}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# FlashInfer's attention kernels must be JIT-compiled for SM 12.x (RTX PRO 6000) and
# that build fails here; Triton attention supports gpt-oss and needs no nvcc build.
# ATTN_BACKEND=auto lets vLLM choose.
ATTN_BACKEND="${ATTN_BACKEND:-TRITON_ATTN}"
RAW_PREFIX="${RAW_PREFIX:-pilot_2026_10}"
JOB="${SLURM_JOB_ID:-local$$}"

# Job-specific local ports, so several jobs can share a node.
BASE=$(( 20000 + ${SLURM_JOB_ID:-$$} % 10000 * 3 ))
MINIO_PORT=$BASE; VAULT_PORT=$((BASE + 1)); VLLM_PORT=$((BASE + 2))

wait_port() {  # wait_port <port> <seconds>
  for _ in $(seq 1 "$2"); do
    (echo > "/dev/tcp/127.0.0.1/$1") 2>/dev/null && return 0
    sleep 1
  done
  return 1
}

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

echo "job=$JOB node=$(hostname) model=$MODEL tp=$TP attn=$ATTN_BACKEND prefix=$RAW_PREFIX started=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# --- 1. SSH tunnel to vostok -------------------------------------------------
ssh -N -i "$LA_SSH_KEY" \
    -o BatchMode=yes -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
    -L "$MINIO_PORT:localhost:9000" -L "$VAULT_PORT:localhost:8200" \
    "$LA_TUNNEL_HOST" &
PIDS+=($!)
wait_port "$MINIO_PORT" 30 || { echo "SSH tunnel did not come up"; exit 1; }

export VAULT_ADDR="https://localhost:$VAULT_PORT" VAULT_CACERT="$LA_VAULT_CACERT"
export LA_CONFIG_OVERRIDE="logs/override_$JOB.yaml"
cat > "$LA_CONFIG_OVERRIDE" <<EOF
minio:
  url: "localhost:$MINIO_PORT"
llm_providers:
  vllm:
    models:
      - model: "$MODEL"
        base_url: "http://localhost:$VLLM_PORT/v1"
        temperature: 0.1
        max_tokens: ${MAX_TOKENS:-8000}
        skip: false
EOF

# --- 2. Environment ------------------------------------------------------------
uv sync --all-packages --group slurm

# FlashInfer reads the CUDA version from the toolkit (CUDA_HOME, else /usr/local/cuda),
# not from torch. The node's system toolkit is older than 12.9, so FlashInfer disables
# its Blackwell (SM 12.x) kernels and gpt-oss fails to load. Use the pip CUDA toolkit
# that vLLM installs into the venv instead.
NVCC=$(find .venv/lib/python3*/site-packages/nvidia -path '*/bin/nvcc' -type f 2>/dev/null | head -1)
[ -n "$NVCC" ] || { echo "no pip nvcc in .venv (nvidia-cuda-nvcc); FlashInfer would use the old system CUDA"; exit 1; }
export CUDA_HOME="$(cd "$(dirname "$NVCC")/.." && pwd)"
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME ($("$NVCC" --version | grep -o 'release [0-9.]*'))"

# Fail fast (before loading a 65 GB model) if MinIO is not usable through the tunnel.
uv run --all-packages --group slurm python - <<'EOF'
from minio import Minio
from config import load_app_config
c = load_app_config()
m = Minio(c.minio.url, access_key=c.minio.access_key, secret_key=c.minio.secret_key, secure=c.minio.secure)
assert m.bucket_exists(c.minio.raw_articles_bucket), "raw articles bucket not found"
print("MinIO OK via", c.minio.url)
EOF

# --- 3. Model server -----------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ATTN_ARGS=(); [ "$ATTN_BACKEND" = auto ] || ATTN_ARGS=(--attention-backend "$ATTN_BACKEND")
export PYTHONUNBUFFERED=1   # the log is a file, so Python would otherwise buffer it in 8 KB blocks
uv run --all-packages --group slurm vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$VLLM_PORT" \
    --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    "${ATTN_ARGS[@]}" \
    > "logs/vllm_$JOB.log" 2>&1 &
PIDS+=($!)
echo "waiting for vLLM (log: logs/vllm_$JOB.log)..."
wait_port "$VLLM_PORT" 3600 || { echo "vLLM server did not start"; tail -50 "logs/vllm_$JOB.log"; exit 1; }
until curl -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null; do sleep 5; done
echo "vLLM ready $(date)"

# --- 4. Label --------------------------------------------------------------------
ARGS=(--providers "$MODEL" --raw-prefix "$RAW_PREFIX" --upload --skip-existing --out "logs/results_$JOB")
if [ -n "${PMIDS_FILE:-}" ]; then ARGS+=(--pmids-file "$PMIDS_FILE"); else ARGS+=(--from-prefix "$RAW_PREFIX"); fi
if [ -n "${LIMIT:-}" ]; then ARGS+=(--limit "$LIMIT"); fi

uv run --all-packages --group slurm python data_generation/labeler.py "${ARGS[@]}"

echo "finished $(date)"
