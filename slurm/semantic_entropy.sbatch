#!/bin/bash
# =============================================================================
# Semantic entropy over reason texts (weak_supervision/semantic_entropy.py).
#
# Reads each provider's original answers and entropy/s01..s05 resamples from MinIO
# through an SSH tunnel (same setup as label_vllm.sbatch), clusters them with a
# bidirectional-entailment NLI model on the GPU, and writes
# synthetic-data-v3/entropy/semantic_entropy.parquet. Rerun as resample passes land.
#
#   sbatch slurm/semantic_entropy.sbatch            # all samples
#   LIMIT=200 sbatch slurm/semantic_entropy.sbatch  # smoke test, not uploaded
# =============================================================================

#SBATCH --job-name=la_sem_entropy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sem_entropy_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# shellcheck disable=SC1091
source slurm/cluster.env
export HF_HOME VAULT_ROLE_ID VAULT_SECRET_ID_FILE
[ -n "${VAULT_ROLE_ID:-}" ] || { echo "VAULT_ROLE_ID is empty in slurm/cluster.env"; exit 1; }
for f in "$VAULT_SECRET_ID_FILE" "$LA_VAULT_CACERT" "$LA_SSH_KEY"; do
  [ -r "$f" ] || { echo "missing or unreadable: $f (see slurm/cluster.env)"; exit 1; }
done

JOB="${SLURM_JOB_ID:-local$$}"
BASE=$(( 20000 + ${SLURM_JOB_ID:-$$} % 10000 * 3 ))
MINIO_PORT=$BASE; VAULT_PORT=$((BASE + 1))

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

echo "job=$JOB node=$(hostname) started=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

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
EOF

# The slurm group brings torch and transformers (via vLLM).
uv sync --all-packages --group slurm
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PWD/data_generation:$PWD/weak_supervision"
ARGS=()
if [ -n "${LIMIT:-}" ]; then ARGS+=(--limit "$LIMIT" --no-upload); fi
uv run --all-packages --group slurm --with pandas --with pyarrow --with sentencepiece \
  python weak_supervision/semantic_entropy.py "${ARGS[@]}"

echo "finished $(date)"
