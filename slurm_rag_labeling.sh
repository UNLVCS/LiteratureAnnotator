#!/bin/bash
# =============================================================================
# SLURM job: RAG labeling with vLLM native (offline inference)
# Node:      falcon9  (LocalQ partition, 4× A100)
# =============================================================================

#SBATCH --job-name=rag_labeling
#SBATCH --partition=LocalQ
#SBATCH --nodelist=falcon9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # headroom for multiprocessing workers
#SBATCH --gres=gpu:4                # all 4 A100s (tensor_parallel_size=4)
#SBATCH --mem=64G
#SBATCH --time=UNLIMITED
#SBATCH --output=logs/rag_labeling_%j.out
#SBATCH --error=logs/rag_labeling_%j.err

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
echo "Started:    $(date)"
echo "Project:    $PROJECT_ROOT"
echo "============================================================"

# Verify GPUs are visible
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------
# uv: sync the workspace + slurm dependency group into .venv/
#
# --all-packages: install every workspace member's dependencies (minio, redis,
# pinecone, langchain, …), not just those reachable from the virtual root.
# --group slurm: also install vllm + pyyaml from [dependency-groups.slurm].
# On re-runs this is a fast no-op if the lock file has not changed.
# ---------------------------------------------------------------------------
uv sync --all-packages --group slurm

# vLLM spawns tensor-parallel sub-processes; spawn avoids CUDA fork issues.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ---------------------------------------------------------------------------
# Run via uv so the workspace .venv is used automatically
# ---------------------------------------------------------------------------
uv run --all-packages --group slurm python data_generation/rag_labeling_script_mp.py

echo "============================================================"
echo "Finished:   $(date)"
echo "============================================================"
