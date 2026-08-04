#!/bin/bash
#SBATCH --job-name=rlad-hinter-pool
# The dispatcher supplies partition, nodelist, and workspace-specific log paths.
#SBATCH --output=/tmp/rlad-hinter-pool-%j.out
#SBATCH --error=/tmp/rlad-hinter-pool-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=1-00:00:00

set -euo pipefail

: "${RLAD_REPO_ROOT:?set by autoresearch.hinter.pool start}"
: "${RLAD_AUTORESEARCH_WORK:?set by autoresearch.hinter.pool start}"
: "${AUTORESEARCH_PYTHON:?set by autoresearch.hinter.pool start}"

cd "${RLAD_REPO_ROOT}"
mkdir -p "${RLAD_AUTORESEARCH_WORK}/logs"

echo "[hinter-pool] allocation=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST}"
echo "[hinter-pool] python=${AUTORESEARCH_PYTHON}"
# This cluster defines no GPU gres, so GPUs come from whole exclusive nodes and
# each task is confined to one GPU by CUDA_VISIBLE_DEVICES, not by Slurm.
# Preflight therefore checks that every node exposes the full expected GPU count
# and that pinning a single index yields exactly one visible device.
srun --exclusive --exact --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --cpus-per-task=6 bash -c '
  count="$(nvidia-smi -L | grep -c "^GPU ")"
  if [[ "$count" != "8" ]]; then
    echo "[hinter-pool] expected 8 GPUs on $(hostname), found ${count}" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES=0 "$AUTORESEARCH_PYTHON" -c "
import torch
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
"
  echo "[hinter-pool] $(hostname): validated ${count} GPUs and one-GPU pinning"
'

exec "${AUTORESEARCH_PYTHON}" -m autoresearch.hinter.pool dispatch
