#!/usr/bin/env bash
#SBATCH --job-name=b2w-traces
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${B2W_MANIFEST:?set B2W_MANIFEST}"
: "${B2W_QUERIES:?set B2W_QUERIES}"
: "${B2W_EXPERIMENT:?set B2W_EXPERIMENT}"
: "${B2W_OUTPUT:?set B2W_OUTPUT}"
source "${B2W_VENV:-.venv}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# The suite command reads every channel's exact codec settings from the frozen YAML
# and writes one immutable sub-store plus a suite manifest.
b2w traces collect-micro-suite \
  --manifest "$B2W_MANIFEST" \
  --queries "$B2W_QUERIES" \
  --experiment "$B2W_EXPERIMENT" \
  --output "$B2W_OUTPUT" \
  --base-seed "${B2W_BASE_SEED:-0}" \
  --device cuda
