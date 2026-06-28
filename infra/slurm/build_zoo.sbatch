#!/usr/bin/env bash
#SBATCH --job-name=b2w-zoo
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${B2W_CONFIG:?set B2W_CONFIG}"
: "${B2W_OUTPUT:?set B2W_OUTPUT}"
module load cuda 2>/dev/null || true
source "${B2W_VENV:-.venv}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

plan_path="${B2W_PLAN_OUTPUT:-${B2W_OUTPUT}.plan.json}"
b2w zoo plan --config "$B2W_CONFIG" --kind micro --output "$plan_path"
b2w zoo build-micro --config "$B2W_CONFIG" --output "$B2W_OUTPUT" --device cuda
