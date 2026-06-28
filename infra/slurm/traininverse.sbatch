#!/usr/bin/env bash
#SBATCH --job-name=b2w-inverse
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
: "${B2W_MANIFEST:?set B2W_MANIFEST}"
: "${B2W_TRACES_ROOT:?set B2W_TRACES_ROOT}"
: "${B2W_CONFIG:?set B2W_CONFIG}"
: "${B2W_OUTPUT:?set B2W_OUTPUT}"
source "${B2W_VENV:-.venv}/bin/activate"
channels=(tokens sample_histogram topk logits logit_sketch)
channel="${channels[$SLURM_ARRAY_TASK_ID]}"
seed="${B2W_SEED:-20260621}"
args=(
  b2w train inverse
  --manifest "$B2W_MANIFEST"
  --traces "$B2W_TRACES_ROOT/$channel"
  --config "$B2W_CONFIG"
  --output "$B2W_OUTPUT/$channel/seed-$seed"
  --seed "$seed"
  --device cuda
  --run-name "$channel-seed-$seed"
)
if [[ -n "${B2W_TRACKING_CONFIG:-}" ]]; then
  args+=(--tracking-config "$B2W_TRACKING_CONFIG")
fi
"${args[@]}"
