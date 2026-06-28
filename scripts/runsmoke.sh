#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
python -m behavior2weights.cli doctor
python -m behavior2weights.cli smoke --output "${1:-artifacts/smoke}" --overwrite
