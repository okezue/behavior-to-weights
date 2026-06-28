#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-$root/dist/behavior-to-weights-repo.zip}"
mkdir -p "$(dirname "$output")"
rm -f "$output"

(
  cd "$root"
  zip -q -r "$output" . \
    -x '.git/*' \
       '.pytest_cache/*' \
       '.ruff_cache/*' \
       '.mypy_cache/*' \
       '**/__pycache__/*' \
       '*.pyc' \
       '*.pyo' \
       '*.egg-info/*' \
       '.venv/*' \
       'dist/*'
)
printf '%s\n' "$output"
