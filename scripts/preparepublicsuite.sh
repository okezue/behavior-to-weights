#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/public data/queries
b2w zoo build-public --config configs/model_zoo/public_models.yaml --output data/public/targets.jsonl
# Choose a tokenizer family and pin an immutable revision before running the next line.
b2w probes build-hf --prompts data/query_templates/general_prompts.jsonl \
  --tokenizer EleutherAI/gpt-neox-20b --sequence-length 128 \
  --output data/queries/pythia.jsonl
