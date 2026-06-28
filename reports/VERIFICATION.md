# Verification Report

**Status:** validated locally on CPU on June 21, 2026.

This report distinguishes executable validation from scientific evidence. The checked-in smoke runs prove that the repository's control flow, schemas, storage contracts, reconstruction path, and analysis utilities execute; they are not powered estimates of the research hypotheses.

## Automated checks

- `pytest -q`: **40 passed, 1 deselected**, with two non-fatal PyTorch nested-tensor warnings.
- `ruff check src tests scripts infra/aws`: **passed**.
- `mypy src/behavior2weights`: **passed; no issues in 62 source files**.
- Python byte-compilation, shell/Sbatch syntax, and YAML/JSON/JSONL parsing: **passed**.
- Editable package installation and `b2w` CLI discovery: **passed**.
- Reference study lock: **all 4 frozen-file SHA-256 values match**.
- Canonical claim-to-artifact mapping: `docs/RESEARCH_SPECIFICATION.md`.
- Exact verified CPU/Python 3.13 dependency closure: `requirements/verified-cpu-py313.lock` (GPU/HF/AIM environments require separate immutable locks).

The codebase contains 62 package Python files (8,078 source lines) and 15 test modules (917 test lines). This exported artifact is not itself a Git working tree, so a commit SHA is unavailable; production studies must freeze a Git commit or container digest.

## End-to-end smoke execution

The final CPU run created 24 target checkpoints/edits from 8 lineages, collected a trace tensor of shape `[24, 32, 16]`, trained the inverse model for 12 steps, reconstructed a held-out target, and evaluated both parameter and functional error.

- inverse weight NRMSE: `0.108605571`
- population-mean checkpoint NRMSE: `0.119980544`
- held-out functional forward KL: `0.001120552`
- status: `completed`

These numbers are execution diagnostics only. The smoke test has too few independent test lineages and too little training to support significance or performance claims.

## Observation-channel validation

The final model zoo was recollected through all five channels: tokens, sample_histogram, topk, logits, logit_sketch. Every trace store passed checksum, target/query-order, tensor-shape, and finite-value validation. Exact sampled token IDs and exact top-k indices/log-probabilities were preserved in auxiliary tensors. Full logits were stored at vocabulary width; the logit-sketch channel used its declared seeded projection.

## Analytic Tier 0

The linear-softmax experiment completed 180 independent systems across 36 factorial cells. It exercises full-rank exact orbit recovery, rank-deficient designs, exact probability observations, and finite sampled-output observations.

## Preflight design projections

| Design | Independent lineages | Targets | Raw FP32 checkpoint payload | Split summary |
|---|---:|---:|---:|---|
| Tier 1 micro confirmatory | 2,560 | 23,040 | 2.057 GiB | {'ood': 0, 'test': 512, 'train': 1792, 'validation': 256} |
| Tier 2 text pilot | 32 | 192 | 6.047 GiB | {'ood': 0, 'test': 6, 'train': 23, 'validation': 3} |
| Public external-validity suite | 19 | 72 | 172.614 GiB | all observational targets marked OOD |

The Tier-2 text configuration is explicitly a pipeline/variance pilot, not a powered confirmatory design. The Tier-1 design exceeds the SAP's pooled minimum of 144 evaluable test lineages; checkpoints, prompts, edits, and inverse seeds remain nested repeated measurements rather than independent samples.

## Deliberately unexecuted external work

This environment had no CUDA device and did not contain AIM, Transformers, Datasets, or PyArrow. Consequently, the following are implemented and documented but were not run here: Hugging Face dataset/checkpoint downloads, full Tier-1+ GPU experiments, AWS Batch and Slurm jobs, self-hosted AIM runtime integration, and distributed/multi-node execution. Those paths require organization-specific credentials, quotas, storage, license review, immutable Hub revisions, and scheduler configuration.

## Reproduction commands

```bash
pip install -e '.[dev]'
pytest -q
ruff check src tests scripts infra/aws
mypy src/behavior2weights
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 b2w smoke --output artifacts/final-smoke --overwrite
b2w traces collect-micro-suite \
  --manifest artifacts/final-smoke/zoo/targets.jsonl \
  --queries artifacts/final-smoke/queries.jsonl \
  --experiment configs/experiments/channel_smoke.yaml \
  --output artifacts/final-smoke/traces-suite
b2w analytic softmax \
  --config configs/experiments/tier0_softmax_smoke.yaml \
  --output artifacts/final-tier0-smoke
```

Machine-readable details are in `reports/verification.json`; raw test/lint/type-check logs are stored beside this report.
