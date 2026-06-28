# Behavior-to-Weights

**Amortized black-box system identification of transformers from behavioral traces.**

This repository implements a research program in which an inverse model observes query/response traces from a target transformer and estimates a posterior over:

1. target identity and lineage;
2. architecture and training properties;
3. controlled circuits or parameter edits;
4. functionally equivalent reconstructed models; and
5. symmetry-aligned parameter coordinates.

The project is designed around a crucial distinction: a transformer's literal parameter array is generally not uniquely identifiable because heads, MLP neurons, and other parameter groups admit exact or approximate symmetries. Primary analyses therefore report **functional reconstruction**, **circuit localization**, and **parameter-orbit recovery**, not just raw coordinate error.

## What is implemented

- A transparent decoder-only micro-transformer with explicit Q/K/V/O matrices.
- Deterministic factorial model-zoo generation across architectures, seeds, datasets, optimizers, training steps, and controlled interventions, plus a no-compute planner that projects exact lineage/target/split/storage counts.
- Black-box observation channels: sampled tokens, repeated-sample histograms, compact top-k index/log-probability pairs, full logits, and seeded random logit sketches. Exact sampled IDs and top-k arrays are persisted as auxiliary trace tensors.
- A set-invariant trace encoder plus a coordinate-conditioned probabilistic hypernetwork decoder.
- Streaming blockwise checkpoint reconstruction; the inverse model never has to emit billions of decimal numbers as text.
- Exact head/MLP permutation transformations, deterministic canonicalization, and Hungarian orbit alignment.
- Active-query baselines based on posterior-predictive disagreement and expected information gain.
- Lineage-disjoint train/validation/test/OOD splits and checksum-addressed manifests.
- Lineage-cluster bootstrap, randomization tests, mixed-effects models, Holm correction, and clustered power simulation.
- Pinned AIM 3.29.1 local/remote tracking plus a dependency-free JSONL tracker.
- Hugging Face dataset preparation, public model adapters, Transformer-NFN model-zoo download wiring, Slurm templates, and AWS Batch placeholders.
- A CPU-sized end-to-end smoke experiment and tests.

## Start here

Python 3.11 is recommended for the fullest optional-dependency compatibility.

```bash
python -m venv .venv
source .venv/bin/activate
# Reproduces the verified CPU/Python 3.13 environment; use pyproject ranges on other platforms.
pip install -r requirements/verified-cpu-py313.lock
pip install -e . --no-deps

# Environment and optional-package check
b2w doctor

# Inspect a production config without launching training
b2w zoo plan \
  --config configs/model_zoo/micro_confirmatory.yaml \
  --output artifacts/plans/micro-confirmatory.json

# Executes zoo -> traces -> inverse training -> reconstruction -> evaluation
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  b2w smoke --output artifacts/smoke --overwrite

pytest
```

The smoke pipeline is an execution check, not a powered scientific result. It intentionally trains tiny models for only a few steps.

## Reproduce a real experiment

```bash
# 0. Refuse accidental over-scale runs: inspect exact projected counts first
b2w zoo plan \
  --config configs/model_zoo/micro_confirmatory.yaml \
  --output artifacts/tier1/zoo-plan.json

# 1. Build target checkpoints and a lineage-safe manifest
b2w zoo build-micro \
  --config configs/model_zoo/micro_confirmatory.yaml \
  --output data/zoo/tier1

# 2. Build a reusable candidate-query bank
b2w probes build \
  --config configs/experiments/tier1_micro.yaml \
  --output data/queries/tier1.jsonl

# 3. Collect all preregistered interfaces into separate immutable stores
b2w traces collect-micro-suite \
  --manifest data/zoo/tier1/targets.jsonl \
  --queries data/queries/tier1.jsonl \
  --experiment configs/experiments/tier1_micro.yaml \
  --output data/traces/tier1

# 4. Train one coordinate-level inverse model per architecture family
b2w train inverse \
  --manifest data/zoo/tier1/targets.jsonl \
  --traces data/traces/tier1/logits \
  --config configs/models/inverse_micro.yaml \
  --tracking-config configs/tracking/aim_remote.yaml \
  --output artifacts/tier1/inverse

# 5. Run the locked target-level evaluation and baselines
b2w evaluate micro \
  --manifest data/zoo/tier1/targets.jsonl \
  --traces data/traces/tier1/logits \
  --checkpoint artifacts/tier1/inverse/best \
  --budgets 8,16,32,64,128 \
  --output artifacts/tier1/evaluation.jsonl
```

For pooled cross-architecture experiments, train the property/circuit heads over the combined trace corpus, then route parameter reconstruction through the predicted architecture-specific decoder. A raw coordinate loss is never computed between incompatible architectures.


### Tier-0 analytic check

The exact softmax gauge and query-rank predictions can be tested before training transformers:

```bash
b2w analytic softmax \
  --config configs/experiments/tier0_softmax_pilot.yaml \
  --output artifacts/tier0-pilot
```

## Experiment tiers

| Tier | Targets | Approximate target size | Purpose |
|---|---:|---:|---|
| 0 | analytic/simulated | linear softmax to one block | prove identifiability limits and gauge freedoms |
| 1 | 1,000s–10,000s | 1K–500K parameters | exact symmetries, exhaustive probes, high-powered circuit studies |
| 2 | 1,000s | 0.5M–30M | TinyStories/WikiText/synthetic language, unseen lineage and architecture tests |
| 3 | 100s | 30M–500M | realistic decoder LMs, low-rank edits, checkpoint trajectories |
| 4 | public suites | 100M–3B+ | external validity, fingerprinting, property inference, functional extraction |

The canonical implementation-to-experiment map is [`docs/RESEARCH_SPECIFICATION.md`](docs/RESEARCH_SPECIFICATION.md). Detailed hypotheses and inferential rules are specified before data collection in [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md) and [`docs/STATISTICAL_ANALYSIS_PLAN.md`](docs/STATISTICAL_ANALYSIS_PLAN.md).

## Repository map

```text
src/behavior2weights/
  models/          target and inverse architectures; weight address spaces
  zoo/             factorial zoo generation, interventions, manifests
  traces/          observation channels, collection, immutable stores
  probes/          fixed and active query policies
  train/           inverse corpus and training loop
  evaluation/      symmetry alignment, functional and calibration metrics
  interpret/       transcript patching and behavior-to-weight attribution
  stats/           clustered inference, multiplicity, and power
  tracking/        AIM and JSONL tracking
  targets/         micro and Hugging Face adapters
  compute/         runtime, Slurm, and smoke orchestration
configs/           tier, data, model, and tracking configurations
infra/             AIM, Slurm, Docker, and AWS deployment assets
docs/              protocol, SAP, runbook, data, security, and references
```

The checked-in plans currently expand to 2,560 independent Tier-1 lineages and 23,040 checkpoint/edit targets; 32 Tier-2 text-pilot lineages and 192 targets; and 19 public-model lineages with 72 revision targets. The planner's byte estimate is raw FP32 unique-parameter payload only and deliberately excludes optimizer state, traces, compression overhead, and training compute.

## Core scientific rules

- **The independent unit is a training lineage**, not a prompt, token, checkpoint, or intervention.
- **All descendants of a lineage remain in one split.** No seed, initialization, checkpoint, edit, or near-duplicate crosses the boundary.
- **Confirmatory test data are frozen and blinded.** Query-policy tuning and threshold selection use only train/validation lineages.
- **Observation interfaces are analyzed separately.** A token-only result cannot be described as logit-level extraction.
- **Weight claims are symmetry-aware.** Report raw, canonicalized, orbit-aligned, and functional metrics together.
- **Security claims are decomposed.** Fingerprinting, property inference, functional extraction, parameter extraction, and memorized-data extraction are different outcomes.
- **All failures remain in the denominator** unless they match a preregistered technical exclusion.

## Data and public model suites

The repository includes executable adapters or manifests for:

- synthetic Markov, copy, induction, and modular-arithmetic corpora;
- `karpathy/tinystories-gpt4-clean`, `roneneldan/TinyStories`, and `Salesforce/wikitext`;
- the Transformer-NFN MNIST and AG News transformer zoos;
- TinyStories 1M/3M/8M/28M/33M models;
- EleutherAI Pythia 14M through 2.8B checkpoint trajectories; and
- Hugging Face SmolLM2 135M/360M/1.7B base/instruct pairs.

Remote datasets and checkpoints are not silently downloaded by tests. Preparation commands write fingerprints, revisions, checksums where available, and license metadata. Production runs must pin immutable commits rather than floating `main` revisions.

A custom Tier-2 language-model zoo is executable from raw text through checkpoints:

```bash
b2w data train-tokenizer \
  --config configs/tokenizers/tinystories_bpe_2048.yaml \
  --output data/tokenizers/tinystories-bpe-2048

b2w data prepare-hf \
  --config configs/datasets/tinystories_clean_bpe2048.yaml \
  --output data/hf/tinystories-clean-bpe2048

b2w zoo build-text \
  --config configs/model_zoo/text_tier2_pilot.yaml \
  --dataset data/hf/tinystories-clean-bpe2048 \
  --output data/zoo/tier2-text
```

## AIM

AIM is optional. The default JSONL tracker always works, including on Python versions for which AIM wheels may lag.

```bash
cd infra/aim
docker compose up -d
export AIM_REPO='aim://localhost:53800'
```

Then pass the tracking configuration to either training command:

```bash
b2w train inverse ... --tracking-config configs/tracking/aim_remote.yaml
b2w train properties ... --tracking-config configs/tracking/aim_remote.yaml
```

See [`docs/AIM.md`](docs/AIM.md).

## Safety and responsible use

This code is intended for controlled research on models you own, have permission to query, or that are explicitly released for research. Do not use it to evade access controls, violate API terms, extract confidential training data, or target third-party systems without authorization. See [`docs/SECURITY_AND_ETHICS.md`](docs/SECURITY_AND_ETHICS.md).

## Citation

See [`CITATION.cff`](CITATION.cff) and [`docs/REFERENCES.md`](docs/REFERENCES.md).
