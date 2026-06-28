# Behavior-to-Weights Research Specification

## 1. Purpose

This repository studies **amortized black-box system identification of transformers**. A learned inverse model receives a finite behavioral transcript

\[
T=\{(x_i,o_i)\}_{i=1}^n
\]

from a target model and estimates a posterior over target properties, mechanisms, functional behavior, and symmetry-equivalent parameter coordinates. The intended scientific output is not merely a checkpoint prediction; it is a measured hierarchy of what internal information is identifiable from each black-box interface and query budget.

This specification is the canonical index for the implementation. Detailed rationale and analysis rules live in:

- `docs/EXPERIMENT_DESIGN.md` — hypotheses, tiers, endpoints, and stage gates;
- `docs/STATISTICAL_ANALYSIS_PLAN.md` — estimands, power, inference, multiplicity, and failures;
- `docs/ARCHITECTURE.md` — software/model architecture and storage contracts;
- `docs/RUNBOOK.md` — execution order and frozen-study workflow;
- `docs/MODEL_MATRIX.md` — target populations and held-out panels;
- `docs/SECURITY_AND_ETHICS.md` — authorized-use boundary and disclosure policy.

## 2. Scope and non-goals

### In scope

1. Identify target lineage, architecture, training factors, checkpoint stage, and controlled interventions.
2. Reconstruct target behavior on unseen prompts.
3. Recover parameter coordinates within a declared compatible architecture and after exact symmetry alignment.
4. Learn active probes that reduce posterior uncertainty using only training/validation lineages.
5. trace which transcript evidence causes the inverse model to infer a target mechanism or parameter block.
6. quantify interface leakage, query efficiency, uncertainty calibration, and generalization to held-out factors/families.

### Out of scope

- claiming a unique literal checkpoint where parameter symmetries make it undefined;
- treating prompts, tokens, checkpoints, or parameter coordinates as independent statistical samples;
- bypassing model-access controls or extracting systems without authorization;
- inferring memorized training examples merely because functional or parameter extraction succeeds;
- comparing raw coordinate vectors across incompatible tensor shapes.

## 3. Identifiability target

The primary parameter object is an equivalence class \([\theta]\), not an arbitrary storage ordering. Exact head and MLP-neuron permutations are implemented for the transparent transformer. Evaluation reports:

1. raw storage-space error;
2. deterministic-canonicalization error;
3. Hungarian orbit-aligned error;
4. functional divergence on unseen inputs.

For finite transcripts, the inverse estimator is probabilistic:

\[
q_\phi([\theta], z_{\mathrm{properties}}, z_{\mathrm{circuits}}\mid T).
\]

Posterior quality is measured by held-out NLL, interval coverage, standardized residuals, sharpness, and calibration stratified by tensor role/layer and OOD panel.

## 4. Independent unit and split invariant

A **lineage** is one initialization and training trajectory. All checkpoints, interventions, quantized variants, and edited descendants of that trajectory remain in one split. Lineage ID—not target ID, prompt, response, or inverse seed—is the independent unit for confirmatory inference.

Split assignment is deterministic and hash-sorted under a frozen salt. Named OOD factor values are assigned before the remaining train/validation/test allocation. Every analysis must verify that one lineage occurs in exactly one split.

Implementation:

- schema: `src/behavior2weights/schemas.py`;
- split policy and manifest validation: `src/behavior2weights/zoo/manifest.py`;
- target planners: `src/behavior2weights/zoo/planning.py`;
- frozen reference lock: `artifacts/reference-study.lock.json`.

## 5. Target populations

### Tier 0 — analytic identifiability

Systems: linear-softmax models with exact additive gauge removal, varied query rank, exact probabilities, and finite sampled outputs.

- smoke: `configs/experiments/tier0_softmax_smoke.yaml`;
- pilot: `configs/experiments/tier0_softmax_pilot.yaml`;
- confirmatory: `configs/experiments/tier0_softmax_confirmatory.yaml`;
- implementation: `src/behavior2weights/analytic/softmax.py`.

Confirmatory cells use 1,000 independent systems. The primary check is agreement between design rank, observable dimension, exact orbit recovery, and held-out functional divergence.

### Tier 1 — transparent micro-transformers

The checked-in confirmatory population contains:

- 4 architectures;
- 4 synthetic tasks;
- 40 initialization seeds;
- 4 dataset/order seeds;
- 1 controlled optimizer setting;
- 5 checkpoints and 4 controlled edits per lineage.

This expands to 2,560 independent lineages and 23,040 nested targets, with 512 deterministic test lineages before any named OOD reservation.

Files:

- zoo: `configs/model_zoo/micro_confirmatory.yaml`;
- experiment: `configs/experiments/tier1_micro.yaml`;
- target model: `src/behavior2weights/models/microtransformer.py`;
- zoo builder: `src/behavior2weights/zoo/micro.py`;
- exact interventions: `src/behavior2weights/zoo/interventions.py`;
- preflight projection: `artifacts/plans/micro-confirmatory.json`.

This tier is the primary high-powered mechanism and parameter-orbit laboratory.

### Tier 2 — custom small language models

The executable pipeline trains a byte-level BPE tokenizer, packs immutable Hugging Face text corpora into fixed blocks, and trains transparent decoder LMs from 0.86M to 22.97M parameters.

- tokenizer: `configs/tokenizers/tinystories_bpe_2048.yaml`;
- prepared dataset: `configs/datasets/tinystories_clean_bpe2048.yaml`;
- text-zoo pilot: `configs/model_zoo/text_tier2_pilot.yaml`;
- experiment design: `configs/experiments/tier2_small.yaml`;
- tokenizer/data code: `src/behavior2weights/data/tokenizer.py`, `hftext.py`;
- target training: `src/behavior2weights/zoo/text.py`.

The checked-in 32-lineage/192-target configuration is a pipeline and variance pilot. Confirmatory Tier 2 requires at least 512 lineages across at least eight architecture×corpus strata and at least 144 evaluable test lineages. Each corpus is prepared and frozen separately; coordinate decoders are trained only within shape-compatible architecture families.

### Tier 3 — medium language models

The planned 30M–500M tier tests sampled-coordinate training, blockwise reconstruction, low-rank edits, quantization, and checkpoint trajectories. The frozen design surface is in `configs/experiments/tier3_medium.yaml`.

Target distributed training is intentionally scheduler-specific. The repository supplies manifests, trace collection, inverse training, evaluation, tracking, Slurm, and AWS Batch interfaces; operators must provide immutable images, credentials, quotas, and storage.

### Tier 4 — public external validity

The public suite contains 19 model lineages and 72 revision targets across:

- TinyStories 1M/3M/8M/28M/33M;
- Pythia 14M through 2.8B trajectories;
- SmolLM2 135M/360M/1.7B base/instruct pairs.

Files:

- source specification: `configs/model_zoo/public_models.yaml`;
- expanded manifest: `artifacts/public-manifest/targets.jsonl`;
- plan: `artifacts/plans/public-models.json`;
- adapter: `src/behavior2weights/targets/huggingface.py`;
- protocol: `configs/experiments/tier4_public.yaml`.

All checked-in public targets are marked OOD. Revision names such as `main` are placeholders for exploration; a confirmatory run must replace them with immutable Hub commit hashes. Public checkpoints from one training trajectory are repeated measures, not independent lineages.

The Transformer-NFN MNIST/AG News zoo has a separate reviewed download adapter in `src/behavior2weights/zoo/external.py`.

## 6. Dataset and query contracts

Target-training corpora, candidate queries, active-policy training candidates, and functional holdouts are separate immutable artifacts.

Synthetic tasks:

- Markov transition sources;
- copy;
- induction;
- modular addition;
- balanced mixtures.

Text presets:

- `karpathy/tinystories-gpt4-clean`;
- `roneneldan/TinyStories`;
- `Salesforce/wikitext` / WikiText-103 raw.

Query banks contain immutable IDs, source labels, partitions, token IDs, and checksums. Recommended candidate composition is natural, random-token, synthetic diagnostic, rare-token, adversarially optimized training-lineage candidates, and neutral controls. Functional holdouts never enter inverse training or query-policy tuning.

Public-model query banks are tokenizer-family-specific. Original text prompt IDs are preserved so semantically matched prompts can be compared across tokenizers without treating token IDs as shared.

## 7. Observation interfaces

Each channel is collected and analyzed separately:

1. `tokens` — one sampled token, hash-bucket encoded when needed, exact ID retained;
2. `sample_histogram` — repeated samples, bucket frequencies plus exact sampled IDs;
3. `topk` — compact normalized indices/log-probabilities plus exact top-k arrays;
4. `logits` — centered full vocabulary logits; silent truncation is prohibited;
5. `logit_sketch` — seeded random projection for controlled dimensionality reduction.

Stochastic response RNG is keyed by target ID, query ID, channel, repeat index, and frozen salt. Reordering a manifest or distributed shard therefore cannot change sampled outputs.

Implementation:

- codecs: `src/behavior2weights/traces/observations.py`;
- collection: `src/behavior2weights/traces/collector.py`;
- immutable storage: `src/behavior2weights/traces/store.py`;
- validator: `scripts/validateartifacts.py`.

## 8. Inverse model

The reference inverse model has four components:

1. query token encoder;
2. channel-aware observation encoder;
3. permutation-invariant set encoder over prompt/response pairs;
4. coordinate-conditioned Gaussian hypernetwork decoder.

The decoder receives a latent transcript statistic and a semantic scalar address describing tensor, role, layer, shape, row/column, and normalized global position. It predicts a mean and log variance for arbitrary parameter blocks. Full checkpoint reconstruction is streamed in bounded chunks.

A pooled classifier uses the same trace representation to predict architecture, task/corpus, checkpoint stage, optimizer, and intervention labels across compatible trace corpora. Incompatible raw tensors are never assigned a shared coordinate loss.

Files:

- inverse model: `src/behavior2weights/models/inverse.py`;
- property model: `src/behavior2weights/models/property.py`;
- weight address space: `src/behavior2weights/models/weightspace.py`;
- coordinate corpus: `src/behavior2weights/train/corpus.py`;
- trainers: `src/behavior2weights/train/inverse.py`, `property.py`.

## 9. Active identification

All methods select from one frozen candidate bank. Confirmatory deployable policies are:

- random nested-prefix order;
- training-lineage population disagreement;
- posterior expected information gain;
- a learned scorer trained only on training/validation lineages.

The locked evaluator currently implements nested-budget random and leakage-safe training-population disagreement orders. Checkpoints within a lineage are averaged before utility is computed. Oracle target-aware selection is allowed only as a clearly labeled upper bound.

Files:

- policies: `src/behavior2weights/probes/active.py`;
- locked evaluator: `src/behavior2weights/evaluation/runner.py`.

## 10. Controlled mechanisms and causal tests

Implemented edit families include:

- attention-head ablation;
- MLP-neuron ablation;
- low-rank/LoRA-style matrix edits;
- sparse weight edits.

Each edit records its parent target, layer/tensor location, magnitude, rank/count, and deterministic seed. Mechanistic success is evaluated by localization rank/MRR and intervention transfer: apply the inferred edit to the parent/reference model and test whether it reproduces the target behavioral delta on held-out diagnostic prompts.

Interpretability analyses include behavior-to-weight Jacobians, integrated gradients, and transcript patching in `src/behavior2weights/interpret/evidence.py`. Discovery prompts and circuits must be validated on separate lineages/interventions.

## 11. Baselines

At minimum, each compatible condition includes:

- population-mean checkpoint;
- nearest behavioral neighbor;
- linear/ridge trace-to-weight regression for tractable cells;
- direct per-target optimization/distillation where compute permits;
- shuffled-response and shuffled-target negative controls;
- path-only/manifest-order leakage controls.

Baselines use identical query budgets and observation channels. Hyperparameters are selected on validation lineages only.

## 12. Endpoints

### Parameter-space

- raw NRMSE and cosine similarity;
- canonicalized NRMSE;
- orbit-aligned NRMSE and cosine similarity;
- per-role, per-layer, and per-tensor-block error;
- posterior NLL and interval coverage.

### Functional

- forward and reverse KL on held-out prompts;
- probability L1 distance;
- top-1 agreement;
- task accuracy/loss where defined;
- adversarially searched discrepancy under a frozen search budget.

### Mechanistic/property

- architecture/training-factor accuracy or AUROC;
- edit localization MRR/top-k accuracy;
- intervention-transfer effect recovery;
- evidence-map stability across inverse seeds.

### Query efficiency

- normalized AUC over the fixed log2 query-budget grid;
- minimum budget to reach a preregistered threshold;
- posterior entropy reduction per query.

## 13. Confirmatory inference

The co-primary family contains four lineage-paired tests:

1. full logits versus sampled tokens on functional-KL AUC;
2. active versus random queries on functional-KL AUC;
3. active versus random queries on orbit-NRMSE AUC;
4. inverse model versus strongest preregistered baseline on controlled-edit MRR.

Holm controls familywise error at 0.05. The primary estimate is a paired lineage-level contrast with a 10,000-resample lineage-cluster bootstrap confidence interval. A paired sign-randomization test and prespecified mixed-effects model are robustness analyses.

The pooled target is at least 144 evaluable test lineages, inflated for technical failure after pilot-estimated ICC/missingness simulation. Repeated checkpoints, prompts, edit descendants, and inverse seeds do not increase independent sample size.

Implementation:

- inference: `src/behavior2weights/stats/inference.py`;
- power: `src/behavior2weights/stats/power.py`;
- result aggregation: `scripts/aggregateresults.py`;
- full rules: `docs/STATISTICAL_ANALYSIS_PLAN.md`.

## 14. Generalization claims

Every result names the held-out axis. Required panels include unseen:

- initialization seed/lineage;
- data order;
- corpus/task;
- width interpolation;
- depth extrapolation;
- intervention type/rank/location;
- tokenizer;
- observation channel;
- public model family.

Report ID score, OOD score, absolute gap, direction-consistent OOD/ID ratio, lineage-cluster interval, independent lineage/family count, and failure rate. A pooled number never substitutes for stratum results.

## 15. Tracking and compute

The scientific source of truth is immutable manifests/configs/result tables, not an experiment dashboard.

Tracking backends:

- dependency-free local JSONL;
- AIM 3.29.1 local filesystem;
- self-hosted AIM server via `aim://host:53800`.

Infrastructure:

- root Docker image;
- self-hosted AIM Docker Compose stack;
- Slurm templates for planning, zoo building, trace suites, and inverse arrays;
- credential-free AWS Batch definition/submission templates.

Organization-specific credentials, ECR/S3 resources, IAM roles, network controls, GPU queues, and immutable public checkpoint revisions are intentionally placeholders.

## 16. Artifact acceptance criteria

A study is eligible for analysis only when:

1. the exact preflight plan matches observed lineage/target counts;
2. all lineage split invariants hold;
3. checkpoint and trace hashes validate;
4. saved target sentinels reproduce within frozen tolerances;
5. all query IDs/order and channel codec configs match the frozen bank;
6. auxiliary sampled/top-k tensors have declared shapes and finite values;
7. negative leakage controls pass;
8. every missing target has a preregistered failure/exclusion code;
9. one tidy `ResultRecord` exists for every expected target×condition×budget×metric cell;
10. analysis is generated programmatically from the locked result table.

## 17. Execution DAG

```text
freeze configs/SAP/commit
        ↓
preflight zoo plan
        ↓
prepare tokenizer + corpora ── or ── synthetic generator
        ↓
train/download target zoo
        ↓
validate target manifest/checkpoints
        ↓
build immutable query + functional-holdout banks
        ↓
collect each observation channel
        ↓
validate trace stores and auxiliary tensors
        ↓
train inverse/property models on train lineages
        ↓
select hyperparameters on validation lineages
        ↓
run locked target-level evaluation once
        ↓
aggregate to lineage estimands + multiplicity correction
        ↓
mechanistic discovery/confirmation and external replication
```

## 18. Minimal executable proof

```bash
pip install -r requirements/verified-cpu-py313.lock
pip install -e . --no-deps
pytest -q
b2w zoo plan --config configs/model_zoo/micro_confirmatory.yaml
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  b2w smoke --output artifacts/final-smoke --overwrite
```

The checked-in `reports/VERIFICATION.md` records the exact locally executed checks and clearly separates them from unexecuted GPU/network experiments.
