# Experimental Design

## 1. Objective

Estimate what internal information about a transformer is identifiable from black-box behavioral traces, how query design and output interface affect identifiability, and what evidence an amortized inverse model uses to infer target mechanisms.

For target parameters \(\theta\), a trace

\[
T_n = \{(x_i, o_i)\}_{i=1}^n
\]

contains prompts \(x_i\) and observations \(o_i\) from one specified output interface. The inverse model estimates

\[
q_\phi([\theta], z, a \mid T_n),
\]

where \([\theta]\) is a parameter-symmetry orbit, \(z\) denotes mechanisms or controlled edits, and \(a\) denotes architecture/training properties.

The study does not assume that an arbitrary finite transcript uniquely determines literal floating-point storage. It measures a hierarchy of recovery claims.

## 2. Recovery hierarchy

| Level | Estimand | Minimum successful evidence |
|---|---|---|
| L0 | chance/control | inverse model does not exploit manifest order, file metadata, or query leakage |
| L1 | identity/fingerprint | held-out target or lineage retrieval above a preregistered baseline |
| L2 | property inference | architecture, dataset, optimizer, step bin, tuning method, or quantization classification/calibration |
| L3 | mechanism inference | controlled edit or circuit location recovered on unseen lineages |
| L4 | functional reconstruction | reconstructed model matches target on unseen and adversarially selected inputs |
| L5 | parameter-orbit recovery | low symmetry-aligned error with calibrated uncertainty |
| L6 | literal storage recovery | exact bytes after a declared canonical gauge; not presumed achievable |

Claims are made at the highest level supported by all required metrics. L1 success does not imply L5; L5 does not imply training-data extraction.

## 3. Research questions and confirmatory hypotheses

### RQ1 — Observation interface

How much additional target information is available from full logits, top-k log-probabilities, repeated samples, and one sampled token?

**H1:** At a fixed query budget, full-logit traces improve held-out functional KL and symmetry-aligned NRMSE over token-only traces. The primary contrast is paired within lineage and query bank.

### RQ2 — Query selection

Can learned or posterior-disagreement probes reduce query complexity?

**H2:** Active selection improves area under the recovery-versus-log-query-budget curve over a random-query policy using the same candidate bank and target observations.

### RQ3 — Mechanism before coordinates

Are controlled circuit edits recoverable before exact weights?

**H3:** Edit localization reaches prespecified top-k accuracy at query budgets for which parameter NRMSE remains above its success threshold.

### RQ4 — External generalization

Does the inverse model generalize beyond seen seeds, datasets, architectures, and model families?

**H4:** Performance on lineage-disjoint and factor-held-out targets exceeds non-amortized and nearest-neighbor baselines, with a separately reported OOD degradation ratio.

### RQ5 — Information geometry

Is recoverability concentrated in a low-dimensional observable parameter subspace?

**H5:** Empirical behavior-to-weight Jacobian/Fisher spectra are heavy-tailed, and high-observability directions explain more functional variance than norm-matched low-observability directions.

### RQ6 — Evidence used by the inverse model

Which prompts, output positions, tokens, or logit directions cause specific inferred mechanisms?

This is exploratory but causally validated through transcript patching, observation ablation, and intervention transfer.

## 4. Factorial target population

Each target record declares the complete data-generating factors:

- architecture family, depth, width, head count, MLP ratio, normalization, positional encoding, tied embeddings;
- initialization distribution and seed;
- tokenizer and vocabulary;
- corpus/task and data-order seed;
- optimizer, learning rate, schedule, weight decay, batch size;
- training step/checkpoint;
- post-training method, adapter rank, quantization;
- controlled intervention and parent checkpoint.

A **lineage** is the maximal group sharing initialization and training trajectory. Checkpoints and all edited descendants are nested inside it.

### Split construction

1. Reserve named factor values for OOD evaluation before training the inverse model.
2. Hash-sort remaining lineage IDs with a public salt.
3. Allocate complete lineages to train/validation/test.
4. Verify no checkpoint hash, parent ID, initialization, data-order seed, or near-duplicate tensor signature crosses a boundary.
5. Freeze manifests and store SHA-256 checksums.

No prompt/response from a test target may be used to train a query scorer, normalizer, architecture router, canonicalizer, or early-stopping rule.

## 5. Experiment tiers

### Tier 0 — Analytic identifiability

**Systems:** multinomial logistic regression, embedding-plus-softmax, one attention head without MLP, one MLP block, and one transformer block.

**Population:** at least 1,000 randomly generated systems per cell, with exhaustive inputs whenever the finite domain is at most 100,000 strings.

**Purpose:**

- recover known additive softmax gauge freedoms;
- compare numerical rank of the query Jacobian with empirical posterior contraction;
- validate uncertainty coverage when the true equivalence class is known;
- expose implementation bugs before learned-language experiments.

**Success criteria:** rank/equivalence predictions agree with numerical recovery in at least 99% of well-conditioned simulated systems; failures are stratified by condition number.

### Tier 1 — Micro-transformer mechanism laboratory

**Sizes:** approximately 1K, 5K, 20K, 100K, and 500K parameters.

**Architecture cells:** at least 12 combinations of depth 1–4, width 8–64, heads 1–8, and MLP ratio 2–4. Keep tokenizer/vocabulary fixed inside each coordinate-recovery family.

**Training tasks:** Markov sources, copy, induction, modular arithmetic, mixed synthetic language, and at least one deliberately spurious-correlate task.

**Checked-in reference confirmatory population:**

- 4 architectures × 4 tasks × 40 initialization seeds × 4 dataset seeds × 1 optimizer = **2,560 independent lineages**;
- 160 lineages per architecture×task cell and 640 per architecture;
- 5 checkpoints plus 4 final-step controlled edits per lineage = **23,040 targets**;
- 70/10/20 lineage split = 1,792 train, 256 validation, and **512 test lineages**;
- named OOD factor values can be removed from this allocation and assigned wholly to OOD.

This intentionally exceeds the SAP's pooled minimum of 144 evaluable test lineages and permits architecture/task-stratified estimates. A compute-reduced confirmatory design may use a preregistered balanced incomplete factorial, but it must retain the independent-lineage power target. Checkpoints/edits are repeated measurements, not additions to independent \(n\). Run `b2w zoo plan --config configs/model_zoo/micro_confirmatory.yaml` before launch.

**Queries:** exhaustive strings for the smallest cells; 8,192 fixed candidates otherwise. Budgets: 1, 2, 4, …, 2,048.

**Primary role:** high-powered tests of H1–H3, exact symmetry validation, and mechanistic tracing of the inverse model.

### Tier 2 — Small language models

**Sizes:** 0.5M–30M parameters.

**Architectures:** custom GPT-like cells plus public TinyStories 1M/3M/8M/28M/33M checkpoints where licenses and model structure permit.

**Corpora:** TinyStories clean, TinyStories original, WikiText-103 raw, and synthetic mixtures. Each custom architecture receives independent model seeds and independent data-order seeds.

**Recommended population:** at least 512 independent lineages across at least eight architecture×corpus strata, with an explicit blocked allocation of at least **144 evaluable test lineages** balanced across strata; three checkpoints and two controlled low-rank/sparse edits per lineage. The checked-in 32-lineage/192-target text configuration is a pipeline and variance pilot, not the confirmatory population.

**Budgets:** 16–8,192 queries. Candidate prompts are partitioned into natural, random-token, synthetic diagnostic, and learned-active sets.

**Primary role:** H4 generalization across unseen seeds, data order, task mixtures, and selected architecture factors.

### Tier 3 — Medium language models

**Sizes:** 30M–500M parameters.

**Design:** 128–256 independent custom lineages where compute permits, plus Pythia 31M/70M/160M/410M trajectories. For a public trajectory with only one initialization, checkpoints are repeated observations of one lineage and cannot establish seed-level generalization by themselves.

**Interventions:** LoRA edits at multiple ranks/layers, sparse factual/synthetic-rule edits, head/MLP ablations where architecture permits, 8-bit/4-bit quantization, and instruction-tuning subsets.

**Primary role:** transfer from Tier 2, low-rank delta recovery, realistic functional extraction, and scaling curves. Confirmatory claims require at least 80 independent custom lineages or are labeled exploratory/external replication.

### Tier 4 — Public external-validity suite

**Models:** Pythia 14M–2.8B selected steps; SmolLM2 135M/360M/1.7B base/instruct pairs; other permissively licensed, revision-pinned families added by manifest.

**Claims:** identity, property, lineage/step inference, active-query transfer, functional extraction, and compressed adapter recovery. Literal cross-family weight error is not computed.

**Evaluation:** leave-one-family-out; leave-one-size-bin-out; tokenizer-held-out; base-to-instruct and unquantized-to-quantized transfer. Report family-level estimates and raw per-family values because the number of independent public families is small.

## 6. Controlled interventions

Every edited model has a parent and machine-readable ground truth.

1. **Attention-head ablation:** zero selected Q/K/V rows and output-projection columns.
2. **MLP-neuron ablation:** zero input row/bias and output column.
3. **Low-rank edit:** add \(\alpha AB/\sqrt r\) to one declared matrix.
4. **Sparse edit:** perturb exact stored indices with signed, scale-normalized deltas.
5. **Circuit insertion/removal:** train paired synthetic tasks or transplant a known module.
6. **Fine-tuning/adapters:** fixed dataset and rank, with base checkpoint retained.
7. **Quantization:** deterministic method, group size, calibration data, and software revision.

Intervention magnitude is calibrated on validation lineages to span barely behavioral, moderate, and obvious effects. Thresholds are frozen before test collection.

## 7. Observation channels

Each channel is collected and stored separately:

- `tokens`: one sample per query;
- `sample_histogram`: empirical frequencies from a fixed number of repeated samples;
- `topk`: values and indices at declared \(k\);
- `logits`: full final-position logits, centered to remove a non-identifiable additive constant;
- `logit_sketch`: a seeded random projection of centered logits.

Temperature, sampling RNG, top-k, number of repeats, precision, logit post-processing, and server implementation are immutable trace metadata. No result may compare channels using different prompts or budgets unless explicitly modeled.

## 8. Inverse architecture

1. A token encoder maps each prompt to a fixed representation.
2. An observation encoder maps the declared channel to the same trace width.
3. A channel embedding prevents accidental interface conflation.
4. A set transformer summarizes an unordered set of query/response pairs.
5. Property and circuit heads predict discrete/continuous target properties.
6. A coordinate decoder receives the trace latent and an address `(architecture hypothesis, tensor role, layer, tensor ID, normalized coordinates)` and emits mean/log-variance.
7. Full checkpoints are decoded in blocks from the same latent.

For incompatible architectures, use an architecture router and architecture-specific address space/decoder. The architecture may be supplied for conditional recovery or marginalized over using the predicted posterior.

## 9. Baselines

Every primary comparison includes:

- train-population mean checkpoint;
- nearest behavioral neighbor;
- ridge/linear trace-to-weight map where tractable;
- non-amortized direct optimization/distillation against the same query transcript;
- random and natural-query policies;
- oracle posterior-disagreement selection as an upper diagnostic bound;
- identity-only classifier to detect when reconstruction gains reduce to retrieval;
- shuffled-target and shuffled-response negative controls.

A retrieval audit removes the nearest train lineage and repeats evaluation. Compositional held-outs test combinations never present during training.

## 10. Endpoints

### Weight-space

- raw NRMSE and cosine similarity;
- deterministic-canonical NRMSE;
- optimal orbit-aligned NRMSE;
- per-layer, tensor-role, and Fisher-eigenmode errors;
- posterior NLL, interval coverage, and standardized residuals.

### Functional

- forward and reverse KL on unseen prompts;
- top-1/token-distribution agreement;
- sequence-level divergence under matched decoding randomness;
- held-out task loss/perplexity;
- adversarial active-query agreement;
- intervention-transfer effect.

### Mechanistic

- edit type accuracy/AUROC;
- layer/head/neuron mean reciprocal rank and top-k localization;
- overlap with ground-truth changed parameter mask;
- causal reproduction after applying inferred edit to the parent;
- evidence sparsity and minimal witness queries.

### Query efficiency

Plot each endpoint against \(\log_2\) query budget and summarize by preregistered AUC. Also report the minimum budget reaching fixed success thresholds.

## 11. Leakage controls

- Paths, filenames, target IDs, manifest order, tensor checksums, and response RNG seeds are never model inputs.
- Responses are regenerated after randomly permuting target order; predictions must match.
- Tokenizers and prompt templates are either fixed or declared target properties.
- Query banks are generated before target split labels are exposed to the policy.
- Test traces are encrypted or access-controlled until model selection freezes.
- Duplicate and near-duplicate checkpoints are detected by exact hash and random-projection signatures.
- An inverse model trained on shuffled target labels must remain at chance/population-mean performance.

## 12. Stage gates

Proceed only when the previous gate passes:

1. **Implementation gate:** all unit tests and analytic invariance tests pass.
2. **Tier-0 gate:** numerical identifiability matches theory and posterior coverage is calibrated.
3. **Pilot gate:** no split leakage; effect directions stable across at least three fresh pilot seeds.
4. **Preregistration gate:** hypotheses, metrics, sample sizes, exclusions, and code commit frozen.
5. **Confirmatory gate:** one locked run; no tuning on test lineages.
6. **Replication gate:** new model zoo, new data-order seeds, and at least one unseen architecture/family.

## 13. Generalization matrix

Every paper table labels exactly which axes are held out:

- seed/initialization;
- data order;
- dataset/task;
- optimizer/schedule;
- checkpoint range;
- architecture width/depth/head count;
- tokenizer/vocabulary;
- intervention type/magnitude;
- model family;
- output channel;
- query distribution.

“Generalizes” is never used without naming the held-out axes.
