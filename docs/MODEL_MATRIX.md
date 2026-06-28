# Model and Generalization Matrix

## Checked-in executable cells

### Tier 1 reference confirmatory config

| Architecture | Exact parameters | Layers | Width | Heads | MLP width | Context | Lineages per task | All-task lineages | Targets/lineage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1-A | 1,896 | 1 | 8 | 1 | 16 | 32 | 160 | 640 | 9 |
| T1-B | 4,816 | 1 | 16 | 2 | 32 | 32 | 160 | 640 | 9 |
| T1-C | 15,200 | 2 | 24 | 3 | 64 | 32 | 160 | 640 | 9 |
| T1-D | 73,968 | 3 | 48 | 6 | 128 | 32 | 160 | 640 | 9 |

The full cross has 2,560 independent lineages and 23,040 targets. Each lineage contributes five checkpoints and four final-step edits. The deterministic 70/10/20 policy yields 1,792/256/512 train/validation/test lineages before any named OOD exclusions. Raw FP32 unique-parameter checkpoint payload is approximately 2.06 GiB; traces and optimizer state are additional.

### Tier 2 text pipeline pilot

| Architecture | Exact parameters | Layers | Width | Heads | MLP width | Context | Pilot lineages | Targets/lineage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T2-A | 865,344 | 4 | 96 | 4 | 384 | 256 | 8 | 6 |
| T2-B | 2,552,320 | 6 | 160 | 5 | 640 | 256 | 8 | 6 |
| T2-C | 7,432,704 | 8 | 256 | 8 | 1,024 | 256 | 8 | 6 |
| T2-D | 22,965,504 | 12 | 384 | 12 | 1,536 | 256 | 8 | 6 |

This pilot has 32 lineages and 192 targets, with four checkpoints and two edits per lineage. Its raw FP32 checkpoint payload is approximately 6.05 GiB. It validates the tokenizer→Arrow corpus→transparent LM→trace→inverse pipeline; it is not powered for the confirmatory Tier-2 claims. The confirmatory allocation requires at least 144 evaluable test lineages.

Generate machine-readable values rather than copying this table:

```bash
b2w zoo plan --config configs/model_zoo/micro_confirmatory.yaml
b2w zoo plan --config configs/model_zoo/text_tier2_pilot.yaml
```

Counts are independent lineages unless explicitly labeled targets. Checkpoints and edits are nested repeated measurements.

## Factor allocation

Use a balanced incomplete factorial rather than a completely crossed design when compute is limiting. Each primary architecture stratum must contain:

- at least four model seeds per data-order seed;
- at least four data-order seeds per corpus/optimizer cell;
- when making an optimizer-transfer claim, include AdamW and at least one contrast optimizer in a separately planned stratum; the checked-in Tier-1 reference controls optimizer to AdamW;
- early, middle, and final checkpoints;
- intervention magnitude balanced independently of target seed;
- equal split allocation within architecture/task strata when integer counts permit.

Do not use the target ID hash as a random seed for model training; IDs are derived after all factors are fixed.

## Public external-validity manifest

### TinyStories

- `roneneldan/TinyStories-1M`
- `roneneldan/TinyStories-3M`
- `roneneldan/TinyStories-8M`
- `roneneldan/TinyStories-28M`
- `roneneldan/TinyStories-33M`

These models are useful for size and post-training contrasts but are not assumed to provide independent initialization replicates.

### Pythia

- `EleutherAI/pythia-14m`
- `EleutherAI/pythia-31m`
- `EleutherAI/pythia-70m`
- `EleutherAI/pythia-160m`
- `EleutherAI/pythia-410m`
- `EleutherAI/pythia-1b`
- `EleutherAI/pythia-1.4b`
- `EleutherAI/pythia-2.8b`

Select immutable revisions from `step0`, early log-spaced steps, and `step1000`–`step143000`. Pythia checkpoints share training data order and, within a model, one lineage; analyze trajectory observations as repeated measures.

The 14M repository changed in February 2026: pin the intended standard-Pile or `-deduped` repository and commit explicitly. Never rely on a floating model name for a longitudinal replication.

### SmolLM2

- `HuggingFaceTB/SmolLM2-135M`
- `HuggingFaceTB/SmolLM2-360M`
- `HuggingFaceTB/SmolLM2-1.7B`
- the corresponding `-Instruct` models

Use these for family/size/base-versus-instruct transfer, not literal cross-architecture coordinate comparison.

### Transformer-NFN

The public Small Transformer Zoo contains large MNIST and AG News checkpoint collections and is wired through:

```bash
b2w data download-transformer-nfn --name mnist --output data/external
b2w data download-transformer-nfn --name ag_news --output data/external
```

After the first reviewed download, record the archive SHA-256 in the production config.

## Held-out generalization panels

| Panel | Train | Test | Primary claim |
|---|---|---|---|
| seed | seeds A | unseen seeds B | amortization beyond retrieval |
| lineage | early/checkpoints of train lineages only | entirely new lineages | core ID generalization |
| data order | orders A | orders B | training stochasticity |
| corpus | synthetic/TinyStories | WikiText or held-out synthetic grammar | data distribution |
| architecture interpolation | widths 16,32,64 | width 48 | interpolation |
| architecture extrapolation | depth ≤4 | depth 6/8 | extrapolation |
| edit | ablation/LoRA ranks seen | new layer, rank, or edit type | mechanism compositionality |
| tokenizer | tokenizer A | tokenizer B | trace abstraction |
| channel | logits/top-k | token-only | interface transfer |
| family | all but one public family | held-out family | external validity |

Each panel receives its own OOD label and cannot be pooled into a single unspecified “OOD” result.
