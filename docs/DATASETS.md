# Datasets and Query Banks

## Principles

Target-training data and inverse-model query data are separate artifacts.

- **Target-training corpus:** determines what each target model learns.
- **Candidate query bank:** prompts used to interrogate trained targets.
- **Functional holdout:** prompts never shown to the inverse model, used only for reconstruction evaluation.
- **Active-policy training bank:** prompts and utilities from train/validation targets only.

No target test response can enter query-policy training, preprocessing calibration, or target normalization.

## Synthetic corpora

`behavior2weights.data.synthetic` generates deterministic integer-token sequences for:

- `markov`: a seed-specific transition matrix;
- `copy`: prefix, delimiter, copied suffix;
- `induction`: repeated key/value patterns;
- `modular_addition`: structured arithmetic token sequences;
- `mixture`: balanced shuffled combination of the four.

Each dataset config records vocabulary, sequence length, example counts, task seed, and split-generation offsets. Splits are generated independently rather than slicing one stream.

Synthetic data are the preferred Tier-0/Tier-1 substrate because exact algorithms, exhaustive input spaces, and controlled spurious features can be introduced.

## Hugging Face text datasets

The preparation command downloads, tokenizes, concatenates, and packs fixed causal-LM blocks:

```bash
pip install -e '.[hf]'

b2w data prepare-hf \
  --preset tinystories_clean \
  --output data/hf/tinystories-clean

b2w data prepare-hf \
  --preset wikitext103 \
  --output data/hf/wikitext103
```

Included presets:

| Preset | Dataset | Default subset | Default tokenizer |
|---|---|---|---|
| `tinystories_clean` | `karpathy/tinystories-gpt4-clean` | default | GPT-2 |
| `tinystories` | `roneneldan/TinyStories` | default | GPT-2 |
| `wikitext103` | `Salesforce/wikitext` | `wikitext-103-raw-v1` | GPT-2 |

The output manifest records dataset fingerprints, row counts, tokenizer identity, token IDs, and the full preparation config. Production configs must pin dataset and tokenizer revisions. A dataset update creates a new dataset ID rather than mutating an existing experiment.

### Custom small-vocabulary tokenizer and Tier-2 zoo

For tractable coordinate reconstruction, the repository can train a byte-level BPE tokenizer and transparent decoder LMs rather than inheriting GPT-2's 50,257-token embedding/output matrices:

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

The tokenizer manifest records the exact dataset fingerprint, requested and realized vocabulary size, special tokens, and generated files. The prepared-corpus manifest records fixed block length and split fingerprints. `build-text` rejects vocabulary or context mismatches before training, records validation loss at every saved checkpoint, and assigns all checkpoints and edits from one trajectory to the same lineage split.


## Public-model tokenizer families

Public black-box traces are collected only within tokenizer-compatible families. The checked-in public manifest pins one query-bank tokenizer source per family: GPT-Neo for TinyStories checkpoints, `EleutherAI/gpt-neox-20b` for the Pythia suite, and the SmolLM2-135M tokenizer files for all SmolLM2 sizes and base/instruct variants. A collector rejects manifests that mix vocabulary/tokenizer identities in one trace store.

Before confirmatory execution, replace mutable model names with immutable Hub commit hashes for both model and tokenizer revisions, build a separate tokenized query bank per tokenizer family, and preserve the original text prompt IDs so analyses can compare semantically matched prompts across families without pretending token IDs are shared.

### Licensing

Licenses are upstream-controlled and must be rechecked at the pinned revision. At repository design time, TinyStories pages identify CDLA-Sharing-1.0; WikiText and each public model have their own terms. This repository does not redistribute those datasets or checkpoints.

## Query bank partitions

Every `QueryRecord` has a `source` and `partition`. Recommended source proportions:

- 25% natural corpus contexts;
- 25% random-token contexts;
- 20% synthetic diagnostics targeting known algorithms;
- 10% rare-token/low-probability contexts;
- 10% adversarially optimized candidates on train lineages;
- 10% fixed neutral controls.

A query bank should be much larger than the largest budget. The active policy selects from exactly the same candidate bank as random and heuristic policies.

### Fixed budget grid

Use powers of two whenever feasible:

```text
1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
```

For each lineage, all methods share the same observation realization where mathematically possible. Token sampling randomness is paired by seed; if adaptive methods choose different prompts, response RNG is a deterministic function of target, query, channel, repeat index, and preregistered salt.

## Functional evaluation sets

At least four held-out sets are required:

1. in-distribution natural prompts;
2. random-token prompts;
3. task/circuit diagnostic prompts;
4. adversarial prompts selected after reconstruction, using only the reconstructed and target black-box interfaces under a fixed search budget.

The fourth set measures worst-case functional discrepancy but must not be used to improve the reconstruction being evaluated.

## Data integrity

- Save immutable manifests and SHA-256 checksums.
- Store exact Hub commit/revision, not only repository name.
- Check for prompt duplicates across inverse input and functional holdout.
- Normalize text before duplicate detection and also compare token IDs.
- Record failed/filtered examples with reasons.
- Never write API keys, access tokens, or proprietary prompts into manifests or AIM.
