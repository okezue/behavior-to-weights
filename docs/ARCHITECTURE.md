# System Architecture

## Data flow

```text
Target manifest + immutable query bank
              │
              ▼
       target-family adapter
              │
              ▼
  channel-specific trace collector ───► trace bundle + checksums
              │
              ▼
       set-invariant trace encoder
              │
              ├──► property/circuit posterior heads
              │
              ▼
        latent sufficient statistic z
              │
              ▼
 coordinate-conditioned probabilistic decoder
              │
              ├──► arbitrary tensor blocks
              ├──► full reconstructed checkpoint
              └──► uncertainty by coordinate
```

## Target adapters

`TargetAdapter` is the only component that knows how to instantiate a target family. The included adapters support the transparent micro-transformer and Hugging Face causal LMs. Future adapters should expose:

- deterministic model loading from a revision-pinned record;
- logits with an explicitly defined position/interface;
- floating-point state dictionaries where permitted;
- architecture metadata and parameter-role mapping;
- optional activation/circuit hooks.

Remote-code execution is disabled by default in the Hugging Face adapter.

## Weight addressing

`AddressSpace` converts a state dictionary into a deterministic list of `TensorSpec` values. Each scalar address is represented by:

- architecture-local tensor ID;
- semantic role (Q/K/V/O, MLP input/output, embedding, norm, head);
- layer ID;
- normalized flat, row, and column coordinates;
- tensor size/shape/rank features;
- global normalized address.

Large checkpoints should be read by safetensors slices or memory mapping. The reference in-memory corpus is intentionally simple for Tier 1 and smoke tests.

## Trace encoder

The query token encoder uses token and positional embeddings plus a small transformer. It produces one vector per prompt. The observation encoder consumes a fixed-width representation of the selected output interface. A channel embedding prevents the same numeric vector from being interpreted identically across, for example, token one-hot and empirical-frequency channels.

Prompt/response pair vectors are processed by a transformer with no pair-position embedding. A learned summary token therefore implements a permutation-invariant set encoder, modulo floating-point ordering effects. Query masking permits variable budgets.

## Decoder

The decoder is a hypernetwork:

```text
(weight-address embedding, trace latent) -> mean, log variance
```

It can decode a random coordinate sample for training or every coordinate in bounded chunks for reconstruction. This avoids a vocabulary of textual floating-point numbers and decouples output size from context length.

## Multi-architecture operation

There are two supported regimes:

1. **Conditional coordinate recovery:** the target architecture is supplied, and one decoder/address space is trained per compatible family.
2. **Joint inference:** a pooled trace encoder predicts an architecture posterior; architecture-specific decoders are evaluated under candidate architectures and combined or routed.

A future common weight-space model may use globally canonical tensor roles and neural functional networks, but the repository never compares raw coordinate vectors of different shapes.

## Symmetry handling

The micro-transformer implements exact head and MLP-neuron permutation transformations. Evaluation reports:

- raw storage-space error;
- error after deterministic canonicalization;
- Hungarian orbit-aligned error;
- functional divergence before and after alignment.

The canonicalizer is a baseline gauge choice, not a proof of globally continuous canonical coordinates. Functional equivalence remains the arbiter.

## Active identification

A candidate query bank is generated independently of test targets. Policies select from this fixed bank:

- random;
- natural-order/frequency heuristic;
- posterior-predictive ensemble disagreement;
- expected diagonal-Gaussian information gain;
- learned query scorer trained to predict validation-lineage entropy reduction.

The oracle policy may use true candidate target ensembles only as an upper bound and is never reported as deployable.

## Storage contracts

- `targets.jsonl`: one validated `TargetRecord` per checkpoint/edit.
- `queries.jsonl`: one `QueryRecord` per candidate prompt.
- `traces.safetensors` + `manifest.json`: `[target, query, feature]` observations, shared inputs, and exact channel-specific `auxiliary.*` tensors.
- inverse checkpoint: model safetensors, model config, address space, standardizer, metadata.
- result table: one `ResultRecord` per target×condition×metric.

Every immutable binary artifact has a SHA-256 entry in its manifest. Paths are relative at rest and resolved at load time.

Observation codecs are deliberately explicit:

- token and sample-histogram channels use deterministic hash buckets when the requested feature width is smaller than the vocabulary; exact sampled token IDs are retained in `auxiliary.output_ids`;
- the top-k channel stores a compact feature vector of normalized token indices and log-probabilities, while exact indices and values are retained in auxiliary tensors;
- the full-logit channel rejects feature widths smaller than the vocabulary rather than silently truncating; use the seeded logit-sketch channel for dimensionality reduction;
- stochastic outputs are keyed by target ID, query ID, channel, repeat index, and frozen salt so results are invariant to distributed sharding and manifest order.

## Scaling substitutions

| Reference component | Large-scale replacement |
|---|---|
| in-memory target vectors | sharded/memory-mapped safetensors coordinate reader |
| one-process collection | model-server pool + deterministic work queue |
| dense full logits | compressed float16/logit sketch shards with audited error |
| one decoder process | FSDP/DTensor or tensor-parallel coordinate decoder |
| local query bank | object-store versioned dataset |
| JSONL tracker | self-hosted AIM remote server |
| local launch | Slurm array or AWS Batch array job |
