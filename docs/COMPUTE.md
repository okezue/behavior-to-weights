# Compute and Scaling Runbook

## Local

Use the smoke pipeline first. For micro experiments, cap CPU threads:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
b2w smoke --output artifacts/smoke --overwrite
```

## Slurm

`infra/slurm` contains array templates. Recommended decomposition:

1. one array element per target lineage for target training;
2. one array element per target×channel shard for trace collection;
3. one distributed job per architecture decoder/inverse seed;
4. one CPU array for target-level evaluation;
5. one CPU job for locked statistical aggregation.

All workers write immutable shards to a shared/object store and register a success marker only after checksum verification. Never let multiple workers update one JSONL manifest concurrently; merge deterministic shard manifests afterward.

## AWS Batch

`infra/aws` includes a submission script and CloudFormation-style placeholders. Operators must provide:

- AWS account/region;
- ECR image URI and immutable digest;
- S3 bucket/prefix;
- Batch job queue/definition;
- IAM roles with least privilege;
- networking/subnets/security groups;
- instance types and Spot/on-demand policy;
- optional FSx/EFS cache.

No credentials are included. Jobs use instance/task roles. Store target checkpoints and traces in versioned, encrypted S3 prefixes and write a checksum manifest before marking completion.

## Preflight projection

Never launch a zoo build without materializing its plan:

```bash
b2w zoo plan \
  --config configs/model_zoo/micro_confirmatory.yaml \
  --output artifacts/plans/micro-confirmatory.json
```

The projection reports exact factor cardinalities, independent lineage and nested target counts, split counts, per-architecture parameter counts, and raw FP32 checkpoint payload. It intentionally excludes optimizer states, trace stores, container layers, object-store versions, and retry overhead; measure those in the pilot.

## Scale estimates

The dominant costs differ by tier:

- Tier 1: number of lineages and traces; target models are cheap.
- Tier 2: target training and full-logit storage.
- Tier 3: target training plus inverse decoder memory bandwidth.
- Tier 4: checkpoint download/storage and target inference.

Before a full run, execute a 1% pilot that measures:

- target tokens/second;
- queries/second per channel;
- compressed bytes per target×query;
- inverse coordinates/second;
- GPU memory and communication;
- failure and retry rate.

Use measured values to generate a cost sheet; do not rely on generic estimates.

## Trace storage controls

For vocabulary \(V\), full float32 logits cost `4*V` bytes per query before compression. Alternatives:

- float16/bfloat16 with a validation error audit;
- top-k values/indices;
- seeded random sketches;
- centered logits with per-row scale quantization;
- on-the-fly target serving when storage is more expensive than inference.

Compression method is an experimental factor, not an invisible implementation detail.

## Distributed inverse training

Recommended progression:

1. one GPU, sampled coordinates;
2. gradient accumulation;
3. DDP over target batches;
4. FSDP/DTensor for the trace encoder/decoder;
5. sharded coordinate readers and asynchronous prefetch;
6. architecture-specific experts routed by a pooled trace encoder.

Coordinate sampling should be stratified by tensor role so large embeddings do not dominate every minibatch. Full-checkpoint evaluation remains blockwise.
