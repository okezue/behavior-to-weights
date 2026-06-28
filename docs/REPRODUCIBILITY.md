# Reproducibility and Artifact Integrity

## Required frozen artifacts

Every experiment release contains:

- resolved YAML config;
- Git commit and patch/dirty flag;
- Python and dependency lock;
- target, query, and trace manifests with hashes;
- split salt and explicit OOD rules;
- model/inverse seeds and deterministic settings;
- scheduler/container image digest;
- result table and analysis script commit;
- exclusion/failure log;
- final paper tables generated from the result table.


## Dependency and image locks

The checked-in `requirements/verified-cpu-py313.lock` records the exact dependency closure used for local CPU verification. It is platform-specific and must not be treated as a CUDA/Hugging Face/AIM lock. Every production study must freeze a separately resolved lock for its Python/CUDA platform and selected extras, then record the immutable container digest and driver/runtime metadata.

The project package should be installed with `--no-deps` after applying a frozen lock, preventing an editable install from silently changing resolved versions.

## Randomness domains

Use independent named seeds for:

- target initialization;
- target data order;
- target optimization stochasticity;
- intervention generation;
- query-bank generation;
- query-policy tie breaking;
- target sampling responses;
- inverse initialization;
- inverse minibatches and coordinate samples;
- bootstrap/permutation tests.

Never reuse an opaque global seed for all domains. Derive child seeds with a stable hash of `(domain, entity ID, frozen salt)`.

## Determinism

`seed_everything` seeds Python, NumPy, and PyTorch and enables deterministic algorithms where supported. Determinism does not guarantee cross-hardware bit identity. Record hardware and compare with declared numerical tolerances.

For small CPU kernels, set a bounded OpenMP thread count. Excessive thread pools can make micro experiments slower and alter floating-point accumulation order.

## Integrity checks

Before launching a model zoo, materialize the exact design and storage projection:

```bash
b2w zoo plan --config configs/model_zoo/micro_confirmatory.yaml \
  --output artifacts/plans/micro-confirmatory.json
```

Treat unexpected lineage, target, split, parameter, or storage counts as a configuration error.

Before analysis:

1. verify all checkpoint and trace checksums;
2. instantiate targets and reproduce saved validation loss/logit sentinels;
3. confirm lineages occupy exactly one split;
4. scan exact and random-projection duplicate signatures across splits;
5. verify query/functional-holdout disjointness;
6. rerun exact symmetry tests;
7. compare resolved configs across workers;
8. verify expected target counts and failure accounting;
9. verify every trace tensor, including `auxiliary.*` sampled IDs/top-k arrays, has the declared shape and finite values;
10. confirm full-logit traces were never silently truncated and every projected channel records its projection/hash configuration.

## Containerization

The root Dockerfile defines the project runtime. GPU deployments should build from a CUDA/PyTorch base matched to the cluster driver and retain the same Python package lock. Record the image digest, not just a mutable tag.

## Replication levels

- **Exact rerun:** same manifests, seeds, image, and hardware class.
- **Computational replication:** same protocol and data, new inverse seeds.
- **Target replication:** new target lineages and data-order seeds.
- **Conceptual replication:** new architecture/model family and query bank.

A robust claim should survive at least target replication; cross-family claims require conceptual replication.
