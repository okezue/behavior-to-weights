# Slurm launch templates

The templates use environment variables so no cluster-specific paths or credentials are committed. Create `logs/` before submission, pin the repository commit and container/environment digest in the job record, and freeze all resolved configs.

## Build the transparent model zoo

```bash
mkdir -p logs
export B2W_VENV=$PWD/.venv
export B2W_CONFIG=configs/model_zoo/micro_confirmatory.yaml
export B2W_OUTPUT=/shared/b2w/zoo/tier1
sbatch infra/slurm/build_zoo.sbatch
```

`build_zoo.sbatch` writes the exact lineage/target/storage plan before starting training. Treat a mismatch from the preregistered plan as a failed launch.

## Collect the complete observation suite

```bash
export B2W_MANIFEST=/shared/b2w/zoo/tier1/targets.jsonl
export B2W_QUERIES=/shared/b2w/queries/tier1/queries.jsonl
export B2W_EXPERIMENT=configs/experiments/tier1_micro.yaml
export B2W_OUTPUT=/shared/b2w/traces/tier1
export B2W_BASE_SEED=20260621
sbatch infra/slurm/collect_traces.sbatch
```

The collector reads all five channel definitions from the frozen experiment YAML, preserving exact top-k/sample auxiliaries and projection settings. It writes a `suite_manifest.json` plus one immutable directory per channel.

## Train inverse models

`train_inverse.sbatch` uses an array over the five channel stores. Submit separate arrays for preregistered inverse seeds, or wrap it in a second scheduler dimension. The target lineage split and trace stores must remain fixed across seeds and methods.

```bash
export B2W_MANIFEST=/shared/b2w/zoo/tier1/targets.jsonl
export B2W_TRACES_ROOT=/shared/b2w/traces/tier1
export B2W_CONFIG=configs/models/inverse_micro.yaml
export B2W_OUTPUT=/shared/b2w/runs/tier1
export B2W_SEED=20260621
export B2W_TRACKING_CONFIG=configs/tracking/aim_remote.yaml
sbatch infra/slurm/train_inverse.sbatch
```
