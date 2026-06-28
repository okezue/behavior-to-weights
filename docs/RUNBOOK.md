# End-to-End Runbook

## Phase A — Validate the environment

```bash
pip install -e '.[dev,hf]'
b2w doctor
pytest
b2w smoke --output artifacts/smoke --overwrite
```

Archive the smoke report with the environment but do not mix it into scientific results.

## Phase B — Freeze a pilot

1. Copy a tier config to a dated study directory.
2. Define architecture/corpus/optimizer factors and named OOD values.
3. Generate the exact target/storage plan and inspect lineage counts:

   ```bash
   b2w zoo plan --config studies/<study>/micro_zoo.yaml --output studies/<study>/zoo-plan.json
   ```
4. Freeze query bank and functional holdouts.
5. Run 5–10% pilot lineages.
6. Measure variance, ICC, failures, storage, and compute.
7. Run clustered power simulation.
8. Freeze confirmatory config, commit, manifests, and SAP:

   ```bash
   python scripts/freeze_study.py \
     studies/<study>/experiment.yaml studies/<study>/zoo.yaml \
     studies/<study>/inverse.yaml studies/<study>/SAP.md \
     --study-id <immutable-study-id> \
     --output studies/<study>/study.lock.json
   ```

## Phase C — Build target zoo

```bash
# Mandatory dry run before allocating compute
b2w zoo plan \
  --config studies/<study>/micro_zoo.yaml \
  --output studies/<study>/zoo-plan.json

b2w zoo build-micro \
  --config studies/<study>/micro_zoo.yaml \
  --output s3-or-local/zoo

# Tier-2 text alternative, after `b2w data train-tokenizer` and `prepare-hf`
b2w zoo build-text \
  --config studies/<study>/text_zoo.yaml \
  --dataset data/hf/<prepared-corpus> \
  --output s3-or-local/zoo

b2w zoo summary s3-or-local/zoo/targets.jsonl
```

Validate expected lineages/checkpoints/interventions and checksums. Do not start trace collection on an incomplete manifest.

## Phase D — Collect traces

Run every channel as a separate, immutable dataset. Use the same query IDs and paired RNG mapping.

```bash
b2w traces collect-micro-suite \
  --manifest data/zoo/targets.jsonl \
  --queries data/queries/candidates.jsonl \
  --experiment studies/<study>/experiment.yaml \
  --output data/traces
```

Perform trace sentinel checks: dimensions, finite values, expected probability sums, top-k consistency, exact auxiliary sampled-ID/top-k tensor shapes, and repeatability.

## Phase E — Train inverse models

Train multiple inverse seeds on identical train/validation lineages. Select hyperparameters using validation-lineage AUC and calibration, never test targets. For coordinate recovery, train one compatible architecture family at a time. Use `--tracking-config configs/tracking/aim_remote.yaml` for self-hosted AIM or omit it for the local JSONL tracker.

For cross-architecture targets, train the pooled property/circuit model separately:

```bash
b2w train properties \
  --manifest data/zoo/targets.jsonl \
  --traces data/traces/logits \
  --config configs/models/property_small.yaml \
  --tracking-config configs/tracking/aim_remote.yaml \
  --output artifacts/properties
```

Keep:

- last and best checkpoints;
- all validation decisions;
- target coordinate sampling distribution;
- resolved model/training config;
- tracker run ID;
- failure/restart history.

## Phase F — Locked evaluation

1. Mark the code/config commit as release candidate.
2. Unblind test trace location once.
3. Run every preregistered method and budget.
4. Emit tidy target-level `ResultRecord` rows.
5. Check count completeness before looking at aggregate effects.
6. Run the locked evaluators and frozen analysis script:

   ```bash
   b2w evaluate micro --manifest ... --traces ... --checkpoint ... --output results.jsonl
   b2w evaluate properties --manifest ... --traces ... --checkpoint ... --output properties.jsonl
   python scripts/aggregate_results.py results.jsonl \
     --contrast method \
     --fixed-query-policy random \
     --metric weight_nrmse:lower \
     --metric functional_forward_kl:lower \
     --output analysis.json
   ```

7. Store raw and adjusted p-values, CIs, failures, and stratum estimates.

Any code correction after unblinding is documented. Material changes trigger a fresh held-out replication.

## Phase G — Mechanistic analysis

Use discovery lineages to identify witness prompts and inverse circuits. Validate them on a separate confirmatory intervention set. Apply inferred edits to parent models and test intervention transfer.

## Phase H — Release

Publish or archive:

- source commit;
- configs and manifests;
- permitted trace/checkpoint artifacts or generation instructions;
- result tables;
- analysis outputs;
- model cards/data statements;
- security review and disclosure record;
- known limitations and failed replications.
