# AIM Experiment Tracking

AIM is optional. The scientific source of truth remains immutable configs, manifests, result tables, and checkpoint hashes; AIM provides searchable run metadata and metric visualization.

## Supported API line

The repository pins AIM 3.29.1 and uses its documented `aim.Run` interface. Repositories may be local filesystem paths or remote `aim://host:53800` URIs. The supplied server image uses Python 3.11; every training job can instead use the dependency-free `JsonlTracker`. Workers in one study must use the same pinned AIM image and configuration.

## Local self-hosting

```bash
cd infra/aim
docker compose up -d

# UI
open http://localhost:43800

# Training clients
export AIM_REPO='aim://localhost:53800'
```

The compose stack runs a tracking server and UI over one persistent volume. It is development/research infrastructure; add TLS/auth/network controls before shared or remote deployment.

## Config

```yaml
tracking:
  backend: aim
  repo: aim://aim.internal.example:53800
  experiment: b2w-tier2-confirmatory
  enabled: true
  tags: [confirmatory, tier2, logits]
  system_tracking_interval: 10.0
```

Pass the file directly to training:

```bash
b2w train inverse ... --tracking-config configs/tracking/aim_remote.yaml
b2w train properties ... --tracking-config configs/tracking/aim_remote.yaml
```

The Python factory creates the tracker. If AIM is unavailable, choose `backend: jsonl`; the framework never silently drops metrics.

## Required metadata

Each run logs:

- Git commit and dirty status;
- complete resolved config;
- target/query/trace manifest hashes;
- architecture ID and split;
- method, channel, budget, replicate;
- dependency/runtime summary;
- scheduler/cloud job IDs;
- training and validation metrics;
- terminal status and failure reason.

Do not log secrets or full sensitive prompts.

## Run hierarchy

Recommended experiment names:

```text
b2w/<study>/<tier>/<phase>
```

Contexts:

```text
subset=train|validation|test
channel=tokens|histogram|topk|logits|sketch
budget=<integer>
architecture=<id>
held_out_axis=<name>
```

Target-level scientific results are still exported to tidy JSONL/Parquet; AIM plots are not manually scraped for analysis.
