from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from behavior2weights.schemas import QueryRecord
from behavior2weights.traces.store import load_trace_bundle
from behavior2weights.utils import read_jsonl
from behavior2weights.zoo.manifest import load_manifest, manifest_summary, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate immutable B2W manifests and trace stores"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--traces", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--verify-checkpoints", action="store_true")
    args = parser.parse_args()

    records = load_manifest(args.manifest, resolve_paths=args.verify_checkpoints)
    validate_manifest(records, verify_files=args.verify_checkpoints)
    report: dict[str, object] = {"manifest": manifest_summary(records), "errors": []}
    target_ids = {record.target_id for record in records}

    query_records: list[QueryRecord] | None = None
    if args.queries:
        query_records = [QueryRecord.model_validate(row) for row in read_jsonl(args.queries)]
        query_ids = [record.query_id for record in query_records]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("duplicate query IDs")
        lengths = {len(record.input_ids) for record in query_records}
        if len(lengths) != 1:
            raise ValueError("query bank contains mixed sequence lengths")
        report["queries"] = {"count": len(query_records), "sequence_lengths": sorted(lengths)}

    if args.traces:
        bundle = load_trace_bundle(args.traces, verify=True)
        missing_targets = sorted(set(bundle.target_ids) - target_ids)
        if missing_targets:
            raise ValueError(f"trace store has targets absent from manifest: {missing_targets[:5]}")
        if query_records and tuple(record.query_id for record in query_records) != bundle.query_ids:
            raise ValueError("trace query order differs from the frozen query bank")
        if not torch.isfinite(bundle.observations).all():
            raise ValueError("trace observations contain non-finite values")
        nonfinite_auxiliary = [
            key
            for key, value in bundle.auxiliary.items()
            if value.is_floating_point() and not torch.isfinite(value).all()
        ]
        if nonfinite_auxiliary:
            raise ValueError(
                f"trace auxiliary tensors contain non-finite values: {nonfinite_auxiliary}"
            )
        report["traces"] = {
            "channel": bundle.channel.value,
            "shape": list(bundle.observations.shape),
            "feature_dim": bundle.feature_dim,
            "auxiliary": {
                key: list(value.shape) for key, value in sorted(bundle.auxiliary.items())
            },
        }
    report["status"] = "valid"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
