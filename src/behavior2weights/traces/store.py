from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from behavior2weights.schemas import ObservationChannel
from behavior2weights.utils import file_sha256


@dataclasses.dataclass(frozen=True, slots=True)
class TraceBundle:
    target_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    input_ids: Tensor
    observations: Tensor
    channel: ObservationChannel
    feature_dim: int
    auxiliary: dict[str, Tensor] = dataclasses.field(default_factory=dict)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [queries, sequence]")
        if self.observations.ndim != 3:
            raise ValueError("observations must have shape [targets, queries, features]")
        if self.observations.shape[0] != len(self.target_ids):
            raise ValueError("target_ids length does not match observations")
        if self.observations.shape[1] != len(self.query_ids):
            raise ValueError("query_ids length does not match observations")
        if self.input_ids.shape[0] != len(self.query_ids):
            raise ValueError("input_ids query dimension does not match query_ids")
        if self.observations.shape[2] != self.feature_dim:
            raise ValueError("feature_dim does not match observations")
        for key, tensor in self.auxiliary.items():
            if tensor.ndim < 2:
                raise ValueError(f"auxiliary tensor {key!r} must start with target and query axes")
            if tensor.shape[:2] != self.observations.shape[:2]:
                raise ValueError(
                    f"auxiliary tensor {key!r} has incompatible target/query dimensions"
                )

    def subset_targets(self, indices: Sequence[int]) -> TraceBundle:
        index = torch.tensor(indices, dtype=torch.long)
        return dataclasses.replace(
            self,
            target_ids=tuple(self.target_ids[position] for position in indices),
            observations=self.observations[index],
            auxiliary={key: value[index] for key, value in self.auxiliary.items()},
        )

    def subset_queries(self, indices: Sequence[int]) -> TraceBundle:
        index = torch.tensor(indices, dtype=torch.long)
        return dataclasses.replace(
            self,
            query_ids=tuple(self.query_ids[position] for position in indices),
            input_ids=self.input_ids[index],
            observations=self.observations[:, index],
            auxiliary={key: value[:, index] for key, value in self.auxiliary.items()},
        )


def save_trace_bundle(bundle: TraceBundle, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensor_path = directory / "traces.safetensors"
    tensors = {
        "input_ids": bundle.input_ids.cpu().contiguous(),
        "observations": bundle.observations.cpu().contiguous(),
    }
    tensors.update(
        {
            f"auxiliary.{key}": value.cpu().contiguous()
            for key, value in sorted(bundle.auxiliary.items())
        }
    )
    save_file(
        tensors,
        str(tensor_path),
        metadata={
            "channel": bundle.channel.value,
            "target_count": str(len(bundle.target_ids)),
            "query_count": str(len(bundle.query_ids)),
        },
    )
    manifest = {
        "schema_version": 2,
        "channel": bundle.channel.value,
        "feature_dim": bundle.feature_dim,
        "target_ids": list(bundle.target_ids),
        "query_ids": list(bundle.query_ids),
        "auxiliary_keys": sorted(bundle.auxiliary),
        "tensor_file": tensor_path.name,
        "tensor_sha256": file_sha256(tensor_path),
        "metadata": bundle.metadata,
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def load_trace_bundle(directory: str | Path, *, verify: bool = True) -> TraceBundle:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    tensor_path = directory / manifest["tensor_file"]
    if verify and file_sha256(tensor_path) != manifest["tensor_sha256"]:
        raise ValueError(f"Trace tensor checksum mismatch: {tensor_path}")
    tensors = load_file(str(tensor_path))
    auxiliary = {
        key.removeprefix("auxiliary."): value
        for key, value in tensors.items()
        if key.startswith("auxiliary.")
    }
    return TraceBundle(
        target_ids=tuple(manifest["target_ids"]),
        query_ids=tuple(manifest["query_ids"]),
        input_ids=tensors["input_ids"].long(),
        observations=tensors["observations"].float(),
        channel=ObservationChannel(manifest["channel"]),
        feature_dim=int(manifest["feature_dim"]),
        auxiliary=auxiliary,
        metadata=dict(manifest.get("metadata", {})),
    )
