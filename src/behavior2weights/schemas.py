from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ObservationChannel(StrEnum):
    TOKENS = "tokens"
    SAMPLE_HISTOGRAM = "sample_histogram"
    TOPK = "topk"
    LOGITS = "logits"
    LOGIT_SKETCH = "logit_sketch"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    OOD = "ood"


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InterventionRecord(StrictSchema):
    kind: str
    label: str
    layer: int | None = None
    head: int | None = None
    neuron: int | None = None
    tensor_name: str | None = None
    rank: int | None = None
    scale: float | None = None
    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TargetRecord(StrictSchema):
    target_id: str
    family_id: str
    lineage_id: str
    architecture_id: str
    task_id: str
    dataset_id: str
    seed: int
    checkpoint_path: Path
    checkpoint_sha256: str | None = None
    training_step: int | None = None
    split: Split | None = None
    parent_target_id: str | None = None
    interventions: list[InterventionRecord] = Field(default_factory=list)
    factors: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def convert_checkpoint_path(cls, value: Any) -> Path:
        return Path(value)


class QueryRecord(StrictSchema):
    query_id: str
    input_ids: list[int]
    source: str
    partition: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceRecord(StrictSchema):
    target_id: str
    query_id: str
    channel: ObservationChannel
    input_ids: list[int]
    observation: list[float]
    output_ids: list[int] = Field(default_factory=list)
    topk_indices: list[int] = Field(default_factory=list)
    topk_values: list[float] = Field(default_factory=list)
    temperature: float = 1.0
    sample_count: int = 1
    rng_seed: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResultRecord(StrictSchema):
    run_id: str
    target_id: str
    lineage_id: str
    split: Split
    tier: str
    method: str
    channel: ObservationChannel
    query_policy: str = "random"
    query_budget: int
    replicate: int
    metric: str
    value: float
    metadata: dict[str, Any] = Field(default_factory=dict)
