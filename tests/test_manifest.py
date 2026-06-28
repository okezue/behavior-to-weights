from __future__ import annotations

from pathlib import Path

import pytest

from behavior2weights.schemas import Split, TargetRecord
from behavior2weights.zoo.manifest import SplitPolicy, assign_lineage_splits, validate_manifest


def record(target: str, lineage: str) -> TargetRecord:
    return TargetRecord(
        target_id=target,
        family_id="micro-transformer",
        lineage_id=lineage,
        architecture_id="a",
        task_id="markov",
        dataset_id="synthetic",
        seed=0,
        checkpoint_path=Path(f"{target}.safetensors"),
    )


def test_lineage_split_never_leaks() -> None:
    rows = [
        record(f"t-{lineage}-{checkpoint}", f"l-{lineage}")
        for lineage in range(12)
        for checkpoint in range(3)
    ]
    assigned = assign_lineage_splits(rows, SplitPolicy())
    mapping: dict[str, Split | None] = {}
    for row in assigned:
        mapping.setdefault(row.lineage_id, row.split)
        assert mapping[row.lineage_id] == row.split
    assert {row.split for row in assigned} >= {Split.TRAIN, Split.VALIDATION, Split.TEST}


def test_manifest_rejects_lineage_in_two_splits() -> None:
    rows = [
        record("a", "same").model_copy(update={"split": Split.TRAIN}),
        record("b", "same").model_copy(update={"split": Split.TEST}),
    ]
    with pytest.raises(ValueError, match="multiple splits"):
        validate_manifest(rows)
