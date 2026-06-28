from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from behavior2weights.schemas import Split, TargetRecord
from behavior2weights.utils import file_sha256, read_jsonl, stable_hash, write_jsonl


@dataclasses.dataclass(frozen=True, slots=True)
class SplitPolicy:
    train_fraction: float = 0.70
    validation_fraction: float = 0.10
    test_fraction: float = 0.20
    salt: str = "behavior2weights-v1"
    ood_factor_values: dict[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-8:
            raise ValueError("train, validation, and test fractions must sum to one")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) < 0:
            raise ValueError("split fractions cannot be negative")


def _is_ood(record: TargetRecord, policy: SplitPolicy) -> bool:
    for factor, values in policy.ood_factor_values.items():
        if str(record.factors.get(factor)) in values:
            return True
    return False


def assign_lineage_splits(
    records: Sequence[TargetRecord], policy: SplitPolicy | None = None
) -> list[TargetRecord]:
    """Assign every checkpoint from a lineage to the same split.

    Allocation is hash-sorted but uses explicit counts, rather than independent Bernoulli draws,
    so small pilot zoos still receive non-empty validation/test splits when possible.
    """

    policy = policy or SplitPolicy()
    ood_lineages = {record.lineage_id for record in records if _is_ood(record, policy)}
    lineages = sorted(
        {record.lineage_id for record in records if record.lineage_id not in ood_lineages},
        key=lambda value: stable_hash({"lineage": value, "salt": policy.salt}, length=64),
    )
    count = len(lineages)
    validation_count = round(count * policy.validation_fraction)
    test_count = round(count * policy.test_fraction)
    if count >= 3 and policy.validation_fraction > 0:
        validation_count = max(1, validation_count)
    if count >= 3 and policy.test_fraction > 0:
        test_count = max(1, test_count)
    if validation_count + test_count >= count:
        overflow = validation_count + test_count - max(count - 1, 0)
        test_count = max(0, test_count - overflow)
    train_count = count - validation_count - test_count
    mapping: dict[str, Split] = {}
    for lineage in lineages[:train_count]:
        mapping[lineage] = Split.TRAIN
    for lineage in lineages[train_count : train_count + validation_count]:
        mapping[lineage] = Split.VALIDATION
    for lineage in lineages[train_count + validation_count :]:
        mapping[lineage] = Split.TEST
    mapping.update({lineage: Split.OOD for lineage in ood_lineages})
    return [record.model_copy(update={"split": mapping[record.lineage_id]}) for record in records]


def validate_manifest(records: Sequence[TargetRecord], *, verify_files: bool = False) -> None:
    target_ids = [record.target_id for record in records]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("target_id values must be unique")
    lineage_split: dict[str, Split | None] = {}
    for record in records:
        previous = lineage_split.setdefault(record.lineage_id, record.split)
        if previous != record.split:
            raise ValueError(f"lineage {record.lineage_id} appears in multiple splits")
        if verify_files:
            if not record.checkpoint_path.exists():
                raise FileNotFoundError(record.checkpoint_path)
            if (
                record.checkpoint_sha256
                and file_sha256(record.checkpoint_path) != record.checkpoint_sha256
            ):
                raise ValueError(f"checksum mismatch for {record.checkpoint_path}")


def save_manifest(records: Sequence[TargetRecord], path: str | Path) -> None:
    validate_manifest(records)
    write_jsonl(path, [record.model_dump(mode="json") for record in records])


def load_manifest(path: str | Path, *, resolve_paths: bool = True) -> list[TargetRecord]:
    path = Path(path)
    records = [TargetRecord.model_validate(row) for row in read_jsonl(path)]
    if resolve_paths:
        records = [
            record.model_copy(
                update={
                    "checkpoint_path": record.checkpoint_path
                    if record.checkpoint_path.is_absolute()
                    else (path.parent / record.checkpoint_path).resolve()
                }
            )
            for record in records
        ]
    validate_manifest(records)
    return records


def manifest_summary(records: Iterable[TargetRecord]) -> dict[str, Any]:
    rows = list(records)
    return {
        "targets": len(rows),
        "lineages": len({row.lineage_id for row in rows}),
        "architectures": len({row.architecture_id for row in rows}),
        "families": sorted({row.family_id for row in rows}),
        "splits": {split.value: sum(row.split == split for row in rows) for split in Split},
        "tasks": sorted({row.task_id for row in rows}),
        "intervened_targets": sum(bool(row.interventions) for row in rows),
    }
