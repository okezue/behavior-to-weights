from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from behavior2weights.models.micro_transformer import MicroTransformerConfig
from behavior2weights.schemas import Split
from behavior2weights.zoo.manifest import SplitPolicy
from behavior2weights.zoo.micro import MicroZooConfig
from behavior2weights.zoo.public import PublicManifestConfig
from behavior2weights.zoo.text import TextZooConfig


@dataclasses.dataclass(frozen=True, slots=True)
class ArchitectureProjection:
    architecture_index: int
    configuration: dict[str, Any]
    parameters: int
    raw_parameter_bytes_fp32: int
    lineages: int
    checkpoint_targets: int
    intervention_targets: int
    total_targets: int
    projected_raw_checkpoint_bytes_fp32: int


@dataclasses.dataclass(frozen=True, slots=True)
class ZooProjection:
    kind: str
    name: str | None
    architectures: tuple[ArchitectureProjection, ...]
    lineages: int
    checkpoint_targets: int
    intervention_targets: int
    targets: int
    checkpoints_per_lineage: int | None
    interventions_per_lineage: int | None
    targets_per_lineage: int | None
    projected_raw_checkpoint_bytes_fp32: int
    lineage_splits: dict[str, int]
    target_splits: dict[str, int]
    factor_cardinalities: dict[str, int]
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["projected_raw_checkpoint_gib_fp32"] = self.projected_raw_checkpoint_bytes_fp32 / (
            1024**3
        )
        for architecture in payload["architectures"]:
            architecture["projected_raw_checkpoint_gib_fp32"] = architecture[
                "projected_raw_checkpoint_bytes_fp32"
            ] / (1024**3)
        return payload


def micro_parameter_count(config: MicroTransformerConfig) -> int:
    """Count unique trainable parameters without instantiating the model."""
    width = config.d_model
    embeddings = config.vocab_size * width + config.max_seq_len * width
    attention_weights = 4 * width * width
    mlp_weights = 2 * width * config.d_ff
    layer_norms = 4 * width
    linear_biases = (5 * width + config.d_ff) if config.bias else 0
    per_block = attention_weights + mlp_weights + layer_norms + linear_biases
    final_layer_norm = 2 * width
    output_projection = 0 if config.tie_embeddings else config.vocab_size * width
    return embeddings + config.n_layers * per_block + final_layer_norm + output_projection


def _split_counts(factor_rows: Sequence[Mapping[str, Any]], policy: SplitPolicy) -> dict[str, int]:
    ood = 0
    in_distribution = 0
    for factors in factor_rows:
        is_ood = any(
            str(factors.get(factor)) in {str(value) for value in values}
            for factor, values in policy.ood_factor_values.items()
        )
        if is_ood:
            ood += 1
        else:
            in_distribution += 1
    validation = round(in_distribution * policy.validation_fraction)
    test = round(in_distribution * policy.test_fraction)
    if in_distribution >= 3 and policy.validation_fraction > 0:
        validation = max(1, validation)
    if in_distribution >= 3 and policy.test_fraction > 0:
        test = max(1, test)
    if validation + test >= in_distribution:
        overflow = validation + test - max(in_distribution - 1, 0)
        test = max(0, test - overflow)
    train = in_distribution - validation - test
    return {
        Split.TRAIN.value: train,
        Split.VALIDATION.value: validation,
        Split.TEST.value: test,
        Split.OOD.value: ood,
    }


def _factor_cardinalities(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    values: dict[str, set[str]] = {}
    for row in rows:
        for key, value in row.items():
            values.setdefault(key, set()).add(str(value))
    return {key: len(unique) for key, unique in sorted(values.items())}


def plan_micro_zoo(
    config: MicroZooConfig,
    *,
    split_policy: SplitPolicy | None = None,
    name: str | None = None,
) -> ZooProjection:
    split_policy = split_policy or SplitPolicy()
    checkpoints_per_lineage = len(config.checkpoint_steps)
    interventions_per_lineage = sum(spec.count for spec in config.interventions)
    targets_per_lineage = checkpoints_per_lineage + interventions_per_lineage
    architecture_plans: list[ArchitectureProjection] = []
    factor_rows: list[dict[str, Any]] = []

    lineages_per_architecture = (
        len(config.tasks)
        * len(config.model_seeds)
        * len(config.dataset_seeds)
        * len(config.optimizers)
    )
    for architecture_index, architecture in enumerate(config.architectures):
        parameter_count = micro_parameter_count(architecture)
        lineages = lineages_per_architecture
        checkpoint_targets = lineages * checkpoints_per_lineage
        intervention_targets = lineages * interventions_per_lineage
        total_targets = checkpoint_targets + intervention_targets
        architecture_plans.append(
            ArchitectureProjection(
                architecture_index=architecture_index,
                configuration=dataclasses.asdict(architecture),
                parameters=parameter_count,
                raw_parameter_bytes_fp32=parameter_count * 4,
                lineages=lineages,
                checkpoint_targets=checkpoint_targets,
                intervention_targets=intervention_targets,
                total_targets=total_targets,
                projected_raw_checkpoint_bytes_fp32=total_targets * parameter_count * 4,
            )
        )
        for task in config.tasks:
            for model_seed in config.model_seeds:
                for dataset_seed in config.dataset_seeds:
                    for optimizer in config.optimizers:
                        factor_rows.append(
                            {
                                "architecture_index": architecture_index,
                                "d_model": architecture.d_model,
                                "n_layers": architecture.n_layers,
                                "n_heads": architecture.n_heads,
                                "d_ff": architecture.d_ff,
                                "task": task,
                                "model_seed": model_seed,
                                "dataset_seed": dataset_seed,
                                "optimizer": optimizer.name,
                                "learning_rate": optimizer.learning_rate,
                            }
                        )

    lineage_splits = _split_counts(factor_rows, split_policy)
    target_splits = {split: count * targets_per_lineage for split, count in lineage_splits.items()}
    return ZooProjection(
        kind="micro",
        name=name,
        architectures=tuple(architecture_plans),
        lineages=len(factor_rows),
        checkpoint_targets=sum(item.checkpoint_targets for item in architecture_plans),
        intervention_targets=sum(item.intervention_targets for item in architecture_plans),
        targets=sum(item.total_targets for item in architecture_plans),
        checkpoints_per_lineage=checkpoints_per_lineage,
        interventions_per_lineage=interventions_per_lineage,
        targets_per_lineage=targets_per_lineage,
        projected_raw_checkpoint_bytes_fp32=sum(
            item.projected_raw_checkpoint_bytes_fp32 for item in architecture_plans
        ),
        lineage_splits=lineage_splits,
        target_splits=target_splits,
        factor_cardinalities=_factor_cardinalities(factor_rows),
        notes=(
            "Storage is the raw unique-parameter payload at FP32; optimizer state, manifests, "
            "safetensors headers, traces, and inverse-model artifacts are excluded.",
            "Lineages, not checkpoints or interventions, are the independent statistical units.",
        ),
    )


def plan_text_zoo(
    config: TextZooConfig,
    *,
    split_policy: SplitPolicy | None = None,
    name: str | None = None,
) -> ZooProjection:
    split_policy = split_policy or SplitPolicy()
    checkpoints_per_lineage = len(config.checkpoint_steps)
    interventions_per_lineage = sum(spec.count for spec in config.interventions)
    targets_per_lineage = checkpoints_per_lineage + interventions_per_lineage
    lineages_per_architecture = (
        len(config.model_seeds) * len(config.data_order_seeds) * len(config.optimizers)
    )
    architecture_plans: list[ArchitectureProjection] = []
    factor_rows: list[dict[str, Any]] = []
    for architecture_index, architecture in enumerate(config.architectures):
        parameter_count = micro_parameter_count(architecture)
        lineages = lineages_per_architecture
        checkpoint_targets = lineages * checkpoints_per_lineage
        intervention_targets = lineages * interventions_per_lineage
        total_targets = checkpoint_targets + intervention_targets
        architecture_plans.append(
            ArchitectureProjection(
                architecture_index=architecture_index,
                configuration=dataclasses.asdict(architecture),
                parameters=parameter_count,
                raw_parameter_bytes_fp32=parameter_count * 4,
                lineages=lineages,
                checkpoint_targets=checkpoint_targets,
                intervention_targets=intervention_targets,
                total_targets=total_targets,
                projected_raw_checkpoint_bytes_fp32=total_targets * parameter_count * 4,
            )
        )
        for model_seed in config.model_seeds:
            for data_order_seed in config.data_order_seeds:
                for optimizer in config.optimizers:
                    factor_rows.append(
                        {
                            "architecture_index": architecture_index,
                            "d_model": architecture.d_model,
                            "n_layers": architecture.n_layers,
                            "n_heads": architecture.n_heads,
                            "d_ff": architecture.d_ff,
                            "model_seed": model_seed,
                            "data_order_seed": data_order_seed,
                            "optimizer": optimizer.name,
                            "learning_rate": optimizer.learning_rate,
                        }
                    )
    lineage_splits = _split_counts(factor_rows, split_policy)
    target_splits = {split: count * targets_per_lineage for split, count in lineage_splits.items()}
    return ZooProjection(
        kind="text",
        name=name,
        architectures=tuple(architecture_plans),
        lineages=len(factor_rows),
        checkpoint_targets=sum(item.checkpoint_targets for item in architecture_plans),
        intervention_targets=sum(item.intervention_targets for item in architecture_plans),
        targets=sum(item.total_targets for item in architecture_plans),
        checkpoints_per_lineage=checkpoints_per_lineage,
        interventions_per_lineage=interventions_per_lineage,
        targets_per_lineage=targets_per_lineage,
        projected_raw_checkpoint_bytes_fp32=sum(
            item.projected_raw_checkpoint_bytes_fp32 for item in architecture_plans
        ),
        lineage_splits=lineage_splits,
        target_splits=target_splits,
        factor_cardinalities=_factor_cardinalities(factor_rows),
        notes=(
            "Storage is the raw unique-parameter payload at FP32; optimizer state, manifests, "
            "safetensors headers, traces, and inverse-model artifacts are excluded.",
            "Dataset preparation/tokenizer storage and training FLOPs are not included.",
        ),
    )


def plan_public_zoo(
    config: PublicManifestConfig,
    *,
    name: str | None = None,
) -> ZooProjection:
    lineages = len({(model.model_name, model.external_family) for model in config.models})
    targets = sum(len(model.revisions) for model in config.models)
    split_targets = Counter[str]()
    split_lineages: dict[str, set[tuple[str, str]]] = {split.value: set() for split in Split}
    parameter_bytes = 0
    architecture_rows: list[ArchitectureProjection] = []
    for architecture_index, model in enumerate(config.models):
        model_targets = len(model.revisions)
        split_targets[model.split.value] += model_targets
        split_lineages[model.split.value].add((model.model_name, model.external_family))
        parameter_count = int(model.parameter_count or 0)
        raw_bytes = parameter_count * 4
        parameter_bytes += raw_bytes * model_targets
        architecture_rows.append(
            ArchitectureProjection(
                architecture_index=architecture_index,
                configuration={
                    "model_name": model.model_name,
                    "external_family": model.external_family,
                    "architecture_id": model.architecture_id,
                    "revisions": list(model.revisions),
                },
                parameters=parameter_count,
                raw_parameter_bytes_fp32=raw_bytes,
                lineages=1,
                checkpoint_targets=model_targets,
                intervention_targets=0,
                total_targets=model_targets,
                projected_raw_checkpoint_bytes_fp32=raw_bytes * model_targets,
            )
        )
    return ZooProjection(
        kind="public",
        name=name,
        architectures=tuple(architecture_rows),
        lineages=lineages,
        checkpoint_targets=targets,
        intervention_targets=0,
        targets=targets,
        checkpoints_per_lineage=None,
        interventions_per_lineage=None,
        targets_per_lineage=None,
        projected_raw_checkpoint_bytes_fp32=parameter_bytes,
        lineage_splits={split: len(values) for split, values in split_lineages.items()},
        target_splits={split.value: split_targets[split.value] for split in Split},
        factor_cardinalities={
            "external_family": len({model.external_family for model in config.models}),
            "model_name": len({model.model_name for model in config.models}),
            "revision": len({revision for model in config.models for revision in model.revisions}),
        },
        notes=(
            "Zero-valued parameter/storage estimates indicate a model spec omitted parameter_count.",
            "Public-model revisions are observational targets, not assumed independent lineages.",
        ),
    )


def plan_zoo_config(raw: Mapping[str, Any], *, kind: str = "auto") -> ZooProjection:
    resolved = kind
    if resolved == "auto":
        if "models" in raw:
            resolved = "public"
        elif "dataset_directory" in raw:
            resolved = "text"
        elif "zoo" in raw:
            zoo = raw["zoo"]
            resolved = "micro" if "tasks" in zoo or "dataset_seeds" in zoo else "text"
        else:
            raise ValueError("Could not infer zoo kind; pass --kind micro, text, or public")
    if resolved == "public":
        return plan_public_zoo(PublicManifestConfig.from_dict(dict(raw)), name=raw.get("name"))
    policy = SplitPolicy(**raw.get("split_policy", {}))
    zoo_raw = raw.get("zoo", raw)
    if resolved == "micro":
        return plan_micro_zoo(
            MicroZooConfig.from_dict(zoo_raw), split_policy=policy, name=raw.get("name")
        )
    if resolved == "text":
        return plan_text_zoo(
            TextZooConfig.from_dict(zoo_raw), split_policy=policy, name=raw.get("name")
        )
    raise ValueError(f"Unsupported zoo kind: {resolved}")
