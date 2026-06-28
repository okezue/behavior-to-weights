from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from behavior2weights.baselines.behavior import mean_checkpoint, nearest_behavior_neighbor
from behavior2weights.data.synthetic import SyntheticDatasetConfig, generate_sequences
from behavior2weights.evaluation.alignment import align_micro_state_dict
from behavior2weights.evaluation.metrics import (
    cosine_similarity,
    functional_metrics,
    normalized_rmse,
    posterior_calibration,
)
from behavior2weights.models.inverse import CHANNEL_TO_ID
from behavior2weights.models.micro_transformer import MicroTransformer
from behavior2weights.probes.active import training_population_order
from behavior2weights.schemas import ResultRecord, Split
from behavior2weights.traces.store import load_trace_bundle
from behavior2weights.train.corpus import InverseTrainingCorpus
from behavior2weights.train.inverse import load_inverse_checkpoint
from behavior2weights.utils import stable_hash, write_jsonl
from behavior2weights.zoo.manifest import load_manifest


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationConfig:
    query_budgets: tuple[int, ...] = (8, 16, 32, 64)
    query_policies: tuple[str, ...] = ("random", "population_disagreement")
    splits: tuple[Split, ...] = (Split.TEST, Split.OOD)
    functional_examples: int = 256
    functional_batch_size: int = 128
    decoder_chunk_size: int = 65_536
    seed: int = 20260621
    replicate: int = 0
    tier: str = "tier1"
    run_id: str = "evaluation"
    include_mean_baseline: bool = True
    include_nearest_baseline: bool = True
    canonicalize: bool = True

    def __post_init__(self) -> None:
        if not self.query_budgets or min(self.query_budgets) <= 0:
            raise ValueError("query_budgets must contain positive values")
        if self.functional_examples <= 0:
            raise ValueError("functional_examples must be positive")
        allowed = {"random", "population_disagreement"}
        if not self.query_policies or set(self.query_policies) - allowed:
            raise ValueError(f"query_policies must be a non-empty subset of {sorted(allowed)}")


def _raw_log_variance(
    standardizer: Any, standardized: torch.Tensor, role_ids: torch.Tensor
) -> torch.Tensor:
    scales = torch.tensor(
        [standardizer.scales[int(role)] for role in role_ids.tolist()], dtype=standardized.dtype
    )
    return standardized + 2.0 * scales.log()


def _holdout_queries(
    corpus: InverseTrainingCorpus, target_index: int, count: int, seed: int
) -> torch.Tensor:
    target = corpus.targets[target_index]
    config = corpus.model_config
    task = target.record.task_id
    if task in {"markov", "copy", "induction", "modular_addition", "mixture"}:
        dataset_config = SyntheticDatasetConfig(
            task=task,  # type: ignore[arg-type]
            vocab_size=config.vocab_size,
            seq_len=config.max_seq_len,
            train_examples=1,
            validation_examples=1,
            test_examples=1,
            seed=seed,
        )
        return generate_sequences(dataset_config, count, seed=seed)[:, :-1]
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        config.vocab_size,
        (count, config.max_seq_len - 1),
        generator=generator,
    )


def _model_from_vector(corpus: InverseTrainingCorpus, vector: torch.Tensor) -> MicroTransformer:
    state = corpus.address_space.unvectorize(vector, template=corpus.targets[0].state_dict)
    model = MicroTransformer(corpus.model_config)
    model.load_state_dict(state, strict=True)
    return model


def _aligned_vector(
    corpus: InverseTrainingCorpus, prediction: torch.Tensor, target_index: int
) -> torch.Tensor:
    prediction_state = corpus.address_space.unvectorize(
        prediction, template=corpus.targets[target_index].state_dict
    )
    aligned, _ = align_micro_state_dict(
        prediction_state,
        corpus.targets[target_index].state_dict,
        corpus.model_config,
    )
    return corpus.address_space.vectorize(aligned)


def _result_rows(
    *,
    config: EvaluationConfig,
    target_index: int,
    corpus: InverseTrainingCorpus,
    method: str,
    query_policy: str,
    budget: int,
    vector: torch.Tensor,
    holdout: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> list[ResultRecord]:
    target = corpus.targets[target_index]
    target_vector = target.vector
    aligned = _aligned_vector(corpus, vector, target_index)
    model = _model_from_vector(corpus, vector)
    target_model = _model_from_vector(corpus, target_vector)
    functional = functional_metrics(
        model,
        target_model,
        holdout,
        batch_size=config.functional_batch_size,
    )
    metrics = {
        "weight_nrmse": normalized_rmse(vector, target_vector),
        "weight_cosine": cosine_similarity(vector, target_vector),
        "orbit_weight_nrmse": normalized_rmse(aligned, target_vector),
        "orbit_weight_cosine": cosine_similarity(aligned, target_vector),
        **{f"functional_{key}": value for key, value in functional.items()},
    }
    return [
        ResultRecord(
            run_id=config.run_id,
            target_id=target.record.target_id,
            lineage_id=target.record.lineage_id,
            split=target.record.split or Split.TEST,
            tier=config.tier,
            method=method,
            channel=corpus.traces.channel,
            query_policy=query_policy,
            query_budget=budget,
            replicate=config.replicate,
            metric=name,
            value=float(value),
            metadata=metadata or {},
        )
        for name, value in metrics.items()
    ]


@torch.no_grad()
def evaluate_micro_inverse(
    *,
    manifest_path: str | Path,
    traces_directory: str | Path,
    checkpoint_directory: str | Path,
    output_path: str | Path,
    architecture_id: str | None = None,
    config: EvaluationConfig | None = None,
    device: str = "cpu",
) -> list[ResultRecord]:
    """Locked, target-level evaluation for one compatible micro-transformer architecture."""

    config = config or EvaluationConfig()
    records = load_manifest(manifest_path, resolve_paths=False)
    traces = load_trace_bundle(traces_directory)
    corpus = InverseTrainingCorpus(
        records,
        traces,
        architecture_id=architecture_id,
        manifest_root=Path(manifest_path).parent,
        canonicalize=config.canonicalize,
    )
    model, standardizer = load_inverse_checkpoint(checkpoint_directory, device=device)
    model.eval()
    budgets = tuple(sorted(set(config.query_budgets)))
    policies = tuple(dict.fromkeys(config.query_policies))
    if max(budgets) > len(traces.query_ids):
        raise ValueError("a requested query budget exceeds the trace bank")
    train_indices = corpus.indices_for_split(Split.TRAIN)
    evaluation_indices = [
        index for split in config.splits for index in corpus.indices_for_split(split)
    ]
    if not train_indices:
        raise ValueError("evaluation requires training targets for baselines and query ranking")
    if not evaluation_indices:
        raise ValueError("no targets match requested evaluation splits")

    train_vectors = torch.stack([corpus.target_vector(index) for index in train_indices])
    population_mean = mean_checkpoint(train_vectors)
    train_trace_rows = corpus.trace_indices[torch.tensor(train_indices)]
    training_observations = traces.observations[train_trace_rows]
    training_lineages = [corpus.targets[index].record.lineage_id for index in train_indices]
    policy_orders = {
        policy: training_population_order(
            training_observations,
            training_lineages,
            policy=policy,
            seed=config.seed + policy_index * 1_000_003,
        ).indices
        for policy_index, policy in enumerate(policies)
    }

    rows: list[ResultRecord] = []
    predicted_cache: dict[tuple[str, int, int], torch.Tensor] = {}
    target_id_to_index = {
        target.record.target_id: index for index, target in enumerate(corpus.targets)
    }

    for query_policy in policies:
        order = policy_orders[query_policy]
        for budget in budgets:
            query_indices = order[:budget]
            query_hash = stable_hash(query_indices.tolist(), length=32)
            for target_index in evaluation_indices:
                trace_row = int(corpus.trace_indices[target_index].item())
                input_ids = traces.input_ids[query_indices].unsqueeze(0).to(device)
                observations = traces.observations[trace_row, query_indices].unsqueeze(0).to(device)
                channel_ids = torch.full(
                    (1, budget),
                    CHANNEL_TO_ID[traces.channel],
                    dtype=torch.long,
                    device=device,
                )
                latent = model.encode(input_ids, observations, channel_ids)
                standardized_mean, standardized_log_variance = model.decode_all(
                    latent, corpus.address_space, chunk_size=config.decoder_chunk_size
                )
                prediction = standardizer.inverse_transform(
                    standardized_mean.squeeze(0), corpus.role_ids
                )
                predicted_cache[(query_policy, target_index, budget)] = prediction
                holdout = _holdout_queries(
                    corpus,
                    target_index,
                    config.functional_examples,
                    config.seed + 17 * target_index + budget,
                )
                common_metadata = {
                    "query_index_hash": query_hash,
                    "architecture_id": corpus.records[0].architecture_id,
                    "query_policy_fit_split": "train",
                }
                rows.extend(
                    _result_rows(
                        config=config,
                        target_index=target_index,
                        corpus=corpus,
                        method="inverse_posterior_mean",
                        query_policy=query_policy,
                        budget=budget,
                        vector=prediction,
                        holdout=holdout,
                        metadata=common_metadata,
                    )
                )
                raw_log_variance = _raw_log_variance(
                    standardizer,
                    standardized_log_variance.squeeze(0),
                    corpus.role_ids,
                )
                calibration = posterior_calibration(
                    corpus.target_vector(target_index), prediction, raw_log_variance
                )
                for name, value in calibration.items():
                    rows.append(
                        ResultRecord(
                            run_id=config.run_id,
                            target_id=corpus.targets[target_index].record.target_id,
                            lineage_id=corpus.targets[target_index].record.lineage_id,
                            split=corpus.targets[target_index].record.split or Split.TEST,
                            tier=config.tier,
                            method="inverse_posterior_mean",
                            channel=traces.channel,
                            query_policy=query_policy,
                            query_budget=budget,
                            replicate=config.replicate,
                            metric=f"posterior_{name}",
                            value=float(value),
                            metadata=common_metadata,
                        )
                    )
                if config.include_mean_baseline:
                    rows.extend(
                        _result_rows(
                            config=config,
                            target_index=target_index,
                            corpus=corpus,
                            method="mean_checkpoint",
                            query_policy=query_policy,
                            budget=budget,
                            vector=population_mean,
                            holdout=holdout,
                            metadata=common_metadata,
                        )
                    )
                if config.include_nearest_baseline:
                    nearest_local = nearest_behavior_neighbor(
                        traces.observations[train_trace_rows][:, query_indices],
                        traces.observations[trace_row, query_indices],
                    )
                    nearest_index = train_indices[nearest_local]
                    rows.extend(
                        _result_rows(
                            config=config,
                            target_index=target_index,
                            corpus=corpus,
                            method="nearest_behavior_neighbor",
                            query_policy=query_policy,
                            budget=budget,
                            vector=corpus.target_vector(nearest_index),
                            holdout=holdout,
                            metadata={
                                **common_metadata,
                                "neighbor_target_id": corpus.targets[
                                    nearest_index
                                ].record.target_id,
                            },
                        )
                    )

            # Controlled-edit localization compares independently inferred parent/child deltas.
            for child_index in evaluation_indices:
                child = corpus.targets[child_index]
                parent_id = child.record.parent_target_id
                if not parent_id or parent_id not in target_id_to_index:
                    continue
                parent_index = target_id_to_index[parent_id]
                cache_key = (query_policy, parent_index, budget)
                if cache_key not in predicted_cache:
                    trace_row = int(corpus.trace_indices[parent_index].item())
                    input_ids = traces.input_ids[query_indices].unsqueeze(0).to(device)
                    observations = (
                        traces.observations[trace_row, query_indices].unsqueeze(0).to(device)
                    )
                    channel_ids = torch.full(
                        (1, budget),
                        CHANNEL_TO_ID[traces.channel],
                        dtype=torch.long,
                        device=device,
                    )
                    latent = model.encode(input_ids, observations, channel_ids)
                    mean, _ = model.decode_all(
                        latent,
                        corpus.address_space,
                        chunk_size=config.decoder_chunk_size,
                    )
                    predicted_cache[cache_key] = standardizer.inverse_transform(
                        mean.squeeze(0), corpus.role_ids
                    )
                true_delta = (child.vector - corpus.target_vector(parent_index)).abs()
                predicted_delta = (
                    predicted_cache[(query_policy, child_index, budget)]
                    - predicted_cache[cache_key]
                ).abs()
                labels = (true_delta > 1e-12).cpu().numpy().astype(int)
                scores = predicted_delta.cpu().numpy()
                if labels.min() == labels.max():
                    continue
                localization = {
                    "edit_coordinate_average_precision": average_precision_score(labels, scores),
                    "edit_coordinate_auroc": roc_auc_score(labels, scores),
                }
                changed = int(labels.sum())
                topk = torch.topk(predicted_delta, min(changed, predicted_delta.numel())).indices
                localization["edit_coordinate_recall_at_true_k"] = float(
                    labels[topk.cpu().numpy()].mean()
                )
                for name, value in localization.items():
                    rows.append(
                        ResultRecord(
                            run_id=config.run_id,
                            target_id=child.record.target_id,
                            lineage_id=child.record.lineage_id,
                            split=child.record.split or Split.TEST,
                            tier=config.tier,
                            method="inverse_delta_localization",
                            channel=traces.channel,
                            query_policy=query_policy,
                            query_budget=budget,
                            replicate=config.replicate,
                            metric=name,
                            value=float(value),
                            metadata={
                                "parent_target_id": parent_id,
                                "changed_coordinates": changed,
                                "query_index_hash": query_hash,
                            },
                        )
                    )

    write_jsonl(output_path, [row.model_dump(mode="json") for row in rows])
    summary_path = Path(output_path).with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "rows": len(rows),
                "targets": len(evaluation_indices),
                "lineages": len(
                    {corpus.targets[index].record.lineage_id for index in evaluation_indices}
                ),
                "budgets": list(budgets),
                "query_policies": list(policies),
                "channel": traces.channel.value,
                "architecture_id": corpus.records[0].architecture_id,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return rows
