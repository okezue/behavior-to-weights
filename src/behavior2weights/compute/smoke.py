from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from typing import Any

import torch

from behavior2weights.evaluation.metrics import functional_metrics, normalized_rmse
from behavior2weights.models.inverse import BehaviorToWeights, InverseModelConfig
from behavior2weights.models.micro_transformer import MicroTransformer, MicroTransformerConfig
from behavior2weights.probes.query_bank import QueryBankConfig, build_query_bank, save_query_bank
from behavior2weights.schemas import ObservationChannel, Split
from behavior2weights.targets.micro import MicroTransformerAdapter
from behavior2weights.traces.collector import CollectionConfig, collect_target_traces
from behavior2weights.traces.observations import ObservationConfig
from behavior2weights.tracking.jsonl import JsonlTracker
from behavior2weights.train.corpus import InverseTrainingCorpus
from behavior2weights.train.inverse import (
    InverseTrainingConfig,
    load_inverse_checkpoint,
    train_inverse_model,
)
from behavior2weights.utils import atomic_write_json, seed_everything
from behavior2weights.zoo.manifest import SplitPolicy, manifest_summary
from behavior2weights.zoo.micro import (
    InterventionSpec,
    MicroZooConfig,
    OptimizerSpec,
    build_micro_zoo,
)


def run_smoke_pipeline(output_directory: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    # Tiny CPU kernels become dramatically slower under large default OpenMP thread pools.
    torch.set_num_threads(1)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    output = Path(output_directory)
    if overwrite:
        shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(20260621)

    architecture = MicroTransformerConfig(
        vocab_size=16,
        max_seq_len=8,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
    )
    zoo_config = MicroZooConfig(
        architectures=(architecture,),
        tasks=("markov",),
        model_seeds=tuple(range(8)),
        dataset_seeds=(11,),
        optimizers=(OptimizerSpec(name="adamw", learning_rate=5e-3),),
        train_steps=6,
        checkpoint_steps=(0, 6),
        batch_size=32,
        train_examples=256,
        validation_examples=64,
        test_examples=64,
        interventions=(InterventionSpec(kind="attention_head_ablation", count=1),),
    )
    zoo = build_micro_zoo(
        zoo_config,
        output / "zoo",
        split_policy=SplitPolicy(
            train_fraction=0.625, validation_fraction=0.125, test_fraction=0.25
        ),
    )

    query_config = QueryBankConfig(
        vocab_size=architecture.vocab_size,
        seq_len=architecture.max_seq_len - 1,
        random_queries=16,
        natural_queries=16,
        seed=303,
    )
    queries = build_query_bank(query_config)
    save_query_bank(queries, output / "queries.jsonl")
    adapter = MicroTransformerAdapter(manifest_root=output / "zoo")
    trace_bundle = collect_target_traces(
        zoo.records,
        queries,
        adapter.load,
        ObservationConfig(
            channel=ObservationChannel.LOGITS,
            vocab_size=architecture.vocab_size,
            feature_dim=architecture.vocab_size,
        ),
        CollectionConfig(batch_size=64, device="cpu", base_seed=91),
        output_directory=output / "traces" / "logits",
    )

    corpus = InverseTrainingCorpus(
        zoo.records,
        trace_bundle,
        architecture_id=zoo.records[0].architecture_id,
        manifest_root=output / "zoo",
        canonicalize=True,
    )
    inverse_config = InverseModelConfig(
        vocab_size=architecture.vocab_size,
        max_seq_len=architecture.max_seq_len,
        observation_dim=architecture.vocab_size,
        trace_width=16,
        trace_heads=2,
        query_layers=1,
        set_layers=1,
        latent_dim=16,
        address_width=16,
        decoder_width=32,
        decoder_layers=2,
        max_tensors=64,
        max_layers=8,
        dropout=0.0,
    )
    tracker = JsonlTracker(output / "tracking", experiment="smoke", run_name="inverse")
    training_config = InverseTrainingConfig(
        steps=12,
        batch_size=4,
        query_budgets=(8, 16),
        coordinate_count=128,
        learning_rate=1e-3,
        warmup_steps=2,
        validation_every=4,
        validation_batches=2,
        early_stopping_patience=10,
        seed=44,
        device="cpu",
    )
    result = train_inverse_model(
        BehaviorToWeights(inverse_config),
        corpus,
        training_config,
        output / "inverse",
        tracker=tracker,
    )

    model, standardizer = load_inverse_checkpoint(result.best_checkpoint)
    test_indices = corpus.indices_for_split(Split.TEST)
    if not test_indices:
        raise RuntimeError("smoke split did not produce a test target")
    target_index = test_indices[0]
    query_indices = torch.arange(len(trace_bundle.query_ids))
    trace_row = int(corpus.trace_indices[target_index].item())
    input_ids = trace_bundle.input_ids[query_indices].unsqueeze(0)
    observations = trace_bundle.observations[trace_row, query_indices].unsqueeze(0)
    channel_id = list(ObservationChannel).index(trace_bundle.channel)
    channel_ids = torch.full((1, len(query_indices)), channel_id, dtype=torch.long)
    with torch.no_grad():
        latent = model.encode(input_ids, observations, channel_ids)
        standardized_mean, _ = model.decode_all(latent, corpus.address_space, chunk_size=1_024)
    prediction_vector = standardizer.inverse_transform(
        standardized_mean.squeeze(0), corpus.role_ids
    )
    target_vector = corpus.target_vector(target_index)

    predicted_state = corpus.address_space.unvectorize(
        prediction_vector,
        template=corpus.targets[target_index].state_dict,
    )
    prediction_model = MicroTransformer(architecture)
    prediction_model.load_state_dict(predicted_state, strict=False)
    target_model = MicroTransformer(architecture)
    target_model.load_state_dict(corpus.targets[target_index].state_dict, strict=False)
    holdout_generator = torch.Generator().manual_seed(999)
    holdout_queries = torch.randint(
        architecture.vocab_size,
        (64, architecture.max_seq_len - 1),
        generator=holdout_generator,
    )
    functional = functional_metrics(prediction_model, target_model, holdout_queries)
    train_vectors = torch.stack(
        [corpus.target_vector(index) for index in corpus.indices_for_split(Split.TRAIN)]
    )
    mean_baseline = train_vectors.mean(dim=0)

    report = {
        "status": "completed",
        "zoo": manifest_summary(zoo.records),
        "trace_shape": list(trace_bundle.observations.shape),
        "inverse_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "target_parameters": corpus.address_space.total_parameters,
        "training": {
            "steps_completed": result.steps_completed,
            "best_validation_nll": result.best_validation_nll,
        },
        "evaluation_target": corpus.targets[target_index].record.target_id,
        "weight_nrmse": normalized_rmse(prediction_vector, target_vector),
        "mean_checkpoint_nrmse": normalized_rmse(mean_baseline, target_vector),
        "functional": functional,
        "note": "Smoke results validate execution only; they are not powered scientific estimates.",
    }
    atomic_write_json(output / "smoke_report.json", report)
    return report
