from __future__ import annotations

import json

import torch

from behavior2weights.analytic.softmax import (
    LinearSoftmaxExperimentConfig,
    canonical_softmax_parameters,
    probabilities_from_canonical,
    recover_linear_softmax,
    run_linear_softmax_experiment,
)


def test_full_rank_probability_queries_recover_softmax_orbit_exactly() -> None:
    generator = torch.Generator().manual_seed(5)
    input_dim = 3
    classes = 4
    weight = torch.randn(classes, input_dim, generator=generator, dtype=torch.float64)
    bias = torch.randn(classes, generator=generator, dtype=torch.float64)
    beta = canonical_softmax_parameters(weight, bias)
    inputs = torch.cat([torch.zeros(1, input_dim), torch.eye(input_dim)], dim=0).double()
    design = torch.cat([inputs, torch.ones(len(inputs), 1)], dim=1)
    probabilities = probabilities_from_canonical(design, beta)

    recovery = recover_linear_softmax(design, probabilities, observation_channel="probabilities")
    assert recovery.design_rank == input_dim + 1
    assert recovery.observable_dimension == recovery.orbit_dimension
    assert torch.allclose(recovery.recovered_beta, beta, atol=1e-10, rtol=1e-10)


def test_rank_deficient_queries_report_unobservable_dimensions() -> None:
    design = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    beta = torch.tensor([[1.0, -0.5], [0.2, 0.1]], dtype=torch.float64)
    probabilities = probabilities_from_canonical(design, beta)
    recovery = recover_linear_softmax(design, probabilities, observation_channel="probabilities")
    assert recovery.design_rank == 1
    assert recovery.observable_dimension < recovery.orbit_dimension
    assert recovery.design_condition_number == float("inf")


def test_tier0_runner_writes_strict_json_artifacts(tmp_path) -> None:
    summary = run_linear_softmax_experiment(
        LinearSoftmaxExperimentConfig(
            input_dims=(2,),
            class_counts=(3,),
            query_counts=(2, 3),
            query_strategies=("basis",),
            observation_channels=("probabilities",),
            systems_per_cell=2,
            holdout_queries=8,
        ),
        tmp_path,
    )
    assert summary["systems"] == 4
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert any(row["design_condition_number"] is None for row in rows)
    assert any(row["exact_recovery"] for row in rows)
