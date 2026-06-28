from __future__ import annotations

import torch

from behavior2weights.probes.active import training_population_order


def test_population_disagreement_uses_lineages_not_target_rows() -> None:
    # The two first rows are checkpoints from one lineage. Candidate 1 varies strongly between
    # those checkpoints but not between lineage means; candidate 0 genuinely separates lineages.
    observations = torch.tensor(
        [
            [[0.0], [0.0], [1.0]],
            [[0.0], [100.0], [1.0]],
            [[10.0], [50.0], [1.0]],
        ]
    )
    ranked = training_population_order(
        observations,
        ["lineage-a", "lineage-a", "lineage-b"],
        policy="population_disagreement",
        seed=7,
    )
    assert int(ranked.indices[0]) == 0
    assert ranked.utilities[0] > ranked.utilities[1]


def test_random_candidate_order_is_reproducible_and_complete() -> None:
    observations = torch.zeros(2, 7, 3)
    first = training_population_order(observations, ["a", "b"], policy="random", seed=11)
    second = training_population_order(observations, ["a", "b"], policy="random", seed=11)
    assert torch.equal(first.indices, second.indices)
    assert sorted(first.indices.tolist()) == list(range(7))
