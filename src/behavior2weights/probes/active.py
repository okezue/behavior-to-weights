from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn


@dataclasses.dataclass(frozen=True, slots=True)
class ActiveSelection:
    indices: Tensor
    utilities: Tensor
    policy: str


def random_select(candidate_count: int, budget: int, *, seed: int = 0) -> ActiveSelection:
    if not 0 < budget <= candidate_count:
        raise ValueError("budget must be in (0, candidate_count]")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(candidate_count, generator=generator)[:budget]
    return ActiveSelection(indices=indices, utilities=torch.zeros(budget), policy="random")


def ensemble_disagreement(
    candidate_observations: Tensor,
    *,
    budget: int,
    selected: Sequence[int] = (),
) -> ActiveSelection:
    """Select prompts with maximal posterior-predictive disagreement.

    candidate_observations has shape [posterior_samples, candidates, features]. In the exact
    experiment this is produced by posterior-sampled reconstructed targets; for oracle analysis it
    can be produced by the true candidate checkpoint set.
    """

    if candidate_observations.ndim != 3:
        raise ValueError("candidate_observations must have [samples, candidates, features]")
    utilities = candidate_observations.float().var(dim=0, unbiased=False).mean(dim=-1)
    if selected:
        utilities = utilities.clone()
        utilities[torch.tensor(list(selected), dtype=torch.long)] = -torch.inf
    available = torch.isfinite(utilities).sum().item()
    if budget > available:
        raise ValueError("budget exceeds unselected candidates")
    values, indices = torch.topk(utilities, budget)
    return ActiveSelection(indices=indices, utilities=values, policy="ensemble_disagreement")


def expected_parameter_information_gain(
    predicted_means: Tensor,
    predicted_log_variances: Tensor,
    *,
    budget: int,
) -> ActiveSelection:
    """Diagonal-Gaussian approximation to expected posterior entropy reduction.

    Inputs are per-candidate posterior predictions with shape [candidates, samples, coordinates].
    Utility combines between-posterior disagreement and within-posterior uncertainty.
    """

    if predicted_means.shape != predicted_log_variances.shape or predicted_means.ndim != 3:
        raise ValueError("means/log-variances must share [candidates, samples, coordinates] shape")
    epistemic = predicted_means.var(dim=1, unbiased=False)
    aleatoric = predicted_log_variances.exp().mean(dim=1)
    utilities = torch.log1p(epistemic / aleatoric.clamp_min(1e-8)).mean(dim=-1)
    values, indices = torch.topk(utilities, budget)
    return ActiveSelection(indices=indices, utilities=values, policy="expected_information_gain")


class LearnedQueryScorer(nn.Module):
    def __init__(self, query_width: int, posterior_width: int, hidden_width: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(query_width + posterior_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, query_embeddings: Tensor, posterior_state: Tensor) -> Tensor:
        if query_embeddings.ndim != 3:
            raise ValueError("query_embeddings must have [batch, candidates, width]")
        expanded = posterior_state[:, None, :].expand(-1, query_embeddings.shape[1], -1)
        return cast(
            Tensor, self.network(torch.cat([query_embeddings, expanded], dim=-1)).squeeze(-1)
        )


def training_population_order(
    observations: Tensor,
    lineage_ids: Sequence[str],
    *,
    policy: str = "population_disagreement",
    seed: int = 0,
) -> ActiveSelection:
    """Rank a fixed candidate bank using training lineages only.

    Args:
        observations: Tensor with shape ``[training_targets, candidates, features]``.
        lineage_ids: One lineage identifier per target. Targets from the same training trajectory
            are averaged before utilities are computed, so checkpoints and edits do not inflate the
            effective sample size.
        policy: ``random`` or ``population_disagreement``.
        seed: Tie-breaking/random-policy seed.

    The returned order is fixed before any validation/test response is inspected, which makes it a
    leakage-safe active-query baseline for locked evaluations. Prefixes of one order define every
    query budget, ensuring nested budget curves.
    """

    if observations.ndim != 3:
        raise ValueError("observations must have [targets, candidates, features]")
    if observations.shape[0] != len(lineage_ids):
        raise ValueError("lineage_ids must have one entry per target")
    if observations.shape[0] == 0 or observations.shape[1] == 0:
        raise ValueError("observations cannot be empty")
    candidate_count = observations.shape[1]
    if policy == "random":
        return random_select(candidate_count, candidate_count, seed=seed)
    if policy != "population_disagreement":
        raise ValueError(
            "policy must be 'random' or 'population_disagreement' for a frozen trace ranking"
        )

    grouped: dict[str, list[int]] = {}
    for index, lineage_id in enumerate(lineage_ids):
        grouped.setdefault(str(lineage_id), []).append(index)
    lineage_means = torch.stack(
        [observations[indices].float().mean(dim=0) for indices in grouped.values()]
    )
    utilities = lineage_means.var(dim=0, unbiased=False).mean(dim=-1)

    # Stable random tie-breaking avoids candidate-index artifacts when utilities are identical.
    generator = torch.Generator().manual_seed(seed)
    ties = torch.rand(candidate_count, generator=generator)
    order = sorted(
        range(candidate_count),
        key=lambda index: (-float(utilities[index]), float(ties[index])),
    )
    indices = torch.tensor(order, dtype=torch.long)
    return ActiveSelection(
        indices=indices,
        utilities=utilities[indices],
        policy="population_disagreement",
    )
