from __future__ import annotations

import numpy as np
import pandas as pd

from behavior2weights.stats.inference import (
    holm_adjust,
    paired_cluster_bootstrap,
    paired_cluster_permutation_test,
)
from behavior2weights.stats.power import analytic_paired_sample_size


def test_clustered_inference_uses_lineage_pairs() -> None:
    rows = []
    for lineage in range(20):
        rows.extend(
            [
                {
                    "lineage_id": f"l{lineage}",
                    "method": "a",
                    "metric": "score",
                    "value": 1.0 + lineage / 100,
                },
                {
                    "lineage_id": f"l{lineage}",
                    "method": "b",
                    "metric": "score",
                    "value": 0.7 + lineage / 100,
                },
            ]
        )
    estimate = paired_cluster_bootstrap(
        pd.DataFrame(rows), method_a="a", method_b="b", metric="score", resamples=1000
    )
    assert np.isclose(estimate.estimate, 0.3)
    assert estimate.lower > 0
    p = paired_cluster_permutation_test([0.3] * 12, permutations=5000, alternative="greater")
    assert p < 0.01


def test_holm_and_power_helpers() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert all(0 <= value <= 1 for value in adjusted)
    assert analytic_paired_sample_size(0.5) > 10
