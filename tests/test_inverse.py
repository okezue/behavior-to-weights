from __future__ import annotations

import torch

from behavior2weights.models.inverse import BehaviorToWeights, InverseModelConfig, gaussian_nll
from behavior2weights.models.micro_transformer import MicroTransformer, MicroTransformerConfig
from behavior2weights.models.weight_space import AddressSpace


def test_inverse_shapes_and_set_invariance() -> None:
    config = InverseModelConfig(
        vocab_size=16,
        max_seq_len=8,
        observation_dim=16,
        trace_width=16,
        trace_heads=2,
        query_layers=1,
        set_layers=1,
        latent_dim=12,
        address_width=12,
        decoder_width=24,
        decoder_layers=2,
        max_tensors=64,
        max_layers=8,
    )
    model = BehaviorToWeights(config).eval()
    inputs = torch.randint(16, (2, 7, 6))
    observations = torch.randn(2, 7, 16)
    channels = torch.zeros(2, 7, dtype=torch.long)
    target = MicroTransformer(
        MicroTransformerConfig(
            vocab_size=16, max_seq_len=8, d_model=8, n_heads=2, n_layers=1, d_ff=12
        )
    )
    space = AddressSpace.from_state_dict(target.ordered_state_dict())
    descriptors = space.descriptors(torch.arange(23))
    with torch.no_grad():
        output = model(inputs, observations, channels, descriptors)
        permutation = torch.tensor([6, 0, 4, 1, 5, 2, 3])
        permuted = model(
            inputs[:, permutation],
            observations[:, permutation],
            channels[:, permutation],
            descriptors,
        )
    assert output["weight_mean"].shape == (2, 23)
    assert output["weight_log_variance"].shape == (2, 23)
    assert torch.allclose(output["latent"], permuted["latent"], atol=2e-5, rtol=2e-5)
    assert gaussian_nll(
        torch.zeros(2, 23), output["weight_mean"], output["weight_log_variance"]
    ).isfinite()
