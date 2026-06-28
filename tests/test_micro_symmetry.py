from __future__ import annotations

import torch

from behavior2weights.models.micro_transformer import (
    MicroTransformer,
    MicroTransformerConfig,
    canonicalize_state_dict,
    permute_attention_heads,
    permute_mlp_neurons,
)


def test_head_and_neuron_permutations_preserve_function() -> None:
    torch.manual_seed(7)
    config = MicroTransformerConfig(
        vocab_size=16, max_seq_len=8, d_model=8, n_heads=2, n_layers=1, d_ff=12
    )
    reference = MicroTransformer(config).eval()
    state = reference.ordered_state_dict()
    permuted = permute_attention_heads(state, config, 0, torch.tensor([1, 0]))
    permuted = permute_mlp_neurons(permuted, 0, torch.arange(config.d_ff - 1, -1, -1))
    candidate = MicroTransformer(config).eval()
    candidate.load_state_dict(permuted)
    inputs = torch.randint(config.vocab_size, (17, config.max_seq_len - 1))
    assert torch.allclose(reference(inputs), candidate(inputs), atol=1e-6, rtol=1e-5)


def test_canonicalization_matches_permuted_orbit() -> None:
    torch.manual_seed(11)
    config = MicroTransformerConfig(
        vocab_size=16, max_seq_len=8, d_model=8, n_heads=2, n_layers=1, d_ff=10
    )
    state = MicroTransformer(config).ordered_state_dict()
    permuted = permute_attention_heads(state, config, 0, torch.tensor([1, 0]))
    permuted = permute_mlp_neurons(permuted, 0, torch.randperm(config.d_ff))
    first = canonicalize_state_dict(state, config)
    second = canonicalize_state_dict(permuted, config)
    assert first.keys() == second.keys()
    for name in first:
        assert torch.equal(first[name], second[name]), name
