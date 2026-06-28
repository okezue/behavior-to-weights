from __future__ import annotations

import dataclasses
import math
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F


@dataclasses.dataclass(frozen=True, slots=True)
class MicroTransformerConfig:
    vocab_size: int = 32
    max_seq_len: int = 16
    d_model: int = 16
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 32
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    tie_embeddings: bool = False
    bias: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        if self.vocab_size < 4:
            raise ValueError("vocab_size must be at least 4")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MicroTransformerConfig:
        fields = {field.name for field in dataclasses.fields(cls)}
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"Unknown MicroTransformerConfig fields: {sorted(unknown)}")
        return cls(**dict(data))


class CausalSelfAttention(nn.Module):
    causal_mask: Tensor

    def __init__(self, config: MicroTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", mask, persistent=False)
        self.last_attention: Tensor | None = None
        self.last_head_output: Tensor | None = None

    def forward(self, x: Tensor, *, capture: bool = False) -> Tensor:
        batch, seq_len, width = x.shape
        heads = self.config.n_heads
        head_dim = self.config.head_dim

        def shape_projection(projection: nn.Linear) -> Tensor:
            return cast(Tensor, projection(x).view(batch, seq_len, heads, head_dim).transpose(1, 2))

        q = shape_projection(self.q_proj)
        k = shape_projection(self.k_proj)
        v = shape_projection(self.v_proj)
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        scores = scores.masked_fill(self.causal_mask[:seq_len, :seq_len], float("-inf"))
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        head_output = attention @ v
        merged = head_output.transpose(1, 2).contiguous().view(batch, seq_len, width)
        if capture:
            self.last_attention = attention.detach()
            self.last_head_output = head_output.detach()
        else:
            self.last_attention = None
            self.last_head_output = None
        return cast(Tensor, self.o_proj(merged))


class FeedForward(nn.Module):
    def __init__(self, config: MicroTransformerConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.dropout(self.fc2(F.gelu(self.fc1(x), approximate="tanh"))))


class TransformerBlock(nn.Module):
    def __init__(self, config: MicroTransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x: Tensor, *, capture: bool = False) -> Tensor:
        x = x + self.attn(self.ln1(x), capture=capture)
        x = x + self.mlp(self.ln2(x))
        return x


class MicroTransformer(nn.Module):
    """A deliberately transparent decoder-only Transformer for identification experiments."""

    def __init__(self, config: MicroTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        *,
        capture: bool = False,
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}"
            )
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}"
            )
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        hidden = self.dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden, capture=capture)
        hidden = self.final_ln(hidden)
        logits = self.lm_head(hidden)
        return (logits, hidden) if return_hidden else logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        output = input_ids
        for _ in range(max_new_tokens):
            context = output[:, -self.config.max_seq_len :]
            logits = self(context)[:, -1, :] / temperature
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            output = torch.cat([output, next_token], dim=1)
        return output

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )

    def ordered_state_dict(self) -> OrderedDict[str, Tensor]:
        return OrderedDict((name, tensor.detach()) for name, tensor in self.state_dict().items())

    def save(self, path: str | Path, *, metadata: Mapping[str, str] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config_metadata = {
            f"config.{key}": str(value) for key, value in dataclasses.asdict(self.config).items()
        }
        if metadata:
            config_metadata.update({str(key): str(value) for key, value in metadata.items()})
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in self.state_dict().items()
            },
            str(path),
            metadata=config_metadata,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: MicroTransformerConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> MicroTransformer:
        model = cls(config)
        state = load_file(str(path), device=str(device))
        model.load_state_dict(state, strict=True)
        return model.to(device)


def flatten_state_dict(
    state_dict: Mapping[str, Tensor],
) -> tuple[Tensor, list[tuple[str, tuple[int, ...]]]]:
    tensors: list[Tensor] = []
    spec: list[tuple[str, tuple[int, ...]]] = []
    for name, tensor in state_dict.items():
        if not tensor.is_floating_point():
            continue
        tensors.append(tensor.detach().reshape(-1))
        spec.append((name, tuple(tensor.shape)))
    if not tensors:
        return torch.empty(0), spec
    return torch.cat(tensors), spec


def unflatten_state_dict(
    vector: Tensor,
    spec: list[tuple[str, tuple[int, ...]]],
    *,
    template: Mapping[str, Tensor] | None = None,
) -> OrderedDict[str, Tensor]:
    result: OrderedDict[str, Tensor] = OrderedDict()
    offset = 0
    for name, shape in spec:
        count = math.prod(shape)
        if offset + count > vector.numel():
            raise ValueError("Vector is shorter than the state specification")
        value = vector[offset : offset + count].reshape(shape)
        if template is not None and name in template:
            value = value.to(dtype=template[name].dtype, device=template[name].device)
        result[name] = value
        offset += count
    if offset != vector.numel():
        raise ValueError(f"Vector has {vector.numel() - offset} trailing values")
    return result


def permute_attention_heads(
    state_dict: Mapping[str, Tensor],
    config: MicroTransformerConfig,
    layer: int,
    permutation: Tensor,
) -> OrderedDict[str, Tensor]:
    if permutation.tolist() == list(range(config.n_heads)):
        return OrderedDict((name, value.clone()) for name, value in state_dict.items())
    if sorted(permutation.tolist()) != list(range(config.n_heads)):
        raise ValueError("permutation must contain each head index exactly once")
    result = OrderedDict((name, value.clone()) for name, value in state_dict.items())
    head_dim = config.head_dim
    index = torch.cat(
        [torch.arange(head * head_dim, (head + 1) * head_dim) for head in permutation.tolist()]
    )
    prefix = f"blocks.{layer}.attn"
    for projection in ("q_proj", "k_proj", "v_proj"):
        weight_name = f"{prefix}.{projection}.weight"
        result[weight_name] = result[weight_name][index, :]
        bias_name = f"{prefix}.{projection}.bias"
        if bias_name in result:
            result[bias_name] = result[bias_name][index]
    output_name = f"{prefix}.o_proj.weight"
    result[output_name] = result[output_name][:, index]
    return result


def permute_mlp_neurons(
    state_dict: Mapping[str, Tensor],
    layer: int,
    permutation: Tensor,
) -> OrderedDict[str, Tensor]:
    result = OrderedDict((name, value.clone()) for name, value in state_dict.items())
    prefix = f"blocks.{layer}.mlp"
    fc1_weight = f"{prefix}.fc1.weight"
    fc1_bias = f"{prefix}.fc1.bias"
    fc2_weight = f"{prefix}.fc2.weight"
    width = result[fc1_weight].shape[0]
    if sorted(permutation.tolist()) != list(range(width)):
        raise ValueError("permutation must contain each MLP neuron exactly once")
    result[fc1_weight] = result[fc1_weight][permutation, :]
    if fc1_bias in result:
        result[fc1_bias] = result[fc1_bias][permutation]
    result[fc2_weight] = result[fc2_weight][:, permutation]
    return result


def canonicalize_state_dict(
    state_dict: Mapping[str, Tensor], config: MicroTransformerConfig
) -> OrderedDict[str, Tensor]:
    """Choose a deterministic representative of common head/neuron permutation orbits.

    The signature intentionally mixes norms and deterministic projections. It is a useful baseline,
    not a proof of a globally continuous canonical gauge; primary analyses should still report
    orbit-aligned and functional metrics.
    """

    result = OrderedDict((name, value.clone()) for name, value in state_dict.items())
    for layer in range(config.n_layers):
        prefix = f"blocks.{layer}.attn"
        signatures: list[tuple[float, ...]] = []
        for head in range(config.n_heads):
            sl = slice(head * config.head_dim, (head + 1) * config.head_dim)
            pieces = [
                result[f"{prefix}.q_proj.weight"][sl],
                result[f"{prefix}.k_proj.weight"][sl],
                result[f"{prefix}.v_proj.weight"][sl],
                result[f"{prefix}.o_proj.weight"][:, sl],
            ]
            flattened = torch.cat([piece.reshape(-1).float() for piece in pieces])
            projection = torch.linspace(-1.0, 1.0, flattened.numel(), device=flattened.device)
            signatures.append(
                (
                    float(flattened.norm().item()),
                    float((flattened * projection).sum().item()),
                    float(flattened.mean().item()),
                )
            )
        permutation = torch.tensor(sorted(range(config.n_heads), key=signatures.__getitem__))
        result = permute_attention_heads(result, config, layer, permutation)

        mlp_prefix = f"blocks.{layer}.mlp"
        fc1 = result[f"{mlp_prefix}.fc1.weight"]
        fc2 = result[f"{mlp_prefix}.fc2.weight"]
        signatures_mlp: list[tuple[float, ...]] = []
        for neuron in range(config.d_ff):
            flattened = torch.cat([fc1[neuron].reshape(-1).float(), fc2[:, neuron].float()])
            projection = torch.linspace(-0.5, 0.5, flattened.numel(), device=flattened.device)
            signatures_mlp.append(
                (
                    float(flattened.norm().item()),
                    float((flattened * projection).sum().item()),
                    float(flattened.mean().item()),
                )
            )
        neuron_permutation = torch.tensor(
            sorted(range(config.d_ff), key=signatures_mlp.__getitem__)
        )
        result = permute_mlp_neurons(result, layer, neuron_permutation)
    return result
