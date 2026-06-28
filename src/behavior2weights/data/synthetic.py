from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Literal

import torch
from torch import Tensor

SyntheticTask = Literal["markov", "copy", "induction", "modular_addition", "mixture"]


@dataclasses.dataclass(frozen=True, slots=True)
class SyntheticDatasetConfig:
    task: SyntheticTask = "mixture"
    vocab_size: int = 32
    seq_len: int = 16
    train_examples: int = 2_048
    validation_examples: int = 256
    test_examples: int = 256
    seed: int = 0
    noise: float = 0.05

    def __post_init__(self) -> None:
        if self.vocab_size < 8:
            raise ValueError("vocab_size must be at least 8")
        if self.seq_len < 6:
            raise ValueError("seq_len must be at least 6")
        if min(self.train_examples, self.validation_examples, self.test_examples) <= 0:
            raise ValueError("all split sizes must be positive")


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _markov_sequences(config: SyntheticDatasetConfig, count: int, seed: int) -> Tensor:
    generator = _generator(seed)
    transition_logits = torch.randn(config.vocab_size, config.vocab_size, generator=generator)
    transition_logits += 2.5 * torch.eye(config.vocab_size)
    probabilities = transition_logits.softmax(dim=-1)
    result = torch.empty(count, config.seq_len, dtype=torch.long)
    result[:, 0] = torch.randint(config.vocab_size, (count,), generator=generator)
    for position in range(1, config.seq_len):
        row_probabilities = probabilities[result[:, position - 1]]
        result[:, position] = torch.multinomial(row_probabilities, 1, generator=generator).squeeze(
            1
        )
    return result


def _copy_sequences(config: SyntheticDatasetConfig, count: int, seed: int) -> Tensor:
    generator = _generator(seed)
    delimiter = config.vocab_size - 1
    usable = config.vocab_size - 1
    prefix_len = max(2, (config.seq_len - 1) // 2)
    prefix = torch.randint(usable, (count, prefix_len), generator=generator)
    rows: list[Tensor] = []
    for row in prefix:
        sequence = torch.cat([row, torch.tensor([delimiter]), row])[: config.seq_len]
        if sequence.numel() < config.seq_len:
            padding = torch.randint(
                usable, (config.seq_len - sequence.numel(),), generator=generator
            )
            sequence = torch.cat([sequence, padding])
        rows.append(sequence)
    return torch.stack(rows)


def _induction_sequences(config: SyntheticDatasetConfig, count: int, seed: int) -> Tensor:
    generator = _generator(seed)
    result = torch.randint(config.vocab_size, (count, config.seq_len), generator=generator)
    for row in range(count):
        first_position = int(torch.randint(0, config.seq_len - 4, (1,), generator=generator).item())
        second_position = int(
            torch.randint(first_position + 2, config.seq_len - 1, (1,), generator=generator).item()
        )
        key = int(torch.randint(config.vocab_size, (1,), generator=generator).item())
        value = int(torch.randint(config.vocab_size, (1,), generator=generator).item())
        result[row, first_position] = key
        result[row, first_position + 1] = value
        result[row, second_position] = key
        result[row, second_position + 1] = value
    return result


def _modular_addition_sequences(config: SyntheticDatasetConfig, count: int, seed: int) -> Tensor:
    generator = _generator(seed)
    modulus = max(4, config.vocab_size - 4)
    plus_token = config.vocab_size - 4
    equals_token = config.vocab_size - 3
    separator_token = config.vocab_size - 2
    rows: list[Tensor] = []
    for _ in range(count):
        tokens: list[int] = []
        while len(tokens) < config.seq_len:
            a = int(torch.randint(modulus, (1,), generator=generator).item())
            b = int(torch.randint(modulus, (1,), generator=generator).item())
            tokens.extend([a, plus_token, b, equals_token, (a + b) % modulus, separator_token])
        rows.append(torch.tensor(tokens[: config.seq_len], dtype=torch.long))
    return torch.stack(rows)


def generate_sequences(
    config: SyntheticDatasetConfig,
    count: int,
    *,
    seed: int | None = None,
) -> Tensor:
    seed = config.seed if seed is None else seed
    if config.task == "markov":
        return _markov_sequences(config, count, seed)
    if config.task == "copy":
        return _copy_sequences(config, count, seed)
    if config.task == "induction":
        return _induction_sequences(config, count, seed)
    if config.task == "modular_addition":
        return _modular_addition_sequences(config, count, seed)
    if config.task == "mixture":
        tasks: tuple[SyntheticTask, ...] = ("markov", "copy", "induction", "modular_addition")
        sizes = [count // len(tasks)] * len(tasks)
        for index in range(count % len(tasks)):
            sizes[index] += 1
        chunks: list[Tensor] = []
        for index, (task, size) in enumerate(zip(tasks, sizes, strict=True)):
            sub_config = dataclasses.replace(config, task=task)
            chunks.append(generate_sequences(sub_config, size, seed=seed + 10_003 * (index + 1)))
        merged = torch.cat(chunks)
        return merged[torch.randperm(count, generator=_generator(seed + 77))]
    raise ValueError(f"Unknown synthetic task: {config.task}")


def build_splits(config: SyntheticDatasetConfig) -> dict[str, Tensor]:
    return {
        "train": generate_sequences(config, config.train_examples, seed=config.seed),
        "validation": generate_sequences(
            config, config.validation_examples, seed=config.seed + 1_000_003
        ),
        "test": generate_sequences(config, config.test_examples, seed=config.seed + 2_000_003),
    }


class SequenceBatcher:
    def __init__(
        self,
        sequences: Tensor,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if sequences.ndim != 2:
            raise ValueError("sequences must have shape [examples, sequence]")
        self.sequences = sequences
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def batches(self, epoch: int = 0) -> Iterable[tuple[Tensor, Tensor]]:
        if self.shuffle:
            indices = torch.randperm(len(self.sequences), generator=_generator(self.seed + epoch))
        else:
            indices = torch.arange(len(self.sequences))
        for start in range(0, len(indices), self.batch_size):
            batch = self.sequences[indices[start : start + self.batch_size]]
            yield batch[:, :-1], batch[:, 1:]
