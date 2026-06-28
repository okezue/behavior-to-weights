from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, cast


@dataclasses.dataclass(frozen=True, slots=True)
class HFTextDatasetConfig:
    dataset_name: str
    dataset_config: str | None = None
    text_column: str = "text"
    tokenizer_name: str = "gpt2"
    sequence_length: int = 256
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    num_proc: int = 1
    revision: str | None = None
    tokenizer_revision: str | None = None
    max_train_examples: int | None = None
    max_validation_examples: int | None = None
    max_test_examples: int | None = None


PRESETS: dict[str, HFTextDatasetConfig] = {
    "tinystories_clean": HFTextDatasetConfig(
        dataset_name="karpathy/tinystories-gpt4-clean",
        tokenizer_name="gpt2",
        sequence_length=256,
    ),
    "tinystories": HFTextDatasetConfig(
        dataset_name="roneneldan/TinyStories",
        tokenizer_name="gpt2",
        sequence_length=256,
    ),
    "wikitext103": HFTextDatasetConfig(
        dataset_name="Salesforce/wikitext",
        dataset_config="wikitext-103-raw-v1",
        tokenizer_name="gpt2",
        sequence_length=256,
    ),
}


def prepare_hf_causal_lm_dataset(
    config: HFTextDatasetConfig,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Download, tokenize, concatenate, and fixed-block a Hugging Face text dataset."""

    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install behavior2weights[hf] to prepare Hugging Face datasets"
        ) from error

    output_directory = Path(output_directory)
    manifest_path = output_directory / "dataset_manifest.json"
    if manifest_path.exists() and not overwrite:
        load_from_disk(str(output_directory / "arrow"))
        return manifest_path

    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        revision=config.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        revision=config.tokenizer_revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    split_names = {
        "train": config.train_split,
        "validation": config.validation_split,
        "test": config.test_split,
    }
    limits = {
        "train": config.max_train_examples,
        "validation": config.max_validation_examples,
        "test": config.max_test_examples,
    }
    selected: dict[str, Any] = {}
    for canonical, source in split_names.items():
        if source not in dataset:
            continue
        split = dataset[source]
        limit = limits[canonical]
        if limit is not None:
            split = split.select(range(min(limit, len(split))))
        selected[canonical] = split
    dataset = DatasetDict(selected)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        texts = [text if isinstance(text, str) else "" for text in batch[config.text_column]]
        return cast(
            dict[str, Any],
            tokenizer(texts, add_special_tokens=False, return_attention_mask=False),
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        num_proc=config.num_proc,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing text",
    )

    block_size = config.sequence_length + 1

    def group(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated = sum(batch["input_ids"], [])
        usable = len(concatenated) // block_size * block_size
        blocks = [
            concatenated[index : index + block_size] for index in range(0, usable, block_size)
        ]
        return {"input_ids": blocks}

    grouped = tokenized.map(
        group,
        batched=True,
        num_proc=config.num_proc,
        desc="Packing fixed-length causal-LM blocks",
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    grouped.save_to_disk(str(output_directory / "arrow"))
    manifest = {
        "schema_version": 1,
        "config": dataclasses.asdict(config),
        "dataset_fingerprint": {name: split._fingerprint for name, split in grouped.items()},
        "rows": {name: len(split) for name, split in grouped.items()},
        "tokenizer": {
            "name_or_path": tokenizer.name_or_path,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path
