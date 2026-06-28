from __future__ import annotations

from behavior2weights.schemas import Split
from behavior2weights.zoo.micro import MicroZooConfig
from behavior2weights.zoo.public import PublicManifestConfig, build_public_manifest


def test_micro_config_expands_integer_ranges() -> None:
    config = MicroZooConfig.from_dict(
        {
            "architectures": [
                {
                    "vocab_size": 16,
                    "max_seq_len": 8,
                    "d_model": 8,
                    "n_heads": 2,
                    "n_layers": 1,
                    "d_ff": 12,
                }
            ],
            "model_seeds": {"start": 2, "stop": 8, "step": 2},
            "dataset_seeds": [1],
            "train_steps": 4,
            "checkpoint_steps": [0, 4],
        }
    )
    assert config.model_seeds == (2, 4, 6)


def test_public_manifest_expands_revisions(tmp_path) -> None:
    config = PublicManifestConfig.from_dict(
        {
            "models": [
                {
                    "model_name": "example/model",
                    "revisions": ["step0", "step16"],
                    "external_family": "example",
                    "split": "ood",
                }
            ]
        }
    )
    records = build_public_manifest(config, tmp_path / "targets.jsonl")
    assert len(records) == 2
    assert {record.training_step for record in records} == {0, 16}
    assert all(record.split == Split.OOD for record in records)


def test_text_zoo_config_expands_ranges_and_rejects_unknown_fields() -> None:
    import pytest

    from behavior2weights.zoo.text import TextZooConfig

    config = TextZooConfig.from_dict(
        {
            "architectures": [
                {
                    "vocab_size": 64,
                    "max_seq_len": 16,
                    "d_model": 16,
                    "n_heads": 2,
                    "n_layers": 1,
                    "d_ff": 32,
                }
            ],
            "model_seeds": {"start": 10, "stop": 14, "step": 2},
            "data_order_seeds": [3],
            "train_steps": 4,
            "checkpoint_steps": [0, 4],
        }
    )
    assert config.model_seeds == (10, 12)
    assert config.checkpoint_steps == (0, 4)

    with pytest.raises(ValueError, match="Unknown TextZooConfig fields"):
        TextZooConfig.from_dict(
            {
                "architectures": [
                    {
                        "vocab_size": 64,
                        "max_seq_len": 16,
                        "d_model": 16,
                        "n_heads": 2,
                        "n_layers": 1,
                        "d_ff": 32,
                    }
                ],
                "train_steps": 0,
                "checkpoint_steps": [0],
                "not_a_field": True,
            }
        )


def test_tokenizer_training_config_normalizes_special_tokens() -> None:
    from behavior2weights.data.tokenizer import TokenizerTrainingConfig

    config = TokenizerTrainingConfig.from_dict(
        {
            "dataset_name": "example/corpus",
            "special_tokens": ["<pad>", "<unk>", "<eos>"],
        }
    )
    assert config.special_tokens == ("<pad>", "<unk>", "<eos>")


def test_evaluation_config_rejects_unknown_query_policy() -> None:
    import pytest

    from behavior2weights.evaluation.runner import EvaluationConfig

    with pytest.raises(ValueError, match="query_policies"):
        EvaluationConfig(query_policies=("test_leaking_oracle",))


def test_checked_in_public_families_have_one_tokenizer_per_family() -> None:
    from pathlib import Path

    from behavior2weights.utils import load_yaml
    from behavior2weights.zoo.public import PublicManifestConfig

    path = Path(__file__).resolve().parents[1] / "configs" / "model_zoo" / "public_models.yaml"
    config = PublicManifestConfig.from_dict(load_yaml(path))
    grouped: dict[str, set[str]] = {}
    for model in config.models:
        grouped.setdefault(model.external_family, set()).add(
            model.tokenizer_name or model.model_name
        )
    assert all(len(tokenizers) == 1 for tokenizers in grouped.values())
    assert {model.model_name for model in config.models} >= {
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-2.8b",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    }
