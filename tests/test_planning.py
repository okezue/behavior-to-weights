from __future__ import annotations

from behavior2weights.models.micro_transformer import MicroTransformer, MicroTransformerConfig
from behavior2weights.zoo.manifest import SplitPolicy
from behavior2weights.zoo.micro import MicroZooConfig
from behavior2weights.zoo.planning import (
    micro_parameter_count,
    plan_micro_zoo,
    plan_zoo_config,
)


def test_closed_form_parameter_count_matches_module() -> None:
    for tied in (False, True):
        for bias in (False, True):
            config = MicroTransformerConfig(
                vocab_size=19,
                max_seq_len=11,
                d_model=12,
                n_heads=3,
                n_layers=2,
                d_ff=31,
                tie_embeddings=tied,
                bias=bias,
            )
            expected = sum(parameter.numel() for parameter in MicroTransformer(config).parameters())
            assert micro_parameter_count(config) == expected


def test_micro_plan_counts_lineages_targets_splits_and_storage() -> None:
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
                },
                {
                    "vocab_size": 16,
                    "max_seq_len": 8,
                    "d_model": 12,
                    "n_heads": 3,
                    "n_layers": 2,
                    "d_ff": 24,
                },
            ],
            "tasks": ["copy", "markov"],
            "model_seeds": [1, 2, 3],
            "dataset_seeds": [4, 5],
            "optimizers": [{"name": "adamw", "learning_rate": 0.001}],
            "train_steps": 4,
            "checkpoint_steps": [0, 2, 4],
            "interventions": [
                {"kind": "attention_head_ablation", "count": 2},
                {"kind": "lora_edit", "count": 1},
            ],
        }
    )
    projection = plan_micro_zoo(config, split_policy=SplitPolicy())
    assert projection.lineages == 24
    assert projection.checkpoints_per_lineage == 3
    assert projection.interventions_per_lineage == 3
    assert projection.targets_per_lineage == 6
    assert projection.targets == 144
    assert sum(projection.lineage_splits.values()) == 24
    assert sum(projection.target_splits.values()) == 144
    assert projection.projected_raw_checkpoint_bytes_fp32 == sum(
        row.total_targets * row.parameters * 4 for row in projection.architectures
    )


def test_auto_planner_recognizes_text_and_public_configs() -> None:
    architecture = {
        "vocab_size": 16,
        "max_seq_len": 8,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 12,
    }
    text = plan_zoo_config(
        {
            "name": "text",
            "dataset_directory": "data/example",
            "zoo": {
                "architectures": [architecture],
                "model_seeds": [1, 2],
                "data_order_seeds": [3],
                "train_steps": 4,
                "checkpoint_steps": [0, 4],
            },
        }
    )
    assert text.kind == "text"
    assert text.lineages == 2
    assert text.targets == 4

    public = plan_zoo_config(
        {
            "models": [
                {
                    "model_name": "example/model",
                    "external_family": "example",
                    "revisions": ["a", "b"],
                    "parameter_count": 10,
                }
            ]
        }
    )
    assert public.kind == "public"
    assert public.targets == 2
    assert public.projected_raw_checkpoint_bytes_fp32 == 80
