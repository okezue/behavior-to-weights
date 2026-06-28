from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Protocol

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from behavior2weights.models.inverse import (
    BehaviorToWeights,
    InverseModelConfig,
    gaussian_nll,
    property_loss,
)
from behavior2weights.models.weight_space import WeightStandardizer
from behavior2weights.schemas import Split
from behavior2weights.train.corpus import InverseTrainingCorpus
from behavior2weights.utils import seed_everything


class TrackerLike(Protocol):
    def set_params(self, params: dict[str, Any]) -> None: ...
    def track(
        self, value: float, *, name: str, step: int, context: dict[str, Any] | None = None
    ) -> None: ...
    def close(self, status: str = "completed") -> None: ...


@dataclasses.dataclass(frozen=True, slots=True)
class InverseTrainingConfig:
    steps: int = 2_000
    batch_size: int = 16
    query_budgets: tuple[int, ...] = (16, 32, 64)
    coordinate_count: int = 2_048
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    variance_regularization: float = 1e-4
    property_loss_weight: float = 0.1
    validation_every: int = 100
    validation_batches: int = 8
    early_stopping_patience: int = 10
    seed: int = 20260621
    device: str = "cpu"
    mixed_precision: bool = False

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0 or self.coordinate_count <= 0:
            raise ValueError("steps, batch_size, and coordinate_count must be positive")
        if not self.query_budgets or min(self.query_budgets) <= 0:
            raise ValueError("query_budgets must contain positive values")


@dataclasses.dataclass(frozen=True, slots=True)
class TrainResult:
    best_checkpoint: Path
    best_validation_nll: float
    steps_completed: int
    history: tuple[dict[str, float], ...]


def _learning_rate_scale(step: int, config: InverseTrainingConfig) -> float:
    if step < config.warmup_steps:
        return (step + 1) / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


@torch.no_grad()
def evaluate_inverse_nll(
    model: BehaviorToWeights,
    corpus: InverseTrainingCorpus,
    indices: list[int],
    standardizer: WeightStandardizer,
    config: InverseTrainingConfig,
    *,
    seed: int = 0,
) -> float:
    if not indices:
        return float("nan")
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    device = next(model.parameters()).device
    budget = max(config.query_budgets)
    for _ in range(config.validation_batches):
        batch = corpus.sample_batch(
            indices,
            batch_size=min(config.batch_size, max(len(indices), 1)),
            query_budget=budget,
            coordinate_count=config.coordinate_count,
            standardizer=standardizer,
            generator=generator,
        )
        batch = _to_device(batch, device)
        output = model(
            batch["input_ids"],
            batch["observations"],
            batch["channel_ids"],
            batch["descriptors"],
            query_mask=batch["query_mask"],
        )
        loss = gaussian_nll(
            batch["target_weights"],
            output["weight_mean"],
            output["weight_log_variance"],
        )
        losses.append(float(loss.item()))
    return float(sum(losses) / len(losses))


def save_inverse_checkpoint(
    model: BehaviorToWeights,
    standardizer: WeightStandardizer,
    corpus: InverseTrainingCorpus,
    directory: str | Path,
    *,
    metadata: dict[str, Any],
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights_path = directory / "inverse_model.safetensors"
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()},
        str(weights_path),
    )
    (directory / "model_config.json").write_text(
        json.dumps(dataclasses.asdict(model.config), indent=2, sort_keys=True) + "\n"
    )
    standardizer.save(directory / "weight_standardizer.json")
    corpus.address_space.to_json(directory / "address_space.json")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return weights_path


def load_inverse_checkpoint(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[BehaviorToWeights, WeightStandardizer]:
    directory = Path(directory)
    config = InverseModelConfig.from_dict(json.loads((directory / "model_config.json").read_text()))
    model = BehaviorToWeights(config)
    model.load_state_dict(
        load_file(str(directory / "inverse_model.safetensors"), device=str(device))
    )
    return model.to(device), WeightStandardizer.load(directory / "weight_standardizer.json")


def train_inverse_model(
    model: BehaviorToWeights,
    corpus: InverseTrainingCorpus,
    config: InverseTrainingConfig,
    output_directory: str | Path,
    *,
    tracker: TrackerLike | None = None,
) -> TrainResult:
    seed_everything(config.seed)
    device = torch.device(config.device)
    model = model.to(device)
    train_indices = corpus.indices_for_split(Split.TRAIN)
    validation_indices = corpus.indices_for_split(Split.VALIDATION)
    if not train_indices:
        raise ValueError("corpus contains no training targets")
    if not validation_indices:
        validation_indices = corpus.indices_for_split(Split.TEST) or train_indices[:1]
    standardizer = corpus.fit_standardizer(train_indices)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed + 1)
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
    output_directory = Path(output_directory)
    history: list[dict[str, float]] = []
    best_validation = float("inf")
    best_path = output_directory / "best"
    stale_evaluations = 0

    if tracker:
        tracker.set_params(
            {
                "training": dataclasses.asdict(config),
                "model": dataclasses.asdict(model.config),
                "architecture_id": corpus.records[0].architecture_id,
                "train_targets": len(train_indices),
                "validation_targets": len(validation_indices),
            }
        )

    try:
        for step in range(config.steps):
            model.train()
            budget = config.query_budgets[
                min(
                    len(config.query_budgets) - 1,
                    step * len(config.query_budgets) // config.steps,
                )
            ]
            batch = corpus.sample_batch(
                train_indices,
                batch_size=config.batch_size,
                query_budget=budget,
                coordinate_count=config.coordinate_count,
                standardizer=standardizer,
                generator=generator,
            )
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=config.mixed_precision and device.type in {"cuda", "cpu"},
            ):
                output = model(
                    batch["input_ids"],
                    batch["observations"],
                    batch["channel_ids"],
                    batch["descriptors"],
                    query_mask=batch["query_mask"],
                )
                weight_nll = gaussian_nll(
                    batch["target_weights"],
                    output["weight_mean"],
                    output["weight_log_variance"],
                )
                variance_penalty = output["weight_log_variance"].square().mean()
                auxiliary = (
                    property_loss(output["properties"], {})
                    if output["properties"]
                    else torch.zeros((), device=weight_nll.device, dtype=weight_nll.dtype)
                )
                loss = (
                    weight_nll
                    + config.variance_regularization * variance_penalty
                    + config.property_loss_weight * auxiliary
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            lr_scale = _learning_rate_scale(step, config)
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * lr_scale

            row = {
                "step": float(step),
                "loss": float(loss.detach().item()),
                "weight_nll": float(weight_nll.detach().item()),
                "query_budget": float(budget),
                "learning_rate": float(config.learning_rate * lr_scale),
            }
            history.append(row)
            if tracker:
                for name in ("loss", "weight_nll", "learning_rate"):
                    tracker.track(row[name], name=name, step=step, context={"subset": "train"})

            should_validate = (step + 1) % config.validation_every == 0 or step + 1 == config.steps
            if should_validate:
                validation_nll = evaluate_inverse_nll(
                    model,
                    corpus,
                    validation_indices,
                    standardizer,
                    config,
                    seed=config.seed + 10_000 + step,
                )
                history[-1]["validation_nll"] = validation_nll
                if tracker:
                    tracker.track(
                        validation_nll,
                        name="weight_nll",
                        step=step,
                        context={"subset": "validation"},
                    )
                if validation_nll < best_validation:
                    best_validation = validation_nll
                    stale_evaluations = 0
                    save_inverse_checkpoint(
                        model,
                        standardizer,
                        corpus,
                        best_path,
                        metadata={
                            "best_validation_nll": best_validation,
                            "step": step,
                            "architecture_id": corpus.records[0].architecture_id,
                        },
                    )
                else:
                    stale_evaluations += 1
                    if stale_evaluations >= config.early_stopping_patience:
                        break
        if not best_path.exists():
            save_inverse_checkpoint(
                model,
                standardizer,
                corpus,
                best_path,
                metadata={"best_validation_nll": best_validation, "step": len(history) - 1},
            )
        if tracker:
            tracker.close("completed")
    except Exception:
        if tracker:
            tracker.close("failed")
        raise

    return TrainResult(
        best_checkpoint=best_path,
        best_validation_nll=best_validation,
        steps_completed=len(history),
        history=tuple(history),
    )
