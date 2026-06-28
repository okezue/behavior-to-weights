from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def normalized_rmse(prediction: Tensor, target: Tensor, epsilon: float = 1e-12) -> float:
    numerator = (prediction.float() - target.float()).square().mean().sqrt()
    denominator = target.float().square().mean().sqrt().clamp_min(epsilon)
    return float((numerator / denominator).item())


def cosine_similarity(prediction: Tensor, target: Tensor) -> float:
    return float(
        F.cosine_similarity(prediction.float().reshape(1, -1), target.float().reshape(1, -1)).item()
    )


def state_dict_metrics(
    prediction: Mapping[str, Tensor], target: Mapping[str, Tensor]
) -> dict[str, Any]:
    shared = [name for name in target if name in prediction and target[name].is_floating_point()]
    if not shared:
        raise ValueError("no shared floating-point tensors")
    predicted_vector = torch.cat([prediction[name].reshape(-1).cpu() for name in shared])
    target_vector = torch.cat([target[name].reshape(-1).cpu() for name in shared])
    per_tensor = {
        name: {
            "nrmse": normalized_rmse(prediction[name], target[name]),
            "cosine": cosine_similarity(prediction[name], target[name]),
        }
        for name in shared
    }
    return {
        "nrmse": normalized_rmse(predicted_vector, target_vector),
        "cosine": cosine_similarity(predicted_vector, target_vector),
        "per_tensor": per_tensor,
    }


@torch.no_grad()
def functional_metrics(
    prediction_model: nn.Module,
    target_model: nn.Module,
    input_ids: Tensor,
    *,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    prediction_model = prediction_model.to(device).eval()
    target_model = target_model.to(device).eval()
    kl_sum = 0.0
    reverse_kl_sum = 0.0
    agreement_sum = 0.0
    probability_l1_sum = 0.0
    count = 0
    for start in range(0, len(input_ids), batch_size):
        batch = input_ids[start : start + batch_size].to(device)
        prediction = prediction_model(batch)
        target = target_model(batch)
        if hasattr(prediction, "logits"):
            prediction = prediction.logits
        if hasattr(target, "logits"):
            target = target.logits
        prediction_logits = prediction[:, -1, :].float()
        target_logits = target[:, -1, :].float()
        prediction_log_probs = F.log_softmax(prediction_logits, dim=-1)
        target_log_probs = F.log_softmax(target_logits, dim=-1)
        target_probs = target_log_probs.exp()
        prediction_probs = prediction_log_probs.exp()
        batch_count = len(batch)
        kl_sum += float(
            F.kl_div(prediction_log_probs, target_probs, reduction="batchmean").item() * batch_count
        )
        reverse_kl_sum += float(
            F.kl_div(target_log_probs, prediction_probs, reduction="batchmean").item() * batch_count
        )
        agreement_sum += float(
            (prediction_logits.argmax(-1) == target_logits.argmax(-1)).float().sum().item()
        )
        probability_l1_sum += float(
            (prediction_probs - target_probs).abs().sum(dim=-1).sum().item()
        )
        count += batch_count
    return {
        "forward_kl": kl_sum / count,
        "reverse_kl": reverse_kl_sum / count,
        "top1_agreement": agreement_sum / count,
        "probability_l1": probability_l1_sum / count,
    }


def posterior_calibration(
    target: Tensor,
    mean: Tensor,
    log_variance: Tensor,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
) -> dict[str, float]:
    normal = torch.distributions.Normal(0.0, 1.0)
    standard_deviation = torch.exp(0.5 * log_variance)
    output: dict[str, float] = {}
    for level in levels:
        quantile = normal.icdf(torch.tensor((1 + level) / 2, device=target.device))
        covered = (target - mean).abs() <= quantile * standard_deviation
        output[f"coverage_{level:.2f}"] = float(covered.float().mean().item())
        output[f"coverage_error_{level:.2f}"] = abs(output[f"coverage_{level:.2f}"] - level)
    z = (target - mean) / standard_deviation.clamp_min(1e-12)
    output["standardized_residual_mean"] = float(z.mean().item())
    output["standardized_residual_std"] = float(z.std(unbiased=False).item())
    return output


def localization_metrics(scores: Tensor, correct_indices: Tensor) -> dict[str, float]:
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if correct_indices.numel() == 0:
        raise ValueError("correct_indices cannot be empty")
    order = torch.argsort(scores, descending=True)
    correct = set(correct_indices.long().tolist())
    ranks = [rank + 1 for rank, index in enumerate(order.tolist()) if index in correct]
    best_rank = min(ranks)
    return {
        "reciprocal_rank": 1.0 / best_rank,
        "top1": float(best_rank <= 1),
        "top3": float(best_rank <= 3),
        "top5": float(best_rank <= 5),
        "best_rank": float(best_rank),
    }
