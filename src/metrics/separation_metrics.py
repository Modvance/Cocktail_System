from __future__ import annotations

import torch

from src.losses.sisdr import sisdr_value


def batch_sisdr(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return sisdr_value(estimate, target)


def batch_sisdri(estimate: torch.Tensor, mixture: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return batch_sisdr(estimate, target) - batch_sisdr(mixture, target)


def batch_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    error = target - estimate
    ratio = torch.sum(target * target, dim=-1).clamp_min(eps) / torch.sum(error * error, dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))


def batch_sdri(estimate: torch.Tensor, mixture: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return batch_sdr(estimate, target) - batch_sdr(mixture, target)


def batch_sir(estimate: torch.Tensor, mixture: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    interference = mixture - target
    target_power = torch.sum(target * target, dim=-1).clamp_min(eps)
    interference_residual = torch.sum((estimate - target) * (estimate - target), dim=-1).clamp_min(eps)
    interference_power = torch.sum(interference * interference, dim=-1).clamp_min(eps)
    ratio = target_power / torch.minimum(interference_residual, interference_power)
    return 10.0 * torch.log10(ratio.clamp_min(eps))


def batch_sar(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    artifact = estimate - target
    ratio = torch.sum(target * target, dim=-1).clamp_min(eps) / torch.sum(artifact * artifact, dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))
