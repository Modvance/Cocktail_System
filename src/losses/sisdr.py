from __future__ import annotations

import torch
import torch.nn as nn


class SISDRLoss(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        estimate = estimate.float()
        target = target.float()
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        target_energy = torch.sum(target * target, dim=-1, keepdim=True).clamp_min(self.eps)
        projection = torch.sum(estimate * target, dim=-1, keepdim=True) * target / target_energy
        noise = estimate - projection
        ratio = torch.sum(projection * projection, dim=-1).clamp_min(self.eps) / torch.sum(noise * noise, dim=-1).clamp_min(self.eps)
        sisdr = 10.0 * torch.log10(ratio.clamp_min(self.eps))
        return -sisdr.mean()


def sisdr_value(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    estimate = estimate.float()
    target = target.float()
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    target_energy = torch.sum(target * target, dim=-1, keepdim=True).clamp_min(eps)
    projection = torch.sum(estimate * target, dim=-1, keepdim=True) * target / target_energy
    noise = estimate - projection
    ratio = torch.sum(projection * projection, dim=-1).clamp_min(eps) / torch.sum(noise * noise, dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))
