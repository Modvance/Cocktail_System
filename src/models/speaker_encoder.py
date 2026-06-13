from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = hidden_dim
        self.conv = nn.Sequential(*layers)
        self.attention = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.proj = nn.Linear(hidden_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, enrollment_feat: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(enrollment_feat)
        weights = torch.softmax(self.attention(hidden), dim=-1)
        mean = torch.sum(hidden * weights, dim=-1)
        var = torch.sum(((hidden - mean.unsqueeze(-1)) ** 2) * weights, dim=-1)
        std = torch.sqrt(var.clamp_min(1e-8))
        pooled = torch.cat([mean, std], dim=-1)
        embedding = self.proj(pooled)
        embedding = self.norm(embedding)
        return F.normalize(embedding, p=2, dim=-1)
