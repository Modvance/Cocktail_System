from __future__ import annotations

import torch
import torch.nn as nn


class MaskEstimator(nn.Module):
    def __init__(self, hidden_dim: int, mask_hidden_dim: int, mask_layers: int, freq_bins: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=mask_hidden_dim,
            num_layers=mask_layers,
            dropout=dropout if mask_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.proj = nn.Linear(mask_hidden_dim * 2, freq_bins)
        self.activation = nn.Sigmoid()

    def forward(self, conditioned_feat: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(conditioned_feat)
        mask = self.proj(hidden)
        return self.activation(mask).transpose(1, 2)
