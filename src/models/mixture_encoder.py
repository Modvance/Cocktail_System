from __future__ import annotations

import torch
import torch.nn as nn


class MixtureEncoder(nn.Module):
    def __init__(self, freq_bins: int, hidden_dim: int, lstm_hidden_dim: int, lstm_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(freq_bins, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.output_proj = nn.Linear(lstm_hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mixture_log_mag: torch.Tensor) -> torch.Tensor:
        hidden = mixture_log_mag.transpose(1, 2)
        hidden = self.input_proj(hidden)
        hidden = self.norm(hidden)
        hidden, _ = self.lstm(hidden)
        hidden = self.output_proj(hidden)
        return self.dropout(hidden)
