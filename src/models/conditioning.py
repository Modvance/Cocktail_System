from __future__ import annotations

import math

import torch
import torch.nn as nn


class TargetConditioningModule(nn.Module):
    def __init__(self, hidden_dim: int, speaker_embed_dim: int, use_enrollment: bool, use_film: bool, use_attention: bool) -> None:
        super().__init__()
        self.use_enrollment = use_enrollment
        self.use_film = use_film
        self.use_attention = use_attention
        self.gamma = nn.Linear(speaker_embed_dim, hidden_dim)
        self.beta = nn.Linear(speaker_embed_dim, hidden_dim)
        self.query = nn.Linear(speaker_embed_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.concat_proj = nn.Linear(hidden_dim + speaker_embed_dim, hidden_dim)

    def forward(self, mixture_feat: torch.Tensor, speaker_embedding: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.use_enrollment or speaker_embedding is None:
            return mixture_feat, None

        hidden = mixture_feat
        if self.use_film:
            gamma = self.gamma(speaker_embedding).unsqueeze(1)
            beta = self.beta(speaker_embedding).unsqueeze(1)
            hidden = gamma * hidden + beta

        if not self.use_attention:
            expanded = speaker_embedding.unsqueeze(1).expand(-1, hidden.size(1), -1)
            return self.concat_proj(torch.cat([hidden, expanded], dim=-1)), None

        query = self.query(speaker_embedding).unsqueeze(1)
        key = self.key(hidden)
        value = self.value(hidden)
        scores = torch.sum(query * key, dim=-1) / math.sqrt(hidden.size(-1))
        weights = torch.softmax(scores, dim=-1)
        context = torch.sum(weights.unsqueeze(-1) * value, dim=1)
        context_expand = context.unsqueeze(1).expand(-1, hidden.size(1), -1)
        fused = self.fusion(torch.cat([hidden, context_expand], dim=-1))
        return hidden + fused, weights
