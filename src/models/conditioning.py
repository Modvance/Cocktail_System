from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            gamma = torch.tanh(self.gamma(speaker_embedding)).unsqueeze(1)
            beta = self.beta(speaker_embedding).unsqueeze(1)
            hidden = hidden * (1.0 + gamma) + beta

        expanded = speaker_embedding.unsqueeze(1).expand(-1, hidden.size(1), -1)
        hidden = hidden + self.concat_proj(torch.cat([hidden, expanded], dim=-1))

        if not self.use_attention:
            return hidden, None

        query = F.normalize(self.query(speaker_embedding), dim=-1).unsqueeze(1)
        key = F.normalize(self.key(hidden), dim=-1)
        value = self.value(hidden)
        scores = torch.sum(query * key, dim=-1) * math.sqrt(hidden.size(-1))
        weights = torch.sigmoid(scores)
        gated_hidden = hidden * (1.0 + weights.unsqueeze(-1))
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        context = torch.sum(value * weights.unsqueeze(-1), dim=1) / denom
        context_expand = context.unsqueeze(1).expand(-1, hidden.size(1), -1)
        fused = self.fusion(torch.cat([gated_hidden, context_expand], dim=-1))
        return hidden + gated_hidden + fused, weights
