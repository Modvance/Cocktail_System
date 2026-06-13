from __future__ import annotations

import torch
import torch.nn as nn

from src.models.feature_extractor import AudioFeatureExtractor


class WaveformReconstructor(nn.Module):
    def __init__(self, feature_extractor: AudioFeatureExtractor) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, estimated_mag: torch.Tensor, mixture_phase: torch.Tensor, length: int) -> torch.Tensor:
        complex_spec = torch.polar(estimated_mag, mixture_phase)
        return self.feature_extractor.istft(complex_spec, length=length)
