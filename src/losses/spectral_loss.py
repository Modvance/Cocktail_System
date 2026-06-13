from __future__ import annotations

import torch
import torch.nn as nn

from src.models.feature_extractor import AudioFeatureExtractor


class SpectralMagnitudeLoss(nn.Module):
    def __init__(self, feature_extractor: AudioFeatureExtractor) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.loss = nn.L1Loss()

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        estimate_spec = self.feature_extractor.stft(estimate)
        target_spec = self.feature_extractor.stft(target)
        estimate_log_mag = self.feature_extractor.log_mag(estimate_spec)
        target_log_mag = self.feature_extractor.log_mag(target_spec)
        return self.loss(estimate_log_mag, target_log_mag)
