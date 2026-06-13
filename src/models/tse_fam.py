from __future__ import annotations

import torch
import torch.nn as nn

from src.models.conditioning import TargetConditioningModule
from src.models.feature_extractor import AudioFeatureExtractor
from src.models.mask_estimator import MaskEstimator
from src.models.mixture_encoder import MixtureEncoder
from src.models.reconstructor import WaveformReconstructor
from src.models.speaker_encoder import SpeakerEncoder


class TSEFAM(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        feature_cfg = config["feature"]
        model_cfg = config["model"]
        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=config["data"]["sample_rate"],
            n_fft=feature_cfg["n_fft"],
            win_length=feature_cfg["win_length"],
            hop_length=feature_cfg["hop_length"],
            n_mels=feature_cfg["n_mels"],
            f_min=feature_cfg.get("f_min", 80.0),
            f_max=feature_cfg.get("f_max", 7600.0),
            enrollment_feature_type=feature_cfg.get("enrollment_feature_type", "log_mel"),
            eps=feature_cfg.get("eps", 1e-8),
        )
        enroll_input_dim = self.feature_extractor.n_mels if feature_cfg.get("enrollment_feature_type", "log_mel") == "log_mel" else self.feature_extractor.freq_bins
        self.speaker_encoder = SpeakerEncoder(
            input_dim=enroll_input_dim,
            hidden_dim=model_cfg["speaker_hidden_dim"],
            embed_dim=model_cfg["speaker_embed_dim"],
            num_layers=model_cfg["speaker_num_layers"],
            dropout=model_cfg["dropout"],
        )
        self.mixture_encoder = MixtureEncoder(
            freq_bins=self.feature_extractor.freq_bins,
            hidden_dim=model_cfg["hidden_dim"],
            lstm_hidden_dim=model_cfg["lstm_hidden_dim"],
            lstm_layers=model_cfg["lstm_layers"],
            dropout=model_cfg["dropout"],
        )
        self.conditioning = TargetConditioningModule(
            hidden_dim=model_cfg["hidden_dim"],
            speaker_embed_dim=model_cfg["speaker_embed_dim"],
            use_enrollment=model_cfg.get("use_enrollment", True),
            use_film=model_cfg.get("use_film", True),
            use_attention=model_cfg.get("use_attention", True),
        )
        self.mask_estimator = MaskEstimator(
            hidden_dim=model_cfg["hidden_dim"],
            mask_hidden_dim=model_cfg.get("mask_hidden_dim", model_cfg["hidden_dim"]),
            mask_layers=model_cfg.get("mask_layers", 1),
            freq_bins=self.feature_extractor.freq_bins,
            dropout=model_cfg["dropout"],
        )
        self.reconstructor = WaveformReconstructor(self.feature_extractor)

    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.feature_extractor(mixture, enrollment)
        speaker_embedding = None
        if self.conditioning.use_enrollment:
            speaker_embedding = self.speaker_encoder(features["enrollment_feat"])
        mixture_feat = self.mixture_encoder(features["mixture_log_mag"])
        conditioned_feat, attention = self.conditioning(mixture_feat, speaker_embedding)
        mask = self.mask_estimator(conditioned_feat)
        estimated_mag = mask * features["mixture_mag"]
        estimated_waveform = self.reconstructor(estimated_mag, features["mixture_phase"], length=mixture.shape[-1])
        return {
            "estimated_waveform": estimated_waveform,
            "estimated_mag": estimated_mag,
            "mask": mask,
            "attention": attention,
            "speaker_embedding": speaker_embedding,
            "mixture_mag": features["mixture_mag"],
            "mixture_phase": features["mixture_phase"],
        }
