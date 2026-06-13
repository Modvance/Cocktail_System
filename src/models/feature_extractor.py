from __future__ import annotations

import math

import torch
import torch.nn as nn


class AudioFeatureExtractor(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        enrollment_feature_type: str = "log_mag",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.enrollment_feature_type = enrollment_feature_type
        self.eps = float(eps)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.register_buffer("mel_filterbank", self._build_mel_filterbank(), persistent=False)

    @property
    def freq_bins(self) -> int:
        return self.n_fft // 2 + 1

    def _hz_to_mel(self, hz: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filterbank(self) -> torch.Tensor:
        freq_bins = self.freq_bins
        min_mel = self._hz_to_mel(torch.tensor(self.f_min))
        max_mel = self._hz_to_mel(torch.tensor(self.f_max))
        mel_points = torch.linspace(min_mel, max_mel, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        fft_freqs = torch.linspace(0.0, self.sample_rate / 2.0, freq_bins)
        filters = torch.zeros(self.n_mels, freq_bins)
        for idx in range(self.n_mels):
            left, center, right = hz_points[idx], hz_points[idx + 1], hz_points[idx + 2]
            up = (fft_freqs - left) / max((center - left).item(), 1e-8)
            down = (right - fft_freqs) / max((right - center).item(), 1e-8)
            filters[idx] = torch.maximum(torch.zeros_like(fft_freqs), torch.minimum(up, down))
        return filters

    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            center=True,
            return_complex=True,
        )

    def istft(self, complex_spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(complex_spec.device),
            center=True,
            length=length,
        )

    def log_mag(self, complex_spec: torch.Tensor) -> torch.Tensor:
        return torch.log(complex_spec.abs() + self.eps)

    def log_mel(self, complex_spec: torch.Tensor) -> torch.Tensor:
        magnitude = complex_spec.abs()
        mel = torch.matmul(self.mel_filterbank.to(magnitude.device), magnitude)
        return torch.log(mel + self.eps)

    def reconstruct_from_mag_phase(self, magnitude: torch.Tensor, phase: torch.Tensor, length: int) -> torch.Tensor:
        complex_spec = torch.polar(magnitude, phase)
        return self.istft(complex_spec, length=length)

    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor) -> dict[str, torch.Tensor]:
        mixture_complex = self.stft(mixture)
        enrollment_complex = self.stft(enrollment)
        mixture_mag = mixture_complex.abs()
        mixture_log_mag = self.log_mag(mixture_complex)
        mixture_phase = torch.angle(mixture_complex)
        if self.enrollment_feature_type == "log_mel":
            enrollment_feat = self.log_mel(enrollment_complex)
        else:
            enrollment_feat = self.log_mag(enrollment_complex)
        return {
            "mixture_complex": mixture_complex,
            "mixture_mag": mixture_mag,
            "mixture_log_mag": mixture_log_mag,
            "mixture_phase": mixture_phase,
            "enrollment_feat": enrollment_feat,
        }
