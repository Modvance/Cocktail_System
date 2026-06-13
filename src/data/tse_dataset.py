from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.audio import load_audio
from src.utils.config import resolve_path


class TSEDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        sample_rate: int,
        mixture_duration: float,
        enrollment_duration: float,
        training: bool = False,
        gain_augment_db: float = 0.0,
        limit: int | None = None,
    ) -> None:
        self.csv_path = resolve_path(csv_path)
        self.sample_rate = int(sample_rate)
        self.mixture_length = int(round(float(mixture_duration) * self.sample_rate))
        self.enrollment_length = int(round(float(enrollment_duration) * self.sample_rate))
        self.training = training
        self.gain_augment_db = float(gain_augment_db)
        with self.csv_path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.rows = rows[:limit] if limit is not None else rows

    def __len__(self) -> int:
        return len(self.rows)

    def _apply_gain(self, mixture: np.ndarray, target: np.ndarray, enrollment: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.training or self.gain_augment_db <= 0.0:
            return mixture, target, enrollment
        gain_db = float(np.random.uniform(-self.gain_augment_db, self.gain_augment_db))
        scale = float(10.0 ** (gain_db / 20.0))
        return mixture * scale, target * scale, enrollment * scale

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        mixture = load_audio(row["mixture_path"], self.sample_rate, self.mixture_length)
        target = load_audio(row["target_path"], self.sample_rate, self.mixture_length)
        enrollment = load_audio(row["enrollment_path"], self.sample_rate, self.enrollment_length)
        mixture, target, enrollment = self._apply_gain(mixture, target, enrollment)

        snr_db = float("nan") if row["snr_db"] in {"", "None"} else float(row["snr_db"])
        return {
            "mixture": torch.from_numpy(mixture.copy()).float(),
            "target": torch.from_numpy(target.copy()).float(),
            "enrollment": torch.from_numpy(enrollment.copy()).float(),
            "sample_id": row["sample_id"],
            "lang": row["lang"],
            "num_speakers": int(row["num_speakers"]),
            "snr_db": snr_db,
            "tir_db": float(row["tir_db"]),
            "overlap_mode": row["overlap_mode"],
            "target_speaker": row["target_speaker"],
            "interferer_speakers": row["interferer_speakers"],
        }


def create_dataloader(
    csv_path: str | Path,
    sample_rate: int,
    mixture_duration: float,
    enrollment_duration: float,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    training: bool = False,
    gain_augment_db: float = 0.0,
    limit: int | None = None,
) -> DataLoader:
    dataset = TSEDataset(
        csv_path=csv_path,
        sample_rate=sample_rate,
        mixture_duration=mixture_duration,
        enrollment_duration=enrollment_duration,
        training=training,
        gain_augment_db=gain_augment_db,
        limit=limit,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
