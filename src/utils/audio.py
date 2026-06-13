from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

from src.utils.config import resolve_path


def require_soundfile() -> None:
    if sf is None:
        raise RuntimeError("soundfile is required. Install it with `pip install soundfile`.")


def trim_or_pad(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[0] >= length:
        return np.asarray(audio[:length], dtype=np.float32)
    return np.pad(np.asarray(audio, dtype=np.float32), (0, length - audio.shape[0]))


def load_audio(path: str | Path, target_sr: int, length: int | None = None) -> np.ndarray:
    require_soundfile()
    resolved = resolve_path(path)
    audio, sample_rate = sf.read(str(resolved), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != target_sr:
        raise RuntimeError(f"Sample rate mismatch for {resolved}: expected {target_sr}, got {sample_rate}")
    if length is not None:
        audio = trim_or_pad(audio, length)
    return audio


def write_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    require_soundfile()
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(resolved), np.asarray(audio, dtype=np.float32), sample_rate, subtype="FLOAT")
