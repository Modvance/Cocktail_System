from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import soundfile as sf
except ImportError:
    sf = None


REPO_ROOT = Path(__file__).resolve().parents[2]
PREVIEW_FILES = ["meta.json", "mixture.wav", "target_clean.wav", "enrollment.wav", "noise.wav"]
WAVEFORM_FILES = ["mixture.wav", "target_clean.wav", "enrollment.wav", "noise.wav"]
TRACK_COLORS = {
    "mixture.wav": "#2563eb",
    "target_clean.wav": "#16a34a",
    "enrollment.wav": "#9333ea",
    "noise.wav": "#ea580c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create lightweight preview directories from generated samples.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def require_soundfile() -> None:
    if sf is None:
        raise RuntimeError("soundfile is required for preview generation. Install it with `pip install soundfile`.")


def load_mono_audio(path: Path) -> np.ndarray:
    require_soundfile()
    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32)


def compute_spectrogram(audio: np.ndarray, fft_size: int = 512, hop_size: int = 128) -> np.ndarray:
    if audio.size < fft_size:
        audio = np.pad(audio, (0, fft_size - audio.size))
    window = np.hanning(fft_size).astype(np.float32)
    frames = []
    for start in range(0, max(audio.size - fft_size + 1, 1), hop_size):
        chunk = audio[start:start + fft_size]
        if chunk.size < fft_size:
            chunk = np.pad(chunk, (0, fft_size - chunk.size))
        spectrum = np.fft.rfft(chunk * window)
        frames.append(np.abs(spectrum))
    if not frames:
        frames.append(np.zeros(fft_size // 2 + 1, dtype=np.float32))
    spec = np.stack(frames, axis=1)
    spec = np.log10(np.maximum(spec, 1e-6))
    spec = (spec - spec.min()) / max(spec.max() - spec.min(), 1e-6)
    return spec


def draw_waveform_preview(sample_dir: Path, output_path: Path) -> None:
    width = 1400
    track_height = 120
    padding = 24
    gap = 18
    total_height = padding * 2 + len(WAVEFORM_FILES) * track_height + (len(WAVEFORM_FILES) - 1) * gap
    image = Image.new("RGB", (width, total_height), "white")
    draw = ImageDraw.Draw(image)

    for index, name in enumerate(WAVEFORM_FILES):
        audio_path = sample_dir / name
        top = padding + index * (track_height + gap)
        bottom = top + track_height
        draw.rectangle((0, top, width - 1, bottom), outline="#d1d5db", width=1)
        draw.text((12, top + 8), name, fill="#111827")
        if not audio_path.exists():
            continue
        audio = load_mono_audio(audio_path)
        if audio.size == 0:
            continue
        step = max(1, int(np.ceil(audio.size / (width - 80))))
        reduced = audio[: step * ((audio.size + step - 1) // step)]
        reduced = reduced[: (reduced.size // step) * step]
        if reduced.size == 0:
            continue
        envelope = np.max(np.abs(reduced.reshape(-1, step)), axis=1)
        usable_width = min(envelope.size, width - 80)
        mid_y = top + track_height / 2
        amp = (track_height - 32) / 2
        color = TRACK_COLORS[name]
        for x in range(usable_width):
            value = float(envelope[x])
            x_pos = 70 + x
            y0 = mid_y - value * amp
            y1 = mid_y + value * amp
            draw.line((x_pos, y0, x_pos, y1), fill=color, width=1)
        draw.line((70, mid_y, 70 + usable_width, mid_y), fill="#9ca3af", width=1)

    image.save(output_path)


def draw_spectrogram_preview(sample_dir: Path, output_path: Path) -> None:
    width = 1400
    track_height = 120
    padding = 24
    gap = 18
    total_height = padding * 2 + len(WAVEFORM_FILES) * track_height + (len(WAVEFORM_FILES) - 1) * gap
    image = Image.new("RGB", (width, total_height), "white")
    draw = ImageDraw.Draw(image)

    for index, name in enumerate(WAVEFORM_FILES):
        audio_path = sample_dir / name
        top = padding + index * (track_height + gap)
        bottom = top + track_height
        draw.rectangle((0, top, width - 1, bottom), outline="#d1d5db", width=1)
        draw.text((12, top + 8), name, fill="#111827")
        if not audio_path.exists():
            continue
        audio = load_mono_audio(audio_path)
        if audio.size == 0:
            continue
        spec = compute_spectrogram(audio)
        spec_img = np.uint8(np.clip(spec[::-1, :] * 255.0, 0, 255))
        spec_rgb = np.stack([spec_img, np.minimum(spec_img + 30, 255), np.minimum(255 - spec_img // 2, 255)], axis=-1).astype(np.uint8)
        spec_pil = Image.fromarray(spec_rgb, mode="RGB").resize((width - 80, track_height - 28))
        image.paste(spec_pil, (70, top + 20))

    image.save(output_path)


def main() -> None:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    output_root = resolve_path(args.output_root)
    dataset_name = dataset_root.name
    preview_root = output_root / dataset_name
    preview_root.mkdir(parents=True, exist_ok=True)

    all_sample_dirs = sorted(path for path in dataset_root.rglob("sample_*") if path.is_dir())
    rng = random.Random(args.seed)
    if len(all_sample_dirs) <= args.limit:
        sample_dirs = all_sample_dirs
    else:
        sample_dirs = sorted(rng.sample(all_sample_dirs, args.limit))

    manifest = []
    for sample_dir in sample_dirs:
        target_dir = preview_root / sample_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in PREVIEW_FILES:
            source = sample_dir / name
            if source.exists():
                shutil.copy2(source, target_dir / name)
                copied.append(name)
        draw_waveform_preview(sample_dir, target_dir / "waveform.png")
        draw_spectrogram_preview(sample_dir, target_dir / "spectrogram.png")
        manifest.append({
            "sample": sample_dir.name,
            "copied_files": copied,
            "waveform": "waveform.png",
            "spectrogram": "spectrogram.png",
        })

    (preview_root / "preview_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote preview data to {preview_root}")


if __name__ == "__main__":
    main()
