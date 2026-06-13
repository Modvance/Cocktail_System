from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.utils.audio import load_audio
from src.utils.config import resolve_path

TRACK_COLORS = {
    "mixture": "#2563eb",
    "target": "#16a34a",
    "estimate": "#dc2626",
    "enrollment": "#9333ea",
}


def compute_spectrogram(audio: np.ndarray, fft_size: int = 512, hop_size: int = 128) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
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


def save_waveform_compare(output_path: str | Path, tracks: list[tuple[str, np.ndarray]]) -> Path:
    resolved = resolve_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    width = 1400
    track_height = 120
    padding = 24
    gap = 18
    total_height = padding * 2 + len(tracks) * track_height + max(len(tracks) - 1, 0) * gap
    image = Image.new("RGB", (width, total_height), "white")
    draw = ImageDraw.Draw(image)

    for index, (name, audio) in enumerate(tracks):
        top = padding + index * (track_height + gap)
        bottom = top + track_height
        draw.rectangle((0, top, width - 1, bottom), outline="#d1d5db", width=1)
        draw.text((12, top + 8), name, fill="#111827")
        audio = np.asarray(audio, dtype=np.float32)
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
        color = TRACK_COLORS.get(name, "#111827")
        for x in range(usable_width):
            value = float(envelope[x])
            x_pos = 70 + x
            y0 = mid_y - value * amp
            y1 = mid_y + value * amp
            draw.line((x_pos, y0, x_pos, y1), fill=color, width=1)
        draw.line((70, mid_y, 70 + usable_width, mid_y), fill="#9ca3af", width=1)

    image.save(resolved)
    return resolved


def save_spectrogram_compare(output_path: str | Path, tracks: list[tuple[str, np.ndarray]]) -> Path:
    resolved = resolve_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    width = 1400
    track_height = 120
    padding = 24
    gap = 18
    total_height = padding * 2 + len(tracks) * track_height + max(len(tracks) - 1, 0) * gap
    image = Image.new("RGB", (width, total_height), "white")
    draw = ImageDraw.Draw(image)

    for index, (name, audio) in enumerate(tracks):
        top = padding + index * (track_height + gap)
        bottom = top + track_height
        draw.rectangle((0, top, width - 1, bottom), outline="#d1d5db", width=1)
        draw.text((12, top + 8), name, fill="#111827")
        spec = compute_spectrogram(np.asarray(audio, dtype=np.float32))
        spec_img = np.uint8(np.clip(spec[::-1, :] * 255.0, 0, 255))
        spec_rgb = np.stack([spec_img, np.minimum(spec_img + 30, 255), np.minimum(255 - spec_img // 2, 255)], axis=-1).astype(np.uint8)
        spec_pil = Image.fromarray(spec_rgb, mode="RGB").resize((width - 80, track_height - 28))
        image.paste(spec_pil, (70, top + 20))

    image.save(resolved)
    return resolved


def save_matrix_heatmap(output_path: str | Path, matrix: np.ndarray, title: str) -> Path:
    resolved = resolve_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    matrix = matrix - matrix.min()
    matrix = matrix / max(float(matrix.max()), 1e-6)
    matrix_img = np.uint8(np.clip(matrix * 255.0, 0, 255))
    heatmap_rgb = np.stack([
        matrix_img,
        np.minimum(matrix_img + 40, 255),
        np.minimum(255 - matrix_img // 3, 255),
    ], axis=-1).astype(np.uint8)
    body = Image.fromarray(heatmap_rgb, mode="RGB").resize((1320, 320))
    image = Image.new("RGB", (1400, 380), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), title, fill="#111827")
    draw.rectangle((39, 39, 1361, 361), outline="#d1d5db", width=1)
    image.paste(body, (40, 40))
    image.save(resolved)
    return resolved


def save_metric_bar_chart(output_path: str | Path, rows: list[dict[str, Any]], label_key: str, value_key: str, title: str) -> Path:
    resolved = resolve_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    width = 1200
    height = 700
    left = 90
    right = 40
    top = 60
    bottom = 120
    chart_width = width - left - right
    chart_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 20), title, fill="#111827")
    draw.line((left, top, left, top + chart_height), fill="#111827", width=2)
    draw.line((left, top + chart_height, left + chart_width, top + chart_height), fill="#111827", width=2)

    labels = [str(row[label_key]) for row in rows]
    values = [float(row[value_key]) for row in rows]
    if not values:
        image.save(resolved)
        return resolved
    max_abs = max(max(abs(v) for v in values), 1.0)
    zero_y = top + chart_height / 2 if min(values) < 0.0 else top + chart_height
    bar_width = max(chart_width // max(len(values) * 2, 1), 24)
    gap = max((chart_width - bar_width * len(values)) // max(len(values), 1), 12)

    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = left + gap // 2 + idx * (bar_width + gap)
        x1 = x0 + bar_width
        scaled = (value / max_abs) * (chart_height / 2 if min(values) < 0.0 else chart_height)
        y1 = zero_y
        y0 = zero_y - scaled
        if y0 > y1:
            y0, y1 = y1, y0
        draw.rectangle((x0, y0, x1, y1), fill="#2563eb", outline="#1d4ed8")
        draw.text((x0, top + chart_height + 16), label, fill="#111827")
        draw.text((x0, y0 - 18 if value >= 0 else y1 + 4), f"{value:.2f}", fill="#111827")

    image.save(resolved)
    return resolved


def compose_overview(output_dir: str | Path, image_names: list[str], output_name: str = "overview.png") -> Path:
    output_root = resolve_path(output_dir)
    images = [Image.open(output_root / name).convert("RGB") for name in image_names if (output_root / name).exists()]
    if not images:
        raise RuntimeError(f"No images found under {output_root}")
    width = max(image.width for image in images)
    total_height = sum(image.height for image in images) + 16 * (len(images) - 1)
    canvas = Image.new("RGB", (width, total_height), "white")
    top = 0
    for image in images:
        canvas.paste(image, (0, top))
        top += image.height + 16
    output_path = output_root / output_name
    canvas.save(output_path)
    return output_path


def save_case_visualizations(
    output_dir: str | Path,
    mixture: np.ndarray,
    enrollment: np.ndarray,
    estimate: np.ndarray,
    target: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    attention: np.ndarray | None = None,
) -> list[Path]:
    output_root = resolve_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    waveform_tracks = [("mixture", mixture)]
    spectrogram_tracks = [("mixture", mixture)]
    if target is not None:
        waveform_tracks.append(("target", target))
        spectrogram_tracks.append(("target", target))
    waveform_tracks.extend([("estimate", estimate), ("enrollment", enrollment)])
    spectrogram_tracks.extend([("estimate", estimate), ("enrollment", enrollment)])

    outputs = [
        save_waveform_compare(output_root / "waveform_compare.png", waveform_tracks),
        save_spectrogram_compare(output_root / "spectrogram_compare.png", spectrogram_tracks),
    ]
    if mask is not None:
        outputs.append(save_matrix_heatmap(output_root / "mask.png", np.asarray(mask)[::-1, :], "estimated mask"))
    if attention is not None:
        outputs.append(save_matrix_heatmap(output_root / "attention_weights.png", np.asarray(attention)[None, :], "attention weights"))
    overview_names = [path.name for path in outputs]
    outputs.append(compose_overview(output_root, overview_names))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create TSE visualizations.")
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--enrollment", required=True)
    parser.add_argument("--estimate", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--target")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mixture = load_audio(args.mixture, args.sample_rate)
    enrollment = load_audio(args.enrollment, args.sample_rate)
    estimate = load_audio(args.estimate, args.sample_rate)
    target = load_audio(args.target, args.sample_rate) if args.target else None
    outputs = save_case_visualizations(
        output_dir=args.out_dir,
        mixture=mixture,
        enrollment=enrollment,
        estimate=estimate,
        target=target,
    )
    print({"output_dir": str(resolve_path(args.out_dir)), "files": [str(path) for path in outputs]})


if __name__ == "__main__":
    main()
