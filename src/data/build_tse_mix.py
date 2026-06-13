from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import soundfile as sf
except ImportError:
    sf = None


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FIELDS = [
    "sample_id",
    "split",
    "dataset",
    "lang",
    "mixture_path",
    "target_path",
    "enrollment_path",
    "noise_path",
    "num_speakers",
    "target_speaker",
    "interferer_speakers",
    "snr_db",
    "tir_db",
    "overlap_mode",
    "duration",
    "enroll_duration",
    "target_source_files",
    "enrollment_source_files",
]


class SampleBuildError(RuntimeError):
    pass


def load_yaml(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def require_soundfile() -> None:
    if sf is None:
        raise RuntimeError("soundfile is required for real audio synthesis. Install it with `pip install soundfile`.")


def load_audio(path_str: str, target_sr: int) -> np.ndarray:
    require_soundfile()
    path = resolve_path(path_str)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != target_sr:
        raise RuntimeError(f"Sample rate mismatch for {path}: expected {target_sr}, got {sample_rate}")
    return np.asarray(audio, dtype=np.float32)


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    require_soundfile()
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="FLOAT")


def peak_limit(audio: np.ndarray, limit: float) -> np.ndarray:
    current_peak = peak(audio)
    if current_peak <= limit or current_peak <= 1e-8:
        return audio.copy()
    return audio * (limit / current_peak)


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float32))))


def normalize_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    current = rms(audio)
    if current <= 1e-8:
        return audio.copy()
    return audio * (target_rms / current)


def peak(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def active_ratio(audio: np.ndarray, threshold: float = 1e-3) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) > threshold))


def trim_or_pad(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.size >= length:
        return audio[:length].copy()
    return np.pad(audio, (0, length - audio.size))


def add_fade(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    if fade_samples <= 0 or audio.size == 0:
        return audio
    fade_samples = min(fade_samples, audio.size // 2)
    if fade_samples <= 0:
        return audio
    faded = audio.copy()
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    faded[:fade_samples] *= ramp
    faded[-fade_samples:] *= ramp[::-1]
    return faded


def assemble_segment(rows: list[dict[str, str]], duration_sec: float, sample_rate: int, target_rms_value: float, min_active_ratio: float, normalize: bool = True) -> tuple[np.ndarray, list[str]]:
    if not rows:
        raise SampleBuildError("No source rows available to assemble segment")

    target_length = int(round(duration_sec * sample_rate))
    parts: list[np.ndarray] = []
    used_files: list[str] = []
    silence = np.zeros(int(0.12 * sample_rate), dtype=np.float32)

    for row in rows:
        audio = load_audio(row["audio_path"], sample_rate)
        if normalize:
            audio = normalize_rms(audio, target_rms_value)
        audio = add_fade(audio, int(0.01 * sample_rate))
        if audio.size == 0:
            continue
        used_files.append(row["audio_path"])
        parts.append(audio)
        if sum(part.size for part in parts) >= target_length:
            break
        parts.append(silence)

    if not used_files:
        raise SampleBuildError("Failed to assemble non-empty segment")

    segment = trim_or_pad(np.concatenate(parts), target_length)
    if active_ratio(segment) < min_active_ratio:
        raise SampleBuildError("Assembled segment active ratio below threshold")
    return segment, used_files


def choose_target_and_enrollment(candidates: list[dict[str, str]], enrollment_duration: float, target_duration: float, sample_rate: int, target_rms_value: float, min_active_ratio: float, rng: random.Random) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    shuffled = candidates[:]
    rng.shuffle(shuffled)

    for pivot in range(1, len(shuffled)):
        target_rows = shuffled[:pivot]
        enrollment_rows = shuffled[pivot:]
        if not enrollment_rows:
            continue
        try:
            target_audio, target_sources = assemble_segment(target_rows, target_duration, sample_rate, target_rms_value, min_active_ratio)
            enrollment_audio, enrollment_sources = assemble_segment(enrollment_rows, enrollment_duration, sample_rate, target_rms_value, min_active_ratio)
        except SampleBuildError:
            continue
        if set(target_sources).intersection(enrollment_sources):
            continue
        return target_audio, target_sources, enrollment_audio, enrollment_sources
    raise SampleBuildError("Could not construct disjoint target/enrollment segments")


def layout_interferer(audio: np.ndarray, sample_length: int, overlap_mode: str, rng: random.Random) -> np.ndarray:
    base = trim_or_pad(audio, sample_length)
    if overlap_mode == "full_overlap":
        return base
    max_offset = min(sample_length // 10, int(0.8 * 16000))
    offset = rng.randint(0, max_offset)
    placed = np.zeros(sample_length, dtype=np.float32)
    keep = min(sample_length - offset, base.size)
    if keep > 0:
        placed[offset:offset + keep] = base[:keep]
    return placed


def sample_num_speakers(split: str, speaker_ratio_cfg: dict[str, Any], rng: random.Random) -> int:
    ratio_cfg = speaker_ratio_cfg[split]
    return 2 if rng.random() < float(ratio_cfg[2]) else 3


def sample_overlap_mode(overlap_cfg: dict[str, float], rng: random.Random) -> str:
    return "full_overlap" if rng.random() < float(overlap_cfg["full_overlap"]) else "random_offset"


def sample_snr(split: str, sampling_cfg: dict[str, Any], rng: random.Random) -> float:
    if split == "train":
        low, high = sampling_cfg["train_snr_range_db"]
        return round(rng.uniform(float(low), float(high)), 2)
    return float(rng.choice(sampling_cfg["eval_snr_values_db"]))


def sample_tir(tir_range: list[float], rng: random.Random) -> float:
    return round(rng.uniform(float(tir_range[0]), float(tir_range[1])), 2)


def scale_interferers_for_tir(target_audio: np.ndarray, interferer_mix: np.ndarray, tir_db: float) -> np.ndarray:
    target_power = float(np.mean(np.square(target_audio)))
    interferer_power = float(np.mean(np.square(interferer_mix)))
    if interferer_power <= 1e-8:
        return interferer_mix.copy()
    desired_ratio = 10.0 ** (tir_db / 10.0)
    desired_interferer_power = target_power / max(desired_ratio, 1e-8)
    scale = math.sqrt(desired_interferer_power / interferer_power)
    return interferer_mix * scale


def scale_noise_for_snr(speech_mix: np.ndarray, noise_audio: np.ndarray, snr_db: float) -> np.ndarray:
    speech_power = float(np.mean(np.square(speech_mix)))
    noise_power = float(np.mean(np.square(noise_audio)))
    if noise_power <= 1e-8:
        return np.zeros_like(noise_audio)
    alpha = math.sqrt(speech_power / (noise_power * (10.0 ** (snr_db / 10.0))))
    return noise_audio * alpha


def limit_peak_bundle(audios: list[np.ndarray], peak_limit: float) -> list[np.ndarray]:
    current_peak = max(peak(audio) for audio in audios)
    if current_peak <= peak_limit or current_peak <= 1e-8:
        return [audio.copy() for audio in audios]
    scale = peak_limit / current_peak
    return [audio * scale for audio in audios]


def actual_snr_db(speech_mix: np.ndarray, noise_audio: np.ndarray) -> float | None:
    noise_power = float(np.mean(np.square(noise_audio)))
    if noise_power <= 1e-8:
        return None
    speech_power = float(np.mean(np.square(speech_mix)))
    if speech_power <= 1e-8:
        return None
    return 10.0 * math.log10(speech_power / noise_power)


def actual_tir_db(target_audio: np.ndarray, interferer_mix: np.ndarray) -> float | None:
    interferer_power = float(np.mean(np.square(interferer_mix)))
    if interferer_power <= 1e-8:
        return None
    target_power = float(np.mean(np.square(target_audio)))
    if target_power <= 1e-8:
        return None
    return 10.0 * math.log10(target_power / interferer_power)


def build_sample(sample_id: str, split: str, dataset_name: str, dataset_cfg: dict[str, Any], config: dict[str, Any], speech_rows: list[dict[str, str]], noise_rows: list[dict[str, str]], sample_root: Path, rng: random.Random, save_sources: bool, lang: str | None = None) -> dict[str, Any]:
    audio_cfg = config["audio"]
    quality_cfg = config["quality"]
    sampling_cfg = config["sampling"]
    sample_rate = int(audio_cfg["sample_rate"])
    target_duration = float(audio_cfg["target_duration"])
    enrollment_duration = float(audio_cfg["enrollment_duration"])
    sample_length = int(round(target_duration * sample_rate))

    by_speaker: dict[str, list[dict[str, str]]] = {}
    for row in speech_rows:
        by_speaker.setdefault(row["speaker_id"], []).append(row)
    eligible_speakers = [speaker for speaker, rows in by_speaker.items() if len(rows) >= 2]
    if len(eligible_speakers) < 2:
        raise SampleBuildError("Need at least two eligible speakers with >=2 utterances")

    num_speakers = sample_num_speakers(split, sampling_cfg["speaker_ratio"], rng)
    overlap_mode = sample_overlap_mode(sampling_cfg["overlap_ratio"], rng)
    tir_db = sample_tir(sampling_cfg["tir_range_db"], rng)
    snr_db = sample_snr(split, sampling_cfg, rng)

    target_speaker = rng.choice(eligible_speakers)
    target_audio, target_sources, enrollment_audio, enrollment_sources = choose_target_and_enrollment(
        by_speaker[target_speaker],
        enrollment_duration,
        target_duration,
        sample_rate,
        float(audio_cfg["target_rms"]),
        float(quality_cfg["active_ratio_threshold"]),
        rng,
    )

    interferer_speakers = rng.sample([speaker for speaker in eligible_speakers if speaker != target_speaker], num_speakers - 1)
    interferer_tracks: list[np.ndarray] = []
    saved_source_payloads: list[tuple[str, np.ndarray]] = []
    for index, interferer_speaker in enumerate(interferer_speakers, start=1):
        interferer_audio, interferer_sources = assemble_segment(
            rng.sample(by_speaker[interferer_speaker], len(by_speaker[interferer_speaker])),
            target_duration,
            sample_rate,
            float(audio_cfg["target_rms"]),
            float(quality_cfg["active_ratio_threshold"]),
        )
        laid_out = layout_interferer(interferer_audio, sample_length, overlap_mode, rng)
        interferer_tracks.append(laid_out)
        if save_sources:
            saved_source_payloads.append((f"source_{index}.wav", laid_out))

    interferer_mix = np.sum(interferer_tracks, axis=0) if interferer_tracks else np.zeros(sample_length, dtype=np.float32)
    interferer_mix = scale_interferers_for_tir(target_audio, interferer_mix, tir_db)
    speech_mix = target_audio + interferer_mix

    add_noise = bool(noise_rows) and rng.random() < float(sampling_cfg["noise_prob"])
    if add_noise:
        noise_row = rng.choice(noise_rows)
        raw_noise = assemble_segment([noise_row], target_duration, sample_rate, float(audio_cfg["target_rms"]), 0.0, normalize=False)[0]
        noise_audio = scale_noise_for_snr(speech_mix, raw_noise, snr_db)
        noise_path_value = noise_row["audio_path"]
    else:
        noise_audio = np.zeros(sample_length, dtype=np.float32)
        noise_path_value = ""

    mixture = speech_mix + noise_audio
    limited_target, limited_interferer, limited_noise, limited_mixture = limit_peak_bundle(
        [target_audio, interferer_mix, noise_audio, mixture],
        float(audio_cfg["peak_limit"]),
    )

    final_enrollment = peak_limit(enrollment_audio, float(audio_cfg["peak_limit"]))
    final_target_active_ratio = active_ratio(limited_target)
    final_enrollment_active_ratio = active_ratio(final_enrollment)
    active_ratio_floor = float(quality_cfg["active_ratio_threshold"]) + 0.01
    if final_target_active_ratio < active_ratio_floor:
        raise SampleBuildError(f"Final target active ratio below threshold: {final_target_active_ratio:.3f}")
    if final_enrollment_active_ratio < active_ratio_floor:
        raise SampleBuildError(f"Final enrollment active ratio below threshold: {final_enrollment_active_ratio:.3f}")

    write_audio(sample_root / "target_clean.wav", limited_target, sample_rate)
    write_audio(sample_root / "enrollment.wav", final_enrollment, sample_rate)
    write_audio(sample_root / "noise.wav", limited_noise, sample_rate)
    write_audio(sample_root / "mixture.wav", limited_mixture, sample_rate)
    for file_name, audio in saved_source_payloads:
        write_audio(sample_root / file_name, audio, sample_rate)

    meta = {
        "sample_id": sample_id,
        "split": split,
        "dataset": dataset_name,
        "lang": lang or dataset_cfg["language"],
        "sample_rate": sample_rate,
        "target_speaker": target_speaker,
        "interferer_speakers": interferer_speakers,
        "num_speakers": num_speakers,
        "overlap_mode": overlap_mode,
        "target_duration": target_duration,
        "enrollment_duration": enrollment_duration,
        "snr_db_target": snr_db,
        "tir_db_target": tir_db,
        "snr_db_actual": actual_snr_db(limited_target + limited_interferer, limited_noise),
        "tir_db_actual": actual_tir_db(limited_target, limited_interferer),
        "target_source_files": target_sources,
        "enrollment_source_files": enrollment_sources,
        "noise_source_file": noise_path_value,
        "active_ratio": {
            "target_clean": final_target_active_ratio,
            "enrollment": final_enrollment_active_ratio,
            "mixture": active_ratio(limited_mixture),
        },
        "peak": {
            "target_clean": peak(limited_target),
            "enrollment": peak(final_enrollment),
            "noise": peak(limited_noise),
            "mixture": peak(limited_mixture),
        },
    }
    (sample_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "sample_id": sample_id,
        "split": split,
        "dataset": dataset_name,
        "lang": lang or dataset_cfg["language"],
        "mixture_path": str((sample_root / "mixture.wav").relative_to(REPO_ROOT)),
        "target_path": str((sample_root / "target_clean.wav").relative_to(REPO_ROOT)),
        "enrollment_path": str((sample_root / "enrollment.wav").relative_to(REPO_ROOT)),
        "noise_path": str((sample_root / "noise.wav").relative_to(REPO_ROOT)),
        "num_speakers": num_speakers,
        "target_speaker": target_speaker,
        "interferer_speakers": json.dumps(interferer_speakers, ensure_ascii=False),
        "snr_db": snr_db,
        "tir_db": tir_db,
        "overlap_mode": overlap_mode,
        "duration": target_duration,
        "enroll_duration": enrollment_duration,
        "target_source_files": json.dumps(target_sources, ensure_ascii=False),
        "enrollment_source_files": json.dumps(enrollment_sources, ensure_ascii=False),
    }


def sample_language(language_ratio: dict[str, float], rng: random.Random) -> str:
    en_ratio = float(language_ratio.get("en", 0.5))
    return "en" if rng.random() < en_ratio else "zh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pilot TSE datasets from manifests.")
    parser.add_argument("--config", default="configs/dataset_build.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", default="train")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save-sources", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    manifests_root = resolve_path(config["paths"]["manifests_root"])
    generated_root = resolve_path(config["paths"]["generated_root"])
    dataset_name = args.dataset
    dataset_cfg = config["datasets"][dataset_name]
    rng = random.Random(args.seed if args.seed is not None else config.get("seed", 42))
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]

    bilingual_manifests: dict[str, dict[str, list[dict[str, str]]]] = {}
    if dataset_cfg["language"] == "bilingual":
        for lang, manifest_map in dataset_cfg["speech_manifests"].items():
            bilingual_manifests[lang] = {}
            for split, manifest_name in manifest_map.items():
                bilingual_manifests[lang][split] = read_manifest(manifests_root / manifest_name)

    dataset_root = generated_root / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {"dataset": dataset_name, "splits": {}, "mode": "audio_enabled"}
    quality_lines = ["# Quality Report", "", f"dataset: {dataset_name}", ""]
    dataset_lang_counts: dict[str, int] = {}

    for split in splits:
        noise_manifest_name = dataset_cfg["noise_manifests"][split]
        noise_rows = read_manifest(manifests_root / noise_manifest_name)
        requested = args.num_samples or dataset_cfg["sample_counts"][split]
        split_rows: list[dict[str, Any]] = []
        split_root = dataset_root / split
        split_root.mkdir(parents=True, exist_ok=True)

        if dataset_cfg["language"] == "bilingual":
            manifest_label = json.dumps({lang: dataset_cfg["speech_manifests"][lang][split] for lang in dataset_cfg["speech_manifests"]}, ensure_ascii=False)
        else:
            manifest_label = dataset_cfg["speech_manifests"][split]
            speech_rows = read_manifest(manifests_root / manifest_label)

        built = 0
        attempts = 0
        lang_counts: dict[str, int] = {}
        max_attempts = max(requested * int(config["runtime"].get("max_attempts_per_sample", 50)), requested)
        errors: list[str] = []
        while built < requested and attempts < max_attempts:
            attempts += 1
            sample_id = f"sample_{built + 1:06d}"
            sample_root = split_root / sample_id
            sample_root.mkdir(parents=True, exist_ok=True)
            try:
                sample_lang = None
                if dataset_cfg["language"] == "bilingual":
                    sample_lang = sample_language(dataset_cfg["language_ratio"], rng)
                    current_speech_rows = bilingual_manifests[sample_lang][split]
                else:
                    current_speech_rows = speech_rows
                row = build_sample(sample_id, split, dataset_name, dataset_cfg, config, current_speech_rows, noise_rows, sample_root, rng, args.save_sources or bool(config["runtime"].get("save_sources", False)), lang=sample_lang)
            except (SampleBuildError, RuntimeError) as exc:
                errors.append(f"{sample_id}: {exc}")
                for child in sample_root.iterdir():
                    child.unlink()
                sample_root.rmdir()
                continue
            split_rows.append(row)
            built += 1
            lang_key = row["lang"]
            lang_counts[lang_key] = lang_counts.get(lang_key, 0) + 1
            dataset_lang_counts[lang_key] = dataset_lang_counts.get(lang_key, 0) + 1

        csv_path = dataset_root / f"{split}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(split_rows)

        speaker_counts = {
            "2": sum(1 for row in split_rows if int(row["num_speakers"]) == 2),
            "3": sum(1 for row in split_rows if int(row["num_speakers"]) == 3),
        }
        stats["splits"][split] = {
            "requested_samples": requested,
            "generated_samples": built,
            "attempts": attempts,
            "errors_preview": errors[:10],
            "language_counts": lang_counts,
            "speaker_counts": speaker_counts,
        }
        quality_lines.extend([
            f"## {split}",
            f"- requested_samples: {requested}",
            f"- generated_samples: {built}",
            f"- attempts: {attempts}",
            f"- manifest: {manifest_label}",
            f"- noise_manifest: {noise_manifest_name}",
            f"- language_counts: {json.dumps(lang_counts, ensure_ascii=False)}",
            f"- speaker_counts: {json.dumps(speaker_counts, ensure_ascii=False)}",
            "",
        ])

    stats["language_counts"] = dataset_lang_counts
    (dataset_root / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (dataset_root / "quality_report.md").write_text("\n".join(quality_lines) + "\n", encoding="utf-8")
    print(f"Generated dataset outputs under {dataset_root}")


if __name__ == "__main__":
    main()
