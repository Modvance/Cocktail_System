from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_FILES = ["meta.json", "mixture.wav", "target_clean.wav", "enrollment.wav", "noise.wav"]
AUDIO_FILES = ["mixture.wav", "target_clean.wav", "enrollment.wav", "noise.wav"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check generated dataset structure and metadata.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", default="configs/dataset_build.yaml")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(config_path: str) -> dict[str, Any]:
    import yaml

    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def require_soundfile() -> None:
    if sf is None:
        raise RuntimeError("soundfile is required for dataset checking. Install it with `pip install soundfile`.")


def read_audio_info(path: Path) -> dict[str, Any]:
    require_soundfile()
    info = sf.info(str(path))
    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        mono_for_stats = audio.mean(axis=1)
    else:
        mono_for_stats = np.asarray(audio, dtype=np.float32)
    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_sec": info.duration,
        "peak": float(np.max(np.abs(audio))) if np.size(audio) else 0.0,
        "active_ratio": float(np.mean(np.abs(mono_for_stats) > 1e-3)) if mono_for_stats.size else 0.0,
        "is_silent": bool(np.max(np.abs(audio)) <= 1e-6) if np.size(audio) else True,
        "clipped": bool(np.max(np.abs(audio)) >= 0.999) if np.size(audio) else False,
        "power": float(np.mean(np.square(mono_for_stats))) if mono_for_stats.size else 0.0,
    }


def expected_duration(name: str, audio_cfg: dict[str, Any]) -> float:
    if name == "enrollment.wav":
        return float(audio_cfg["enrollment_duration"])
    return float(audio_cfg["target_duration"])


def inspect_sample(sample_root: Path, config: dict[str, Any], split_counts: dict[str, dict[int, int]], lang_counts: dict[str, int]) -> dict[str, Any]:
    result = {"sample": str(sample_root.relative_to(REPO_ROOT)), "missing": [], "audio": {}, "issues": []}
    for name in REQUIRED_FILES:
        path = sample_root / name
        if not path.exists():
            result["missing"].append(name)

    meta = None
    meta_path = sample_root / "meta.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        result["meta"] = meta
        lang_value = meta.get("lang")
        if lang_value:
            lang_counts[lang_value] = lang_counts.get(lang_value, 0) + 1
        split_name = meta.get("split")
        num_speakers = meta.get("num_speakers")
        if split_name and isinstance(num_speakers, int):
            split_counts.setdefault(split_name, {})
            split_counts[split_name][num_speakers] = split_counts[split_name].get(num_speakers, 0) + 1

    for name in AUDIO_FILES:
        path = sample_root / name
        if not path.exists():
            continue
        info = read_audio_info(path)
        result["audio"][name] = info

        duration_target = expected_duration(name, config["audio"])
        if abs(info["duration_sec"] - duration_target) > 0.05:
            result["issues"].append(f"{name} duration mismatch: {info['duration_sec']:.3f}s")
        if info["sample_rate"] != int(config["audio"]["sample_rate"]):
            result["issues"].append(f"{name} sample rate mismatch: {info['sample_rate']}")
        if info["clipped"]:
            result["issues"].append(f"{name} appears clipped")
        if info["peak"] > float(config["audio"]["peak_limit"]) + 1e-4:
            result["issues"].append(f"{name} peak exceeds limit: {info['peak']:.4f}")
        if name in {"mixture.wav", "target_clean.wav", "enrollment.wav"} and info["is_silent"]:
            result["issues"].append(f"{name} is silent")
        if name in {"target_clean.wav", "enrollment.wav"} and info["active_ratio"] < float(config["quality"]["active_ratio_threshold"]):
            result["issues"].append(f"{name} active ratio below threshold: {info['active_ratio']:.3f}")

    if meta is not None:
        target_sources = set(meta.get("target_source_files", []))
        enrollment_sources = set(meta.get("enrollment_source_files", []))
        if target_sources.intersection(enrollment_sources):
            result["issues"].append("target/enrollment source files overlap")
        if meta.get("target_speaker") is None:
            result["issues"].append("missing target speaker in meta")
        if not meta.get("interferer_speakers"):
            result["issues"].append("missing interferer speakers in meta")

        actual_snr = meta.get("snr_db_actual")
        target_snr = meta.get("snr_db_target")
        if actual_snr is not None and target_snr is not None:
            if abs(float(actual_snr) - float(target_snr)) > float(config["quality"]["snr_tolerance_db"]):
                result["issues"].append(f"snr mismatch: actual={float(actual_snr):.2f}, target={float(target_snr):.2f}")

        actual_tir = meta.get("tir_db_actual")
        target_tir = meta.get("tir_db_target")
        if actual_tir is not None and target_tir is not None:
            if abs(float(actual_tir) - float(target_tir)) > float(config["quality"]["tir_tolerance_db"]):
                result["issues"].append(f"tir mismatch: actual={float(actual_tir):.2f}, target={float(target_tir):.2f}")

    return result


def summarize_speaker_counts(split_counts: dict[str, dict[int, int]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split, counts in split_counts.items():
        summary[split] = {str(num_speakers): count for num_speakers, count in sorted(counts.items())}
    return summary


def ratio_issues(config: dict[str, Any], dataset_root: Path, split_counts: dict[str, dict[int, int]], lang_counts: dict[str, int]) -> list[str]:
    issues: list[str] = []
    sampling_cfg = config["sampling"]
    for split, counts in split_counts.items():
        total = sum(counts.values())
        if total == 0:
            continue
        expected = sampling_cfg["speaker_ratio"][split]
        actual_two = counts.get(2, 0) / total
        actual_three = counts.get(3, 0) / total
        if abs(actual_two - float(expected[2])) > 0.2:
            issues.append(f"{split} 2-speaker ratio deviates: actual={actual_two:.3f}, expected={float(expected[2]):.3f}")
        if abs(actual_three - float(expected[3])) > 0.2:
            issues.append(f"{split} 3-speaker ratio deviates: actual={actual_three:.3f}, expected={float(expected[3]):.3f}")

    dataset_cfg = config["datasets"].get(dataset_root.name, {})
    if dataset_cfg.get("language") == "bilingual":
        total_lang = sum(lang_counts.values())
        if total_lang > 0:
            expected_ratio = dataset_cfg.get("language_ratio", {})
            for lang, expected in expected_ratio.items():
                actual = lang_counts.get(lang, 0) / total_lang
                if abs(actual - float(expected)) > 0.2:
                    issues.append(f"language ratio for {lang} deviates: actual={actual:.3f}, expected={float(expected):.3f}")
    return issues


def main() -> None:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    config = load_config(args.config)
    sample_dirs = sorted(path for path in dataset_root.rglob("sample_*") if path.is_dir())
    split_counts: dict[str, dict[int, int]] = {}
    lang_counts: dict[str, int] = {}
    results = [inspect_sample(sample_dir, config, split_counts, lang_counts) for sample_dir in sample_dirs]
    missing_count = sum(1 for item in results if item["missing"])
    issue_count = sum(1 for item in results if item["issues"])
    dataset_level_issues = ratio_issues(config, dataset_root, split_counts, lang_counts)
    print(json.dumps({
        "dataset_root": str(dataset_root),
        "dataset": dataset_root.name,
        "sample_count": len(results),
        "samples_with_missing_required_files": missing_count,
        "samples_with_issues": issue_count,
        "dataset_level_issues": dataset_level_issues,
        "speaker_count_summary": summarize_speaker_counts(split_counts),
        "language_count_summary": lang_counts,
        "results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
