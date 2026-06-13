from __future__ import annotations

import argparse
import csv
import shutil
import wave
from pathlib import Path
from typing import Any

import yaml

try:
    import soundfile as sf
except ImportError:
    sf = None


REPO_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(message, flush=True)


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


def split_path_candidates(source_cfg: dict[str, Any], split_rel: str) -> list[Path]:
    root = resolve_path(source_cfg["extracted_root"])
    rel_path = Path(split_rel)
    candidates = [root / rel_path, root / root.name / rel_path, root / rel_path.name]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def resolve_split_path(source_cfg: dict[str, Any], split_rel: str) -> Path:
    for candidate in split_path_candidates(source_cfg, split_rel):
        if candidate.exists():
            return candidate
    return split_path_candidates(source_cfg, split_rel)[0]


def list_audio_files(root: Path, allowed_extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    for ext in allowed_extensions:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def read_audio_metadata(path: Path) -> dict[str, Any]:
    if sf is not None:
        info = sf.info(str(path))
        return {
            "sample_rate": info.samplerate,
            "num_channels": info.channels,
            "duration_sec": info.duration,
        }

    suffix = path.suffix.lower()
    if suffix == ".wav":
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            sample_rate = wf.getframerate()
            return {
                "sample_rate": sample_rate,
                "num_channels": wf.getnchannels(),
                "duration_sec": frames / max(sample_rate, 1),
            }
    if suffix == ".flac" and shutil.which("soxi"):
        import subprocess

        output = subprocess.run(["soxi", str(path)], check=True, capture_output=True, text=True).stdout
        sample_rate = None
        channels = None
        duration_sec = None
        for line in output.splitlines():
            if "Sample Rate" in line:
                sample_rate = int(line.split(":", 1)[1].strip())
            elif "Channels" in line:
                channels = int(line.split(":", 1)[1].strip())
            elif "Duration" in line:
                raw = line.split(":", 1)[1].strip().split("=")[0].strip()
                h, m, s = raw.split(":")
                duration_sec = int(h) * 3600 + int(m) * 60 + float(s)
        return {
            "sample_rate": sample_rate,
            "num_channels": channels,
            "duration_sec": duration_sec,
        }
    return {
        "sample_rate": None,
        "num_channels": None,
        "duration_sec": None,
    }


def speech_identifiers(source_name: str, split_path: Path, audio_path: Path) -> tuple[str, str]:
    relative_parts = audio_path.relative_to(split_path).parts
    if source_name in {"librispeech", "aishell1"}:
        speaker_id = relative_parts[0]
        utterance_id = audio_path.stem
    else:
        speaker_id = "unknown"
        utterance_id = audio_path.stem
    return speaker_id, utterance_id


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_speech_manifest(source_name: str, split_name: str, split_path: Path, language: str, allowed_extensions: set[str]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    audio_files = list_audio_files(split_path, allowed_extensions)
    log(f"[build_manifest]   {source_name}/{split_name}: found {len(audio_files)} speech files")
    for index, audio_path in enumerate(audio_files, start=1):
        if index == 1 or index % 5000 == 0:
            log(f"[build_manifest]   {source_name}/{split_name}: processing {index}/{len(audio_files)}")
        try:
            metadata = read_audio_metadata(audio_path)
        except Exception as exc:
            skipped += 1
            if skipped <= 3:
                log(f"[build_manifest]   {source_name}/{split_name}: skip unreadable {audio_path.relative_to(REPO_ROOT)} ({exc})")
            continue
        speaker_id, utterance_id = speech_identifiers(source_name, split_path, audio_path)
        rows.append(
            {
                "dataset": source_name,
                "split": split_name,
                "speaker_id": speaker_id,
                "utterance_id": utterance_id,
                "audio_path": str(audio_path.relative_to(REPO_ROOT)),
                "duration_sec": metadata["duration_sec"],
                "sample_rate": metadata["sample_rate"],
                "num_channels": metadata["num_channels"],
                "language": language,
                "source_relpath": str(audio_path.relative_to(split_path)),
            }
        )
    return rows, skipped


def build_noise_manifest(source_name: str, split_name: str, split_path: Path, allowed_extensions: set[str]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    audio_files = list_audio_files(split_path, allowed_extensions)
    log(f"[build_manifest]   {source_name}/{split_name}: found {len(audio_files)} noise files")
    for index, audio_path in enumerate(audio_files, start=1):
        if index == 1 or index % 5000 == 0:
            log(f"[build_manifest]   {source_name}/{split_name}: processing {index}/{len(audio_files)}")
        try:
            metadata = read_audio_metadata(audio_path)
        except Exception as exc:
            skipped += 1
            if skipped <= 3:
                log(f"[build_manifest]   {source_name}/{split_name}: skip unreadable {audio_path.relative_to(REPO_ROOT)} ({exc})")
            continue
        rows.append(
            {
                "dataset": source_name,
                "split": split_name,
                "noise_id": audio_path.stem,
                "audio_path": str(audio_path.relative_to(REPO_ROOT)),
                "duration_sec": metadata["duration_sec"],
                "sample_rate": metadata["sample_rate"],
                "num_channels": metadata["num_channels"],
                "source_relpath": str(audio_path.relative_to(split_path)),
            }
        )
    return rows, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized manifests from extracted source datasets.")
    parser.add_argument("--config", default="configs/sources.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log(f"[build_manifest] loading config: {args.config}")
    sources_config = load_yaml(args.config)
    manifests_root = resolve_path("data/manifests")
    speech_exts = set(sources_config["spot_check"]["allowed_speech_extensions"])
    noise_exts = set(sources_config["spot_check"]["allowed_noise_extensions"])
    written: list[Path] = []

    for source_name, source_cfg in sources_config["sources"].items():
        log(f"[build_manifest] source: {source_name}")
        for split_name, split_rel in source_cfg["splits"].items():
            split_path = resolve_split_path(source_cfg, split_rel)
            log(f"[build_manifest]   split={split_name} path={split_path}")
            if not split_path.exists():
                log(f"[build_manifest]   split={split_name} missing, skip")
                continue
            if source_cfg["kind"] == "speech":
                rows, skipped = build_speech_manifest(source_name, split_name, split_path, source_cfg["language"], speech_exts)
                fieldnames = [
                    "dataset",
                    "split",
                    "speaker_id",
                    "utterance_id",
                    "audio_path",
                    "duration_sec",
                    "sample_rate",
                    "num_channels",
                    "language",
                    "source_relpath",
                ]
            else:
                rows, skipped = build_noise_manifest(source_name, split_name, split_path, noise_exts)
                fieldnames = [
                    "dataset",
                    "split",
                    "noise_id",
                    "audio_path",
                    "duration_sec",
                    "sample_rate",
                    "num_channels",
                    "source_relpath",
                ]
            out_path = manifests_root / f"{source_name}_{split_name}.csv"
            write_csv(out_path, fieldnames, rows)
            written.append(out_path)
            log(f"[build_manifest]   wrote {out_path} ({len(rows)} rows, skipped {skipped} unreadable)")

    log(f"[build_manifest] finished; wrote {len(written)} manifest files")


if __name__ == "__main__":
    main()
