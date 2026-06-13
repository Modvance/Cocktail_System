from __future__ import annotations

import argparse
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


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
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


def read_audio_header(path: Path) -> dict[str, Any]:
    if sf is not None:
        info = sf.info(str(path))
        return {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration_sec": info.duration,
        }

    suffix = path.suffix.lower()
    if suffix == ".wav":
        with wave.open(str(path), "rb") as wf:
            return {
                "sample_rate": wf.getframerate(),
                "channels": wf.getnchannels(),
                "frames": wf.getnframes(),
                "duration_sec": wf.getnframes() / max(wf.getframerate(), 1),
            }
    if suffix == ".flac" and shutil.which("soxi"):
        import subprocess

        output = subprocess.run(["soxi", str(path)], check=True, capture_output=True, text=True).stdout
        info: dict[str, Any] = {"sample_rate": None, "channels": None, "duration_sec": None, "frames": None}
        for line in output.splitlines():
            if "Sample Rate" in line:
                info["sample_rate"] = int(line.split(":", 1)[1].strip())
            elif "Channels" in line:
                info["channels"] = int(line.split(":", 1)[1].strip())
            elif "Duration" in line:
                raw = line.split(":", 1)[1].strip().split("=")[0].strip()
                h, m, s = raw.split(":")
                info["duration_sec"] = int(h) * 3600 + int(m) * 60 + float(s)
        return info
    return {"sample_rate": None, "channels": None, "frames": None, "duration_sec": None}


def build_report(config: dict[str, Any]) -> str:
    report_lines = ["# Source Dataset Report", ""]
    speech_exts = set(config["spot_check"]["allowed_speech_extensions"])
    noise_exts = set(config["spot_check"]["allowed_noise_extensions"])
    files_per_split = int(config["spot_check"].get("files_per_split", 3))

    for source_name, source_cfg in config["sources"].items():
        log(f"[check_sources] scanning source: {source_name}")
        report_lines.append(f"## {source_name}")
        report_lines.append("")
        extracted_root = resolve_path(source_cfg["extracted_root"])
        report_lines.append(f"- extracted_root: `{extracted_root}`")
        report_lines.append(f"- exists: `{extracted_root.exists()}`")
        report_lines.append("")

        allowed = speech_exts if source_cfg["kind"] == "speech" else noise_exts
        for split_name, split_rel in source_cfg.get("splits", {}).items():
            split_path = resolve_split_path(source_cfg, split_rel)
            log(f"[check_sources]   split={split_name} path={split_path}")
            report_lines.append(f"### split `{split_name}`")
            report_lines.append(f"- path: `{split_path}`")
            report_lines.append(f"- exists: `{split_path.exists()}`")
            if split_path.exists():
                audio_files = list_audio_files(split_path, allowed)
                log(f"[check_sources]   split={split_name} audio_count={len(audio_files)}")
                report_lines.append(f"- audio_count: `{len(audio_files)}`")
                valid_samples = 0
                invalid_samples: list[str] = []
                for sample_path in audio_files:
                    try:
                        header = read_audio_header(sample_path)
                    except Exception as exc:
                        invalid_samples.append(f"{sample_path.relative_to(REPO_ROOT)} ({exc})")
                        continue
                    if valid_samples < files_per_split:
                        report_lines.append(
                            "- sample: `{}` | sr={} | ch={} | dur={}".format(
                                sample_path.relative_to(REPO_ROOT),
                                header.get("sample_rate"),
                                header.get("channels"),
                                header.get("duration_sec"),
                            )
                        )
                    valid_samples += 1
                report_lines.append(f"- readable_audio_count: `{valid_samples}`")
                report_lines.append(f"- unreadable_audio_count: `{len(invalid_samples)}`")
                for bad_path in invalid_samples[:3]:
                    report_lines.append(f"- unreadable_example: `{bad_path}`")
            report_lines.append("")
    return "\n".join(report_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check extracted source datasets and write a report.")
    parser.add_argument("--config", default="configs/sources.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log(f"[check_sources] loading config: {args.config}")
    config = load_config(args.config)
    report = build_report(config)
    report_path = resolve_path(config["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    log(f"[check_sources] wrote source report to {report_path}")


if __name__ == "__main__":
    main()
