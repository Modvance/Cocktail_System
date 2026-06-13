from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PROGRESS_BAR_WIDTH = 28


def log(message: str) -> None:
    print(message, flush=True)


def render_progress(prefix: str, current: int, total: int) -> None:
    total = max(total, 1)
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(PROGRESS_BAR_WIDTH * ratio)
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} ({ratio * 100:5.1f}%)", end="", flush=True)
    if current >= total:
        print(flush=True)


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


def pick_downloader(preferred: list[str]) -> str | None:
    for name in preferred:
        if shutil.which(name):
            return name
    return None


def split_path_candidates(source_cfg: dict[str, Any], split_rel: str) -> list[Path]:
    root = resolve_path(source_cfg["extracted_root"])
    rel_path = Path(split_rel)
    candidates = [root / rel_path, root / root.name / rel_path, root / rel_path.name]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def split_exists(source_cfg: dict[str, Any], split_rel: str) -> bool:
    return any(path.exists() for path in split_path_candidates(source_cfg, split_rel))


def outer_extraction_complete(source_cfg: dict[str, Any]) -> bool:
    return all(split_exists(source_cfg, split_rel) for split_rel in source_cfg.get("splits", {}).values())


def nested_archive_target_dir(archive_path: Path, extracted_root: Path) -> Path | None:
    with tarfile.open(archive_path, "r:*" ) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) >= 2:
                return extracted_root / parts[0] / parts[1]
            if len(parts) == 1:
                return extracted_root / parts[0]
    return None


def nested_archives_complete(source_cfg: dict[str, Any]) -> bool:
    nested_cfg = source_cfg.get("nested_archives")
    if not nested_cfg:
        return True
    extracted_root = resolve_path(source_cfg["extracted_root"])
    nested_root = extracted_root / nested_cfg["root"]
    if not nested_root.exists():
        return False
    archives = sorted(nested_root.glob(nested_cfg["pattern"]))
    if not archives:
        return False
    for archive_path in archives:
        target_dir = nested_archive_target_dir(archive_path, extracted_root)
        if target_dir is None or not target_dir.exists():
            return False
    return True


def source_complete(source_cfg: dict[str, Any]) -> bool:
    return outer_extraction_complete(source_cfg) and nested_archives_complete(source_cfg)


def repair_known_layouts(source_name: str, extracted_root: Path) -> str | None:
    if source_name == "librispeech":
        message = flatten_nested_root(extracted_root, "LibriSpeech")
    elif source_name == "wham_noise":
        message = flatten_nested_root(extracted_root, "wham_noise")
    else:
        return None
    if message.startswith("no nested "):
        return None
    return message


def ensure_archive(source_name: str, archive: dict[str, Any], downloads_root: Path, downloader: str | None) -> tuple[Path | None, str]:
    manual_archive = archive.get("manual_archive")
    if manual_archive:
        manual_path = resolve_path(manual_archive)
        if manual_path.exists():
            return manual_path, f"using manual archive for {source_name}"

    archive_path = downloads_root / archive["filename"]
    if archive_path.exists() and archive_path.stat().st_size > 0:
        return archive_path, f"archive already exists: {archive_path.name}"

    urls = archive.get("urls") or []
    if not urls:
        return None, f"missing archive {archive['filename']} and no download URL configured"
    if downloader is None:
        return None, f"missing archive {archive['filename']} and no downloader available"

    url = urls[0]
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if downloader == "aria2c":
        command = [
            "aria2c",
            "--continue=true",
            "--dir",
            str(archive_path.parent),
            "--out",
            archive_path.name,
            url,
        ]
    else:
        command = ["wget", "-c", "-O", str(archive_path), url]

    log(f"[{source_name}] downloading {archive['filename']} with {downloader} ...")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        return None, f"download failed for {archive['filename']}: {exc}"
    return archive_path, f"downloaded {archive_path.name}"


def safe_extract_tar(archive_path: Path, destination: Path, prefix: str) -> None:
    with tarfile.open(archive_path, "r:*" ) as tar:
        members = tar.getmembers()
        total = len(members)
        render_progress(prefix, 0, total)
        for index, member in enumerate(members, start=1):
            try:
                tar.extract(member, destination, filter="data")
            except TypeError:
                tar.extract(member, destination)
            if index == total or index == 1 or index % 500 == 0:
                render_progress(prefix, index, total)


def safe_extract_zip(archive_path: Path, destination: Path, prefix: str) -> None:
    with zipfile.ZipFile(archive_path) as zf:
        members = zf.infolist()
        total = len(members)
        render_progress(prefix, 0, total)
        for index, member in enumerate(members, start=1):
            zf.extract(member, destination)
            if index == total or index == 1 or index % 500 == 0:
                render_progress(prefix, index, total)


def flatten_nested_root(extracted_root: Path, nested_name: str) -> str:
    nested_root = extracted_root / nested_name
    if not nested_root.exists() or not nested_root.is_dir():
        return f"no nested {nested_name} directory detected"

    moved = 0
    skipped = 0
    for child in list(nested_root.iterdir()):
        target = extracted_root / child.name
        if target.exists():
            skipped += 1
            continue
        shutil.move(str(child), str(target))
        moved += 1

    remaining = list(nested_root.iterdir())
    if not remaining:
        nested_root.rmdir()
    return f"flattened nested {nested_name} directory; moved {moved} entries, skipped {skipped} existing entries"


def extract_archive(source_name: str, archive_path: Path, destination: Path) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive_path.suffixes)
    prefix = f"[{source_name}] extracting {archive_path.name}"
    log(f"{prefix} ...")
    if suffixes.endswith((".tar.gz", ".tgz", ".tar")):
        safe_extract_tar(archive_path, destination, prefix)
        if source_name == "librispeech":
            flatten_message = flatten_nested_root(destination, "LibriSpeech")
            return f"extracted {archive_path.name} to {destination}; {flatten_message}"
        return f"extracted {archive_path.name} to {destination}"
    if suffixes.endswith(".zip"):
        safe_extract_zip(archive_path, destination, prefix)
        if source_name == "wham_noise":
            flatten_message = flatten_nested_root(destination, "wham_noise")
            return f"extracted {archive_path.name} to {destination}; {flatten_message}"
        return f"extracted {archive_path.name} to {destination}"
    return f"skipped extraction for unsupported archive type: {archive_path.name}"


def extract_nested_archives(source_name: str, source_cfg: dict[str, Any]) -> list[str]:
    nested_cfg = source_cfg.get("nested_archives")
    if not nested_cfg:
        return []

    extracted_root = resolve_path(source_cfg["extracted_root"])
    nested_root = extracted_root / nested_cfg["root"]
    if not nested_root.exists():
        return [f"[{source_name}] nested archive root missing: {nested_root}"]

    archives = sorted(nested_root.glob(nested_cfg["pattern"]))
    summaries: list[str] = []
    total = len(archives)
    log(f"[{source_name}] found {total} nested archives under {nested_root}")
    for index, archive_path in enumerate(archives, start=1):
        try:
            target_dir = nested_archive_target_dir(archive_path, extracted_root)
            if target_dir is not None and target_dir.exists():
                log(f"[{source_name}] nested {index}/{total}: skip {archive_path.name} -> {target_dir.relative_to(extracted_root)}")
                summaries.append(f"[{source_name}] nested extraction skipped: {target_dir.relative_to(extracted_root)}")
                continue
            log(f"[{source_name}] nested {index}/{total}: extracting {archive_path.name} ...")
            safe_extract_tar(archive_path, extracted_root, f"[{source_name}] nested {index}/{total} {archive_path.name}")
            summaries.append(f"[{source_name}] extracted nested archive {archive_path.name}")
        except Exception as exc:
            summaries.append(f"[{source_name}] nested extraction failed for {archive_path.name}: {exc}")
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract source datasets for TSE construction.")
    parser.add_argument("--config", default="configs/sources.yaml")
    parser.add_argument("--dataset", action="append", help="Limit to one or more sources (librispeech, aishell1, wham_noise)")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    downloads_root = resolve_path(config["downloads_root"])
    downloader = pick_downloader(config.get("preferred_downloaders", []))
    selected = set(args.dataset or config["sources"].keys())

    summaries: list[str] = []
    log(f"Preparing sources: {', '.join(sorted(selected))}")
    for source_name, source_cfg in config["sources"].items():
        if source_name not in selected:
            continue

        log(f"=== [{source_name}] start ===")
        extracted_root = resolve_path(source_cfg["extracted_root"])
        repair_message = repair_known_layouts(source_name, extracted_root)
        if repair_message is not None:
            log(f"[{source_name}] {repair_message}")
            summaries.append(f"[{source_name}] {repair_message}")
        if source_complete(source_cfg) and not args.force_extract and not args.download_only:
            message = f"[{source_name}] source already prepared; skipping download and extraction"
            log(message)
            summaries.append(message)
            continue

        archives = source_cfg.get("archives", [])
        total_archives = len(archives)
        for index, archive in enumerate(archives, start=1):
            archive_cfg = {**archive}
            if source_cfg.get("manual_archive"):
                archive_cfg["manual_archive"] = source_cfg["manual_archive"]

            archive_path = downloads_root / archive_cfg["filename"]
            log(f"[{source_name}] archive {index}/{total_archives}: {archive_cfg['filename']}")
            if not args.extract_only:
                if args.skip_download and not archive_path.exists() and not archive_cfg.get("manual_archive"):
                    message = f"[{source_name}] skipped download but archive missing: {archive_cfg['filename']}"
                    log(message)
                    summaries.append(message)
                    continue
                archive_path_result, message = ensure_archive(source_name, archive_cfg, downloads_root, downloader)
                log(f"[{source_name}] {message}")
                summaries.append(f"[{source_name}] {message}")
                if archive_path_result is None:
                    continue
                archive_path = archive_path_result
            else:
                if archive_cfg.get("manual_archive"):
                    archive_path = resolve_path(archive_cfg["manual_archive"])
                elif not archive_path.exists():
                    message = f"[{source_name}] cannot extract missing archive: {archive_cfg['filename']}"
                    log(message)
                    summaries.append(message)
                    continue
                log(f"[{source_name}] extract-only mode for {archive_cfg['filename']}")
                summaries.append(f"[{source_name}] extract-only mode for {archive_cfg['filename']}")

            if args.download_only:
                continue
            try:
                if args.force_extract or not source_complete(source_cfg):
                    message = extract_archive(source_name, archive_path, extracted_root)
                    log(f"[{source_name}] {message}")
                    summaries.append(f"[{source_name}] {message}")
                else:
                    message = f"[{source_name}] extraction skipped; source already prepared"
                    log(message)
                    summaries.append(message)
            except Exception as exc:
                message = f"[{source_name}] extraction failed for {archive_path.name}: {exc}"
                log(message)
                summaries.append(message)

        if not args.download_only:
            nested_summaries = extract_nested_archives(source_name, source_cfg)
            summaries.extend(nested_summaries)
            complete_message = f"[{source_name}] complete={source_complete(source_cfg)}"
            log(complete_message)
            summaries.append(complete_message)
        log(f"=== [{source_name}] done ===")

    log("All selected sources processed.")
    print("\n".join(summaries))


if __name__ == "__main__":
    main()
