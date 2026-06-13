from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.utils.config import resolve_path


def append_csv_row(path: str | Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    exists = resolved.exists()
    with resolved.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: Any) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: str | Path, content: str) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
