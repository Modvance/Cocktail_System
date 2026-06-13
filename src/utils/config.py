from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def select_device(requested: str | None) -> torch.device:
    if requested and requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested and requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
