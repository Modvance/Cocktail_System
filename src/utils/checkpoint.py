from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.utils.config import resolve_path, save_yaml


def save_checkpoint(
    path: str | Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
            "config": config,
        },
        resolved,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(resolve_path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload


def save_config_copy(save_dir: str | Path, config: dict[str, Any]) -> None:
    resolved = resolve_path(save_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    save_yaml(resolved / "config_used.yaml", config)
