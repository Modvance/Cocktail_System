from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

import torch

from src.data.tse_dataset import create_dataloader
from src.losses.sisdr import SISDRLoss
from src.losses.spectral_loss import SpectralMagnitudeLoss
from src.models.tse_fam import TSEFAM
from src.utils.config import load_yaml, select_device
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a forward/backward sanity check for TSE-FAM.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config["seed"]))
    device = select_device(config.get("device"))
    model = TSEFAM(config).to(device)
    data_cfg = config["data"]
    train_cfg = config["train"]
    loader = create_dataloader(
        csv_path=data_cfg["train_csv"],
        sample_rate=data_cfg["sample_rate"],
        mixture_duration=data_cfg["mixture_duration"],
        enrollment_duration=data_cfg["enrollment_duration"],
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=False,
        training=False,
        limit=2,
    )
    batch = next(iter(loader))
    mixture = batch["mixture"].to(device)
    target = batch["target"].to(device)
    enrollment = batch["enrollment"].to(device)
    outputs = model(mixture, enrollment)
    sisdr_loss = SISDRLoss()
    mag_loss = SpectralMagnitudeLoss(model.feature_extractor)
    loss = sisdr_loss(outputs["estimated_waveform"], target) + mag_loss(outputs["estimated_waveform"], target)
    if not torch.isfinite(loss):
        raise RuntimeError("Sanity check failed: loss is not finite")
    loss.backward()
    mask = outputs["mask"]
    if outputs["estimated_waveform"].shape != target.shape:
        raise RuntimeError(f"Waveform shape mismatch: {outputs['estimated_waveform'].shape} vs {target.shape}")
    if mask.dim() != 3:
        raise RuntimeError(f"Mask dim mismatch: {mask.shape}")
    print({
        "device": str(device),
        "estimated_waveform_shape": tuple(outputs["estimated_waveform"].shape),
        "mask_shape": tuple(mask.shape),
        "loss": float(loss.item()),
    })


if __name__ == "__main__":
    main()
