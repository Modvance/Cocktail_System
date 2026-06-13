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
from src.metrics.separation_metrics import batch_sisdri
from src.models.tse_fam import TSEFAM
from src.utils.config import load_yaml, select_device
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit a small batch for TSE-FAM debugging.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config["seed"]))
    device = select_device(config.get("device"))
    model = TSEFAM(config).to(device)
    data_cfg = config["data"]
    train_cfg = config["train"]
    loss_cfg = config["loss"]
    loader = create_dataloader(
        csv_path=data_cfg["train_csv"],
        sample_rate=data_cfg["sample_rate"],
        mixture_duration=data_cfg["mixture_duration"],
        enrollment_duration=data_cfg["enrollment_duration"],
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=False,
        training=True,
        gain_augment_db=train_cfg.get("gain_augment_db", 0.0),
        limit=args.num_samples,
    )
    batch = next(iter(loader))
    mixture = batch["mixture"].to(device)
    target = batch["target"].to(device)
    enrollment = batch["enrollment"].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    sisdr_loss = SISDRLoss()
    mag_loss = SpectralMagnitudeLoss(model.feature_extractor)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(mixture, enrollment)
        loss_sisdr = sisdr_loss(outputs["estimated_waveform"], target)
        loss_mag = mag_loss(outputs["estimated_waveform"], target)
        loss = loss_cfg["sisdr_weight"] * loss_sisdr + loss_cfg["mag_weight"] * loss_mag
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            with torch.no_grad():
                sisdri = batch_sisdri(outputs["estimated_waveform"], mixture, target).mean().item()
            print({"step": step, "loss": float(loss.item()), "si_sdri": sisdri})


if __name__ == "__main__":
    main()
