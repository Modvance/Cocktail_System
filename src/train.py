from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from typing import Any

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.tse_dataset import create_dataloader
from src.losses.sisdr import SISDRLoss
from src.losses.spectral_loss import SpectralMagnitudeLoss
from src.metrics.separation_metrics import batch_sisdr, batch_sisdri
from src.models.tse_fam import TSEFAM
from src.utils.audio import write_audio
from src.utils.checkpoint import load_checkpoint, save_checkpoint, save_config_copy
from src.utils.config import load_yaml, resolve_path, select_device
from src.utils.logger import append_csv_row
from src.utils.seed import seed_everything
from src.visualize import save_case_visualizations


LOG_FIELDS = [
    "epoch",
    "train_loss",
    "train_sisdr_loss",
    "train_mag_loss",
    "valid_loss",
    "valid_sisdr",
    "valid_sisdr_i",
    "lr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TSE-FAM.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    return parser.parse_args()


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return batch["mixture"].to(device), batch["target"].to(device), batch["enrollment"].to(device)


def run_epoch(
    model: TSEFAM,
    loader,
    optimizer,
    sisdr_loss_fn: SISDRLoss,
    mag_loss_fn: SpectralMagnitudeLoss,
    loss_cfg: dict[str, float],
    grad_clip: float,
    device: torch.device,
    training: bool,
    amp_enabled: bool,
    collect_quality_metrics: bool,
) -> dict[str, float]:
    model.train(training)
    total_loss = 0.0
    total_sisdr_loss = 0.0
    total_mag_loss = 0.0
    total_sisdr = 0.0
    total_sisdri = 0.0
    steps = 0
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled and training)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.set_grad_enabled(training):
        for step_idx, batch in enumerate(loader, start=1):
            mixture, target, enrollment = move_batch_to_device(batch, device)
            with torch.autocast(device_type=autocast_device, enabled=amp_enabled):
                outputs = model(mixture, enrollment)
            sisdr_loss = sisdr_loss_fn(outputs["estimated_waveform"], target)
            mag_loss = mag_loss_fn(outputs["estimated_waveform"], target)
            loss = loss_cfg["sisdr_weight"] * sisdr_loss + loss_cfg["mag_weight"] * mag_loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {step_idx}: "
                    f"sisdr_loss={float(sisdr_loss.detach().cpu())}, "
                    f"mag_loss={float(mag_loss.detach().cpu())}, "
                    f"waveform_finite={bool(torch.isfinite(outputs['estimated_waveform']).all().item())}, "
                    f"mask_finite={bool(torch.isfinite(outputs['mask']).all().item())}"
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if not torch.isfinite(grad_norm):
                        raise RuntimeError(f"Non-finite grad norm at step {step_idx}: {float(grad_norm.detach().cpu())}")
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if not torch.isfinite(grad_norm):
                        raise RuntimeError(f"Non-finite grad norm at step {step_idx}: {float(grad_norm.detach().cpu())}")
                    optimizer.step()
            if collect_quality_metrics:
                with torch.no_grad():
                    sisdr = batch_sisdr(outputs["estimated_waveform"], target).mean().item()
                    sisdri = batch_sisdri(outputs["estimated_waveform"], mixture, target).mean().item()
                total_sisdr += sisdr
                total_sisdri += sisdri
            total_loss += float(loss.item())
            total_sisdr_loss += float(sisdr_loss.item())
            total_mag_loss += float(mag_loss.item())
            steps += 1
    if steps == 0:
        raise RuntimeError("No batches were processed.")
    metrics = {
        "loss": total_loss / steps,
        "sisdr_loss": total_sisdr_loss / steps,
        "mag_loss": total_mag_loss / steps,
    }
    if collect_quality_metrics:
        metrics["sisdr"] = total_sisdr / steps
        metrics["sisdri"] = total_sisdri / steps
    return metrics


def save_validation_examples(model: TSEFAM, config: dict[str, Any], device: torch.device, epoch: int, limit: int = 3) -> None:
    data_cfg = config["data"]
    train_cfg = config["train"]
    save_dir = resolve_path(f"results/validation_samples/{resolve_path(config['checkpoint']['save_dir']).name}/epoch_{epoch:03d}")
    loader = create_dataloader(
        csv_path=data_cfg["valid_csv"],
        sample_rate=data_cfg["sample_rate"],
        mixture_duration=data_cfg["mixture_duration"],
        enrollment_duration=data_cfg["enrollment_duration"],
        batch_size=1,
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=False,
        training=False,
        limit=limit,
        persistent_workers=train_cfg.get("persistent_workers"),
        prefetch_factor=train_cfg.get("prefetch_factor"),
    )
    sample_rate = int(data_cfg["sample_rate"])
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            sample_id = batch["sample_id"][0]
            sample_dir = save_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            mixture = batch["mixture"].to(device)
            target = batch["target"].to(device)
            enrollment = batch["enrollment"].to(device)
            outputs = model(mixture, enrollment)
            mixture_np = mixture.squeeze(0).detach().cpu().numpy()
            target_np = target.squeeze(0).detach().cpu().numpy()
            enrollment_np = enrollment.squeeze(0).detach().cpu().numpy()
            estimate_np = outputs["estimated_waveform"].squeeze(0).detach().cpu().numpy()
            write_audio(sample_dir / f"{sample_id}_mixture.wav", mixture_np, sample_rate)
            write_audio(sample_dir / f"{sample_id}_target.wav", target_np, sample_rate)
            write_audio(sample_dir / f"{sample_id}_enrollment.wav", enrollment_np, sample_rate)
            write_audio(sample_dir / f"{sample_id}_estimated.wav", estimate_np, sample_rate)
            save_case_visualizations(
                output_dir=sample_dir,
                mixture=mixture_np,
                enrollment=enrollment_np,
                estimate=estimate_np,
                target=target_np,
                mask=outputs["mask"].squeeze(0).detach().cpu().numpy(),
                attention=None if outputs["attention"] is None else outputs["attention"].squeeze(0).detach().cpu().numpy(),
            )
    model.train(model_was_training)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config["seed"]))
    device = select_device(config.get("device"))
    data_cfg = config["data"]
    train_cfg = config["train"]
    loss_cfg = config["loss"]
    checkpoint_cfg = config["checkpoint"]

    train_loader = create_dataloader(
        csv_path=data_cfg["train_csv"],
        sample_rate=data_cfg["sample_rate"],
        mixture_duration=data_cfg["mixture_duration"],
        enrollment_duration=data_cfg["enrollment_duration"],
        batch_size=train_cfg["batch_size"],
        shuffle=train_cfg.get("shuffle_train", True),
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=train_cfg.get("drop_last_train", True),
        training=True,
        gain_augment_db=train_cfg.get("gain_augment_db", 0.0),
        persistent_workers=train_cfg.get("persistent_workers"),
        prefetch_factor=train_cfg.get("prefetch_factor"),
    )
    valid_loader = create_dataloader(
        csv_path=data_cfg["valid_csv"],
        sample_rate=data_cfg["sample_rate"],
        mixture_duration=data_cfg["mixture_duration"],
        enrollment_duration=data_cfg["enrollment_duration"],
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=False,
        training=False,
        persistent_workers=train_cfg.get("persistent_workers"),
        prefetch_factor=train_cfg.get("prefetch_factor"),
    )

    model = TSEFAM(config).to(device)
    optimizer = Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    sisdr_loss_fn = SISDRLoss()
    mag_loss_fn = SpectralMagnitudeLoss(model.feature_extractor)

    start_epoch = 1
    best_metric = float("-inf")
    if args.resume:
        payload = load_checkpoint(args.resume, model=model, optimizer=optimizer, scheduler=scheduler, map_location=device)
        start_epoch = int(payload["epoch"]) + 1
        best_metric = float(payload.get("best_metric", best_metric))

    save_dir = checkpoint_cfg["save_dir"]
    save_config_copy(save_dir, config)
    log_path = f"{save_dir}/train_log.csv"

    amp_enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    validate_every = int(train_cfg.get("validate_every", 1))
    save_examples_every = int(train_cfg.get("save_examples_every", 0))
    train_metrics_every = int(train_cfg.get("train_metrics_every", validate_every))

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        should_validate = (epoch % validate_every == 0) or (epoch == train_cfg["epochs"])
        should_collect_train_metrics = (epoch % train_metrics_every == 0) or should_validate
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            sisdr_loss_fn=sisdr_loss_fn,
            mag_loss_fn=mag_loss_fn,
            loss_cfg=loss_cfg,
            grad_clip=train_cfg["grad_clip"],
            device=device,
            training=True,
            amp_enabled=amp_enabled,
            collect_quality_metrics=should_collect_train_metrics,
        )
        valid_metrics = {"loss": float("nan"), "sisdr": float("nan"), "sisdri": float("nan")}
        if should_validate:
            valid_metrics = run_epoch(
                model=model,
                loader=valid_loader,
                optimizer=optimizer,
                sisdr_loss_fn=sisdr_loss_fn,
                mag_loss_fn=mag_loss_fn,
                loss_cfg=loss_cfg,
                grad_clip=train_cfg["grad_clip"],
                device=device,
                training=False,
                amp_enabled=amp_enabled,
                collect_quality_metrics=True,
            )
            scheduler.step(valid_metrics["sisdri"])
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_sisdr_loss": train_metrics["sisdr_loss"],
            "train_mag_loss": train_metrics["mag_loss"],
            "valid_loss": valid_metrics["loss"],
            "valid_sisdr": valid_metrics["sisdr"],
            "valid_sisdr_i": valid_metrics["sisdri"],
            "lr": current_lr,
        }
        append_csv_row(log_path, LOG_FIELDS, row)
        save_checkpoint(f"{save_dir}/last.pt", epoch, model, optimizer, scheduler, best_metric, config)
        is_best = should_validate and valid_metrics["sisdri"] >= best_metric
        if is_best:
            best_metric = valid_metrics["sisdri"]
            save_checkpoint(f"{save_dir}/best.pt", epoch, model, optimizer, scheduler, best_metric, config)
        should_save_examples = False
        if save_examples_every > 0 and epoch % save_examples_every == 0:
            should_save_examples = True
        if is_best and train_cfg.get("save_examples_on_best", True):
            should_save_examples = True
        if should_save_examples:
            save_validation_examples(model, config, device, epoch, limit=min(3, train_cfg["batch_size"] + 1))
        print(row)


if __name__ == "__main__":
    main()
