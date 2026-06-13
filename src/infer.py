from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import torch

from src.models.tse_fam import TSEFAM
from src.utils.audio import load_audio, write_audio
from src.utils.checkpoint import load_checkpoint
from src.utils.config import resolve_path, select_device
from src.visualize import save_case_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TSE-FAM inference.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--enrollment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device")
    parser.add_argument("--save_fig", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(resolve_path(args.ckpt), map_location="cpu", weights_only=False)
    config = payload["config"]
    device = select_device(args.device or config.get("device"))
    model = TSEFAM(config).to(device)
    load_checkpoint(args.ckpt, model=model, map_location=device)
    model.eval()
    sample_rate = int(config["data"]["sample_rate"])
    mixture = load_audio(args.mixture, sample_rate)
    enrollment = load_audio(args.enrollment, sample_rate)
    mixture_tensor = torch.from_numpy(mixture).float().unsqueeze(0).to(device)
    enrollment_tensor = torch.from_numpy(enrollment).float().unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(mixture_tensor, enrollment_tensor)
    estimate = outputs["estimated_waveform"].squeeze(0).cpu().numpy()
    write_audio(args.out, estimate, sample_rate)
    figures = []
    if args.save_fig:
        figure_dir = resolve_path(args.out).parent / f"{resolve_path(args.out).stem}_figures"
        figures = [str(path) for path in save_case_visualizations(
            output_dir=figure_dir,
            mixture=mixture,
            enrollment=enrollment,
            estimate=estimate,
            mask=outputs["mask"].squeeze(0).detach().cpu().numpy(),
            attention=None if outputs["attention"] is None else outputs["attention"].squeeze(0).detach().cpu().numpy(),
        )]
    print({"output": str(resolve_path(args.out)), "sample_rate": sample_rate, "num_samples": int(estimate.shape[0]), "figures": figures})


if __name__ == "__main__":
    main()
