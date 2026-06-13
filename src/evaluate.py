from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from typing import Any

import torch

from src.data.tse_dataset import create_dataloader
from src.metrics.metric_utils import METRIC_FIELDS, build_group_rows, group_rows, summarize_metrics
from src.metrics.separation_metrics import batch_sar, batch_sdr, batch_sdri, batch_sir, batch_sisdr, batch_sisdri
from src.models.tse_fam import TSEFAM
from src.utils.audio import write_audio
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_yaml, resolve_path, select_device
from src.utils.logger import write_csv, write_json, write_markdown
from src.visualize import save_case_visualizations, save_metric_bar_chart

PER_SAMPLE_FIELDS = [
    "sample_id",
    "lang",
    "num_speakers",
    "snr_db",
    "tir_db",
    "overlap_mode",
    "si_sdr_in",
    "si_sdr_out",
    "si_sdri",
    "sdr_in",
    "sdr_out",
    "sdri",
    "sir",
    "sar",
]

GROUP_KEYS = [
    ("num_speakers", "group_by_num_speakers.csv"),
    ("snr_db", "group_by_snr.csv"),
    ("lang", "group_by_lang.csv"),
    ("overlap_mode", "group_by_overlap.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TSE-FAM on a test CSV.")
    parser.add_argument("--config")
    parser.add_argument("--ckpt")
    parser.add_argument("--test_csv")
    parser.add_argument("--out_dir")
    parser.add_argument("--eval_config")
    parser.add_argument("--device")
    parser.add_argument("--num_examples", type=int, default=3, help="number of examples to export for each category")
    args = parser.parse_args()
    if args.eval_config:
        return args
    required = ["config", "ckpt", "test_csv", "out_dir"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join('--' + name for name in missing)}")
    return args


def save_example_bundle(out_dir: Path, sample_id: str, batch_index: int, batch: dict[str, Any], outputs: dict[str, Any], category: str | None = None) -> None:
    example_dir = out_dir / "audio_examples"
    if category:
        example_dir = example_dir / category
    example_dir = example_dir / sample_id
    example_dir.mkdir(parents=True, exist_ok=True)
    mixture = batch["mixture"][batch_index].detach().cpu().numpy()
    target = batch["target"][batch_index].detach().cpu().numpy()
    enrollment = batch["enrollment"][batch_index].detach().cpu().numpy()
    estimate = outputs["estimated_waveform"][batch_index].detach().cpu().numpy()
    sample_rate = int(outputs["sample_rate"])
    write_audio(example_dir / "mixture.wav", mixture, sample_rate)
    write_audio(example_dir / "target_clean.wav", target, sample_rate)
    write_audio(example_dir / "enrollment.wav", enrollment, sample_rate)
    write_audio(example_dir / "estimated_target.wav", estimate, sample_rate)
    attention = outputs["attention"]
    mask = outputs["mask"][batch_index].detach().cpu().numpy()
    attention_np = None if attention is None else attention[batch_index].detach().cpu().numpy()
    save_case_visualizations(
        output_dir=example_dir,
        mixture=mixture,
        enrollment=enrollment,
        estimate=estimate,
        target=target,
        mask=mask,
        attention=attention_np,
    )


def select_example_indices(rows: list[dict[str, Any]], num_examples: int) -> list[tuple[str, int]]:
    if not rows or num_examples <= 0:
        return []
    ranked_indices = sorted(range(len(rows)), key=lambda idx: (rows[idx]["si_sdri"], rows[idx]["sample_id"]))
    ranked_positions = {idx: pos for pos, idx in enumerate(ranked_indices)}
    median_idx = ranked_indices[len(ranked_indices) // 2]
    median_value = rows[median_idx]["si_sdri"]
    selected: list[tuple[str, int]] = []

    def take(category: str, candidates: list[int]) -> None:
        taken = 0
        for idx in candidates:
            selected.append((category, idx))
            taken += 1
            if taken >= num_examples:
                break

    take("worst", ranked_indices)
    take("best", list(reversed(ranked_indices)))
    median_candidates = sorted(
        ranked_indices,
        key=lambda idx: (
            abs(ranked_positions[idx] - ranked_positions[median_idx]),
            abs(rows[idx]["si_sdri"] - median_value),
            rows[idx]["sample_id"],
        ),
    )
    take("median", median_candidates)

    order = {"best": 0, "median": 1, "worst": 2}
    category_sort = {
        "best": lambda idx: (-rows[idx]["si_sdri"], rows[idx]["sample_id"]),
        "median": lambda idx: (abs(rows[idx]["si_sdri"] - median_value), rows[idx]["sample_id"]),
        "worst": lambda idx: (rows[idx]["si_sdri"], rows[idx]["sample_id"]),
    }
    return sorted(selected, key=lambda item: (order[item[0]], *category_sort[item[0]](item[1])))


def write_selected_examples_markdown(out_dir: Path, selected_examples: list[dict[str, Any]]) -> None:
    lines = ["# Selected Audio Examples", "", "Examples are chosen by `si_sdri`.", ""]
    grouped: dict[str, list[dict[str, Any]]] = {"best": [], "median": [], "worst": []}
    for example in selected_examples:
        grouped.setdefault(example["category"], []).append(example)

    for category in ["best", "median", "worst"]:
        examples = grouped.get(category, [])
        if not examples:
            continue
        lines.append(f"## {category}")
        for example in examples:
            row = example["row"]
            lines.extend([
                f"- sample_id: {row['sample_id']}, si_sdri: {row['si_sdri']:.4f}, si_sdr_out: {row['si_sdr_out']:.4f}, snr_db: {row['snr_db']}, num_speakers: {row['num_speakers']}, lang: {row['lang']}, overlap_mode: {row['overlap_mode']}",
            ])
        lines.append("")
    write_markdown(out_dir / "audio_examples" / "selected_examples.md", "\n".join(lines))


def write_summary_markdown(out_dir: Path, summary: dict[str, float]) -> None:
    lines = ["# Evaluation Summary", f"- sample_count: {int(summary['sample_count'])}"]
    for key in METRIC_FIELDS:
        if key in summary:
            lines.append(f"- {key}: {summary[key]:.4f}")
    write_markdown(out_dir / "metrics_summary.md", "\n".join(lines))


def write_group_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    for key_name, file_name in GROUP_KEYS:
        groups = group_rows(rows, key_name)
        group_rows_payload = build_group_rows(groups, key_name, metric_fields=["si_sdri", "sdri", "sir", "sar"])
        write_csv(out_dir / file_name, [key_name, "count", "si_sdri", "sdri", "sir", "sar"], group_rows_payload)


def write_figures(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_metric_bar_chart(
        figures_dir / "speaker_count_results.png",
        build_group_rows(group_rows(rows, "num_speakers"), "num_speakers", metric_fields=["si_sdri"]),
        label_key="num_speakers",
        value_key="si_sdri",
        title="SI-SDRi by speaker count",
    )
    save_metric_bar_chart(
        figures_dir / "snr_results.png",
        build_group_rows(group_rows(rows, "snr_db"), "snr_db", metric_fields=["si_sdri"]),
        label_key="snr_db",
        value_key="si_sdri",
        title="SI-SDRi by SNR",
    )
    save_metric_bar_chart(
        figures_dir / "language_results.png",
        build_group_rows(group_rows(rows, "lang"), "lang", metric_fields=["si_sdri"]),
        label_key="lang",
        value_key="si_sdri",
        title="SI-SDRi by language",
    )
    save_metric_bar_chart(
        figures_dir / "overlap_results.png",
        build_group_rows(group_rows(rows, "overlap_mode"), "overlap_mode", metric_fields=["si_sdri"]),
        label_key="overlap_mode",
        value_key="si_sdri",
        title="SI-SDRi by overlap mode",
    )


def evaluate_run(run: dict[str, Any], device_override: str | None = None, num_examples_override: int | None = None) -> dict[str, float]:
    config = load_yaml(run["config"])
    device = select_device(device_override or config.get("device"))
    data_cfg = config["data"]
    train_cfg = config["train"]
    model = TSEFAM(config).to(device)
    load_checkpoint(run["ckpt"], model=model, map_location=device)
    model.eval()

    loader = create_dataloader(
        csv_path=run["test_csv"],
        sample_rate=data_cfg["sample_rate"],
        mixture_duration=data_cfg["mixture_duration"],
        enrollment_duration=data_cfg["enrollment_duration"],
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=False,
        training=False,
    )

    rows: list[dict[str, Any]] = []
    example_payloads: dict[str, dict[str, Any]] = {}
    out_dir = resolve_path(run["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    num_examples = int(run.get("num_examples", 3) if num_examples_override is None else num_examples_override)

    with torch.no_grad():
        for batch in loader:
            mixture = batch["mixture"].to(device)
            target = batch["target"].to(device)
            enrollment = batch["enrollment"].to(device)
            outputs = model(mixture, enrollment)
            outputs["sample_rate"] = int(data_cfg["sample_rate"])
            si_sdr_in = batch_sisdr(mixture, target).cpu()
            si_sdr_out = batch_sisdr(outputs["estimated_waveform"], target).cpu()
            si_sdri = batch_sisdri(outputs["estimated_waveform"], mixture, target).cpu()
            sdr_in = batch_sdr(mixture, target).cpu()
            sdr_out = batch_sdr(outputs["estimated_waveform"], target).cpu()
            sdri = batch_sdri(outputs["estimated_waveform"], mixture, target).cpu()
            sir = batch_sir(outputs["estimated_waveform"], mixture, target).cpu()
            sar = batch_sar(outputs["estimated_waveform"], target).cpu()
            batch_size = mixture.size(0)

            mixture_cpu = batch["mixture"].detach().cpu()
            target_cpu = batch["target"].detach().cpu()
            enrollment_cpu = batch["enrollment"].detach().cpu()
            estimated_cpu = outputs["estimated_waveform"].detach().cpu()
            mask_cpu = outputs["mask"].detach().cpu()
            attention = outputs["attention"]
            attention_cpu = None if attention is None else attention.detach().cpu()

            for idx in range(batch_size):
                snr_value = float(batch["snr_db"][idx])
                sample_id = batch["sample_id"][idx]
                row = {
                    "sample_id": sample_id,
                    "lang": batch["lang"][idx],
                    "num_speakers": int(batch["num_speakers"][idx]),
                    "snr_db": "clean" if snr_value != snr_value else snr_value,
                    "tir_db": float(batch["tir_db"][idx]),
                    "overlap_mode": batch["overlap_mode"][idx],
                    "si_sdr_in": float(si_sdr_in[idx].item()),
                    "si_sdr_out": float(si_sdr_out[idx].item()),
                    "si_sdri": float(si_sdri[idx].item()),
                    "sdr_in": float(sdr_in[idx].item()),
                    "sdr_out": float(sdr_out[idx].item()),
                    "sdri": float(sdri[idx].item()),
                    "sir": float(sir[idx].item()),
                    "sar": float(sar[idx].item()),
                }
                rows.append(row)
                example_payloads[sample_id] = {
                    "mixture": mixture_cpu[idx].numpy(),
                    "target": target_cpu[idx].numpy(),
                    "enrollment": enrollment_cpu[idx].numpy(),
                    "estimate": estimated_cpu[idx].numpy(),
                    "mask": mask_cpu[idx].numpy(),
                    "attention": None if attention_cpu is None else attention_cpu[idx].numpy(),
                    "sample_rate": int(data_cfg["sample_rate"]),
                }

    selected_examples = []
    for category, row_idx in select_example_indices(rows, num_examples):
        row = rows[row_idx]
        payload = example_payloads[row["sample_id"]]
        example_dir = out_dir / "audio_examples" / category / row["sample_id"]
        example_dir.mkdir(parents=True, exist_ok=True)
        write_audio(example_dir / "mixture.wav", payload["mixture"], payload["sample_rate"])
        write_audio(example_dir / "target_clean.wav", payload["target"], payload["sample_rate"])
        write_audio(example_dir / "enrollment.wav", payload["enrollment"], payload["sample_rate"])
        write_audio(example_dir / "estimated_target.wav", payload["estimate"], payload["sample_rate"])
        save_case_visualizations(
            output_dir=example_dir,
            mixture=payload["mixture"],
            enrollment=payload["enrollment"],
            estimate=payload["estimate"],
            target=payload["target"],
            mask=payload["mask"],
            attention=payload["attention"],
        )
        selected_examples.append({"category": category, "row": row})

    if selected_examples:
        write_selected_examples_markdown(out_dir, selected_examples)

    write_csv(out_dir / "metrics_per_sample.csv", PER_SAMPLE_FIELDS, rows)
    summary = {"sample_count": len(rows), **summarize_metrics(rows)}
    write_json(out_dir / "metrics_summary.json", summary)
    write_summary_markdown(out_dir, summary)
    write_group_outputs(out_dir, rows)
    write_figures(out_dir, rows)
    return summary


def resolve_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.eval_config:
        return [{
            "config": args.config,
            "ckpt": args.ckpt,
            "test_csv": args.test_csv,
            "out_dir": args.out_dir,
            "num_examples": args.num_examples,
        }]

    eval_cfg = load_yaml(args.eval_config)
    if "runs" in eval_cfg:
        runs = []
        for run in eval_cfg["runs"]:
            runs.append({
                "config": run["config"],
                "ckpt": run["ckpt"],
                "test_csv": run["test_csv"],
                "out_dir": run["out_dir"],
                "num_examples": run.get("num_examples", eval_cfg.get("num_examples", args.num_examples)),
            })
        return runs

    return [{
        "config": args.config or eval_cfg["config"],
        "ckpt": args.ckpt or eval_cfg["ckpt"],
        "test_csv": args.test_csv or eval_cfg["test_csv"],
        "out_dir": args.out_dir or eval_cfg["out_dir"],
        "num_examples": eval_cfg.get("num_examples", args.num_examples),
    }]


def main() -> None:
    args = parse_args()
    runs = resolve_runs(args)
    results = []
    for run in runs:
        summary = evaluate_run(run, device_override=args.device, num_examples_override=run.get("num_examples"))
        results.append({"out_dir": str(resolve_path(run["out_dir"])), **summary})
    if len(results) == 1:
        print(results[0])
    else:
        print({"runs": results})


if __name__ == "__main__":
    main()
