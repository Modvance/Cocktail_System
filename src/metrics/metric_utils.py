from __future__ import annotations

from collections import defaultdict
from typing import Any


METRIC_FIELDS = [
    "si_sdr_in",
    "si_sdr_out",
    "si_sdri",
    "sdr_in",
    "sdr_out",
    "sdri",
    "sir",
    "sar",
]


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def summarize_metrics(rows: list[dict[str, Any]], metric_fields: list[str] | None = None) -> dict[str, float]:
    fields = metric_fields or METRIC_FIELDS
    return {key: mean_metric(rows, key) for key in fields}


def group_rows(rows: list[dict[str, Any]], key_name: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[format_group_value(key_name, row[key_name])].append(row)
    return dict(groups)


def format_group_value(key_name: str, value: Any) -> str:
    if key_name == "snr_db":
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if numeric != numeric:
            return "clean"
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:g}"
    return str(value)


def build_group_rows(groups: dict[str, list[dict[str, Any]]], key_name: str, metric_fields: list[str] | None = None) -> list[dict[str, Any]]:
    fields = metric_fields or METRIC_FIELDS
    rows = []
    for key in sorted(groups):
        items = groups[key]
        row: dict[str, Any] = {key_name: key, "count": len(items)}
        for metric_name in fields:
            row[metric_name] = mean_metric(items, metric_name)
        rows.append(row)
    return rows
