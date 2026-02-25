#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print per-dataset uncertainty metrics from unc_summary_*.json files."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="runs/unc_summary_*_test*.json",
        help="Glob pattern for summary JSON files.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["dice_at_80", "dice_at_90", "aurc", "auroc", "auprc"],
        help="Metric keys to display.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_metric(metric_name: str, entry: Dict[str, float]) -> str:
    mean_key = f"{metric_name}_mean"
    std_key = f"{metric_name}_std"
    if mean_key in entry and std_key in entry:
        return f"{entry[mean_key]:.4f}±{entry[std_key]:.4f}"
    value = entry.get(metric_name)
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    args = parse_args()
    files = sorted(Path(".").glob(args.pattern))
    if not files:
        print(f"No summary files matched pattern '{args.pattern}'.")
        return
    grouped: Dict[str, Dict[str, Dict[str, float]]] = {}
    for path in files:
        data = load_summary(path)
        for key, metrics in data.items():
            try:
                dataset, split, method = key.split(":")
            except ValueError:
                continue
            grouped.setdefault(f"{dataset}:{split}", {})[method] = metrics
    if not grouped:
        print("No dataset entries found.")
        return
    for dataset_split in sorted(grouped):
        dataset, split = dataset_split.split(":")
        print(f"\n=== {dataset.upper()} [{split}] ===")
        methods = grouped[dataset_split]
        header = ["Method"] + args.metrics + ["Runs", "Samples"]
        row_fmt = "{:<25}" + "  {:>16}" * len(args.metrics) + "  {:>6}  {:>8}"
        print(row_fmt.format(*header))
        for method in sorted(methods):
            entry = methods[method]
            row = [method]
            for metric in args.metrics:
                row.append(format_metric(metric, entry))
            run_count = entry.get("run_count", entry.get("runs", "–"))
            samples = entry.get("num_samples", entry.get("samples", "–"))
            row.append(str(run_count) if run_count is not None else "–")
            row.append(str(samples) if samples is not None else "–")
            print(row_fmt.format(*row))


if __name__ == "__main__":
    main()
