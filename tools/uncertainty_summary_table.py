#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate uncertainty summaries into a mean±std table."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="runs/unc_summary_*_test.json",
        help="Glob pattern for summary JSON files (default: runs/unc_summary_*_test.json).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["dice_at_80", "dice_at_90", "aurc", "auroc", "auprc"],
        help="Metric keys to include in the table.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=[],
        help="Datasets to exclude when aggregating (match dataset name in dataset:split:method).",
    )
    return parser.parse_args()


def load_summary(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_metrics(
    paths: List[Path], metric_keys: List[str], exclude: List[str]
) -> Dict[str, Dict[str, List[float]]]:
    def normalize(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    exclude_set = {normalize(name) for name in exclude}
    bucket: Dict[str, Dict[str, List[float]]] = {}
    for path in paths:
        data = load_summary(path)
        for key, metrics in data.items():
            try:
                dataset, _, method = key.split(":")
            except ValueError:
                continue
            if normalize(dataset) in exclude_set:
                continue
            method_bucket = bucket.setdefault(method, {k: [] for k in metric_keys})
            for metric in metric_keys:
                value = metrics.get(metric)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                method_bucket[metric].append(float(value))
    return bucket


def format_mean_std(values: List[float]) -> str:
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.4f}±0.0000"
    return f"{mean(values):.4f}±{stdev(values):.4f}"


def main() -> None:
    args = parse_args()
    paths = sorted(Path(".").glob(args.pattern))
    if not paths:
        print(f"No summary files matched pattern '{args.pattern}'.")
        return
    bucket = collect_metrics(paths, args.metrics, args.exclude)
    if not bucket:
        print("No metrics found in the provided summary files.")
        return
    header = ["Method"] + args.metrics + ["N"]
    row_format = "{:<25}" + "  {:>18}" * len(args.metrics) + "  {:>4}"
    print(row_format.format(*header))
    for method in sorted(bucket):
        metric_values = bucket[method]
        counts = [len(metric_values[m]) for m in args.metrics if metric_values[m]]
        n = min(counts) if counts else 0
        cells = [method]
        for metric in args.metrics:
            cells.append(format_mean_std(metric_values[metric]))
        cells.append(str(n))
        print(row_format.format(*cells))


if __name__ == "__main__":
    main()
