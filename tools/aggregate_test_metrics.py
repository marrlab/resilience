#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

METRIC_KEYS = ["pixel_accuracy", "mean_iou", "boundary_f1"]

def collect_metrics(runs_root: Path) -> dict[str, dict[str, list[float]]]:
    datasets: dict[str, dict[str, list[float]]] = {}
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        for metrics_path in run_dir.glob("eval_*_test/metrics.json"):
            try:
                dataset = metrics_path.parent.name.split("_")[1]
            except IndexError:
                continue
            data = json.loads(metrics_path.read_text())
            metrics = data.get("metrics", {})
            bucket = datasets.setdefault(dataset, {k: [] for k in METRIC_KEYS})
            for key in METRIC_KEYS:
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    bucket[key].append(float(value))
    return datasets


def summarize(metrics: dict[str, list[float]]) -> dict[str, tuple[float, float, int]]:
    summary: dict[str, tuple[float, float, int]] = {}
    for key, values in metrics.items():
        if not values:
            continue
        m = mean(values)
        summary[key] = (m, stdev(values) if len(values) > 1 else 0.0, len(values))
    return summary


def main() -> None:
    runs_root = Path("runs")
    datasets = collect_metrics(runs_root)
    rows = []
    for dataset in sorted(datasets):
        summary = summarize(datasets[dataset])
        if not summary:
            continue
        rows.append((dataset, summary))

    if not rows:
        print("No evaluation metrics found under runs/*/eval_*_test/metrics.json")
        return

    header = ["Dataset"] + [f"{key} (mean±std)" for key in METRIC_KEYS]
    print("\t\t".join(header))
    for dataset, summary in rows:
        cells = [dataset]
        for key in METRIC_KEYS:
            if key in summary:
                mean_val, std_val, count = summary[key]
                cells.append(f"{mean_val:.4f}±{std_val:.4f} (n={count})")
            else:
                cells.append("N/A")
        print("\t\t".join(cells))

if __name__ == "__main__":
    main()
