from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

MetricDict = Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation metrics across datasets/runs."
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="runs",
        help="Root directory containing experiment runs (default: runs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON file to save aggregated results.",
    )
    return parser.parse_args()


def collect_metrics(root: Path) -> Dict[str, List[MetricDict]]:
    records: Dict[str, List[MetricDict]] = defaultdict(list)
    for metrics_path in root.rglob("metrics.json"):
        parent = metrics_path.parent
        try:
            data = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        if parent.name.startswith("eval_"):
            dataset = data.get("dataset", "unknown")
            split = data.get("split", "unknown")
            metrics = data.get("metrics")
            if not isinstance(metrics, dict):
                continue
            key = f"{dataset}:{split}"
            records[key].append(
                {
                    "pixel_accuracy": float(metrics.get("pixel_accuracy", 0.0)),
                    "mean_iou": float(metrics.get("mean_iou", 0.0)),
                    "boundary_f1": float(metrics.get("boundary_f1", 0.0)),
                }
            )
        else:
            if not isinstance(data, list):
                continue
            if not parent.name:
                dataset = "unknown"
            else:
                dataset = parent.name.split("_")[0]
            split = "val"
            best = max(
                data,
                key=lambda entry: entry.get("miou", entry.get("pixel_acc", 0.0)),
            )
            metrics = {
                "pixel_accuracy": float(best.get("pixel_acc", 0.0)),
                "mean_iou": float(best.get("miou", 0.0)),
                "boundary_f1": float(best.get("dice", 0.0)),
            }
            key = f"{dataset}:{split}"
            records[key].append(metrics)
    return records


def aggregate(records: Dict[str, List[MetricDict]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, metrics_list in sorted(records.items()):
        dataset, split = key.split(":", 1)
        table: Dict[str, Dict[str, float]] = {}
        for metric_name in ["pixel_accuracy", "mean_iou", "boundary_f1"]:
            values = [
                float(metric.get(metric_name, 0.0))
                for metric in metrics_list
                if metric_name in metric
            ]
            if not values:
                continue
            mean = statistics.mean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            table[metric_name] = {"mean": mean, "std": std, "runs": len(values)}
        summary[f"{dataset}:{split}"] = table
    return summary


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    records = collect_metrics(runs_dir)
    if not records:
        print("No evaluation metrics.json files found under", runs_dir)
        return
    summary = aggregate(records)
    for key, metrics in summary.items():
        dataset, split = key.split(":", 1)
        print(f"{dataset} [{split}]")
        for metric_name, stats in metrics.items():
            mean = stats["mean"]
            std = stats["std"]
            runs = stats["runs"]
            print(
                f"  {metric_name}: mean={mean:.4f}, std={std:.4f} (n={runs})"
            )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("Saved aggregated results to", output_path)


if __name__ == "__main__":
    main()
