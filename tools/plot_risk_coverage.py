#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot risk-coverage curves for uncertainty methods.")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["dsb2018", "monuseg", "rus", "nuinsseg", "isic2017", "kvasirseg", "clinicdb", "drive"],
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["single", "stoptime", "stability", "flicker", "resilience", "tta", "disagreement"],
    )
    parser.add_argument("--output", type=str, default="runs/risk_coverage")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--average",
        action="store_true",
        help="Also plot the dataset-averaged risk-coverage curves with mean±std envelopes.",
    )
    parser.add_argument(
        "--avg_exclude",
        type=str,
        nargs="+",
        default=[],
        help="Datasets to exclude when building the averaged plot.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_records(run_dir: Path, dataset: str, split: str, method: str) -> Optional[List[Dict]]:
    quality_path = run_dir / f"quality_{dataset}_{split}.json"
    unc_dir = run_dir / f"uncertainty_{dataset}_{split}_{method}"
    unc_path = unc_dir / f"uncertainty_{dataset}_{split}_{method}.json"
    if not quality_path.exists() or not unc_path.exists():
        return None
    quality = load_json(quality_path).get("records", [])
    unc = load_json(unc_path).get("records", [])
    return join_quality_uncertainty(quality, unc)


def join_quality_uncertainty(quality: List[Dict], uncertainty: List[Dict]) -> List[Dict]:
    key = lambda rec: rec.get("sample_id", rec.get("index"))
    q_map = {key(rec): rec for rec in quality}
    records: List[Dict] = []
    for u_rec in uncertainty:
        k = key(u_rec)
        if k not in q_map:
            continue
        merged = {**q_map[k], **u_rec}
        records.append(merged)
    return records


def gather_records(
    runs_dir: Path, dataset: str, split: str, methods: List[str]
) -> Dict[str, List[Dict]]:
    collected: Dict[str, List[Dict]] = {}
    for exp_dir in sorted(runs_dir.glob(f"{dataset}_*")):
        for method in methods:
            joined = load_records(exp_dir, dataset, split, method)
            if not joined:
                continue
            collected.setdefault(method, []).extend(joined)
    return collected


def compute_risk_curve(records: List[Dict]) -> Optional[Dict[str, np.ndarray]]:
    if not records:
        return None
    sorted_records = sorted(records, key=lambda r: r["unc_boundary_mean"])
    dice = np.array([r.get("dice", 0.0) for r in sorted_records], dtype=np.float64)
    errors = 1.0 - dice
    coverage = np.arange(1, len(dice) + 1, dtype=np.float64) / len(dice)
    selective_risk = np.cumsum(errors) / np.arange(1, len(dice) + 1, dtype=np.float64)
    return {"coverage": coverage, "risk": selective_risk, "dice": dice}


def coverage_stat(curve: Dict[str, np.ndarray], threshold: float) -> float:
    coverage = curve["coverage"]
    idx = max(0, min(len(coverage) - 1, int(np.ceil(threshold * len(coverage)) - 1)))
    return float(1.0 - curve["risk"][idx])


def aurc(curve: Dict[str, np.ndarray]) -> float:
    return float(np.trapezoid(curve["risk"], curve["coverage"]))


def plot_dataset(
    dataset: str,
    split: str,
    curves: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
    dpi: int,
) -> None:
    if not curves:
        print(f"[risk-coverage] Skipping {dataset}: no curves available.")
        return
    plt.figure(figsize=(6, 4))
    for method, data in curves.items():
        plt.plot(data["coverage"], data["risk"], label=f"{method} (AURC={aurc(data):.3f})")
    plt.xlabel("Coverage")
    plt.ylabel("Selective risk (1 - Dice)")
    plt.title(f"{dataset.upper()} risk-coverage")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset}_{split}_rc.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"[risk-coverage] Saved plot to {out_path}")


def plot_average(
    split: str,
    curves_by_method: Dict[str, List[Dict[str, np.ndarray]]],
    output_dir: Path,
    dpi: int,
) -> None:
    if not curves_by_method:
        print("[risk-coverage] No curves available for averaging.")
        return
    coverage = np.linspace(0.0, 1.0, 200)
    plt.figure(figsize=(6, 4))
    for method, curves in sorted(curves_by_method.items()):
        if not curves:
            continue
        samples = []
        for curve in curves:
            interp = np.interp(
                coverage,
                curve["coverage"],
                curve["risk"],
                left=curve["risk"][0],
                right=curve["risk"][-1],
            )
            samples.append(interp)
        stacked = np.stack(samples, axis=0)
        mean_risk = stacked.mean(axis=0)
        plt.plot(coverage, mean_risk, label=method)
    plt.xlabel("Coverage")
    plt.ylabel("Selective risk (1 - Dice)")
    plt.title(f"Averaged risk-coverage ({split})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"average_{split}_rc.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"[risk-coverage] Saved averaged plot to {out_path}")


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output)
    avg_curves: Dict[str, List[Dict[str, np.ndarray]]] = {}
    avg_exclude = {name.lower() for name in args.avg_exclude}
    for dataset in args.datasets:
        records_by_method = gather_records(runs_dir, dataset, args.split, args.methods)
        curves: Dict[str, Dict[str, np.ndarray]] = {}
        for method in args.methods:
            records = records_by_method.get(method)
            if not records:
                continue
            curve = compute_risk_curve(records)
            if curve is None:
                continue
            curves[method] = curve
            if dataset.lower() not in avg_exclude:
                avg_curves.setdefault(method, []).append(curve)
            print(
                f"{dataset} | {method} | "
                f"AURC={aurc(curve):.4f} Dice@80={coverage_stat(curve, 0.8):.4f} "
                f"Dice@90={coverage_stat(curve, 0.9):.4f} "
                f"n={len(curve['coverage'])}"
            )
        plot_dataset(dataset, args.split, curves, output_dir, args.dpi)
    if args.average:
        plot_average(args.split, avg_curves, output_dir, args.dpi)


if __name__ == "__main__":
    main()
