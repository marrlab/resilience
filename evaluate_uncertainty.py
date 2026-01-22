from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty scores using saved quality labels."
    )
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["dsb2018", "monuseg", "rus"],
        help="Datasets to evaluate.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--bad_quantile",
        type=float,
        default=0.2,
        help="Fraction of lowest-Dice samples to mark as bad (default: 0.2).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["single"],
        help="Uncertainty estimation methods to evaluate.",
    )
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_records(run_dir: Path, dataset: str, split: str, method: str) -> Tuple[List[Dict], List[Dict]]:
    quality_path = run_dir / f"quality_{dataset}_{split}.json"
    unc_dir = run_dir / f"uncertainty_{dataset}_{split}_{method}"
    unc_path = unc_dir / f"uncertainty_{dataset}_{split}_{method}.json"
    if not quality_path.exists() or not unc_path.exists():
        raise FileNotFoundError(
            f"Missing quality or uncertainty files for {dataset} {split} under {run_dir}"
        )
    q = load_json(quality_path)["records"]
    u = load_json(unc_path)["records"]
    return q, u


def join_quality_uncertainty(
    quality: List[Dict], uncertainty: List[Dict]
) -> List[Dict]:
    key = lambda rec: rec.get("sample_id", rec.get("index"))
    q_map = {key(rec): rec for rec in quality}
    records = []
    for u_rec in uncertainty:
        k = key(u_rec)
        if k not in q_map:
            continue
        rec = {
            **q_map[k],
            **u_rec,
            "sample_key": k,
        }
        records.append(rec)
    return records


def mark_bad_samples(records: List[Dict], quantile: float) -> None:
    if not records:
        return
    dice_scores = np.array([rec.get("dice", 0.0) for rec in records], dtype=np.float64)
    threshold = float(np.quantile(dice_scores, quantile))
    for rec, dice in zip(records, dice_scores):
        rec["bad_label"] = float(dice <= threshold)
    return


def risk_coverage_metrics(records: List[Dict]) -> Dict[str, float]:
    sorted_records = sorted(records, key=lambda r: r["unc_boundary_mean"])
    if not sorted_records:
        return {}
    dice = np.array([r.get("dice", 0.0) for r in sorted_records], dtype=np.float64)
    prefix_mean = np.cumsum(dice) / np.arange(1, len(dice) + 1)
    coverage = np.arange(1, len(dice) + 1) / len(dice)
    risk = 1.0 - prefix_mean

    def coverage_dice(threshold: float) -> Tuple[float, int]:
        k = max(1, math.ceil(threshold * len(dice)))
        return float(prefix_mean[k - 1]), k

    dice_80, k80 = coverage_dice(0.8)
    dice_90, k90 = coverage_dice(0.9)
    tau_80 = float(sorted_records[k80 - 1]["unc_boundary_mean"])
    tau_90 = float(sorted_records[k90 - 1]["unc_boundary_mean"])

    aurc = np.trapezoid(risk, coverage)
    stats = {
        "dice_at_80": dice_80,
        "dice_at_90": dice_90,
        "aurc": float(aurc),
        "tau_80": tau_80,
        "tau_90": tau_90,
        "num_bad": float(sum(1 for r in records if r.get("bad_label", 0.0) > 0.5)),
        "num_good": float(sum(1 for r in records if r.get("bad_label", 0.0) <= 0.5)),
        "dice_min": float(dice.min()),
        "dice_median": float(np.median(dice)),
        "dice_max": float(dice.max()),
    }
    return stats


def error_detection_metrics(records: List[Dict]) -> Dict[str, float]:
    if not records:
        return {}
    scores = np.array([r["unc_boundary_mean"] for r in records], dtype=np.float64)
    labels = np.array([r.get("bad_label", 0.0) for r in records], dtype=np.float64)
    unique = np.unique(labels)
    if unique.size < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            auroc = roc_auc_score(labels, scores)
            auprc = average_precision_score(labels, scores)
    return {"auroc": float(auroc), "auprc": float(auprc)}


def summarize_dataset(dataset: str, split: str, records: List[List[Dict]]) -> Dict[str, float]:
    combined = []
    for run_records in records:
        combined.extend(run_records)
    rc = risk_coverage_metrics(combined)
    ed = error_detection_metrics(combined)
    summary = {**rc, **ed, "num_samples": len(combined)}
    return summary


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    aggregate: Dict[str, Dict[str, float]] = {}

    for dataset in args.datasets:
        for method in args.methods:
            dataset_records: List[List[Dict]] = []
            bad_q = args.bad_quantile
            for exp_dir in sorted(runs_dir.glob(f"{dataset}_*")):
                quality_path = exp_dir / f"quality_{dataset}_{args.split}.json"
                if not quality_path.exists():
                    continue
                try:
                    quality, uncertainty = load_records(exp_dir, dataset, args.split, method)
                except FileNotFoundError:
                    continue
                joined = join_quality_uncertainty(quality, uncertainty)
                if not joined:
                    continue
                mark_bad_samples(joined, bad_q)
                dataset_records.append(joined)
                rc = risk_coverage_metrics(joined)
                ed = error_detection_metrics(joined)
                print(
                    f"{dataset} | {method} | {exp_dir.name} | "
                    f"Dice@80={rc['dice_at_80']:.4f} (tau={rc['tau_80']:.4f}) "
                    f"Dice@90={rc['dice_at_90']:.4f} (tau={rc['tau_90']:.4f}) "
                    f"AURC={rc['aurc']:.4f} AUROC={ed['auroc']:.4f} AUPRC={ed['auprc']:.4f} | "
                    f"bad={rc['num_bad']:.0f}, good={rc['num_good']:.0f} | "
                    f"Dice[min/med/max]={rc['dice_min']:.4f}/{rc['dice_median']:.4f}/{rc['dice_max']:.4f}"
                )
            if dataset_records:
                summary = summarize_dataset(dataset, args.split, dataset_records)
                aggregate[f"{dataset}:{args.split}:{method}"] = summary
                print(
                    f"{dataset} [{args.split}] {method} aggregate | "
                    f"Dice@80={summary['dice_at_80']:.4f} (tau={summary['tau_80']:.4f}) "
                    f"Dice@90={summary['dice_at_90']:.4f} (tau={summary['tau_90']:.4f}) "
                    f"AURC={summary['aurc']:.4f} AUROC={summary['auroc']:.4f} "
                    f"AUPRC={summary['auprc']:.4f} | bad={summary['num_bad']:.0f}, "
                    f"good={summary['num_good']:.0f} | Dice[min/med/max]={summary['dice_min']:.4f}/"
                    f"{summary['dice_median']:.4f}/{summary['dice_max']:.4f} "
                    f"(n={summary['num_samples']})"
                )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2)
        print(f"Saved aggregate metrics to {out_path}")


if __name__ == "__main__":
    main()
