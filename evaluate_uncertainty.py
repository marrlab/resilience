from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

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
        default=[
            "dsb2018",
            "monuseg",
            "rus",
            "nuinsseg",
            "isic2017",
            "kvasirseg",
            "clinicdb",
            "drive",
            "promise12",
        ],
        help="Datasets to evaluate.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--fusion_pairs",
        type=str,
        nargs="+",
        default=None,
        help="Optional fusion pairs specified as method1,method2 (e.g., single,tta).",
    )
    parser.add_argument(
        "--fusion_metric",
        type=str,
        choices=["aurc", "dice_at_80", "dice_at_90"],
        default="aurc",
        help="Metric optimized when tuning weighted fusion.",
    )
    parser.add_argument(
        "--fusion_alpha_steps",
        type=int,
        default=21,
        help="Number of points in [0,1] grid when tuning alpha (default: 21).",
    )
    parser.add_argument(
        "--fusion_val_split",
        type=str,
        default="val",
        help="Split used for computing fusion statistics (default: val).",
    )
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


def summarize_run_stats(run_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not run_metrics:
        return {}
    keys = run_metrics[0].keys()
    stats: Dict[str, float] = {"run_count": len(run_metrics)}
    for key in keys:
        values = [metrics.get(key) for metrics in run_metrics if key in metrics]
        values = [float(v) for v in values if v is not None and not np.isnan(v)]
        if not values:
            continue
        stats[f"{key}_mean"] = float(np.mean(values))
        stats[f"{key}_std"] = float(np.std(values))
    return stats


def _record_key(rec: Dict) -> Optional[str]:
    return rec.get("sample_key") or rec.get("sample_id") or rec.get("index")


def build_record_map(records: Iterable[Dict]) -> Dict[str, Dict]:
    mapping: Dict[str, Dict] = {}
    for rec in records:
        key = _record_key(rec)
        if key is not None:
            mapping[str(key)] = rec
    return mapping


def intersect_record_keys(maps: List[Dict[str, Dict]]) -> List[str]:
    if not maps:
        return []
    key_sets = [set(m.keys()) for m in maps]
    common = set.intersection(*key_sets)
    return sorted(common)


def fuse_rank_average_records(
    records_list: List[List[Dict]],
    method_name: str,
    source_methods: Tuple[str, ...],
) -> List[Dict]:
    maps = [build_record_map(records) for records in records_list]
    keys = intersect_record_keys(maps)
    if not keys:
        return []
    rank_lists: List[np.ndarray] = []
    for mapping in maps:
        scores = np.array([mapping[key]["unc_boundary_mean"] for key in keys], dtype=np.float64)
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(order), dtype=np.float64)
        rank_lists.append(ranks)
    avg_rank = np.mean(np.stack(rank_lists, axis=0), axis=0)
    fused_records: List[Dict] = []
    for idx, key in enumerate(keys):
        base = dict(maps[0][key])
        fused_score = float(avg_rank[idx])
        base["unc_mean"] = fused_score
        base["unc_boundary_mean"] = fused_score
        base["unc_boundary_p95"] = fused_score
        base["method"] = method_name
        base["fused_from"] = list(source_methods)
        fused_records.append(base)
    return fused_records


def compute_standard_stats(records: List[Dict]) -> Tuple[float, float]:
    if not records:
        return 0.0, 1.0
    scores = np.array([rec["unc_boundary_mean"] for rec in records], dtype=np.float64)
    mean = float(scores.mean())
    std = float(scores.std())
    if std <= 0:
        std = 1.0
    return mean, std


def create_weighted_records(
    records_a: List[Dict],
    records_b: List[Dict],
    stats_a: Tuple[float, float],
    stats_b: Tuple[float, float],
    alpha: float,
    method_name: str,
    source_methods: Tuple[str, str],
) -> List[Dict]:
    map_a = build_record_map(records_a)
    map_b = build_record_map(records_b)
    keys = intersect_record_keys([map_a, map_b])
    if not keys:
        return []
    mean_a, std_a = stats_a
    mean_b, std_b = stats_b
    fused_records: List[Dict] = []
    for key in keys:
        rec_a = map_a[key]
        rec_b = map_b[key]
        score_a = (rec_a["unc_boundary_mean"] - mean_a) / std_a
        score_b = (rec_b["unc_boundary_mean"] - mean_b) / std_b
        fused_score = alpha * score_a + (1.0 - alpha) * score_b
        fused_rec = dict(rec_a)
        fused_rec["unc_mean"] = float(fused_score)
        fused_rec["unc_boundary_mean"] = float(fused_score)
        fused_rec["unc_boundary_p95"] = float(fused_score)
        fused_rec["method"] = method_name
        fused_rec["fused_from"] = list(source_methods)
        fused_rec["fusion_alpha"] = float(alpha)
        fused_records.append(fused_rec)
    return fused_records


def tune_weighted_alpha(
    records_a: List[Dict],
    records_b: List[Dict],
    stats_a: Tuple[float, float],
    stats_b: Tuple[float, float],
    metric: str,
    steps: int,
    source_methods: Tuple[str, str],
) -> Optional[float]:
    if steps < 2:
        steps = 2
    best_alpha: Optional[float] = None
    best_value: Optional[float] = None
    maximize = metric in {"dice_at_80", "dice_at_90"}
    for alpha in np.linspace(0.0, 1.0, steps):
        fused = create_weighted_records(records_a, records_b, stats_a, stats_b, alpha, "fusion_tmp", source_methods)
        if not fused:
            continue
        rc = risk_coverage_metrics(fused)
        value = rc.get(metric)
        if value is None:
            continue
        if best_value is None:
            best_value = value
            best_alpha = float(alpha)
            continue
        if maximize and value > best_value:
            best_value = value
            best_alpha = float(alpha)
        elif not maximize and value < best_value:
            best_value = value
            best_alpha = float(alpha)
    return best_alpha


def load_and_prepare_records(
    exp_dir: Path,
    dataset: str,
    split: str,
    method: str,
    bad_quantile: float,
) -> Optional[List[Dict]]:
    quality_path = exp_dir / f"quality_{dataset}_{split}.json"
    if not quality_path.exists():
        return None
    try:
        quality, uncertainty = load_records(exp_dir, dataset, split, method)
    except FileNotFoundError:
        return None
    joined = join_quality_uncertainty(quality, uncertainty)
    if not joined:
        return None
    mark_bad_samples(joined, bad_quantile)
    return joined


def log_run_metrics(dataset: str, method: str, run_name: str, records: List[Dict]) -> None:
    rc = risk_coverage_metrics(records)
    ed = error_detection_metrics(records)
    if not rc:
        return
    print(
        f"{dataset} | {method} | {run_name} | "
        f"Dice@80={rc['dice_at_80']:.4f} (tau={rc['tau_80']:.4f}) "
        f"Dice@90={rc['dice_at_90']:.4f} (tau={rc['tau_90']:.4f}) "
        f"AURC={rc['aurc']:.4f} AUROC={ed['auroc']:.4f} AUPRC={ed['auprc']:.4f} | "
        f"bad={rc['num_bad']:.0f}, good={rc['num_good']:.0f} | "
        f"Dice[min/med/max]={rc['dice_min']:.4f}/{rc['dice_median']:.4f}/{rc['dice_max']:.4f}"
    )


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    aggregate: Dict[str, Dict[str, float]] = {}
    any_summary = False
    fusion_pairs: List[Tuple[str, str]] = []
    if args.fusion_pairs:
        for pair_str in args.fusion_pairs:
            parts = [p.strip() for p in pair_str.split(",") if p.strip()]
            if len(parts) != 2:
                raise ValueError(f"Fusion pair '{pair_str}' must be formatted as method1,method2")
            fusion_pairs.append((parts[0], parts[1]))

    for dataset in args.datasets:
        dataset_records: Dict[str, List[List[Dict]]] = {}
        bad_q = args.bad_quantile
        for exp_dir in sorted(runs_dir.glob(f"{dataset}_*")):
            per_method_records: Dict[str, List[Dict]] = {}
            for method in args.methods:
                joined = load_and_prepare_records(exp_dir, dataset, args.split, method, bad_q)
                if not joined:
                    continue
                per_method_records[method] = joined
                dataset_records.setdefault(method, []).append(joined)
                log_run_metrics(dataset, method, exp_dir.name, joined)

            # Fusion baselines per run
            for method_a, method_b in fusion_pairs:
                if method_a not in per_method_records or method_b not in per_method_records:
                    continue
                rank_name = f"fusion_rank_{method_a}_{method_b}"
                fused_rank = fuse_rank_average_records(
                    [per_method_records[method_a], per_method_records[method_b]],
                    rank_name,
                    (method_a, method_b),
                )
                if fused_rank:
                    dataset_records.setdefault(rank_name, []).append(fused_rank)
                    log_run_metrics(dataset, rank_name, exp_dir.name, fused_rank)

                val_a = load_and_prepare_records(
                    exp_dir, dataset, args.fusion_val_split, method_a, bad_q
                )
                val_b = load_and_prepare_records(
                    exp_dir, dataset, args.fusion_val_split, method_b, bad_q
                )
                if not val_a or not val_b:
                    continue
                stats_a = compute_standard_stats(val_a)
                stats_b = compute_standard_stats(val_b)
                best_alpha = tune_weighted_alpha(
                    val_a,
                    val_b,
                    stats_a,
                    stats_b,
                    args.fusion_metric,
                    args.fusion_alpha_steps,
                    (method_a, method_b),
                )
                if best_alpha is None:
                    continue
                weighted_name = f"fusion_weighted_{method_a}_{method_b}"
                fused_weighted = create_weighted_records(
                    per_method_records[method_a],
                    per_method_records[method_b],
                    stats_a,
                    stats_b,
                    best_alpha,
                    weighted_name,
                    (method_a, method_b),
                )
                if fused_weighted:
                    dataset_records.setdefault(weighted_name, []).append(fused_weighted)
                    print(
                        f"{dataset} | {weighted_name} | {exp_dir.name} | "
                        f"alpha={best_alpha:.3f} tuned on split '{args.fusion_val_split}'"
                    )
                    log_run_metrics(dataset, weighted_name, exp_dir.name, fused_weighted)

        for method, records_list in dataset_records.items():
            if not records_list:
                continue
            run_metrics: List[Dict[str, float]] = []
            for run_records in records_list:
                rc = risk_coverage_metrics(run_records)
                if not rc:
                    continue
                ed = error_detection_metrics(run_records)
                run_metrics.append({**rc, **ed})
            summary = summarize_dataset(dataset, args.split, records_list)
            summary.update(summarize_run_stats(run_metrics))
            aggregate[f"{dataset}:{args.split}:{method}"] = summary
            any_summary = True
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
        if aggregate:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(aggregate, f, indent=2)
            print(f"Saved aggregate metrics to {out_path}")
        else:
            print(
                f"No aggregate metrics computed for datasets {args.datasets}; "
                "skipping output."
            )


if __name__ == "__main__":
    main()
