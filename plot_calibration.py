#!/usr/bin/env python3
"""Fit validation-only failure probabilities and plot test-set calibration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATASETS = [
    "dsb2018", "monuseg", "rus", "nuinsseg", "isic2017",
    "kvasirseg", "clinicdb", "drive", "promise12", "raabin",
]
METHODS = [
    "resilience_single_latent", "resilience_double_latent",
    "resilience_single_rgb", "resilience_double_rgb",
]
PROTOCOL_KEYS = [
    "checkpoint", "method", "resilience_mode", "perturb_rgb", "input_conditioning",
    "steps", "resilience_relax_steps", "resilience_noise", "perturbation_count",
    "dropout_rate", "mc_dropout_samples", "noise_trials", "num_trials",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--datasets", "--dataset", nargs="+", default=DATASETS)
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        help="Saved method IDs, not AUROC summary tables.")
    parser.add_argument("--calibration_split", default="val", help="Split used to fit the mapping.")
    parser.add_argument("--split", default="test", help="Independent evaluation split.")
    parser.add_argument("--run_name", default=None, help="Optional specific run directory.")
    parser.add_argument("--output", default="runs/calibration", help="Output directory.")
    parser.add_argument("--failure_threshold", "--dice_error_threshold", type=float, default=0.8,
                        help="Failure iff original Dice < this fixed threshold (default 0.8).")
    parser.add_argument("--score_key", default="unc_boundary_mean")
    parser.add_argument("--calibrator", choices=["logistic", "isotonic", "auto"], default="logistic")
    parser.add_argument("--logistic_c", type=float, default=1.0, help="Fixed logistic regularization C.")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--selection_metric", choices=["brier", "log_loss"], default="brier")
    parser.add_argument("--min_isotonic_samples", type=int, default=100,
                        help="Minimum validation count before auto compares isotonic.")
    parser.add_argument("--min_isotonic_class_count", type=int, default=10,
                        help="Minimum failures AND successes before auto compares isotonic.")
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--strategy", choices=["quantile", "uniform"], default="quantile")
    parser.add_argument("--confidence_level", type=float, default=0.95)
    grouping = parser.add_mutually_exclusive_group()
    grouping.add_argument("--group_key", help="Patient/slide/video ID field in saved per-image records.")
    grouping.add_argument("--group_file", help="JSON mapping dataset -> split -> sample_id -> group ID.")
    parser.add_argument("--bootstrap_samples", type=int, default=1000,
                        help="Group bootstrap repetitions when grouping is provided.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def record_id(record):
    value = record.get("sample_id", record.get("index"))
    if value is None:
        raise ValueError("Every record must have sample_id or index.")
    return str(value)


def index_records(records, source):
    result = {}
    for record in records:
        key = record_id(record)
        if key in result:
            raise ValueError(f"Duplicate sample ID {key!r} in {source}.")
        result[key] = record
    return result


def load_image_results(run_dir, dataset, split, method, score_key, threshold,
                       group_key=None, group_mapping=None):
    path = run_dir / f"uncertainty_{dataset}_{split}_{method}" / f"uncertainty_{dataset}_{split}_{method}.json"
    document = load_json(path)
    for key, expected in (("dataset", dataset), ("split", split), ("method", method)):
        if key in document and document[key] != expected:
            raise ValueError(f"{path}: {key} does not match the requested value.")
    uncertainty = document.get("records")
    if not isinstance(uncertainty, list) or not uncertainty:
        raise ValueError(f"{path}: expected non-empty per-image records; summary tables are insufficient.")
    index_records(uncertainty, path)
    quality_path = run_dir / f"quality_{dataset}_{split}.json"
    quality = index_records(load_json(quality_path)["records"], quality_path) if quality_path.exists() else {}
    rows = []
    for entry in uncertainty:
        key = record_id(entry)
        known_baseline = entry.get("quality_source") == "resilience_baseline" and "dice" in entry
        if not known_baseline and key not in quality:
            raise ValueError(f"{path}: missing original-prediction Dice for {key}; generate quality labels.")
        merged = {**quality.get(key, {}), **entry}
        dice = float(entry["dice"] if known_baseline else quality[key]["dice"])
        if score_key not in entry:
            raise ValueError(f"{path}: record {key} has no {score_key}.")
        u = float(entry[score_key])
        if not np.isfinite(u) or not np.isfinite(dice) or not 0 <= dice <= 1:
            raise ValueError(f"{path}: invalid score or Dice for {key}.")
        if method.startswith("resilience") and not 0 <= u <= 1:
            raise ValueError(f"{path}: resilience must be in [0, 1].")
        row = {
            "sample_id": key, "index": entry.get("index"), "u": u, "dice": dice,
            "failure": int(dice < threshold),
            "quality_source": "resilience_baseline" if known_baseline else str(quality_path),
        }
        for field in ("image_path", "case_dir"):
            if merged.get(field):
                row[field] = str(merged[field])
        if group_key:
            if merged.get(group_key) is None:
                raise ValueError(f"{path}: {key} has no group field {group_key!r}.")
            row["group_id"] = str(merged[group_key])
        elif group_mapping is not None:
            group = group_mapping.get(dataset, {}).get(split, {}).get(key)
            if group is None:
                raise ValueError(f"Group file has no group for {dataset}/{split}/{key}.")
            row["group_id"] = str(group)
        rows.append(row)
    return document, rows, path


def validate_splits(calibration_doc, test_doc, calibration, test):
    """Reject observed overlap and changed inference settings, without using test labels."""
    for key in PROTOCOL_KEYS:
        left, right = calibration_doc.get(key), test_doc.get(key)
        if left is not None and right is not None and left != right:
            raise ValueError(f"Validation/test protocol mismatch for {key}: {left!r} != {right!r}.")
    def identities(rows):
        found = set()
        for row in rows:
            source = row.get("image_path") or row.get("case_dir")
            if source:
                found.add(("path", os.path.normcase(os.path.normpath(source))))
            # Split-local numeric indices cannot establish identity across splits.
            elif row["sample_id"] != str(row.get("index")):
                found.add(("sample_id", row["sample_id"]))
        return found
    overlap = identities(calibration) & identities(test)
    if overlap:
        raise ValueError(f"Calibration/test data overlap: {len(overlap)} shared sample(s).")
    groups_cal = {row["group_id"] for row in calibration if "group_id" in row}
    groups_test = {row["group_id"] for row in test if "group_id" in row}
    if groups_cal & groups_test:
        raise ValueError("Calibration/test splits share patient/slide/video groups.")
    required = ["checkpoint", "method"]
    if str(calibration_doc.get("method", "")).startswith("resilience"):
        required += ["steps", "resilience_noise", "resilience_relax_steps",
                     "resilience_mode", "perturb_rgb", "input_conditioning"]
    return {
        "unverified_protocol_fields": [
            key for key in required if calibration_doc.get(key) is None or test_doc.get(key) is None
        ],
        "calibration_identifiable_samples": len(identities(calibration)),
        "test_identifiable_samples": len(identities(test)),
        "group_overlap_checked": bool(groups_cal),
    }


def fit_mapping(kind, u, failures, c=1.0):
    if len(np.unique(failures)) != 2:
        raise ValueError("Calibration data must contain both failures and successes; change the calibration set, not test labels.")
    if kind == "isotonic":
        return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(u, failures)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
    return model.fit(u.reshape(-1, 1), failures)


def predict_probability(model, u):
    if isinstance(model, IsotonicRegression):
        return model.predict(u)
    return model.predict_proba(u.reshape(-1, 1))[:, 1]


def mapping_parameters(model):
    if isinstance(model, IsotonicRegression):
        return {"kind": "isotonic", "increasing": True, "out_of_bounds": "clip",
                "x_thresholds": model.X_thresholds_.tolist(), "y_thresholds": model.y_thresholds_.tolist()}
    scaler, logistic = model.steps[0][1], model.steps[1][1]
    a = float(logistic.coef_[0, 0] / scaler.scale_[0])
    b = float(logistic.intercept_[0] - a * scaler.mean_[0])
    return {"kind": "logistic", "a": a, "b": b, "C": float(logistic.C),
            "negative_slope": a < 0, "formula": "1 / (1 + exp(-(a*u + b)))"}


def select_mapping(u, failures, groups, args):
    """Select exclusively within validation, then refit on all validation images."""
    if len(np.unique(failures)) != 2:
        raise ValueError("Calibration data must contain both failures and successes.")
    details = {"requested": args.calibrator, "selection_metric": args.selection_metric,
               "cv_scores": {}, "cv_folds_used": 0}
    selected = args.calibrator
    if selected == "auto":
        selected = "logistic"
        counts = np.bincount(failures, minlength=2)
        eligible = len(u) >= args.min_isotonic_samples and counts.min() >= args.min_isotonic_class_count
        n_splits = min(args.cv_folds, int(counts.min()))
        if groups is not None:
            n_splits = min(n_splits, len(np.unique(groups)))
        if eligible and n_splits >= 2:
            splitter = (StratifiedKFold(n_splits, shuffle=True, random_state=args.seed) if groups is None
                        else StratifiedGroupKFold(n_splits, shuffle=True, random_state=args.seed))
            folds = list(splitter.split(u.reshape(-1, 1), failures, groups))
            usable = all(len(np.unique(failures[train])) == 2 for train, _ in folds)
            if usable:
                for kind in ("logistic", "isotonic"):
                    heldout = np.empty(len(u), dtype=float)
                    for train, held in folds:
                        model = fit_mapping(kind, u[train], failures[train], args.logistic_c)
                        heldout[held] = predict_probability(model, u[held])
                    details["cv_scores"][kind] = {
                        "brier": float(brier_score_loss(failures, heldout)),
                        "log_loss": float(log_loss(failures, heldout, labels=[0, 1])),
                    }
                details["cv_folds_used"] = n_splits
                selected = min(details["cv_scores"], key=lambda kind: details["cv_scores"][kind][args.selection_metric])
            else:
                details["fallback_reason"] = "Group folds could not retain both classes in every training fold."
        else:
            details["fallback_reason"] = "Insufficient validation samples/classes for isotonic comparison."
    details["selected"] = selected
    return fit_mapping(selected, u, failures, args.logistic_c), details


def wilson_interval(failures, count, confidence_level):
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    proportion = failures / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    lower = 0.0 if failures == 0 else max(0.0, float(center - half))
    upper = 1.0 if failures == count else min(1.0, float(center + half))
    return lower, upper


def calibration_bins(failures, probabilities, n_bins=10, strategy="quantile",
                     confidence_level=0.95, groups=None, bootstrap_samples=1000, seed=42):
    # Match sklearn.calibration.calibration_curve's binning, including tied scores.
    edges = (np.percentile(probabilities, np.linspace(0, 100, n_bins + 1))
             if strategy == "quantile" else np.linspace(0, 1, n_bins + 1))
    assignments = np.searchsorted(edges[1:-1], probabilities)
    bins = []
    memberships = []
    for index in range(n_bins):
        chosen = assignments == index
        count = int(chosen.sum())
        if not count:
            continue
        mistakes = int(failures[chosen].sum())
        lower, upper = wilson_interval(mistakes, count, confidence_level)
        bins.append({
            "bin": index, "lower_edge": float(edges[index]), "upper_edge": float(edges[index + 1]),
            "count": count, "failures": mistakes,
            "mean_probability": float(probabilities[chosen].mean()),
            "fraction_failed": mistakes / count, "ci_lower": lower, "ci_upper": upper,
            "ci_method": "wilson", "group_count": None,
        })
        memberships.append(chosen)
    if groups is not None:
        unique, inverse = np.unique(groups, return_inverse=True)
        if len(unique) < 2:
            raise ValueError("Group confidence intervals require at least two independent test groups.")
        rng = np.random.default_rng(seed)
        boot = [[] for _ in bins]
        for _ in range(bootstrap_samples):
            weights = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))[inverse]
            for index, chosen in enumerate(memberships):
                denominator = weights[chosen].sum()
                if denominator:
                    boot[index].append(float(np.dot(weights[chosen], failures[chosen]) / denominator))
        alpha = (1 - confidence_level) / 2
        for row, chosen, values in zip(bins, memberships, boot):
            row["group_count"] = int(len(np.unique(groups[chosen])))
            row["bootstrap_valid_samples"] = len(values)
            if row["group_count"] < 2 or len(values) < 2:
                row["ci_lower"], row["ci_upper"] = None, None
                row["ci_method"] = "insufficient_independent_groups"
            else:
                row["ci_lower"], row["ci_upper"] = [float(x) for x in np.quantile(values, [alpha, 1 - alpha])]
                row["ci_method"] = "cluster_bootstrap_fixed_bins"
    return bins


def evaluate_configuration(calibration, test, args):
    u_val = np.array([row["u"] for row in calibration], dtype=float)
    y_val = np.array([row["failure"] for row in calibration], dtype=int)
    groups_val = np.array([row["group_id"] for row in calibration]) if "group_id" in calibration[0] else None
    # Fit and choose the mapping before accessing test outcomes.
    model, selection = select_mapping(u_val, y_val, groups_val, args)
    u_test = np.array([row["u"] for row in test], dtype=float)
    probabilities = predict_probability(model, u_test)
    y_test = np.array([row["failure"] for row in test], dtype=int)
    groups_test = np.array([row["group_id"] for row in test]) if "group_id" in test[0] else None
    bins = calibration_bins(y_test, probabilities, args.n_bins, args.strategy,
                            args.confidence_level, groups_test, args.bootstrap_samples, args.seed)
    has_two_classes = len(np.unique(y_test)) == 2
    ece = sum(row["count"] * abs(row["mean_probability"] - row["fraction_failed"]) for row in bins) / len(test)
    return {
        "mapping": mapping_parameters(model), "selection": selection,
        "calibration_count": len(calibration), "calibration_failures": int(y_val.sum()),
        "test_count": len(test), "test_failures": int(y_test.sum()),
        "metrics": {
            "brier": float(brier_score_loss(y_test, probabilities)),
            "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
            "auroc_uncertainty": float(roc_auc_score(y_test, u_test)) if has_two_classes else None,
            "auroc_probability": float(roc_auc_score(y_test, probabilities)) if has_two_classes else None,
            "ece": float(ece),
        },
        "bins_requested": args.n_bins, "bins_used": len(bins), "bins": bins,
        "calibration_records": calibration,
        "records": [{**row, "failure_probability": float(p)} for row, p in zip(test, probabilities)],
    }


def plot_results(dataset, split, run_name, results, output_dir, dpi, threshold, confidence_level):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for method, result in results.items():
        bins = result["bins"]
        x = np.array([row["mean_probability"] for row in bins])
        y = np.array([row["fraction_failed"] for row in bins])
        low = np.array([row["ci_lower"] for row in bins], dtype=float)
        high = np.array([row["ci_upper"] for row in bins], dtype=float)
        line, = axes[0].plot(x, y, "o-", label=f"{method} ({result['mapping']['kind']})", markersize=4)
        # Draw intervals directly: percentile bootstrap intervals need not bracket the point estimate.
        axes[0].vlines(x, low, high, color=line.get_color(), alpha=0.7)
        for row in bins:
            axes[0].annotate(f"{row['failures']}/{row['count']}",
                             (row["mean_probability"], row["fraction_failed"]),
                             xytext=(3, 5), textcoords="offset points", fontsize=7, color=line.get_color())
        axes[1].hist([row["failure_probability"] for row in result["records"]],
                     bins=np.linspace(0, 1, 11), histtype="step", label=method)
    axes[0].set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.06),
                xlabel="Mean predicted probability of segmentation failure",
                ylabel="Observed proportion of segmentation failures")
    axes[0].set_title(f"Failure: Dice < {threshold:g}; {confidence_level:.0%} intervals\nLabels: failures/images")
    axes[1].set(xlim=(0, 1), xlabel="Predicted failure probability", ylabel="Number of images",
                title="Test probability distribution")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle(f"{dataset} | {run_name} | {split}")
    figure.tight_layout()
    stem = output_dir / f"{dataset}_{split}_calibration"
    for extension in ("png", "pdf"):
        figure.savefig(stem.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def validate_args(args):
    if args.calibration_split == args.split:
        raise ValueError("Calibration and evaluation splits must be different.")
    if not 0 <= args.failure_threshold <= 1 or not np.isfinite(args.failure_threshold):
        raise ValueError("Failure threshold must be in [0, 1].")
    if args.n_bins < 1 or args.cv_folds < 2 or args.bootstrap_samples < 2:
        raise ValueError("Need n_bins >= 1, cv_folds >= 2, bootstrap_samples >= 2.")
    if not 0 < args.confidence_level < 1 or not np.isfinite(args.logistic_c) or args.logistic_c <= 0:
        raise ValueError("Need confidence_level in (0,1) and positive finite logistic_c.")
    if args.min_isotonic_samples < 2 or args.min_isotonic_class_count < 1:
        raise ValueError("Invalid minimum isotonic calibration counts.")


def main():
    args = parse_args()
    validate_args(args)
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    group_mapping = load_json(args.group_file) if args.group_file else None
    completed = 0
    for dataset in args.datasets:
        run_dirs = ([runs_dir / args.run_name] if args.run_name else sorted(runs_dir.glob(f"{dataset}_*")))
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            results = {}
            for method in args.methods:
                try:
                    val_doc, calibration, val_path = load_image_results(
                        run_dir, dataset, args.calibration_split, method, args.score_key,
                        args.failure_threshold, args.group_key, group_mapping)
                    test_doc, test, test_path = load_image_results(
                        run_dir, dataset, args.split, method, args.score_key,
                        args.failure_threshold, args.group_key, group_mapping)
                except FileNotFoundError as error:
                    print(f"[calibration] Skipping {run_dir.name}/{method}: missing {error.filename or error}.")
                    continue
                checks = validate_splits(val_doc, test_doc, calibration, test)
                result = evaluate_configuration(calibration, test, args)
                result.update({
                    "dataset": dataset, "split": args.split, "method": method, "run": run_dir.name,
                    "calibration_split": args.calibration_split, "failure_threshold": args.failure_threshold,
                    "score_key": args.score_key, "seed": args.seed, "strategy": args.strategy,
                    "confidence_level": args.confidence_level, "group_key": args.group_key,
                    "group_file": args.group_file, "split_checks": checks,
                    "calibration_source": str(val_path), "test_source": str(test_path),
                    "protocol": {key: val_doc[key] for key in PROTOCOL_KEYS if key in val_doc},
                    "interval_scope": "test sampling with fitted mapping and bin boundaries held fixed",
                })
                results[method] = result
                destination = Path(args.output) / run_dir.name
                destination.mkdir(parents=True, exist_ok=True)
                with (destination / f"calibration_{dataset}_{args.split}_{method}.json").open("w", encoding="utf-8") as handle:
                    json.dump(result, handle, indent=2, allow_nan=False)
                print(f"{dataset} | {run_dir.name} | {method}: "
                      f"mapping={result['mapping']['kind']} bins={result['bins_used']}/{args.n_bins} "
                      f"Brier={result['metrics']['brier']:.4f} test failures={result['test_failures']}/{result['test_count']}")
                if checks["unverified_protocol_fields"]:
                    print(f"  Protocol fields absent in legacy files: {checks['unverified_protocol_fields']}")
                completed += 1
            if results:
                destination = Path(args.output) / run_dir.name
                plot_results(dataset, args.split, run_dir.name, results, destination, args.dpi,
                             args.failure_threshold, args.confidence_level)
                print(f"[calibration] Saved figures to {destination}")
    if not completed:
        raise FileNotFoundError("No complete validation/test result pairs found; compute both splits first.")


if __name__ == "__main__":
    main()
