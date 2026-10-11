"""Statistical and file-format regression checks for calibration plotting."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import plot_calibration as calibration


@contextlib.contextmanager
def workspace_tempdir():
    with tempfile.TemporaryDirectory(prefix="calibration_test_", dir=ROOT) as name:
        folder = Path(name).resolve()
        assert ROOT.resolve() in folder.parents
        yield folder


def arguments(**overrides):
    with patch.object(sys, "argv", ["plot_calibration.py"]):
        args = calibration.parse_args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def rows(prefix, count, seed=1, grouped=False):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, count)
    failures = rng.random(count) < (0.1 + 0.8 * u)
    result = []
    for i in range(count):
        row = {"sample_id": f"{prefix}_{i}", "index": i, "u": float(u[i]),
               "dice": 0.5 if failures[i] else 0.9, "failure": int(failures[i])}
        if grouped:
            row["group_id"] = f"{prefix}_patient_{i // 4}"
        result.append(row)
    return result


def write_results(folder, split, method, entries, **metadata):
    path = folder / f"uncertainty_dsb2018_{split}_{method}" / f"uncertainty_dsb2018_{split}_{method}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "dataset": "dsb2018", "split": split, "method": method,
        "checkpoint": str(folder / "best.pt"), "steps": 64,
        "resilience_noise": 0.02, "resilience_relax_steps": 12,
        "resilience_mode": "double" if "double" in method else "single",
        "perturb_rgb": method.endswith("_rgb"), "input_conditioning": "original_rgb",
        "records": [
            {**row, "unc_boundary_mean": row["u"], "quality_source": "resilience_baseline"}
            for row in entries
        ],
        **metadata,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class CalibrationTests(unittest.TestCase):
    def test_bin_points_match_sklearn_and_keep_ties_together(self):
        y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
        p = np.array([0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.7, 0.7, 0.7, 0.9])
        for strategy in ("quantile", "uniform"):
            bins = calibration.calibration_bins(y, p, 10, strategy)
            fraction, mean = calibration_curve(y, p, n_bins=10, strategy=strategy)
            np.testing.assert_allclose([b["fraction_failed"] for b in bins], fraction)
            np.testing.assert_allclose([b["mean_probability"] for b in bins], mean)
            self.assertEqual(sum(b["count"] for b in bins), len(y))
            self.assertEqual(sum(b["failures"] for b in bins), int(y.sum()))
        constant = calibration.calibration_bins(y, np.full(len(y), 0.3), 10)
        self.assertEqual(len(constant), 1)
        self.assertEqual(constant[0]["count"], len(y))

    def test_wilson_intervals_are_nonzero_at_zero_or_all_failures(self):
        low, high = calibration.wilson_interval(0, 10, 0.95)
        self.assertEqual(low, 0)
        self.assertGreater(high, 0)
        low, high = calibration.wilson_interval(10, 10, 0.95)
        self.assertGreater(low, 0)
        self.assertAlmostEqual(high, 1)

    def test_group_bootstrap_respects_perfectly_correlated_patients(self):
        y = np.array([0] * 20 + [1] * 20)
        p = np.full(40, 0.5)
        groups = np.array(["patient_a"] * 20 + ["patient_b"] * 20)
        bins = calibration.calibration_bins(y, p, 1, groups=groups, bootstrap_samples=200)
        self.assertEqual(bins[0]["group_count"], 2)
        self.assertEqual(bins[0]["ci_method"], "cluster_bootstrap_fixed_bins")
        self.assertEqual(bins[0]["ci_lower"], 0)
        self.assertEqual(bins[0]["ci_upper"], 1)
        with self.assertRaisesRegex(ValueError, "two independent"):
            calibration.calibration_bins(y, p, groups=np.array(["one"] * 40))

    def test_logistic_mapping_coefficients_reproduce_probabilities(self):
        data = rows("val", 200)
        u = np.array([row["u"] for row in data])
        y = np.array([row["failure"] for row in data])
        model = calibration.fit_mapping("logistic", u, y)
        params = calibration.mapping_parameters(model)
        computed = 1 / (1 + np.exp(-(params["a"] * u + params["b"])))
        np.testing.assert_allclose(computed, calibration.predict_probability(model, u))
        self.assertGreater(params["a"], 0)

    def test_mapping_and_cv_selection_do_not_depend_on_test_labels(self):
        val = rows("val", 200, 4)
        test = rows("test", 60, 5)
        args = arguments(calibrator="auto")
        first = calibration.evaluate_configuration(val, test, args)
        flipped = [{**row, "failure": 1 - row["failure"], "dice": 1 - row["dice"]} for row in test]
        second = calibration.evaluate_configuration(val, flipped, args)
        self.assertEqual(first["mapping"], second["mapping"])
        self.assertEqual(first["selection"], second["selection"])
        np.testing.assert_array_equal([r["failure_probability"] for r in first["records"]],
                                      [r["failure_probability"] for r in second["records"]])

    def test_auto_falls_back_when_validation_is_small(self):
        data = rows("val", 30, 1)
        u = np.array([row["u"] for row in data])
        y = np.array([row["failure"] for row in data])
        _, selection = calibration.select_mapping(u, y, None, arguments(calibrator="auto"))
        self.assertEqual(selection["selected"], "logistic")
        self.assertEqual(selection["cv_scores"], {})
        with self.assertRaisesRegex(ValueError, "both failures and successes"):
            calibration.fit_mapping("logistic", u, np.zeros(len(u), dtype=int))

    def test_cross_validation_never_splits_a_patient(self):
        u = np.arange(120, dtype=float) / 120
        groups = np.repeat(np.arange(30).astype(str), 4)
        y = np.tile([0, 0, 1, 1], 30)
        fitting = calibration.fit_mapping
        training_sets = []
        def capture(kind, score, failures, c):
            training_sets.append(np.rint(score * 120).astype(int))
            return fitting(kind, score, failures, c)
        with patch.object(calibration, "fit_mapping", side_effect=capture):
            _, selection = calibration.select_mapping(u, y, groups, arguments(calibrator="auto"))
        self.assertEqual(selection["cv_folds_used"], 5)
        for training in training_sets:
            selected = set(training)
            for first in range(0, 120, 4):
                self.assertIn(sum(i in selected for i in range(first, first + 4)), (0, 4))

    def test_loader_uses_original_dice_and_strict_failure_threshold(self):
        method = "resilience_single_latent"
        with workspace_tempdir() as folder:
            entries = [
                {"sample_id": "a", "index": 0, "u": 0.2, "dice": 0.8},
                {"sample_id": "b", "index": 1, "u": 0.4, "dice": 0.79},
            ]
            write_results(folder, "val", method, entries)
            (folder / "quality_dsb2018_val.json").write_text(
                json.dumps({"records": [{"sample_id": "a", "dice": 0}, {"sample_id": "b", "dice": 1}]}),
                encoding="utf-8")
            _, loaded, _ = calibration.load_image_results(folder, "dsb2018", "val", method,
                                                          "unc_boundary_mean", 0.8)
            self.assertEqual([r["dice"] for r in loaded], [0.8, 0.79])
            self.assertEqual([r["failure"] for r in loaded], [0, 1])
            group_map = {"dsb2018": {"val": {"a": "patient_1", "b": "patient_2"}}}
            _, loaded, _ = calibration.load_image_results(folder, "dsb2018", "val", method,
                                                          "unc_boundary_mean", 0.8, group_mapping=group_map)
            self.assertEqual(loaded[1]["group_id"], "patient_2")

    def test_legacy_quality_join_and_bad_records(self):
        method = "resilience"
        with workspace_tempdir() as folder:
            path = write_results(folder, "val", method, rows("val", 10))
            document = json.loads(path.read_text(encoding="utf-8"))
            for entry in document["records"]:
                entry.pop("quality_source")
                entry.pop("dice")
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing original"):
                calibration.load_image_results(folder, "dsb2018", "val", method, "unc_boundary_mean", 0.8)
            quality = {"records": [{"sample_id": f"val_{i}", "dice": 0.6 if i % 2 else 0.9} for i in range(10)]}
            (folder / "quality_dsb2018_val.json").write_text(json.dumps(quality), encoding="utf-8")
            _, loaded, _ = calibration.load_image_results(folder, "dsb2018", "val", method, "unc_boundary_mean", 0.8)
            self.assertEqual(len(loaded), 10)
            document["records"].append(document["records"][0])
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                calibration.load_image_results(folder, "dsb2018", "val", method, "unc_boundary_mean", 0.8)

    def test_overlap_and_protocol_mismatch_are_rejected(self):
        val, test = rows("val", 20), rows("test", 20)
        docs = {"method": "resilience_single_latent", "resilience_noise": 0.02}
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            calibration.validate_splits(docs, {**docs, "resilience_noise": 0.1}, val, test)
        with self.assertRaisesRegex(ValueError, "data overlap"):
            calibration.validate_splits(docs, docs, val, val)
        val[0]["group_id"] = "shared_patient"
        test[0]["group_id"] = "shared_patient"
        with self.assertRaisesRegex(ValueError, "share patient"):
            calibration.validate_splits(docs, docs, val, test)
        with self.assertRaisesRegex(ValueError, "must be different"):
            calibration.validate_args(arguments(calibration_split="test"))

    def test_cli_exports_separate_runs_figures_and_per_image_json(self):
        methods = ["resilience_single_latent", "resilience_double_latent"]
        with workspace_tempdir() as folder:
            runs = folder / "runs"
            for run in ("dsb2018_a", "dsb2018_b"):
                for method in methods:
                    write_results(runs / run, "val", method, rows("val", 140, 11))
                    write_results(runs / run, "test", method, rows("test", 70, 12))
            output = folder / "plots"
            cli = ["plot_calibration.py", "--datasets", "dsb2018", "--runs_dir", str(runs),
                   "--methods"] + methods + ["--calibrator", "auto", "--output", str(output)]
            with patch.object(sys, "argv", cli), contextlib.redirect_stdout(io.StringIO()):
                calibration.main()
            for run in ("dsb2018_a", "dsb2018_b"):
                self.assertTrue((output / run / "dsb2018_test_calibration.png").exists())
                self.assertTrue((output / run / "dsb2018_test_calibration.pdf").exists())
                for method in methods:
                    result = json.loads((output / run / f"calibration_dsb2018_test_{method}.json").read_text(encoding="utf-8"))
                    self.assertEqual(result["calibration_count"], 140)
                    self.assertEqual(result["test_count"], 70)
                    self.assertEqual(len(result["records"]), 70)
                    self.assertEqual(sum(b["count"] for b in result["bins"]), 70)
                    self.assertEqual(result["split_checks"]["unverified_protocol_fields"], [])
                    self.assertTrue(all(0 <= r["failure_probability"] <= 1 for r in result["records"]))
                    self.assertIn("auroc_uncertainty", result["metrics"])

    def test_group_bins_with_only_one_patient_do_not_claim_precision(self):
        failures = np.array([0] * 10 + [1] * 10)
        probabilities = np.array([0.1] * 10 + [0.9] * 10)
        groups = np.array(["patient_a"] * 10 + ["patient_b"] * 10)
        bins = calibration.calibration_bins(failures, probabilities, 2,
                                             groups=groups, bootstrap_samples=200)
        for row in bins:
            self.assertEqual(row["group_count"], 1)
            self.assertIsNone(row["ci_lower"])
            self.assertIsNone(row["ci_upper"])



if __name__ == "__main__":
    unittest.main()
