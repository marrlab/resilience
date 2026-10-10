"""Regression coverage for perturbation scope, timing, pairing and saved scores."""
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
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ablation_sweep
import compute_uncertainty
import evaluate_uncertainty
from NCA import BackboneNCA
from evaluate import prepare_state, select_logits
from resilience import (
    experiment_seed, isolated_rng, perturb_and_recover, perturb_state,
    recover_state, result_method, rollout_metadata, validate_parameters,
)


class CountingModel:
    input_channels = 3

    def __init__(self):
        self.updates = 0

    def __call__(self, state, steps, conditioning):
        state = state.clone()
        for _ in range(steps):
            state[..., :3] = conditioning
            # Deliberately change RGB, to verify that recovery re-clamps it.
            state[..., :3] += 100
            state[..., 3:] += 1
            self.updates += 1
        state[..., :3] = conditioning
        return state


@contextlib.contextmanager
def workspace_tempdir():
    with tempfile.TemporaryDirectory(prefix="resilience_test_", dir=ROOT) as name:
        folder = Path(name).resolve()
        assert ROOT.resolve() in folder.parents
        yield folder


class ResilienceMechanismTests(unittest.TestCase):
    def test_noise_scope_and_matching_latent_draws(self):
        original = torch.ones(2, 4, 5, 6)
        saved = original.clone()
        latent = perturb_state(original, 3, 0.2, False, 13)
        rgb = perturb_state(original, 3, 0.2, True, 13)
        self.assertTrue(torch.equal(original, saved))
        self.assertTrue(torch.equal(latent[..., :3], original[..., :3]))
        self.assertFalse(torch.equal(rgb[..., :3], original[..., :3]))
        self.assertTrue(torch.equal(latent[..., 3:], rgb[..., 3:]))
        self.assertFalse(torch.equal(latent[..., 3:], original[..., 3:]))

    def test_exact_timing_and_second_injection_uses_recovered_state(self):
        model = CountingModel()
        baseline = torch.zeros(1, 2, 2, 5)
        observed = []
        endpoints = list(perturb_and_recover(
            model, baseline, 0.1, 3, "both", False, 9,
            lambda state, cycle, step: observed.append((cycle, step, state.clone())),
        ))
        self.assertEqual(model.updates, 6)
        self.assertEqual([(cycle, step) for cycle, step, _ in observed],
                         [(1, 0), (1, 1), (1, 2), (1, 3),
                          (2, 0), (2, 1), (2, 2), (2, 3)])
        first_noise = observed[0][2] - baseline
        second_noise = observed[4][2] - observed[3][2]
        self.assertFalse(torch.equal(first_noise, second_noise))
        # If the second cycle restarted at baseline it would lack these three updates.
        expected = baseline + first_noise + second_noise
        expected[..., 3:] += 6
        self.assertTrue(torch.allclose(endpoints[-1][1], expected))
        self.assertTrue(torch.equal(baseline, torch.zeros_like(baseline)))
        self.assertEqual(rollout_metadata("double", False, 64, 12)["injection_steps"], [64, 76])
        self.assertEqual(rollout_metadata("double", False, 64, 12)["evaluation_step"], 88)

    def test_rgb_is_fixed_to_each_cycles_noisy_input(self):
        model = CountingModel()
        baseline = torch.ones(1, 2, 2, 5)
        frames = []
        list(perturb_and_recover(
            model, baseline, 0.2, 2, "double", True, 71,
            lambda state, cycle, step: frames.append((cycle, step, state.clone())),
        ))
        first_rgb = frames[0][2][..., :3]
        second_rgb = frames[3][2][..., :3]
        self.assertFalse(torch.equal(first_rgb, baseline[..., :3]))
        self.assertFalse(torch.equal(first_rgb, second_rgb))
        for cycle, _, state in frames:
            self.assertTrue(torch.equal(state[..., :3], first_rgb if cycle == 1 else second_rgb))

    def test_single_double_and_both_match_with_stochastic_nca(self):
        torch.manual_seed(19)
        model = BackboneNCA(6, 0.5, torch.device("cpu"), hidden_size=8, input_channels=3)
        with torch.no_grad():
            model.fc1.weight.normal_(0, 0.1)
        state = torch.randn(2, 5, 5, 6)
        initial_rng = torch.random.get_rng_state().clone()
        with torch.no_grad():
            for rgb in (False, True):
                both = dict(perturb_and_recover(model, state, 0.02, 3, "both", rgb, 12))
                for mode in ("single", "double"):
                    standalone = dict(perturb_and_recover(model, state, 0.02, 3, mode, rgb, 12))
                    self.assertTrue(torch.equal(both[mode], standalone[mode]))
                frames = []
                recorded = dict(perturb_and_recover(
                    model, state, 0.02, 3, "both", rgb, 12,
                    lambda current, cycle, step: frames.append(current.clone()),
                ))
                self.assertTrue(torch.equal(recorded["double"], both["double"]))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), initial_rng))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_scope_pairing_and_rng_restoration(self):
        device = torch.device("cuda")
        model = BackboneNCA(6, 0.5, device, hidden_size=8, input_channels=3)
        with torch.no_grad():
            model.fc1.weight.normal_(0, 0.1)
        state = torch.randn(1, 5, 5, 6, device=device)
        before_cpu = torch.random.get_rng_state().clone()
        before_cuda = torch.cuda.get_rng_state().clone()
        with torch.no_grad():
            both = dict(perturb_and_recover(model, state, 0.02, 2, "both", True, 12))
            double = dict(perturb_and_recover(model, state, 0.02, 2, "double", True, 12))
        self.assertTrue(torch.equal(both["double"], double["double"]))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before_cpu))
        self.assertTrue(torch.equal(torch.cuda.get_rng_state(), before_cuda))

    def test_invalid_parameters_fail_before_rollout(self):
        for sigma, steps in [(-0.1, 2), (float("nan"), 2), (float("inf"), 2), (0.02, 0), (0.02, -1)]:
            with self.assertRaises(ValueError):
                validate_parameters(sigma, steps)

    def test_terminal_options_on_both_scripts(self):
        for module, required in [
            (compute_uncertainty, []),
            (ablation_sweep, ["--checkpoint", "dummy.pt", "--dataset", "dsb2018"]),
        ]:
            with patch.object(sys, "argv", ["script"] + required):
                args = module.parse_args()
                self.assertEqual(args.resilience_mode, "single")
                self.assertFalse(args.perturb_rgb)
            with patch.object(sys, "argv", ["script"] + required + [
                "--resilience_mode", "both", "--perturb_rgb",
            ]):
                args = module.parse_args()
                self.assertEqual(args.resilience_mode, "both")
                self.assertTrue(args.perturb_rgb)
            with patch.object(sys, "argv", ["script"] + required + ["--no_perturb_rgb"]):
                self.assertFalse(module.parse_args().perturb_rgb)


class ResiliencePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_fixture(self):
        torch.manual_seed(17)
        model = BackboneNCA(6, 0.5, torch.device("cpu"), hidden_size=8, input_channels=3)
        with torch.no_grad():
            model.fc1.weight.normal_(0, 0.15)
        model.eval()
        images = torch.rand(3, 3, 6, 6)
        masks = torch.zeros(3, 6, 6, dtype=torch.long)
        # Include good and bad examples for an actual AUC calculation.
        for offset in (0, 2):
            batch = images[offset:offset + 2]
            with torch.no_grad(), isolated_rng(experiment_seed(42, "dsb2018", offset, "baseline"), torch.device("cpu")):
                state = model(prepare_state(batch, 6), steps=3)
                pred = select_logits(state, 2).argmax(dim=1)
            masks[offset:offset + len(batch)] = pred
        masks[1] = 1 - masks[1]
        loader = DataLoader(TensorDataset(images, masks), batch_size=2, shuffle=False)
        args = argparse.Namespace(
            resilience_mode="both", perturb_rgb=False, resilience_noise=0.02,
            resilience_relax_steps=2, seed=42, batch_size=2, split="test",
            boundary_radius=3, ignore_index=255,
        )
        meta = [{"sample_id": f"case_{i}"} for i in range(3)]
        return model, loader, args, meta

    def compute(self, folder, model, loader, args, meta):
        checkpoint = folder / "best.pt"
        with contextlib.redirect_stdout(io.StringIO()):
            compute_uncertainty.generate_resilience(
                args, "dsb2018", checkpoint, model, loader, 2, meta, 3, torch.device("cpu")
            )
        result = {}
        for mode in ("single", "double"):
            method = result_method(mode, args.perturb_rgb)
            path = folder / f"uncertainty_dsb2018_test_{method}" / f"uncertainty_dsb2018_test_{method}.json"
            if path.exists():
                result[mode] = json.loads(path.read_text(encoding="utf-8"))
        return result

    def test_saved_endpoints_quality_and_frame_metadata(self):
        model, loader, args, meta = self.make_fixture()
        with workspace_tempdir() as folder:
            for rgb in (False, True):
                args.perturb_rgb = rgb
                outputs = self.compute(folder, model, loader, args, meta)
                for mode, summary in outputs.items():
                    count = 1 if mode == "single" else 2
                    self.assertEqual(summary["perturbation_count"], count)
                    self.assertEqual(summary["evaluation_step"], 3 + count * 2)
                    self.assertEqual(len(summary["records"]), 3)
                    for rec in summary["records"]:
                        baseline = np.load(rec["pred_mask"])
                        recovered = np.load(rec["recovered_pred_mask"])
                        self.assertAlmostEqual(
                            rec["unc_mean"], 1 - ablation_sweep.compute_iou(baseline, recovered)
                        )
                        self.assertTrue(np.array_equal(np.load(rec["unc_map"]), baseline != recovered))
                        frames = np.load(rec["resilience_frames"])
                        self.assertEqual(len(frames), 1 + count * 3)
                        self.assertEqual(len(rec["resilience_frame_labels"]), len(frames))
                        self.assertEqual(rec["injection_frame_indices"], [1] if count == 1 else [1, 4])
                        self.assertTrue(np.array_equal(frames[0], baseline))
                        self.assertTrue(np.array_equal(frames[-1], recovered))
                for single, double in zip(outputs["single"]["records"], outputs["double"]["records"]):
                    self.assertEqual(single["dice"], double["dice"])
                    self.assertTrue(np.array_equal(np.load(single["pred_mask"]), np.load(double["pred_mask"])))
                    self.assertTrue(np.array_equal(np.load(single["resilience_frames"]), np.load(double["resilience_frames"])[:4]))

    def test_separate_runs_match_both_even_on_later_batches(self):
        model, loader, args, meta = self.make_fixture()
        with workspace_tempdir() as folder:
            both = self.compute(folder / "both", model, loader, args, meta)
            for mode in ("single", "double"):
                args.resilience_mode = mode
                separate = self.compute(folder / mode, model, loader, args, meta)[mode]
                for a, b in zip(both[mode]["records"], separate["records"]):
                    self.assertEqual(a["unc_mean"], b["unc_mean"])
                    self.assertEqual(a["dice"], b["dice"])
                    self.assertTrue(np.array_equal(np.load(a["pred_mask"]), np.load(b["pred_mask"])))
                    self.assertTrue(np.array_equal(np.load(a["recovered_pred_mask"]), np.load(b["recovered_pred_mask"])))

    def test_evaluation_uses_exact_baseline_quality_without_external_labels(self):
        model, loader, args, meta = self.make_fixture()
        with workspace_tempdir() as folder:
            outputs = self.compute(folder, model, loader, args, meta)
            method = result_method("single", False)
            prepared = evaluate_uncertainty.load_and_prepare_records(
                folder, "dsb2018", "test", method, 0.2, False, 0.0
            )
            self.assertEqual(len(prepared), 3)
            self.assertEqual([rec["dice"] for rec in prepared],
                             [rec["dice"] for rec in outputs["single"]["records"]])
            metrics = evaluate_uncertainty.risk_coverage_metrics(prepared)
            self.assertIn("aurc", metrics)
            quality = {"records": [{"sample_id": f"case_{i}", "dice": -1} for i in range(3)]}
            (folder / "quality_dsb2018_test.json").write_text(json.dumps(quality), encoding="utf-8")
            prepared = evaluate_uncertainty.load_and_prepare_records(
                folder, "dsb2018", "test", method, 0.2, False, 0.0
            )
            self.assertTrue(all(rec["dice"] >= 0 for rec in prepared))

    def test_ablation_matches_computation_and_scores_whole_batch(self):
        model, loader, args, meta = self.make_fixture()
        with workspace_tempdir() as folder:
            checkpoint = folder / "best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "args": {"channel_n": 6, "fire_rate": 0.5, "hidden_size": 8,
                         "input_channels": 3, "steps_max": 3, "dropout_rate": 0.0},
            }, checkpoint)
            for rgb in (False, True):
                args.perturb_rgb = rgb
                outputs = self.compute(folder, model, loader, args, meta)
                output = folder / f"ablation_{rgb}.json"
                cli = ["ablation_sweep.py", "--checkpoint", str(checkpoint),
                       "--dataset", "dsb2018", "--batch_size", "2", "--device", "cpu",
                       "--sigmas", "0.02", "--t_primes", "2", "--resilience_mode", "both",
                       "--output", str(output), "--perturb_rgb" if rgb else "--no_perturb_rgb"]
                with patch.object(sys, "argv", cli), patch.object(
                    ablation_sweep, "build_split_dataloader", return_value=(loader, 2, [])
                ), contextlib.redirect_stdout(io.StringIO()):
                    ablation_sweep.main()
                grid = json.loads(output.read_text(encoding="utf-8"))["results_grid"]
                self.assertEqual(len(grid), 2)
                for mode in ("single", "double"):
                    entry = grid[ablation_sweep.grid_key(mode, rgb, 0.02, 2)]
                    self.assertEqual(entry["indices"], [0, 1, 2])
                    self.assertEqual(entry["resilience_scores"],
                                     [rec["unc_mean"] for rec in outputs[mode]["records"]])
                    self.assertEqual(entry["baseline_dice"],
                                     [rec["dice"] for rec in outputs[mode]["records"]])

    def test_command_line_computation_and_evaluation_dispatch(self):
        model, loader, args, meta = self.make_fixture()
        with workspace_tempdir() as folder:
            run_dir = folder / "dsb2018_fixture"
            run_dir.mkdir()
            checkpoint = run_dir / "best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "args": {"channel_n": 6, "fire_rate": 0.5, "hidden_size": 8,
                         "input_channels": 3, "steps_max": 3, "dropout_rate": 0.0},
            }, checkpoint)
            legacy_dir = run_dir / "uncertainty_dsb2018_test_resilience"
            legacy_dir.mkdir()
            sentinel = legacy_dir / "keep.txt"
            sentinel.write_text("original result", encoding="utf-8")
            methods = []
            for rgb in (False, True):
                cli = ["compute_uncertainty.py", "--runs_dir", str(folder),
                       "--datasets", "dsb2018", "--methods", "resilience",
                       "--resilience_mode", "both", "--resilience_relax_steps", "2",
                       "--device", "cpu", "--batch_size", "2",
                       "--perturb_rgb" if rgb else "--no_perturb_rgb"]
                with patch.object(sys, "argv", cli), patch.object(
                    compute_uncertainty, "build_split_dataloader", return_value=(loader, 2, [])
                ), contextlib.redirect_stdout(io.StringIO()):
                    compute_uncertainty.main()
                methods.extend(result_method(mode, rgb) for mode in ("single", "double"))
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "original result")
            output = folder / "comparison.json"
            cli = ["evaluate_uncertainty.py", "--runs_dir", str(folder),
                   "--datasets", "dsb2018", "--methods"] + methods + ["--output", str(output)]
            with patch.object(sys, "argv", cli), contextlib.redirect_stdout(io.StringIO()):
                evaluate_uncertainty.main()
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(summary), 4)
            for method in methods:
                self.assertEqual(summary[f"dsb2018:test:{method}"]["num_samples"], 3)
                for metric in ("aurc", "auroc", "auprc"):
                    self.assertIn(metric, summary[f"dsb2018:test:{method}"])

    def test_gif_export_accepts_injection_labels(self):
        from tools.visualize_uncertainty import create_resilience_gif
        from PIL import Image
        model, loader, args, meta = self.make_fixture()
        with workspace_tempdir() as folder:
            rec = self.compute(folder, model, loader, args, meta)["double"]["records"][0]
            path = folder / "double.gif"
            with contextlib.redirect_stdout(io.StringIO()):
                create_resilience_gif(
                    np.full((96, 96, 3), 128, dtype=np.uint8),
                    Path(rec["resilience_frames"]), path, rec["resilience_frame_labels"],
                )
            with Image.open(path) as animation:
                self.assertGreater(animation.n_frames, 1)
            with self.assertRaises(ValueError):
                create_resilience_gif(
                    np.zeros((96, 96, 3), dtype=np.uint8),
                    Path(rec["resilience_frames"]), folder / "bad.gif", ["Baseline"],
                )



if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
