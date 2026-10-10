"""Sweep noise magnitude and recovery length for single/double resilience."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from NCA import BackboneNCA
from dataloader import build_split_dataloader
from evaluate import (
    DATASET_DEFAULT_ROOTS, prepare_state, resolve_data_root,
    select_logits, sanitize_targets,
)
from resilience import (
    add_resilience_arguments, experiment_seed, isolated_rng, perturb_and_recover,
    result_method, rollout_metadata, selected_modes, validate_parameters,
)


def binarize(prob_map, threshold=0.5):
    return (prob_map > threshold).astype(np.uint8)


def compute_dice(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    return float(2.0 * intersection / total) if total else 1.0


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union else 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep noise sigma / recovery steps and score error-detection AUROC."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULT_ROOTS), required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=None, help="NCA steps to reach the original state S_T.")
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--dice_error_threshold", type=float, default=0.8)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument(
        "--t_primes", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24],
        help="Recovery steps after each injection; double has two recovery stages.",
    )
    add_resilience_arguments(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Optional results JSON path.")
    return parser.parse_args()


def grid_key(mode, perturb_rgb, sigma, t_prime):
    return f"{result_method(mode, perturb_rgb)}_sigma_{sigma}_T_{t_prime}"


def main() -> None:
    args = parse_args()
    for sigma in args.sigmas:
        for t_prime in args.t_primes:
            validate_parameters(sigma, t_prime)
    modes = selected_modes(args.resilience_mode)
    checkpoint_path = Path(args.checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    steps = int(args.steps if args.steps is not None else ckpt_args.get("steps_max", 64))
    if steps < 1:
        raise ValueError("--steps must be at least 1.")
    model = BackboneNCA(
        channel_n=int(ckpt_args.get("channel_n", 64)),
        fire_rate=float(ckpt_args.get("fire_rate", 0.5)), device=device,
        hidden_size=int(ckpt_args.get("hidden_size", 128)),
        input_channels=int(ckpt_args.get("input_channels", 3)), steps_default=steps,
        dropout_rate=float(ckpt_args.get("dropout_rate", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data_root = resolve_data_root(args.dataset, args.data_root)
    loader, num_classes, _ = build_split_dataloader(
        dataset_name=args.dataset, split=args.split, batch_size=args.batch_size,
        image_size=tuple(args.image_size), num_workers=args.num_workers,
        pin_memory=True, root=data_root, ignore_index=args.ignore_index,
        subset=args.subset, shuffle=False,
    )
    if num_classes != 2:
        raise ValueError("Resilience ablations support binary (two-logit) segmentation.")

    results_grid = {
        grid_key(mode, args.perturb_rgb, sigma, t_prime): {
            **rollout_metadata(mode, args.perturb_rgb, steps, t_prime),
            "resilience_noise": sigma, "resilience_scores": [], "is_error": [],
            "indices": [], "baseline_dice": [],
        }
        for mode in modes for sigma in args.sigmas for t_prime in args.t_primes
    }
    print(
        f"Starting ablation: dataset={args.dataset}, split={args.split}, "
        f"mode={args.resilience_mode}, perturb_rgb={args.perturb_rgb}"
    )
    print(f"Total samples to process: {len(loader.dataset)}")

    global_offset = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = sanitize_targets(
                targets.to(device, non_blocking=True), num_classes, args.ignore_index
            ).cpu().numpy()
            state = prepare_state(images, model.channel_n)
            conditioning = state[..., :model.input_channels].clone()
            baseline_seed = experiment_seed(args.seed, args.dataset, global_offset, "baseline")
            resilience_seed = experiment_seed(args.seed, args.dataset, global_offset, "perturbation")
            with isolated_rng(baseline_seed, device):
                baseline = model(state, steps=steps, conditioning=conditioning)
            original_masks = F.softmax(select_logits(baseline, num_classes), dim=1).argmax(
                dim=1
            ).cpu().numpy().astype(np.uint8)
            baseline_dice = []
            for original, target in zip(original_masks, targets):
                valid = target != args.ignore_index
                baseline_dice.append(compute_dice(original.astype(bool) & valid, (target > 0) & valid))
            errors = [int(dice < args.dice_error_threshold) for dice in baseline_dice]

            for sigma in args.sigmas:
                for t_prime in args.t_primes:
                    for mode, recovered in perturb_and_recover(
                        model, baseline, sigma, t_prime, args.resilience_mode,
                        args.perturb_rgb, resilience_seed,
                    ):
                        recovered_masks = F.softmax(
                            select_logits(recovered, num_classes), dim=1
                        ).argmax(dim=1).cpu().numpy().astype(np.uint8)
                        entry = results_grid[grid_key(mode, args.perturb_rgb, sigma, t_prime)]
                        entry["resilience_scores"].extend(
                            1.0 - compute_iou(original, final)
                            for original, final in zip(original_masks, recovered_masks)
                        )
                        entry["is_error"].extend(errors)
                        entry["baseline_dice"].extend(baseline_dice)
                        entry["indices"].extend(range(global_offset, global_offset + len(original_masks)))
            global_offset += len(original_masks)
    print(f"Processed {global_offset} images.")

    summary = {}
    for key, entry in results_grid.items():
        labels = np.asarray(entry["is_error"])
        scores = np.asarray(entry["resilience_scores"])
        # AUC is undefined when there are no samples or only one quality class.
        auroc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else None
        summary[key] = auroc
        displayed = f"{auroc:.4f}" if auroc is not None else "undefined (one/no class)"
        print(f"{key} | AUROC: {displayed}")
    valid_results = {key: value for key, value in summary.items() if value is not None}
    best_config = max(valid_results, key=valid_results.get) if valid_results else None
    best_auroc = valid_results[best_config] if best_config else None
    print(f"Best configuration: {best_config} | AUROC: {best_auroc}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": str(checkpoint_path), "dataset": args.dataset,
                "split": args.split, "data_root": data_root,
                "seed": args.seed, "batch_size": args.batch_size, "steps": steps,
                "resilience_mode": args.resilience_mode, "perturb_rgb": args.perturb_rgb,
                "dice_error_threshold": args.dice_error_threshold,
                "sigmas": args.sigmas, "t_primes": args.t_primes,
                "auroc_grid": summary, "results_grid": results_grid,
                "best_config": best_config, "best_auroc": best_auroc,
            }, f, indent=2)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
