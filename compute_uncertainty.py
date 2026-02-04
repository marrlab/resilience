from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from NCA import BackboneNCA
from dataloader import build_split_dataloader
from evaluate import (
    prepare_state,
    select_logits,
    sanitize_targets,
    DATASET_DEFAULT_ROOTS,
)
EPS = 1e-6
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute uncertainty maps and scalar scores for each validation/test image."
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
        help="Datasets to process.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--image_size", type=int, nargs=2, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", type=str, default="*best.pt")
    parser.add_argument("--boundary_radius", type=int, default=3)
    parser.add_argument("--save_png", action="store_true")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["single"],
        help="Uncertainty estimation methods to compute (default: single).",
    )
    parser.add_argument("--stoptime_samples", type=int, default=5)
    parser.add_argument("--stoptime_min_steps", type=int, default=None)
    parser.add_argument("--stoptime_max_steps", type=int, default=None)
    parser.add_argument("--stability_window", type=int, default=3)
    parser.add_argument(
        "--flicker_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for flicker mask binarization (default 0.5).",
    )
    parser.add_argument(
        "--flicker_window",
        type=int,
        default=None,
        help="Number of final steps to consider for flicker (default: entire rollout).",
    )
    parser.add_argument(
        "--resilience_noise",
        type=float,
        default=0.02,
        help="Stddev of Gaussian noise applied to resilience perturbation.",
    )
    parser.add_argument(
        "--resilience_relax_steps",
        type=int,
        default=12,
        help="Number of relaxation steps for resilience uncertainty.",
    )
    parser.add_argument(
        "--tta_max_transforms",
        type=int,
        default=None,
        help="Limit number of geometric transforms used for TTA (default: use all).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_data_root(dataset: str, override: Optional[str]) -> Optional[str]:
    if override:
        return override
    for candidate in DATASET_DEFAULT_ROOTS.get(dataset.lower(), []):
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None


def unwrap_dataset(dataset):
    indices = None
    current = dataset
    while isinstance(current, Subset):
        subset_indices = list(current.indices)
        if indices is None:
            indices = subset_indices
        else:
            indices = [indices[i] for i in subset_indices]
        current = current.dataset
    if indices is None:
        indices = list(range(len(current)))
    return current, indices


def _find_case_image(case_dir: Path) -> Optional[Path]:
    images_dir = case_dir / "images"
    if images_dir.exists():
        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{case_dir.name}{ext}"
            if candidate.exists():
                return candidate
        files = sorted(
            [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        if files:
            return files[0]
    return None


def get_sample_metadata(dataset, index: int) -> Dict[str, Optional[str]]:
    if hasattr(dataset, "samples"):
        entry = dataset.samples[index]
        if isinstance(entry, tuple):
            image_path, mask_path = entry
        else:
            image_path, mask_path = entry, None
        image_path = Path(image_path)
        meta = {
            "sample_id": image_path.stem,
            "image_path": str(image_path),
        }
        if mask_path is not None:
            meta["mask_path"] = str(mask_path)
        if hasattr(dataset, "multi_mask_paths"):
            extra = dataset.multi_mask_paths[index]
            if extra:
                meta["multi_mask_paths"] = [str(path) for path in extra]
        return meta
    if hasattr(dataset, "cases"):
        case_dir = Path(dataset.cases[index])
        image_path = _find_case_image(case_dir)
        meta = {"sample_id": case_dir.name, "case_dir": str(case_dir)}
        if image_path is not None:
            meta["image_path"] = str(image_path)
        return meta
    return {"sample_id": str(index)}


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 2 * radius + 1, 2 * radius + 1), dtype=torch.float32)
    out = F.conv2d(tensor, kernel, padding=radius)
    return (out > 0).squeeze().numpy().astype(bool)


def binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 2 * radius + 1, 2 * radius + 1), dtype=torch.float32)
    total = kernel.numel()
    out = F.conv2d(tensor, kernel, padding=radius)
    return (out >= total).squeeze().numpy().astype(bool)


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    dil = binary_dilate(mask, radius)
    ero = binary_erode(mask, radius)
    band = dil & (~ero)
    if not band.any():
        # fallback to dilated boundary from edges
        grad = np.zeros_like(mask, dtype=bool)
        grad[:-1, :] |= mask[:-1, :] != mask[1:, :]
        grad[1:, :] |= mask[:-1, :] != mask[1:, :]
        grad[:, :-1] |= mask[:, :-1] != mask[:, 1:]
        grad[:, 1:] |= mask[:, :-1] != mask[:, 1:]
        band = binary_dilate(grad.astype(np.uint8), radius)
    return band


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    if probs.size(1) == 1:
        p = torch.clamp(probs[:, 0], EPS, 1 - EPS)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    p = torch.clamp(probs, EPS, 1.0)
    return -(p * torch.log(p)).sum(dim=1)


@dataclass(frozen=True)
class _TTATransform:
    rotations: int = 0
    flip_h: bool = False
    flip_v: bool = False


def _apply_tta_transform(tensor: torch.Tensor, transform: _TTATransform) -> torch.Tensor:
    out = tensor
    if transform.rotations:
        out = torch.rot90(out, transform.rotations, dims=(-2, -1))
    if transform.flip_h:
        out = torch.flip(out, dims=(-1,))
    if transform.flip_v:
        out = torch.flip(out, dims=(-2,))
    return out


def _invert_tta_transform(tensor: torch.Tensor, transform: _TTATransform) -> torch.Tensor:
    out = tensor
    if transform.flip_v:
        out = torch.flip(out, dims=(-2,))
    if transform.flip_h:
        out = torch.flip(out, dims=(-1,))
    if transform.rotations:
        out = torch.rot90(out, (4 - transform.rotations) % 4, dims=(-2, -1))
    return out


def _default_tta_transforms() -> List[_TTATransform]:
    return [
        _TTATransform(rotations=0, flip_h=False, flip_v=False),
        _TTATransform(rotations=0, flip_h=True, flip_v=False),
        _TTATransform(rotations=0, flip_h=False, flip_v=True),
        _TTATransform(rotations=0, flip_h=True, flip_v=True),
        _TTATransform(rotations=1, flip_h=False, flip_v=False),
        _TTATransform(rotations=2, flip_h=False, flip_v=False),
        _TTATransform(rotations=3, flip_h=False, flip_v=False),
        _TTATransform(rotations=1, flip_h=True, flip_v=False),
    ]


def save_entropy_map(entropy: np.ndarray, path: Path, save_png: bool) -> None:
    np.save(path, entropy)
    if save_png:
        norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + EPS)
        Image.fromarray((norm * 255).astype(np.uint8)).save(path.with_suffix(".png"))


def generate_uncertainty(
    args: argparse.Namespace, dataset: str, checkpoint_path: Path, method: str
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    channel_n = int(ckpt_args.get("channel_n", 64))
    fire_rate = float(ckpt_args.get("fire_rate", 0.5))
    hidden_size = int(ckpt_args.get("hidden_size", 128))
    input_channels = int(ckpt_args.get("input_channels", 3))
    steps = args.steps or int(ckpt_args.get("steps_max", 64))
    if args.image_size:
        image_size = tuple(args.image_size)
    else:
        ckpt_size = ckpt_args.get("image_size")
        if isinstance(ckpt_size, (list, tuple)) and len(ckpt_size) == 2:
            image_size = (int(ckpt_size[0]), int(ckpt_size[1]))
        else:
            image_size = None
    data_root = resolve_data_root(dataset, ckpt_args.get("data_root"))

    loader, num_classes, _ = build_split_dataloader(
        dataset_name=dataset,
        split=args.split,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
        pin_memory=True,
        root=data_root,
        ignore_index=args.ignore_index,
        subset=None,
        shuffle=False,
    )
    base_dataset, order = unwrap_dataset(loader.dataset)
    sample_meta = [get_sample_metadata(base_dataset, idx) for idx in order]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    requires_model = method not in {"disagreement"}
    model: Optional[BackboneNCA] = None
    if requires_model:
        model = BackboneNCA(
            channel_n=channel_n,
            fire_rate=fire_rate,
            device=device,
            hidden_size=hidden_size,
            input_channels=input_channels,
            steps_default=steps,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

    output_dir = (
        checkpoint_path.parent / f"uncertainty_{dataset}_{args.split}_{method}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, float]] = []

    with torch.no_grad():
        if method == "single":
            for batch_idx, (images, targets) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                targets = sanitize_targets(
                    targets.to(device, non_blocking=True), num_classes, args.ignore_index
                )
                state = prepare_state(images, channel_n)
                logits_state = model(state, steps=steps)
                logits = select_logits(logits_state, num_classes)
                probs = torch.softmax(logits, dim=1)
                entropy = compute_entropy(probs).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
                prob_np = probs[:, 1 if probs.size(1) > 1 else 0].cpu().numpy()

                for i in range(entropy.shape[0]):
                    global_index = batch_idx * args.batch_size + i
                    meta = sample_meta[global_index] if global_index < len(sample_meta) else {}
                    sample_id = meta.get("sample_id", f"sample_{global_index}")
                    entropy_map = entropy[i]
                    pred_mask = preds[i]
                    prob_map = prob_np[i]
                    entropy_path = output_dir / f"{sample_id}_uncertainty.npy"
                    save_entropy_map(entropy_map, entropy_path, args.save_png)
                    pred_path = output_dir / f"{sample_id}_pred.npy"
                    np.save(pred_path, pred_mask)
                    prob_path = output_dir / f"{sample_id}_prob.npy"
                    np.save(prob_path, prob_map)

                    unc_mean = float(entropy_map.mean())
                    boundary = boundary_band(pred_mask.astype(bool), args.boundary_radius)
                    if boundary.any():
                        boundary_entropy = entropy_map[boundary]
                        unc_boundary_mean = float(boundary_entropy.mean())
                        unc_boundary_p95 = float(np.percentile(boundary_entropy, 95))
                    else:
                        unc_boundary_mean = unc_mean
                        unc_boundary_p95 = unc_mean

                    record = {
                        "index": global_index,
                        "sample_id": sample_id,
                        "unc_mean": unc_mean,
                        "unc_boundary_mean": unc_boundary_mean,
                        "unc_boundary_p95": unc_boundary_p95,
                        "unc_map": str(entropy_path),
                        "entropy_map": str(entropy_path),
                        "pred_mask": str(pred_path),
                        "prob_map": str(prob_path),
                        "method": method,
                    }
                    record.update(meta)
                    records.append(record)
        elif method == "stoptime":
            k_samples = args.stoptime_samples
            stop_min = args.stoptime_min_steps or ckpt_args.get("steps_min") or steps
            stop_max = args.stoptime_max_steps or ckpt_args.get("steps_max") or steps
            stop_min = int(stop_min)
            stop_max = max(int(stop_max), stop_min + 1)
            for batch_idx, (images, targets) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                targets = sanitize_targets(
                    targets.to(device, non_blocking=True), num_classes, args.ignore_index
                )
                prob_stack = []
                pred_stack = []
                for _ in range(k_samples):
                    steps_k = random.randint(stop_min, stop_max)
                    state = prepare_state(images, channel_n)
                    logits_state = model(state, steps=steps_k)
                    logits = select_logits(logits_state, num_classes)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    prob_stack.append(probs)
                    pred_stack.append(np.argmax(probs, axis=1))
                prob_stack = np.stack(prob_stack, axis=0)  # K x N x C x H x W
                pred_stack = np.stack(pred_stack, axis=0)
                mean_probs = prob_stack.mean(axis=0)
                var_probs = prob_stack.var(axis=0)

                if num_classes <= 1:
                    scalar_mean = mean_probs[:, 0]
                    scalar_var = var_probs[:, 0]
                    mean_mask = scalar_mean >= 0.5
                elif num_classes == 2:
                    scalar_mean = mean_probs[:, 1]
                    scalar_var = var_probs[:, 1]
                    mean_mask = scalar_mean >= 0.5
                else:
                    scalar_mean = mean_probs.mean(axis=1)
                    scalar_var = var_probs.sum(axis=1)
                    mean_mask = mean_probs.argmax(axis=1)

                for i in range(mean_probs.shape[0]):
                    global_index = batch_idx * args.batch_size + i
                    meta = sample_meta[global_index] if global_index < len(sample_meta) else {}
                    sample_id = meta.get("sample_id", f"sample_{global_index}")
                    var_map = scalar_var[i]
                    mean_prob_map = scalar_mean[i]
                    entropy_path = output_dir / f"{sample_id}_variance.npy"
                    np.save(entropy_path, var_map)
                    pred_path = output_dir / f"{sample_id}_pred.npy"
                    np.save(pred_path, mean_mask[i].astype(np.uint8))
                    prob_path = output_dir / f"{sample_id}_prob.npy"
                    np.save(prob_path, mean_prob_map)

                    unc_mean = float(var_map.mean())
                    boundary = boundary_band(mean_mask[i].astype(bool), args.boundary_radius)
                    if boundary.any():
                        boundary_values = var_map[boundary]
                        unc_boundary_mean = float(boundary_values.mean())
                        unc_boundary_p95 = float(np.percentile(boundary_values, 95))
                    else:
                        unc_boundary_mean = unc_mean
                        unc_boundary_p95 = unc_mean

                    record = {
                        "index": global_index,
                        "sample_id": sample_id,
                        "unc_mean": unc_mean,
                        "unc_boundary_mean": unc_boundary_mean,
                        "unc_boundary_p95": unc_boundary_p95,
                        "unc_map": str(entropy_path),
                        "variance_map": str(entropy_path),
                        "pred_mask": str(pred_path),
                        "prob_map": str(prob_path),
                        "method": method,
                    }
                    record.update(meta)
                    records.append(record)
        elif method == "stability":
            window = max(1, min(args.stability_window, steps - 1))
            for batch_idx, (images, targets) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                targets = sanitize_targets(
                    targets.to(device, non_blocking=True), num_classes, args.ignore_index
                )
                state = prepare_state(images, channel_n)
                prob_series: List[np.ndarray] = []
                for _ in range(steps):
                    state = model.update(state, fire_rate=None)
                    logits = select_logits(state, num_classes)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    prob_series.append(probs)
                prob_stack = np.stack(prob_series, axis=0)  # T x B x C x H x W
                final_probs = prob_stack[-1]
                if num_classes <= 1:
                    scalar_probs = prob_stack[:, :, 0]
                    final_mask = scalar_probs[-1] >= 0.5
                elif num_classes == 2:
                    scalar_probs = prob_stack[:, :, 1]
                    final_mask = scalar_probs[-1] >= 0.5
                else:
                    scalar_probs = prob_stack.max(axis=2)
                    final_mask = final_probs.argmax(axis=1)

                diff_maps = []
                for offset in range(1, window + 1):
                    curr = scalar_probs[-offset]
                    prev = scalar_probs[-offset - 1]
                    diff_maps.append(np.abs(curr - prev))
                diff_maps = np.mean(diff_maps, axis=0)

                for i in range(diff_maps.shape[0]):
                    global_index = batch_idx * args.batch_size + i
                    meta = sample_meta[global_index] if global_index < len(sample_meta) else {}
                    sample_id = meta.get("sample_id", f"sample_{global_index}")
                    unc_map = diff_maps[i]
                    entropy_path = output_dir / f"{sample_id}_stability.npy"
                    np.save(entropy_path, unc_map)
                    if num_classes <= 2:
                        pred_mask = final_mask[i].astype(np.uint8)
                    else:
                        pred_mask = (final_mask[i] > 0).astype(np.uint8)
                    pred_path = output_dir / f"{sample_id}_pred.npy"
                    np.save(pred_path, pred_mask)
                    prob_path = output_dir / f"{sample_id}_prob.npy"
                    if num_classes <= 1:
                        prob_map = scalar_probs[-1][i]
                    elif num_classes == 2:
                        prob_map = scalar_probs[-1][i]
                    else:
                        prob_map = final_probs[i].max(axis=0)
                    np.save(prob_path, prob_map)

                    unc_mean = float(unc_map.mean())
                    boundary = boundary_band(pred_mask.astype(bool), args.boundary_radius)
                    if boundary.any():
                        boundary_values = unc_map[boundary]
                        unc_boundary_mean = float(boundary_values.mean())
                        unc_boundary_p95 = float(np.percentile(boundary_values, 95))
                    else:
                        unc_boundary_mean = unc_mean
                        unc_boundary_p95 = unc_mean

                    record = {
                        "index": global_index,
                        "sample_id": sample_id,
                        "unc_mean": unc_mean,
                        "unc_boundary_mean": unc_boundary_mean,
                        "unc_boundary_p95": unc_boundary_p95,
                        "unc_map": str(entropy_path),
                        "pred_mask": str(pred_path),
                        "prob_map": str(prob_path),
                        "method": method,
                    }
                    record.update(meta)
                    records.append(record)
                    record.update(meta)
                    records.append(record)
        elif method == "flicker":
            thresh = args.flicker_threshold
            window = args.flicker_window
            for batch_idx, (images, targets) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                targets = sanitize_targets(
                    targets.to(device, non_blocking=True), num_classes, args.ignore_index
                )
                state = prepare_state(images, channel_n)
                prob_series: List[np.ndarray] = []
                for _ in range(steps):
                    state = model.update(state, fire_rate=None)
                    logits = select_logits(state, num_classes)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    prob_series.append(probs)
                prob_stack = np.stack(prob_series, axis=0)
                if window is not None and window < steps:
                    prob_stack = prob_stack[-window:]
                if num_classes <= 1:
                    scalar_probs = prob_stack[:, :, 0]
                elif num_classes == 2:
                    scalar_probs = prob_stack[:, :, 1]
                else:
                    scalar_probs = prob_stack.max(axis=2)
                binary_masks = (scalar_probs >= thresh).astype(np.uint8)
                flips = np.abs(np.diff(binary_masks, axis=0))
                flicker_map = flips.mean(axis=0)
                final_mask = binary_masks[-1]
                final_prob = scalar_probs[-1]

                for i in range(flicker_map.shape[0]):
                    global_index = batch_idx * args.batch_size + i
                    meta = sample_meta[global_index] if global_index < len(sample_meta) else {}
                    sample_id = meta.get("sample_id", f"sample_{global_index}")
                    unc_map = flicker_map[i]
                    entropy_path = output_dir / f"{sample_id}_flicker.npy"
                    np.save(entropy_path, unc_map)
                    pred_mask = final_mask[i].astype(np.uint8)
                    pred_path = output_dir / f"{sample_id}_pred.npy"
                    np.save(pred_path, pred_mask)
                    prob_path = output_dir / f"{sample_id}_prob.npy"
                    np.save(prob_path, final_prob[i])

                    unc_mean = float(unc_map.mean())
                    boundary = boundary_band(pred_mask.astype(bool), args.boundary_radius)
                    if boundary.any():
                        boundary_values = unc_map[boundary]
                        unc_boundary_mean = float(boundary_values.mean())
                        unc_boundary_p95 = float(np.percentile(boundary_values, 95))
                    else:
                        unc_boundary_mean = unc_mean
                        unc_boundary_p95 = unc_mean

                    record = {
                        "index": global_index,
                        "sample_id": sample_id,
                        "unc_mean": unc_mean,
                        "unc_boundary_mean": unc_boundary_mean,
                        "unc_boundary_p95": unc_boundary_p95,
                        "unc_map": str(entropy_path),
                        "pred_mask": str(pred_path),
                        "prob_map": str(prob_path),
                        "method": method,
                    }
                    record.update(meta)
                    records.append(record)
        elif method == "resilience":
            noise_std = args.resilience_noise
            relax_steps = args.resilience_relax_steps
            for batch_idx, (images, targets) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                targets = sanitize_targets(
                    targets.to(device, non_blocking=True), num_classes, args.ignore_index
                )
                state = prepare_state(images, channel_n)
                for _ in range(steps):
                    state = model.update(state, fire_rate=None)
                logits = select_logits(state, num_classes)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                if num_classes <= 1:
                    prob_map = probs[:, 0]
                    pred_mask = prob_map >= 0.5
                elif num_classes == 2:
                    prob_map = probs[:, 1]
                    pred_mask = prob_map >= 0.5
                else:
                    prob_map = probs.max(axis=1)
                    pred_mask = probs.argmax(axis=1)

                perturbed_state = state.clone()
                noise = torch.randn_like(perturbed_state) * noise_std
                perturbed_state = perturbed_state + noise
                for _ in range(relax_steps):
                    perturbed_state = model.update(perturbed_state, fire_rate=None)
                logits_relaxed = select_logits(perturbed_state, num_classes)
                probs_relaxed = torch.softmax(logits_relaxed, dim=1).cpu().numpy()
                if num_classes <= 1:
                    prob_relaxed = probs_relaxed[:, 0]
                    pred_relaxed = prob_relaxed >= 0.5
                elif num_classes == 2:
                    prob_relaxed = probs_relaxed[:, 1]
                    pred_relaxed = prob_relaxed >= 0.5
                else:
                    prob_relaxed = probs_relaxed.max(axis=1)
                    pred_relaxed = probs_relaxed.argmax(axis=1)

                for i in range(prob_map.shape[0]):
                    global_index = batch_idx * args.batch_size + i
                    meta = sample_meta[global_index] if global_index < len(sample_meta) else {}
                    sample_id = meta.get("sample_id", f"sample_{global_index}")
                    mask_a = pred_mask[i].astype(np.uint8)
                    mask_b = pred_relaxed[i].astype(np.uint8)
                    intersection = np.logical_and(mask_a, mask_b).sum()
                    union = np.logical_or(mask_a, mask_b).sum()
                    iou = intersection / union if union > 0 else 1.0
                    unc_resilience = 1.0 - iou
                    record = {
                        "index": global_index,
                        "sample_id": sample_id,
                        "unc_mean": float(unc_resilience),
                        "unc_boundary_mean": float(unc_resilience),
                        "unc_boundary_p95": float(unc_resilience),
                        "unc_map": str(output_dir / f"{sample_id}_resilience.npy"),
                        "pred_mask": str(output_dir / f"{sample_id}_pred.npy"),
                        "prob_map": str(output_dir / f"{sample_id}_prob.npy"),
                        "method": method,
                    }
                    np.save(output_dir / f"{sample_id}_resilience.npy", mask_a.astype(np.uint8))
                    np.save(output_dir / f"{sample_id}_pred.npy", mask_b)
                    np.save(output_dir / f"{sample_id}_prob.npy", prob_relaxed[i])
                    record.update(meta)
                    records.append(record)
        elif method == "tta":
            transforms = _default_tta_transforms()
            if args.tta_max_transforms is not None:
                limit = max(1, args.tta_max_transforms)
                transforms = transforms[:limit]
            if not transforms:
                raise ValueError("TTA requires at least one transform.")
            for batch_idx, (images, targets) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                targets = sanitize_targets(
                    targets.to(device, non_blocking=True), num_classes, args.ignore_index
                )
                aggregated_probs: List[torch.Tensor] = []
                for transform in transforms:
                    transformed = _apply_tta_transform(images, transform)
                    state = prepare_state(transformed, channel_n)
                    logits_state = model(state, steps=steps)
                    logits = select_logits(logits_state, num_classes)
                    probs = torch.softmax(logits, dim=1)
                    probs = _invert_tta_transform(probs, transform)
                    aggregated_probs.append(probs)
                stacked = torch.stack(aggregated_probs, dim=0)
                mean_probs = stacked.mean(dim=0)
                var_probs = stacked.var(dim=0, unbiased=False)
                entropy = compute_entropy(mean_probs).cpu().numpy()
                pred_idx = torch.argmax(mean_probs, dim=1, keepdim=True)
                preds = pred_idx.squeeze(1).cpu().numpy().astype(np.uint8)
                if num_classes <= 1:
                    prob_map = mean_probs[:, 0]
                    variance_map = var_probs[:, 0]
                elif num_classes == 2:
                    prob_map = mean_probs[:, 1]
                    variance_map = var_probs[:, 1]
                else:
                    prob_map = torch.gather(mean_probs, 1, pred_idx).squeeze(1)
                    variance_map = torch.gather(var_probs, 1, pred_idx).squeeze(1)
                prob_np = prob_map.cpu().numpy()
                variance_np = variance_map.cpu().numpy()
                for i in range(entropy.shape[0]):
                    global_index = batch_idx * args.batch_size + i
                    meta = sample_meta[global_index] if global_index < len(sample_meta) else {}
                    sample_id = meta.get("sample_id", f"sample_{global_index}")
                    entropy_map = entropy[i]
                    pred_mask = preds[i]
                    prob_map_i = prob_np[i]
                    variance_map_i = variance_np[i]
                    entropy_path = output_dir / f"{sample_id}_tta_entropy.npy"
                    save_entropy_map(entropy_map, entropy_path, args.save_png)
                    pred_path = output_dir / f"{sample_id}_pred.npy"
                    np.save(pred_path, pred_mask)
                    prob_path = output_dir / f"{sample_id}_prob.npy"
                    np.save(prob_path, prob_map_i)
                    var_path = output_dir / f"{sample_id}_tta_variance.npy"
                    np.save(var_path, variance_map_i)

                    unc_mean = float(entropy_map.mean())
                    boundary = boundary_band(pred_mask.astype(bool), args.boundary_radius)
                    if boundary.any():
                        boundary_entropy = entropy_map[boundary]
                        unc_boundary_mean = float(boundary_entropy.mean())
                        unc_boundary_p95 = float(np.percentile(boundary_entropy, 95))
                        boundary_variance = variance_map_i[boundary]
                        variance_boundary_mean = float(boundary_variance.mean())
                    else:
                        unc_boundary_mean = unc_mean
                        unc_boundary_p95 = unc_mean
                        variance_boundary_mean = float(variance_map_i.mean())

                    record = {
                        "index": global_index,
                        "sample_id": sample_id,
                        "unc_mean": unc_mean,
                        "unc_boundary_mean": unc_boundary_mean,
                        "unc_boundary_p95": unc_boundary_p95,
                        "unc_map": str(entropy_path),
                        "entropy_map": str(entropy_path),
                        "variance_map": str(var_path),
                        "variance_mean": float(variance_map_i.mean()),
                        "variance_boundary_mean": variance_boundary_mean,
                        "pred_mask": str(pred_path),
                        "prob_map": str(prob_path),
                        "method": method,
                        "tta_transform_count": len(transforms),
                    }
                    record.update(meta)
                    records.append(record)
        elif method == "disagreement":
            for batch_idx, (_, targets) in enumerate(loader):
                batch_size = targets.size(0)
                for i in range(batch_size):
                    global_index = batch_idx * args.batch_size + i
                    meta = sample_meta[global_index] if global_index < len(sample_meta) else {}
                    sample_id = meta.get("sample_id", f"sample_{global_index}")
                    extra_masks = meta.get("multi_mask_paths") or []
                    target_np = targets[i].cpu().numpy()
                    height, width = target_np.shape[-2], target_np.shape[-1]
                    if extra_masks:
                        mask_arrays = []
                        for path in extra_masks:
                            mask_img = (
                                Image.open(path)
                                .convert("L")
                                .resize((width, height), Image.NEAREST)
                            )
                            mask_arr = (np.array(mask_img, dtype=np.uint8) > 0).astype(
                                np.float32
                            )
                            mask_arrays.append(mask_arr)
                        if mask_arrays:
                            stack = np.stack(mask_arrays, axis=0)
                            prob_map = stack.mean(axis=0)
                        else:
                            prob_map = np.zeros((height, width), dtype=np.float32)
                    else:
                        prob_map = np.zeros((height, width), dtype=np.float32)
                    variance_map = prob_map * (1.0 - prob_map)
                    pred_mask = (prob_map >= 0.5).astype(np.uint8)
                    var_path = output_dir / f"{sample_id}_disagreement.npy"
                    np.save(var_path, variance_map)
                    pred_path = output_dir / f"{sample_id}_pred.npy"
                    np.save(pred_path, pred_mask)
                    prob_path = output_dir / f"{sample_id}_prob.npy"
                    np.save(prob_path, prob_map)
                    unc_mean = float(variance_map.mean())
                    boundary = boundary_band(pred_mask.astype(bool), args.boundary_radius)
                    if boundary.any():
                        boundary_values = variance_map[boundary]
                        unc_boundary_mean = float(boundary_values.mean())
                        unc_boundary_p95 = float(np.percentile(boundary_values, 95))
                    else:
                        unc_boundary_mean = unc_mean
                        unc_boundary_p95 = unc_mean
                    record = {
                        "index": global_index,
                        "sample_id": sample_id,
                        "unc_mean": unc_mean,
                        "unc_boundary_mean": unc_boundary_mean,
                        "unc_boundary_p95": unc_boundary_p95,
                        "unc_map": str(var_path),
                        "pred_mask": str(pred_path),
                        "prob_map": str(prob_path),
                        "method": method,
                        "annotator_count": len(extra_masks),
                    }
                    record.update(meta)
                    records.append(record)
        else:
            raise ValueError(f"Unsupported method '{method}'")

    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset": dataset,
        "split": args.split,
        "method": method,
        "records": records,
        "boundary_radius": args.boundary_radius,
    }
    with open(
        output_dir / f"uncertainty_{dataset}_{args.split}_{method}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)
    print(f"[{dataset}|{method}] Saved {len(records)} uncertainty entries to {output_dir}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    for dataset in args.datasets:
        for exp_dir in sorted(runs_dir.glob(f"{dataset}_*")):
            for checkpoint_path in sorted(exp_dir.glob(args.pattern)):
                for method in args.methods:
                    generate_uncertainty(args, dataset, checkpoint_path, method)


if __name__ == "__main__":
    main()
