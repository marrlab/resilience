from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from NCA import BackboneNCA
from dataloader import build_split_dataloader


DATASET_DEFAULT_ROOTS = {
    "voc": [
        "datasets/voc/VOCdevkit/VOC2012",
        "datasets/VOC2012_train_val/VOC2012_train_val",
    ],
    "camvid": ["datasets/CamVid", "datasets/camvid"],
    "dsb2018": ["datasets/dsb2018"],
    "monuseg": [
        "datasets/MoNuSeg/MoNuSeg 2018 Training Data",
        "datasets/MoNuSeg",
    ],
    "rus": [
        "datasets/US/RUS",
        "datasets/us/RUS",
        "datasets/US/abdominal_US/abdominal_US/RUS",
    ],
    "nuinsseg": [
        "datasets/NuInsSeg",
        "datasets/nuinsseg",
    ],
    "isic2017": [
        "datasets/isic/isic2017_task1",
        "datasets/isic2017_task1",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an NCA checkpoint on a dataset split.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--dataset",
        choices=["voc", "camvid", "dsb2018", "monuseg", "rus", "nuinsseg", "isic2017"],
        required=True,
        help="Dataset name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument("--data_root", type=str, default=None, help="Optional dataset root override.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional resize (width height).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=None, help="Number of NCA steps during eval.")
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--samples", type=int, default=20, help="Number of qualitative samples.")
    parser.add_argument("--subset", type=int, default=None, help="Limit number of test samples.")
    parser.add_argument("--channel_n", type=int, default=None)
    parser.add_argument("--fire_rate", type=float, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--input_channels", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
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


def prepare_state(images: torch.Tensor, channel_n: int) -> torch.Tensor:
    batch, channels, height, width = images.shape
    state = images.permute(0, 2, 3, 1).contiguous()
    if channel_n > channels:
        pad = channel_n - channels
        zeros = torch.zeros(
            (batch, height, width, pad), device=images.device, dtype=images.dtype
        )
        state = torch.cat((state, zeros), dim=-1)
    else:
        state = state[..., :channel_n]
    return state


def select_logits(state: torch.Tensor, num_classes: int) -> torch.Tensor:
    logits = state[..., -num_classes:]
    return logits.permute(0, 3, 1, 2).contiguous()


def sanitize_targets(
    targets: torch.Tensor, num_classes: int, ignore_index: int
) -> torch.Tensor:
    invalid = targets >= num_classes
    if ignore_index >= 0:
        invalid &= targets != ignore_index
    if invalid.any():
        targets = targets.clone()
        replacement = ignore_index if ignore_index >= 0 else num_classes - 1
        targets[invalid] = replacement
    return targets


class SegmentationMetricTracker:
    def __init__(self, num_classes: int, ignore_index: int, device: torch.device) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.cm = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = torch.argmax(logits, dim=1)
        if self.ignore_index >= 0:
            valid = targets != self.ignore_index
            preds = preds[valid]
            targets = targets[valid]
        if targets.numel() == 0:
            return
        k = self.num_classes
        indices = targets.view(-1) * k + preds.view(-1)
        counts = torch.bincount(indices, minlength=k * k)
        self.cm += counts.double().view(k, k).to(self.device)

    def compute(self) -> Tuple[float, float]:
        cm = self.cm
        total = cm.sum()
        pixel_acc = (torch.trace(cm) / total).item() if total > 0 else 0.0
        diag = torch.diag(cm)
        denom_iou = cm.sum(1) + cm.sum(0) - diag
        valid = denom_iou > 0
        miou = (diag[valid] / denom_iou[valid]).mean().item() if valid.any() else 0.0
        return pixel_acc, miou


def compute_boundary_map(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[1:, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    boundary[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 2 * radius + 1, 2 * radius + 1), dtype=torch.float32)
    out = F.conv2d(tensor, kernel, padding=radius)
    return (out > 0).squeeze().numpy().astype(bool)


def boundary_f1_score(gt: np.ndarray, pred: np.ndarray, radius: int = 2) -> float:
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    gt_boundary = compute_boundary_map(gt)
    pred_boundary = compute_boundary_map(pred)
    if not gt_boundary.any() and not pred_boundary.any():
        return 1.0
    if not gt_boundary.any() or not pred_boundary.any():
        return 0.0
    gt_dil = dilate(gt_boundary, radius)
    pred_dil = dilate(pred_boundary, radius)
    precision = (pred_boundary & gt_dil).sum() / max(pred_boundary.sum(), 1)
    recall = (gt_boundary & pred_dil).sum() / max(gt_boundary.sum(), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def save_sample(
    output_dir: Path,
    index: int,
    image: torch.Tensor,
    target: torch.Tensor,
    pred: np.ndarray,
) -> None:
    sample_dir = output_dir / f"{index:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    image_np = np.clip(image_np, 0.0, 1.0)
    Image.fromarray((image_np * 255).astype(np.uint8)).save(sample_dir / "image.png")
    target_np = target.cpu().numpy().astype(np.uint8)
    Image.fromarray(target_np, mode="L").save(sample_dir / "mask_gt.png")
    Image.fromarray(pred.astype(np.uint8), mode="L").save(sample_dir / "mask_pred.png")


def extract_param(
    name: str,
    cli_value: Optional[float],
    ckpt_args: Optional[Dict[str, float]],
    default: Optional[float],
) -> float:
    if cli_value is not None:
        return cli_value
    if ckpt_args and name in ckpt_args:
        return ckpt_args[name]
    if default is not None:
        return default
    raise ValueError(f"Parameter '{name}' must be provided via CLI or checkpoint.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})

    channel_n = int(extract_param("channel_n", args.channel_n, ckpt_args, 64))
    fire_rate = float(extract_param("fire_rate", args.fire_rate, ckpt_args, 0.5))
    hidden_size = int(extract_param("hidden_size", args.hidden_size, ckpt_args, 128))
    input_channels = int(extract_param("input_channels", args.input_channels, ckpt_args, 3))
    eval_steps = int(
        extract_param("steps_max", args.steps, ckpt_args, ckpt_args.get("steps_max", 64))
    )

    image_size = tuple(args.image_size) if args.image_size else None
    data_root = resolve_data_root(args.dataset, args.data_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    loader, num_classes, class_names = build_split_dataloader(
        dataset_name=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
        pin_memory=True,
        root=data_root,
        ignore_index=args.ignore_index,
        subset=args.subset,
        shuffle=False,
    )

    model = BackboneNCA(
        channel_n=channel_n,
        fire_rate=fire_rate,
        device=device,
        hidden_size=hidden_size,
        input_channels=input_channels,
        steps_default=eval_steps,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tracker = SegmentationMetricTracker(num_classes, args.ignore_index, device)
    boundary_scores: List[float] = []
    qualitative_dir = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint_path.parent / f"eval_{args.dataset}_{args.split}"
    )
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = qualitative_dir / "qualitative"
    samples_dir.mkdir(exist_ok=True)

    saved_samples = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = sanitize_targets(
                targets.to(device, non_blocking=True), num_classes, args.ignore_index
            )
            state = prepare_state(images, channel_n)
            logits_state = model(state, steps=eval_steps)
            logits = select_logits(logits_state, num_classes)
            tracker.update(logits, targets)
            preds = torch.argmax(logits, dim=1)
            for b in range(preds.size(0)):
                pred_np = preds[b].cpu().numpy().astype(np.uint8)
                target_np = targets[b].cpu().numpy()
                target_np = np.where(target_np == args.ignore_index, 0, target_np)
                boundary_scores.append(boundary_f1_score(target_np, pred_np))
                if saved_samples < args.samples:
                    save_sample(samples_dir, saved_samples, images[b], targets[b], pred_np)
                    saved_samples += 1

    pixel_acc, miou = tracker.compute()
    boundary_mean = float(np.mean(boundary_scores)) if boundary_scores else 0.0

    metrics = {
        "pixel_accuracy": pixel_acc,
        "mean_iou": miou,
        "boundary_f1": boundary_mean,
    }

    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset": args.dataset,
        "split": args.split,
        "data_root": data_root,
        "seed": args.seed,
        "steps": eval_steps,
        "class_names": class_names,
        "metrics": metrics,
        "checkpoint_hparams": ckpt_args,
        "eval_hparams": {
            "channel_n": channel_n,
            "fire_rate": fire_rate,
            "hidden_size": hidden_size,
            "input_channels": input_channels,
            "ignore_index": args.ignore_index,
            "image_size": image_size,
            "batch_size": args.batch_size,
        },
    }

    metrics_path = qualitative_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"Evaluation complete. PixelAcc={pixel_acc:.4f}, mIoU={miou:.4f}, "
        f"BoundaryF1={boundary_mean:.4f}"
    )
    print(f"Saved qualitative outputs and metrics to {qualitative_dir}")


if __name__ == "__main__":
    main()
