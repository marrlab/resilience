from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from NCA import BackboneNCA
from dataloader import build_dataloaders

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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an NCA model for segmentation.")
    parser.add_argument(
        "--dataset",
        choices=["voc", "camvid", "dsb2018", "monuseg"],
        required=True,
        help="Dataset to use.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional override for dataset root directory.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--steps_min", type=int, default=32)
    parser.add_argument("--steps_max", type=int, default=64)
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=255,
        help="Label to ignore in the loss and metrics (set to -1 to disable).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Resize (width height). Skip to keep original resolution.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable pin_memory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp_name", type=str, default=None)

    parser.add_argument("--channel_n", type=int, default=64)
    parser.add_argument("--fire_rate", type=float, default=0.5)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument(
        "--input_channels",
        type=int,
        default=3,
        help="Number of channels from the input image copied into the NCA state.",
    )
    parser.add_argument(
        "--val_steps",
        type=int,
        default=None,
        help="Number of NCA steps during validation (defaults to --steps_max).",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=None,
        help="Optional gradient clipping norm.",
    )
    parser.add_argument(
        "--train_subset",
        type=int,
        default=None,
        help="Limit number of training samples (e.g., 4 for overfitting checks).",
    )
    parser.add_argument(
        "--val_subset",
        type=int,
        default=None,
        help="Limit number of validation samples.",
    )
    parser.add_argument(
        "--eval_on_train",
        action="store_true",
        help="Run validation metrics on the training split instead of the validation split.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
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


def prepare_nca_state(images: torch.Tensor, channel_n: int) -> torch.Tensor:
    """Convert BCHW tensors to BHWC state and pad hidden channels if needed."""
    if channel_n <= 0:
        raise ValueError("channel_n must be positive.")
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
    """Extract logits from the last channels and return BCHW tensors."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if state.size(-1) < num_classes:
        raise ValueError(
            f"State has only {state.size(-1)} channels, cannot slice {num_classes} logits."
        )
    logits = state[..., -num_classes:]
    return logits.permute(0, 3, 1, 2).contiguous()


def sanitize_targets(
    targets: torch.Tensor, num_classes: int, ignore_index: int
) -> torch.Tensor:
    """Clamp unexpected labels to ignore_index (or last class if ignore disabled)."""
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
        with torch.no_grad():
            k = self.num_classes
            indices = targets.view(-1) * k + preds.view(-1)
            counts = torch.bincount(indices, minlength=k * k)
            self.cm += counts.double().view(k, k).to(self.device)

    def compute(self) -> Tuple[float, float, float]:
        cm = self.cm
        total = cm.sum()
        pixel_acc = (torch.trace(cm) / total).item() if total > 0 else 0.0
        diag = torch.diag(cm)
        denom_iou = cm.sum(1) + cm.sum(0) - diag
        valid_iou = denom_iou > 0
        miou = (
            (diag[valid_iou] / denom_iou[valid_iou]).mean().item()
            if valid_iou.any()
            else 0.0
        )
        denom_dice = cm.sum(1) + cm.sum(0)
        valid_dice = denom_dice > 0
        dice = (
            (2.0 * diag[valid_dice] / denom_dice[valid_dice]).mean().item()
            if valid_dice.any()
            else 0.0
        )
        return pixel_acc, miou, dice


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    channel_n: int,
    steps_min: int,
    steps_max: int,
    num_classes: int,
    grad_clip: Optional[float],
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = sanitize_targets(
            targets.to(device, non_blocking=True), num_classes, criterion.ignore_index
        )
        steps = random.randint(steps_min, steps_max)
        state = prepare_nca_state(images, channel_n)
        logits_state = model(state, steps=steps)
        logits = select_logits(logits_state, num_classes)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    channel_n: int,
    steps: int,
    num_classes: int,
    ignore_index: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    tracker = SegmentationMetricTracker(num_classes, ignore_index, device)
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = sanitize_targets(
                targets.to(device, non_blocking=True), num_classes, criterion.ignore_index
            )
            state = prepare_nca_state(images, channel_n)
            logits_state = model(state, steps=steps)
            logits = select_logits(logits_state, num_classes)
            loss = criterion(logits, targets)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            tracker.update(logits, targets)
    pixel_acc, miou, dice = tracker.compute()
    return {
        "loss": total_loss / max(total_samples, 1),
        "pixel_acc": pixel_acc,
        "miou": miou,
        "dice": dice,
    }


def main() -> None:
    args = parse_args()
    if args.steps_min > args.steps_max:
        raise ValueError("--steps_min must be <= --steps_max.")
    if args.input_channels > args.channel_n:
        raise ValueError("--input_channels cannot exceed --channel_n.")
    set_seed(args.seed)
    image_size = tuple(args.image_size) if args.image_size else None
    pin_memory = not args.no_pin_memory
    data_root = resolve_data_root(args.dataset, args.data_root)
    train_loader, val_loader, num_classes, _ = build_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        root=data_root,
        ignore_index=args.ignore_index,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
    )
    if args.eval_on_train:
        val_loader = train_loader
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = BackboneNCA(
        channel_n=args.channel_n,
        fire_rate=args.fire_rate,
        device=device,
        hidden_size=args.hidden_size,
        input_channels=args.input_channels,
        steps_default=args.steps_max,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    val_steps = args.val_steps or args.steps_max

    exp_name = (
        args.exp_name
        if args.exp_name
        else f"{args.dataset}_nca_c{args.channel_n}_s{args.steps_min}-{args.steps_max}"
    )
    exp_dir = Path("runs") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = exp_dir / "metrics.json"
    checkpoints_dir = exp_dir

    best_miou = 0.0
    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            channel_n=args.channel_n,
            steps_min=args.steps_min,
            steps_max=args.steps_max,
            num_classes=num_classes,
            grad_clip=args.grad_clip,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            channel_n=args.channel_n,
            steps=val_steps,
            num_classes=num_classes,
            ignore_index=args.ignore_index,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "pixel_acc": val_metrics["pixel_acc"],
                "miou": val_metrics["miou"],
                "dice": val_metrics["dice"],
            }
        )
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        is_best = val_metrics["miou"] > best_miou
        if is_best:
            best_miou = val_metrics["miou"]
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_miou": best_miou,
            "args": vars(args),
        }
        torch.save(checkpoint, checkpoints_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, checkpoints_dir / "best.pt")

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"pixel_acc={val_metrics['pixel_acc']:.4f} | "
            f"mIoU={val_metrics['miou']:.4f} | "
            f"dice={val_metrics['dice']:.4f} | "
            f"best_mIoU={best_miou:.4f}"
        )


if __name__ == "__main__":
    main()
