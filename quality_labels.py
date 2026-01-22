from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Subset

from NCA import BackboneNCA
from dataloader import build_split_dataloader
from evaluate import (
    prepare_state,
    select_logits,
    sanitize_targets,
    boundary_f1_score,
    DATASET_DEFAULT_ROOTS,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate quality labels for all runs/datasets."
    )
    parser.add_argument(
        "--runs_dir", type=str, default="runs", help="Directory containing experiment runs."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["dsb2018", "monuseg", "rus"],
        help="Datasets to evaluate (default: dsb2018 monuseg rus).",
    )
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override dataset root (applied to all datasets).",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, nargs=2, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pattern", type=str, default="*best.pt", help="Glob pattern for checkpoints."
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


def dice_coefficient(target: np.ndarray, pred: np.ndarray) -> float:
    target_bin = target > 0
    pred_bin = pred > 0
    intersection = np.logical_and(target_bin, pred_bin).sum()
    total = target_bin.sum() + pred_bin.sum()
    return (2 * intersection) / total if total > 0 else 1.0


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
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
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
        return meta
    if hasattr(dataset, "cases"):
        case_dir = Path(dataset.cases[index])
        image_path = _find_case_image(case_dir)
        meta = {"sample_id": case_dir.name, "case_dir": str(case_dir)}
        if image_path is not None:
            meta["image_path"] = str(image_path)
        return meta
    return {"sample_id": str(index)}


def generate_quality_labels(
    args: argparse.Namespace,
    dataset: str,
    checkpoint_path: Path,
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

    data_root = resolve_data_root(dataset, args.data_root or ckpt_args.get("data_root"))

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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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

    base_dataset, order = unwrap_dataset(loader.dataset)
    sample_meta = [get_sample_metadata(base_dataset, idx) for idx in order]

    records: List[Dict[str, float]] = []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = sanitize_targets(
                targets.to(device, non_blocking=True), num_classes, args.ignore_index
            )
            state = prepare_state(images, channel_n)
            logits_state = model(state, steps=steps)
            logits = select_logits(logits_state, num_classes)
            preds = torch.argmax(logits, dim=1)
            for i in range(preds.size(0)):
                pred_np = preds[i].cpu().numpy().astype(np.uint8)
                target_np = targets[i].cpu().numpy()
                target_clean = np.where(target_np == args.ignore_index, 0, target_np)
                dice = dice_coefficient(target_clean, pred_np)
                boundary = boundary_f1_score(target_clean, pred_np)
                bad = float(dice < args.iou_threshold)
                global_index = batch_idx * args.batch_size + i
                meta = sample_meta[global_index] if global_index < len(sample_meta) else {}
                records.append(
                    {
                        "index": global_index,
                        "dice": dice,
                        "boundary_f1": boundary,
                        "bad_label": bad,
                        **meta,
                    }
                )

    output_dir = checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"quality_{dataset}_{args.split}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "dataset": dataset,
                "split": args.split,
                "seed": args.seed,
                "threshold": args.iou_threshold,
                "records": records,
            },
            f,
            indent=2,
        )
    print(f"[{dataset}] Wrote {len(records)} quality labels to {output_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    for dataset in args.datasets:
        for pattern in runs_dir.glob(f"{dataset}_*"):
            checkpoint_candidates = sorted(pattern.glob(args.pattern))
            if not checkpoint_candidates:
                continue
            checkpoint_path = checkpoint_candidates[0]
            generate_quality_labels(args, dataset, checkpoint_path)


if __name__ == "__main__":
    main()
