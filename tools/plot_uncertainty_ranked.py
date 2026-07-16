#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot most certain and most uncertain samples per method."
    )
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "dsb2018",
            # "monuseg",
            # "rus",
            "nuinsseg",
            "isic2017",
            "kvasirseg",
            "clinicdb",
            # "drive",
            # "promise12",
        ],
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["single", "stoptime", "stability", "flicker", "resilience", "tta", "disagreement"],
    )
    parser.add_argument("--run_name", type=str, default=None, help="Optional specific run directory.")
    parser.add_argument("--output", type=str, default="runs/uncertainty_ranked")
    parser.add_argument(
        "--sample_ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific sample IDs to plot (bypasses ranked selection).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="Number of lowest/highest uncertainty samples to plot per method.",
    )
    parser.add_argument(
        "--dice_threshold",
        type=float,
        default=None,
        help="Optional Dice threshold for highlighting the most certain correct sample.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_run_dir(runs_dir: Path, dataset: str, run_name: Optional[str]) -> Path:
    if run_name:
        run_dir = runs_dir / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir
    candidates = sorted(runs_dir.glob(f"{dataset}_*"))
    if not candidates:
        raise FileNotFoundError(f"No runs found for dataset {dataset}")
    return candidates[0]


def load_quality_map(run_dir: Path, dataset: str, split: str) -> Dict[str, Dict]:
    path = run_dir / f"quality_{dataset}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Quality labels missing: {path}")
    records = load_json(path).get("records", [])
    mapping = {}
    for rec in records:
        sid = str(rec.get("sample_id") or rec.get("index"))
        mapping[sid] = rec
    return mapping


def rasterize_monuseg_mask(xml_path: Path, size: Tuple[int, int]) -> np.ndarray:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for region in root.findall(".//Region"):
        vertices = region.find("Vertices")
        if vertices is None:
            continue
        pts = [
            (float(vertex.attrib["X"]), float(vertex.attrib["Y"]))
            for vertex in vertices
            if "X" in vertex.attrib and "Y" in vertex.attrib
        ]
        if len(pts) >= 3:
            draw.polygon(pts, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)


def load_mask(record: Dict, target_shape: Tuple[int, int]) -> np.ndarray:
    mask_path = record.get("mask_path")
    mask = np.zeros(target_shape, dtype=np.uint8)
    if mask_path:
        path = Path(mask_path)
        if path.suffix.lower() == ".xml":
            mask = rasterize_monuseg_mask(path, target_shape[::-1])
        else:
            mask = np.array(Image.open(path).convert("L"))
    elif record.get("case_dir"):
        mask_dir = Path(record["case_dir"]) / "masks"
        masks = sorted([p for p in mask_dir.glob("*") if p.suffix.lower() in [".png", ".tif", ".tiff"]])
        mask = np.zeros(target_shape, dtype=np.uint8)
        for p in masks:
            mask_img = Image.open(p).convert("L").resize(target_shape[::-1], Image.NEAREST)
            mask = np.maximum(mask, (np.array(mask_img) > 0).astype(np.uint8))
    if mask.shape != target_shape:
        mask = np.array(Image.fromarray(mask).resize(target_shape[::-1], Image.NEAREST))
    return (mask > 0).astype(np.uint8)


def normalize_map(array: np.ndarray) -> np.ndarray:
    arr = array.astype(np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    if heatmap.shape != image.shape[:2]:
        heatmap = np.array(Image.fromarray(heatmap).resize(image.shape[1::-1], Image.BILINEAR))
    heat_color = plt.get_cmap("magma")(normalize_map(heatmap))[..., :3]
    overlay = (0.6 * heat_color + 0.4 * (image / 255.0)).clip(0.0, 1.0)
    return (overlay * 255).astype(np.uint8)


def load_uncertainty_records(run_dir: Path, dataset: str, split: str, method: str) -> List[Dict]:
    path = run_dir / f"uncertainty_{dataset}_{split}_{method}" / f"uncertainty_{dataset}_{split}_{method}.json"
    if not path.exists():
        raise FileNotFoundError(f"Uncertainty file missing: {path}")
    return load_json(path).get("records", [])


def is_empty_prediction(record: Dict) -> bool:
    flag = record.get("_empty_pred")
    if flag is not None:
        return flag
    pred_path = record.get("pred_mask")
    if not pred_path:
        record["_empty_pred"] = True
        return True
    path = Path(pred_path)
    if not path.exists():
        record["_empty_pred"] = True
        return True
    try:
        arr = np.load(path)
    except Exception:
        record["_empty_pred"] = True
        return True
    empty = not np.any(arr)
    record["_empty_pred"] = empty
    return empty


def select_best_correct(records: List[Dict], dice_threshold: Optional[float]) -> Optional[Dict]:
    if dice_threshold is None:
        return None
    sorted_records = sorted(records, key=lambda r: r["unc_boundary_mean"])
    for rec in sorted_records:
        if rec.get("dice", 0.0) >= dice_threshold:
            return rec
    return None


def select_most_certain(records: List[Dict]) -> Dict:
    sorted_records = sorted(records, key=lambda r: r["unc_boundary_mean"])
    if not sorted_records:
        raise RuntimeError("No records available for selection.")
    return sorted_records[0]


def select_most_uncertain(records: List[Dict]) -> Dict:
    sorted_records = sorted(records, key=lambda r: r["unc_boundary_mean"])
    if not sorted_records:
        raise RuntimeError("No records available for selection.")
    return sorted_records[-1]


def prepare_visuals(
    quality_meta: Dict,
    record: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_path = quality_meta.get("image_path")
    if not image_path:
        raise FileNotFoundError("image_path missing in quality metadata.")
    heatmap_path = record.get("unc_map") or record.get("entropy_map") or record.get("variance_map")
    if not heatmap_path:
        raise FileNotFoundError("Uncertainty map not found in record.")
    heatmap = np.load(heatmap_path)

    image = Image.open(image_path).convert("RGB")
    if image.size != heatmap.shape[::-1]:
        image = image.resize(heatmap.shape[::-1], Image.BILINEAR)
    image_np = np.array(image)

    gt_mask = load_mask(quality_meta, heatmap.shape)

    pred_path = record.get("pred_mask")
    if not pred_path:
        raise FileNotFoundError("pred_mask path missing in record.")
    pred = np.load(pred_path)
    if pred.shape != heatmap.shape:
        pred = np.array(Image.fromarray(pred.astype(np.uint8)).resize(heatmap.shape[::-1], Image.NEAREST))
    pred = pred.astype(np.uint8)

    return image_np, gt_mask, pred, heatmap


def plot_method(
    method: str,
    samples: List[Tuple[str, Dict]],
    quality_map: Dict[str, Dict],
    output_dir: Path,
    dataset: str,
    split: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(len(samples), 4, figsize=(4 * 4, 4 * len(samples)))
    axes = np.atleast_2d(axes)
    row_labels = [label for label, _ in samples]

    for row, (label, record) in enumerate(samples):
        sid = record["sample_id"]
        meta = quality_map.get(sid)
        if meta is None:
            raise KeyError(f"Quality metadata missing for {sid}")
        image_np, gt_np, pred_np, heatmap_np = prepare_visuals(meta, record)

        dice = meta.get("dice", 0.0)
        unc = record.get("unc_boundary_mean", 0.0)

        visuals = [
            (image_np, f"{label} - Input"),
            (gt_np, "Ground Truth"),
            (pred_np, "Prediction"),
            (overlay_heatmap(image_np, heatmap_np), f"{method} unc={unc:.3f} Dice={dice:.3f}"),
        ]

        for col, (img, title) in enumerate(visuals):
            ax = axes[row, col]
            cmap = "gray" if img.ndim == 2 else None
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset}_{split}_{method}_ranked.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[plot_uncertainty_ranked] Saved {method} figure to {out_path}")


def process_dataset(args: argparse.Namespace, dataset: str) -> None:
    runs_dir = Path(args.runs_dir)
    run_dir = find_run_dir(runs_dir, dataset, args.run_name)
    quality_map = load_quality_map(run_dir, dataset, args.split)
    output_dir = Path(args.output)

    for method in args.methods:
        try:
            records = load_uncertainty_records(run_dir, dataset, args.split, method)
        except FileNotFoundError as exc:
            print(exc)
            continue

        if args.sample_ids:
            label_samples = []
            for sid in args.sample_ids:
                record = next((rec for rec in records if str(rec.get("sample_id") or rec.get("index")) == sid), None)
                if record:
                    label_samples.append((f"Sample {sid}", record))
                else:
                    print(f"[plot_uncertainty_ranked] {dataset} {method}: sample {sid} not found.")
        else:
            sorted_records = sorted(records, key=lambda r: r["unc_boundary_mean"])
            if not sorted_records:
                print(f"[plot_uncertainty_ranked] {dataset} {method}: no records.")
                continue
            num = max(1, args.num_samples)
            low_samples = sorted_records[:num]
            high_samples = sorted_records[-num:] if len(sorted_records) >= num else sorted_records[::-1]
            label_samples = []
            for idx, rec in enumerate(low_samples, 1):
                label_samples.append((f"Lowest {idx}", rec))
            best_correct = select_best_correct(sorted_records, args.dice_threshold)
            if best_correct is not None:
                label_samples.append((f"Lowest w/ Dice ≥ {args.dice_threshold}", best_correct))
            for idx, rec in enumerate(reversed(high_samples), 1):
                label_samples.append((f"Highest {idx}", rec))

        unique_samples = []
        seen = set()
        for label, rec in label_samples:
            sid = rec["sample_id"]
            if sid in seen:
                continue
            seen.add(sid)
            unique_samples.append((label, rec))
        if not unique_samples:
            continue
        plot_method(method, unique_samples, quality_map, output_dir, dataset, args.split, args.dpi)


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        print(f"[plot_uncertainty_ranked] Processing dataset: {dataset}")
        process_dataset(args, dataset)


if __name__ == "__main__":
    main()
