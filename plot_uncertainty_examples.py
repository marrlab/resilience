from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot qualitative uncertainty examples.")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["dsb2018", "monuseg", "rus", "nuinsseg", "isic2017"],
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default="runs/uncertainty_plots")
    parser.add_argument("--lowest", type=int, default=2)
    parser.add_argument("--mid", type=int, default=2)
    parser.add_argument("--highest", type=int, default=4)
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["single"],
        help="Uncertainty methods to visualize.",
    )
    return parser.parse_args()


def load_records(run_dir: Path, dataset: str, split: str, method: str) -> List[Dict]:
    unc_path = (
        run_dir
        / f"uncertainty_{dataset}_{split}_{method}"
        / f"uncertainty_{dataset}_{split}_{method}.json"
    )
    if not unc_path.exists():
        return []
    with open(unc_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("records", [])


def rasterize_monuseg_mask(xml_path: Path, size: tuple[int, int]) -> np.ndarray:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mask = Image.new("L", size, 0)
    draw = Image.Draw.Draw(mask)
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


def load_mask(record: Dict, image_shape: tuple[int, int]) -> np.ndarray:
    mask_path = record.get("mask_path")
    if not mask_path:
        case_dir = record.get("case_dir")
        if case_dir:
            mask_dir = Path(case_dir) / "masks"
            masks = sorted([p for p in mask_dir.glob("*") if p.suffix.lower() in [".png", ".tif", ".tiff"]])
            mask = np.zeros(image_shape, dtype=np.uint8)
            for m in masks:
                mask_img = Image.open(m).convert("L").resize(image_shape[::-1], Image.NEAREST)
                mask = np.maximum(mask, (np.array(mask_img) > 0).astype(np.uint8))
            return mask
        return np.zeros(image_shape, dtype=np.uint8)
    path = Path(mask_path)
    if path.suffix.lower() == ".xml":
        mask = rasterize_monuseg_mask(path, image_shape[::-1])
    else:
        mask = np.array(Image.open(path).convert("L"))
    if mask.shape != image_shape:
        mask = np.array(
            Image.fromarray(mask).resize(image_shape[::-1], Image.NEAREST), dtype=np.uint8
        )
    return mask


def select_examples(records: List[Dict], args: argparse.Namespace) -> List[Dict]:
    sorted_records = sorted(records, key=lambda r: r["unc_boundary_mean"])
    n = len(sorted_records)
    if n == 0:
        return []
    lowest = sorted_records[: min(args.lowest, n)]
    highest = sorted_records[max(0, n - args.highest) : n]
    mid_start = max(0, n // 2 - args.mid // 2)
    mid = sorted_records[mid_start : mid_start + args.mid]
    selected = lowest + mid + highest
    # ensure unique sample_id by preserving order but dropping duplicates
    seen = set()
    deduped = []
    for rec in selected:
        key = rec.get("sample_id", rec.get("index"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)
    return deduped


def plot_examples(dataset: str, records: List[Dict], output_dir: Path, split: str, method: str) -> None:
    if not records:
        print(f"No records for {dataset} {split}, skipping.")
        return
    rows = len(records)
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.0))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, rec in enumerate(records):
        image_path = rec.get("image_path")
        if not image_path and rec.get("case_dir"):
            case_dir = Path(rec["case_dir"])
            images_dir = case_dir / "images"
            candidates = sorted(images_dir.glob("*"))
            if candidates:
                image_path = str(candidates[0])
        if not image_path:
            continue
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        pred = np.load(rec["pred_mask"])
        unc_path = rec.get("unc_map") or rec.get("entropy_map") or rec.get("variance_map")
        if not unc_path:
            continue
        entropy = np.load(unc_path)
        gt = load_mask(rec, pred.shape)

        titles = [
            f"{rec.get('sample_id','sample')} (Dice {rec.get('dice',0):.2f})",
            "Ground Truth",
            "Prediction",
            f"Entropy (unc={rec['unc_boundary_mean']:.2f})",
        ]
        visuals = [image_np, gt, pred, entropy]
        cmaps = [None, "gray", "gray", "magma"]
        for col in range(cols):
            ax = axes[idx, col]
            ax.imshow(visuals[col], cmap=cmaps[col])
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.set_title(titles[col], fontsize=10)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset}_{split}_{method}_uncertainty_examples.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved qualitative plot for {dataset} to {out_path}")


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    output_dir = Path(args.output)
    for dataset in args.datasets:
        for method in args.methods:
            combined_records: List[Dict] = []
            for exp_dir in sorted(runs_dir.glob(f"{dataset}_*")):
                records = load_records(exp_dir, dataset, args.split, method)
                combined_records.extend(records)
            selected = select_examples(combined_records, args)
            plot_examples(dataset, selected, output_dir, args.split, method)


if __name__ == "__main__":
    main()
