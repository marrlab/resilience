#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize uncertainty maps and resilience behaviour for a single sample."
    )
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sample", type=str, required=True, help="Sample identifier or 1-based index.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--methods", nargs="+", default=["single", "stoptime", "stability", "flicker", "resilience", "tta", "disagreement"])
    parser.add_argument("--run_name", type=str, default=None, help="Specific run directory (defaults to first runs/<dataset>_*).")
    parser.add_argument("--output", type=str, default="runs/uncertainty_visuals")
    parser.add_argument("--figure_dpi", type=int, default=200)
    parser.add_argument("--create_gif", action="store_true", help="Create resilience GIF if resilience method is included.")
    return parser.parse_args()


def find_run_dir(runs_dir: Path, dataset: str, run_name: Optional[str]) -> Path:
    if run_name:
        run_dir = runs_dir / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir
    candidates = sorted(runs_dir.glob(f"{dataset}_*"))
    if not candidates:
        raise FileNotFoundError(f"No runs found under {runs_dir} matching {dataset}_*")
    return candidates[0]


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_sample_id(sample_arg: str, records: List[Dict]) -> str:
    if sample_arg.isdigit():
        idx = int(sample_arg) - 1
        if idx < 0 or idx >= len(records):
            raise IndexError(f"Sample index {sample_arg} is out of range (1-{len(records)}).")
        return str(records[idx].get("sample_id") or records[idx].get("index"))
    return sample_arg


def find_record(records: List[Dict], sample_id: str) -> Dict:
    for rec in records:
        sid = str(rec.get("sample_id") or rec.get("index"))
        if sid == sample_id:
            return rec
    raise KeyError(f"Sample '{sample_id}' not found.")


def load_quality_record(run_dir: Path, dataset: str, split: str, sample_token: str) -> Tuple[Dict, str]:
    q_path = run_dir / f"quality_{dataset}_{split}.json"
    if not q_path.exists():
        raise FileNotFoundError(f"Quality labels not found: {q_path}")
    records = load_json(q_path).get("records", [])
    resolved_id = resolve_sample_id(sample_token, records)
    return find_record(records, resolved_id), resolved_id


def load_uncertainty_record(run_dir: Path, dataset: str, split: str, method: str, sample_id: str) -> Dict:
    u_dir = run_dir / f"uncertainty_{dataset}_{split}_{method}"
    u_path = u_dir / f"uncertainty_{dataset}_{split}_{method}.json"
    if not u_path.exists():
        raise FileNotFoundError(f"Uncertainty file not found: {u_path}")
    records = load_json(u_path).get("records", [])
    return find_record(records, sample_id)


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


def load_mask_from_metadata(record: Dict, target_shape: Tuple[int, int]) -> np.ndarray:
    mask_path = record.get("mask_path")
    mask: np.ndarray
    if mask_path:
        path = Path(mask_path)
        if path.suffix.lower() == ".xml":
            mask = rasterize_monuseg_mask(path, target_shape[::-1])
        else:
            mask = np.array(Image.open(path).convert("L"))
    else:
        case_dir = record.get("case_dir")
        if case_dir:
            mask_dir = Path(case_dir) / "masks"
            masks = sorted([p for p in mask_dir.glob("*") if p.suffix.lower() in [".png", ".tif", ".tiff"]])
            mask = np.zeros(target_shape, dtype=np.uint8)
            for m in masks:
                mask_img = Image.open(m).convert("L").resize(target_shape[::-1], Image.NEAREST)
                mask = np.maximum(mask, (np.array(mask_img) > 0).astype(np.uint8))
        else:
            mask = np.zeros(target_shape, dtype=np.uint8)
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


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    if heatmap.shape != image.shape[:2]:
        heatmap = np.array(Image.fromarray(heatmap).resize(image.shape[1::-1], Image.BILINEAR))
    heat_color = plt.get_cmap("magma")(normalize_map(heatmap))[..., :3]
    overlay = (alpha * heat_color + (1 - alpha) * (image / 255.0)).clip(0.0, 1.0)
    return (overlay * 255).astype(np.uint8)


def build_figure(image: np.ndarray, gt_mask: np.ndarray, method_maps: List[Tuple[str, np.ndarray, float]], output_path: Path, dpi: int) -> None:
    target_shape = method_maps[0][1].shape if method_maps else image.shape[:2]
    if image.shape[:2] != target_shape:
        image = np.array(Image.fromarray(image).resize(target_shape[::-1], Image.BILINEAR))
    if gt_mask.shape != target_shape:
        gt_mask = np.array(
            Image.fromarray((gt_mask * 255).astype(np.uint8)).resize(target_shape[::-1], Image.NEAREST)
        )
        gt_mask = (gt_mask > 127).astype(np.uint8)
    cols = 2 + len(method_maps)
    fig, axes = plt.subplots(1, cols, figsize=(3.5 * cols, 4))
    axes = np.atleast_1d(axes)

    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth")

    for ax, (method, heatmap, score) in zip(axes[2:], method_maps):
        ax.imshow(overlay_heatmap(image, heatmap))
        ax.set_title(f"{method}\nunc={score:.3f}" if score is not None else method)
        ax.axis("off")

    for ax in axes[:2]:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"[visualize_uncertainty] Saved figure to {output_path}")


def create_resilience_gif(image: np.ndarray, frames_path: Path, output_path: Path) -> None:
    frames = np.load(frames_path)

    def overlay_mask(mask: np.ndarray, color: Tuple[int, int, int], label: str) -> Image.Image:
        mask_resized = mask
        if mask.shape != image.shape[:2]:
            mask_resized = np.array(
                Image.fromarray(mask.astype(np.uint8)).resize(image.shape[1::-1], Image.NEAREST)
            )
        overlay = image.copy()
        mask_bool = mask_resized.astype(bool)
        overlay[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * np.array(color)).astype(np.uint8)
        pil_img = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil_img)
        draw.text((10, 10), label, fill="white")
        return pil_img

    gif_frames: List[Image.Image] = []
    for idx, mask in enumerate(frames):
        if idx == 0:
            label = "Baseline"
        else:
            label = f"Relax step {idx}"
        gif_frames.append(overlay_mask(mask, (255, 128, 0), label))

    if not gif_frames:
        return
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=400,
        loop=0,
    )
    print(f"[visualize_uncertainty] Saved resilience GIF to {output_path}")


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    run_dir = find_run_dir(runs_dir, args.dataset, args.run_name)
    quality_rec, sample_id = load_quality_record(run_dir, args.dataset, args.split, args.sample)

    image_path = quality_rec.get("image_path")
    if not image_path:
        raise FileNotFoundError("image_path missing in quality labels.")
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    gt_mask = load_mask_from_metadata(quality_rec, image_np.shape[:2])

    method_maps: List[Tuple[str, np.ndarray, float]] = []
    resilience_frames_path: Optional[Path] = None
    for method in args.methods:
        try:
            rec = load_uncertainty_record(run_dir, args.dataset, args.split, method, sample_id)
        except (FileNotFoundError, KeyError):
            print(f"[visualize_uncertainty] Skipping {method}: record not found.")
            continue
        map_path = rec.get("unc_map") or rec.get("entropy_map") or rec.get("variance_map")
        if not map_path:
            print(f"[visualize_uncertainty] Skipping {method}: no map path.")
            continue
        heatmap = np.load(map_path)
        method_maps.append((method, heatmap, rec.get("unc_boundary_mean")))
        if method == "resilience":
            frames_path = rec.get("resilience_frames")
            if frames_path:
                p = Path(frames_path)
                if p.exists():
                    resilience_frames_path = p

    if not method_maps:
        raise RuntimeError("No uncertainty methods available for plotting.")

    output_dir = Path(args.output)
    fig_path = output_dir / f"{args.dataset}_{sample_id}_{args.split}_uncertainty.png"
    build_figure(image_np, gt_mask, method_maps, fig_path, args.figure_dpi)

    if args.create_gif and resilience_frames_path:
        gif_path = output_dir / f"{args.dataset}_{sample_id}_{args.split}_resilience.gif"
        create_resilience_gif(image_np, resilience_frames_path, gif_path)


if __name__ == "__main__":
    main()
