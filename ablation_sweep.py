import argparse
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# Import existing modules from your codebase
from NCA import BackboneNCA
from dataloader import build_split_dataloader
from evaluate import (
    DATASET_DEFAULT_ROOTS,
    prepare_state,
    resolve_data_root,
    select_logits,
    sanitize_targets,
)

# -------------------------
# Utility Functions
# -------------------------
def binarize(prob_map, threshold=0.5):
    """Binarize probability maps to binary masks."""
    return (prob_map > threshold).astype(np.uint8)

def compute_dice(mask1, mask2):
    """Compute Dice coefficient between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum()
    if union == 0:
        return 1.0
    return 2. * intersection / union

def compute_iou(mask1, mask2):
    """Compute Intersection over Union (IoU) between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0
    return intersection / (union + 1e-8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep noise sigma / recovery steps and score error-detection AUROC."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_DEFAULT_ROOTS.keys()),
        required=True,
        help="Dataset name.",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate.")
    parser.add_argument("--data_root", type=str, default=None, help="Optional dataset root override.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=[256, 256],
        help="Resize (width height).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=None, help="Number of NCA steps to reach S_T.")
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--subset", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--dice_error_threshold", type=float, default=0.8)
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.05, 0.1],
        help="Perturbation noise std values to sweep.",
    )
    parser.add_argument(
        "--t_primes",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16, 20, 24],
        help="Recovery step counts to sweep.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional path to save results JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sigmas = args.sigmas
    t_primes = args.t_primes

    # Load Model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})

    channel_n = int(ckpt_args.get("channel_n", 64))
    fire_rate = float(ckpt_args.get("fire_rate", 0.5))
    hidden_size = int(ckpt_args.get("hidden_size", 128))
    input_channels = int(ckpt_args.get("input_channels", 3))
    dropout_rate = float(ckpt_args.get("dropout_rate", 0.0))
    steps = int(args.steps if args.steps is not None else ckpt_args.get("steps_max", 64))

    model = BackboneNCA(
        channel_n=channel_n,
        fire_rate=fire_rate,
        device=device,
        hidden_size=hidden_size,
        input_channels=input_channels,
        steps_default=steps,
        dropout_rate=dropout_rate,
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Setup Data
    image_size = tuple(args.image_size)
    data_root = resolve_data_root(args.dataset, args.data_root)
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

    results_grid = {f"sigma_{s}_T_{t}": {"resilience_scores": [], "is_error": []} for s in sigmas for t in t_primes}

    print(f"Starting Ablation Study on dataset='{args.dataset}' split='{args.split}' (root={data_root})")
    print(f"Total samples to process: {len(loader.dataset)}")

    # Grid Search Execution
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = sanitize_targets(targets.to(device, non_blocking=True), num_classes, args.ignore_index)
            target_mask = targets[0].cpu().numpy()

            state_init = prepare_state(images, channel_n)
            conditioning = state_init[..., :input_channels].clone()

            S_T = model(state_init, steps=steps, conditioning=conditioning)
            logits = select_logits(S_T, num_classes)
            probs = F.softmax(logits, dim=1)

            prob_map = probs[0, 1].cpu().numpy() if num_classes == 2 else probs[0, 0].cpu().numpy()
            m = binarize(prob_map, 0.5)

            base_dice = compute_dice(m, target_mask)
            is_error = 1 if base_dice < args.dice_error_threshold else 0

            for s in sigmas:
                perturbed_state_init = S_T.clone()
                noise = torch.randn_like(perturbed_state_init[..., input_channels:]) * s
                perturbed_state_init[..., input_channels:] += noise

                for t_prime in t_primes:
                    S_TT = model(perturbed_state_init.clone(), steps=t_prime, conditioning=conditioning)
                    logits_perturbed = select_logits(S_TT, num_classes)
                    probs_perturbed = F.softmax(logits_perturbed, dim=1)

                    prob_map_perturbed = probs_perturbed[0, 1].cpu().numpy() if num_classes == 2 else probs_perturbed[0, 0].cpu().numpy()
                    m_prime = binarize(prob_map_perturbed, 0.5)

                    iou = compute_iou(m, m_prime)
                    resilience = 1.0 - iou

                    key = f"sigma_{s}_T_{t_prime}"
                    results_grid[key]["resilience_scores"].append(resilience)
                    results_grid[key]["is_error"].append(is_error)

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} images...")

    # Evaluation & Report
    print("\n===== ABLATION RESULTS (AUROC for Error Detection) =====")
    best_auroc = 0
    best_config = ""
    summary = {}

    for s in sigmas:
        for t in t_primes:
            key = f"sigma_{s}_T_{t}"
            y_true = np.array(results_grid[key]["is_error"])
            y_scores = np.array(results_grid[key]["resilience_scores"])

            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_scores)
            else:
                auroc = 0.5

            summary[key] = auroc
            print(f"Sigma: {s:<6} | T': {t:<4} | AUROC: {auroc:.4f}")

            if auroc > best_auroc:
                best_auroc = auroc
                best_config = key

    print("-" * 50)
    print(f"BEST CONFIGURATION: {best_config} with AUROC = {best_auroc:.4f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": str(checkpoint_path),
                    "dataset": args.dataset,
                    "split": args.split,
                    "data_root": data_root,
                    "sigmas": sigmas,
                    "t_primes": t_primes,
                    "auroc_grid": summary,
                    "best_config": best_config,
                    "best_auroc": best_auroc,
                },
                f,
                indent=2,
            )
        print(f"Saved results to {output_path}")


if __name__ == '__main__':
    main()
