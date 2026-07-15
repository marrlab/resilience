import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from PIL import Image

# Import existing modules from your codebase
from NCA import BackboneNCA
from evaluate import prepare_state, select_logits, sanitize_targets

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

# Custom Light Dataset to bypass dataloader structured constraints
class SimpleISICValDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, img_size=(256, 256)):
        self.base_path = Path(base_path)
        self.img_size = img_size
        self.img_dir = self.base_path / "ISIC-2017_Validation_Data"
        self.mask_dir = self.base_path / "ISIC-2017_Validation_Part1_GroundTruth"
        
        # Collect all images matching the template, ignoring superpixels
        self.img_paths = sorted([
            p for p in self.img_dir.glob("*.jpg") 
            if "superpixels" not in p.name
        ])
        
    def __len__(self):
        return len(self.img_paths)
        
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_name = img_path.stem + "_segmentation.png"
        mask_path = self.mask_dir / mask_name
        
        image = Image.open(img_path).convert("RGB").resize(self.img_size)
        mask = Image.open(mask_path).convert("L").resize(self.img_size, resample=Image.NEAREST)
        
        image_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        if mask_tensor.max() > 1:
            mask_tensor = (mask_tensor > 127).long()
            
        return image_tensor, mask_tensor

# -------------------------
# Execution Main Guard
# -------------------------
if __name__ == '__main__':
    # Configuration & Grid Setup
    checkpoint_path = Path("runs/isic2017_baseline/best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sigmas = [0.005, 0.01, 0.02, 0.05, 0.1]
    t_primes = [4, 8, 12, 16, 20, 24]

    # Load Model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})

    channel_n = int(args.get("channel_n", 64))
    fire_rate = float(args.get("fire_rate", 0.5))
    hidden_size = int(args.get("hidden_size", 128))
    input_channels = int(args.get("input_channels", 3))
    steps = int(args.get("steps_max", 64))

    model = BackboneNCA(
        channel_n=channel_n,
        fire_rate=fire_rate,
        device=device,
        hidden_size=hidden_size,
        input_channels=input_channels,
        steps_default=steps
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Setup Data
    val_dataset = SimpleISICValDataset(base_path="datasets/isic/isic2017_task1", img_size=(256, 256))
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    num_classes = 2

    results_grid = {f"sigma_{s}_T_{t}": {"resilience_scores": [], "is_error": []} for s in sigmas for t in t_primes}

    print(f"Starting Ablation Study on custom validation layout...")
    print(f"Total samples to process: {len(loader.dataset)}")

    # Grid Search Execution
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = sanitize_targets(targets.to(device, non_blocking=True), num_classes, 255)
            target_mask = targets[0].cpu().numpy()
            
            state_init = prepare_state(images, channel_n)
            conditioning = state_init[..., :input_channels].clone()
            
            S_T = model(state_init, steps=steps, conditioning=conditioning)
            logits = select_logits(S_T, num_classes)
            probs = F.softmax(logits, dim=1)
            
            prob_map = probs[0, 1].cpu().numpy() if num_classes == 2 else probs[0, 0].cpu().numpy()
            m = binarize(prob_map, 0.5)
            
            base_dice = compute_dice(m, target_mask)
            is_error = 1 if base_dice < 0.8 else 0 
            
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

    for s in sigmas:
        for t in t_primes:
            key = f"sigma_{s}_T_{t}"
            y_true = np.array(results_grid[key]["is_error"])
            y_scores = np.array(results_grid[key]["resilience_scores"])
            
            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_scores)
            else:
                auroc = 0.5
                
            print(f"Sigma: {s:<6} | T': {t:<4} | AUROC: {auroc:.4f}")
            
            if auroc > best_auroc:
                best_auroc = auroc
                best_config = key

    print("-" * 50)
    print(f"BEST CONFIGURATION: {best_config} with AUROC = {best_auroc:.4f}")