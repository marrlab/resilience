# Neural Cellular Automata for Medical Segmentation and Uncertainty

This repository contains the training, evaluation, and uncertainty estimation stack we use to study Neural Cellular Automata (NCA) based segmentation on a diverse set of biomedical datasets. In addition to classical nuclei benchmarks (DSB2018, MoNuSeg) we support augmented “hard” splits, real/synthetic ultrasound (RUS), hand-curated NuInsSeg tissues, and ISIC 2017 Task 1 lesions with multi-annotator labels used to build disagreement-aware baselines.

## Key Capabilities

- **End-to-end NCA training / evaluation** via `train.py` and `evaluate.py` with configurable steps, channel budgets, image sizes, and datasets.
- **Unified dataloader** (`dataloader.py`) for VOC, CamVid, DSB2018, MoNuSeg (with automatic XML rasterization + augmentation), NuInsSeg (with automatic split generation + augmentation), RUS ultrasound, and ISIC 2017 Task 1 (binary lesions with multi-annotator metadata).
- **Uncertainty tooling**:
  - `quality_labels.py` computes Dice / boundary scores per sample and records annotator disagreement if available.
  - `compute_uncertainty.py` implements single forward entropy, stop-time, stability, flicker, resilience, test-time augmentation (TTA), and the ISIC-specific *disagreement* baseline derived from annotator variance.
  - `evaluate_uncertainty.py` reports Dice@80/90, AURC, AUROC, AUPRC, and adds two fusion baselines (rank-average and validation-tuned weighted fusion) to combine the best signals.
  - `plot_uncertainty_examples.py` creates qualitative panels.
- **Augmented “hard” splits** for DSB2018 (photometric + geometric noise), MoNuSeg, and NuInsSeg (`*_aug` splits are generated on-the-fly if missing).

## Installation

```bash
python -m venv nca_env
source nca_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Ensure the matching CUDA builds of torch/torchvision if training on GPU.
```

## Dataset Preparation

| Dataset     | Expected Location                                      | Notes |
|-------------|--------------------------------------------------------|-------|
| VOC 2012    | `datasets/voc/VOCdevkit/VOC2012` or `datasets/VOC2012_train_val` | Standard PASCAL VOC segmentation |
| CamVid      | `datasets/CamVid`                                      | Requires `train/val/test` and label folders |
| DSB2018     | `datasets/dsb2018`                                     | Contains `stage1_train`; `_aug` splits auto-generated |
| MoNuSeg     | `datasets/MoNuSeg`                                     | Run `python datasets/MoNuSeg/split_monuseg.py` first |
| RUS Ultrasound | `datasets/US/RUS/{images,annotations}/{train,val,test}` | Provide masks for training/validation splits |
| NuInsSeg    | `datasets/NuInsSeg`                                    | Run `python datasets/NuInsSeg/split_nuinsseg.py --root datasets/NuInsSeg --force` once; `_aug` splits are generated on demand |
| ISIC 2017 Task 1 | `datasets/isic/isic2017_task1`                     | Keep the official `ISIC-2017_<split>_Data` and `*_Part1_GroundTruth` folders. Additional annotator masks (if present) are automatically discovered for disagreement baselines. |

All scripts accept `--data_root` to override the automatically detected locations when needed.

## Training & Evaluation

Train on any supported dataset:

```bash
# Example: NuInsSeg baseline
python train.py \
  --dataset nuinsseg \
  --data_root datasets/NuInsSeg \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-3 \
  --steps_min 32 --steps_max 64 \
  --image_size 512 512 \
  --num_workers 4 \
  --device cuda \
  --exp_name nuinsseg_baseline
```

Evaluate a checkpoint on a target split:

```bash
python evaluate.py \
  --checkpoint runs/nuinsseg_baseline/best.pt \
  --dataset nuinsseg \
  --split test \
  --data_root datasets/NuInsSeg \
  --batch_size 2 \
  --num_workers 4 \
  --image_size 512 512 \
  --device cuda
```

`evaluate.py` reports pixel accuracy, mIoU, Dice, and boundary F1.

## Uncertainty Pipeline

1. **Quality labels** (Dice, boundary F1, annotator stats):

   ```bash
   python quality_labels.py \
     --datasets dsb2018 monuseg rus nuinsseg isic2017 \
     --split test \
     --runs_dir runs \
     --pattern "*best.pt" \
     --batch_size 2 \
     --num_workers 4 \
     --device cuda
   ```

2. **Uncertainty computation** (per-method maps + scalars). Available methods: `single`, `stoptime`, `stability`, `flicker`, `resilience`, `tta`, `disagreement`.

   ```bash
   python compute_uncertainty.py \
     --datasets dsb2018 nuinsseg isic2017 \
     --split test test_aug \
     --methods single stoptime stability flicker resilience tta disagreement \
     --runs_dir runs \
     --batch_size 2 \
     --num_workers 4 \
     --device cuda \
     --save_png
   ```

   - `--tta_max_transforms` controls how many geometric transforms are used.
   - `disagreement` requires multi-annotator metadata (ISIC 2017); it falls back to zeros otherwise.

3. **Evaluation + fusion** using `evaluate_uncertainty.py`. This script loads the quality/uncertainty JSON pairs per run and computes Dice@80/90, AURC, AUROC, and AUPRC. Optional fusion baselines:

   ```bash
   python evaluate_uncertainty.py \
     --datasets dsb2018 nuinsseg isic2017 \
     --split test \
     --methods single tta stability disagreement \
     --fusion_pairs tta,stability tta,single \
     --fusion_metric aurc \
     --fusion_alpha_steps 21 \
     --fusion_val_split val \
     --runs_dir runs \
     --output runs/unc_summary_test.json
   ```

   - `fusion_rank_<m1>_<m2>`: average normalized ranks of each method.
   - `fusion_weighted_<m1>_<m2>`: learn `α` on the validation split (grid search) after standardizing each uncertainty score.

4. **Qualitative panels**:

   ```bash
   python plot_uncertainty_examples.py \
     --datasets nuinsseg isic2017 \
     --split test \
     --methods single tta disagreement \
     --runs_dir runs \
     --output runs/uncertainty_plots
   ```

## ISIC-Specific Notes

- `dataloader.ISIC2017Dataset` stores multi-annotator masks (when available). Downstream scripts expose `multi_mask_paths` metadata.
- `quality_labels.py` logs per-sample annotator variance, maximum disagreement, and pairwise Dice statistics.
- The `disagreement` uncertainty method computes pixel-wise variance and the average mask, serving as an upper bound for how much label ambiguity exists per image.

## Augmented “Hard” Splits

- **DSB2018**: requesting `split=test_aug` or `train_aug` triggers `_ensure_dsb_augmented_split`, which applies geometric and photometric noise to each case.
- **MoNuSeg**: `split=test_aug` uses `_ensure_monuseg_augmented_split` to rasterize XML annotations, apply augmentations, and cache the results.
- **NuInsSeg**: `split=test_aug` calls `_ensure_nuinsseg_augmented_split` to create noisy counterparts of each split image/mask.

Use `--split test` for in-distribution evaluation and `--split test_aug` to stress the models with synthetic perturbations.

## Automated Jobs

The repository includes SLURM helpers such as `run_dsb2018.sbatch`, `run_monuseg.sbatch`, and `run_isic.sbatch`. Update these scripts with your environment-specific parameters (partition, time, etc.) before submitting them to a cluster.

## Repository Structure

```
├── NCA.py                     # Backbone NCA implementation
├── dataloader.py              # Dataset builders and augmenters
├── train.py / evaluate.py     # Segmentation training and evaluation
├── quality_labels.py          # Dice/boundary tracking + annotator stats
├── compute_uncertainty.py     # Uncertainty map generation (multiple methods)
├── evaluate_uncertainty.py    # Risk/coverage, ROC/AUC, fusion baselines
├── plot_uncertainty_examples.py # Qualitative grids
├── datasets/
│   ├── NuInsSeg/split_nuinsseg.py
│   ├── MoNuSeg/split_monuseg.py
│   └── ...
└── runs/                      # Default output directory for checkpoints + metrics
```

## Troubleshooting

- **Missing dataset roots**: all loaders call `_ensure_exists` and raise descriptive errors. Use `--data_root` to point to the correct location or create symlinks under `datasets/`.
- **RUS annotations**: ensure `datasets/US/RUS/annotations/{train,val,test}` exist. Training will fail otherwise.
- **NuInsSeg splits**: run `datasets/NuInsSeg/split_nuinsseg.py` once before training; re-run with `--force` to rebuild.
- **Torch install**: match the CUDA version on your system. For cluster jobs, load the corresponding module before `pip install`.

## Citation & License

This repository is part of an ongoing NCA-based uncertainty study. Please contact the maintainers for up-to-date citation and licensing information before redistributing or publishing results derived from this codebase.
