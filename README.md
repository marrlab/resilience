# Neural Cellular Automata for Medical Segmentation and Uncertainty

This repository contains the training, evaluation, and uncertainty estimation stack we use to study Neural Cellular Automata (NCA) based segmentation on a diverse set of biomedical datasets. In addition to classical nuclei benchmarks (DSB2018, MoNuSeg) we support augmented “hard” splits, real/synthetic ultrasound (RUS), hand-curated NuInsSeg tissues, and ISIC 2017 Task 1 lesions with multi-annotator labels used to build disagreement-aware baselines.

## Key Capabilities

- **End-to-end NCA training / evaluation** via `train.py` and `evaluate.py` with configurable steps, channel budgets, image sizes, and datasets.
- **Optional dropout and MC-dropout baseline** in the shared NCA update network. Dropout is disabled by default, enabled at training time with `--dropout_rate`, and sampled at test time with the `mc_dropout` uncertainty method.
- **Unified dataloader** (`dataloader.py`) for VOC, CamVid, DSB2018, MoNuSeg (with automatic XML rasterization + augmentation), NuInsSeg (with automatic split generation + augmentation), RUS ultrasound, and ISIC 2017 Task 1 (binary lesions with multi-annotator metadata).
- **Uncertainty tooling**:
  - `quality_labels.py` computes Dice / boundary scores per sample and records annotator disagreement if available.
  - `compute_uncertainty.py` implements single forward entropy, MC dropout, stop-time, stability, flicker, resilience, test-time augmentation (TTA), and the ISIC-specific *disagreement* baseline derived from annotator variance.
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

Train the otherwise identical dropout baseline by passing a non-zero probability:

```bash
python train.py \
  --dataset nuinsseg \
  --data_root datasets/NuInsSeg \
  --dropout_rate 0.1 \
  --exp_name nuinsseg_mc_dropout
```

`--dropout_rate` (also available as `--dropout`) defaults to `0.0`. The value is
saved in the checkpoint and is restored automatically by evaluation and uncertainty
scripts. Ordinary validation/evaluation disables dropout, as expected.

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

2. **Uncertainty computation** (per-method maps + scalars). Available methods: `single`, `mc_dropout`, `stoptime`, `stability`, `flicker`, `resilience`, `tta`, `disagreement`.

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

   For a checkpoint trained with non-zero `--dropout_rate`, run the MC-dropout baseline with:

   ```bash
   python compute_uncertainty.py \
     --datasets nuinsseg \
     --split test \
     --methods mc_dropout \
     --mc_dropout_samples 20 \
     --runs_dir runs \
     --device cuda
   ```

   MC dropout keeps the rest of the model in evaluation mode, activates only the
   update-network dropout layer, and uses predictive mutual information as its primary
   uncertainty map/score. Predictive entropy and variance maps are saved as well.
   Checkpoints trained with the default `--dropout_rate 0` are skipped for this method.

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

## Single and Double Resilience Perturbations

Both compute_uncertainty.py and ablation_sweep.py accept:

| Option | Meaning |
|---|---|
| --resilience_mode single | One injection, followed by one recovery stage (default). |
| --resilience_mode double | Two sequential injections, each followed by a recovery stage. |
| --resilience_mode both | Save single and double results from the same original state and first recovery. |
| --no_perturb_rgb | Add noise only to non-input channels; keep original RGB fixed (default). |
| --perturb_rgb | Add noise to all channels, including RGB/input channels. |

The original prediction is always obtained with the original RGB held fixed.
During each recovery stage the input channels are clamped to that stage's input:
original RGB for latent-only noise, or noisy RGB when RGB noise is enabled.
In double RGB mode, the second fresh noise is added to the already recovered
state, including its previously perturbed RGB. No clipping is applied to Gaussian
noise. This keeps the RGB perturbation effective instead of immediately restoring
the original image, and avoids unconstrained RGB drift from learned updates.

For initial rollout length T and recovery length T':

- Single: inject at T, recover T' updates, evaluate at T + T'.
- Double: inject at T, recover T' updates, inject again at T + T',
  recover another T' updates, evaluate at T + 2T'.
- Each injection uses independent Gaussian noise with the same standard deviation.
- Both endpoints are compared to the original mask at T using 1 - IoU.
- --resilience_relax_steps (computation) and each --t_primes value (ablation)
  mean recovery steps **per injection**, not the total recovery budget.

Compute paired single/double results without RGB noise:

~~~powershell
python compute_uncertainty.py --datasets dsb2018 --split test --methods resilience --resilience_mode both --no_perturb_rgb --resilience_noise 0.02 --resilience_relax_steps 12 --seed 42
~~~

Compute the corresponding paired results with RGB noise:

~~~powershell
python compute_uncertainty.py --datasets dsb2018 --split test --methods resilience --resilience_mode both --perturb_rgb --resilience_noise 0.02 --resilience_relax_steps 12 --seed 42
~~~

Results use distinct method IDs:

| Mode | Without RGB noise | With RGB noise |
|---|---|---|
| Single | resilience_single_latent | resilience_single_rgb |
| Double | resilience_double_latent | resilience_double_rgb |

The directory and JSON filename follow the existing convention, for example
runs/<run>/uncertainty_dsb2018_test_resilience_double_latent/uncertainty_dsb2018_test_resilience_double_latent.json.
The old ..._resilience outputs are left in place. Repeating the same configuration
overwrites its own output; use separate run folders to retain different seeds or
noise/recovery settings.

Evaluate all four variants together after computing both RGB settings:

~~~powershell
python evaluate_uncertainty.py --datasets dsb2018 --split test --methods resilience_single_latent resilience_double_latent resilience_single_rgb resilience_double_rgb --output runs/unc_summary_resilience_comparison_test.json
~~~

New resilience records include Dice and boundary F1 measured on the exact
original mask being probed. Evaluation uses these values even if a separate
quality file contains a different stochastic prediction. New resilience outputs
can therefore be evaluated without first generating quality_*.json.
Other uncertainty methods continue to require their quality-label files.

pred_mask and prob_map describe the original prediction.
recovered_pred_mask and recovered_prob_map describe the selected endpoint.
unc_map is the pixel-wise mask disagreement; the ranking score remains the
scalar 1 - IoU, stored in the existing uncertainty-score fields.
Saved frames and labels identify both injections and their recovery stages.
JSON metadata records the selected mode, RGB choice, seed, noise magnitude,
injection times and final evaluation time.

Run the paired ablation grid:

~~~powershell
python ablation_sweep.py --checkpoint runs/dsb2018_baseline/best.pt --dataset dsb2018 --resilience_mode both --no_perturb_rgb --sigmas 0.02 --t_primes 12 --seed 42 --output runs/ablation_resilience_latent.json
~~~

Repeat with --perturb_rgb and a different output filename for the RGB comparison.
Grid entries include both their method ID and parameters, and contain per-image
scores and original Dice. Every sample in a batch is scored. AUROC is recorded
as null if the evaluation set contains only one quality class.

Baseline, injection and recovery randomness are isolated. With the same checkpoint,
dataset order, batch size, image size, rollout settings and seed, standalone
single/double runs match their endpoints in both mode. Additional recovery steps
do not change the random sequence used for later images. Keep these settings
identical across comparisons. The same seed also pairs the latent noise between
RGB settings. Fire masks are reproducible, but GPU reproducibility across hardware
or different PyTorch builds is not guaranteed.

The corrected rollout does not reproduce the old compute_uncertainty.py
resilience branch exactly: that branch perturbed all channels and allowed RGB to
evolve without clamping, including during the initial rollout. Existing checkpoints
remain compatible; no retraining is needed. Recompute inference results for the
new comparisons.

For a journal comparison, additionally run single perturbation with twice the
recovery length to control for the extra update budget in double mode.
Plotting/evaluation scripts accept the saved method IDs via --methods; the
visualization tool creates separate GIFs with injection labels for each variant.

Run the focused regression tests with:

~~~powershell
python -m unittest discover -s tests -p "test_resilience.py" -v
~~~

## Calibration of Segmentation Failure Probabilities

Use plot_calibration.py after computing per-image uncertainty results for both
validation and test. The script fits a separate mapping for each dataset, run and
saved method ID; it does not pool different checkpoints or change NCA weights.

Failure is defined as **original-prediction Dice < --failure_threshold**.
The default threshold is 0.8, matching ablation_sweep.py's default. The existing
evaluate_uncertainty.py instead labels the lowest-Dice quantile as failures, so
its AUROC can differ. This script recomputes AUROC using the same fixed failure
definition as the calibration plot. Choose the threshold before inspecting test
results.

First generate paired validation and test results using identical inference
settings (including noise magnitude, recovery length and RGB choice):

~~~powershell
python compute_uncertainty.py --datasets dsb2018 --split val --methods resilience --resilience_mode both --no_perturb_rgb --resilience_noise 0.02 --resilience_relax_steps 12
python compute_uncertainty.py --datasets dsb2018 --split test --methods resilience --resilience_mode both --no_perturb_rgb --resilience_noise 0.02 --resilience_relax_steps 12
~~~

Plot the single/double comparison with the logistic baseline:

~~~powershell
python plot_calibration.py --datasets dsb2018 --methods resilience_single_latent resilience_double_latent --calibration_split val --split test --failure_threshold 0.8 --n_bins 10 --calibrator logistic --output runs/calibration
~~~

Both --dataset and --datasets accept one or more dataset names. --runs_dir uses
the existing runs/<dataset>_* convention; --run_name restricts analysis to one
run. Use the corresponding *_rgb method IDs after generating RGB-noise results.
Legacy methods such as resilience and single are also supported when their
matching quality_<dataset>_<split>.json contains original-prediction Dice.

New resilience files already contain exact baseline Dice. That value takes
precedence over separate quality-label files. The script refuses duplicate IDs,
missing Dice, invalid values, detected validation/test sample or group overlap,
and conflicting saved protocol settings. Missing result pairs are skipped.
Legacy files may lack enough metadata to verify the full protocol; the output
records which checks could not be made. Split-local numeric indices alone cannot
prove sample independence.

The default calibrator learns p(failure | u) = sigmoid(a*u + b) on validation.
Logistic regression uses standardized scores and fixed L2 regularization C=1;
--logistic_c changes this value. The saved coefficients a and b apply directly
to the original uncertainty scale. No class balancing is applied, since failure
prevalence must be retained. Both failure classes must occur in the calibration
set; the script refuses to fit if validation has only successes or only failures.

Use --calibrator auto to compare logistic and increasing isotonic regression by
cross-validation **within validation**, then refit the selected mapping on all
validation images. Selection uses --selection_metric brier (default) or log_loss.
By default isotonic is considered only with at least 100 validation images and
10 examples of each failure class; these are configurable practical thresholds,
not guarantees against overfitting. Insufficient data or unusable group folds
fall back to logistic. --calibrator isotonic explicitly fits isotonic without
automatic selection.

Calibration plots use 10 quantile bins by default; choose --n_bins 5 for smaller
test sets, or --strategy uniform for equal-width probability bins. Binning follows
[scikit-learn calibration_curve](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html).
Tied probabilities stay together, and empty bins are omitted, so fewer than the
requested number of plotted points can occur. Labels show failures/image count.
The horizontal axis is mean predicted failure probability; the vertical axis is
observed failure fraction. A second panel shows the test probability histogram.

Each run produces:

- runs/calibration/<run>/<dataset>_<split>_calibration.png
- runs/calibration/<run>/<dataset>_<split>_calibration.pdf
- runs/calibration/<run>/calibration_<dataset>_<split>_<method>.json for every method

The JSON contains fitted mapping parameters, validation CV scores, test Brier
score/log loss/ECE, raw-uncertainty AUROC, calibrated-probability AUROC, per-bin
counts/failures/confidence intervals, validation u/Dice/failure records, and
test u/Dice/failure/predicted-probability records. AUROC is null if the test set
has only one outcome class. ECE depends on binning and is descriptive; it is not
used to select a mapping on test. Brier and log loss assess overall probability
quality as well as calibration, as explained in the
[scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

For related images, supply --group_key patient_id (or slide_id/video_id) if that
field exists in saved records. Alternatively, --group_file groups.json reads:

~~~json
{
  "dsb2018": {
    "val": {"case_val_001": "patient_A", "case_val_002": "patient_A"},
    "test": {"case_test_001": "patient_B", "case_test_002": "patient_B"}
  }
}
~~~

The mapping must cover every analyzed image. The script checks group independence
across the supplied splits; it does not create or repair dataset splits.
With groups, validation CV keeps whole groups together and confidence intervals
use a cluster bootstrap (--bootstrap_samples 1000, --seed 42). Without groups,
95% Wilson intervals assume independent images. Bins containing fewer than two
independent groups have null interval bounds rather than misleading precision.
These intervals describe test-sampling uncertainty conditional on the fitted
mapping and fixed bin boundaries; they do not include uncertainty from refitting
the calibration mapping.

Run the statistical and file-format regression tests with:

~~~powershell
python -m unittest discover -s tests -p "test_calibration.py" -v
~~~
