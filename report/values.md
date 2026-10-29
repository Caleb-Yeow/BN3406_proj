# Values for the report

Generated from the project outputs on the final run (all 15 models). Every number below can be regenerated with the scripts in src/. Test = 195 held-out cases, never used for training or model selection.

## 1. Dataset and split

- BraTS 2023 GLI training set: **1251 cases** from **1133 patients**; 118 patients have more than one scan (timepoint), so the split is done by patient.
- Each case: 4 co-registered MRI volumes (T1n, T1c, T2w, T2-FLAIR), 240 × 240 × 155 voxels, 1 mm isotropic, skull-stripped; labels 0 background, 1 NCR, 2 ED, 3 ET.
- Split 70 / 15 / 15 by patient, random seed 42:

| Set | Patients | Cases | 2D slices | Slices with tumour | % with tumour |
|---|---|---|---|---|---|
| train | 793 | 869 | 121004 | 56080 | 46.3% |
| val | 170 | 187 | 26043 | 12286 | 47.2% |
| test | 170 | 195 | 27074 | 13071 | 48.3% |
| total | 1133 | 1251 | 174121 | 81437 | 46.8% |

- Axial slices kept per case (slices containing brain): mean 139, range 98–155 of 155.
- No patient appears in more than one set (checked: train ∩ val ∩ test patients = ∅): 0 overlaps.

**Class imbalance** (voxels in the preprocessed training slices):

| Class | % of all voxels | % of tumour voxels |
|---|---|---|
| Background | 98.15% |  |
| NCR | 0.27% | 14.7% |
| ED | 1.16% | 62.3% |
| ET | 0.42% | 22.9% |

Tumour is 1.85% of voxels, so background is ~53× more common than tumour. This is why voxel accuracy is ~99.6% for every model and is not informative.

**Test cases with an absent sub-region in the ground truth** (Dice = 1 if the model also predicts nothing, 0 if it predicts any; recall is undefined and skipped):

- NCR absent in 4 of 195 test cases (2.1%)
- ED absent in 0 of 195 test cases (0.0%)
- ET absent in 2 of 195 test cases (1.0%)

## 2. Preprocessing, model and training (implementation details)

- Normalisation: z-score per modality and per volume, computed over brain voxels only (background left at 0).
- Crop: 192 × 192 in-plane, centred on the brain, no resizing; axial slices without brain discarded.
- Augmentation (training only): left-right flip p = 0.5, rotation uniform in ±10° (bilinear for images, nearest-neighbour for the mask).
- Model: MONAI 2D U-Net, channels [32, 64, 128, 256, 512], strides [2, 2, 2, 2] (4 down-sampling levels, bottleneck at 12 × 12), 0 residual units, instance norm + PReLU (MONAI defaults), 4 output classes with softmax.
- Parameters: **2,640,460** with 4 input channels, **2,640,172** with 3 (only the first convolution differs).
- Loss: Dice + cross-entropy, equal weights (λ = 1 each), Dice computed over the whole batch.
- Optimiser: Adam, learning rate 0.0001, batch size 8, no weight decay, no LR schedule.
- Epoch = one pass over all 121,004 training slices = 15,126 iterations.
- Early stopping: stop when mean validation Dice (NCR, ED, ET) has not improved for 5 epochs; max 40 epochs; the checkpoint with the best validation Dice is used for testing.
- Seeds [0, 1, 2] change weight initialisation, slice order and augmentation; the data split is fixed.
- Mixed precision (float16 autocast) training.

**Compute**

- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), Windows 11; Python 3.11.15, PyTorch 2.13.0+cu126, MONAI 1.6.0.
- Time per epoch (training + validation): median 5.8 min (range 4.9–18.6; 4 slower epochs over 10 min, likely from other load on the laptop).
- Epochs per run: mean 13.4 (range 9–17); best epoch mean 8.4 (range 4–12). No run reached the 40-epoch limit.
- Total training time for all 15 runs: **20.4 GPU-hours** (mean 1.36 h per run).

| Run | Epochs | Best epoch | Best val Dice | Train time (h) |
|---|---|---|---|---|
| all, seed 0 | 10 | 5 | 0.8566 | 1.16 |
| all, seed 1 | 14 | 9 | 0.8552 | 1.50 |
| all, seed 2 | 13 | 8 | 0.8585 | 1.23 |
| no T1n, seed 0 | 17 | 12 | 0.8540 | 1.68 |
| no T1n, seed 1 | 16 | 11 | 0.8570 | 1.61 |
| no T1n, seed 2 | 13 | 8 | 0.8579 | 1.63 |
| no T1c, seed 0 | 15 | 10 | 0.6674 | 1.52 |
| no T1c, seed 1 | 10 | 5 | 0.6694 | 0.94 |
| no T1c, seed 2 | 9 | 4 | 0.6578 | 1.16 |
| no T2w, seed 0 | 14 | 9 | 0.8484 | 1.47 |
| no T2w, seed 1 | 14 | 9 | 0.8483 | 1.22 |
| no T2w, seed 2 | 14 | 9 | 0.8498 | 1.36 |
| no FLAIR, seed 0 | 12 | 7 | 0.8475 | 1.22 |
| no FLAIR, seed 1 | 17 | 12 | 0.8460 | 1.48 |
| no FLAIR, seed 2 | 13 | 8 | 0.8477 | 1.19 |

Best validation Dice (mean of NCR, ED, ET), mean ± std over 3 seeds: all 0.8568 ± 0.0016; no T1n 0.8563 ± 0.0021; no T1c 0.6648 ± 0.0062; no T2w 0.8488 ± 0.0009; no FLAIR 0.8471 ± 0.0010.

## 3. Test results

Full metric table (mean ± std over seeds): see report/outline.md and outputs/results/summary.csv. Each seed's value is the mean over the 195 test cases; mean and std are then taken over the 3 seeds.

**Change vs the 4-modality baseline** (Δ = variant − baseline; Dice higher is better, HD95 lower is better). p = two-sided paired Wilcoxon signed-rank test over the 195 test cases (each case averaged over the 3 seeds). With 4 variants × 5 regions = 20 Dice tests, the Bonferroni threshold is p < 0.0025.

| Removed | ΔDice NCR | ΔDice ED | ΔDice ET | ΔDice WT | ΔDice TC |
|---|---|---|---|---|---|
| T1n | -0.014 (p 0.194) | -0.002 (p 0.113) | -0.007 (p 3.6e-06) * | -0.000 (p 0.969) | -0.004 (p 3.1e-04) * |
| T1c | -0.245 (p < 1e-10) * | -0.090 (p < 1e-10) * | -0.343 (p < 1e-10) * | -0.013 (p < 1e-10) * | -0.207 (p < 1e-10) * |
| T2w | +0.004 (p 0.700) | -0.007 (p 2.7e-04) * | +0.003 (p 0.675) | -0.006 (p 6.4e-07) * | +0.004 (p 0.886) |
| FLAIR | -0.002 (p 0.663) | -0.059 (p < 1e-10) * | -0.004 (p 0.041) | -0.040 (p < 1e-10) * | -0.003 (p 0.061) |

\* significant after Bonferroni correction.

| Removed | ΔHD95 NCR (mm) | ΔHD95 ED | ΔHD95 ET | ΔHD95 WT | ΔHD95 TC |
|---|---|---|---|---|---|
| T1n | +1.9 (p 0.251) | +0.1 (p 0.616) | +1.0 (p 0.042) | -0.4 (p 0.466) | -0.3 (p 0.025) |
| T1c | +4.3 (p < 1e-10) * | +1.6 (p < 1e-10) * | +8.4 (p < 1e-10) * | +0.0 (p 9.7e-04) * | +6.2 (p < 1e-10) * |
| T2w | -0.2 (p 0.197) | -0.4 (p 0.536) | +1.1 (p 0.470) | -1.0 (p 0.141) | -0.8 (p 0.150) |
| FLAIR | +0.0 (p 0.967) | +0.8 (p 2.6e-10) * | -0.7 (p 0.852) | +0.4 (p 6.6e-07) * | -0.7 (p 0.657) |

**Relative Dice drop** (Δ / baseline Dice):

| Removed | NCR | ED | ET | WT | TC |
|---|---|---|---|---|---|
| T1n | -1.9% | -0.2% | -0.9% | -0.0% | -0.5% |
| T1c | -33.3% | -10.8% | -40.2% | -1.5% | -23.5% |
| T2w | +0.5% | -0.8% | +0.3% | -0.7% | +0.4% |
| FLAIR | -0.2% | -7.1% | -0.4% | -4.4% | -0.3% |

**Modality ranking** (most important = largest Dice drop when removed):

| Rank | NCR | ED | ET | WT | TC |
|---|---|---|---|---|---|
| #1 | T1C | T1C | T1C | FLAIR | T1C |
| #2 | T1N | FLAIR | T1N | T1C | T1N |
| #3 | FLAIR | T2W | FLAIR | T2W | FLAIR |
| #4 | T2W | T1N | T2W | T1N | T2W |

**Share of test cases where the variant is worse than the baseline** (Dice lower by more than 0.01, seed-averaged):

| Removed | NCR | ED | ET | WT | TC |
|---|---|---|---|---|---|
| T1n | 25% | 22% | 22% | 10% | 23% |
| T1c | 91% | 96% | 99% | 44% | 96% |
| T2w | 22% | 26% | 14% | 17% | 15% |
| FLAIR | 22% | 89% | 17% | 89% | 18% |

**Baseline per-case Dice distribution** (seed-averaged, 195 test cases):

| Region | Mean | Median | IQR | Cases with Dice < 0.5 |
|---|---|---|---|---|
| NCR | 0.736 | 0.875 | 0.656–0.935 | 38 (19%) |
| ED | 0.833 | 0.888 | 0.778–0.927 | 7 (4%) |
| ET | 0.854 | 0.895 | 0.843–0.942 | 10 (5%) |
| WT | 0.915 | 0.939 | 0.893–0.963 | 0 (0%) |
| TC | 0.880 | 0.946 | 0.884–0.966 | 11 (6%) |

**Seed variability**: std of Dice over 3 seeds ranges 0.001–0.009 across all experiments and sub-regions, much smaller than the T1c effect (-0.343 ET) and the FLAIR effect on ED (-0.059).

**Cases in the overlay figure** (seed 0, chosen by baseline mean Dice of NCR, ED, ET). Dice NCR / ED / ET per model:

| Case | GT voxels NCR / ED / ET | all | no T1n | no T1c | no T2w | no FLAIR |
|---|---|---|---|---|---|---|
| 01393-000 | 4,462 / 49,804 / 12,461 | 0.98 / 0.97 / 0.96 | 0.98 / 0.97 / 0.97 | 0.81 / 0.94 / 0.82 | 0.98 / 0.97 / 0.97 | 0.98 / 0.94 / 0.96 |
| 00259-000 | 9,107 / 75,181 / 15,306 | 0.96 / 0.97 / 0.96 | 0.97 / 0.97 / 0.96 | 0.91 / 0.93 / 0.75 | 0.96 / 0.97 / 0.96 | 0.97 / 0.94 / 0.96 |
| 01480-000 | 10,557 / 24,143 / 0 | 0.00 / 0.45 / 0.00 | 0.00 / 0.47 / 0.00 | 0.03 / 0.49 / 0.00 | 0.00 / 0.37 / 0.00 | 0.00 / 0.36 / 0.00 |
| 01179-000 | 0 / 105,314 / 0 | 0.00 / 0.72 / 0.00 | 0.00 / 0.67 / 0.00 | 0.00 / 0.74 / 0.00 | 0.00 / 0.77 / 0.00 | 0.00 / 0.68 / 1.00 |

The two worst cases are partly an effect of the empty-region rule: 01179-000 has no NCR or ET in the ground truth and 01480-000 has no ET, so a handful of false-positive voxels gives Dice 0 for that region (the no-FLAIR model predicted no ET for 01179-000 and gets ET Dice 1.00). 01480-000 is also a genuine failure: its 10,557 NCR voxels are missed by every model and partly labelled ED. Dice is unstable for small or absent regions; this is worth a sentence in the limitations.

## 4. Key numbers for the abstract / conclusion

- Baseline (all 4 modalities) test Dice: NCR 0.736, ED 0.833, ET 0.854 (WT 0.915, TC 0.880).
- Removing T1c: ET -0.343, NCR -0.245, ED -0.090, TC -0.207; ET HD95 +8.4 mm.
- Removing FLAIR: ED -0.059, WT -0.040; NCR and ET change by ≤ 0.004.
- Removing T1n or T2w: every sub-region Dice changes by ≤ 0.014.
