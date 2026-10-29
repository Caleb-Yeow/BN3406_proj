# BN3406 Group 4: Which MRI modalities does a glioma U-Net rely on?

BraTS 2023 Adult Glioma Segmentation (project Task 4), with a leave-one-modality-out study.
Report due **5 Dec 2026**. Presentation on **12 Nov 2026**.

**Research question.** How does the segmentation accuracy of a 2D U-Net for glioma sub-regions (NCR, ED, ET)
change when each MRI modality (T1n, T1c, T2w, T2-FLAIR) is removed, and which modalities does the model rely
on most for each sub-region?

**Hypothesis.** Removing T1c hurts ET Dice the most (T1c shows contrast enhancement, which defines ET).
Removing FLAIR hurts ED Dice the most (FLAIR shows edema and infiltration). T1n matters least overall, and
removing T2w alone causes only a small drop because it overlaps with FLAIR.

Labels: 0 background, 1 NCR (necrotic core), 2 ED (edema), 3 ET (enhancing tumour).

## Experiments

| Experiment | Input channels | Role |
|---|---|---|
| `all`    | T1n, T1c, T2w, T2-FLAIR | baseline |
| `no_t1n` | T1c, T2w, T2-FLAIR | variant |
| `no_t1c` | T1n, T2w, T2-FLAIR | variant |
| `no_t2w` | T1n, T1c, T2-FLAIR | variant |
| `no_t2f` | T1n, T1c, T2w | variant |

Each is trained with seeds 0, 1, 2, so there are **15 runs**. Only the input channels change between experiments;
the split, preprocessing, augmentation, architecture, loss and optimiser are identical (all in
[configs/base.yaml](configs/base.yaml)).

| Item | Choice |
|---|---|
| Split | 70 / 15 / 15 by patient, seed 42. Cases are grouped by patient ID (`BraTS-GLI-00052-000` and `-001` are the same patient), so no patient appears in two sets |
| Preprocessing | Z-score per modality over brain voxels (background stays 0), 192 x 192 crop around the brain, no resizing, axial slices with brain only |
| Augmentation | Left-right flip (p = 0.5), rotation ±10°, same transform for image and mask |
| Model | MONAI 2D U-Net, channels 32 → 512, 4 downsampling levels, no residual units, `in_channels` = number of modalities, 4 output classes |
| Loss | Dice + cross-entropy, equal weights |
| Training | Adam, lr 1e-4, batch 8, up to 40 full epochs (all ~121k training slices each), early stopping (patience 5) on mean validation Dice |
| Model selection | Checkpoint with the best mean validation Dice (NCR, ED, ET). The test set is used once, at the end |
| Test metrics | Per case in 3D: Dice, IoU, HD95, recall for NCR / ED / ET, background recall, voxel accuracy, plus BraTS WT and TC Dice |
| Extra analyses | 3 seeds (mean ± std); per-sub-region Dice and HD95; Dice drop vs baseline with paired Wilcoxon test; modality ranking; overlay figures of best and worst cases |

## Layout

```
configs/base.yaml        all settings, including the 5 experiments
data/raw/                BraTS 2023 GLI training cases, unzipped (one folder per case)
data/splits/             train.csv, val.csv, test.csv (case, patient)
data/processed/          <case>_img.npy (D,4,192,192) float16, <case>_seg.npy, slices.csv
src/
  check_data.py          confirm all cases have t1n/t1c/t2w/t2f/seg
  make_splits.py         patient-level 70/15/15 split
  preprocess.py          normalise, crop, slice
  dataset.py             2D slice dataset, picks modality channels, augmentation
  common.py              config, model, helpers
  train.py               train one experiment + seed
  evaluate.py            3D per-case test metrics
  analyze.py             summary, Dice drop vs baseline, ranking, plots
  visualize.py           ground-truth vs prediction overlays for every experiment
  report_values.py       every number for the report -> report/values.md
  run_all.py             train + evaluate all 15 runs, then analyse (resumes where it stopped)
checkpoints/             <exp>_seed<k>.pt
outputs/logs/            per-epoch training logs (CSV)
outputs/results/         per-case test metrics, summary.csv, delta.csv, ranking.csv
outputs/figures/         delta_dice.png, val_curves.png, overlays_seed<k>.png
notebooks/               exploration
report/                  MICCAI report (see report/outline.md)
slides/                  presentation
```

## How to run

Use the `BN3406` conda env (PyTorch + CUDA, MONAI, nibabel). From this folder:

```powershell
conda activate BN3406

# 1. Download the data (link below), unzip the GLI training set into data/raw/, then check it
python src/check_data.py

# 2. Split and preprocess (once)
python src/make_splits.py
python src/preprocess.py --workers 4

# 3. Quick sanity run before committing GPU time
python src/train.py --exp all --seed 0

# 4. Everything: 15 runs + evaluation + analysis. Safe to stop and restart
python src/run_all.py

# Or split runs across group members' machines (same splits, same config), then copy
# checkpoints/ and outputs/ back together and run analyze.py
python src/run_all.py --seeds 1
python src/analyze.py
python src/visualize.py --seed 0
```

Dataset: https://nusu-my.sharepoint.com/:f:/g/personal/e0920791_u_nus_edu/IgBRO1z1vKP9SqA7Y-yEJK0ZAbqM3w8DQ1XO9gUjW6oO1PA?e=kchpdy

## Compute and disk (measured on an RTX 4050 Laptop, 6 GB)

- A full epoch is ~15,000 iterations + validation on ~26k slices: about 7.5 min with mixed precision,
  under 1 GB of GPU memory. A run is at most 40 epochs (~5 h); early stopping usually ends it sooner.
  All 15 runs take roughly a day on this laptop.
- Every epoch saves `checkpoints/<run>_last.pt`. If training stops (sleep, crash, restart), run
  `run_all.py` again: finished runs are skipped and the interrupted run resumes from its last epoch.
- `data/processed/` is 54 GB (1,251 cases, 174,121 brain slices, float16). Check free space first.
- Keep the laptop plugged in and stop it from sleeping while training, or the run pauses.
- On Windows each DataLoader worker takes ~20 s to start. Set `num_workers: 0` in the config for quick debugging.

## Notes

- Voxel accuracy is dominated by background (most voxels are healthy brain) and will look high for every model.
  Report it because the group template asks for it, but base conclusions on per-class Dice, HD95 and recall.
- HD95 is in mm (BraTS is 1 mm isotropic). If exactly one of prediction and ground truth is empty, HD95 = 373.13
  (BraTS convention); if both are empty, Dice = 1 and HD95 = 0.
- Some cases have no NCR or ET at all. Recall for an absent class is left blank (NaN) and skipped in averages.
- The decision sheet says "the full 5 modalities"; it should read 4. There are 5 *models*: 1 baseline + 4 variants.
