# Report outline

MICCAI template, 6–8 pages single-column (or 3–4 double-column), excluding references.
Weights are from the grading rubric; sources are the files that feed each section.

| Section | Weight | Content | Source |
|---|---|---|---|
| Abstract | | Question, method, main finding (which modality matters for which sub-region) | |
| 1. Introduction | 10% | Gliomas, multi-modal MRI and what each sequence shows; missing modalities in practice; our question and hypothesis | README |
| 2. Related Work | 10% | U-Net and nnU-Net on BraTS; BraTS 2023 top methods; missing-modality work (e.g. HeMIS, modality dropout, BraTS 2023 synthesis track); prior modality-importance studies | |
| 3. Method | 20% | 2D U-Net architecture (figure), input channels per experiment, Dice + CE loss, training and early stopping | configs/base.yaml, src/train.py |
| 4. Experiments | 25% (with Results) | Patient-level split and counts, preprocessing, augmentation, implementation details (GPU, time per run), metrics and how empty classes are handled | data/splits/, README |
| 5. Results | | Table: mean ± std over 3 seeds for Dice / IoU / HD95 / recall per sub-region, 5 rows. Figure: Dice drop vs baseline. Figure: overlays. Validation curves (appendix) | outputs/results/summary.csv, delta.csv, outputs/figures/ |
| 6. Discussion | 25% | Was the hypothesis supported, per sub-region? Ranking of modalities and significance; best and worst cases and why; seed variability; limitations (2D context, label-wise vs region-wise metrics, single dataset); improvements (3D, modality dropout training, synthesis of missing modality) | ranking.csv, overlays |
| 7. Conclusion | | One paragraph | |
| Clarity and reproducibility | 10% | Fixed seeds, config file, code layout, commands to reproduce | README |

## Results table (from outputs/results/summary.csv, 195 test cases)

| Metric (test, mean ± std over 3 seeds) | all | no T1n | no T1c | no T2w | no FLAIR |
|---|---|---|---|---|---|
| Accuracy | 0.9960 ± 0.0001 | 0.9960 ± 0.0000 | 0.9931 ± 0.0001 | 0.9959 ± 0.0001 | 0.9950 ± 0.0001 |
| Recall background | 0.9990 ± 0.0002 | 0.9990 ± 0.0001 | 0.9987 ± 0.0001 | 0.9990 ± 0.0001 | 0.9985 ± 0.0003 |
| Dice NCR | 0.736 ± 0.003 | 0.722 ± 0.005 | 0.491 ± 0.009 | 0.740 ± 0.006 | 0.735 ± 0.002 |
| Dice ED | 0.833 ± 0.006 | 0.831 ± 0.002 | 0.743 ± 0.003 | 0.826 ± 0.001 | 0.774 ± 0.006 |
| Dice ET | 0.854 ± 0.003 | 0.846 ± 0.005 | 0.511 ± 0.006 | 0.857 ± 0.001 | 0.850 ± 0.005 |
| Dice WT | 0.915 ± 0.004 | 0.915 ± 0.001 | 0.902 ± 0.005 | 0.909 ± 0.001 | 0.875 ± 0.004 |
| Dice TC | 0.880 ± 0.001 | 0.876 ± 0.005 | 0.673 ± 0.008 | 0.884 ± 0.002 | 0.877 ± 0.003 |
| IoU NCR | 0.653 ± 0.005 | 0.639 ± 0.005 | 0.373 ± 0.007 | 0.657 ± 0.006 | 0.651 ± 0.003 |
| IoU ED | 0.736 ± 0.008 | 0.733 ± 0.002 | 0.616 ± 0.004 | 0.726 ± 0.001 | 0.658 ± 0.006 |
| IoU ET | 0.770 ± 0.004 | 0.760 ± 0.006 | 0.369 ± 0.006 | 0.772 ± 0.001 | 0.766 ± 0.005 |
| HD95 NCR (mm) | 16.5 ± 3.1 | 18.4 ± 0.1 | 20.8 ± 1.7 | 16.3 ± 3.3 | 16.6 ± 1.3 |
| HD95 ED (mm) | 7.8 ± 0.4 | 7.9 ± 0.9 | 9.4 ± 0.6 | 7.5 ± 0.4 | 8.6 ± 0.9 |
| HD95 ET (mm) | 8.8 ± 1.6 | 9.8 ± 0.9 | 17.2 ± 0.8 | 10.0 ± 0.5 | 8.2 ± 1.6 |
| Recall NCR | 0.772 ± 0.012 | 0.767 ± 0.008 | 0.511 ± 0.039 | 0.760 ± 0.015 | 0.766 ± 0.009 |
| Recall ED | 0.825 ± 0.024 | 0.822 ± 0.009 | 0.748 ± 0.012 | 0.819 ± 0.003 | 0.760 ± 0.017 |
| Recall ET | 0.861 ± 0.005 | 0.861 ± 0.006 | 0.568 ± 0.009 | 0.858 ± 0.010 | 0.850 ± 0.002 |

WT = whole tumour (NCR+ED+ET), TC = tumour core (NCR+ET). IoU/HD95/recall for WT and TC are also in summary.csv.

Paired Wilcoxon test vs `all` (outputs/results/delta.csv): removing T1c lowers NCR, ED and ET Dice (all p < 1e-28);
removing FLAIR lowers ED Dice (p ≈ 1e-28) and WT Dice (p ≈ 2e-30); removing T1n or T2w changes Dice by ≤ 0.015.

## Figures

| File | Use in |
|---|---|
| outputs/figures/delta_dice.png | Results: Dice drop vs baseline per sub-region (main figure) |
| outputs/figures/overlays_seed0.png | Results/Discussion: 2 best + 2 worst test cases, every model. Colours: blue NCR, orange ED, pink ET. Add a legend in the caption |
| outputs/figures/metrics_overview.png | Results: every metric (Dice, IoU, HD95, recall per sub-region; accuracy, background recall) for all 5 models in one figure |
| outputs/figures/metrics/*.png | The same metrics as separate figures: dice, iou, hd95, recall, accuracy, recall_bg |
| outputs/figures/hd95_boxplot.png | Results: per-case HD95 distribution (medians, robust to empty-region cases) |
| outputs/figures/val_curves.png | Appendix: validation Dice per epoch |

Discussion points from the overlays: 01393-000 and 00259-000 are near-perfect. 01480-000 is a genuine failure (its NCR
is missed by every model and partly labelled ED). Both worst cases are also penalised by the empty-region rule
(01179-000 has no NCR or ET, 01480-000 no ET, so a few false-positive voxels give Dice 0). See report/values.md.

All other numbers for the report (dataset counts, parameters, training time, p-values, per-case statistics):
report/values.md, regenerated by `python src/report_values.py`.
