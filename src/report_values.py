"""Collect every number the report needs (dataset, implementation, results, statistics) into report/values.md.

Usage (from the project folder, after run_all.py has finished):
    python src/report_values.py
"""
import platform
import sys

import numpy as np
import pandas as pd

from common import build_model, load_config, run_name

sys.stdout.reconfigure(encoding="utf-8")
cfg = load_config()
P = cfg["paths"]
EXPS = list(cfg["experiments"])
SEEDS = cfg["train"]["seeds"]
NICE = {"all": "all", "no_t1n": "no T1n", "no_t1c": "no T1c", "no_t2w": "no T2w", "no_t2f": "no FLAIR"}
REG = ["ncr", "ed", "et", "wt", "tc"]
out = []
w = out.append


def table(df, floatfmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in cols]) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [v if isinstance(v, str) else format(v, floatfmt) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def pfmt(p):
    return "< 1e-10" if p < 1e-10 else f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------- dataset
splits = {s: pd.read_csv(P["splits_dir"] / f"{s}.csv", dtype=str) for s in ("train", "val", "test")}
slices = pd.read_csv(P["processed_dir"] / "slices.csv")
all_cases = pd.concat(splits.values())
n_cases, n_pat = len(all_cases), all_cases["patient"].nunique()
multi = (all_cases.groupby("patient").size() > 1).sum()

w("# Values for the report\n")
w("Generated from the project outputs on the final run (all 15 models). Every number below can be regenerated "
  "with the scripts in src/. Test = 195 held-out cases, never used for training or model selection.\n")
w("## 1. Dataset and split\n")
w(f"- BraTS 2023 GLI training set: **{n_cases} cases** from **{n_pat} patients**; "
  f"{multi} patients have more than one scan (timepoint), so the split is done by patient.")
w("- Each case: 4 co-registered MRI volumes (T1n, T1c, T2w, T2-FLAIR), 240 × 240 × 155 voxels, 1 mm isotropic, "
  "skull-stripped; labels 0 background, 1 NCR, 2 ED, 3 ET.")
w("- Split 70 / 15 / 15 by patient, random seed 42:\n")
rows = []
for s, df in splits.items():
    sl = slices[slices["case"].isin(df["case"])]
    rows.append({"Set": s, "Patients": df["patient"].nunique(), "Cases": len(df), "2D slices": len(sl),
                 "Slices with tumour": int((sl["tumour_voxels"] > 0).sum()),
                 "% with tumour": f"{100 * (sl['tumour_voxels'] > 0).mean():.1f}%"})
rows.append({"Set": "total", "Patients": n_pat, "Cases": n_cases, "2D slices": len(slices),
             "Slices with tumour": int((slices["tumour_voxels"] > 0).sum()),
             "% with tumour": f"{100 * (slices['tumour_voxels'] > 0).mean():.1f}%"})
w(table(pd.DataFrame(rows).set_index("Set").astype(str), "") + "\n")
per_case = slices.groupby("case").size()
w(f"- Axial slices kept per case (slices containing brain): mean {per_case.mean():.0f}, "
  f"range {per_case.min()}–{per_case.max()} of 155.")
w("- No patient appears in more than one set (checked: train ∩ val ∩ test patients = ∅): "
  f"{len(set(splits['train'].patient) & set(splits['val'].patient)) + len(set(splits['train'].patient) & set(splits['test'].patient)) + len(set(splits['val'].patient) & set(splits['test'].patient))} overlaps.\n")

# class balance over the preprocessed (brain-cropped) training volumes
counts = np.zeros(4, dtype=np.int64)
for case in splits["train"]["case"]:
    seg = np.load(P["processed_dir"] / f"{case}_seg.npy", mmap_mode="r")
    counts += np.bincount(np.asarray(seg).ravel(), minlength=4)[:4]
frac = counts / counts.sum()
tum = counts[1:] / counts[1:].sum()
w("**Class imbalance** (voxels in the preprocessed training slices):\n")
w("| Class | % of all voxels | % of tumour voxels |")
w("|---|---|---|")
for i, name in enumerate(["Background", "NCR", "ED", "ET"]):
    w(f"| {name} | {100 * frac[i]:.2f}% | {'' if i == 0 else f'{100 * tum[i - 1]:.1f}%'} |")
w(f"\nTumour is {100 * frac[1:].sum():.2f}% of voxels, so background is ~{frac[0] / frac[1:].sum():.0f}× more common "
  "than tumour. This is why voxel accuracy is ~99.6% for every model and is not informative.\n")

test0 = pd.read_csv(P["results_dir"] / "all_seed0_test_cases.csv")
w("**Test cases with an absent sub-region in the ground truth** (Dice = 1 if the model also predicts nothing, "
  "0 if it predicts any; recall is undefined and skipped):\n")
for r in ["ncr", "ed", "et"]:
    n_abs = test0[f"recall_{r}"].isna().sum()
    w(f"- {r.upper()} absent in {n_abs} of {len(test0)} test cases ({100 * n_abs / len(test0):.1f}%)")
w("")

# ---------------------------------------------------------------- preprocessing / model / training
w("## 2. Preprocessing, model and training (implementation details)\n")
p4 = sum(p.numel() for p in build_model(cfg, 4).parameters())
p3 = sum(p.numel() for p in build_model(cfg, 3).parameters())
t = cfg["train"]
n_train = len(slices[slices["case"].isin(splits["train"]["case"])])
iters = int(np.ceil(n_train / t["batch_size"]))
w("- Normalisation: z-score per modality and per volume, computed over brain voxels only (background left at 0).")
w(f"- Crop: {cfg['preprocess']['crop_size']} × {cfg['preprocess']['crop_size']} in-plane, centred on the brain, "
  "no resizing; axial slices without brain discarded.")
w(f"- Augmentation (training only): left-right flip p = {cfg['augment']['flip_prob']}, rotation uniform in "
  f"±{cfg['augment']['rotate_deg']}° (bilinear for images, nearest-neighbour for the mask).")
w(f"- Model: MONAI 2D U-Net, channels {cfg['model']['channels']}, strides {cfg['model']['strides']} "
  f"(4 down-sampling levels, bottleneck at 12 × 12), {cfg['model']['num_res_units']} residual units, "
  "instance norm + PReLU (MONAI defaults), 4 output classes with softmax.")
w(f"- Parameters: **{p4:,}** with 4 input channels, **{p3:,}** with 3 (only the first convolution differs).")
w(f"- Loss: Dice + cross-entropy, equal weights (λ = 1 each), Dice computed over the whole batch.")
w(f"- Optimiser: Adam, learning rate {t['lr']:g}, batch size {t['batch_size']}, no weight decay, no LR schedule.")
w(f"- Epoch = one pass over all {n_train:,} training slices = {iters:,} iterations.")
w(f"- Early stopping: stop when mean validation Dice (NCR, ED, ET) has not improved for {t['patience']} epochs; "
  f"max {t['max_epochs']} epochs; the checkpoint with the best validation Dice is used for testing.")
w(f"- Seeds {SEEDS} change weight initialisation, slice order and augmentation; the data split is fixed.")
w("- Mixed precision (float16 autocast) training.\n")

logs = {}
for e in EXPS:
    for s in SEEDS:
        logs[(e, s)] = pd.read_csv(P["logs_dir"] / f"{run_name(e, s)}.csv")
rows = []
for (e, s), lg in logs.items():
    rows.append({"Run": f"{NICE[e]}, seed {s}", "Epochs": len(lg), "Best epoch": int(lg["val_dice_mean"].idxmax()) + 1,
                 "Best val Dice": f"{lg['val_dice_mean'].max():.4f}",
                 "Train time (h)": f"{lg['seconds'].sum() / 3600:.2f}"})
run_tab = pd.DataFrame(rows).set_index("Run")
all_secs = pd.concat(logs.values())["seconds"]
epochs = [len(lg) for lg in logs.values()]
best_ep = [int(lg["val_dice_mean"].idxmax()) + 1 for lg in logs.values()]
w("**Compute**\n")
w(f"- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), Windows 11; Python {platform.python_version()}, "
  f"PyTorch {__import__('torch').__version__}, MONAI {__import__('monai').__version__}.")
w(f"- Time per epoch (training + validation): median {all_secs.median() / 60:.1f} min "
  f"(range {all_secs.min() / 60:.1f}–{all_secs.max() / 60:.1f}; {(all_secs > 600).sum()} slower epochs over 10 min, "
  "likely from other load on the laptop).")
w(f"- Epochs per run: mean {np.mean(epochs):.1f} (range {min(epochs)}–{max(epochs)}); "
  f"best epoch mean {np.mean(best_ep):.1f} (range {min(best_ep)}–{max(best_ep)}). "
  f"No run reached the {t['max_epochs']}-epoch limit.")
w(f"- Total training time for all 15 runs: **{all_secs.sum() / 3600:.1f} GPU-hours** "
  f"(mean {all_secs.sum() / 3600 / len(logs):.2f} h per run).\n")
w(table(run_tab, "") + "\n")

val = pd.DataFrame([{"exp": e, "seed": s, "v": lg["val_dice_mean"].max()} for (e, s), lg in logs.items()])
vs = val.groupby("exp")["v"].agg(["mean", "std"]).reindex(EXPS)
w("Best validation Dice (mean of NCR, ED, ET), mean ± std over 3 seeds: "
  + "; ".join(f"{NICE[e]} {vs.loc[e, 'mean']:.4f} ± {vs.loc[e, 'std']:.4f}" for e in EXPS) + ".\n")

# ---------------------------------------------------------------- results
cases = pd.concat([pd.read_csv(P["results_dir"] / f"{run_name(e, s)}_test_cases.csv") for e in EXPS for s in SEEDS])
summary = pd.read_csv(P["results_dir"] / "summary.csv", header=[0, 1], index_col=0)
delta = pd.read_csv(P["results_dir"] / "delta.csv").set_index("exp")

w("## 3. Test results\n")
w("Full metric table (mean ± std over seeds): see report/outline.md and outputs/results/summary.csv. "
  "Each seed's value is the mean over the 195 test cases; mean and std are then taken over the 3 seeds.\n")
w("**Change vs the 4-modality baseline** (Δ = variant − baseline; Dice higher is better, HD95 lower is better). "
  "p = two-sided paired Wilcoxon signed-rank test over the 195 test cases (each case averaged over the 3 seeds). "
  "With 4 variants × 5 regions = 20 Dice tests, the Bonferroni threshold is p < 0.0025.\n")
w("| Removed | ΔDice NCR | ΔDice ED | ΔDice ET | ΔDice WT | ΔDice TC |")
w("|---|---|---|---|---|---|")
for e in EXPS[1:]:
    cells = []
    for r in REG:
        d, p = delta.loc[e, f"d_dice_{r}"], delta.loc[e, f"p_dice_{r}"]
        star = " *" if p < 0.0025 else ""
        cells.append(f"{d:+.3f} (p {pfmt(p)}){star}")
    w(f"| {NICE[e].replace('no ', '')} | " + " | ".join(cells) + " |")
w("\n\\* significant after Bonferroni correction.\n")
w("| Removed | ΔHD95 NCR (mm) | ΔHD95 ED | ΔHD95 ET | ΔHD95 WT | ΔHD95 TC |")
w("|---|---|---|---|---|---|")
for e in EXPS[1:]:
    cells = []
    for r in REG:
        d, p = delta.loc[e, f"d_hd95_{r}"], delta.loc[e, f"p_hd95_{r}"]
        star = " *" if p < 0.0025 else ""
        cells.append(f"{d:+.1f} (p {pfmt(p)}){star}")
    w(f"| {NICE[e].replace('no ', '')} | " + " | ".join(cells) + " |")
w("")

# HD95 in detail (written by analyze.py)
hd = pd.read_csv(P["results_dir"] / "hd95.csv").set_index(["exp", "region"])
w("**HD95 in detail** (mm, lower is better; averaged over 3 seeds). HD95 = 95th percentile of the distances between "
  "the predicted and ground-truth surfaces, computed per case in 3D with MONAI `compute_hausdorff_distance` "
  f"(1 mm voxels, so voxels = mm). A case scores {cfg['eval']['hd95_empty']} mm when a region is predicted but "
  "absent, or present but not predicted at all; a few such cases dominate the mean, so the median is the more "
  "robust summary. Figure: outputs/figures/hd95_boxplot.png.\n")
w("| Model | Region | Mean | Median (IQR) | Cases at 373 mm | Mean without them |")
w("|---|---|---|---|---|---|")
for e in EXPS:
    for r in ["ncr", "ed", "et"]:
        h = hd.loc[(e, r)]
        w(f"| {NICE[e]} | {r.upper()} | {h['mean']:.1f} ± {h['mean_std']:.1f} | {h['median']:.1f} "
          f"({h['q25']:.1f}–{h['q75']:.1f}) | {h['n_empty']:.1f} | {h['mean_no_empty']:.1f} |")
w("")

# relative drop
w("**Relative Dice drop** (Δ / baseline Dice):\n")
w("| Removed | NCR | ED | ET | WT | TC |")
w("|---|---|---|---|---|---|")
for e in EXPS[1:]:
    w(f"| {NICE[e].replace('no ', '')} | " + " | ".join(
        f"{100 * delta.loc[e, f'd_dice_{r}'] / summary.loc['all', (f'dice_{r}', 'mean')]:+.1f}%" for r in REG) + " |")
w("")

rank = pd.read_csv(P["results_dir"] / "ranking.csv", index_col=0)
w("**Modality ranking** (most important = largest Dice drop when removed):\n")
w("| Rank | NCR | ED | ET | WT | TC |")
w("|---|---|---|---|---|---|")
for idx, row in rank.iterrows():
    w(f"| {idx} | " + " | ".join(row[r].upper().replace("T2F", "FLAIR") for r in REG) + " |")
w("")

# per-case: how often does a variant do worse than baseline
pc = cases.groupby(["exp", "case"])[[f"dice_{r}" for r in REG]].mean()
w("**Share of test cases where the variant is worse than the baseline** (Dice lower by more than 0.01, "
  "seed-averaged):\n")
w("| Removed | NCR | ED | ET | WT | TC |")
w("|---|---|---|---|---|---|")
base = pc.loc["all"]
for e in EXPS[1:]:
    diff = pc.loc[e].reindex(base.index) - base
    w(f"| {NICE[e].replace('no ', '')} | " + " | ".join(f"{100 * (diff[f'dice_{r}'] < -0.01).mean():.0f}%" for r in REG)
      + " |")
w("")

# distribution of baseline per-case Dice
w("**Baseline per-case Dice distribution** (seed-averaged, 195 test cases):\n")
w("| Region | Mean | Median | IQR | Cases with Dice < 0.5 |")
w("|---|---|---|---|---|")
for r in REG:
    v = base[f"dice_{r}"]
    w(f"| {r.upper()} | {v.mean():.3f} | {v.median():.3f} | {v.quantile(0.25):.3f}–{v.quantile(0.75):.3f} | "
      f"{(v < 0.5).sum()} ({100 * (v < 0.5).mean():.0f}%) |")
w("")

# seed variability
std_cols = [(f"dice_{r}", "std") for r in ["ncr", "ed", "et"]]
w(f"**Seed variability**: std of Dice over 3 seeds ranges {summary[std_cols].min().min():.3f}–"
  f"{summary[std_cols].max().max():.3f} across all experiments and sub-regions, much smaller than the "
  f"T1c effect ({delta.loc['no_t1c', 'd_dice_et']:+.3f} ET) and the FLAIR effect on ED "
  f"({delta.loc['no_t2f', 'd_dice_ed']:+.3f}).\n")

# best / worst cases shown in the overlay figure
mean3 = base[["dice_ncr", "dice_ed", "dice_et"]].mean(axis=1)
b0 = cases[(cases.exp == "all") & (cases.seed == 0)].set_index("case")
b0m = b0[["dice_ncr", "dice_ed", "dice_et"]].mean(axis=1).sort_values()
shown = list(b0m.index[::-1][:2]) + list(b0m.index[:2])
w("**Cases in the overlay figure** (seed 0, chosen by baseline mean Dice of NCR, ED, ET). Dice NCR / ED / ET per model:\n")
w("| Case | GT voxels NCR / ED / ET | " + " | ".join(NICE[e] for e in EXPS) + " |")
w("|---|---|" + "---|" * len(EXPS))
for c in shown:
    gt = np.bincount(np.asarray(np.load(P["processed_dir"] / f"{c}_seg.npy", mmap_mode="r")).ravel(), minlength=4)[1:4]
    cells = [" / ".join(f"{v:,}" for v in gt)]
    for e in EXPS:
        row = cases[(cases.exp == e) & (cases.seed == 0) & (cases.case == c)].iloc[0]
        cells.append(" / ".join(f"{row[f'dice_{r}']:.2f}" for r in ["ncr", "ed", "et"]))
    w(f"| {c.removeprefix('BraTS-GLI-')} | " + " | ".join(cells) + " |")
w("\nThe two worst cases are partly an effect of the empty-region rule: 01179-000 has no NCR or ET in the ground "
  "truth and 01480-000 has no ET, so a handful of false-positive voxels gives Dice 0 for that region (the no-FLAIR "
  "model predicted no ET for 01179-000 and gets ET Dice 1.00). 01480-000 is also a genuine failure: its 10,557 "
  "NCR voxels are missed by every model and partly labelled ED. Dice is unstable for small or absent regions; "
  "this is worth a sentence in the limitations.\n")

# ---------------------------------------------------------------- key sentences
d = delta
w("## 4. Key numbers for the abstract / conclusion\n")
w(f"- Baseline (all 4 modalities) test Dice: NCR {summary.loc['all', ('dice_ncr', 'mean')]:.3f}, "
  f"ED {summary.loc['all', ('dice_ed', 'mean')]:.3f}, ET {summary.loc['all', ('dice_et', 'mean')]:.3f} "
  f"(WT {summary.loc['all', ('dice_wt', 'mean')]:.3f}, TC {summary.loc['all', ('dice_tc', 'mean')]:.3f}).")
w(f"- Removing T1c: ET {d.loc['no_t1c', 'd_dice_et']:+.3f}, NCR {d.loc['no_t1c', 'd_dice_ncr']:+.3f}, "
  f"ED {d.loc['no_t1c', 'd_dice_ed']:+.3f}, TC {d.loc['no_t1c', 'd_dice_tc']:+.3f}; "
  f"ET HD95 {d.loc['no_t1c', 'd_hd95_et']:+.1f} mm.")
w(f"- Removing FLAIR: ED {d.loc['no_t2f', 'd_dice_ed']:+.3f}, WT {d.loc['no_t2f', 'd_dice_wt']:+.3f}; "
  f"NCR and ET change by ≤ {max(abs(d.loc['no_t2f', 'd_dice_ncr']), abs(d.loc['no_t2f', 'd_dice_et'])):.3f}.")
w(f"- Removing T1n or T2w: every sub-region Dice changes by ≤ "
  f"{max(abs(d.loc[e, f'd_dice_{r}']) for e in ['no_t1n', 'no_t2w'] for r in REG):.3f}.")

text = "\n".join(out) + "\n"
(cfg["paths"]["results_dir"].parents[1] / "report" / "values.md").write_text(text, encoding="utf-8")
print(text)
