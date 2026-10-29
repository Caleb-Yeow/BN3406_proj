"""Overlay predicted masks from every experiment against the ground truth.

Picks the best and worst test cases by baseline mean Dice (or the cases you pass),
shows the axial slice with the most tumour, one column per model.

Usage (from the project folder):
    python src/visualize.py --seed 0
    python src/visualize.py --seed 0 --cases BraTS-GLI-00052-000 BraTS-GLI-00366-000
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd
import torch

from common import load_case, load_config, read_split, run_name
from evaluate import load_model, predict_case

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

# NCR blue, ED orange, ET pink, as in the project brief
CMAP = ListedColormap([(0, 0, 0, 0), (0.0, 0.45, 0.70, 0.6), (0.90, 0.62, 0.0, 0.6), (0.80, 0.47, 0.65, 0.6)])
GRAY = plt.cm.gray.with_extremes(bad="black")  # z-scored background is 0, show it black


def pick_cases(cfg, seed, n):
    f = cfg["paths"]["results_dir"] / f"{run_name('all', seed)}_test_cases.csv"
    if not f.exists():
        return read_split(cfg, "test")[:n]
    df = pd.read_csv(f)
    df["mean_dice"] = df[["dice_ncr", "dice_ed", "dice_et"]].mean(axis=1)
    ranked = df.sort_values("mean_dice")["case"].tolist()
    best_then_worst = ranked[-(n - n // 2):][::-1] + ranked[:n // 2]
    return list(dict.fromkeys(best_then_worst))  # drop repeats when the test set is small


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--n", type=int, default=4, help="number of cases when --cases is not given")
    ap.add_argument("--background", default="t1c", help="modality shown under the overlays")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exps = [e for e in cfg["experiments"]
            if (cfg["paths"]["checkpoints_dir"] / f"{run_name(e, args.seed)}.pt").exists()]
    models = {e: load_model(cfg, e, args.seed, device) for e in exps}
    cases = args.cases or pick_cases(cfg, args.seed, args.n)
    bg = cfg["modalities"].index(args.background)

    cols = ["Ground truth"] + [e if e == "all" else e.replace("no_", "no ") for e in exps]
    fig, axes = plt.subplots(len(cases), len(cols), figsize=(2.2 * len(cols), 2.2 * len(cases)), squeeze=False)
    for r, case in enumerate(cases):
        img, seg = load_case(cfg, case)
        s = int(np.argmax((np.asarray(seg) > 0).sum(axis=(1, 2))))
        masks = [seg[s]] + [predict_case(m, img[s:s + 1], ch, device)[0] for m, ch in models.values()]
        for c, mask in enumerate(masks):
            ax = axes[r, c]
            ax.imshow(np.ma.masked_equal(np.rot90(img[s, bg].astype(np.float32)), 0), cmap=GRAY)
            ax.imshow(np.rot90(mask), cmap=CMAP, vmin=0, vmax=3, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
        axes[r, 0].set_ylabel(case.removeprefix("BraTS-GLI-"), fontsize=8)

    fig.tight_layout()
    out = cfg["paths"]["figures_dir"] / f"overlays_seed{args.seed}.png"
    fig.savefig(out, dpi=200)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
