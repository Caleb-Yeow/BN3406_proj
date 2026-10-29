"""Aggregate test results across experiments and seeds.

Writes to outputs/results/:
  summary.csv  mean and std over seeds of each metric, per experiment
  delta.csv    change vs the 4-modality baseline, with a paired Wilcoxon test over test cases
  ranking.csv  modalities ranked by Dice drop for each sub-region
And to outputs/figures/: delta_dice.png, val_curves.png

Usage (from the project folder):
    python src/analyze.py
"""
import argparse

import matplotlib
import pandas as pd
from scipy.stats import wilcoxon

from common import CLASS_NAMES, load_config, run_name

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASELINE = "all"
REGIONS = list(CLASS_NAMES.values())
KEY_METRICS = (["accuracy", "recall_bg"]
               + [f"{m}_{r}" for m in ("dice", "iou", "hd95", "recall") for r in REGIONS + ["wt", "tc"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res_dir, fig_dir = cfg["paths"]["results_dir"], cfg["paths"]["figures_dir"]

    files = sorted(res_dir.glob(f"*_{args.split}_cases.csv"))
    if not files:
        raise SystemExit(f"No *_{args.split}_cases.csv in {res_dir}. Run evaluate.py first.")
    cases = pd.concat([pd.read_csv(f) for f in files])
    metrics = [m for m in KEY_METRICS if m in cases]
    order = [e for e in cfg["experiments"] if e in set(cases["exp"])]

    # Mean over test cases per run, then mean/std over seeds
    per_run = cases.groupby(["exp", "seed"])[metrics].mean()
    summary = per_run.groupby("exp").agg(["mean", "std"]).reindex(order)
    summary.to_csv(res_dir / "summary.csv")

    # Change vs baseline; Wilcoxon on per-case values averaged over seeds
    per_case = cases.groupby(["exp", "case"])[metrics].mean()
    rows = []
    if BASELINE in order:
        base = per_run.loc[BASELINE].mean()
        for exp in order:
            if exp == BASELINE:
                continue
            delta = per_run.loc[exp].mean() - base
            row = {"exp": exp, "removed": exp.removeprefix("no_")}
            for m in metrics:
                row[f"d_{m}"] = delta[m]
                if m.startswith(("dice", "hd95")):
                    a = per_case.loc[BASELINE][m]
                    b = per_case.loc[exp][m].reindex(a.index)
                    row[f"p_{m}"] = wilcoxon(a, b, nan_policy="omit").pvalue if (a - b).abs().sum() > 0 else 1.0
            rows.append(row)
    delta = pd.DataFrame(rows)
    delta.to_csv(res_dir / "delta.csv", index=False)

    if len(delta):
        rank = pd.DataFrame({r: delta.sort_values(f"d_dice_{r}")["removed"].tolist() for r in REGIONS + ["wt", "tc"]},
                            index=[f"#{i + 1}" for i in range(len(delta))])
        rank.to_csv(res_dir / "ranking.csv")
        print("Modalities ranked by Dice drop when removed (most important first):")
        print(rank.to_string(), "\n")

        # Grouped bar chart of Dice change per sub-region, error bars = std over seeds
        seed_delta = per_run.sub(per_run.loc[BASELINE].mean(), axis=1) if BASELINE in order else None
        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.8 / len(REGIONS)
        for i, r in enumerate(REGIONS):
            m = f"dice_{r}"
            g = seed_delta[m].drop(BASELINE, level="exp").groupby("exp")
            mean, std = g.mean().reindex(delta["exp"]), g.std().reindex(delta["exp"])
            ax.bar([j + i * width for j in range(len(delta))], mean, width, yerr=std, capsize=3, label=r.upper())
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks([j + width for j in range(len(delta))], [f"no {r}" for r in delta["removed"]])
        ax.set_ylabel("Change in test Dice vs 4-modality baseline")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "delta_dice.png", dpi=200)

    # Validation curves, one line per run: colour = experiment, line style = seed
    styles = ["-", "--", ":"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, exp in enumerate(cfg["experiments"]):
        for j, seed in enumerate(cfg["train"]["seeds"]):
            f = cfg["paths"]["logs_dir"] / f"{run_name(exp, seed)}.csv"
            if f.exists():
                log = pd.read_csv(f)
                ax.plot(log["epoch"], log["val_dice_mean"], color=f"C{i}", ls=styles[j % len(styles)], lw=1,
                        label=exp.replace("_", " ") if j == 0 else None)
    if ax.lines:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean validation Dice (NCR, ED, ET)")
        ax.legend(fontsize=8, ncol=5, loc="lower center", title="solid / dashed / dotted = seed 0 / 1 / 2",
                  title_fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "val_curves.png", dpi=200)

    cols = [(f"{m}_{r}", "mean") for m in ("dice", "hd95") for r in REGIONS]
    print(summary[[c for c in cols if c in summary]].round(4).to_string())
    print(f"\n-> {res_dir}, {fig_dir}")


if __name__ == "__main__":
    main()
