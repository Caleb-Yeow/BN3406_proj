"""Evaluate a trained model on the test set, case by case in 3D.

Slice predictions are stacked back into a volume per case, then per class (NCR, ED, ET)
we compute Dice, IoU, HD95 (mm, data is 1 mm isotropic) and recall, plus background
recall, voxel accuracy, and the BraTS regions WT (1+2+3) and TC (1+3).

Usage (from the project folder):
    python src/evaluate.py --exp all --seed 0
"""
import argparse

import numpy as np
import pandas as pd
import torch
from monai.metrics import compute_hausdorff_distance

from common import CLASS_NAMES, build_model, load_case, load_config, read_split, run_name

REGIONS = {"wt": (1, 2, 3), "tc": (1, 3)}


def load_model(cfg, exp, seed, device):
    ckpt = torch.load(cfg["paths"]["checkpoints_dir"] / f"{run_name(exp, seed)}.pt", map_location=device)
    model = build_model(cfg, len(ckpt["channels"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt["channels"]


@torch.no_grad()
def predict_case(model, img, channels, device, batch_size=32):
    """img: (D, 4, H, W) -> predicted labels (D, H, W) uint8."""
    out = []
    for i in range(0, len(img), batch_size):
        x = torch.from_numpy(img[i:i + batch_size][:, channels].astype(np.float32)).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(x).argmax(1).cpu().numpy().astype(np.uint8))
    return np.concatenate(out)


def binary_metrics(p, g, hd95_empty):
    tp = np.logical_and(p, g).sum()
    ps, gs = p.sum(), g.sum()
    if gs == 0 and ps == 0:
        return {"dice": 1.0, "iou": 1.0, "hd95": 0.0, "recall": np.nan}
    if gs == 0 or ps == 0:
        return {"dice": 0.0, "iou": 0.0, "hd95": hd95_empty, "recall": 0.0 if gs else np.nan}
    hd = compute_hausdorff_distance(torch.from_numpy(p)[None, None], torch.from_numpy(g)[None, None],
                                    percentile=95).item()
    return {"dice": 2 * tp / (ps + gs), "iou": tp / (ps + gs - tp), "hd95": hd, "recall": tp / gs}


def case_metrics(pred, seg, hd95_empty):
    row = {"accuracy": (pred == seg).mean(), "recall_bg": ((pred == 0) & (seg == 0)).sum() / max((seg == 0).sum(), 1)}
    for c, name in CLASS_NAMES.items():
        for k, v in binary_metrics(pred == c, seg == c, hd95_empty).items():
            row[f"{k}_{name}"] = v
    for name, labels in REGIONS.items():
        for k, v in binary_metrics(np.isin(pred, labels), np.isin(seg, labels), hd95_empty).items():
            row[f"{k}_{name}"] = v
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, channels = load_model(cfg, args.exp, args.seed, device)

    rows = []
    for case in read_split(cfg, args.split):
        img, seg = load_case(cfg, case)
        pred = predict_case(model, img, channels, device)
        rows.append({"case": case, "exp": args.exp, "seed": args.seed,
                     **case_metrics(pred, np.asarray(seg), cfg["eval"]["hd95_empty"])})

    df = pd.DataFrame(rows)
    out = cfg["paths"]["results_dir"] / f"{run_name(args.exp, args.seed)}_{args.split}_cases.csv"
    df.to_csv(out, index=False)
    cols = [f"{m}_{c}" for m in ("dice", "hd95") for c in CLASS_NAMES.values()]
    print(df[cols].mean().round(4).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
