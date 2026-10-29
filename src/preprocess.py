"""Turn each BraTS case into a stack of normalised, cropped 2D axial slices.

Per case:
  1. Z-score each modality over its non-zero (brain) voxels; background stays 0.
  2. Crop 192 x 192 in-plane around the brain bounding box (no resizing).
  3. Keep axial slices that contain brain.
  4. Save image (D, 4, 192, 192) float16 and seg (D, 192, 192) uint8 as .npy.

Also writes data/processed/slices.csv (case, slice, tumour voxels) and reports
any tumour voxels lost by the crop.

Usage (from the project folder):
    python src/preprocess.py --workers 4
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import load_config, read_split


def find_case_dirs(raw):
    return {p.parent.name: p.parent for p in raw.rglob("*-seg.nii.gz")}


def zscore(vol):
    mask = vol > 0
    if mask.any():
        v = vol[mask]
        vol = np.where(mask, (vol - v.mean()) / (v.std() + 1e-8), 0)
    return vol


def crop_start(lo, hi, size, full):
    """Start index of a `size` window centred on [lo, hi], kept inside [0, full]."""
    centre = (lo + hi) // 2
    return int(np.clip(centre - size // 2, 0, full - size))


def process_case(case, case_dir, out_dir, modalities, size, keep):
    vols = [np.asarray(nib.load(case_dir / f"{case}-{m}.nii.gz").dataobj, dtype=np.float32) for m in modalities]
    seg = np.asarray(nib.load(case_dir / f"{case}-seg.nii.gz").dataobj).astype(np.uint8)
    img = np.stack([zscore(v) for v in vols])            # (4, X, Y, Z)
    brain = (img != 0).any(axis=0)                       # (X, Y, Z)

    xs, ys = np.where(brain.any(axis=2))
    x0 = crop_start(xs.min(), xs.max(), size, brain.shape[0])
    y0 = crop_start(ys.min(), ys.max(), size, brain.shape[1])
    img = img[:, x0:x0 + size, y0:y0 + size]
    seg_c = seg[x0:x0 + size, y0:y0 + size]
    lost = int((seg > 0).sum() - (seg_c > 0).sum())

    z = np.where(brain.any(axis=(0, 1)))[0] if keep == "brain" else np.arange(brain.shape[2])
    img = np.ascontiguousarray(img[..., z].transpose(3, 0, 1, 2)).astype(np.float16)  # (D, 4, H, W)
    seg_c = np.ascontiguousarray(seg_c[..., z].transpose(2, 0, 1))                       # (D, H, W)

    np.save(out_dir / f"{case}_img.npy", img)
    np.save(out_dir / f"{case}_seg.npy", seg_c)
    rows = [{"case": case, "slice": i, "z": int(zi), "tumour_voxels": int((seg_c[i] > 0).sum())}
            for i, zi in enumerate(z)]
    return rows, lost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = cfg["preprocess"]

    dirs = find_case_dirs(cfg["paths"]["raw_dir"])
    cases = [c for split in ("train", "val", "test") for c in read_split(cfg, split)]
    fn = partial(process_case, out_dir=cfg["paths"]["processed_dir"], modalities=cfg["modalities"],
                 size=p["crop_size"], keep=p["keep_slices"])

    rows, lost = [], {}
    with ProcessPoolExecutor(args.workers) as ex:
        futures = [ex.submit(fn, c, dirs[c]) for c in cases]
        for c, f in tqdm(zip(cases, futures), total=len(cases)):
            r, n = f.result()
            rows += r
            if n:
                lost[c] = n

    df = pd.DataFrame(rows)
    df.to_csv(cfg["paths"]["processed_dir"] / "slices.csv", index=False)
    print(f"{len(cases)} cases, {len(df)} slices, {(df.tumour_voxels > 0).mean():.1%} contain tumour")
    print(f"Cases with tumour voxels outside the crop: {len(lost)}"
          + (f" (worst: {max(lost, key=lost.get)}, {max(lost.values())} voxels)" if lost else ""))


if __name__ == "__main__":
    main()
