"""Check that the BraTS 2023 GLI training data in data/raw is complete.

Usage (from the project folder):
    python src/check_data.py
"""
import argparse

import nibabel as nib
import numpy as np

from common import load_config

SUFFIXES = ["t1n", "t1c", "t2w", "t2f", "seg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    RAW = load_config(ap.parse_args().config)["paths"]["raw_dir"]
    # A case folder is any folder holding a *-seg.nii.gz file, so nested unzip layouts still work
    case_dirs = sorted({p.parent for p in RAW.rglob("*-seg.nii.gz")})
    print(f"Found {len(case_dirs)} cases under {RAW}")
    if not case_dirs:
        print("No cases found. Unzip the BraTS training data into data/raw/.")
        return

    incomplete = []
    for d in case_dirs:
        missing = [s for s in SUFFIXES if not (d / f"{d.name}-{s}.nii.gz").exists()]
        if missing:
            incomplete.append((d.name, missing))
    print(f"Incomplete cases: {len(incomplete)}")
    for name, missing in incomplete[:10]:
        print(f"  {name}: missing {missing}")

    # Inspect one case to confirm shape and label values
    d = case_dirs[0]
    img = nib.load(d / f"{d.name}-t1c.nii.gz")
    seg = np.asarray(nib.load(d / f"{d.name}-seg.nii.gz").dataobj)
    print(f"\nExample case: {d.name}")
    print(f"  shape={img.shape}, spacing={img.header.get_zooms()}")
    print(f"  seg labels={np.unique(seg).tolist()} (expected 0 BG, 1 NCR, 2 ED, 3 ET)")
    print(f"\nCase folders live in: {case_dirs[0].parent}")


if __name__ == "__main__":
    main()
