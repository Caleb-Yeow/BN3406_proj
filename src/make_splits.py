"""Split BraTS cases 70/15/15 at the patient level with a fixed seed.

BraTS-GLI-00052-000 and BraTS-GLI-00052-001 are two timepoints of patient 00052,
so cases are grouped by patient before splitting to keep a patient in one set only.

Usage (from the project folder):
    python src/make_splits.py
"""
import argparse

import numpy as np
import pandas as pd

from common import load_config


def patient_id(case):
    return case.split("-")[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw, out, s = cfg["paths"]["raw_dir"], cfg["paths"]["splits_dir"], cfg["split"]

    cases = sorted({p.parent.name for p in raw.rglob("*-seg.nii.gz")})
    if not cases:
        raise SystemExit(f"No cases under {raw}. Unzip the BraTS training data there first.")
    patients = sorted({patient_id(c) for c in cases})

    rng = np.random.default_rng(s["seed"])
    rng.shuffle(patients)
    n_train = round(s["train"] * len(patients))
    n_val = round(s["val"] * len(patients))
    groups = {
        "train": set(patients[:n_train]),
        "val": set(patients[n_train:n_train + n_val]),
        "test": set(patients[n_train + n_val:]),
    }

    for name, pats in groups.items():
        df = pd.DataFrame({"case": [c for c in cases if patient_id(c) in pats]})
        df["patient"] = df["case"].map(patient_id)
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(pats):4d} patients, {len(df):4d} cases")


if __name__ == "__main__":
    main()
