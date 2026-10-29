"""Shared helpers: config loading, paths, seeding, model construction."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# torch and MONAI are imported inside the functions that need them: on Windows each process that
# imports torch reserves ~2 GB of memory, which light scripts like progress.py don't need.

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / "configs" / "base.yaml"
CLASS_NAMES = {1: "ncr", 2: "ed", 3: "et"}


def merge(base, override):
    out = dict(base)
    for k, v in override.items():
        out[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path=None):
    path = Path(path or DEFAULT_CONFIG)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # `base: base.yaml` inherits every setting from that file and overrides only what is listed
    if "base" in cfg:
        with open(path.parent / cfg.pop("base")) as f:
            cfg = merge(yaml.safe_load(f), cfg)
    # Resolve relative paths against the project folder
    cfg["paths"] = {k: (PROJECT / v).resolve() for k, v in cfg["paths"].items()}
    for p in cfg["paths"].values():
        p.mkdir(parents=True, exist_ok=True)
    return cfg


def channel_indices(cfg, exp):
    """Indices into the processed (t1n, t1c, t2w, t2f) channel axis for an experiment."""
    return [cfg["modalities"].index(m) for m in cfg["experiments"][exp]]


def run_name(exp, seed):
    return f"{exp}_seed{seed}"


def training_done(cfg, name):
    """True once a run's log shows it reached max_epochs or stopped early."""
    log = cfg["paths"]["logs_dir"] / f"{name}.csv"
    if not log.exists():
        return False
    df = pd.read_csv(log)
    if df.empty:
        return False
    epochs_since_best = len(df) - 1 - int(df["val_dice_mean"].idxmax())
    return len(df) >= cfg["train"]["max_epochs"] or epochs_since_best >= cfg["train"]["patience"]


def read_split(cfg, name):
    return pd.read_csv(cfg["paths"]["splits_dir"] / f"{name}.csv")["case"].tolist()


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg, in_channels):
    from monai.networks.nets import UNet
    m = cfg["model"]
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=m["out_channels"],
        channels=m["channels"],
        strides=m["strides"],
        num_res_units=m["num_res_units"],
    )


def load_case(cfg, case):
    """Processed arrays: image (D, 4, H, W) float16, seg (D, H, W) uint8. Memory-mapped."""
    d = cfg["paths"]["processed_dir"]
    return np.load(d / f"{case}_img.npy", mmap_mode="r"), np.load(d / f"{case}_seg.npy", mmap_mode="r")
