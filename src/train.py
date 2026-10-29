"""Train one 2D U-Net for one experiment and seed.

The checkpoint with the highest mean validation Dice (NCR, ED, ET) is kept.
The test set is never touched here.

After every epoch the full training state is saved to checkpoints/<run>_last.pt, so an
interrupted run (crash, restart, laptop sleep) continues from its last finished epoch
when started again. The file is deleted once the run finishes.

Usage (from the project folder):
    python src/train.py --exp all --seed 0
    python src/train.py --exp no_t1c --seed 1
"""
import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader, RandomSampler

from common import (CLASS_NAMES, build_model, channel_indices, load_config, read_split, run_name, set_seed,
                    training_done)
from dataset import SliceDataset


@torch.no_grad()
def validate(model, loader, device, amp):
    """Dice per tumour class, pooled over every validation slice."""
    model.eval()
    inter = torch.zeros(4, device=device)
    total = torch.zeros(4, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)[:, 0]
        with torch.autocast(device.type, enabled=amp):
            pred = model(x).argmax(1)
        for c in CLASS_NAMES:
            p, g = pred == c, y == c
            inter[c] += (p & g).sum()
            total[c] += p.sum() + g.sum()
    dice = {name: (2 * inter[c] / total[c].clamp(min=1)).item() for c, name in CLASS_NAMES.items()}
    dice["mean"] = sum(dice.values()) / len(CLASS_NAMES)
    return dice


def save_atomic(obj, path):
    """Write to a temp file first so a crash mid-save never leaves a corrupt checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    t, paths = cfg["train"], cfg["paths"]
    name = run_name(args.exp, args.seed)
    if training_done(cfg, name):
        print(f"{name}: already finished, see {paths['logs_dir'] / f'{name}.csv'}")
        return
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = t["amp"] and device.type == "cuda"

    channels = channel_indices(cfg, args.exp)
    train_ds = SliceDataset(cfg, read_split(cfg, "train"), channels, augment=True)
    val_ds = SliceDataset(cfg, read_split(cfg, "val"), channels)
    n = min(t["samples_per_epoch"] or len(train_ds), len(train_ds))
    gen = torch.Generator().manual_seed(args.seed)
    # Persistent workers: on Windows each worker start re-imports torch/MONAI (~20 s)
    loader_args = dict(num_workers=t["num_workers"], pin_memory=True, persistent_workers=t["num_workers"] > 0)
    train_dl = DataLoader(train_ds, t["batch_size"], sampler=RandomSampler(train_ds, num_samples=n, generator=gen),
                          **loader_args)
    # Fewer validation workers: they sit idle during training but each holds ~2 GB on Windows.
    # Validation results do not depend on the number of workers.
    val_workers = t.get("val_num_workers", t["num_workers"])
    val_dl = DataLoader(val_ds, t["val_batch_size"], num_workers=val_workers, pin_memory=True,
                        persistent_workers=val_workers > 0)
    print(f"{name}: modalities={cfg['experiments'][args.exp]}, "
          f"{len(train_ds)} train slices ({n}/epoch), {len(val_ds)} val slices, device={device}")

    model = build_model(cfg, len(channels)).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, batch=True, lambda_dice=1.0, lambda_ce=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    scaler = torch.amp.GradScaler(enabled=amp)

    ckpt = paths["checkpoints_dir"] / f"{name}.pt"
    last = paths["checkpoints_dir"] / f"{name}_last.pt"
    log_path = paths["logs_dir"] / f"{name}.csv"
    log, best, stale, start_epoch = [], -1.0, 0, 1
    if last.exists():
        s = torch.load(last, map_location="cpu", weights_only=False)  # RNG states must stay on CPU
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["optimizer"])
        scaler.load_state_dict(s["scaler"])
        gen.set_state(s["sampler_rng"])
        torch.set_rng_state(s["torch_rng"])
        np.random.set_state(s["numpy_rng"])
        random.setstate(s["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(s["cuda_rng"])
        log, best, stale, start_epoch = s["log"], s["best"], s["stale"], s["epoch"] + 1
        pd.DataFrame(log).to_csv(log_path, index=False)
        print(f"Resuming from epoch {start_epoch} (best val dice so far {best:.4f})")

    for epoch in range(start_epoch, t["max_epochs"] + 1):
        if stale >= t["patience"]:
            break
        start = time.time()
        model.train()
        running = 0.0
        for step, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(x)
            if step % 2000 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)}  loss {loss.item():.4f}  "
                      f"{time.time() - start:.0f}s", flush=True)

        val = validate(model, val_dl, device, amp)
        row = {"epoch": epoch, "train_loss": running / n, **{f"val_dice_{k}": v for k, v in val.items()},
               "seconds": time.time() - start}
        log.append(row)

        improved = val["mean"] > best
        if improved:
            best, stale = val["mean"], 0
            save_atomic({"model": model.state_dict(), "exp": args.exp, "seed": args.seed,
                         "channels": channels, "epoch": epoch, "val_dice": val}, ckpt)
        else:
            stale += 1
        save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                     "sampler_rng": gen.get_state(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                     "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                     "log": log, "best": best, "stale": stale, "epoch": epoch}, last)
        pd.DataFrame(log).to_csv(log_path, index=False)
        print(f"epoch {epoch:3d}  loss {row['train_loss']:.4f}  val dice {val['mean']:.4f} "
              f"(NCR {val['ncr']:.3f} ED {val['ed']:.3f} ET {val['et']:.3f})  "
              f"{row['seconds']:.0f}s{'  *' if improved else ''}", flush=True)

    if stale >= t["patience"]:
        print(f"Early stop: no improvement for {t['patience']} epochs")
    last.unlink(missing_ok=True)
    print(f"Best val dice {best:.4f} -> {ckpt}")


if __name__ == "__main__":
    main()
