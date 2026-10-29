"""Show training progress for all runs as progress bars. Read-only: safe while training runs.

Usage (from the project folder):
    python src/progress.py            # show once
    python src/progress.py --watch    # refresh every 30 s, Ctrl+C to stop
"""
import argparse
import os
import re
import sys
import time

import pandas as pd

from common import load_config, run_name, training_done

STEP_RE = re.compile(r"epoch (\d+) step (\d+)/(\d+)\s+loss \S+\s+(\d+)s")
TRAIN_RE = re.compile(r"train\.py --exp (\S+) --seed (\d+)")


def bar(frac, width=30):
    frac = min(max(frac, 0.0), 1.0)
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled) + f" {frac:6.1%}"


def hours(seconds):
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def current_step(cfg):
    """(epoch, step, total_steps, seconds) from the last progress line of run_all.out, if any."""
    out = cfg["paths"]["logs_dir"] / "run_all.out"
    if not out.exists():
        return None
    lines = out.read_text(errors="ignore").splitlines()[-20:]
    for line in reversed(lines):
        if line.startswith("epoch "):  # an epoch just finished, the next has no steps yet
            return None
        m = STEP_RE.search(line)
        if m:
            return tuple(int(g) for g in m.groups())
    return None


def started_run(cfg):
    """Name of the run whose train.py run_all launched most recently, if any."""
    out = cfg["paths"]["logs_dir"] / "run_all.out"
    if not out.exists():
        return None
    matches = TRAIN_RE.findall(out.read_text(errors="ignore"))
    return run_name(matches[-1][0], int(matches[-1][1])) if matches else None


def show(cfg):
    paths, t = cfg["paths"], cfg["train"]
    runs = [(exp, seed) for seed in t["seeds"] for exp in cfg["experiments"]]
    rows, durations, current = [], [], None
    started = started_run(cfg)  # a run has no log or _last.pt until its first epoch ends
    for exp, seed in runs:
        name = run_name(exp, seed)
        log_path = paths["logs_dir"] / f"{name}.csv"
        log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
        results = paths["results_dir"] / f"{name}_test_cases.csv"
        trained = training_done(cfg, name)
        test = ""
        if results.exists():
            r = pd.read_csv(results)
            test = " ".join(f"{r[f'dice_{c}'].mean():.3f}" for c in ("ncr", "ed", "et"))
            status = "done"
            durations.append(log["seconds"].sum())
        elif trained:
            status = "testing"
            current = current or name
        elif len(log) or (paths["checkpoints_dir"] / f"{name}_last.pt").exists() or name == started:
            status = "training"
            current = current or name
        else:
            status = "queued"
        best = f"{log['val_dice_mean'].max():.4f}" if len(log) else ""
        best_ep = int(log["val_dice_mean"].idxmax()) + 1 if len(log) else 0
        rows.append((name, status, len(log), best, best_ep, test, log))

    done = sum(r[1] == "done" for r in rows)
    print(f"BraTS leave-one-modality-out  ·  {time.strftime('%a %d %b %H:%M')}\n")
    print(f"All runs       {bar(done / len(rows))}   {done}/{len(rows)} finished")

    step = current_step(cfg)
    cur = next((r for r in rows if r[0] == current), None)
    if cur and cur[1] == "training":
        name, _, epochs, best, best_ep, _, log = cur
        stale = epochs - best_ep if epochs else 0
        print(f"\nNow training   {name}")
        if step and step[0] == epochs + 1:
            ep, s, total, secs = step
            print(f"  Epoch {ep:<3d}     {bar(s / total)}   step {s}/{total}, {hours(secs)} in")
        else:
            print(f"  Epoch {epochs + 1:<3d}     starting / validating")
        print(f"  Epochs done  {epochs}   (max {t['max_epochs']})")
        if epochs:
            print(f"  Best val Dice {best} at epoch {best_ep}; {stale}/{t['patience']} epochs without "
                  f"improvement (stops at {t['patience']})")
    elif cur and cur[1] == "testing":
        print(f"\nNow testing    {cur[0]} on the test set")

    print(f"\n{'Run':16s} {'Status':9s} {'Epochs':>6s} {'Best val':>9s}   Test Dice NCR / ED / ET")
    for name, status, epochs, best, _, test, _ in rows:
        mark = {"done": "✔", "training": "▶", "testing": "▶", "queued": "·"}[status]
        print(f"{mark} {name:14s} {status:9s} {epochs or '':>6} {best:>9s}   {test}")

    if durations:
        avg = sum(durations) / len(durations)
        left = len(rows) - done - (1 if cur else 0)
        elapsed = 0
        if cur:
            elapsed = cur[6]["seconds"].sum() if len(cur[6]) else 0
            elapsed += step[3] if step and cur[1] == "training" else 0
        remaining = left * avg + max(avg - elapsed, 0)
        print(f"\nAverage run so far: {hours(avg)} training.  Rough time left: ~{hours(remaining)} "
              f"(assumes later runs stop as early as finished ones)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="refresh every --every seconds")
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--config")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(args.config)
    while True:
        if args.watch:
            os.system("cls" if os.name == "nt" else "clear")
        show(cfg)
        if not args.watch:
            break
        print(f"\nRefreshing every {args.every} s. Ctrl+C to stop.")
        time.sleep(args.every)


if __name__ == "__main__":
    main()
