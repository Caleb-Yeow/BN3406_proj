"""Train and evaluate every experiment x seed, skipping runs that are already done.

Usage (from the project folder):
    python src/run_all.py                      # all 5 experiments x 3 seeds = 15 runs
    python src/run_all.py --exps all no_t1c --seeds 0

A run counts as trained once its log shows it finished (40 epochs or early stop);
an unfinished run is started again and resumes from its last saved epoch.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from common import load_config, run_name, training_done

SRC = Path(__file__).resolve().parent


def run(script, *args):
    cmd = [sys.executable, str(SRC / script), *args]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    extra = ["--config", args.config] if args.config else []

    for seed in args.seeds or cfg["train"]["seeds"]:
        for exp in args.exps or cfg["experiments"]:
            name = run_name(exp, seed)
            if not training_done(cfg, name):  # also resumes a run that was interrupted
                run("train.py", "--exp", exp, "--seed", str(seed), *extra)
            if not (cfg["paths"]["results_dir"] / f"{name}_test_cases.csv").exists():
                run("evaluate.py", "--exp", exp, "--seed", str(seed), *extra)
    run("analyze.py", *extra)


if __name__ == "__main__":
    main()
