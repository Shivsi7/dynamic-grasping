#!/usr/bin/env python
"""Train (D), SAC (sbx) in the Drake simulation, or resume a run.

    python scripts/train_rl.py --run-dir runs/sac0
    python scripts/train_rl.py --run-dir runs/sac64 --set train.n_envs=64 sac.batch_size=1024 train.seed=1
    python scripts/train_rl.py --run-dir runs/x --config my.yaml   # YAML with any subset of rl/config.py's keys

Running the same command again resumes from the run's latest checkpoint, with the configuration saved
in the run directory (overrides given then are ignored).  A finished run (run_dir/DONE) exits at once.
SIGUSR1 or SIGTERM (Slurm: time limit near, preemption): checkpoint, exit 0, no DONE.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per simulation process
os.environ.setdefault("HEADLESS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dyngrasp.rl.config import load_config
from dyngrasp.rl.train import train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", help="YAML overriding dyngrasp/rl/config.py DEFAULTS")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="e.g. train.n_envs=64")
    args = ap.parse_args()
    print(train(args.run_dir, load_config(args.config, args.set)), flush=True)


if __name__ == "__main__":
    main()
