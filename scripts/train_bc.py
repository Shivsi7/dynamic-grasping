#!/usr/bin/env python
"""Behaviour cloning of (C) for (D) (dyngrasp/rl/bc.py) on a dataset from scripts/make_bc_demos.py.

    python scripts/train_bc.py --data runs/demos/dataset.npz --run-dir runs/bc0
    python scripts/train_bc.py --data ... --run-dir runs/bc1 --set bc.batch_size=1024 bc.learning_rate=1e-3

Then: python scripts/eval_rl.py runs/bc0/best --scenarios scenarios/rl_experiment30.yaml --workers 8 --out ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # (the closed-loop simulation processes)
os.environ.setdefault("HEADLESS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dyngrasp.rl.bc import train
from dyngrasp.rl.config import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset .npz (make_bc_demos.py merge)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", help="YAML overriding dyngrasp/rl/config.py DEFAULTS")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="e.g. bc.batch_size=1024")
    args = ap.parse_args()
    print(train(args.run_dir, load_config(args.config, args.set), args.data), flush=True)


if __name__ == "__main__":
    main()
