#!/usr/bin/env python
"""Fine-tune the behaviour-cloned (D) policy with SAC in the Drake simulation (dyngrasp/rl/finetune.py), or resume.

    python scripts/finetune_rl.py --run-dir runs/ft0 --set ft.bc_checkpoint=bc/runs/bc_f2n/best ft.demos=demos.npz
    python scripts/finetune_rl.py --run-dir runs/ft1 --config my.yaml   # YAML with any subset of rl/config.py's keys

Running the same command again resumes from the run's checkpoint, with the configuration saved in the run
directory.  SIGUSR1 or SIGTERM: checkpoint, exit 0, no DONE.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("HEADLESS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dyngrasp.rl.config import load_config
from dyngrasp.rl.finetune import finetune


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", help="YAML overriding dyngrasp/rl/config.py DEFAULTS")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="e.g. ft.utd=2")
    args = ap.parse_args()
    print(finetune(args.run_dir, load_config(args.config, args.set)), flush=True)


if __name__ == "__main__":
    main()
