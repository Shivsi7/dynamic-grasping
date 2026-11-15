#!/usr/bin/env python
"""Remote environments for fine-tuning runs (dyngrasp/rl/vecenv.py): N worker processes per run directory, each
connecting to the run's learner (address in RUN_DIR/workers.addr, key in workers.addr.key, written by the learner
with ft.remote_envs > 0) and serving a GraspEnv; after a dropped connection (learner preempted or restarted) they
reconnect to the address then in the file.  Exits when every run has its DONE file.

    python scripts/env_workers.py RUN_DIR:N [RUN_DIR:N ...]
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("HEADLESS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dyngrasp.rl.vecenv import remote_worker


def main():
    runs = [(Path(a.rsplit(":", 1)[0]), int(a.rsplit(":", 1)[1])) for a in sys.argv[1:]]
    if not runs:
        sys.exit(__doc__)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=remote_worker, args=(str(d / "workers.addr"), str(d / "DONE")), daemon=True)
             for d, n in runs for _ in range(n)]
    for p in procs:
        p.start()
    print(f"{len(procs)} workers for {[str(d) for d, _ in runs]}", flush=True)
    reported = set()
    while any(p.is_alive() for p in procs):
        time.sleep(30)
        dead = [k for k, p in enumerate(procs) if not p.is_alive() and p.exitcode and k not in reported]
        if dead:
            print(f"{len(dead)} workers exited with errors: {[procs[k].exitcode for k in dead]}", flush=True)
            reported.update(dead)
    print("every run is DONE", flush=True)


if __name__ == "__main__":
    main()
