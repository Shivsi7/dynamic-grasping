#!/usr/bin/env python
"""Remote environments for fine-tuning runs (dyngrasp/rl/vecenv.py): N worker processes per run directory, each
connecting to the run's learner (address in RUN_DIR/workers.addr, key in workers.addr.key, written by the learner
with ft.remote_envs > 0) and serving a GraspEnv; after a dropped connection (learner preempted or restarted) they
reconnect to the address then in the file.  Every worker process is replaced by a fresh one after --recycle episodes
(GraspEnv's memory grows with every episode).  Exits when every run has its DONE file.

    python scripts/env_workers.py [--recycle 100] RUN_DIR:N [RUN_DIR:N ...]
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
    args = sys.argv[1:]
    recycle = 100
    if args[:1] == ["--recycle"]:
        recycle, args = int(args[1]), args[2:]
    runs = [(Path(a.rsplit(":", 1)[0]), int(a.rsplit(":", 1)[1])) for a in args]
    if not runs:
        sys.exit(__doc__)
    ctx = mp.get_context("spawn")
    slots = [d for d, n in runs for _ in range(n)]

    def start(d):
        p = ctx.Process(target=remote_worker, args=(str(d / "workers.addr"), str(d / "DONE"), recycle), daemon=True)
        p.start()
        return p

    procs = [start(d) for d in slots]
    print(f"{len(procs)} workers for {[str(d) for d, _ in runs]}, recycled every {recycle} episodes", flush=True)
    fresh = errors = 0
    while not all((d / "DONE").exists() for d, _ in runs) or any(p.is_alive() for p in procs):
        time.sleep(10)
        for k, p in enumerate(procs):
            if p.is_alive() or (slots[k] / "DONE").exists():
                continue
            errors += bool(p.exitcode)
            fresh += 1
            procs[k] = start(slots[k])
            if fresh % 100 == 0:
                print(f"{fresh} fresh worker processes so far ({errors} after an error)", flush=True)
    print("every run is DONE", flush=True)


if __name__ == "__main__":
    main()
