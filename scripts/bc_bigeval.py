"""Policies on N fresh final-level (L5) training scenes, as the validation closed loop runs them (config env latency).
Internal, lower-variance check than the 50 validation scenes. Rows are appended as they finish (resumable).

  python scripts/bc_bigeval.py OUT.csv CKPT [CKPT ...] [--n 500] [--seed 777] [--workers 32]
"""
import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["ckpt", "i", "outcome", "t_grasp", "collision_category", "wall_s"]


def one(ckpt, i, seed, level):
    import yaml

    from dyngrasp.harness import run_episode
    from dyngrasp.rl.controller import RLController
    from dyngrasp.rl.scenes import sample_training_scene

    t = time.perf_counter()
    try:
        spec = sample_training_scene(level, np.random.default_rng([seed, i]))
        lat = yaml.safe_load((Path(ckpt) / "config.yaml").read_text())["env"]["latency"]
        r = run_episode(spec, RLController(ckpt), keep_log=False, latency=lat).row()
        row = {k: r[k] for k in ("outcome", "t_grasp", "collision_category")}
    except Exception as exc:  # noqa: BLE001
        row = {"outcome": "error", "t_grasp": None, "collision_category": repr(exc)[:200]}
    return {"ckpt": ckpt, "i": i, **row, "wall_s": time.perf_counter() - t}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out)
    ckpts = [str(Path(c).resolve()) for c in a.ckpts]
    done = set(map(tuple, pd.read_csv(out)[["ckpt", "i"]].values.tolist())) if out.exists() else set()
    todo = [(c, i) for i in range(a.n) for c in ckpts if (c, i) not in done]
    print(f"{len(todo)} episodes to run ({len(done)} done)", flush=True)
    new = not out.exists()
    # Tasks submitted a few at a time: with all queued at once, the pool hung once its workers had retired
    # (max_tasks_per_child) -- 64 workers x 20 tasks, then nothing.
    with ProcessPoolExecutor(a.workers, max_tasks_per_child=20) as ex, out.open("a", newline="") as f:
        w = csv.DictWriter(f, COLS)
        if new:
            w.writeheader()
        it, pending, k = iter(todo), set(), 0
        for c, i in (next(it) for _ in range(min(2 * a.workers, len(todo)))):
            pending.add(ex.submit(one, c, i, a.seed, a.level))
        while pending:
            fut = next(as_completed(pending))
            pending.remove(fut)
            w.writerow(fut.result())
            f.flush()
            k += 1
            if k % 100 == 0:
                print(f"{k} / {len(todo)}  {time.strftime('%H:%M:%S')}", flush=True)
            nxt = next(it, None)
            if nxt is not None:
                pending.add(ex.submit(one, *nxt, a.seed, a.level))
    df = pd.read_csv(out)
    df = df[df.ckpt.isin(ckpts) & (df.i < a.n)]
    s = df.groupby("ckpt").outcome.value_counts(normalize=True).unstack(fill_value=0)
    s["n"] = df.groupby("ckpt").size()
    s["t_grasp_med"] = df[df.outcome == "success"].groupby("ckpt").t_grasp.median()
    s.index = [Path(c).parent.name + "/" + Path(c).name for c in s.index]
    pd.set_option("display.width", 250)
    print(s.round(3).to_string())
