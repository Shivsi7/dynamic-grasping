#!/usr/bin/env python
"""(C) demonstrations as RL transitions: the actions of a BC dataset's episodes (scripts/make_bc_demos.py merge)
replayed open-loop through GraspEnv (final-level grasp tolerances), which computes the training reward.

    python scripts/make_rl_demos.py BC_DATASET.npz --out demos_rl.npz --episodes 6000 --workers 60

After its last action an episode holds still (same opening) until it ends.  Writes obs, act, rew, next_obs, done
(1-step, every episode ends with done = 1), ep_start, ep_len, outcome and level per episode, and a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("HEADLESS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

_ENV = None


def replay(spec_dict: dict, level: int, actions: np.ndarray) -> dict:
    global _ENV
    from dyngrasp.rl.config import load_config
    from dyngrasp.rl.env import GraspEnv
    from dyngrasp.rl.scenes import N_LEVELS
    from dyngrasp.scenario import ScenarioSpec

    if _ENV is None:
        _ENV = GraspEnv(load_config())
        _ENV.set_level(N_LEVELS - 1)
    env = _ENV
    obs, _ = env.reset(options={"spec": ScenarioSpec.from_dict(spec_dict), "level": level})
    hold = np.concatenate([np.zeros(7), actions[-1, 7:]])
    rows = {k: [] for k in ("obs", "act", "rew", "next_obs", "done")}
    k, done, info = 0, False, {}
    while not done:
        a = actions[k] if k < len(actions) else hold
        nxt, r, done, _, info = env.step(a)
        for name, x in zip(rows, (obs, a, r, nxt, done)):
            rows[name].append(x)
        obs, k = nxt, k + 1
    out = {name: np.array(v, np.float32) for name, v in rows.items()}
    out.update(outcome=info["outcome"], level=level, n_actions=len(actions))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=6000, help="a random subset of this many episodes")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.dataset)
    meta = json.loads(str(z["meta"]))["episodes"]
    act, start, length = z["act"], z["ep_start"], z["ep_len"]
    pick = np.sort(np.random.default_rng(args.seed).choice(len(meta), min(args.episodes, len(meta)), replace=False))
    t0 = time.time()
    eps = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(replay, meta[e]["spec"], meta[e]["level"], act[start[e]: start[e] + length[e]]) for e in pick]
        for j, fut in enumerate(futs, 1):
            eps.append(fut.result())
            if j % 200 == 0:
                print(f"{j}/{len(pick)} episodes, {(time.time() - t0) / 60:.1f} min", flush=True)
    ep_len = np.array([len(e["rew"]) for e in eps])
    out = {k: np.concatenate([e[k] for e in eps]) for k in ("obs", "act", "rew", "next_obs", "done")}
    outcomes = np.array([e["outcome"] for e in eps])
    np.savez(args.out, **out, ep_start=np.concatenate([[0], np.cumsum(ep_len)[:-1]]), ep_len=ep_len,
             outcome=outcomes, level=np.array([e["level"] for e in eps]), source_episode=pick)
    summary = {"dataset": str(Path(args.dataset).resolve()), "episodes": len(eps), "transitions": int(ep_len.sum()),
               "outcomes": {o: int(np.sum(outcomes == o)) for o in np.unique(outcomes)},
               "return_mean": float(np.mean([e["rew"].sum() for e in eps])),
               "minutes": (time.time() - t0) / 60}
    Path(args.out + ".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
