#!/usr/bin/env python
"""Success of a (D) policy under TD3-style exploration noise: a = clip(tanh(mean(s)) + sigma eps, -1, 1).

Calibrates ft.td3_explore_std: the exploration noise should leave most of the starting policy's success intact,
otherwise the online buffer fills with failures.  Scenes: the validation scenes (scenes.validation_scenes), graded
by the harness as in training's validation (fixed latency, post-success scoring).

    python scripts/noise_sweep.py --checkpoint <policy dir> --sigmas 0 0.02 0.05 0.1 --n-scenes 24 --workers 8 \
        --out results/noise_sweep.csv
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUTCOMES = ("success", "collision", "dropped", "timeout")


def _run(checkpoint: str, spec_dict: dict, sigma: float, seed: int, latency: float, post_success: bool) -> dict:
    from dyngrasp.harness import run_episode
    from dyngrasp.rl.config import load_config
    from dyngrasp.rl.controller import RLController
    from dyngrasp.scenario import ScenarioSpec

    cfg = load_config(overrides=[f"env.post_success={post_success}"])
    ctrl = RLController(checkpoint, env_cfg=cfg["env"])
    base, rng = ctrl.actor, np.random.default_rng(seed)
    if sigma > 0:
        ctrl.actor = lambda obs: np.clip(base(obs) + sigma * rng.standard_normal(8), -1.0, 1.0)
    r = run_episode(ScenarioSpec.from_dict(spec_dict), ctrl, keep_log=False, latency=latency, settle_after_success=np.inf)
    return {"scene": r.scenario, "sigma": sigma, "outcome": r.outcome, "t_grasp": r.t_grasp,
            "min_clearance": r.min_clearance_obstacles, "collision_with": r.collision_with}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.1])
    ap.add_argument("--n-scenes", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--latency", type=float, default=0.001, help="fixed command latency (s), as in training")
    ap.add_argument("--post-success", type=int, default=1)
    ap.add_argument("--out", default="results/noise_sweep.csv")
    args = ap.parse_args()

    from dyngrasp.rl.scenes import validation_scenes

    specs = validation_scenes()[: args.n_scenes]
    jobs = [(args.checkpoint, s.to_dict(), sg, 1000 * k + i, args.latency, bool(args.post_success))
            for k, sg in enumerate(args.sigmas) for i, s in enumerate(specs)]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(_run, *zip(*jobs)))
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    table = df.groupby("sigma")["outcome"].value_counts(normalize=True).unstack(fill_value=0.0).reindex(columns=OUTCOMES, fill_value=0.0)
    table["t_grasp_median"] = df[df.outcome == "success"].groupby("sigma")["t_grasp"].median()
    pd.set_option("display.width", 160)
    print(f"{len(specs)} validation scenes per sigma, latency {args.latency}, post_success={bool(args.post_success)}")
    print(table.round(3))


if __name__ == "__main__":
    main()
