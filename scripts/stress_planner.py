#!/usr/bin/env python
"""Run controllers over many generated scenarios in parallel and tabulate failures.

    python scripts/stress_planner.py --n 200 --workers 4 --out results/stress
    python scripts/stress_planner.py --scenarios scenarios/dev_v0.yaml --controllers st_rrt naive

Scenarios mix the generator's settings (cart / float mounts, all objects, 0-4
obstacles) unless --scenarios is given.  Measured latency is wall time, so keep
--workers at or below the number of physical cores (default 4).
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HEADLESS", "1")

import numpy as np
import pandas as pd

from dyngrasp import scenario as S
from dyngrasp.objects import OBJECTS


def mixed_set(seed: int, n: int) -> list:
    """n scenarios spread over mounts, objects and obstacle counts."""
    specs = []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        mount = "float" if i % 4 == 3 else "cart"
        n_obst = int(i % 5)  # 0..4
        specs.append(S.sample_scenario(rng, n_obstacles=n_obst, objects=tuple(OBJECTS), name=f"mix_{i:04d}",
                                       seed=seed + i, mount=mount))
    return specs


def run_one(spec_dict: dict, cname: str, latency) -> dict:
    from dyngrasp.baselines import HoldController, NaiveInterceptController
    from dyngrasp.harness import run_episode
    from dyngrasp.planning.controller import SpaceTimeRRTController, STRRTConfig
    from dyngrasp.planning.goals import TailParams
    from dyngrasp.scenario import ScenarioSpec

    factories = {
        "st_rrt": SpaceTimeRRTController,
        "st_rrt_nominal": lambda: SpaceTimeRRTController(STRRTConfig(grasp_set="nominal")),
        "st_rrt_noaccel": lambda: SpaceTimeRRTController(STRRTConfig(a_max=None)),
        "st_rrt_slowtail": lambda: SpaceTimeRRTController(
            STRRTConfig(tail=TailParams(t_descend=0.7, t_lift=0.8))),
        "naive": NaiveInterceptController,
        "hold": HoldController,
    }
    spec = ScenarioSpec.from_dict(spec_dict)
    ctrl = factories[cname]()
    try:
        r = run_episode(spec, ctrl, keep_log=False, latency=latency)
    except Exception as e:  # a crash is a result too: record it
        return {"scenario": spec.name, "controller": cname, "outcome": "crash", "error": repr(e)[:300],
                **spec.summary()}
    row = r.row()
    row["controller"] = cname  # configurations of one controller class share its .name
    if cname.startswith("st_rrt"):
        logs = ctrl.plan_log
        first = logs[0] if logs else {}
        row.update(
            n_goals=max((l.get("n_goals", 0) for l in logs), default=0),
            goal_gen_s=sum(l.get("goal_gen_s", 0.0) for l in logs),
            goal_stats=str(getattr(ctrl.goalgen, "stats", {})),
            first_plan_ok=bool(first.get("rrt_ok", False)),
            n_replans=len(logs),
            n_new_plans=sum(l["outcome"] == "new" for l in logs),
            n_escapes=sum(bool(l.get("escape_ok")) for l in logs),
            replan_wall_max=max((l["wall"] for l in logs[1:]), default=np.nan),
            T_goals=",".join(f"{l['T_goal']:.2f}" for l in logs if "T_goal" in l),
            grasp_used=next((l["grasp"] for l in reversed(logs) if "grasp" in l), ""),
            grasp_goals=";".join(f"{k}:{v['goals']}" for k, v in getattr(ctrl.goalgen, "grasp_stats", {}).items()
                                 if v["goals"]),
        )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--scenarios")
    ap.add_argument("--controllers", nargs="+", default=["st_rrt"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--latency", default="measured")
    ap.add_argument("--out", default="results/stress")
    args = ap.parse_args()
    latency = args.latency if args.latency in ("measured", "none") else float(args.latency)

    specs = S.load_scenario_set(args.scenarios) if args.scenarios else mixed_set(args.seed, args.n)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    S.save_scenario_set(specs, out / "scenarios.yaml")
    jobs = [(s.to_dict(), c) for s in specs for c in args.controllers]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, d, c, latency) for d, c in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            row = f.result()
            rows.append(row)
            print(f"[{k}/{len(jobs)}] {row['scenario']:>12s} {row['controller']:>7s}: {row['outcome']}", flush=True)
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    df = pd.DataFrame(rows)
    print()
    print(df.groupby("controller")["outcome"].value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
