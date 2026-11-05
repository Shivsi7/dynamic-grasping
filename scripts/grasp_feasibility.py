#!/usr/bin/env python
"""Which grasp candidates are actually possible, and why not?

For every scenario and every grasp candidate (dyngrasp.grasps), sweep the
intercept times and ask the planner's goal generator for a goal: an IK solution
for the pre-grasp (with the elbow self-motion search), a collision-free
pre-grasp, and a trackable, collision-free tail.  Each scenario is evaluated
twice:

    kinematic   obstacles removed: can the arm do this grasp on the moving cart at all?
    full        with the obstacles: is it also collision-free at some time?

    python scripts/grasp_feasibility.py --scenarios results/stress_v3_heldout/scenarios.yaml --out results/grasps
    python scripts/grasp_feasibility.py --n 40 --seed 5000

Writes per (scenario, grasp, mode) rows to grasp_feasibility.csv and prints, per
grasp family, the share of scenarios in which it is possible, plus how many
scenarios only become possible thanks to non-nominal grasps.

A goal existing does not mean the object is actually picked up.  `--execute`
instead simulates every candidate on its own (planner restricted to that one
grasp) on obstacle-free scenes for every object and cart speed, and reports
whether the object is grasped, lifted and held:

    python scripts/grasp_feasibility.py --execute --speeds 0.1 0.25 --out results/grasps_exec
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HEADLESS", "1")

import numpy as np
import pandas as pd

from dyngrasp import scenario as S


def evaluate(spec_dict: dict, dT: float) -> list[dict]:
    from dyngrasp.controller import RobotModel
    from dyngrasp.grasps import generate_grasps
    from dyngrasp.planning.collision import SpaceTimeChecker
    from dyngrasp.planning.goals import GoalGenerator
    from dyngrasp.scenario import ScenarioSpec
    from dyngrasp.world import WorldModel

    robot = RobotModel()
    spec0 = ScenarioSpec.from_dict(spec_dict)
    rows = []
    for mode in ("kinematic", "full"):
        spec = copy.deepcopy(spec0)
        if mode == "kinematic":
            spec.obstacles = []
        world = WorldModel(spec)
        ck = SpaceTimeChecker(spec, world)
        for g in generate_grasps(world.object, exclusions=False):
            gg = GoalGenerator(world, robot, 0.9 * robot.v_limits, grasps=[g])
            Ts = gg.candidate_times(1.0, dT=dT, max_count=100)
            gg.start([np.asarray(spec.q0, float)], np.random.default_rng(0), n_random_seeds=0)
            feasible = []
            for T in Ts:
                if gg.generate_for(T, lambda tr, ck=ck: ck.path_free(tr.q, tr.t), ck.state_free, max_per_T=1):
                    feasible.append(float(T))
            st = gg.stats
            rows.append({
                "scenario": spec0.name, "mode": mode, "grasp": g.name, "family": g.family,
                "object": spec0.target.object, "target_speed": spec0.cart.speed,
                "n_obstacles": len(spec0.obstacles), "n_T": len(Ts), "n_feasible_T": len(feasible),
                "possible": bool(feasible), "earliest_T": min(feasible) if feasible else np.nan,
                **{k: v for k, v in st.items() if k != "goals"},
            })
    return rows


def execute(obj: str, grasp_name: str, speed: float) -> dict:
    from dyngrasp.grasps import generate_grasps
    from dyngrasp.harness import run_episode
    from dyngrasp.objects import get_object
    from dyngrasp.planning.controller import SpaceTimeRRTController

    g = next(g for g in generate_grasps(get_object(obj), exclusions=False) if g.name == grasp_name)
    spec = S.no_obstacles(speed, obj)
    c = SpaceTimeRRTController(grasps=[g])
    r = run_episode(spec, c, keep_log=False, latency=0.1)
    planned = any(l.get("grasp") for l in c.plan_log)
    return {"object": obj, "grasp": grasp_name, "family": g.family, "speed": speed, "planned": planned,
            "outcome": r.outcome, "success": r.success, "grasped": r.t_grasp is not None,
            "collision_with": r.collision_with, "collision_category": r.collision_category}


def main_execute(args):
    from dyngrasp.grasps import generate_grasps
    from dyngrasp.objects import OBJECTS

    jobs = [(o, g.name, v) for o in OBJECTS for g in generate_grasps(OBJECTS[o], exclusions=False) for v in args.speeds]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(execute, *j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            r = rows[-1]
            print(f"[{k}/{len(jobs)}] {r['object']:>15s} {r['grasp']:>16s} v={r['speed']:.2f}: {r['outcome']}", flush=True)
            pd.DataFrame(rows).to_csv(out / "grasp_execution.csv", index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nOutcome of executing each family (planned episodes only), all objects and speeds:")
    print(df[df.planned].groupby("family")["outcome"].value_counts().unstack(fill_value=0).to_string())
    print("\nNot even planned (no goal on an obstacle-free scene):",
          df[~df.planned].groupby("family").size().to_dict())
    print("\nSuccess rate per object x family (planned only):")
    print(df[df.planned].groupby(["object", "family"])["success"].mean().unstack().round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--dT", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results/grasps")
    ap.add_argument("--execute", action="store_true", help="simulate each candidate alone (obstacle-free)")
    ap.add_argument("--speeds", nargs="+", type=float, default=[0.1])
    args = ap.parse_args()
    if args.execute:
        return main_execute(args)

    if args.scenarios:
        specs = S.load_scenario_set(args.scenarios)
    else:
        from stress_planner import mixed_set  # same mix as the planner stress test

        specs = mixed_set(args.seed, args.n)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(evaluate, s.to_dict(), args.dT) for s in specs]
        for k, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            print(f"[{k}/{len(specs)}] {rows[-1]['scenario']}", flush=True)
            pd.DataFrame(rows).to_csv(out / "grasp_feasibility.csv", index=False)
    df = pd.DataFrame(rows)
    report(df)


def report(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    print("\nShare of scenarios in which each grasp family is possible (some T has a goal):")
    fam = df.groupby(["mode", "family", "scenario"])["possible"].any().groupby(["mode", "family"]).mean()
    print(fam.unstack("mode").round(2).to_string())

    print("\nNominal top-down grasp vs. all candidates (share of scenarios with any feasible grasp):")
    nominal = df[df.grasp == "top_y+"].groupby(["mode", "scenario"])["possible"].any()
    allg = df.groupby(["mode", "scenario"])["possible"].any()
    t = pd.DataFrame({"nominal": nominal.groupby("mode").mean(), "all": allg.groupby("mode").mean()})
    print(t.round(2).to_string())
    full_n, full_a = nominal.loc["full"], allg.loc["full"]
    print(f"\nScenarios only possible with non-nominal grasps: {int((full_a & ~full_n).sum())} / {len(full_a)}")
    print(f"Scenarios with no feasible grasp at all (full): {int((~full_a).sum())}")

    print("\nWhy grasps fail, full mode (counts over all grasps and T):")
    cols = ["ik_fail", "pregrasp_collides", "tail_infeasible", "tail_collides", "rescued_by_swivel"]
    print(df[df["mode"] == "full"].groupby("family")[cols].sum().to_string())

    print("\nBy cart speed (full, any grasp vs nominal):")
    s = df[df["mode"] == "full"].groupby("scenario").agg(
        speed=("target_speed", "first"), any_=("possible", "any"),
        nominal=("possible", lambda x: bool(x[df.loc[x.index, "grasp"] == "top_y+"].any())))
    print(s.groupby(pd.cut(s.speed, [0, 0.15, 0.22, 0.31]), observed=True)[["nominal", "any_"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
