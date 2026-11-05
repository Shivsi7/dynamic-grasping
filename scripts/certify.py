#!/usr/bin/env python
"""Was abstaining correct?  Label the scenes in which (B) did not grasp.

For every scene the online planner failed on, collect increasingly strong evidence:

  1. exhaustive goal search  no time caps, dense intercept times, many IK seeds and elbow
                             postures, the planner's grasp set and tail limits.  No goal
                             -> "no feasible grasp found" (relative to the grasp set and the
                             IK search: strong evidence, not a mathematical proof).
  2. dynamics lower bound    for each goal, the time-optimal motion from rest at q0 under the
                             planner's joint velocity and acceleration limits, ignoring all
                             obstacles.  If even that arrives after T for every goal, the
                             scene is "dynamically unreachable" (a sound bound, given 1).
  3. oracle                  the same planner with `--wall-scale` x the compute and planning
                             time free (latency "none"), executed in simulation.  A grasp
                             -> "feasible: the online planner missed it" (proof by example).
  4. otherwise               "likely infeasible": goals exist, but even the oracle fails.

The oracle is also run with the acceleration limits off, to show which scenes the limits
themselves make impossible, and (`--sweep`) at increasing acceleration limits, which gives
each scene a *required acceleration*: the smallest limit at which it is possible.  Limits are
per joint, as a fraction of IIWA_ACCEL_REF (the planner uses 0.6, its grasp tail 0.8; 1.0 is
the reference itself).  For the grasp tail the same number comes straight from the exhaustive
search: the smallest peak of joint acceleration / reference among all feasible goals
(searched with no tail limit).

    python scripts/certify.py --results results/stress_v11_accel/results.csv --controller st_rrt_fasttail \\
        --scenarios results/stress_v3_heldout/scenarios.yaml --out results/certify
    python scripts/certify.py --scenarios <set.yaml> --names mix_0034 mix_0054
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HEADLESS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dyngrasp import scenario as S  # noqa: E402
from dyngrasp.planning.controller import IIWA_ACCEL_REF, STRRTConfig, _scaled, accel_vector  # noqa: E402

PLANNER_SCALE = STRRTConfig().a_max[0] / IIWA_ACCEL_REF[0]  # 0.6
TAIL_SCALE = STRRTConfig().tail_a_max[0] / IIWA_ACCEL_REF[0]  # 0.8


def min_time_from_rest(dq: np.ndarray, v_max: np.ndarray, a_max: np.ndarray) -> float:
    """Lower bound on the time to move each joint by |dq| starting at rest, with |v| <= v_max
    and |a| <= a_max (final velocity free, joints independent): accelerate flat out, then
    cruise at v_max.  Per joint: sqrt(2d/a) if v_max is never reached, else d/v + v/(2a)."""
    d = np.abs(dq)
    t_acc = np.sqrt(2 * d / a_max)
    reach_vmax = d > v_max**2 / (2 * a_max)
    t = np.where(reach_vmax, d / v_max + v_max / (2 * a_max), t_acc)
    return float(np.max(t))


def oracle_config(a_max_on: bool, wall_scale: float, scale: float = PLANNER_SCALE):
    """The planner with `wall_scale` x the compute; joint-acceleration limits `scale` x
    IIWA_ACCEL_REF for transfers and max(TAIL_SCALE, scale + 0.2) x it for the tail."""
    return STRRTConfig(
        a_max=_scaled(scale) if a_max_on else None,
        tail_a_max=_scaled(max(TAIL_SCALE, scale + 0.2)),
        wall_scale=wall_scale,
        dT=0.125, max_goal_times=64, max_goals_per_T=6, max_ik_per_T=40, ik_restarts=10,
        max_tails=10, swivel_steps=10, n_random_ik_seeds=3,
    )


def certify(spec_dict: dict, wall_scale: float, sweep: tuple = ()) -> dict:
    from dyngrasp.controller import RobotModel
    from dyngrasp.grasps import generate_grasps
    from dyngrasp.harness import run_episode
    from dyngrasp.planning.collision import SpaceTimeChecker
    from dyngrasp.planning.controller import SpaceTimeRRTController, STRRTConfig
    from dyngrasp.planning.goals import GoalGenerator
    from dyngrasp.scenario import ScenarioSpec
    from dyngrasp.world import WorldModel

    spec = ScenarioSpec.from_dict(spec_dict)
    cfg = STRRTConfig()
    world, robot = WorldModel(spec), RobotModel()
    ck = SpaceTimeChecker(spec, world, cfg.margin_obstacle, cfg.margin_env)
    grasps = [g for g in generate_grasps(world.object) if g.family in cfg.grasp_families]
    gg = GoalGenerator(world, robot, cfg.tail_v_scale * robot.v_limits, cfg.tail, grasps,
                       a_max=accel_vector(cfg.tail_a_max))
    rng = np.random.default_rng(0)
    gg.start([np.asarray(spec.q0, float)], rng, n_random_seeds=3)
    for k in list(gg._seeds):  # random seeds for every grasp, not only the nominal one
        gg._seeds[(k[0], 99)] = rng.uniform(robot.q_lower, robot.q_upper)
    def tail_free(tr):  # as the planner: the tail, and holding its end pose for post_grasp_hold
        return ck.path_free(tr.q, tr.t) and ck.hold_free(tr.q[-1], tr.t[-1], cfg.post_grasp_hold)

    goals = []
    for T in gg.candidate_times(0.0, dT=0.1, max_count=400):
        goals += gg.generate_for(T, tail_free, ck.state_free, max_per_T=30,
                                 swivel_steps=12, max_ik_per_T=200, ik_restarts=10, max_tails=30)

    # the same search with the grasp tail's acceleration limit off: does that limit remove all goals?
    gg_free = GoalGenerator(world, robot, cfg.tail_v_scale * robot.v_limits, cfg.tail, grasps, a_max=None)
    gg_free.start([np.asarray(spec.q0, float)], np.random.default_rng(0), n_random_seeds=3)
    free_goals = []
    for T in gg_free.candidate_times(0.0, dT=0.1, max_count=400):
        free_goals += gg_free.generate_for(T, tail_free, ck.state_free, max_per_T=2,
                                           swivel_steps=12, max_ik_per_T=200, ik_restarts=10, max_tails=30)
    n_free = len(free_goals)
    from dyngrasp.trajectory import motion_limits
    # smallest peak joint acceleration (as a fraction of IIWA_ACCEL_REF) of any feasible grasp
    # tail: possible iff the tail's scale >= this
    ref = np.asarray(IIWA_ACCEL_REF)
    tail_accel_needed = min((float((motion_limits(g.tail)[1] / ref).max()) for g in free_goals), default=np.inf)

    q0 = np.asarray(spec.q0, float)
    v_plan, a_plan = cfg.v_scale * robot.v_limits, accel_vector(cfg.a_max)
    reachable = [g for g in goals if min_time_from_rest(g.q - q0, v_plan, a_plan) <= g.T]

    out = {"scenario": spec.name, "object": spec.target.object, "target_speed": spec.cart.speed,
           "n_obstacles": len(spec.obstacles), "n_goals": len(goals), "n_goals_reachable": len(reachable),
           "n_goals_without_tail_accel_limit": n_free, "tail_accel_needed": round(tail_accel_needed, 2),
           "goal_stats": str({k: v for k, v in gg.stats.items() if v})}
    for tag, on in (("oracle", True), ("oracle_noaccel", False)):
        c = SpaceTimeRRTController(oracle_config(on, wall_scale))
        r = run_episode(spec, c, keep_log=False, latency="none")
        out[f"{tag}_outcome"] = r.outcome
        out[f"{tag}_success"] = bool(r.success)
        out[f"{tag}_grasp"] = next((l["grasp"] for l in reversed(c.plan_log) if "grasp" in l), "")
        out[f"{tag}_accel_max"] = r.cmd_accel_max
    # required planner acceleration: smallest swept limit at which the oracle grasps
    out["planner_accel_needed"] = PLANNER_SCALE if out["oracle_success"] else np.inf
    if not out["oracle_success"] and n_free > 0:
        for a in sweep:
            c = SpaceTimeRRTController(oracle_config(True, wall_scale, a))
            if run_episode(spec, c, keep_log=False, latency="none").success:
                out["planner_accel_needed"] = a
                break
    if out["n_goals"] == 0:
        out["label"] = "no feasible grasp found (exhaustive search)"
    elif out["n_goals_reachable"] == 0:
        out["label"] = "dynamically unreachable (time-optimal bound)"
    elif out["oracle_success"]:
        out["label"] = "FEASIBLE: online planner missed it"
    else:
        out["label"] = "likely infeasible (oracle failed too)"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--results", help="results.csv: certify the scenes this controller did not grasp")
    ap.add_argument("--controller", default="st_rrt")
    ap.add_argument("--names", nargs="*", default=[])
    ap.add_argument("--wall-scale", type=float, default=20.0)
    ap.add_argument("--sweep", nargs="*", type=float, default=[0.8, 1.0, 1.5],
                    help="acceleration limits (fractions of IIWA_ACCEL_REF) to retry the oracle at")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/certify")
    args = ap.parse_args()
    by = {s.name: s for s in S.load_scenario_set(args.scenarios)}
    names = list(args.names)
    if args.results:
        df = pd.read_csv(args.results)
        df = df[df.controller == args.controller]
        names += [n for n in df[~df.success].scenario if n not in names]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(certify, by[n].to_dict(), args.wall_scale, tuple(args.sweep)) for n in names]
        for k, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            print(f"[{k}/{len(names)}] {rows[-1]['scenario']}: {rows[-1]['label']}", flush=True)
            pd.DataFrame(rows).to_csv(out / "certify.csv", index=False)
    df = pd.DataFrame(rows).sort_values("scenario")
    pd.set_option("display.width", 250)
    print("\n" + df[["scenario", "object", "target_speed", "n_obstacles", "n_goals", "n_goals_without_tail_accel_limit",
                     "n_goals_reachable", "tail_accel_needed", "planner_accel_needed",
                     "oracle_outcome", "oracle_noaccel_outcome", "label"]].to_string(index=False))
    print("\n" + df["label"].value_counts().to_string())
    print(f"\nno goal only because of the tail acceleration limit: "
          f"{list(df[(df.n_goals == 0) & (df.n_goals_without_tail_accel_limit > 0)].scenario)}")
    print(f"possible only without acceleration limits (oracle): "
          f"{list(df[df.oracle_noaccel_success & ~df.oracle_success].scenario)}")


if __name__ == "__main__":
    main()
