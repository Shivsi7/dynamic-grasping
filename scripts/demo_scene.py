#!/usr/bin/env python
"""Show a scenario in Meshcat, optionally driven by a controller.

    python scripts/demo_scene.py                        # tall blocker, naive controller
    python scripts/demo_scene.py --preset no_obstacles --controller hold
    python scripts/demo_scene.py --random --seed 3 --n-obstacles 3 --target-speed 0.2
    python scripts/demo_scene.py --scenario scenarios/dev_v0.yaml --index 4
    HEADLESS=1 python scripts/demo_scene.py             # no browser, just the result line
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from dyngrasp import scenario as S  # noqa: E402
from dyngrasp.baselines import HoldController, NaiveInterceptController  # noqa: E402
from dyngrasp.harness import run_episode  # noqa: E402
from dyngrasp.viz import get_meshcat, keep_open  # noqa: E402

CONTROLLERS = {"hold": HoldController, "naive": NaiveInterceptController}


def pick_scenario(args) -> S.ScenarioSpec:
    if args.scenario:
        specs = S.load_scenario_set(args.scenario)
        return specs[args.index]
    if args.random:
        rng = np.random.default_rng(args.seed)
        return S.sample_scenario(
            rng, n_obstacles=args.n_obstacles, target_speed=args.target_speed, name=f"rand_{args.seed}", seed=args.seed
        )
    if args.preset == "no_obstacles":
        return S.no_obstacles(args.target_speed or 0.1, object=args.object)
    preset = S.plank_blocker if args.preset == "plank_blocker" else S.tall_blocker
    return preset(args.target_speed or 0.1, args.obstacle_speed, object=args.object)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=["tall_blocker", "plank_blocker", "no_obstacles"], default="tall_blocker")
    ap.add_argument("--random", action="store_true", help="sample a random scenario instead of a preset")
    ap.add_argument("--scenario", help="YAML scenario set (from run_eval.py / save_scenario_set)")
    ap.add_argument("--index", type=int, default=0, help="which scenario of the set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-obstacles", type=int, default=None)
    ap.add_argument("--target-speed", type=float, default=None)
    ap.add_argument("--obstacle-speed", type=float, default=0.2)
    ap.add_argument("--object", default="soup_can")
    ap.add_argument("--controller", choices=sorted(CONTROLLERS), default="naive")
    ap.add_argument("--keep-going", action="store_true", help="do not stop the episode at the first collision")
    ap.add_argument("--record", action="store_true", help="publish a Meshcat recording to scrub through")
    args = ap.parse_args()

    spec = pick_scenario(args)
    print(f"scenario {spec.name}: {spec.summary()}")
    meshcat = get_meshcat()
    result = run_episode(
        spec,
        CONTROLLERS[args.controller](),
        meshcat=meshcat,
        record=args.record,
        stop_on_collision=not args.keep_going,
    )
    print(
        f"outcome={result.outcome} success={result.success} collided={result.collided} "
        f"t_grasp={result.t_grasp} t_success={result.t_success} "
        f"min_clearance={result.min_clearance_obstacles:.3f} m "
        f"compute={result.compute_mean_ms:.2f} ms/cycle (p95 {result.compute_p95_ms:.2f}) "
        f"sim {result.t_end:.1f} s in {result.sim_wall_s:.1f} s wall"
    )
    for c in result.collisions:
        print(f"  collision at t={c['t']:.2f}: {c['robot_body']} <-> {c['other_body']} ({c['category']})")
    keep_open()


if __name__ == "__main__":
    main()
