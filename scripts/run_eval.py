#!/usr/bin/env python
"""Batch evaluation: every controller on every scenario of a set.

    # 20 random scenarios (seed 0), both baselines, results in results/dev/
    python scripts/run_eval.py --n 20 --seed 0 --out results/dev

    # a sweep: target speed x obstacle count, 5 seeds each
    python scripts/run_eval.py --sweep --out results/sweep

    # re-use a saved scenario set (identical scenarios for a new controller)
    python scripts/run_eval.py --scenarios results/dev/scenarios.yaml --controllers naive --out results/dev_naive

    # the "initial test": the tall post / the plank, at several speeds and crossing times
    python scripts/run_eval.py --tall-blocker --out results/post
    python scripts/run_eval.py --plank-blocker --out results/plank

Add a controller: import it and put it in CONTROLLERS (name -> factory).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from dyngrasp import scenario as S  # noqa: E402
from dyngrasp.baselines import HoldController, NaiveInterceptController  # noqa: E402
from dyngrasp.harness import aggregate, run_batch  # noqa: E402
from dyngrasp.hybrid import HybridController  # noqa: E402
from dyngrasp.reactive import ReactiveQPController  # noqa: E402

CONTROLLERS = {
    "hold": HoldController,
    "naive": NaiveInterceptController,
    "reactive": ReactiveQPController,
    "hybrid": HybridController,
}

SWEEP_TARGET_SPEEDS = [0.05, 0.1, 0.2, 0.3]
SWEEP_N_OBSTACLES = [0, 1, 2, 4]


def build_specs(args) -> list[S.ScenarioSpec]:
    if args.scenarios:
        return S.load_scenario_set(args.scenarios)
    if args.tall_blocker or args.plank_blocker:
        preset = S.plank_blocker if args.plank_blocker else S.tall_blocker
        return [
            preset(v, o, t_cross=S.WARMUP + dt)
            for v, o, dt in itertools.product([0.05, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.5], args.cross_offsets)
        ]
    if args.sweep:
        specs = []
        k = 0
        for v, n in itertools.product(SWEEP_TARGET_SPEEDS, SWEEP_N_OBSTACLES):
            for _ in range(args.seeds_per_cell):
                seed = args.seed + k
                k += 1
                specs.append(
                    S.sample_scenario(
                        S.np.random.default_rng(seed),
                        n_obstacles=n,
                        target_speed=v,
                        objects=tuple(args.objects),
                        name=f"sweep_v{v:.2f}_n{n}_s{seed}",
                        seed=seed,
                    )
                )
        return specs
    return S.generate_scenario_set(args.seed, args.n, objects=tuple(args.objects))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="number of random scenarios")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objects", nargs="+", default=["soup_can"], help="YCB objects to sample from")
    ap.add_argument("--scenarios", help="YAML scenario set to re-use")
    ap.add_argument("--sweep", action="store_true", help="target speed x obstacle count grid")
    ap.add_argument("--seeds-per-cell", type=int, default=5)
    ap.add_argument("--tall-blocker", action="store_true", help="tall post: target speed x obstacle speed x crossing time")
    ap.add_argument("--plank-blocker", action="store_true", help="same grid with the thin plank (wider sweep of the corridor)")
    ap.add_argument("--cross-offsets", nargs="+", type=float, default=[0.5, 1.0, 2.0],
                    help="blocker crossing times, seconds after the target becomes reachable")
    ap.add_argument("--controllers", nargs="+", default=sorted(CONTROLLERS), choices=sorted(CONTROLLERS))
    ap.add_argument("--out", default="results/latest")
    ap.add_argument("--keep-going", action="store_true", help="do not stop episodes at the first collision")
    ap.add_argument("--save-logs", action="store_true", help="save per-episode trajectories (npz)")
    args = ap.parse_args()

    specs = build_specs(args)
    controllers = {name: CONTROLLERS[name] for name in args.controllers}
    print(f"{len(specs)} scenarios x {list(controllers)} -> {args.out}")
    df = run_batch(specs, controllers, out_dir=args.out, stop_on_collision=not args.keep_going, save_logs=args.save_logs)
    pd.set_option("display.width", 200)
    print()
    print(aggregate(df).round(3))
    if args.sweep:
        print()
        print(aggregate(df, by=("controller", "target_speed")).round(3))
        print()
        print(aggregate(df, by=("controller", "n_obstacles")).round(3))


if __name__ == "__main__":
    main()
