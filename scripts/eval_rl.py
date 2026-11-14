#!/usr/bin/env python
"""Evaluate a (D) checkpoint like (A)/(B)/(C): harness.run_episode, measured latency by default.

    python scripts/eval_rl.py runs/sac0/best --scenarios scenarios/rl_experiment30.yaml --workers 8 --out results/rl
    python scripts/eval_rl.py runs/sac0/checkpoints/step_0001000000 --scenarios scenarios/dev_v0.yaml --names dev_003

The checkpoint is a directory with actor.npz and config.yaml (a training checkpoint or run_dir/best).
Writes <out>/results.csv (one EpisodeResult row per scene, as the other controllers' evaluations).
Measured latency is wall time: keep --workers at or below the number of physical cores.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HEADLESS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from dyngrasp.scenario import load_scenario_set


def run_one(checkpoint: str, spec_dict: dict, latency) -> dict:
    from dyngrasp.harness import run_episode
    from dyngrasp.rl.controller import RLController
    from dyngrasp.scenario import ScenarioSpec

    spec = ScenarioSpec.from_dict(spec_dict)
    r = run_episode(spec, RLController(checkpoint), keep_log=False, latency=latency)
    return {**r.row(), "checkpoint": checkpoint}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--scenarios", nargs="+", required=True, help="scenario set YAML file(s)")
    ap.add_argument("--names", nargs="*", help="only these scenes")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--latency", default="measured", help="'measured', 'none' or a fixed delay in seconds")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    latency = args.latency if args.latency in ("measured", "none") else float(args.latency)

    specs = [s for f in args.scenarios for s in load_scenario_set(f)]
    if args.names:
        specs = [s for s in specs if s.name in set(args.names)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, str(Path(args.checkpoint).resolve()), s.to_dict(), latency) for s in specs]
        for k, f in enumerate(as_completed(futs), 1):
            row = f.result()
            rows.append(row)
            print(f"[{k}/{len(specs)}] {row['scenario']:>40s}: {row['outcome']:9s} "
                  f"compute mean/max {row['compute_mean_ms']:.2f}/{row['compute_max_ms']:.1f} ms", flush=True)
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    print(pd.DataFrame(rows)["outcome"].value_counts().to_string())


if __name__ == "__main__":
    main()
