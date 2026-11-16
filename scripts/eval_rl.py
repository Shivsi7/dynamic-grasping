#!/usr/bin/env python
"""Evaluate a (D) checkpoint like (A)/(B)/(C): harness.run_episode, measured latency by default.

    python scripts/eval_rl.py runs/sac0/best --scenarios scenarios/rl_experiment30.yaml --workers 8 --out results/rl
    python scripts/eval_rl.py runs/sac0/checkpoints/step_0001000000 --scenarios scenarios/dev_v0.yaml --names dev_003

    # the full test (214 scenes) as (A)/(B)/(C) were run: 8 in parallel, then every non-success alone (that result counts)
    python scripts/eval_rl.py CKPT --scenarios scenarios/dev_v0.yaml scenarios/preset_grids.yaml \
        scenarios/heldout_mixed_s5000.yaml --workers 8 --rerun-failures --out results/rl_full

The checkpoint is a directory with actor.npz and config.yaml (a training checkpoint or run_dir/best).
Writes <out>/results.csv (one EpisodeResult row per scene, as the other controllers' evaluations, plus its set) and
<out>/decision_ms.npz (per scene: the compute time of each policy decision, ms); with --rerun-failures also
<out>/results_parallel.csv (the first pass).
Measured latency is wall time: keep --workers at or below the number of physical cores.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HEADLESS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dyngrasp.scenario import load_scenario_set


def run_one(checkpoint: str, spec_dict: dict, latency) -> tuple[dict, np.ndarray]:
    from dyngrasp.harness import run_episode
    from dyngrasp.rl.controller import RLController
    from dyngrasp.scenario import ScenarioSpec

    spec = ScenarioSpec.from_dict(spec_dict)
    ctrl, ms = RLController(checkpoint), []
    update = ctrl.update

    def timed(state, world):  # the controller's update wall time on the ticks where the policy decides
        t_next, t0 = ctrl._t_next, time.perf_counter()
        cmd = update(state, world)
        if ctrl._t_next != t_next:
            ms.append((time.perf_counter() - t0) * 1e3)
        return cmd

    ctrl.update = timed
    r = run_episode(spec, ctrl, keep_log=False, latency=latency)
    return {**r.row(), "checkpoint": checkpoint}, np.array(ms, np.float32)


def run_all(checkpoint: str, specs: list, sets: dict, latency, workers: int, csv: Path) -> tuple[list, dict]:
    rows, ms = [], {}
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(run_one, checkpoint, s.to_dict(), latency) for s in specs]
        for k, f in enumerate(as_completed(futs), 1):
            row, ms[row["scenario"]] = f.result()
            rows.append({**row, "set": sets[row["scenario"]]})
            print(f"[{k}/{len(specs)}] {row['scenario']:>40s}: {row['outcome']:9s} "
                  f"compute mean/max {row['compute_mean_ms']:.2f}/{row['compute_max_ms']:.1f} ms", flush=True)
            pd.DataFrame(rows).to_csv(csv, index=False)
    return rows, ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--scenarios", nargs="+", required=True, help="scenario set YAML file(s)")
    ap.add_argument("--names", nargs="*", help="only these scenes")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--latency", default="measured", help="'measured', 'none' or a fixed delay in seconds")
    ap.add_argument("--rerun-failures", action="store_true", help="then rerun every non-success alone; that result counts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    latency = args.latency if args.latency in ("measured", "none") else float(args.latency)

    sets = {s.name: Path(f).stem for f in args.scenarios for s in load_scenario_set(f)}
    specs = [s for f in args.scenarios for s in load_scenario_set(f)]
    if args.names:
        specs = [s for s in specs if s.name in set(args.names)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = str(Path(args.checkpoint).resolve())
    rows, ms = run_all(ckpt, specs, sets, latency, args.workers, out / ("results_parallel.csv" if args.rerun_failures else "results.csv"))
    if args.rerun_failures:
        again = [s for s in specs if s.name in {r["scenario"] for r in rows if r["outcome"] != "success"}]
        print(f"rerunning {len(again)} non-successes one at a time", flush=True)
        rows2, ms2 = run_all(ckpt, again, sets, latency, 1, out / "results_serial.csv")
        rows = [r for r in rows if r["scenario"] not in ms2] + rows2
        ms.update(ms2)
        pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    np.savez(out / "decision_ms.npz", **ms)
    df = pd.DataFrame(rows)
    print(pd.crosstab(df["set"], df["outcome"], margins=True).to_string())
    d = np.concatenate(list(ms.values()))
    print(f"decision compute: median {np.median(d):.2f} ms, p99 {np.percentile(d, 99):.2f} ms, max {d.max():.1f} ms "
          f"({len(d)} decisions)")


if __name__ == "__main__":
    main()
