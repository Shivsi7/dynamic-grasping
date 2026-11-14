#!/usr/bin/env python
"""A (D) evaluation (scripts/eval_rl.py results.csv) per scene next to (A) reactive, (B) st_rrt and (C) hybrid.

    python scripts/compare_rl.py results/bc0/results.csv [--abc scenarios/rl_experiment30_abc.csv] [--name bc]

scenarios/rl_experiment30_abc.csv: (A)/(B)/(C) on scenarios/rl_experiment30.yaml on ORCD (mit_normal, AMD EPYC 7763,
20 CPUs), measured latency, 8 episodes in parallel, then every (B)/(C) non-success rerun serially (the rerun
counts); code of commit 667e1bb.  (Two of its held-out outcomes differ from scenarios/heldout_mixed_s5000_abc.csv,
run on a Mac: mix_0003 (B) timeout there success, mix_0068 (C) collision there timeout.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
OUTCOMES = ("success", "collision", "timeout", "dropped")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--abc", default=str(SCENARIOS / "rl_experiment30_abc.csv"))
    ap.add_argument("--name", default="D")
    args = ap.parse_args()
    d = pd.read_csv(args.results).assign(controller=args.name)
    df = pd.concat([pd.read_csv(args.abc), d[d.scenario.isin(pd.read_csv(args.abc).scenario)]], ignore_index=True)
    methods = ["reactive", "st_rrt", "hybrid", args.name]
    pd.set_option("display.width", 250)
    print(df.pivot(index="scenario", columns="controller", values="outcome")[methods].to_string(), "\n")
    s = df.groupby("controller").outcome.value_counts().unstack(fill_value=0).reindex(index=methods, columns=OUTCOMES, fill_value=0)
    s["t_grasp_median"] = df[df.outcome == "success"].groupby("controller").t_grasp.median()
    s["n"] = df.groupby("controller").size()
    print(s.to_string(), "\n")
    o = df.pivot(index="scenario", columns="controller", values="outcome")
    print(f"{args.name} vs (C) hybrid (rows: hybrid):")
    print(pd.crosstab(o.hybrid, o[args.name]).to_string())


if __name__ == "__main__":
    main()
