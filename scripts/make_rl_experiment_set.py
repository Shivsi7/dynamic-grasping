#!/usr/bin/env python
"""The 30-scene (D) experiment set: 10 held-out mixed scenes + 5 each of dev, pillar, plank and tall post.

    python scripts/make_rl_experiment_set.py      # -> scenarios/rl_experiment30.yaml

The held-out scenes are stratified by how many of (A) reactive, (B) st_rrt and (C) hybrid succeed on
them (0-3; columns scenario, controller, outcome): proportional allocation by largest remainder, at
least one per non-empty stratum.  The others are drawn uniformly.  Seeded.

scenarios/heldout_mixed_s5000_abc.csv (commit 667e1bb, measured latency, 10-core Mac, 8 episodes in
parallel): (A) one parallel pass; (B) and (C) a parallel pass with every non-success rerun serially
(the rerun's outcome kept).  Successes: (A) 52, (B) 63, (C) 70 of 80.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from dyngrasp.scenario import load_scenario_set

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
METHODS = ("reactive", "st_rrt", "hybrid")


def allocate(sizes: dict[int, int], n: int) -> dict[int, int]:
    """n draws over strata: proportional by largest remainder, at least one per non-empty stratum."""
    total = sum(sizes.values())
    quota = {k: n * s / total for k, s in sizes.items()}
    out = {k: int(q) for k, q in quota.items()}
    for k in sorted(sizes, key=lambda k: quota[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    for k, size in sizes.items():  # a non-empty stratum left out takes one from the largest
        if size and not out[k]:
            out[max(out, key=out.get)] -= 1
            out[k] = 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(SCENARIOS / "heldout_mixed_s5000_abc.csv"),
                    help="(A)/(B)/(C) outcomes on heldout_mixed_s5000.yaml")
    ap.add_argument("--out", default=str(SCENARIOS / "rl_experiment30.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    heldout = load_scenario_set(SCENARIOS / "heldout_mixed_s5000.yaml")
    df = pd.read_csv(args.results)
    df = df[df.controller.isin(METHODS)]
    solved = (df.outcome == "success").groupby(df.scenario).sum()
    count = df.groupby("scenario").controller.nunique()
    names = [s.name for s in heldout]
    missing = [n for n in names if count.get(n, 0) != len(METHODS)]
    assert not missing, f"no result of every method for {missing}"
    strata = {k: [n for n in names if solved[n] == k] for k in range(len(METHODS) + 1)}
    picks = []
    for k, m in allocate({k: len(v) for k, v in strata.items()}, 10).items():
        chosen = sorted(str(x) for x in rng.choice(strata[k], m, replace=False)) if m else []
        print(f"held-out solved by {k}/3: {len(strata[k])} scenes, picked {chosen}")
        picks += chosen

    by_name = {s.name: s for s in heldout}
    out = [by_name[n] for n in picks]
    grids = load_scenario_set(SCENARIOS / "preset_grids.yaml")
    for label, pool in (("dev", load_scenario_set(SCENARIOS / "dev_v0.yaml")),
                        *((p, [s for s in grids if s.name.startswith(p)])
                          for p in ("pillar_and_blocker", "plank_blocker", "tall_blocker"))):
        idx = sorted(rng.choice(len(pool), 5, replace=False))
        out += [pool[i] for i in idx]
        print(f"{label}: {[pool[i].name for i in idx]}")

    header = (f"# (D) experiment set: scripts/make_rl_experiment_set.py --seed {args.seed}; held-out scenes\n"
              f"# stratified by (A)/(B)/(C) successes in {Path(args.results).name}\n")
    Path(args.out).write_text(header + yaml.safe_dump([s.to_dict() for s in out], sort_keys=False))
    print(f"{len(out)} scenes -> {args.out}")


if __name__ == "__main__":
    main()
