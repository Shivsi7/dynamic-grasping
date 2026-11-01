#!/usr/bin/env python
"""Tables and figures from one or more results/*/results.csv.

    python scripts/summarize.py results/sweep            # -> results/sweep/summary.md + figures
    python scripts/summarize.py results/a results/b      # several runs, concatenated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dyngrasp.harness import aggregate  # noqa: E402


def load(dirs) -> pd.DataFrame:
    frames = []
    for d in dirs:
        df = pd.read_csv(Path(d) / "results.csv")
        df["run"] = Path(d).name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def rate_plot(df, x, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for name, g in df.groupby("controller"):
        agg = g.groupby(x).agg(
            success=("success", "mean"),
            collided=("collided", "mean"),
            clearance=("min_clearance_obstacles", lambda s: np.nanmean(s.replace(np.inf, np.nan))),
        )
        axes[0].plot(agg.index, agg["success"], "o-", label=name)
        axes[1].plot(agg.index, agg["collided"], "o-", label=name)
        axes[2].plot(agg.index, agg["clearance"], "o-", label=name)
    for ax, title in zip(axes, ["grasp success rate", "collision rate", "min clearance to obstacles [m]"]):
        ax.set_xlabel(x)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None, help="output directory (default: first results dir)")
    args = ap.parse_args()
    df = load(args.dirs)
    out = Path(args.out or args.dirs[0])
    out.mkdir(parents=True, exist_ok=True)

    lines = ["# Summary", "", f"{len(df)} episodes from {', '.join(args.dirs)}", ""]
    lines += ["## By controller", "", aggregate(df).round(3).to_markdown(), ""]
    lines += ["## Outcomes", "", pd.crosstab(df["controller"], df["outcome"]).to_markdown(), ""]
    if df["target_speed"].nunique() > 1:
        df["target_speed_bin"] = df["target_speed"].round(2)
        lines += ["## By target speed", "", aggregate(df, by=("controller", "target_speed_bin")).round(3).to_markdown(), ""]
        rate_plot(df, "target_speed_bin", out / "by_target_speed.png")
    if df["n_obstacles"].nunique() > 1:
        lines += ["## By number of obstacles", "", aggregate(df, by=("controller", "n_obstacles")).round(3).to_markdown(), ""]
        rate_plot(df, "n_obstacles", out / "by_n_obstacles.png")
    failures = df[df["outcome"] != "success"][["scenario", "controller", "outcome", "collision_with", "t_first_collision", "t_grasp"]]
    lines += ["## Failure cases", "", failures.to_markdown(index=False), ""]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
