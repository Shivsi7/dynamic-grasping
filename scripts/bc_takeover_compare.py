#!/usr/bin/env python
"""Two takeover (DAgger) runs of make_bc_demos.py on the same scenes and switch times (same --policy, --seed,
--episodes, --switches), e.g. the expert before and after a change: outcomes, (C)'s grasp time, label lengths, the
velocity change at the switch and wall time, overall and by whether the switch came before or after (C)'s approach
would have started (t_enter - lead_time, 1 s).

    python scripts/bc_takeover_compare.py OLD_PARTS_GLOB NEW_PARTS_GLOB   (quoted globs of run directories)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dyngrasp.harness import shared_robot_model
from dyngrasp.rl.scenes import make_scene
from dyngrasp.world import WorldModel

LEAD = 1.0


def load(pattern: str) -> pd.DataFrame:
    rows = []
    v_max = shared_robot_model().v_limits
    for d in sorted(glob.glob(pattern)):
        for line in (Path(d) / "episodes.jsonl").read_text().splitlines():
            m = json.loads(line)
            if "t_switch" not in m:
                continue
            r = {"part": Path(d).name, "episode": m["episode"], "scene": m["scene"], "level": m["level"], "seed": m["seed"],
                 "t_switch": m["t_switch"], "outcome": m["outcome"], "wall_s": m["wall_s"],
                 "t_grasp": (m.get("c_result") or {}).get("t_grasp"), "n_labels": m.get("n_steps")}
            f = Path(d) / "eps" / f"{m['episode']}.npz"
            if m["outcome"] == "success" and f.exists():
                z = np.load(f)
                v_sw = z["obs"][0, 7:14] * v_max  # commanded velocity at the switch (rad/s)
                r["speed_at_switch"] = float(np.max(np.abs(v_sw)))
                r["dv_switch"] = float(np.max(np.abs(z["act"][0, :7] * v_max - v_sw)))  # first label's end velocity - it
                if len(z["act"]) > 2:
                    r["dv_next"] = float(np.max(np.abs((z["act"][1, :7] - z["obs"][1, 7:14]) * v_max)))
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    old, new = load(sys.argv[1]), load(sys.argv[2])
    key = ["part", "episode"]
    df = old.merge(new, on=key, suffixes=("_old", "_new"))
    assert (df.scene_old == df.scene_new).all() and (df.t_switch_old == df.t_switch_new).all(), "not the same takeovers"
    t_enter = {s: WorldModel(make_scene(lv, sd)).t_enter for s, lv, sd in df[["scene_old", "level_old", "seed_old"]].drop_duplicates().values}
    df["late"] = [t >= t_enter[s] - LEAD - 1e-9 for s, t in zip(df.scene_old, df.t_switch_old)]
    out = {"n_takeovers": len(df), "n_late_switches": int(df.late.sum())}
    for name, g in (("all", df), ("early", df[~df.late]), ("late", df[df.late])):
        o = {}
        for arm in ("old", "new"):
            o[f"outcomes_{arm}"] = g[f"outcome_{arm}"].value_counts().to_dict()
            o[f"success_{arm}"] = float((g[f"outcome_{arm}"] == "success").mean()) if len(g) else None
            o[f"wall_s_median_{arm}"] = float(g[f"wall_s_{arm}"].median()) if len(g) else None
        both = g[(g.outcome_old == "success") & (g.outcome_new == "success")]
        if len(both):
            d = both.t_grasp_new - both.t_grasp_old
            o.update(both_success=len(both), t_grasp_old_mean=float(both.t_grasp_old.mean()),
                     t_grasp_new_mean=float(both.t_grasp_new.mean()), t_grasp_diff_mean=float(d.mean()),
                     t_grasp_diff_median=float(d.median()), t_grasp_diff_se=float(d.std() / np.sqrt(len(d))),
                     n_labels_old_mean=float(both.n_labels_old.mean()), n_labels_new_mean=float(both.n_labels_new.mean()))
            for arm in ("old", "new"):
                o[f"speed_at_switch_median_{arm}"] = float(both[f"speed_at_switch_{arm}"].median())
                o[f"dv_switch_median_{arm}"] = float(both[f"dv_switch_{arm}"].median())
                o[f"dv_switch_p90_{arm}"] = float(both[f"dv_switch_{arm}"].quantile(0.9))
                o[f"dv_next_median_{arm}"] = float(both[f"dv_next_{arm}"].median())
        out[name] = o
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
