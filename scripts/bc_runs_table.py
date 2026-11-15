"""(D) BC runs side by side (run.json, log.csv, eval30/): python scripts/bc_runs_table.py RUN_DIR ..."""
import json
import sys
from pathlib import Path

import pandas as pd

rows = []
for d in map(Path, sys.argv[1:]):
    if not (d / "run.json").exists():
        continue
    run = json.loads((d / "run.json").read_text())
    log = pd.read_csv(d / "log.csv")
    cl = log.dropna(subset=["cl_success"])
    r = {"run": d.name, "episodes": run["n_episodes"], "transitions": run["n_train"] + run["n_val"],
         "epochs": len(log), "best_epoch": run["best_epoch"], "val_mse": run["best_val_mse"],
         "val50_best": run["best_closed_loop"]["success"], "val50_ckpt_mean": cl.cl_success.mean(),
         "val50_ckpt_max": cl.cl_success.max(), "wall_min": run["wall_min"]}
    e = d / "eval30" / "results.csv"
    if (d / "eval30" / "compare.txt").exists():
        x = pd.read_csv(e)
        r.update({f"e30_{o}": int((x.outcome == o).sum()) for o in ("success", "collision", "timeout", "dropped")})
        r["e30_t_grasp"] = x[x.outcome == "success"].t_grasp.median()
    rows.append(r)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
