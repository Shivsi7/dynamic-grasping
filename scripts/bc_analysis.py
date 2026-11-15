"""(D) BC failure analysis: copycat baseline, drift from (C), action modes near obstacles.

  python scripts/bc_analysis.py copycat DATASET CKPT     # MSE on DATASET's held-out episodes vs "repeat the
                                                         # current velocity / gripper" (split of bc.load_dataset)
  python scripts/bc_analysis.py run CKPT OUT_DIR [--workers 30]   # the policy and (C) (cached in OUT_DIR/../c_val.pkl)
                                                                  # on the first 50 validation scenes
  python scripts/bc_analysis.py analyse OUT_DIR DATASET  # drift from (C), phases, nearest demo states
"""
import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

P_CENTRE = np.array([0.5, 0.0, 0.4])
I_V, I_G, I_P, I_EP, I_O = slice(7, 14), 14, slice(16, 19), slice(25, 28), 38  # obs layout (observation.py)


def states_arrays(states):
    return (np.array([s.t for s in states]), np.array([s.q for s in states]),
            np.array([s.X_WG.translation() for s in states]))


def run_c(spec_dict):
    from dyngrasp.harness import run_episode
    from dyngrasp.hybrid import HybridController
    from dyngrasp.rl.demos import Aborted, Recorder
    from dyngrasp.scenario import ScenarioSpec

    spec = ScenarioSpec.from_dict(spec_dict)
    rec = Recorder(HybridController(n_workers=1), 0.1, max_wall_s=300)
    try:
        r = run_episode(spec, rec, keep_log=True, latency="none")
    except Aborted:
        return {"scenario": spec.name, "row": {"outcome": "aborted"}}
    t, q, p = states_arrays(rec.states)
    log = [(e["t"], e["phase"]) for e in r.controller_log if "phase" in e and "t" in e]
    lt = np.array([x[0] for x in log])
    phase = [log[i][1] if i >= 0 else "start" for i in np.searchsorted(lt, t + 1e-9) - 1]
    return {"scenario": spec.name, "row": r.row(), "t": t, "q": q, "p": p, "phase": phase}


def run_bc(ckpt, spec_dict, latency):
    from dyngrasp.harness import run_episode
    from dyngrasp.rl.controller import RLController
    from dyngrasp.rl.demos import Recorder
    from dyngrasp.scenario import ScenarioSpec

    spec = ScenarioSpec.from_dict(spec_dict)
    ctrl = RLController(ckpt)
    actor, seen = ctrl.actor, []

    def rec_actor(o):
        a = actor(o)
        seen.append((o, a))
        return a

    ctrl.actor = rec_actor
    rec = Recorder(ctrl, 0.1)
    r = run_episode(spec, rec, keep_log=False, latency=latency)
    t, q, p = states_arrays(rec.states)
    return {"scenario": spec.name, "row": r.row(), "t": t, "q": q, "p": p,
            "obs": np.array([o for o, _ in seen]), "act": np.array([a for _, a in seen])}


def cmd_run(args):
    import yaml

    from dyngrasp.rl.scenes import validation_scenes

    specs = [s.to_dict() for s in validation_scenes()[:50]]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    latency = yaml.safe_load((Path(args.ckpt) / "config.yaml").read_text())["env"]["latency"]
    c_path = out.parent / "c_val.pkl"
    with ProcessPoolExecutor(args.workers) as ex:
        fc = None if c_path.exists() else [ex.submit(run_c, s) for s in specs]
        fb = [ex.submit(run_bc, str(Path(args.ckpt).resolve()), s, latency) for s in specs]
        if fc is not None:
            c_path.write_bytes(pickle.dumps([f.result() for f in fc]))
        (out / "bc_val.pkl").write_bytes(pickle.dumps([f.result() for f in fb]))
    print("rollouts done", flush=True)


def phase_of(obs, t, t_grasp):
    if not np.isnan(t_grasp) and t >= t_grasp:
        return "lift"
    if obs[I_G] < 0.9:
        return "close"
    if obs[I_O + 10] > 0.5 and obs[I_O + 9] * 0.25 < 0.10:
        return "avoid"
    if np.linalg.norm(obs[I_EP]) * 0.5 < 0.10:
        return "descend"
    return "approach"


def cmd_copycat(args):
    from dyngrasp.rl.bc import load_dataset
    from dyngrasp.rl.controller import NumpyActor

    d = load_dataset(args.dataset, 0.1, 0)
    va = d["val"]
    obs, act = d["obs"][va].astype(np.float64), d["act"][va].astype(np.float64)
    copy = np.concatenate([obs[:, I_V], obs[:, I_G:I_G + 1]], 1)
    actor = NumpyActor(dict(np.load(Path(args.ckpt) / "actor.npz")))
    bc = np.concatenate([actor(obs[i:i + 65536]) for i in range(0, len(obs), 65536)])
    change = np.linalg.norm(act - copy, axis=1)
    top = change >= np.quantile(change, 0.9)

    def m(x, sel=slice(None), cols=slice(None)):
        return float(np.mean((x[sel][:, cols] - act[sel][:, cols]) ** 2))

    res = {"n_val": len(va),
           "mse_bc": m(bc), "mse_copy": m(copy), "mse_zero": m(np.zeros_like(act)),
           "joints_bc": m(bc, cols=slice(0, 7)), "joints_copy": m(copy, cols=slice(0, 7)),
           "grip_bc": m(bc, cols=slice(7, 8)), "grip_copy": m(copy, cols=slice(7, 8)),
           "top10_change_bc": m(bc, top), "top10_change_copy": m(copy, top),
           "mse_bc_vs_copy": float(np.mean((bc - copy) ** 2))}
    print(json.dumps(res, indent=1))


def cmd_analyse(args):
    from dyngrasp.rl.bc import load_dataset

    out = Path(args.out_dir)
    C = {r["scenario"]: r for r in pickle.loads((out.parent / "c_val.pkl").read_bytes())}
    B = pickle.loads((out / "bc_val.pkl").read_bytes())
    rows = []
    for b in B:
        c, row = C[b["scenario"]], b["row"]
        r = {"scenario": b["scenario"], "bc": row["outcome"], "c": c["row"]["outcome"],
             "t_fail": row["t_first_collision"] if row["outcome"] == "collision" else np.nan,
             "with": row.get("collision_category") if row["outcome"] == "collision" else ""}
        if "t" in c:
            n = min(len(c["t"]), len(b["t"]))
            assert np.allclose(c["t"][:n], b["t"][:n])
            dp = np.linalg.norm(b["p"][:n] - c["p"][:n], axis=1)
            dq = np.abs(b["q"][:n] - c["q"][:n]).max(1)
            kp, kq = np.argmax(dp > 0.05) if np.any(dp > 0.05) else None, np.argmax(dq > 0.1) if np.any(dq > 0.1) else None
            tg = row["t_grasp"] if row["t_grasp"] is not None else np.nan
            r.update(t_drift_p=b["t"][kp] if kp is not None else np.nan, t_drift_q=b["t"][kq] if kq is not None else np.nan,
                     phase_drift=phase_of(b["obs"][kp], b["t"][kp], tg) if kp is not None else "",
                     c_phase_drift=c["phase"][kp] if kp is not None else "")
            if r["t_fail"] == r["t_fail"]:
                kf = min(int(r["t_fail"] / 0.1), len(b["obs"]) - 1)
                r["phase_fail"] = phase_of(b["obs"][kf], b["t"][kf], tg)
                r["c_phase_fail"] = c["phase"][min(kf, len(c["phase"]) - 1)]
        rows.append(r)
    import pandas as pd

    pd.set_option("display.width", 250)
    df = pd.DataFrame(rows)
    df["lead_p"] = df.t_fail - df.t_drift_p
    print(df.to_string(float_format=lambda x: f"{x:.2f}"))
    col = df[df.bc == "collision"]
    print("\nBC vs (C) outcomes:\n", pd.crosstab(df.c, df.bc).to_string())
    print(f"\ndrift > 5 cm EE: {df.t_drift_p.notna().sum()}/{len(df)} episodes; median t {df.t_drift_p.median():.2f} s;"
          f" > 0.1 rad: {df.t_drift_q.notna().sum()}, median t {df.t_drift_q.median():.2f} s")
    print("collisions: phase at drift", col.phase_drift.value_counts().to_dict(), " phase at collision",
          col.phase_fail.value_counts().to_dict(), " (C) phase at drift", col.c_phase_drift.value_counts().to_dict())
    print(f"collisions: drift (5 cm) precedes the collision by median {col.lead_p.median():.2f} s "
          f"(quartiles {col.lead_p.quantile(0.25):.2f}, {col.lead_p.quantile(0.75):.2f}); no drift before collision: "
          f"{(col.lead_p.isna() | (col.lead_p < 0)).sum()}/{len(col)}")
    print("non-collision failures: phase at drift", df[(df.bc != "success") & (df.bc != "collision")].phase_drift.value_counts().to_dict())
    print("successes: drift episodes", df[df.bc == "success"].t_drift_p.notna().sum(), "/", (df.bc == "success").sum())

    # modes: nearest demo states of the BC state 0.5 s before an obstacle collision
    d = load_dataset(args.dataset, 0.1, 0)
    obs, act = d["obs"], d["act"]
    z = np.load(args.dataset)
    ep_id = np.repeat(np.arange(len(z["ep_len"])), z["ep_len"])
    tr, va = d["train"], d["val"]
    mu, sd = obs[tr].mean(0), np.maximum(obs[tr].std(0), 1e-3)
    X = ((obs[tr] - mu) / sd).astype(np.float32)
    X2 = np.einsum("ij,ij->i", X, X)
    p = obs[:, I_P] * 0.5
    i5 = np.minimum(np.arange(len(obs)) + 5, len(obs) - 1)
    disp = np.where((ep_id[i5] == ep_id)[:, None], p[i5] - p, np.nan)

    def stats(o, k=20):
        x = ((o - mu) / sd).astype(np.float32)
        dist2 = X2 - 2 * X @ x + x @ x
        idx = np.argpartition(dist2, k)[:k]
        nn, nd = tr[idx], float(np.sqrt(np.maximum(dist2[idx], 0)).mean())
        u = obs[nn, I_O:I_O + 3]
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        dd = disp[nn]
        lat = dd - np.sum(dd * u, 1, keepdims=True) * u
        ok = np.isfinite(lat).all(1) & (np.linalg.norm(lat, axis=1) > 0.01)

        def opp(v):
            if len(v) < 2:
                return np.nan
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            cs = v @ v.T
            return float(np.mean(cs[np.triu_indices(len(v), 1)] < -0.5))

        aj = act[nn, :7]
        aj = aj[np.linalg.norm(aj, axis=1) > 0.05]
        a_mean = act[nn].mean(0)
        return {"nn_dist": nd, "lat_opposite": opp(lat[ok]), "act_opposite": opp(aj),
                "spread": float(np.linalg.norm(act[nn] - a_mean, axis=1).mean()), "a_mean": a_mean}

    rng = np.random.default_rng(0)
    near = va[(obs[va, I_O + 10] > 0.5) & (obs[va, I_O + 9] * 0.25 < 0.10)]
    base_near = [stats(obs[i]) for i in rng.choice(near, min(200, len(near)), replace=False)]
    base_all = [stats(obs[i]) for i in rng.choice(va, 200, replace=False)]
    q = []
    for b in B:
        row = b["row"]
        if row["outcome"] != "collision" or row.get("collision_category") != "obstacle":
            continue
        k = max(0, min(int(row["t_first_collision"] / 0.1) - 5, len(b["obs"]) - 1))
        s = stats(b["obs"][k])
        s.update(scenario=b["scenario"], bc_to_mean=float(np.linalg.norm(b["act"][k] - s["a_mean"])))
        q.append(s)

    def summ(L, name):
        f = {key: np.nanmedian([s[key] for s in L]) for key in ("nn_dist", "lat_opposite", "act_opposite", "spread")}
        f.update(frac_lat_bimodal=float(np.mean([s["lat_opposite"] > 0.1 for s in L if s["lat_opposite"] == s["lat_opposite"]])))
        if "bc_to_mean" in L[0]:
            f["bc_to_nn_mean"] = float(np.median([s["bc_to_mean"] for s in L]))
        print(f"{name:42s} n={len(L):3d} " + " ".join(f"{k}={v:.3f}" for k, v in f.items()))

    print("\nnearest 20 demo states (train split, standardised obs); lat_opposite: fraction of neighbour pairs whose"
          " EE motion over the next 0.5 s, sideways to the nearest obstacle, points in opposite directions (cos < -0.5)")
    summ(q, "BC state 0.5 s before obstacle collision")
    summ(base_near, "baseline: held-out demo states < 10 cm")
    summ(base_all, "baseline: held-out demo states, any")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("copycat"); a.add_argument("dataset"); a.add_argument("ckpt")
    a = sub.add_parser("run"); a.add_argument("ckpt"); a.add_argument("out_dir"); a.add_argument("--workers", type=int, default=30)
    a = sub.add_parser("analyse"); a.add_argument("out_dir"); a.add_argument("dataset")
    args = ap.parse_args()
    {"copycat": cmd_copycat, "run": cmd_run, "analyse": cmd_analyse}[args.cmd](args)
    sys.stdout.flush()
