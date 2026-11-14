#!/usr/bin/env python
"""(D) behaviour-cloning demonstrations from (C) (dyngrasp/rl/demos.py), and the dataset built from them.

    python scripts/make_bc_demos.py generate --out runs/demos/part0 --episodes 500 --workers 16 --seed 0
    python scripts/make_bc_demos.py merge runs/demos/part* --out runs/demos/dataset.npz

generate: (C) on fresh training scenes (scenes.sample_training_scene: never an evaluation or validation scene),
the curriculum level of each drawn with config demos.level_weights; scene e of a run is fixed by (--seed, e).
One process per worker, one episode at a time each ((C) with n_workers=1).  Each (C) success is replayed
open-loop through RLController (the fidelity check; --no-replay skips it).  Writes <out>/episodes.jsonl (every
episode's metadata, failures too) and <out>/eps/<e>.npz (obs, actions of the successes, and obs_replay: the
observations RLController computed in the replay).  Stops starting
episodes after --max-minutes.  Running it again on the same --out skips the episodes already there.
merge: the successes of the given runs (one per scene: duplicates dropped) -> one compressed .npz:
  obs (N, 84) float32, act (N, 8) float32, obs_replay (N, 84) float32 (NaN where the replay had ended or did not
  run), ep_start / ep_len / level / seed / replay_success (E,) per episode (-1: not replayed), meta: JSON with the per-episode metadata (scene spec, (C) result) and the
  generation settings (config, code and (C) commits); and <out>.summary.json with the statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("HEADLESS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from dyngrasp.rl.config import load_config


def _json(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def episode(cfg: dict, seed: int, e: int, out: str, do_replay: bool) -> dict:
    from dyngrasp.harness import shared_robot_model
    from dyngrasp.rl.demos import replay, run_demo
    from dyngrasp.rl.scenes import sample_training_scene

    rng = np.random.default_rng([seed, e])
    w = np.asarray(cfg["demos"]["level_weights"], float)
    level = int(rng.choice(len(w), p=w / w.sum()))
    t = time.perf_counter()
    try:
        spec = sample_training_scene(level, rng)
        d = run_demo(spec, level, cfg)
    except Exception as exc:  # noqa: BLE001  (recorded, not fatal for the run)
        return {"episode": e, "level": level, "outcome": "error", "error": repr(exc), "fingerprint": None,
                "wall_s": time.perf_counter() - t}
    d["episode"] = e
    if d["outcome"] == "success":
        obs, act = d.pop("obs"), d.pop("actions")
        d["n_steps"] = len(act)
        if do_replay:
            t = time.perf_counter()
            r, seen = replay(spec, act, cfg["env"])
            n = min(len(seen), len(obs))
            robot = shared_robot_model()
            dq = np.abs(seen[:n, :7] - obs[:n, :7]) * 0.5 * (robot.q_upper - robot.q_lower)  # measured q, rad
            d.update(replay_q_diff_max=float(dq.max()), replay_outcome=r.outcome, replay_t_grasp=r.t_grasp, replay_wall_s=time.perf_counter() - t,
                     replay_obs_diff_max=float(np.max(np.abs(seen[:n] - obs[:n]))),
                     replay_obs_diff_t0=float(np.max(np.abs(seen[0] - obs[0]))))
        obs_replay = np.full_like(obs, np.nan)
        if do_replay:
            obs_replay[:n] = seen[:n]
        tmp = Path(out) / "eps" / f".{e}.npz"
        np.savez_compressed(tmp, obs=obs, act=act, obs_replay=obs_replay)
        os.replace(tmp, tmp.with_name(f"{e}.npz"))
    return d


def generate(args, cfg):
    out = Path(args.out)
    (out / "eps").mkdir(parents=True, exist_ok=True)
    log = out / "episodes.jsonl"
    done = {json.loads(line)["episode"] for line in log.read_text().splitlines()} if log.exists() else set()
    todo = [e for e in range(args.episodes) if e not in done]
    info = {"seed": args.seed, "episodes": args.episodes, "config": cfg, "code_commit": git("rev-parse", "HEAD"),
            "c_commit": git("log", "-1", "--format=%H", "--", "dyngrasp/hybrid.py"), "replay": not args.no_replay}
    (out / "run.json").write_text(json.dumps(info, indent=1))
    t0, stop_at = time.time(), time.time() + 60 * args.max_minutes
    print(f"{len(todo)} episodes to run ({len(done)} already done), {args.workers} workers", flush=True)
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=args.tasks_per_child) as ex, log.open("a") as f:
        pending = set()
        it = iter(todo)

        def submit():
            e = next(it, None)
            if e is not None and time.time() < stop_at:
                pending.add(ex.submit(episode, cfg, args.seed, e, str(out), not args.no_replay))

        for _ in range(2 * args.workers):
            submit()
        while pending:
            fut = next(as_completed(pending))
            pending.remove(fut)
            d = fut.result()
            f.write(json.dumps(d, default=_json) + "\n")
            f.flush()
            counts[d["outcome"]] = counts.get(d["outcome"], 0) + 1
            rep = f" replay {d['replay_outcome']}" if "replay_outcome" in d else ""
            print(f"[{sum(counts.values())}] ep {d['episode']} L{d['level']} {d['outcome']}{rep} "
                  f"({d['wall_s']:.0f} s)  {counts}  {(time.time() - t0) / 60:.1f} min", flush=True)
            submit()
    print(f"done in {(time.time() - t0) / 60:.1f} min: {counts}", flush=True)


def merge(args):
    eps, runs, seen = [], [], set()
    for d in map(Path, args.runs):
        runs.append(json.loads((d / "run.json").read_text()))
        for line in (d / "episodes.jsonl").read_text().splitlines():
            m = json.loads(line)
            m["run"] = str(d)
            eps.append(m)
    keep = []
    for m in sorted((m for m in eps if m["outcome"] == "success"), key=lambda m: (m["run"], m["episode"])):
        if m["fingerprint"] not in seen:
            seen.add(m["fingerprint"])
            keep.append(m)
    obs, act, obs_replay = [], [], []
    for m in keep:
        z = np.load(Path(m["run"]) / "eps" / f"{m['episode']}.npz")
        obs.append(z["obs"])
        act.append(z["act"])
        obs_replay.append(z["obs_replay"] if "obs_replay" in z else np.full_like(z["obs"], np.nan))
    ep_len = np.array([len(a) for a in act])
    obs, act, obs_replay = (np.concatenate(x).astype(np.float32) for x in (obs, act, obs_replay))
    assert np.all(np.isfinite(obs)) and np.all(np.isfinite(act)) and np.all(np.abs(act) <= 1.0)
    rep = np.array([{"success": 1}.get(m.get("replay_outcome"), 0 if "replay_outcome" in m else -1) for m in keep])

    levels = sorted({m["level"] for m in eps})
    stats = {
        "episodes_run": len(eps),
        "outcomes": {o: sum(m["outcome"] == o for m in eps) for o in sorted({m["outcome"] for m in eps})},
        "c_success_rate": float(np.mean([m["outcome"] == "success" for m in eps])),
        "c_success_rate_per_level": {f"L{k}": float(np.mean([m["outcome"] == "success" for m in eps if m["level"] == k]))
                                     for k in levels},
        "episodes_per_level": {f"L{k}": sum(m["level"] == k for m in eps) for k in levels},
        "kept_episodes": len(keep), "duplicates_dropped": sum(m["outcome"] == "success" for m in eps) - len(keep),
        "transitions": len(act),
        "worker_hours": float(sum(m["wall_s"] + m.get("replay_wall_s", 0.0) for m in eps) / 3600),
        "replayed": int(np.sum(rep >= 0)),
        "replay_success_rate": float(np.mean(rep[rep >= 0] == 1)) if np.any(rep >= 0) else None,
        "replay_outcomes": {o: sum(m.get("replay_outcome") == o for m in keep)
                            for o in sorted({m["replay_outcome"] for m in keep if "replay_outcome" in m})},
        "track_err_max_rad": {q: float(np.quantile([m["track_err_max"] for m in keep], p))
                              for q, p in (("median", 0.5), ("p99", 0.99), ("max", 1.0))},
        "replay_obs_diff_t0_max": max((m.get("replay_obs_diff_t0", 0.0) for m in keep), default=None),
        "replay_q_diff_max_rad": {q: float(np.quantile([m["replay_q_diff_max"] for m in keep if "replay_q_diff_max" in m], p))
                                  for q, p in (("median", 0.5), ("p99", 0.99), ("max", 1.0))}
        if any("replay_q_diff_max" in m for m in keep) else None,
        "action_clipped_per_episode_mean": float(np.mean([m["action_clipped"] for m in keep])),
    }
    meta = {"runs": runs, "episodes": keep, "stats": stats}
    np.savez_compressed(args.out, obs=obs, act=act, obs_replay=obs_replay, ep_start=np.concatenate([[0], np.cumsum(ep_len)[:-1]]), ep_len=ep_len,
                        level=np.array([m["level"] for m in keep]), seed=np.array([m["seed"] for m in keep]),
                        replay_success=rep, meta=np.array(json.dumps(meta, default=_json)))
    Path(str(args.out) + ".summary.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", required=True)
    g.add_argument("--episodes", type=int, required=True)
    g.add_argument("--workers", type=int, default=os.cpu_count())
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-minutes", type=float, default=np.inf)
    g.add_argument("--tasks-per-child", type=int, default=10, help="fresh worker processes (memory)")
    g.add_argument("--no-replay", action="store_true")
    g.add_argument("--config", help="YAML overriding dyngrasp/rl/config.py DEFAULTS")
    g.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    m = sub.add_parser("merge")
    m.add_argument("runs", nargs="+")
    m.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "generate":
        generate(args, load_config(args.config, args.set))
    else:
        merge(args)


if __name__ == "__main__":
    main()
