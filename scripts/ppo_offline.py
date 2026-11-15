#!/usr/bin/env python
"""RL: data for a PPO run made before it, from a fixed policy (the BC policy), as PPO's workers make it
(rollout.run_episode, rollout.takeover_labels): its noisy rollouts on fresh L5 training scenes with the reward (V
pretraining: ppo.value_init) and DAgger takeovers of them by (C) (ppo.dagger_init); never a scripts/bc_bigeval.py check
scene.

    python scripts/ppo_offline.py run --policy DIR --out OUT/part0 --episodes 2000 --seed 0 [--switches 1] [--workers N] [--set KEY=VALUE ...]
    python scripts/ppo_offline.py merge OUT/part* --out OUT/merged    (-> OUT/merged_rollouts.npz, OUT/merged_labels.npz)

run: scene e = sample_training_scene(5, rng([seed, e])); u ~ N(mu(s), exp(ppo.init_log_std)), action tanh(u); then
--switches takeovers (switch drawn uniformly before the rollout's failure).  OUT/episodes.jsonl: a row per scene;
OUT/eps/<e>.npz: obs, u, logp, rew and lab<j>_obs, lab<j>_act for each takeover (C) completed.  Rerunning skips the
scenes done.
merge: rollouts (obs, rew, ep_len, success) and labels (obs, act) of the given runs, each in one .npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("HEADLESS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dyngrasp.rl.config import load_config


def task(cfg: dict, arrays: dict, seed: int, e: int, out: str, switches: int, excluded: frozenset) -> dict:
    from dyngrasp.rl.env import GraspEnv
    from dyngrasp.rl.rollout import NumpyPolicy, run_episode, takeover_labels
    from dyngrasp.rl.scenes import sample_training_scene

    t = time.perf_counter()
    spec = sample_training_scene(5, np.random.default_rng([seed, e]))
    row = {"e": e, "scene": spec.name, "seed": spec.seed}
    if spec.seed in excluded:
        return {**row, "outcome": "excluded", "wall_s": 0.0}
    rng = np.random.default_rng([seed, e, 1])
    env = GraspEnv(cfg, 0)
    env.set_level(5)
    try:
        obs, _ = env.reset(options={"spec": spec, "level": 5})
        ep, executed = run_episode(env, obs, NumpyPolicy(arrays), rng)
    except Exception as exc:  # noqa: BLE001  (recorded)
        return {**row, "outcome": "error", "error": repr(exc)[:300], "wall_s": time.perf_counter() - t}
    finally:
        env.close()
    z = {k: ep[k] for k in ("obs", "u", "logp", "rew")}
    labs = []
    for j in range(switches):
        lab = takeover_labels(spec, executed, ep["t_fail"], cfg, rng)
        if "obs" in lab:
            z[f"lab{j}_obs"], z[f"lab{j}_act"] = np.asarray(lab.pop("obs"), np.float32), np.asarray(lab.pop("act"), np.float32)
        labs.append(lab)
    tmp = Path(out) / "eps" / f".{e}.npz"
    np.savez_compressed(tmp, **z)
    os.replace(tmp, tmp.with_name(f"{e}.npz"))
    return {**row, "outcome": ep["outcome"], "return": float(ep["rew"].sum()), "ep_len": len(ep["rew"]),
            "collision": ep["collision"], "t_fail": ep["t_fail"], "ep_terms": ep["ep_terms"], "takeovers": labs,
            "wall_s": time.perf_counter() - t}


def run(args) -> None:
    from dyngrasp.rl.rollout import check_seeds

    cfg = load_config(None, args.set)
    cfg["demos"]["max_wall_s"] = cfg["ppo"]["label_max_wall_s"]
    arrays = dict(np.load(Path(args.policy) / "actor.npz"))
    arrays["bls"] = np.asarray(cfg["ppo"]["init_log_std"], np.float64)
    out = Path(args.out)
    (out / "eps").mkdir(parents=True, exist_ok=True)
    log = out / "episodes.jsonl"
    done = {json.loads(line)["e"] for line in log.read_text().splitlines()} if log.exists() else set()
    todo = iter([e for e in range(args.episodes) if e not in done])
    (out / "run.json").write_text(json.dumps({"policy": str(Path(args.policy).resolve()), "seed": args.seed,
                                              "switches": args.switches, "config": cfg}, indent=1))
    excluded = frozenset(check_seeds())
    t0, n, labels, counts = time.time(), 0, 0, {}
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=args.tasks_per_child) as ex, log.open("a") as f:
        pending = set()

        def submit():
            e = next(todo, None)
            if e is not None:
                pending.add(ex.submit(task, cfg, arrays, args.seed, e, str(out), args.switches, excluded))

        for _ in range(2 * args.workers):
            submit()
        while pending:
            fut = next(as_completed(pending))
            pending.remove(fut)
            d = fut.result()
            f.write(json.dumps(d, default=float) + "\n")
            f.flush()
            n += 1
            counts[d["outcome"]] = counts.get(d["outcome"], 0) + 1
            labels += sum(t.get("n_labels", 0) for t in d.get("takeovers", []))
            if n % 50 == 0:
                print(f"[{n}] {counts}, {labels} labels  {(time.time() - t0) / 60:.1f} min", flush=True)
            submit()
    print(f"done in {(time.time() - t0) / 60:.1f} min: {counts}", flush=True)


def merge(args) -> None:
    rows = []
    for d in map(Path, args.runs):
        rows += [(d, json.loads(line)) for line in (d / "episodes.jsonl").read_text().splitlines()]
    obs, rew, ep_len, success, lobs, lact = [], [], [], [], [], []
    for d, m in rows:
        if m["outcome"] in ("error", "excluded"):
            continue
        z = np.load(d / "eps" / f"{m['e']}.npz")
        obs.append(z["obs"])
        rew.append(z["rew"])
        ep_len.append(len(z["rew"]))
        success.append(m["outcome"] == "success")
        for j in range(len(m["takeovers"])):
            if f"lab{j}_obs" in z:
                lobs.append(z[f"lab{j}_obs"])
                lact.append(z[f"lab{j}_act"])
    np.savez(f"{args.out}_rollouts.npz", obs=np.concatenate(obs).astype(np.float32), rew=np.concatenate(rew),
             ep_len=np.array(ep_len), success=np.array(success))
    np.savez(f"{args.out}_labels.npz", obs=np.concatenate(lobs).astype(np.float32), act=np.concatenate(lact).astype(np.float32))
    tk = [t for _, m in rows for t in m.get("takeovers", [])]
    print(json.dumps({"scenes": len(rows), "rollouts": len(ep_len), "steps": int(np.sum(ep_len)),
                      "success": float(np.mean(success)),
                      "outcomes": {o: sum(m["outcome"] == o for _, m in rows) for o in sorted({m["outcome"] for _, m in rows})},
                      "takeovers": len(tk), "takeover_success": float(np.mean([t["outcome"] == "success" for t in tk])) if tk else None,
                      "label_episodes": len(lact), "labels": int(sum(len(a) for a in lact))}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--policy", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--episodes", type=int, required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--switches", type=int, default=1)
    r.add_argument("--workers", type=int, default=os.cpu_count())
    r.add_argument("--tasks-per-child", type=int, default=10)
    r.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    m = sub.add_parser("merge")
    m.add_argument("runs", nargs="+")
    m.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else merge(args)


if __name__ == "__main__":
    main()
