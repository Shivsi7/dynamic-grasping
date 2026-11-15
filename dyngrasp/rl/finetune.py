"""(D) SAC fine-tuning of the behaviour-cloned policy (agent.py) on asynchronous environments (vecenv.py).

    finetune(run_dir, cfg)   # resumes from run_dir/ckpt if there is one (with the run's saved config.yaml)

The actor starts from cfg.ft.bc_checkpoint (a BC actor.npz directory), which also stays the reference of the
bc_policy_weight regulariser.  Each batch takes demo_fraction of its rows from cfg.ft.demos (transitions of (C)
demonstrations replayed through GraspEnv, scripts/make_rl_demos.py) and the rest from the online replay
buffer (RLPD).  The first warmup_steps environment steps are collected by the initial (BC) actor with the critics
learning alone.  Rows are n-step (cfg.ft.n_step) with their own discount.  The curriculum starts at
ft.start_level.  Validation (asynchronous, its own processes): the deterministic actor on the 50 validation scenes
through the harness (bc.closed_loop's episode, config env latency), at step 0 and every val_every steps; every
validated actor is kept in vals/<steps>/, the best success rate (ties: fewer collisions) in best/.

SIGUSR1 / SIGTERM: checkpoint and return without DONE (run again to resume).  max_steps: checkpoint, DONE.

Run directory: config.yaml, metrics.jsonl (one line per log_seconds), val.jsonl, vals/<steps>/, best/
(actor.npz, config.yaml, val.json: what RLController and scripts/eval_rl.py load), ckpt/ (agent.pkl,
state.json, buffer.npz).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import shutil
import signal
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax
import numpy as np
import yaml

from dyngrasp.rl import bc
from dyngrasp.rl.agent import ACTION_DIM, Agent
from dyngrasp.rl.curriculum import OUTCOMES, Curriculum
from dyngrasp.rl.observation import OBS_DIM
from dyngrasp.rl.scenes import N_LEVELS, validation_scenes
from dyngrasp.rl.vecenv import AsyncEnvs

STOP_SIGNALS = (signal.SIGUSR1, signal.SIGTERM)
FIELDS = {"obs": OBS_DIM, "act": ACTION_DIM, "rew": 0, "next_obs": OBS_DIM, "done": 0, "disc": 0}


# ---- transitions ------------------------------------------------------------------------------------------------


class NStep:
    """One environment's last n steps -> n-step rows (obs, act, sum gamma^k r_k, next_obs, done, gamma^m)."""

    def __init__(self, n: int, gamma: float):
        self.n, self.gamma = n, gamma
        self.q: deque = deque()

    def add(self, obs, act, rew, next_obs, done) -> list[tuple]:
        self.q.append((obs, act, rew, next_obs, done))
        out = []
        if done:
            while self.q:
                out.append(self._row())
                self.q.popleft()
        elif len(self.q) == self.n:
            out.append(self._row())
            self.q.popleft()
        return out

    def _row(self) -> tuple:
        ret = sum(self.gamma**k * x[2] for k, x in enumerate(self.q))
        return (self.q[0][0], self.q[0][1], ret, self.q[-1][3], float(self.q[-1][4]), self.gamma ** len(self.q))


def nstep_arrays(d: dict, n: int, gamma: float) -> dict:
    """1-step episodes (arrays obs, act, rew, next_obs, done, ep_start, ep_len) -> n-step rows."""
    if n == 1:
        return {**{k: np.asarray(d[k], np.float32) for k in FIELDS if k != "disc"},
                "disc": np.full(len(d["rew"]), gamma, np.float32)}
    rows = []
    for s, length in zip(d["ep_start"], d["ep_len"]):
        ns = NStep(n, gamma)
        for t in range(s, s + length):
            rows += ns.add(d["obs"][t], d["act"][t], float(d["rew"][t]), d["next_obs"][t], bool(d["done"][t]))
    return {k: np.array([r[j] for r in rows], np.float32) for j, k in enumerate(FIELDS)}


class Buffer:
    def __init__(self, capacity: int):
        self.cap, self.ptr, self.size = capacity, 0, 0
        self.a = {k: np.zeros((capacity, w) if w else capacity, np.float32) for k, w in FIELDS.items()}

    def add(self, rows: list[tuple]) -> None:
        for r in rows:
            for k, x in zip(FIELDS, r):
                self.a[k][self.ptr] = x
            self.ptr = (self.ptr + 1) % self.cap
            self.size = min(self.size + 1, self.cap)

    def add_arrays(self, d: dict) -> None:
        n = len(d["rew"])
        assert n <= self.cap - self.size and self.ptr == self.size
        for k in FIELDS:
            self.a[k][self.size: self.size + n] = d[k]
        self.size += n
        self.ptr = self.size % self.cap

    def sample(self, rng: np.random.Generator, shape) -> dict:
        idx = rng.integers(0, self.size, shape)
        return {k: v[idx] for k, v in self.a.items()}

    def save(self, path: Path) -> None:
        tmp = path.with_name(f".{path.name}.tmp.npz")
        np.savez(tmp, ptr=self.ptr, size=self.size, **{k: v[: self.size] for k, v in self.a.items()})
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        z = np.load(path)
        self.size, self.ptr = int(z["size"]), int(z["ptr"])
        for k in FIELDS:
            self.a[k][: self.size] = z[k]


def load_demos(path, n: int, gamma: float) -> dict:
    z = np.load(path)
    return nstep_arrays({k: z[k] for k in (*FIELDS, "ep_start", "ep_len") if k != "disc"}, n, gamma)


# ---- validation ---------------------------------------------------------------------------------------------------


def _ignore_signals() -> None:
    for s in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(s, signal.SIG_IGN)


def save_actor(params, cfg: dict, out: Path) -> Path:
    return bc.save_policy(jax.device_get(params), cfg, out)


class Validator:
    """The validation scenes in a process pool, in the background (one validation at a time)."""

    def __init__(self, cfg: dict, workers: int):
        self.specs = [s.to_dict() for s in validation_scenes()[: cfg["validation"]["n_scenes"]]]
        self.latency = cfg["env"]["latency"]
        self.ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_ignore_signals)
        self.job = None  # (steps, dir, futures)

    @property
    def busy(self) -> bool:
        return self.job is not None

    def start(self, steps: int, policy_dir: Path) -> None:
        futs = [self.ex.submit(bc._run_scene, str(policy_dir), s, self.latency) for s in self.specs]
        self.job = (steps, policy_dir, futs, time.time())

    def result(self) -> dict | None:
        """The finished validation's outcome rates (None if none finished or it failed)."""
        if self.job is None or not all(f.done() for f in self.job[2]):
            return None
        steps, d, futs, t0 = self.job
        self.job = None
        try:
            rows = [f.result() for f in futs]
        except Exception as exc:  # noqa: BLE001  (a broken pool: reported, training goes on)
            print(f"validation at {steps} failed: {exc!r}", flush=True)
            return None
        out = {o: float(np.mean([r["outcome"] == o for r in rows])) for o in OUTCOMES}
        t = [r["t_grasp"] for r in rows if r["outcome"] == "success"]
        out.update(steps=steps, n=len(rows), t_grasp_median=float(np.median(t)) if t else None,
                   wall_s=time.time() - t0, dir=str(d))
        (d / "val.json").write_text(json.dumps({**out, "rows": rows}, indent=1, default=str))
        return out

    def close(self) -> None:
        self.ex.shutdown(wait=False, cancel_futures=True)


# ---- the run ------------------------------------------------------------------------------------------------------


def finetune(run_dir, cfg: dict) -> str:
    run_dir = Path(run_dir)
    if (run_dir / "DONE").exists():
        return (run_dir / "DONE").read_text().strip()
    signals: list[str] = []
    prev = {s: signal.signal(s, lambda signum, frame: signals.append(signal.Signals(signum).name)) for s in STOP_SIGNALS}
    envs = validator = None
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = run_dir / "ckpt"
        if (ckpt / "state.json").exists():
            cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
            st = json.loads((ckpt / "state.json").read_text())
        else:
            (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            st = {"steps": 0, "episodes": 0, "starts": 0, "val_history": [], "best": None, "next_val": 0,
                  "curriculum": None}
        st["starts"] += 1
        f = cfg["ft"]
        rng = np.random.default_rng([f["seed"], st["starts"]])

        actor0 = bc.sbx_params(dict(np.load(Path(f["bc_checkpoint"]) / "actor.npz")))
        agent = Agent(cfg, actor0, seed=f["seed"])
        buf = Buffer(f["buffer_size"])
        if (ckpt / "agent.pkl").exists():
            agent.load_state_numpy(pickle.loads((ckpt / "agent.pkl").read_bytes()))
        if (ckpt / "buffer.npz").exists():
            buf.load(ckpt / "buffer.npz")
        demos = None
        if f["demos"] and f["demo_fraction"] > 0:
            d = load_demos(f["demos"], f["n_step"], cfg["gamma"])
            demos = Buffer(len(d["rew"]))
            demos.add_arrays(d)
            print(f"demonstrations: {demos.size} rows from {f['demos']}", flush=True)
        if f["target_entropy_bc_offset"] is not None:
            if "bc_entropy" not in st:
                x = (demos if demos is not None else buf).sample(rng, 4096)["obs"]
                st["bc_entropy"] = agent.entropy(x, jax.random.PRNGKey(f["seed"]))
            agent.set_target_entropy(st["bc_entropy"] + f["target_entropy_bc_offset"])
            print(f"initial actor entropy {st['bc_entropy']:.2f}: target {st['bc_entropy'] + f['target_entropy_bc_offset']:.2f}",
                  flush=True)
        cur = Curriculum(N_LEVELS, cfg["curriculum"]["window"], cfg["curriculum"]["advance_at"])
        cur.level = f["start_level"]
        if st["curriculum"]:
            cur.load_state_dict(st["curriculum"])
        print(f"start {st['starts']}: {st['steps']} steps, buffer {buf.size}, level {cur.level}, "
              f"device {jax.devices()[0]}", flush=True)

        n = f["n_envs"]
        envs = AsyncEnvs(cfg, n, f["seed"] * 1000 + st["starts"], cur.level)
        validator = Validator(cfg, f["val_workers"])
        obs = envs.reset()
        nsteps = [NStep(f["n_step"], cfg["gamma"]) for _ in range(n)]
        last_act = np.zeros((n, ACTION_DIM), np.float32)
        key = jax.random.PRNGKey(int(rng.integers(2**31)))

        def act(idx):
            nonlocal key
            key, k = jax.random.split(key)
            pad = np.zeros((n, OBS_DIM), np.float32)
            pad[: len(idx)] = obs[idx]
            a = agent.act(pad, k)[: len(idx)]
            last_act[idx] = a
            envs.send(idx, a)

        act(np.arange(n))
        B, K = f["batch_size"], f["updates_per_call"]
        n_demo = round(B * f["demo_fraction"]) if demos is not None else 0
        is_demo = np.concatenate([np.zeros(B - n_demo), np.ones(n_demo)]).astype(np.float32)
        due, metrics = 0.0, {}
        episodes = deque(maxlen=200)
        t_log = t_ckpt = t_start = time.time()
        last = (t_log, st["steps"], int(agent.state["n_updates"]))
        reason = None

        def checkpoint():
            tmp = ckpt.with_name(".ckpt_tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
            (tmp / "agent.pkl").write_bytes(pickle.dumps(agent.state_numpy()))
            saved = {**st, "curriculum": cur.state_dict()}
            if validator.busy:  # interrupted validations run again after a resume
                saved["next_val"] = min(st["next_val"], validator.job[0])
            (tmp / "state.json").write_text(json.dumps(saved, indent=1))
            ckpt.mkdir(exist_ok=True)
            for name in ("agent.pkl", "state.json"):
                os.replace(tmp / name, ckpt / name)
            shutil.rmtree(tmp, ignore_errors=True)
            buf.save(ckpt / "buffer.npz")

        while True:
            results = envs.poll(timeout=0.05)
            for i, (o_next, r, done, next_obs, ep) in results:
                buf.add(nsteps[i].add(obs[i].copy(), last_act[i].copy(), r, next_obs, done))
                obs[i] = o_next
                st["steps"] += 1
                if done:
                    st["episodes"] += 1
                    episodes.append(ep)
                    if cur.record(ep["level"], ep["outcome"]):
                        envs.set_level(cur.level)
                        print(f"curriculum: level {cur.level} at {st['steps']} steps", flush=True)
            if results:
                act(np.array([i for i, _ in results]))
            # learning: utd updates per collected step, in chunks of K
            if buf.size >= f["learning_starts"]:
                due += f["utd"] * len(results)
                for _ in range(f["max_chunks_per_poll"]):
                    if due < K:
                        break
                    on = buf.sample(rng, (K, B - n_demo))
                    if n_demo:
                        dm = demos.sample(rng, (K, n_demo))
                        on = {k: np.concatenate([on[k], dm[k]], 1) for k in on}
                    on["is_demo"] = np.broadcast_to(is_demo, (K, B))
                    metrics = agent.update(on, train_actor=st["steps"] >= f["warmup_steps"])
                    due -= K
            # validation
            if not validator.busy and st["steps"] >= st["next_val"]:
                validator.start(st["steps"], save_actor(agent.state["actor"], cfg, run_dir / "vals" / f"{st['steps']:09d}"))
                st["next_val"] = st["steps"] + f["val_every"]
            v = validator.result()
            if v is not None:
                st["val_history"].append(v)
                with (run_dir / "val.jsonl").open("a") as fh:
                    fh.write(json.dumps(v) + "\n")
                b = st["best"]
                if b is None or (v["success"], -v["collision"]) > (b["success"], -b["collision"]):
                    st["best"] = v
                    shutil.rmtree(run_dir / "best", ignore_errors=True)
                    shutil.copytree(v["dir"], run_dir / "best")
                print(f"validation at {v['steps']}: success {v['success']:.2f} collision {v['collision']:.2f} "
                      f"dropped {v['dropped']:.2f} timeout {v['timeout']:.2f} ({v['wall_s']:.0f} s)", flush=True)
            # logging, checkpoints, stopping
            now = time.time()
            if now - t_log >= f["log_seconds"]:
                t0, s0, u0 = last
                nu = int(agent.state["n_updates"])
                row = {"steps": st["steps"], "episodes": st["episodes"], "updates": nu, "level": cur.level,
                       "steps_per_s": (st["steps"] - s0) / (now - t0), "updates_per_s": (nu - u0) / (now - t0),
                       "buffer": buf.size, "hours": (now - t_start) / 3600}
                if episodes:
                    for o in OUTCOMES:
                        row[o] = float(np.mean([e["outcome"] == o for e in episodes]))
                    row["return"] = float(np.mean([e["return"] for e in episodes]))
                    row["ep_len"] = float(np.mean([e["ep_len"] for e in episodes]))
                    row["good_closes"] = float(np.mean([e["closes"]["good"] for e in episodes]))
                    for term in episodes[0]["ep_terms"]:
                        row[f"term_{term}"] = float(np.mean([e["ep_terms"][term] for e in episodes]))
                row.update(metrics)
                with (run_dir / "metrics.jsonl").open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
                last, t_log = (now, st["steps"], nu), now
            if st["steps"] >= f["max_steps"]:
                reason = f"max_steps ({f['max_steps']}) reached"
            if reason or signals or now - t_ckpt >= 60 * f["checkpoint_minutes"]:
                if reason or signals:
                    envs.drain()  # (their transitions are dropped: the next start begins new episodes)
                checkpoint()
                t_ckpt = time.time()
                print(f"checkpoint at {st['steps']} steps", flush=True)
            if reason:
                (run_dir / "DONE").write_text(reason + "\n")
                return reason
            if signals:
                return f"interrupted by {signals[0]} at {st['steps']} steps (checkpointed; run again to resume)"
    finally:
        for s, h in prev.items():
            signal.signal(s, h)
        if validator is not None:
            validator.close()
        if envs is not None:
            envs.close()
