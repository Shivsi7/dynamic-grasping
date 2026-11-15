"""RL: PPO's episode workers (scripts/env_workers.py --ppo; the learner is ppo.py).  No JAX here: the policy runs in
numpy (NumpyPolicy), so the many worker processes stay small.

serve(conn): connected to the learner, which sends ("init", (config, seed, run directory, excluded scene seeds)), a
worker runs whole GraspEnv episodes at the final level with the latest policy the learner exported
(run_dir/policy/latest.json, read at each episode's start): u ~ N(mean(s), sigma), action to_action(u); it sends each episode
("episode", obs, u, log mu(u), rewards, outcome, policy version).  With probability latest.json's label_fraction it then
runs a DAgger takeover (takeover_labels) and sends ("labels", ...).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ACTION_DIM = 8
GRIP = 7  # the gripper's action dimension: its Gaussian is in action units (ppo.dist_mean)
LOG_2PI = float(np.log(2.0 * np.pi))
CHECK_SEED, CHECK_N = 777, 500  # scripts/bc_bigeval.py's L5 scenes: never trained on


def check_seeds() -> list[int]:
    """The scene seeds of scripts/bc_bigeval.py's L5 check scenes (sample_training_scene(5, rng([777, i])))."""
    from dyngrasp.rl.scenes import sample_training_scene

    return [sample_training_scene(5, np.random.default_rng([CHECK_SEED, i])).seed for i in range(CHECK_N)]


class NumpyPolicy:
    """The worker's policy: mu(s) in float64 numpy, u = mu + sigma eps, log mu(u)."""

    def __init__(self, arrays: dict):
        n = sum(k.startswith("W") and k[1:].isdigit() for k in arrays)
        self.hidden = [(np.asarray(arrays[f"W{i}"], np.float64), np.asarray(arrays[f"b{i}"], np.float64)) for i in range(n)]
        self.mean = (np.asarray(arrays["Wm"], np.float64), np.asarray(arrays["bm"], np.float64))
        self.log_std = np.asarray(arrays["bls"], np.float64)

    def mu(self, obs) -> np.ndarray:
        x = np.asarray(obs, np.float64)
        for W, b in self.hidden:
            x = np.maximum(x @ W + b, 0.0)
        return x @ self.mean[0] + self.mean[1]

    def sample(self, obs, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        """(u, log mu(u)): u ~ N(mean, sigma), mean = mu(s) for the joints, tanh(mu(s)) for the gripper (ppo.dist_mean)."""
        mean = self.mu(obs)
        mean[GRIP:] = np.tanh(mean[GRIP:])
        eps = rng.standard_normal(ACTION_DIM)
        u = mean + np.exp(self.log_std) * eps
        return u, float(np.sum(-0.5 * eps**2 - self.log_std - 0.5 * LOG_2PI))


def to_action(u: np.ndarray) -> np.ndarray:
    """The action of a sample u: tanh(u) for the joints, clip(u, -1, 1) for the gripper."""
    return np.concatenate([np.tanh(u[:GRIP]), np.clip(u[GRIP:], -1.0, 1.0)])


def _t_fail(env, outcome: str) -> float:
    """When the episode went wrong (first collision), else its grasp, else its end: switches are drawn before it."""
    m, t_end = env.h.monitor, env.n_steps * env.cfg["env"]["dt"]
    t = m.collisions[0].t if (outcome == "collision" and m.collisions) else t_end
    return min(t, m.t_grasp) if m.t_grasp is not None else t


def takeover_labels(spec, executed: np.ndarray, t_fail: float, cfg: dict, rng: np.random.Generator) -> dict:
    """DAgger: the episode's executed actions until a switch drawn uniformly in [0, t_fail - 0.2] (policy ticks), then
    (C) (demos.run_takeover); (C)'s actions on the steps from the switch on if it succeeded."""
    from dyngrasp.rl.demos import run_takeover

    t_sw = round(float(rng.uniform(0.0, max(0.1, t_fail - 0.2))), 1)
    k = iter(range(10**6))
    hold = np.concatenate([np.zeros(7), executed[-1, 7:]])

    def replay(obs):
        i = next(k)
        return executed[i] if i < len(executed) else hold

    t = time.perf_counter()
    try:
        d = run_takeover(spec, 5, cfg, replay, t_sw, seed=int(rng.integers(2**31)))
    except Exception as exc:  # noqa: BLE001  (recorded)
        d = {"outcome": "error", "error": repr(exc)[:300]}
    out = {"scene": spec.name, "t_switch": t_sw, "t_fail": t_fail, "outcome": d["outcome"], "wall_s": time.perf_counter() - t,
           "t_grasp": d.get("c_result", {}).get("t_grasp"), "n_policy_steps": d.get("n_policy_steps")}
    if "actions" in d:
        out.update(obs=d["obs"], act=d["actions"], n_labels=len(d["actions"]))
    return out


def run_episode(env, obs, pol: NumpyPolicy, rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    """One GraspEnv episode from obs (just reset) with u ~ NumpyPolicy, action to_action(u): the episode (obs, u, logp,
    rewards, outcome, ...; t_fail: see _t_fail) and its executed actions."""
    ob, us, lp, rw, ex = [], [], [], [], []
    while True:
        u, logp = pol.sample(obs, rng)
        a = to_action(u)
        ob.append(obs)
        us.append(u)
        lp.append(logp)
        ex.append(a)
        obs, r, term, trunc, info = env.step(a)
        rw.append(r)
        if term or trunc:
            break
    ep = {"obs": np.array(ob, np.float32), "u": np.array(us, np.float32), "logp": np.array(lp, np.float64),
          "rew": np.array(rw, np.float64), "outcome": info["outcome"], "scene": env.spec.name, "closes": info["closes"],
          "collision": info["collision"], "ep_terms": info["ep_terms"], "t_fail": _t_fail(env, info["outcome"])}
    return ep, np.array(ex)


def serve(conn, max_episodes: int | None = None) -> bool:
    """A worker on one connection to the learner: episodes (and takeovers) until "close", the connection drops or (True)
    max_episodes episodes."""
    from dyngrasp.rl.env import GraspEnv

    cmd, (cfg, seed, run_dir, excluded) = conn.recv()
    assert cmd == "init"
    excluded = set(excluded)
    pol_dir = Path(run_dir) / "policy"
    env = GraspEnv(cfg, seed)
    env.set_level(5)
    rng = np.random.default_rng(seed)
    pol, version, episodes = None, -1, 0
    try:
        while True:
            if conn.poll(0) and conn.recv()[0] == "close":
                return False
            try:
                ptr = json.loads((pol_dir / "latest.json").read_text())
                if ptr["version"] != version:
                    pol, version = NumpyPolicy(dict(np.load(pol_dir / ptr["file"]))), ptr["version"]
            except Exception:  # noqa: BLE001  (not written yet, or replaced meanwhile: the policy it has)
                if pol is None:
                    time.sleep(1.0)
                    continue
            obs, _ = env.reset()
            while env.spec.seed in excluded:
                obs, _ = env.reset()
            try:
                ep, executed = run_episode(env, obs, pol, rng)
            except Exception as exc:  # noqa: BLE001  (a simulation error: the episode is dropped)
                print(f"GraspEnv.step failed ({exc!r}): episode dropped", flush=True)
                continue
            ep["version"] = version
            conn.send(("episode", ep))
            episodes += 1
            if rng.random() < ptr.get("label_fraction", 0.0):
                lab = takeover_labels(env.spec, executed, ep["t_fail"], cfg, rng)
                lab.update(version=version, policy_outcome=ep["outcome"])
                conn.send(("labels", lab))
                episodes += 1  # (a takeover's simulation leaks memory too)
            if max_episodes and episodes >= max_episodes:
                return True
    except (EOFError, OSError):
        return False
    finally:
        env.close()
