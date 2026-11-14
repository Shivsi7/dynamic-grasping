"""(D) SAC training (sbx): vectorized environments, curriculum, validation, logging, checkpoints, resume.

    train(run_dir, cfg)     # resumes from run_dir/checkpoints/latest if there is one (then the saved
                            # run_dir/config.yaml is used, not `cfg`); stops at convergence on the final
                            # level (no validation improvement in `patience` validations) or at max_steps,
                            # and writes run_dir/DONE.  SIGUSR1 (Slurm, before the time limit) or SIGTERM
                            # (preemption): checkpoint at once and return without DONE; running it again
                            # resumes.  Environment worker processes ignore both signals.

Run directory:
  config.yaml                 the resolved configuration
  tb/                         TensorBoard events (and the same numbers on stdout)
  checkpoints/step_<n>/       model.zip, actor.npz, config.yaml, state.json, rng.pkl, then replay_buffer.pkl
  checkpoints/latest          name of the newest checkpoint (resuming from one whose replay buffer was not
                              written, e.g. killed while saving it, takes the buffer of the newest that has one)
  best/                       the best final-level validation: model.zip, actor.npz, config.yaml, val.json
A checkpoint directory (or best/) is what RLController and scripts/eval_rl.py load.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import shutil
import signal
import time
from collections import deque
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml
from sbx import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from dyngrasp.rl.controller import NumpyActor
from dyngrasp.rl.curriculum import OUTCOMES, Curriculum
from dyngrasp.rl.env import TERMS, make_env
from dyngrasp.rl.scenes import N_LEVELS, validation_scenes

BUFFER = "replay_buffer.pkl"
STOP_SIGNALS = (signal.SIGUSR1, signal.SIGTERM)  # Slurm: time limit near, preemption


def make_vec_env(cfg: dict, n: int, seed: int, scenes=None, subproc: bool = True) -> VecEnv:
    """n environments; with `scenes`, env i runs scenes[i::n] in order (validation)."""
    fns = [make_env(cfg, seed * 10_000 + i, None if scenes is None else list(scenes[i::n])) for i in range(n)]
    return SubprocVecEnv(fns) if subproc else DummyVecEnv(fns)


def new_model(cfg: dict, venv: VecEnv) -> SAC:
    s = cfg["sac"]
    return SAC(
        "MlpPolicy", venv, learning_rate=s["learning_rate"], buffer_size=s["buffer_size"],
        learning_starts=s["learning_starts"], batch_size=s["batch_size"], tau=s["tau"], gamma=cfg["gamma"],
        train_freq=s["train_freq"], ent_coef="auto",
        gradient_steps=max(1, round(s["updates_per_env_step"] * venv.num_envs * s["train_freq"])),
        policy_kwargs={"net_arch": {"pi": s["net_arch"], "qf": s["net_arch"]}, "n_critics": 2,
                       "layer_norm": s["layer_norm"]},
        seed=cfg["train"]["seed"], verbose=0,
    )


# ---- checkpoints ---------------------------------------------------------------------------------------


def latest_checkpoint(run_dir: Path) -> Path | None:
    f = Path(run_dir) / "checkpoints" / "latest"
    return f.parent / f.read_text().strip() if f.exists() else None


def save_actor(model: SAC, cfg: dict, out: Path) -> None:
    """What RLController needs: actor weights (numpy) and the configuration."""
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "actor.npz", **NumpyActor.arrays_from_sbx(model.policy.actor_state.params))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def save_checkpoint(model: SAC, cfg: dict, run_dir: Path, state: dict) -> Path:
    """Everything to resume.  The model and state first (a temporary directory renamed into place, then
    `latest`), the replay buffer last (the slow part); older checkpoints are pruned only after that."""
    root = Path(run_dir) / "checkpoints"
    final = root / f"step_{model.num_timesteps:010d}"
    tmp = root / f".tmp_{final.name}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save(tmp / "model.zip")
    save_actor(model, cfg, tmp)
    (tmp / "state.json").write_text(json.dumps(state, indent=1))
    rng = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "jax_key": np.asarray(model.key),
        "action_space": model.action_space.np_random.bit_generator.state,
        "envs": model.get_env().env_method("get_rng_state"),
    }
    (tmp / "rng.pkl").write_bytes(pickle.dumps(rng))
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    (root / "latest.tmp").write_text(final.name)
    os.replace(root / "latest.tmp", root / "latest")
    model.save_replay_buffer(final / f".{BUFFER}.tmp")
    os.replace(final / f".{BUFFER}.tmp", final / BUFFER)
    old = sorted(p for p in root.glob("step_*") if p.is_dir())[: -cfg["train"]["keep_checkpoints"]]
    for p in old:
        shutil.rmtree(p)
    return final


def load_checkpoint(path: Path, venv: VecEnv) -> tuple[SAC, dict]:
    """The model (with its replay buffer and every random state) and the training state saved at `path`."""
    model = SAC.load(path / "model.zip", env=venv)
    with_buffer = [p for p in sorted(path.parent.glob("step_*"), reverse=True) if p <= path and (p / BUFFER).exists()]
    if with_buffer:
        model.load_replay_buffer(with_buffer[0] / BUFFER)
        if with_buffer[0] != path:
            print(f"{path.name} has no replay buffer (killed while saving it): using {with_buffer[0].name}'s",
                  flush=True)
    else:
        print(f"no saved replay buffer: {path.name} resumes with an empty one", flush=True)
    rng = pickle.loads((path / "rng.pkl").read_bytes())
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    model.key = jnp.asarray(rng["jax_key"])
    model.action_space.np_random.bit_generator.state = rng["action_space"]
    venv._reset_seeds()  # SAC.load re-seeded the envs for their next reset; the saved states win
    for i, s in enumerate(rng["envs"]):
        venv.env_method("set_rng_state", s, indices=[i])
    model._last_obs = None  # new episodes (the simulations themselves are not saved)
    return model, json.loads((path / "state.json").read_text())


# ---- validation ----------------------------------------------------------------------------------------


def validate(model: SAC, val_env: VecEnv, stop=lambda: False) -> dict | None:
    """Deterministic policy, each validation env running its own scene list once (None if `stop()`)."""
    quota = [len(s) for s in val_env.get_attr("scenes")]
    val_env.env_method("restart_list")
    obs = val_env.reset()
    done, rows = [0] * val_env.num_envs, []
    while any(d < q for d, q in zip(done, quota)):
        if stop():
            return None
        actions, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = val_env.step(actions)
        for i, (d, info) in enumerate(zip(dones, infos)):
            if d and done[i] < quota[i]:
                done[i] += 1
                rows.append(info)
    out = {o: float(np.mean([r["outcome"] == o for r in rows])) for o in OUTCOMES}
    out["return"] = float(np.mean([r["episode"]["r"] for r in rows]))
    out["n"] = len(rows)
    return out


# ---- the training loop's callback ------------------------------------------------------------------------


class TrainingCallback(BaseCallback):
    """Curriculum bookkeeping, TensorBoard metrics, periodic checkpoints and validation, the stopping rule."""

    def __init__(self, cfg: dict, run_dir: Path, curriculum: Curriculum, val_env: VecEnv, state: dict):
        super().__init__()
        self.cfg, self.run_dir, self.cur, self.val_env = cfg, Path(run_dir), curriculum, val_env
        v = state.get("validation", {})
        self.val_history: list = v.get("history", [])
        self.best = v.get("best", -1.0)
        self.no_improve = v.get("no_improve", 0)
        self.next_val = v.get("next_val", cfg["validation"]["every_steps"])
        self.stop_reason: str | None = None
        self.signals: list[str] = []  # received (train() fills it): checkpoint and stop, not finished
        self.episodes = deque(maxlen=100)

    def _on_training_start(self) -> None:
        now = time.time()
        self.t_log = self.t_ckpt = now
        self.last = (now, self.num_timesteps, self.model._n_updates)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "outcome" in info:
                self.episodes.append(info)
                if self.cur.record(info["level"], info["outcome"]):
                    self.training_env.env_method("set_level", self.cur.level)
                    print(f"curriculum: level {self.cur.level} at {self.num_timesteps} steps", flush=True)
        now = time.time()
        if now - self.t_log >= self.cfg["train"]["log_seconds"]:
            self.log_metrics(now)
        if now - self.t_ckpt >= 60.0 * self.cfg["train"]["checkpoint_minutes"]:
            self.checkpoint()
        if self.num_timesteps >= self.next_val:
            self.run_validation()
        return self.stop_reason is None and not self.signals

    def log_metrics(self, now: float) -> None:
        rec = self.logger.record
        t0, n0, u0 = self.last
        self.last = (now, self.num_timesteps, self.model._n_updates)
        rec("time/env_steps_per_s", (self.num_timesteps - n0) / (now - t0))
        rec("time/updates_per_s", (self.model._n_updates - u0) / (now - t0))
        rec("curriculum/level", self.cur.level)
        for k in range(N_LEVELS):
            for o, x in self.cur.rates(k).items():
                rec(f"level_{k}/{o}", x)
        if self.episodes:
            for term in TERMS:
                rec(f"episode/{term}", np.mean([e["ep_terms"][term] for e in self.episodes]))
            rec("episode/length", np.mean([e["ep_len"] for e in self.episodes]))
            rec("episode/good_closes", np.mean([e["closes"]["good"] for e in self.episodes]))
            rec("episode/bad_closes", np.mean([e["closes"]["bad"] for e in self.episodes]))
        rb = self.model.replay_buffer
        if rb.size() >= 256:
            data = rb.sample(256)
            qf = self.model.policy.qf_state
            q = np.asarray(qf.apply_fn(qf.params, data.observations.numpy(), data.actions.numpy()))
            rec("train/q_mean", float(q.min(axis=0).mean()))
            rec("train/q_max", float(q.min(axis=0).max()))
        self.model.dump_logs()
        self.t_log = now

    def training_state(self) -> dict:
        return {
            "curriculum": self.cur.state_dict(),
            "num_timesteps": int(self.num_timesteps),
            "n_updates": int(self.model._n_updates),
            "validation": {"history": self.val_history, "best": self.best, "no_improve": self.no_improve,
                           "next_val": self.next_val},
            "stop_reason": self.stop_reason,
        }

    def checkpoint(self) -> Path:
        path = save_checkpoint(self.model, self.cfg, self.run_dir, self.training_state())
        self.t_ckpt = time.time()
        print(f"checkpoint {path.name}", flush=True)
        return path

    def run_validation(self) -> None:
        t = time.time()
        r = validate(self.model, self.val_env, stop=lambda: bool(self.signals))
        if r is None:
            return  # interrupted: validates again after the resume
        r.update(steps=int(self.num_timesteps), level=self.cur.level, wall_s=time.time() - t)
        self.val_history.append(r)
        self.next_val = self.num_timesteps + self.cfg["validation"]["every_steps"]
        for k in (*OUTCOMES, "return"):
            self.logger.record(f"val/{k}", r[k])
        if self.cur.final:
            if r["success"] > self.best:
                self.best, self.no_improve = r["success"], 0
                best = self.run_dir / "best"
                save_actor(self.model, self.cfg, best)
                self.model.save(best / "model.zip")
                (best / "val.json").write_text(json.dumps(r, indent=1))
            else:
                self.no_improve += 1
                if self.no_improve >= self.cfg["validation"]["patience"]:
                    self.stop_reason = f"converged: no validation improvement in {self.no_improve} validations"
        print(f"validation at {self.num_timesteps}: {r}", flush=True)


# ---- entry point -------------------------------------------------------------------------------------------


def train(run_dir: str | Path, cfg: dict, subproc: bool = True) -> str:
    """Train (or resume) in run_dir; returns why it stopped."""
    run_dir = Path(run_dir)
    if (run_dir / "DONE").exists():
        return (run_dir / "DONE").read_text().strip()
    signals: list[str] = []
    prev = {s: signal.signal(s, lambda signum, frame: signals.append(signal.Signals(signum).name)) for s in STOP_SIGNALS}
    venv = val_env = None
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = latest_checkpoint(run_dir)
        if ckpt is not None:
            given, cfg = cfg, yaml.safe_load((run_dir / "config.yaml").read_text())
            changed = sorted(k for k, v in _flat(given).items() if _flat(cfg).get(k) != v)
            if changed:
                print(f"resuming: using the run's saved config.yaml; ignored given values for {changed}", flush=True)
        else:
            (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        tc, vc = cfg["train"], cfg["validation"]
        venv = make_vec_env(cfg, tc["n_envs"], tc["seed"], subproc=subproc)
        val_env = make_vec_env(cfg, vc["n_envs"], tc["seed"], validation_scenes()[: vc["n_scenes"]], subproc=subproc)
        val_env.env_method("set_level", N_LEVELS - 1)  # the final level's grasp tolerances
        cur = Curriculum(N_LEVELS, cfg["curriculum"]["window"], cfg["curriculum"]["advance_at"])
        if ckpt is not None:
            model, state = load_checkpoint(ckpt, venv)
            cur.load_state_dict(state["curriculum"])
            print(f"resumed from {ckpt} at {model.num_timesteps} steps, level {cur.level}", flush=True)
        else:
            model, state = new_model(cfg, venv), {}
        model.set_logger(configure(str(run_dir / "tb"), ["stdout", "tensorboard"]))
        venv.env_method("set_level", cur.level)
        cb = TrainingCallback(cfg, run_dir, cur, val_env, state)
        cb.signals = signals
        model.learn(max(0, tc["max_steps"] - model.num_timesteps), callback=cb, log_interval=None,
                    reset_num_timesteps=False)
        if signals and cb.stop_reason is None:
            cb.checkpoint()
            return f"interrupted by {signals[0]} at {model.num_timesteps} steps (checkpointed; run again to resume)"
        reason = cb.stop_reason or f"max_steps ({tc['max_steps']}) reached"
        cb.stop_reason = reason
        cb.checkpoint()
        (run_dir / "DONE").write_text(reason + "\n")
        return reason
    finally:
        for s, handler in prev.items():
            signal.signal(s, handler)
        for e in (venv, val_env):
            if e is not None:
                e.close()


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out
