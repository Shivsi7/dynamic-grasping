"""(D) training plumbing with tiny networks and in-process environments: checkpoints, resume, validation."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml
from stable_baselines3.common.logger import configure

from dyngrasp import scenario as S
from dyngrasp.rl import train as T
from dyngrasp.rl.config import load_config
from dyngrasp.rl.controller import NumpyActor
from dyngrasp.rl.curriculum import Curriculum
from dyngrasp.rl.scenes import N_LEVELS

TINY = ["train.n_envs=2", "sac.buffer_size=3000", "sac.learning_starts=50", "sac.net_arch=[32,32]",
        "train.log_seconds=1000", "train.checkpoint_minutes=1000", "validation.every_steps=1000000000",
        "validation.n_envs=1", "validation.n_scenes=1"]


def same_tree(a, b) -> bool:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def test_checkpoint_round_trip(tmp_path):
    cfg = load_config(None, TINY)
    venv = T.make_vec_env(cfg, 2, 0, subproc=False)
    model = T.new_model(cfg, venv)
    model.learn(120)
    assert model._n_updates > 0
    state = {"curriculum": Curriculum(N_LEVELS).state_dict(), "anything": [1, 2]}
    path = T.save_checkpoint(model, cfg, tmp_path, state)
    np_state, env_states = np.random.get_state(), venv.env_method("get_rng_state")
    assert T.latest_checkpoint(tmp_path) == path and path.name == "step_0000000120"
    assert {p.name for p in path.iterdir()} == {"model.zip", "replay_buffer.pkl", "actor.npz", "config.yaml",
                                                "state.json", "rng.pkl"}

    venv2 = T.make_vec_env(cfg, 2, 0, subproc=False)
    model2, state2 = T.load_checkpoint(path, venv2)
    assert state2 == state and model2.num_timesteps == 120 and model2._n_updates == model._n_updates
    p1, p2 = model.policy, model2.policy
    assert same_tree(p1.actor_state.params, p2.actor_state.params)
    assert same_tree(p1.actor_state.opt_state, p2.actor_state.opt_state)
    assert same_tree(p1.qf_state.params, p2.qf_state.params)
    assert same_tree(p1.qf_state.target_params, p2.qf_state.target_params)
    assert same_tree(p1.qf_state.opt_state, p2.qf_state.opt_state)
    assert same_tree(model.ent_coef_state.params, model2.ent_coef_state.params)
    assert np.array_equal(np.asarray(model.key), np.asarray(model2.key))
    b1, b2 = model.replay_buffer, model2.replay_buffer
    assert (b1.pos, b1.full) == (b2.pos, b2.full) == (60, False)
    for k in ("observations", "next_observations", "actions", "rewards", "dones"):
        assert np.array_equal(getattr(b1, k), getattr(b2, k)), k
    restored = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(np_state[1:3], restored[1:3]))
    assert venv2.env_method("get_rng_state") == env_states
    assert model2._last_obs is None  # fresh episodes on resume
    obs = np.random.default_rng(0).standard_normal((8, venv.observation_space.shape[0])).astype(np.float32)
    actor = NumpyActor(dict(np.load(path / "actor.npz")))
    np.testing.assert_allclose([actor(o) for o in obs], model2.predict(obs, deterministic=True)[0], atol=1e-5)


def test_a_checkpoint_killed_while_saving_its_buffer_resumes_with_the_previous_buffer(tmp_path, capsys):
    cfg = load_config(None, TINY)
    venv = T.make_vec_env(cfg, 2, 0, subproc=False)
    model = T.new_model(cfg, venv)
    model.learn(80)
    first = T.save_checkpoint(model, cfg, tmp_path, {"n": 1})
    model.learn(40, reset_num_timesteps=False)
    second = T.save_checkpoint(model, cfg, tmp_path, {"n": 2})
    (second / T.BUFFER).unlink()  # as if killed while writing it
    assert T.latest_checkpoint(tmp_path) == second and first.name == "step_0000000080"
    model2, state = T.load_checkpoint(second, T.make_vec_env(cfg, 2, 0, subproc=False))
    assert state == {"n": 2} and model2.num_timesteps == 120
    assert model2.replay_buffer.pos == 40 and "using step_0000000080's" in capsys.readouterr().out


def test_train_finishes_and_resumes(tmp_path):
    cfg = load_config(None, TINY + ["train.max_steps=100"])
    reason = T.train(tmp_path, cfg, subproc=False)
    assert reason.startswith("max_steps") and (tmp_path / "DONE").exists()
    ck = T.latest_checkpoint(tmp_path)
    assert ck.name == "step_0000000100" and (tmp_path / "tb").is_dir()
    assert T.train(tmp_path, cfg, subproc=False) == reason  # a finished run does nothing
    # what a requeued job sees after a kill: no DONE; the saved config (here with a higher cap) is used
    (tmp_path / "DONE").unlink()
    c = yaml.safe_load((tmp_path / "config.yaml").read_text())
    c["train"]["max_steps"] = 200
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(c))
    st = json.loads((ck / "state.json").read_text())
    n_updates = st["n_updates"]
    st["curriculum"]["level"] = 2
    (ck / "state.json").write_text(json.dumps(st))
    T.train(tmp_path, cfg, subproc=False)
    ck2 = T.latest_checkpoint(tmp_path)
    st2 = json.loads((ck2 / "state.json").read_text())
    assert ck2.name == "step_0000000200" and st2["num_timesteps"] == 200 and st2["n_updates"] > n_updates
    assert st2["curriculum"]["level"] == 2
    assert sorted(p.name for p in (tmp_path / "checkpoints").glob("step_*")) == [ck.name, ck2.name]


def train_until_sigusr1(run_dir, cfg, after_s=8.0) -> str:
    timer = threading.Timer(after_s, os.kill, (os.getpid(), signal.SIGUSR1))  # as Slurm before the time limit
    timer.start()
    try:
        return T.train(run_dir, cfg, subproc=False)
    finally:
        timer.cancel()


def test_sigusr1_checkpoints_and_stops_unfinished(tmp_path, capsys):
    cfg = load_config(None, TINY + ["train.max_steps=1000000"])
    reason = train_until_sigusr1(tmp_path, cfg)
    assert reason.startswith("interrupted by SIGUSR1") and not (tmp_path / "DONE").exists()
    st = json.loads((T.latest_checkpoint(tmp_path) / "state.json").read_text())
    assert st["stop_reason"] is None and 0 < st["num_timesteps"] < 1000000
    assert signal.getsignal(signal.SIGUSR1) is signal.SIG_DFL and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    # resuming uses the saved configuration and says which given values it ignores
    capsys.readouterr()
    cfg2 = load_config(None, TINY + ["train.max_steps=500", "sac.batch_size=64"])
    assert train_until_sigusr1(tmp_path, cfg2).startswith("interrupted by SIGUSR1")
    out = capsys.readouterr().out
    assert "ignored given values for ['sac.batch_size', 'train.max_steps']" in out and "resumed from" in out
    st2 = json.loads((T.latest_checkpoint(tmp_path) / "state.json").read_text())
    assert st2["num_timesteps"] > st["num_timesteps"]


def test_sigterm_to_every_process_checkpoints_and_the_next_start_resumes(tmp_path):
    """Slurm preemption: SIGTERM to the learner and its environment workers (a process group)."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_rl.py"
    sets = ["train.n_envs=2", "sac.buffer_size=3000", "sac.learning_starts=50", "sac.net_arch=[32,32]",
            "validation.n_envs=1", "validation.n_scenes=1", "train.max_steps=1000000"]
    cmd = [sys.executable, str(script), "--run-dir", str(tmp_path), "--set", *sets]

    def run_until(sig, to_group):
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})
        lines = []
        for line in p.stdout:  # wait until it trains
            lines.append(line)
            if "Logging to" in line:
                break
        time.sleep(8.0)
        os.killpg(p.pid, sig) if to_group else os.kill(p.pid, sig)
        out = "".join(lines) + p.communicate(timeout=120)[0]
        return p.returncode, out

    rc, out = run_until(signal.SIGTERM, to_group=True)
    assert rc == 0 and "interrupted by SIGTERM" in out, out[-3000:]
    ck = T.latest_checkpoint(tmp_path)
    st = json.loads((ck / "state.json").read_text())
    assert (ck / T.BUFFER).exists() and st["stop_reason"] is None and st["num_timesteps"] > 0
    rc, out = run_until(signal.SIGUSR1, to_group=False)  # as the job script before the time limit
    assert rc == 0 and f"resumed from {ck} at {st['num_timesteps']} steps" in out and "interrupted by SIGUSR1" in out
    assert json.loads((T.latest_checkpoint(tmp_path) / "state.json").read_text())["num_timesteps"] > st["num_timesteps"]


def short_scenes(n):
    out = []
    for v in (0.1, 0.2, 0.3)[:n]:
        s = S.no_obstacles(v)
        s.duration = 0.5
        out.append(s)
    return out


def test_validation_runs_each_scene_once(tmp_path):
    cfg = load_config(None, TINY)
    model = T.new_model(cfg, T.make_vec_env(cfg, 2, 0, subproc=False))
    val_env = T.make_vec_env(cfg, 2, 0, short_scenes(3), subproc=False)
    r = T.validate(model, val_env)
    assert r["n"] == 3 and r["timeout"] == 1.0 and sum(r[o] for o in ("success", "collision", "dropped", "timeout")) == 1
    assert T.validate(model, val_env) == r  # deterministic policy, same scenes


def test_convergence_stops_training_and_keeps_the_best(tmp_path, monkeypatch):
    cfg = load_config(None, TINY)
    model = T.new_model(cfg, T.make_vec_env(cfg, 2, 0, subproc=False))
    model.set_logger(configure(None, []))
    cur = Curriculum(N_LEVELS)
    cur.level = N_LEVELS - 1
    cb = T.TrainingCallback(cfg, tmp_path, cur, None, {})
    cb.init_callback(model)
    results = iter([0.2, 0.3, 0.3, 0.25, 0.3])
    monkeypatch.setattr(T, "validate", lambda m, e, stop=None: {"success": next(results), "collision": 0, "dropped": 0,
                                                       "timeout": 0, "return": 0.0, "n": 50})
    stops = []
    for _ in range(5):
        cb.run_validation()
        stops.append(cb.stop_reason is not None)
    assert stops == [False, False, False, False, True] and cb.best == 0.3
    assert json.loads((tmp_path / "best" / "val.json").read_text())["success"] == 0.3
    assert {"actor.npz", "config.yaml", "model.zip"} <= {p.name for p in (tmp_path / "best").iterdir()}


def test_validation_does_not_stop_before_the_final_level(tmp_path, monkeypatch):
    cfg = load_config(None, TINY)
    model = T.new_model(cfg, T.make_vec_env(cfg, 2, 0, subproc=False))
    model.set_logger(configure(None, []))
    cb = T.TrainingCallback(cfg, tmp_path, Curriculum(N_LEVELS), None, {})
    cb.init_callback(model)
    monkeypatch.setattr(T, "validate", lambda m, e, stop=None: {"success": 0.0, "collision": 0, "dropped": 0, "timeout": 1,
                                                       "return": 0.0, "n": 50})
    for _ in range(5):
        cb.run_validation()
    assert cb.stop_reason is None and not (tmp_path / "best").exists()


@pytest.mark.parametrize("n_envs,freq,ups,expected", [(16, 1, 1.0, 16), (64, 1, 1.0, 64), (4, 2, 0.5, 4)])
def test_gradient_steps_per_call(n_envs, freq, ups, expected, monkeypatch):
    cfg = load_config(None, TINY + [f"sac.train_freq={freq}", f"sac.updates_per_env_step={ups}"])

    class Fake:
        num_envs = n_envs

    seen = {}
    monkeypatch.setattr(T, "SAC", lambda *a, **k: seen.update(k))
    T.new_model(cfg, Fake())
    assert seen["gradient_steps"] == expected and seen["train_freq"] == freq
