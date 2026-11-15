"""RL PPO: returns, the value report, the learner's log-probabilities against the workers', its combined update, V
pretraining and DAgger labels made before a run (scripts/ppo_offline.py), DAgger takeovers from an episode's executed
actions, and a run with remote workers, its gate and its resume."""

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from dyngrasp import scenario as S
from dyngrasp.rl import bc
from dyngrasp.rl.config import load_config
from dyngrasp.rl.controller import NumpyActor
from dyngrasp.rl.observation import OBS_DIM
from dyngrasp.rl.ppo import (
    Imitation,
    Learner,
    gae,
    mc_returns,
    ppo,
    pretrain_value,
    schedule,
    value_report,
)
from dyngrasp.rl.rollout import NumpyPolicy, takeover_labels

ARCH = [32, 32, 32]


def small_cfg(tmp_path, **p):
    actor = bc.save_policy(bc.init_params(ARCH, 0), load_config(), tmp_path / "bc")
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "demos.npz", obs=rng.standard_normal((500, OBS_DIM)).astype(np.float32),
             act=rng.uniform(-1, 1, (500, 8)).astype(np.float32))
    cfg = load_config(overrides=[f"ppo.bc_checkpoint={actor}", f"ppo.demos={tmp_path / 'demos.npz'}", f"ppo.value_arch={ARCH}",
                                 "ppo.minibatch=64", "ppo.imit_batch=32", "ppo.min_dagger=10", "validation.n_scenes=2",
                                 "ppo.val_workers=2"])
    cfg["ppo"].update(p)
    return cfg


def test_gae_and_monte_carlo_returns():
    g, lam = 0.9, 0.8
    rew, val, ep_len = np.array([1.0, 2.0, 3.0, 5.0, -1.0]), np.array([0.5, 1.0, 1.5, 2.0, 0.0]), [3, 2]
    adv = gae(rew, val, ep_len, g, lam)
    d = [1 + g * 1.0 - 0.5, 2 + g * 1.5 - 1.0, 3 - 1.5]
    assert np.allclose(adv[:3], [d[0] + g * lam * (d[1] + g * lam * d[2]), d[1] + g * lam * d[2], d[2]])
    assert np.allclose(adv[3:], [5 + g * 0.0 - 2.0 + g * lam * (-1.0), -1.0])
    assert np.allclose(mc_returns(rew, ep_len, g), [1 + g * 2 + g * g * 3, 2 + g * 3, 3, 5 - g, -1])


def test_value_report_on_perfect_and_constant_values():
    g = np.array([1.0, 0.5, -1.0, -0.5, 2.0])
    r = value_report(g.copy(), g, np.array([2, 2, 1]), np.array([True, False, True]))
    assert np.isclose(r["v_ev"], 1.0) and np.isclose(r["v_corr"], 1.0) and r["v_bias"] == 0.0
    assert np.isclose(r["v0_succ"], 1.5) and np.isclose(r["v0_fail"], -1.0) and np.isclose(r["g_fail"], -0.75)
    assert np.isclose(value_report(np.full(5, g.mean()), g, np.array([5]), np.array([True]))["v_ev"], 0.0)


def test_worker_and_learner_agree_on_the_policy_and_its_log_probability(tmp_path):
    cfg = small_cfg(tmp_path)
    learner = Learner(cfg, dict(np.load(Path(cfg["ppo"]["bc_checkpoint"]) / "actor.npz")))
    arrays = learner.export(ema=False)
    pol, rng = NumpyPolicy(arrays), np.random.default_rng(0)
    obs = rng.standard_normal((20, OBS_DIM))
    us, lps = zip(*(pol.sample(o, rng) for o in obs))
    lp = learner.logp(obs.astype(np.float32), np.array(us, np.float32))
    assert np.allclose(lp, lps, atol=1e-3)  # the importance ratio of fresh data is 1
    assert np.allclose(NumpyActor(arrays)(obs[0]), np.tanh(pol.mu(obs[0])))  # what RLController runs
    assert np.allclose(np.exp(pol.log_std), np.exp(cfg["ppo"]["init_log_std"]), rtol=1e-6)
    bc0 = NumpyActor(dict(np.load(Path(cfg["ppo"]["bc_checkpoint"]) / "actor.npz")))
    assert np.allclose(bc0(obs[0]), NumpyActor(arrays)(obs[0]), atol=1e-6)  # starts as the BC policy


def test_phase_0_update_is_imitation_only_and_ppo_moves_the_policy(tmp_path):
    cfg = small_cfg(tmp_path)
    learner = Learner(cfg, dict(np.load(Path(cfg["ppo"]["bc_checkpoint"]) / "actor.npz")))
    rng = np.random.default_rng(0)
    imitation = Imitation(cfg["ppo"]["demos"], 32, 0.5, 10)
    imitation.add(rng.standard_normal((20, OBS_DIM)), rng.uniform(-1, 1, (20, 8)))
    obs = rng.standard_normal((256, OBS_DIM)).astype(np.float32)
    pol = NumpyPolicy(learner.export(ema=False))
    u = np.array([pol.sample(o, rng)[0] for o in obs], np.float32)  # the policy's own actions
    batch = {"obs": obs, "u": u, "logp": learner.logp(obs, u), "adv": rng.standard_normal(256).astype(np.float32)}
    vb = {"obs": obs, "ret": rng.standard_normal(256).astype(np.float32)}
    import jax

    s0 = schedule(cfg["ppo"], 0, None)
    assert s0["w"] == 0.0 and s0["lam"] == cfg["ppo"]["imit_weight"] and s0["label_fraction"] == cfg["ppo"]["label_fraction"]
    m0 = learner.update(batch, vb, imitation, s0, rng, jax.random.PRNGKey(0))
    assert m0["epochs"] == cfg["ppo"]["epochs"] and np.isfinite(m0["value_loss"]) and np.isfinite(m0["l_ppo"])
    ema_moved = learner.s["ema"]["mean"][1] - learner.s["net"]["mean"][1]
    assert np.any(np.asarray(ema_moved) != 0)  # the EMA lags the trained network
    s1 = schedule(cfg["ppo"], cfg["ppo"]["ramp_batches"] // 2, 0)
    assert 0 < s1["w"] < 1 and np.isclose(s1["lam"], 0.5 * cfg["ppo"]["imit_weight"])
    assert schedule(cfg["ppo"], 10**6, 0)["lam"] == 0.0
    # PPO alone (lam = 0, a fresh learner): the actions with positive advantages become likelier than the others
    learner = Learner(cfg, dict(np.load(Path(cfg["ppo"]["bc_checkpoint"]) / "actor.npz")))
    good = np.arange(256) % 2 == 0
    batch["adv"] = np.where(good, 1.0, -1.0).astype(np.float32)
    batch["logp"] = learner.logp(obs, u)
    learner.update(batch, vb, imitation, {**schedule(cfg["ppo"], 10**6, 0), "c": [0.0] * 8}, rng, jax.random.PRNGKey(1))
    d = learner.logp(obs, u) - batch["logp"]
    assert d[good].mean() > 0 > d[~good].mean()


def test_value_pretraining_and_initial_dagger_labels(tmp_path):
    cfg = small_cfg(tmp_path, minibatch=256)
    learner = Learner(cfg, dict(np.load(Path(cfg["ppo"]["bc_checkpoint"]) / "actor.npz")))
    rng = np.random.default_rng(0)
    ep_len = rng.integers(5, 30, 400)
    c, end = rng.standard_normal(400), np.cumsum(ep_len)
    obs = rng.standard_normal((ep_len.sum(), OBS_DIM)).astype(np.float32)
    obs[:, 0] = np.repeat(c, ep_len)  # the episode's final reward ...
    obs[:, 1] = (np.repeat(end, ep_len) - np.arange(ep_len.sum())) / 30.0  # ... and the steps to it: learnable returns
    rew = np.zeros(ep_len.sum())
    rew[end - 1] = c
    np.savez(tmp_path / "r.npz", obs=obs, rew=rew, ep_len=ep_len, success=rng.random(400) < 0.5)
    rep = pretrain_value(learner, tmp_path / "r.npz", cfg["gamma"], 30, rng)
    assert rep["episodes"] == 400 and rep["after"]["v_ev"] > max(0.3, rep["before"]["v_ev"])
    # initial labels: sampled from the start, never saved (reloaded at each start), the online ones saved and restored
    np.savez(tmp_path / "l.npz", obs=np.ones((30, OBS_DIM), np.float32), act=np.zeros((30, 8), np.float32))
    im = Imitation(cfg["ppo"]["demos"], 32, 0.5, 10, tmp_path / "l.npz")
    assert im.n_dagger == 30 and np.sum(np.all(np.asarray(im(rng)["obs"]) == 1.0, 1)) == 16
    im.add(np.full((5, OBS_DIM), 2.0), np.zeros((5, 8)))
    im.save(tmp_path / "d.npz")
    assert len(np.load(tmp_path / "d.npz")["act"]) == 5
    im2 = Imitation(cfg["ppo"]["demos"], 32, 0.5, 10, tmp_path / "l.npz")
    im2.load(tmp_path / "d.npz")
    assert im2.n_dagger == 35 and np.all(im2.obs[30:] == 2.0)


def test_offline_rollouts_and_takeover_labels_merge(tmp_path, monkeypatch):
    """scripts/ppo_offline.py: two scenes of a policy's noisy rollouts, a takeover each, merged."""
    import argparse
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))  # (importable: its workers unpickle `task`)
    off = importlib.import_module("ppo_offline")
    cfg = small_cfg(tmp_path)
    out = tmp_path / "off" / "part0"
    off.run(argparse.Namespace(policy=cfg["ppo"]["bc_checkpoint"], out=str(out), episodes=2, seed=5, switches=1, workers=2,
                               tasks_per_child=1, set=["ppo.label_max_wall_s=60"]))
    rows = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    assert sorted(r["e"] for r in rows) == [0, 1] and all(len(r["takeovers"]) == 1 for r in rows)
    off.merge(argparse.Namespace(runs=[str(out)], out=str(tmp_path / "off" / "m")))
    z = np.load(tmp_path / "off" / "m_rollouts.npz")
    assert len(z["ep_len"]) == 2 and z["obs"].shape == (z["ep_len"].sum(), OBS_DIM) and len(z["rew"]) == len(z["obs"])
    assert np.isclose(sum(r["return"] for r in rows), z["rew"].sum())
    n_lab = sum(t.get("n_labels", 0) for r in rows for t in r["takeovers"])
    assert len(np.load(tmp_path / "off" / "m_labels.npz")["act"]) == n_lab


def test_takeover_labels_from_an_episodes_executed_actions():
    """The policy held still for 2 s, then (C): labels from the switch on."""
    spec = S.no_obstacles(0.1)
    executed = np.tile(np.concatenate([np.zeros(7), [1.0]]), (60, 1))
    cfg = load_config()
    d = takeover_labels(spec, executed, 2.2, cfg, np.random.default_rng(3))
    assert 0.0 <= d["t_switch"] <= 2.0 and d["outcome"] == "success" and d["n_labels"] == len(d["act"]) > 10
    assert d["obs"].shape == (d["n_labels"], OBS_DIM) and d["n_policy_steps"] == round(d["t_switch"] / 0.1)


def test_ppo_run_with_remote_workers_gates_labels_and_resumes(tmp_path):
    """A run stopped at its gate (stop_at_gate), continued with PPO ramping in to max_steps, then resumed."""
    import yaml

    from dyngrasp.rl.vecenv import remote_worker

    run = tmp_path / "run"
    cfg = small_cfg(tmp_path, batch_steps=150, gate_ev=-1e9, gate_batches=1, min_pretrain=2, ramp_batches=2,
                    label_fraction=1.0, max_steps=600, val_every=2, label_max_wall_s=60.0, checkpoint_minutes=0.01,
                    stop_at_gate=True)
    ctx = mp.get_context("spawn")
    workers = []

    def ensure_workers():  # (they end when they see DONE)
        workers[:] = [w for w in workers if w.is_alive()]
        while len(workers) < 2:
            w = ctx.Process(target=remote_worker, args=(str(run / "workers.addr"), str(run / "DONE"), None, True), daemon=True)
            w.start()
            workers.append(w)

    try:
        ensure_workers()
        assert ppo(run, cfg).startswith("gate open at batch 1")
        st0 = json.loads((run / "ckpt" / "state.json").read_text())
        rows0 = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        assert st0["batch"] == st0["ramp_start"] == 2 and len(rows0) == 2 and all(r["w"] == 0.0 for r in rows0)
        # the fork: on from the gate's checkpoint with its own config
        (run / "DONE").unlink()
        cfg1 = {**cfg, "ppo": {**cfg["ppo"], "stop_at_gate": False, "window": 2}}
        (run / "config.yaml").write_text(yaml.safe_dump(cfg1))
        ensure_workers()
        assert ppo(run, cfg1).startswith("max_steps")
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        assert len(rows) >= 4 and rows[-1]["steps"] >= 600 and rows[2]["ramp_x"] == 0.0 < rows[3]["ramp_x"]
        assert abs(rows[0]["ratio0_mean"] - 1.0) < 0.05
        assert all(k in rows[0] for k in ("v_ev", "v_corr", "v_bias", "log_std", "imit_mse", "success"))
        assert (run / "labels.jsonl").exists() and (run / "final" / "actor.npz").exists()
        assert json.loads((run / "policy" / "latest.json").read_text())["version"] == rows[-1]["batch"]
        (run / "DONE").unlink()
        st = json.loads((run / "ckpt" / "state.json").read_text())
        cfg2 = {**cfg1, "ppo": {**cfg1["ppo"], "max_steps": st["steps"] + 150}}
        (run / "config.yaml").write_text(yaml.safe_dump(cfg2))
        ensure_workers()
        assert ppo(run, cfg2).startswith("max_steps")
        rows2 = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        assert rows2[-1]["batch"] == st["batch"] + 1 and rows2[-1]["ramp_x"] == 1.0
    finally:
        (run / "DONE").touch()
        for w in workers:
            w.join(timeout=30)
            if w.is_alive():
                w.terminate()
