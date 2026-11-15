"""(D) SAC fine-tuning: n-step rows, buffers, the agent's updates, asynchronous environments, a run and its resume."""

import json
from pathlib import Path

import jax
import numpy as np
import yaml

from dyngrasp.rl import bc
from dyngrasp.rl.agent import Agent, actor_forward, sample_action
from dyngrasp.rl.config import load_config
from dyngrasp.rl.controller import RLController
from dyngrasp.rl.finetune import FIELDS, Buffer, NStep, finetune, nstep_arrays
from dyngrasp.rl.observation import OBS_DIM
from dyngrasp.rl.vecenv import AsyncEnvs

ARCH = [32, 32, 32]


def small_cfg(tmp_path, **ft):
    actor = bc.save_policy(bc.init_params(ARCH, 0), load_config(), tmp_path / "bc")
    cfg = load_config(overrides=[f"ft.bc_checkpoint={actor}", f"ft.critic_arch={ARCH}", "ft.batch_size=32",
                                 "ft.updates_per_call=4", "validation.n_scenes=2"])
    cfg["ft"].update(ft)
    return cfg


def test_nstep_rows():
    g = 0.9
    ns = NStep(3, g)
    rows = []
    for t in range(5):
        rows += ns.add(np.full(2, t), t, 1.0 + t, np.full(2, t + 1), t == 4)
    assert [r[1] for r in rows] == [0, 1, 2, 3, 4]
    assert np.isclose(rows[0][2], 1 + 2 * g + 3 * g**2) and rows[0][3][0] == 3 and rows[0][4] == 0.0
    assert np.isclose(rows[0][5], g**3)
    assert np.isclose(rows[2][2], 3 + 4 * g + 5 * g**2) and rows[2][4] == 1.0  # reaches the end
    assert np.isclose(rows[4][2], 5) and rows[4][4] == 1.0 and np.isclose(rows[4][5], g)
    d = {"obs": np.arange(10.0)[:, None], "act": np.zeros((10, 1)), "rew": np.ones(10), "next_obs": np.arange(1.0, 11)[:, None],
         "done": np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1]), "ep_start": [0, 4], "ep_len": [4, 6]}
    one, three = nstep_arrays(d, 1, g), nstep_arrays(d, 3, g)
    assert len(one["rew"]) == len(three["rew"]) == 10 and np.allclose(one["disc"], g)
    assert three["next_obs"][0, 0] == 3 and three["done"][1] == 1 and three["next_obs"][4, 0] == 7


def test_buffer_wraps_samples_and_round_trips(tmp_path):
    b = Buffer(5)
    for t in range(7):
        b.add([(np.full(OBS_DIM, t), np.zeros(8), t, np.zeros(OBS_DIM), 0.0, 0.99)])
    assert b.size == 5 and b.ptr == 2 and sorted(b.a["rew"]) == [2, 3, 4, 5, 6]
    s = b.sample(np.random.default_rng(0), (3, 4))
    assert s["obs"].shape == (3, 4, OBS_DIM) and s["rew"].shape == (3, 4) and set(FIELDS) == set(s)
    b.save(tmp_path / "b.npz")
    c = Buffer(5)
    c.load(tmp_path / "b.npz")
    assert c.size == 5 and c.ptr == 2 and all(np.array_equal(c.a[k], b.a[k]) for k in FIELDS)


def test_agent_updates_and_critic_only_warmup(tmp_path):
    cfg = small_cfg(tmp_path, n_critics=4, critic_subset=2, bc_weight=1.0, bc_policy_weight=1.0, bc_q_filter=True)
    params = bc.sbx_params(dict(np.load(Path(cfg["ft"]["bc_checkpoint"]) / "actor.npz")))
    agent = Agent(cfg, params)
    rng = np.random.default_rng(0)
    K, B = 4, 32
    batch = {"obs": rng.standard_normal((K, B, OBS_DIM)), "act": rng.uniform(-1, 1, (K, B, 8)), "rew": rng.standard_normal((K, B)),
             "next_obs": rng.standard_normal((K, B, OBS_DIM)), "done": (rng.random((K, B)) < 0.1).astype(float),
             "disc": np.full((K, B), 0.99), "is_demo": np.tile(np.arange(B) >= B // 2, (K, 1)).astype(float)}
    actor0 = jax.device_get(agent.state["actor"])
    m = agent.update(batch, train_actor=False)
    assert all(np.isfinite(v) for v in m.values())
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(actor0), jax.tree_util.tree_leaves(jax.device_get(agent.state["actor"]))))
    a0 = float(agent.state["log_alpha"])
    agent.update(batch, train_actor=True)
    assert int(agent.state["n_updates"]) == 2 * K
    assert not np.array_equal(jax.tree_util.tree_leaves(actor0)[0], np.asarray(jax.tree_util.tree_leaves(agent.state["actor"])[0]))
    assert float(agent.state["log_alpha"]) != a0
    # the deterministic action is RLController's (same arrays)
    x = rng.standard_normal((5, OBS_DIM)).astype(np.float32)
    ckpt = bc.save_policy(jax.device_get(agent.state["actor"]), cfg, tmp_path / "p")
    ctrl = RLController(ckpt)
    np.testing.assert_allclose(agent.act(x, jax.random.PRNGKey(0), deterministic=True), np.array([ctrl.actor(o) for o in x]), atol=1e-5)


def test_sampled_log_prob_matches_a_tanh_gaussian():
    mean, log_std = np.zeros((1, 8)), np.full((1, 8), -1.0)
    a, logp = sample_action(mean, log_std, jax.random.PRNGKey(0))
    u = np.arctanh(np.asarray(a))
    ref = np.sum(-0.5 * (u / np.exp(-1.0)) ** 2 + 1.0 - 0.5 * np.log(2 * np.pi) - np.log(1 - np.asarray(a) ** 2))
    assert np.isclose(float(logp[0]), ref, atol=1e-3)
    assert np.asarray(actor_forward(bc.init_params(ARCH, 0), np.zeros((2, OBS_DIM)))[0]).shape == (2, 8)


def test_async_envs_step_independently_and_remote_workers_come_and_go(tmp_path):
    import multiprocessing as mp

    from dyngrasp.rl.vecenv import remote_worker

    addr = tmp_path / "workers.addr"
    envs = AsyncEnvs(load_config(), 1, seed=0, level=0, remote=2, addr_file=addr)
    proc = mp.get_context("spawn").Process(target=remote_worker, args=(str(addr), str(tmp_path / "DONE")), daemon=True)
    proc.start()
    try:
        started = []
        while len(started) < 2:
            started += [(i, kind) for i, kind, _ in envs.poll(1.0)]
        assert sorted(started) == [(0, "start"), (1, "start")] and envs.live == 2
        envs.send([0, 1], np.zeros((2, 8)))
        got = {i: (kind, p) for i, kind, p in envs.drain()}
        assert set(got) == {0, 1} and all(k == "step" for k, _ in got.values())
        o, r, done, _, ep = got[1][1]
        assert o.shape == (OBS_DIM,) and np.isfinite(r) and not done and ep is None
        proc.kill()
        envs.send([1], np.zeros((1, 8)))
        lost = []
        while not lost:
            lost = [i for i, kind, _ in envs.poll(1.0) if kind == "lost"]
        assert lost == [1] and envs.live == 1
    finally:
        envs.close()
        proc.kill()


def test_finetune_runs_validates_and_resumes(tmp_path):
    demos = tmp_path / "demos.npz"
    n = 30
    rng = np.random.default_rng(0)
    np.savez(demos, obs=rng.standard_normal((n, OBS_DIM)), act=rng.uniform(-1, 1, (n, 8)), rew=np.zeros(n),
             next_obs=rng.standard_normal((n, OBS_DIM)), done=np.r_[np.zeros(n - 1), 1.0], ep_start=[0], ep_len=[n])
    cfg = small_cfg(tmp_path, n_envs=2, max_steps=40, learning_starts=10, warmup_steps=20, val_every=10**9,
                    val_workers=2, demos=str(demos), n_step=3, log_seconds=0.0)
    run = tmp_path / "run"
    assert finetune(run, cfg).startswith("max_steps")
    st = json.loads((run / "ckpt" / "state.json").read_text())
    assert st["steps"] >= 40 and len(st["val_history"]) <= 1
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["updates"] > 0
    (run / "DONE").unlink()
    c = yaml.safe_load((run / "config.yaml").read_text())
    c["ft"]["max_steps"] = 80
    (run / "config.yaml").write_text(yaml.safe_dump(c))
    assert finetune(run, cfg).startswith("max_steps")
    st2 = json.loads((run / "ckpt" / "state.json").read_text())
    assert st2["steps"] >= 80 and st2["starts"] == 2
    assert (run / "vals").exists()


def test_residual_mode_starts_at_the_bc_policy_and_exports_for_the_controller(tmp_path):
    cfg = small_cfg(tmp_path, residual_scale=0.3)
    params = jax.tree_util.tree_map(lambda x: x + 0.05, bc.sbx_params(dict(np.load(Path(cfg["ft"]["bc_checkpoint"]) / "actor.npz"))))
    agent = Agent(cfg, params)
    x = np.random.default_rng(0).standard_normal((6, OBS_DIM)).astype(np.float32)
    bc_mean = np.tanh(np.asarray(actor_forward(params, x)[0]))
    np.testing.assert_allclose(agent.act(x, jax.random.PRNGKey(0), deterministic=True), bc_mean, atol=1e-5)
    rng = np.random.default_rng(1)
    K, B = 2, 32
    batch = {"obs": rng.standard_normal((K, B, OBS_DIM)), "act": rng.uniform(-1, 1, (K, B, 8)), "rew": rng.standard_normal((K, B)),
             "next_obs": rng.standard_normal((K, B, OBS_DIM)), "done": np.zeros((K, B)), "disc": np.full((K, B), 0.99),
             "is_demo": np.zeros((K, B))}
    agent.update(batch, train_actor=True)
    a = agent.act(x, jax.random.PRNGKey(0), deterministic=True)
    assert not np.allclose(a, bc_mean, atol=1e-6)
    assert np.all(np.abs(np.arctanh(np.clip(a, -0.999999, 0.999999)) - np.arctanh(np.clip(bc_mean, -0.999999, 0.999999))) <= 0.3 + 1e-4)
    out = tmp_path / "res"
    out.mkdir()
    np.savez(out / "actor.npz", **agent.export_arrays())
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    np.testing.assert_allclose(np.array([RLController(out).actor(o) for o in x]), a, atol=1e-5)


def test_init_log_std_replaces_the_bc_noise():
    cfg = load_config(overrides=[f"ft.critic_arch={ARCH}", "ft.init_log_std=-3.0"])
    agent = Agent(cfg, bc.init_params(ARCH, 0))
    x = np.random.default_rng(0).standard_normal((4, OBS_DIM)).astype(np.float32)
    mean, log_std = agent.pi(agent.state["actor"], agent.state["bc"], x)
    np.testing.assert_allclose(np.asarray(log_std), -3.0)
    np.testing.assert_allclose(np.asarray(mean), np.asarray(actor_forward(bc.init_params(ARCH, 0), x)[0]), atol=1e-6)


def test_awac_and_q_norm_actors_update(tmp_path):
    rng = np.random.default_rng(0)
    K, B = 2, 32
    batch = {"obs": rng.standard_normal((K, B, OBS_DIM)), "act": rng.uniform(-1, 1, (K, B, 8)), "rew": rng.standard_normal((K, B)),
             "next_obs": rng.standard_normal((K, B, OBS_DIM)), "done": np.zeros((K, B)), "disc": np.full((K, B), 0.99),
             "is_demo": np.zeros((K, B))}
    for ft in ({"awac_beta": 0.1, "init_log_std": -3.0}, {"q_norm": True, "bc_policy_weight": 0.4}):
        cfg = small_cfg(tmp_path, **ft)
        agent = Agent(cfg, bc.sbx_params(dict(np.load(Path(cfg["ft"]["bc_checkpoint"]) / "actor.npz"))))
        leaves0 = jax.device_get(jax.tree_util.tree_leaves(agent.state["actor"]))
        m = agent.update(batch, train_actor=True)
        assert all(np.isfinite(v) for v in m.values())
        leaves1 = jax.device_get(jax.tree_util.tree_leaves(agent.state["actor"]))
        assert any(not np.array_equal(a, b) for a, b in zip(leaves0, leaves1))
        if "awac_beta" in ft:  # the log_std head gets no gradient
            assert np.allclose(np.asarray(agent.pi(agent.state["actor"], agent.state["bc"], batch["obs"][0])[1]), -3.0)
