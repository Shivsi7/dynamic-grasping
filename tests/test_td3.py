"""(D) TD3+BC fine-tuning: deterministic acting with clipped exploration noise, smoothed targets, delayed actor and
target updates, no temperature, and a run through finetune() with the noise decay."""

import json
from pathlib import Path

import jax
import numpy as np

from dyngrasp.rl import bc
from dyngrasp.rl.agent import Agent, actor_forward, td3_target_action
from dyngrasp.rl.config import load_config
from dyngrasp.rl.controller import RLController
from dyngrasp.rl.finetune import finetune
from dyngrasp.rl.observation import OBS_DIM

ARCH = [32, 32, 32]


def small_cfg(tmp_path, **ft):
    actor = bc.save_policy(bc.init_params(ARCH, 0), load_config(), tmp_path / "bc")
    cfg = load_config(overrides=[f"ft.bc_checkpoint={actor}", f"ft.critic_arch={ARCH}", "ft.batch_size=32",
                                 "ft.updates_per_call=1", "validation.n_scenes=2", "ft.td3=True"])
    cfg["ft"].update(ft)
    return cfg


def make_batch(rng, K, B):
    return {"obs": rng.standard_normal((K, B, OBS_DIM)), "act": rng.uniform(-1, 1, (K, B, 8)),
            "rew": rng.standard_normal((K, B)), "next_obs": rng.standard_normal((K, B, OBS_DIM)),
            "done": (rng.random((K, B)) < 0.1).astype(float), "disc": np.full((K, B), 0.99),
            "is_demo": np.tile(np.arange(B) >= B // 2, (K, 1)).astype(float)}


def test_td3_acts_deterministically_plus_clipped_exploration_noise(tmp_path):
    cfg = small_cfg(tmp_path, td3_explore_std=0.1)
    params = bc.sbx_params(dict(np.load(Path(cfg["ft"]["bc_checkpoint"]) / "actor.npz")))
    agent = Agent(cfg, params)
    x = np.random.default_rng(0).standard_normal((2000, OBS_DIM)).astype(np.float32)
    mean = np.tanh(np.asarray(actor_forward(params, x)[0]))
    np.testing.assert_allclose(agent.act(x, jax.random.PRNGKey(0), deterministic=True), mean, atol=1e-6)
    a = agent.act(x, jax.random.PRNGKey(1))
    assert np.all(np.abs(a) <= 1.0) and not np.allclose(a, mean)
    inside = np.abs(mean) < 0.5  # where the clip does not bite, the noise has the configured std
    assert abs(np.std((a - mean)[inside]) - 0.1) < 0.01
    agent.explore_std = 0.0  # (what finetune.py's decay sets)
    np.testing.assert_allclose(agent.act(x, jax.random.PRNGKey(2)), mean, atol=1e-6)
    # the deterministic action is RLController's (same arrays)
    ckpt = bc.save_policy(jax.device_get(agent.state["actor"]), cfg, tmp_path / "p")
    np.testing.assert_allclose(np.array([RLController(ckpt).actor(o) for o in x[:5]]), mean[:5], atol=1e-5)


def test_td3_target_action_is_smoothed_within_the_clip():
    mean = np.random.default_rng(0).standard_normal((500, 8)) * 3
    a = np.asarray(td3_target_action(mean, jax.random.PRNGKey(0), 0.2, 0.5))
    assert np.all(np.abs(a) <= 1.0) and np.all(np.abs(a - np.tanh(mean)) <= 0.5 + 1e-6)
    assert not np.allclose(a, np.tanh(mean))
    np.testing.assert_allclose(np.asarray(td3_target_action(mean, jax.random.PRNGKey(0), 0.0, 0.5)), np.tanh(mean), atol=1e-6)


def test_td3_update_delays_actor_and_targets_and_keeps_no_temperature(tmp_path):
    cfg = small_cfg(tmp_path, actor_delay=2, q_norm=True, bc_policy_weight=0.5, bc_weight=0.5, n_critics=2)
    params = bc.sbx_params(dict(np.load(Path(cfg["ft"]["bc_checkpoint"]) / "actor.npz")))
    agent = Agent(cfg, params)
    rng = np.random.default_rng(0)
    leaves = lambda k: jax.device_get(jax.tree_util.tree_leaves(agent.state[k]))
    same = lambda a, b: all(np.array_equal(x, y) for x, y in zip(a, b))
    a0, c0, t0, alpha0 = leaves("actor"), leaves("critic"), leaves("target"), float(agent.state["log_alpha"])
    m = agent.update(make_batch(rng, 1, 32), train_actor=True)  # update 1 of 2: critics only
    assert all(np.isfinite(v) for v in m.values()) and int(agent.state["n_updates"]) == 1
    assert not same(c0, leaves("critic")) and same(a0, leaves("actor")) and same(t0, leaves("target"))
    agent.update(make_batch(rng, 1, 32), train_actor=True)  # update 2: actor and targets move
    assert not same(a0, leaves("actor")) and not same(t0, leaves("target"))
    assert float(agent.state["log_alpha"]) == alpha0  # no temperature update in TD3 mode
    # the unused log_std head keeps its values (no gradient reaches it)
    p0, p1 = params["params"], jax.device_get(agent.state["actor"])["params"]
    n = len(p1) - 2
    np.testing.assert_allclose(p1[f"Dense_{n + 1}"]["kernel"], np.asarray(p0[f"Dense_{n + 1}"]["kernel"]))
    # a critic-only warm-up leaves the actor and targets alone whatever the delay says
    a2 = leaves("actor")
    agent.update(make_batch(rng, 2, 32), train_actor=False)
    assert same(a2, leaves("actor")) and int(agent.state["n_updates"]) == 4
    # q_norm: the RL term is scale-free, so the anchor keeps the policy near the start
    x = rng.standard_normal((64, OBS_DIM)).astype(np.float32)
    drift = np.abs(agent.act(x, jax.random.PRNGKey(0), deterministic=True) - np.tanh(np.asarray(actor_forward(params, x)[0])))
    assert drift.max() < 0.2


def test_td3_finetune_runs_decays_the_noise_and_resumes(tmp_path):
    demos = tmp_path / "demos.npz"
    n = 30
    rng = np.random.default_rng(0)
    np.savez(demos, obs=rng.standard_normal((n, OBS_DIM)), act=rng.uniform(-1, 1, (n, 8)), rew=np.zeros(n),
             next_obs=rng.standard_normal((n, OBS_DIM)), done=np.r_[np.zeros(n - 1), 1.0], ep_start=[0], ep_len=[n])
    cfg = small_cfg(tmp_path, n_envs=2, max_steps=40, learning_starts=10, warmup_steps=20, val_every=10**9,
                    val_workers=2, demos=str(demos), n_step=3, log_seconds=0.0, actor_delay=2, q_norm=True,
                    bc_policy_weight=0.5, td3_explore_std=0.2, td3_explore_final=0.0, td3_explore_steps=80,
                    updates_per_call=4)
    run = tmp_path / "run"
    assert finetune(run, cfg).startswith("max_steps")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["updates"] > 0 and 0.0 < rows[-1]["explore_std"] < 0.2
    with_metrics = [r for r in rows if "alpha" in r]
    assert np.isfinite(rows[-1]["critic_loss"]) and rows[-1]["alpha"] == with_metrics[0]["alpha"]  # never updated
    (run / "DONE").unlink()
    import yaml

    c = yaml.safe_load((run / "config.yaml").read_text())
    c["ft"]["max_steps"] = 90
    (run / "config.yaml").write_text(yaml.safe_dump(c))
    assert finetune(run, cfg).startswith("max_steps")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    st = json.loads((run / "ckpt" / "state.json").read_text())
    assert st["steps"] >= 90 and st["starts"] == 2 and rows[-1]["explore_std"] == 0.0  # fully decayed after 80 steps
