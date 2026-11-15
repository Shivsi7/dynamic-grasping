"""(D) behaviour cloning: (C) -> (D) action conversion, demonstrations and replay, dataset, training, weights."""

import importlib.util
import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pytest

from dyngrasp import scenario as S
from dyngrasp.controller import WSG_OPEN
from dyngrasp.harness import shared_robot_model
from dyngrasp.rl import bc
from dyngrasp.rl.actions import gripper_width
from dyngrasp.rl.config import load_config
from dyngrasp.rl.controller import NumpyActor, RLController
from dyngrasp.rl.demos import (
    C_TICK,
    gripper_action,
    make_processor,
    replay,
    run_demo,
    run_takeover,
    to_actions,
)
from dyngrasp.rl.observation import OBS_DIM

CFG = load_config()
REPO = Path(__file__).resolve().parents[1]


def min_jerk_path(q0, q1, t, t_start, duration):
    s = np.clip((t - t_start) / duration, 0.0, 1.0)
    return q0 + (10 * s**3 - 15 * s**4 + 6 * s**5)[:, None] * (q1 - q0)


# ---- conversion -------------------------------------------------------------------------------------------------


def test_actions_replayed_through_the_ramp_reproduce_the_commanded_trajectory():
    """Two moves from rest (peak 3 rad/s^2, (C) allows 4; jerk up to 28 rad/s^3), starting off the 0.1 s grid; gripper
    closing at 3.537 s.  The ramp's velocity is piecewise linear: it tracks within jerk x dt^3 / 12 ~ 2 mrad."""
    robot, rng = shared_robot_model(), np.random.default_rng(0)
    q0 = rng.uniform(robot.q_lower + 1.0, robot.q_upper - 1.0)
    q1 = q0 + rng.uniform(-1.0, 1.0, 7)
    t = np.arange(0.0, 7.0, C_TICK)
    Q = min_jerk_path(q0, q1, t, 1.234, 1.5) + min_jerk_path(np.zeros(7), rng.uniform(-0.8, 0.8, 7), t, 3.6, 1.2)
    accel = np.abs(np.diff(Q, 2, axis=0)) / C_TICK**2
    assert 1.0 < accel.max() < 4.0
    G = np.where(t >= 3.537, 0.0, WSG_OPEN)
    n = 65
    acts, v_before, wsg_before, err = to_actions(Q, G, q0, n, make_processor(robot, CFG["env"]))
    assert acts.shape == (n, 8) and np.all(np.abs(acts) <= 1.0) and err.max() < 2e-3

    p = make_processor(robot, CFG["env"])  # replay the actions alone
    p.reset(q0, WSG_OPEN)
    for k, a in enumerate(acts):
        np.testing.assert_allclose(p.v, v_before[k], atol=1e-12)
        assert p.wsg == pytest.approx(wsg_before[k])
        p.start(a)
        np.testing.assert_allclose(p.Q, Q[20 * k + 2: 20 * k + 21: 2], atol=2e-3)  # the 10 ms ticks of the step
    np.testing.assert_allclose(acts[:35, 7], 1.0)
    np.testing.assert_allclose(acts[36:, 7], -1.0)  # step 35: commands 3.51 .. 3.60 s
    assert acts[35, 7] == pytest.approx(gripper_action(WSG_OPEN * 6 / 20))  # 6 of its 20 (C) ticks still open
    for w in (0.0, 0.03, WSG_OPEN):
        assert gripper_width(gripper_action(w)) == pytest.approx(w)


# ---- demonstrations from (C) ------------------------------------------------------------------------------------


def test_demo_of_c_and_its_open_loop_replay_succeed():
    spec = S.no_obstacles(0.1)
    d = run_demo(spec, 0, CFG)
    assert d["outcome"] == "success" and d["c_result"]["latency_mode"] == "none"
    obs, acts = d["obs"], d["actions"]
    assert obs.shape == (len(acts), OBS_DIM) and acts.shape[1] == 8 and len(acts) >= spec.duration / 0.1 - 1
    assert obs.dtype == np.float32 and np.all(np.isfinite(obs)) and np.all(np.abs(acts) <= 1.0)
    assert d["track_err_max"] < 0.01 and acts[0, 7] == pytest.approx(1.0) and acts[-1, 7] == pytest.approx(-1.0)
    r, seen = replay(spec, acts, CFG["env"])
    assert r.outcome == "success" and len(seen) == len(obs)
    np.testing.assert_allclose(seen[0], obs[0], atol=1e-6)  # the same observation as RLController's at t = 0
    robot = shared_robot_model()  # measured joint positions: the (D) ramp, commanded every 10 ms, tracks (C)'s 5 ms
    dq = np.abs(seen[:, :7] - obs[:, :7]) * 0.5 * (robot.q_upper - robot.q_lower)  # commands within a few mrad
    assert dq.max() < 0.1


def test_a_demo_over_its_wall_time_is_aborted():
    d = run_demo(S.no_obstacles(0.1), 0, CFG, max_wall_s=0.0)
    assert d["outcome"] == "aborted" and "obs" not in d


# ---- dataset ------------------------------------------------------------------------------------------------------


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_run(d: Path, lengths, outcomes, fingerprints, seed=0):
    (d / "eps").mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"seed": seed}))
    rng = np.random.default_rng(seed)
    with (d / "episodes.jsonl").open("w") as f:
        for e, (n, o, fp) in enumerate(zip(lengths, outcomes, fingerprints)):
            m = {"episode": e, "outcome": o, "level": e % 6, "seed": 10_000_000 + e, "fingerprint": fp, "wall_s": 1.0}
            if o == "success":
                m.update(n_steps=n, track_err_max=1e-4, action_clipped=0, replay_outcome="success" if e else "timeout",
                         replay_obs_diff_t0=0.0, replay_wall_s=1.0)
                np.savez_compressed(d / "eps" / f"{e}.npz", obs=rng.standard_normal((n, OBS_DIM)).astype(np.float32),
                                    act=rng.uniform(-1, 1, (n, 8)).astype(np.float32))
            f.write(json.dumps(m) + "\n")


def test_merged_dataset_shapes_and_split(tmp_path):
    fake_run(tmp_path / "a", [50, 60, 70, 80], ["success", "collision", "success", "success"], ["f0", "f1", "f2", "f3"])
    fake_run(tmp_path / "b", [40, 30], ["success", "success"], ["f3", "f4"], seed=1)  # f3: a duplicate scene
    mod = load_script("make_bc_demos")
    out = tmp_path / "data.npz"
    mod.merge(type("A", (), {"runs": [str(tmp_path / "a"), str(tmp_path / "b")], "out": str(out)}))
    z = np.load(out)
    assert list(z["ep_len"]) == [50, 70, 80, 30] and list(z["ep_start"]) == [0, 50, 120, 200]
    assert z["obs"].shape == (230, OBS_DIM) and z["act"].shape == (230, 8) and z["obs"].dtype == np.float32
    assert np.all(np.isfinite(z["obs"])) and np.all(np.isfinite(z["act"])) and list(z["replay_success"]) == [0, 1, 1, 1]
    assert z["obs_replay"].shape == z["obs"].shape  # (these runs have none: NaN)
    meta = json.loads(str(z["meta"]))
    assert meta["stats"]["duplicates_dropped"] == 1 and meta["stats"]["episodes_run"] == 6
    d = bc.load_dataset(out, 0.25, 0)
    assert len(d["val_episodes"]) == 1 and len(d["train"]) + len(d["val"]) == 230
    e = d["val_episodes"][0]
    assert set(d["val"]) == set(range(z["ep_start"][e], z["ep_start"][e] + z["ep_len"][e]))


# ---- training and weights -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [[], ["bc.ema=0.99", "bc.obs_noise=0.01", "bc.keep=last"]])
def test_bc_training_lowers_the_loss_and_its_policy_runs(tmp_path, extra):
    rng = np.random.default_rng(0)
    n_ep, T = 40, 50
    obs = rng.standard_normal((n_ep * T, OBS_DIM)).astype(np.float32)
    act = np.tanh(obs[:, :5] @ rng.standard_normal((5, 8))).astype(np.float32)
    act[:, 7] = np.sign(obs[:, 0])
    np.savez(tmp_path / "data.npz", obs=obs, act=act, ep_start=np.arange(n_ep) * T, ep_len=np.full(n_ep, T))
    cfg = load_config(None, ["sac.net_arch=[64,64]", "bc.max_epochs=40", "bc.batch_size=64", "bc.learning_rate=1e-3",
                             "bc.closed_loop_every=0", "validation.n_scenes=1", "bc.closed_loop_workers=1", *extra])
    bc.train(tmp_path / "run", cfg, tmp_path / "data.npz")
    log = pd.read_csv(tmp_path / "run" / "log.csv")
    assert log.val_mse.min() < 0.3 * log.val_mse.iloc[0] and log.train_mse.iloc[-1] < 0.3 * log.train_mse.iloc[0]
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    kept = log.val_mse.iloc[-1] if "bc.keep=last" in extra else log.val_mse.min()
    assert run["best_val_mse"] == pytest.approx(kept) and run["best_closed_loop"]["n"] == 1
    assert (tmp_path / "run" / "best" / "closed_loop.json").exists()


def test_bc_weights_load_into_the_controller_and_sbx(tmp_path):
    import gymnasium as gym
    from sbx import SAC
    from sbx.common.policies import SquashedGaussianActor

    params = bc.init_params([64, 64, 64], 3)
    params = jax.tree_util.tree_map(lambda x: x + 0.05 * np.random.default_rng(0).standard_normal(x.shape), params)
    x = np.random.default_rng(1).standard_normal((16, OBS_DIM)).astype(np.float32)
    mean, log_std = (np.asarray(v) for v in bc.forward(params, x))
    dist = SquashedGaussianActor(net_arch=[64, 64, 64], action_dim=8).apply(params, x)  # sbx's own forward
    np.testing.assert_allclose(np.asarray(dist.mode()), np.tanh(mean), atol=1e-5)
    np.testing.assert_allclose(np.asarray(dist.distribution.stddev()), np.exp(np.clip(log_std, -20, 2)), rtol=1e-5)

    ckpt = bc.save_policy(params, load_config(), tmp_path / "policy")
    ctrl = RLController(ckpt)  # what eval_rl.py loads
    np.testing.assert_allclose(np.array([ctrl.actor(o) for o in x]), np.tanh(mean), atol=1e-5)

    class Spaces(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (8,), np.float32)

    model = SAC("MlpPolicy", Spaces(), policy_kwargs={"net_arch": [64, 64, 64]}, seed=0)
    loaded = bc.sbx_params(dict(np.load(ckpt / "actor.npz")))
    assert jax.tree_util.tree_structure(loaded) == jax.tree_util.tree_structure(model.policy.actor_state.params)
    model.policy.actor_state = model.policy.actor_state.replace(params=loaded)
    np.testing.assert_allclose(model.predict(x, deterministic=True)[0], np.tanh(mean), atol=1e-5)
    assert np.array_equal(NumpyActor(dict(np.load(ckpt / "actor.npz"))).mean[0], np.asarray(params["params"]["Dense_3"]["kernel"]))


# ---- takeover: (C) through (D)'s actions (DAgger / DART) ----------------------------------------------------------


def test_c_through_d_actions_from_the_start_grasps_like_c():
    spec = S.no_obstacles(0.1)
    d = run_takeover(spec, 0, CFG, None, 0.0)
    c = run_demo(spec, 0, CFG)
    assert d["outcome"] == "success" and d["n_policy_steps"] == 0 and d["c_result"]["latency_mode"] == "none"
    assert abs(d["c_result"]["t_grasp"] - c["c_result"]["t_grasp"]) < 0.15
    assert d["obs"].shape == (len(d["actions"]), OBS_DIM) and np.array_equal(d["executed"], d["actions"])
    r, _ = replay(spec, d["actions"], CFG["env"])  # the labels are the executed actions: they replay
    assert r.outcome == "success"
    noisy = run_takeover(spec, 0, CFG, None, 0.0, noise_std=0.03, seed=1)  # DART: executed with noise, labelled clean
    assert noisy["outcome"] == "success" and np.abs(noisy["executed"] - noisy["actions"])[:, :7].max() > 0.05


def test_takeover_brakes_a_moving_robot_and_c_grasps():
    """A policy swinging joint 1 at 0.4 v_max until 1.5 s: (C) brakes it, plans from where it stops, and grasps."""
    spec = S.no_obstacles(0.1)
    d = run_takeover(spec, 0, CFG, lambda obs: np.array([0.4, 0, 0, 0, 0, 0, 0, 1.0]), 1.5)
    assert d["outcome"] == "success" and d["n_policy_steps"] == 15 and len(d["executed"]) == 15 + len(d["actions"])
    assert d["obs"][0, 7] > 0.3  # the first labelled step: moving at the policy's speed ...
    assert d["actions"][0, 0] < d["obs"][0, 7] - 0.2  # ... (C) brakes
