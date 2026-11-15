"""(D): action processing, rewards, terminations, observation, curriculum, scene disjointness, controller."""

import itertools

import numpy as np
import pytest
from pydrake.all import RigidTransform, RotationMatrix

from dyngrasp import scenario as S
from dyngrasp.controller import WSG_OPEN
from dyngrasp.harness import run_episode
from dyngrasp.reactive import ReactiveQPController
from dyngrasp.rl import scenes as R
from dyngrasp.rl.actions import ActionProcessor, gripper_width, is_close_event
from dyngrasp.rl.config import DEFAULTS, load_config
from dyngrasp.rl.controller import NumpyActor, RLController
from dyngrasp.rl.curriculum import Curriculum, sample_level
from dyngrasp.rl.env import GraspEnv
from dyngrasp.rl.observation import N_OBSTACLES, OBS_DIM
from dyngrasp.rl.rewards import clearance_potential, close_reward, progress_potential

LO, HI, VMAX, AMAX = -np.ones(7) * 2.0, np.ones(7) * 2.0, np.ones(7) * 1.5, np.ones(7) * 5.0
REWARD = DEFAULTS["reward"]


# ---- action processing --------------------------------------------------------------------------------


def test_ramp_reaches_the_target_velocity_and_integrates_it():
    p = ActionProcessor(LO, HI, VMAX, AMAX, dt=0.1, tick=0.01, margin=0.05)
    p.reset(np.zeros(7))
    a = np.zeros(8)
    a[0] = 0.2  # 0.3 rad/s: within one step's velocity change (a_max dt = 0.5)
    p.start(a)
    assert p.v[0] == pytest.approx(0.3) and np.all(p.v[1:] == 0)
    tau = 0.01 * np.arange(1, 11)
    np.testing.assert_allclose(p.Q[:, 0], 0.3 * tau**2 / (2 * 0.1))  # linear ramp 0 -> 0.3 over 0.1 s
    assert p.q[0] == pytest.approx(0.015) and np.all(p.Q[:, 1:] == 0)
    p.start(a)  # constant velocity from now on
    np.testing.assert_allclose(np.diff(p.Q[:, 0]), 0.003)
    assert np.array_equal(p.q_at(0), p.Q[0]) and np.array_equal(p.q_at(99), p.Q[-1])


def test_velocity_change_is_clipped_to_the_acceleration_limit():
    p = ActionProcessor(LO, HI, VMAX, [5.0, 1.0, 5, 5, 5, 5, 5], dt=0.1, tick=0.01, margin=0.05)
    p.reset(np.zeros(7))
    p.start(np.array([1.0, -1.0, 0, 0, 0, 0, 0, 1.0]))  # targets +-1.5 rad/s
    np.testing.assert_allclose(p.v[:2], [0.5, -0.1])  # a_max x dt
    for _ in range(5):
        p.start(np.array([1.0, -1.0, 0, 0, 0, 0, 0, 1.0]))
    np.testing.assert_allclose(p.v[:2], [1.5, -0.6])  # joint 0 saturates at the target
    p.start(np.array([-1.0, 1.0, 0, 0, 0, 0, 0, 1.0]))
    np.testing.assert_allclose(p.v[:2], [1.0, -0.5])


def test_joint_limits_clip_positions_and_stop_the_joint():
    p = ActionProcessor(LO, HI, VMAX, AMAX, dt=0.1, tick=0.01, margin=0.05)
    p.reset(np.array([1.9, -1.9, 0, 0, 0, 0, 0]))
    p.v = np.array([1.5, -1.5, 0, 0, 0, 0, 0])
    p.start(np.array([1.0, -1.0, 0, 0, 0, 0, 0, 1.0]))
    assert np.all(p.Q[:, 0] <= 1.95 + 1e-12) and np.all(p.Q[:, 1] >= -1.95 - 1e-12)
    assert p.q[0] == pytest.approx(1.95) and p.q[1] == pytest.approx(-1.95)
    assert p.v[0] == 0.0 and p.v[1] == 0.0  # stopped at the clip
    p.start(np.array([-1.0, 1.0, 0, 0, 0, 0, 0, 1.0]))  # moving back in: not clipped, not stopped
    assert p.v[0] == pytest.approx(-0.5) and p.q[0] < 1.95


def test_gripper_mapping_and_close_events():
    assert gripper_width(-1) == 0.0 and gripper_width(1) == pytest.approx(WSG_OPEN)
    assert gripper_width(0) == pytest.approx(WSG_OPEN / 2) and gripper_width(5) == pytest.approx(WSG_OPEN)
    thr = 0.064
    assert is_close_event(WSG_OPEN, 0.0, thr)
    assert is_close_event(thr, thr - 1e-4, thr)
    assert not is_close_event(WSG_OPEN, 0.07, thr)  # partial closing above the object's width is free
    assert not is_close_event(0.03, 0.0, thr)  # already closed
    assert not is_close_event(0.0, WSG_OPEN, thr)  # opening


# ---- rewards --------------------------------------------------------------------------------------------


def test_progress_potential():
    X_grasp = RigidTransform(RotationMatrix.MakeZRotation(0.3), [0.6, 0.0, 0.3])
    at = progress_potential(X_grasp, X_grasp, False, 0.1)
    near = progress_potential(RigidTransform(X_grasp.rotation(), [0.6, 0.0, 0.4]), X_grasp, False, 0.1)
    turned = progress_potential(RigidTransform(RotationMatrix(), [0.6, 0.0, 0.4]), X_grasp, False, 0.1)
    assert at == pytest.approx(0.0, abs=1e-9) and near == pytest.approx(-0.1) and turned == pytest.approx(-0.13)
    assert progress_potential(RigidTransform([0, 0, 1.0]), X_grasp, True, 0.1) == 0.0  # in hand
    k, g = REWARD["k_p"], 0.99
    assert k * (g * near - turned) > 0 > k * (g * turned - near)  # turning towards the grasp pays, away costs


def test_clearance_potential():
    assert clearance_potential(np.inf, 0.04, 0.5) == 0.0 and clearance_potential(0.05, 0.04, 0.5) == 0.0
    assert clearance_potential(0.02, 0.04, 0.5) == pytest.approx(-0.25)
    assert clearance_potential(0.0, 0.04, 0.5) == pytest.approx(-0.5)
    assert clearance_potential(-0.01, 0.04, 0.5) == pytest.approx(-0.625)
    g = 0.99  # moving away from an obstacle pays, towards costs
    assert g * clearance_potential(0.03, 0.04, 0.5) - clearance_potential(0.01, 0.04, 0.5) > 0
    assert g * clearance_potential(0.01, 0.04, 0.5) - clearance_potential(0.03, 0.04, 0.5) < 0


def test_close_reward():
    assert close_reward(True, True, 0.0, REWARD) == pytest.approx(0.5)
    assert close_reward(True, True, 0.05, REWARD) == pytest.approx(0.5 * np.exp(-1))
    assert close_reward(True, False, 0.0, REWARD) == 0.0  # only the first good close pays
    assert close_reward(False, True, 0.0, REWARD) == close_reward(False, False, 0.0, REWARD) == -0.1


# ---- environment: scripted episodes ------------------------------------------------------------------------


def a_policy(env: GraspEnv, A: ReactiveQPController) -> np.ndarray:
    """(A) at 10 Hz: its joint velocity as the target, its gripper command open/closed."""
    st = env.ctrl.state
    v, grip = A.step(st.t, st.q, st.X_WG, env.world, env.cfg["env"]["dt"])
    return np.concatenate([np.clip(v / env.robot.v_limits, -1, 1), [1.0 if grip > 0.5 * WSG_OPEN else -1.0]])


def run_scripted(env, policy, max_steps=400):
    steps = []
    for _ in range(max_steps):
        phi_before = env.phi_p, env.phi_c
        obs, r, term, trunc, info = env.step(policy())
        steps.append((phi_before, (env.phi_p, env.phi_c), r, info))
        assert not trunc and np.all(np.isfinite(obs)) and obs.shape == (OBS_DIM,)
        if term:
            return steps
    raise AssertionError("episode did not end")


def test_scripted_grasp_succeeds_and_the_reward_terms_add_up():
    env = GraspEnv(seed=0)
    env.reset(options={"spec": S.no_obstacles(0.15), "level": 0})
    A = ReactiveQPController(lift_height=0.2)
    A.reset(env.world, env.robot, env.ctrl.state)
    steps = run_scripted(env, lambda: a_policy(env, A))
    info = steps[-1][3]
    assert info["outcome"] == "success" and info["is_success"] and steps[-1][3]["terms"]["terminal"] == 1.0
    assert env.h.monitor.t_success is not None and not env.h.monitor.collisions
    g, k = env.cfg["gamma"], env.cfg["reward"]["k_p"]
    for (pp, pc), (qp, qc), r, inf in steps:  # each term is its definition, and the total is their sum
        assert inf["terms"]["progress"] == pytest.approx(k * (g * qp - pp))
        assert inf["terms"]["clearance"] == pytest.approx(g * qc - pc) == 0.0  # no obstacles
        assert r == pytest.approx(sum(inf["terms"].values()))
    terms = info["ep_terms"]
    assert info["closes"] == {"good": 1, "bad": 0} and 0.0 < terms["close"] <= 0.5
    # discounted, the progress terms telescope to k_p x (start potential - 0): Phi = 0 once in hand
    phi0 = steps[0][0][0]
    assert steps[-1][1][0] == 0.0 and 0.5 < -k * phi0 < 1.5
    assert sum(g**t * s[3]["terms"]["progress"] for t, s in enumerate(steps)) == pytest.approx(-k * phi0)
    assert terms["terminal"] == 1.0 and info["ep_len"] == len(steps)
    assert sum(s[2] for s in steps) == pytest.approx(sum(terms.values()))


def test_collision_ends_the_episode_with_minus_one():
    env = GraspEnv(seed=0)
    env.reset(options={"spec": S.no_obstacles(0.1), "level": 0})
    a = np.zeros(8)
    a[1], a[7] = 1.0, 1.0  # shoulder forward at full speed: down into the conveyor
    steps = run_scripted(env, lambda: a, 100)
    info = steps[-1][3]
    assert info["outcome"] == "collision" and info["terms"]["terminal"] == -1.0
    assert env.h.monitor.collisions[0].category in ("cart", "floor")


def test_drop_ends_the_episode_with_minus_one():
    env = GraspEnv(seed=0)
    env.reset(options={"spec": S.no_obstacles(0.15), "level": 0})
    A = ReactiveQPController(lift_height=0.2)
    A.reset(env.world, env.robot, env.ctrl.state)

    def policy():
        a = a_policy(env, A)
        if env.h.monitor.t_lift is not None:  # lifted: let go before the hold time
            a[7] = 1.0
        return a

    steps = run_scripted(env, policy)
    info = steps[-1][3]
    assert info["outcome"] == "dropped" and info["terms"]["terminal"] == -1.0
    assert env.h.monitor.t_success is None and env.h.monitor.dropped


def test_timeout_and_repeated_bad_closes():
    spec = S.no_obstacles(0.3)
    spec.duration = 1.0
    env = GraspEnv(seed=0)
    env.reset(options={"spec": spec, "level": 5})
    grip = iter([-1, 1, -1, 1, -1, 1, 1, 1, 1, 1])  # three closes, far from the object
    steps = run_scripted(env, lambda: np.array([0, 0, 0, 0, 0, 0, 0, next(grip)], float), 20)
    info = steps[-1][3]
    assert info["outcome"] == "timeout" and info["terms"]["terminal"] == 0.0 and len(steps) == 10
    assert info["closes"] == {"good": 0, "bad": 3} and info["ep_terms"]["close"] == pytest.approx(-0.3)
    assert [s[3]["terms"]["close"] for s in steps[:6]] == [-0.1, 0.0, -0.1, 0.0, -0.1, 0.0]


def test_ticks_command_the_ramp_one_tick_late():
    env = GraspEnv(seed=0)
    env.reset(options={"spec": S.no_obstacles(0.1), "level": 0})
    q0 = env.ctrl.proc.q.copy()
    a = np.array([0.3, -0.2, 0.1, 0.2, -0.3, 0.4, -0.5, 1.0])
    env.step(a)
    Q1 = env.ctrl.proc.Q.copy()
    env.step(-a)
    Q2 = env.ctrl.proc.Q.copy()
    log = np.array(env.h.controller_system.q_cmd_log)
    assert len(log) == 21 and env.ctrl.state.t == pytest.approx(0.2)  # ticks 0, 0.01, ..., 0.2
    np.testing.assert_array_equal(log[0], q0)
    np.testing.assert_array_equal(log[1:11], Q1)
    np.testing.assert_array_equal(log[11:21], Q2)
    assert not np.allclose(Q1[-1], Q2[0])


# ---- observation ---------------------------------------------------------------------------------------------


def obstacle_blocks(obs):
    start = OBS_DIM - 2 - 11 * N_OBSTACLES
    return obs[start:start + 11 * N_OBSTACLES].reshape(N_OBSTACLES, 11)


@pytest.mark.parametrize("which", ["none", "pillar", "four"])
def test_observation_shape_values_and_padding(which):
    rng = np.random.default_rng(3)
    spec = {"none": S.no_obstacles(0.1), "pillar": S.pillar_and_blocker()}.get(which)
    while spec is None:
        s = R.sample_training_scene(3, rng)
        spec = s if len(s.obstacles) == 4 else None
    n = len(spec.obstacles)
    assert n == {"none": 0, "pillar": 2, "four": 4}[which]
    env = GraspEnv(seed=0)
    obs, _ = env.reset(options={"spec": spec, "level": 0})
    for _ in range(3):
        obs, *_ = env.step(np.array([0.5, 0.3, 0, -0.3, 0, 0.2, 0, 1.0]))
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32 and np.all(np.isfinite(obs)) and np.abs(obs).max() < 5
    B = obstacle_blocks(obs)
    assert np.all(B[:n, 10] == 1.0) and np.all(B[n:] == 0.0)
    assert np.all(np.diff(B[:n, 9]) >= 0)  # nearest first


def test_training_and_evaluation_see_the_same_observations():
    """The env's observations (training) equal what RLController computes in a harness episode (evaluation)."""
    spec = S.tall_blocker(0.2, 0.3)
    spec.duration = 1.5
    rng = np.random.default_rng(0)
    actions = rng.uniform(-0.3, 0.3, (15, 8))
    actions[:, 7] = 1.0
    env = GraspEnv(seed=0)
    obs, _ = env.reset(options={"spec": spec, "level": 0})
    train_obs = [obs]
    for a in actions[:14]:
        obs, *_ = env.step(a)
        train_obs.append(obs)

    seen = []

    def replay(o):
        seen.append(o)
        return actions[len(seen) - 1]

    run_episode(spec, RLController(actor=replay), latency=env.cfg["env"]["latency"], keep_log=False)
    assert len(seen) == 15
    np.testing.assert_allclose(np.array(seen), np.array(train_obs), rtol=0, atol=1e-6)


# ---- curriculum ------------------------------------------------------------------------------------------------


def test_curriculum_advances_on_a_full_window_at_the_current_level():
    c = Curriculum(6, window=10, advance_at=0.5)
    for _ in range(9):
        assert not c.record(0, "success")  # window not full yet
    for _ in range(20):
        assert not c.record(1, "success")  # other levels do not count
    assert c.record(0, "collision") and c.level == 1  # full window, 9/10 >= 0.5
    for o in ["timeout"] * 5 + ["success"] * 4:
        assert not c.record(1, o)  # the window restarts at the new level
    assert not c.record(1, "dropped") and c.level == 1  # 4/10 < 0.5
    assert c.record(1, "success") and c.level == 2  # last 10: 5/10
    assert c.rates(1)["success"] == pytest.approx(0.5) and c.rates(0)["collision"] == pytest.approx(0.1)
    d = c.state_dict()
    c2 = Curriculum(6, window=10, advance_at=0.5)
    c2.load_state_dict(d)
    assert c2.state_dict() == d
    c.level = 5
    for _ in range(30):
        assert not c.record(5, "success") and c.level == 5  # the last level stays


def test_level_sampling_mixes_earlier_levels():
    rng = np.random.default_rng(0)
    assert all(sample_level(0, rng, 0.7) == 0 for _ in range(100))
    x = np.array([sample_level(3, rng, 0.7) for _ in range(20_000)])
    assert np.mean(x == 3) == pytest.approx(0.7, abs=0.015) and x.max() == 3
    for k in range(3):
        assert np.mean(x == k) == pytest.approx(0.1, abs=0.01)


# ---- scenes ---------------------------------------------------------------------------------------------------


def test_evaluation_scenes_and_seed_ranges():
    ev = R.eval_scenes()
    kinds = [s.name.split("_v")[0] if "blocker" in s.name else s.name[:3] for s in ev]
    assert len(ev) == 214 and {k: kinds.count(k) for k in set(kinds)} == {
        "dev": 20, "mix": 80, "pillar_and_blocker": 18, "plank_blocker": 48, "tall_blocker": 48}
    eval_seeds = {s.seed for s in ev if s.seed is not None}
    assert eval_seeds == set(range(20)) | set(range(5000, 5080))
    val = R.validation_scenes()
    assert len(val) == 50 and all(R.VAL_SEED <= s.seed < R.TRAIN_SEED_MIN for s in val)
    assert max(eval_seeds) < R.VAL_SEED


def test_training_and_validation_scenes_are_disjoint_from_evaluation():
    ev = {R.fingerprint(s) for s in R.eval_scenes()}
    assert len(ev) == 214  # the fingerprint tells all evaluation scenes apart
    val = [R.fingerprint(s) for s in R.validation_scenes()]
    assert len(set(val)) == 50 and not set(val) & ev
    rng = np.random.default_rng(1)
    train = [R.sample_training_scene(level, rng) for level in range(R.N_LEVELS) for _ in range(25)]
    assert all(s.seed >= R.TRAIN_SEED_MIN for s in train)
    assert not {R.fingerprint(s) for s in train} & (ev | set(val))
    # the generator's version of an evaluation scene would be rejected
    tall = next(s for s in R.eval_scenes() if s.name.startswith("tall"))
    assert R.fingerprint(S.tall_blocker(0.05, 0.1, 3.5)) == R.fingerprint(tall) in R.reserved_fingerprints()


def test_reserved_scenes_are_skipped(monkeypatch):
    first, real, calls = R.eval_scenes()[0], R.make_scene, []

    def fake(level, seed):
        calls.append(seed)
        return first if len(calls) == 1 else real(level, seed)

    monkeypatch.setattr(R, "make_scene", fake)
    spec = R.sample_training_scene(0, np.random.default_rng(0))
    assert len(calls) == 2 and R.fingerprint(spec) != R.fingerprint(first)


def test_validation_file_matches_the_generator():
    val = R.validation_scenes()
    for i in range(3):
        assert R.fingerprint(R.validation_scene(i)) == R.fingerprint(val[i])


def test_levels_use_the_intended_generators():
    rng = np.random.default_rng(2)
    s = [[R.sample_training_scene(k, rng) for _ in range(20)] for k in range(R.N_LEVELS)]
    assert all(not x.obstacles and x.target.object == "soup_can" and x.cart.speed <= 0.1 for x in s[0])
    assert all(not x.obstacles for x in s[1])
    assert all(1 <= len(x.obstacles) <= 2 and all(o.mount == "cart" for o in x.obstacles) for x in s[2])
    assert all(1 <= len(x.obstacles) <= 4 for x in s[3])
    assert all(any(o.name in ("blocker", "pillar") for o in x.obstacles) for x in s[4])


# ---- configuration and controller -----------------------------------------------------------------------------


def test_config_overrides():
    cfg = load_config(None, ["sac.batch_size=1024", "train.n_envs=64", "curriculum.tol_pos=[0.05,0.01]"])
    assert cfg["sac"]["batch_size"] == 1024 and cfg["train"]["n_envs"] == 64 and cfg["curriculum"]["tol_pos"] == [0.05, 0.01]
    assert DEFAULTS["sac"]["batch_size"] == 256
    with pytest.raises(KeyError):
        load_config(None, ["sac.batch=1"])


def test_numpy_actor_matches_sbx():
    import gymnasium as gym
    from sbx import SAC

    class Spaces(gym.Env):
        observation_space = GraspEnv.observation_space
        action_space = GraspEnv.action_space

        def reset(self, seed=None, options=None):
            return np.zeros(OBS_DIM, np.float32), {}

        def step(self, a):
            return np.zeros(OBS_DIM, np.float32), 0.0, False, False, {}

    model = SAC("MlpPolicy", Spaces(), policy_kwargs={"net_arch": [64, 64, 64], "layer_norm": True}, seed=0)
    actor = NumpyActor(NumpyActor.arrays_from_sbx(model.policy.actor_state.params))
    obs = np.random.default_rng(0).standard_normal((32, OBS_DIM)).astype(np.float32)
    ref, _ = model.predict(obs, deterministic=True)
    np.testing.assert_allclose(np.array([actor(o) for o in obs]), ref, atol=1e-5)
    assert len(actor.hidden) == 3


def test_rl_controller_runs_an_evaluation_episode_with_measured_latency():
    rng = np.random.default_rng(0)
    sizes = [OBS_DIM, 512, 512, 512]
    arrays = {f"W{i}": rng.standard_normal((a, b)) * np.sqrt(1 / a) for i, (a, b) in enumerate(itertools.pairwise(sizes))}
    arrays |= {f"b{i}": np.zeros(512) for i in range(3)}
    arrays |= {"Wm": rng.standard_normal((512, 8)) * 0.01, "bm": np.zeros(8)}
    spec = S.tall_blocker(0.1, 0.2)
    spec.duration = 3.0
    actor, calls = NumpyActor(arrays), []
    ctrl = RLController(actor=lambda o: (calls.append(o), actor(o))[1])
    r = run_episode(spec, ctrl)  # latency="measured"
    assert r.latency_mode == "measured" and r.outcome == "timeout" and r.n_cycles >= 295
    assert len(calls) == 30  # the policy runs at 10 Hz
    assert r.compute_max_ms < 50.0 and r.compute_mean_ms < 5.0


def test_post_success_episodes_go_on_to_the_scene_end_and_are_graded_as_the_harness():
    cfg = load_config(overrides=["env.post_success=true"])
    env = GraspEnv(cfg, seed=0)
    spec = S.no_obstacles(0.15)
    env.reset(options={"spec": spec, "level": 0})
    A = ReactiveQPController(lift_height=0.2)
    A.reset(env.world, env.robot, env.ctrl.state)
    steps = run_scripted(env, lambda: a_policy(env, A), 400)
    info = steps[-1][3]
    paid = [s[3]["terms"]["terminal"] for s in steps]
    assert info["outcome"] == "success" and sum(paid) == 1.0 and paid.count(1.0) == 1
    assert steps[-1][3]["t"] >= spec.duration - 1e-9 > env.h.monitor.t_success
    # letting go after success: dropped at the end, terminal rewards add up to the drop's
    env.reset(options={"spec": spec, "level": 0})
    A.reset(env.world, env.robot, env.ctrl.state)

    def let_go():
        a = a_policy(env, A)
        if env.succeeded:
            a[7] = 1.0
        return a

    steps = run_scripted(env, let_go, 400)
    assert steps[-1][3]["outcome"] == "dropped" and sum(s[3]["terms"]["terminal"] for s in steps) == pytest.approx(-1.0)
