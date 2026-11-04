"""The wrapper's latency model: slow computation costs simulated time."""

import time

import numpy as np

from dyngrasp import scenario as S
from dyngrasp.controller import WSG_OPEN, Command, Controller
from dyngrasp.harness import run_episode
from dyngrasp.trajectory import JointTrajectory


def short_spec(duration=1.0):
    spec = S.no_obstacles(0.3)
    spec.duration = duration
    return spec


class SlowPlanner(Controller):
    """First call: think for `think` s, then send a trajectory moving joint 1.  Records what it sees."""

    name = "slow"
    period = 0.01

    def __init__(self, think=0.2, from_now=True):
        self.think, self.from_now = think, from_now
        self.seen = []

    def reset(self, world, robot, state):
        self.calls = 0

    def update(self, state, world):
        self.calls += 1
        self.seen.append((state.t, state.trajectory is not None))
        if self.calls == 1:
            time.sleep(self.think)
            q0 = state.q_cmd
            t0 = state.t if self.from_now else state.t + 0.05
            q1 = q0 + np.array([0.3, 0, 0, 0, 0, 0, 0])
            traj = JointTrajectory([t0, t0 + 0.5, t0 + 0.6], [q0, q1, q1])
            return Command(trajectory=traj, gripper=WSG_OPEN)
        return Command(trajectory=state.trajectory, gripper=WSG_OPEN) if state.trajectory else Command(q=state.q_cmd)


def test_measured_latency_delays_commands_by_the_compute_time():
    c = SlowPlanner(think=0.2)
    r = run_episode(short_spec(), c, latency="measured")
    # the controller is busy for ~0.2 s of sim time: no further calls, and no trajectory before then
    t_second_call = c.seen[1][0]
    assert 0.19 <= t_second_call <= 0.3, t_second_call
    assert r.delay_max_ms >= 190 and r.latency_mode == "measured"
    # the trajectory starts at t = 0 at q0, so taking effect at 0.2 s it would jump: it is
    # rejected, and the arm keeps holding still
    assert r.n_rejected_commands == 1
    assert not c.seen[1][1]
    assert abs(r.log["q"][-1][0] - S.Q_HOME[0]) < 0.01


def test_a_plan_that_starts_after_its_compute_time_is_accepted():
    # plans from t + 0.05 while thinking 0.02 s: arrives in time, the arm follows it
    c = SlowPlanner(think=0.02, from_now=False)
    r = run_episode(short_spec(), c, latency="measured")
    assert r.n_rejected_commands == 0
    assert r.log["q"][-1][0] > 0.25  # joint 1 moved by ~0.3 rad


def test_no_latency_is_the_old_behaviour():
    c = SlowPlanner(think=0.2)
    r = run_episode(short_spec(), c, latency="none")
    assert c.seen[1][0] <= 0.011 and r.n_rejected_commands == 0 and r.delay_max_ms == 0.0


def test_fixed_latency_is_deterministic():
    c = SlowPlanner(think=0.0, from_now=False)
    r = run_episode(short_spec(), c, latency=0.04)
    assert abs(c.seen[1][0] - 0.04) < 1e-6 and abs(r.delay_max_ms - 40.0) < 1e-6
    assert r.n_rejected_commands == 0  # applied at 0.04 <= start of the plan (0.05)
