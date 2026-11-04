"""The generic space-time RRT-Connect on toy problems (no Drake)."""

import numpy as np

from dyngrasp.planning.st_rrt import (
    SpaceTimeRRTConnect,
    STRRTParams,
    min_duration,
    plan_to_safe_state,
)


def edge_checker(point_free, dt=0.01):
    def edge_free(qa, ta, qb, tb):
        n = max(2, int(np.ceil((tb - ta) / dt)) + 1, int(np.ceil(np.max(np.abs(qb - qa)) / 0.01)) + 1)
        for s in np.linspace(0, 1, n):
            if not point_free(qa + s * (qb - qa), ta + s * (tb - ta)):
                return False
        return True

    return edge_free


def assert_valid(res, point_free, v_max, q0, t0, goals_q, goals_t):
    assert res.success, res.reason
    Q, T = res.path_q, res.path_t
    np.testing.assert_allclose(Q[0], q0)
    assert T[0] == t0
    k = res.goal_index
    np.testing.assert_allclose(Q[-1], goals_q[k])
    assert abs(T[-1] - goals_t[k]) < 1e-9
    assert np.all(np.diff(T) > 0)
    for i in range(len(T) - 1):
        assert min_duration(Q[i], Q[i + 1], v_max) <= T[i + 1] - T[i] + 1e-9
    for i in range(len(T) - 1):
        assert edge_checker(point_free, dt=0.002)(Q[i], T[i], Q[i + 1], T[i + 1])


def test_waits_for_a_gate_to_open():
    # 1-D: go from 0 to 1 by t = 5; the band q in [0.4, 0.6] is blocked until t = 3.
    free = lambda q, t: not (0.4 <= q[0] <= 0.6 and t < 3.0)
    v = np.array([1.0])
    pl = SpaceTimeRRTConnect([0.0], [1.0], v, edge_checker(free), STRRTParams(step_q=0.2, step_t=0.5),
                             np.random.default_rng(1))
    gq, gt = np.array([[1.0]]), np.array([5.0])
    res = pl.plan(np.array([0.0]), 0.0, gq, gt, time_budget=2.0)
    assert_valid(res, free, v, [0.0], 0.0, gq, gt)
    # it can only cross the band after it opens
    i = np.argmax(res.path_q[:, 0] > 0.6)
    assert res.path_t[i] > 3.0


def test_dodges_a_moving_disc_in_2d_and_picks_a_reachable_goal():
    # Disc of radius 0.2 sweeping down through the middle of the unit square.
    def free(q, t):
        c = np.array([0.5, 1.2 - 0.3 * t])
        return np.linalg.norm(q - c) > 0.2

    v = np.array([0.5, 0.5])
    pl = SpaceTimeRRTConnect([0, 0], [1, 1], v, edge_checker(free), rng=np.random.default_rng(2),
                             params=STRRTParams(step_q=0.15, step_t=0.4))
    # goal roots at several times; the first one is unreachable (too early)
    gq = np.array([[1.0, 0.5]] * 4)
    gt = np.array([0.5, 2.5, 3.5, 4.5])
    res = pl.plan(np.array([0.0, 0.5]), 0.0, gq, gt, time_budget=3.0)
    assert_valid(res, free, v, [0.0, 0.5], 0.0, gq, gt)
    assert res.goal_index != 0
    # shortcutting keeps validity, endpoints and knot times
    Q, T = pl.shortcut(res.path_q, res.path_t, time_budget=0.5)
    assert len(T) <= len(res.path_t) and T[0] == res.path_t[0] and T[-1] == res.path_t[-1]
    for i in range(len(T) - 1):
        assert min_duration(Q[i], Q[i + 1], v) <= T[i + 1] - T[i] + 1e-9
        assert edge_checker(free, dt=0.002)(Q[i], T[i], Q[i + 1], T[i + 1])


def test_reports_failure_when_no_plan_exists():
    free = lambda q, t: not (0.4 <= q[0] <= 0.6 and t < 10.0)  # blocked past the goal time
    pl = SpaceTimeRRTConnect([0.0], [1.0], np.array([1.0]), edge_checker(free), rng=np.random.default_rng(0))
    res = pl.plan(np.array([0.0]), 0.0, np.array([[1.0]]), np.array([5.0]), time_budget=0.3)
    assert not res.success and res.reason in ("time budget exhausted", "tree full")


def test_escape_reaches_a_state_that_is_safe_to_hold():
    # An obstacle will sweep over q < 0.5 from t = 1 on: get to q >= 0.5 and stay.
    free = lambda q, t: not (q[0] < 0.5 and t >= 1.0)
    safe = lambda q, t: q[0] >= 0.5 and free(q, t)
    pl = SpaceTimeRRTConnect([0.0], [1.0], np.array([1.0]), edge_checker(free), rng=np.random.default_rng(0))
    res = plan_to_safe_state(pl, np.array([0.1]), 0.0, safe, t_max=3.0, time_budget=1.0)
    assert res.success and res.path_q[-1][0] >= 0.5 and res.path_t[-1] < 1.0 + 1e-9
