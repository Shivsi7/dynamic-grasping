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


def test_hermite_keeps_the_knots_and_is_c1():
    from dyngrasp.trajectory import hermite

    T = np.array([0.0, 0.4, 1.0, 1.3])
    Q = np.array([[0.0, 0.0], [0.3, -0.2], [0.1, -0.1], [0.4, 0.2]])
    v0, vN = np.array([0.2, 0.0]), np.zeros(2)
    tr = hermite(T, Q, v0, vN, dt=0.001)
    for t, q in zip(T, Q):  # same space-time knots
        np.testing.assert_allclose(tr.value(t), q, atol=1e-9)
    v = np.diff(tr.q, axis=0) / np.diff(tr.t)[:, None]
    np.testing.assert_allclose(v[0], v0, atol=0.03)
    np.testing.assert_allclose(v[-1], vN, atol=0.03)
    # no velocity jump at the interior knots: left and right velocities agree (C1), whereas
    # the polyline through the same knots jumps by up to ~1 rad/s there
    eps = 1e-4
    for t in T[1:-1]:
        vl = (tr.value(t) - tr.value(t - eps)) / eps
        vr = (tr.value(t + eps) - tr.value(t)) / eps
        assert np.max(np.abs(vl - vr)) < 0.05


def test_acceleration_aware_plan_is_smooth_and_joins_a_moving_start():
    from dyngrasp.trajectory import hermite, motion_limits

    free = lambda q, t: not (0.4 <= q[0] <= 0.6 and t < 3.0)
    v = np.array([1.0])
    pl = SpaceTimeRRTConnect([0.0], [1.0], v, edge_checker(free), STRRTParams(step_q=0.2, step_t=0.5),
                             np.random.default_rng(1))
    pl.a_max = np.array([4.0])
    v_start = np.array([0.15])  # already moving when the plan starts
    gq, gt = np.array([[1.0]]), np.array([6.0])
    res = pl.plan(np.array([0.0]), 0.0, gq, gt, time_budget=3.0, v_start=v_start)
    assert_valid(res, free, v, [0.0], 0.0, gq, gt)
    Q, T = pl.shortcut(res.path_q, res.path_t, 0.5, v_start=v_start)
    tr = hermite(T, Q, v_start, None, dt=0.002)
    _, amax = motion_limits(tr)
    assert amax[0] <= 4.0 * 1.02
    np.testing.assert_allclose((tr.q[1] - tr.q[0]) / (tr.t[1] - tr.t[0]), v_start, atol=0.01)
    for t, q in zip(T, Q):  # the smoothed curve is still free (it passes through the knots,
        assert free(tr.value(t), t)  # and between them it is checked by the controller)
