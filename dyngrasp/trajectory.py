"""Time-indexed joint trajectories that the controller wrapper executes.

A planner returns a `JointTrajectory` (in `Command.trajectory`) instead of a
single set-point: the wrapper samples it at every tick, so the robot keeps
moving along the last accepted plan while the planner is still computing the
next one (see `ControllerSystem` and its latency model).

q(t) is piecewise linear between knots and held constant outside [t[0], t[-1]].
The gripper opening is piecewise constant: gripper[i] applies on [t[i], t[i+1]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class JointTrajectory:
    t: np.ndarray  # (N,) strictly increasing knot times
    q: np.ndarray  # (N, nq) knot positions
    gripper: np.ndarray | None = None  # (N,) opening from each knot on; None = keep Command.gripper

    def __post_init__(self):
        self.t = np.asarray(self.t, float).reshape(-1)
        self.q = np.asarray(self.q, float).reshape(len(self.t), -1)
        if self.gripper is not None:
            self.gripper = np.asarray(self.gripper, float).reshape(-1)
            assert len(self.gripper) == len(self.t)
        assert len(self.t) >= 1
        assert np.all(np.diff(self.t) > 0), "knot times must be strictly increasing"

    @property
    def start_time(self) -> float:
        return float(self.t[0])

    @property
    def end_time(self) -> float:
        return float(self.t[-1])

    def value(self, t: float) -> np.ndarray:
        if t <= self.t[0]:
            return self.q[0].copy()
        if t >= self.t[-1]:
            return self.q[-1].copy()
        i = int(np.searchsorted(self.t, t, side="right")) - 1
        s = (t - self.t[i]) / (self.t[i + 1] - self.t[i])
        return (1.0 - s) * self.q[i] + s * self.q[i + 1]

    def velocity(self, t: float) -> np.ndarray:
        if t < self.t[0] or t >= self.t[-1] or len(self.t) < 2:
            return np.zeros(self.q.shape[1])
        i = int(np.searchsorted(self.t, t, side="right")) - 1
        return (self.q[i + 1] - self.q[i]) / (self.t[i + 1] - self.t[i])

    def gripper_at(self, t: float) -> float | None:
        if self.gripper is None:
            return None
        i = int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1))
        return float(self.gripper[i])

    def max_joint_speed_ratio(self, v_max: np.ndarray) -> float:
        """max over segments and joints of |dq/dt| / v_max (<= 1 means within limits)."""
        if len(self.t) < 2:
            return 0.0
        rates = np.abs(np.diff(self.q, axis=0)) / np.diff(self.t)[:, None]
        return float(np.max(rates / np.asarray(v_max)[None, :]))

    def slice_until(self, t_end: float) -> JointTrajectory:
        """The part on [t0, t_end], with a knot inserted at t_end."""
        keep = self.t < t_end - 1e-9
        t = np.append(self.t[keep], t_end)
        q = np.vstack([self.q[keep], self.value(t_end)])
        g = None if self.gripper is None else np.append(self.gripper[keep], self.gripper_at(t_end))
        return JointTrajectory(t, q, g)

    @staticmethod
    def concatenate(parts: list[JointTrajectory]) -> JointTrajectory:
        """Join trajectories end to start (a knot shared by two parts is kept once)."""
        ts, qs, gs = [], [], []
        has_g = any(p.gripper is not None for p in parts)
        for p in parts:
            g = p.gripper if p.gripper is not None else np.full(len(p.t), np.nan)
            start = 0
            if ts and p.t[0] <= ts[-1][-1] + 1e-9:
                start = 1  # shared knot
            ts.append(p.t[start:])
            qs.append(p.q[start:])
            gs.append(g[start:])
        t = np.concatenate(ts)
        q = np.vstack(qs)
        g = np.concatenate(gs) if has_g else None
        if g is not None:  # fill parts without a gripper schedule with the previous value
            for i in range(len(g)):
                if np.isnan(g[i]):
                    g[i] = g[i - 1] if i > 0 else np.nan
            if np.isnan(g[0]):
                g = None
        return JointTrajectory(t, q, g)


def knot_velocities(T: np.ndarray, Q: np.ndarray, v0: np.ndarray, vN: np.ndarray) -> np.ndarray:
    """Velocities at the knots for a C1 curve through (T, Q): the boundary values at the ends;
    inside, per joint, 0 where the adjacent segments move in opposite directions, else the
    harmonic mean of the two segment velocities (always between them, so at interior knots
    the curve does not overshoot -- Fritsch-Carlson style).  The given boundary velocities
    are not limited this way: a segment that must end at vN can bulge past its end knot."""
    W = np.diff(Q, axis=0) / np.diff(T)[:, None]
    U = np.empty_like(Q)
    U[0], U[-1] = v0, vN
    if len(T) > 2:
        a, b = W[:-1], W[1:]
        same = a * b > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            hm = np.where(same, 2 * a * b / (a + b), 0.0)
        U[1:-1] = hm
    return U


def hermite(T, Q, v0=None, vN=None, dt: float = 0.01, gripper=None) -> JointTrajectory:
    """Sample the C1 cubic Hermite curve through the knots (T, Q) every `dt` (knots kept
    exactly, same times).  v0 / vN: velocities at the first / last knot (default rest).
    gripper: optional per-knot schedule (held constant until the next knot)."""
    T = np.asarray(T, float)
    Q = np.asarray(Q, float)
    n = Q.shape[1]
    if len(T) == 1:
        return JointTrajectory(T, Q, None if gripper is None else np.asarray(gripper, float))
    U = knot_velocities(T, Q, np.zeros(n) if v0 is None else v0, np.zeros(n) if vN is None else vN)
    ts = [T[0]]
    qs = [Q[0]]
    gs = [] if gripper is None else [gripper[0]]
    for k in range(len(T) - 1):
        h = T[k + 1] - T[k]
        m = max(1, int(np.ceil(h / dt - 1e-9)))
        s = np.arange(1, m + 1) / m
        h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
        h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
        P = (h00[:, None] * Q[k] + h10[:, None] * h * U[k] + h01[:, None] * Q[k + 1] + h11[:, None] * h * U[k + 1])
        P[-1] = Q[k + 1]
        ts.extend(T[k] + s * h)
        qs.extend(P)
        if gripper is not None:
            gs.extend([gripper[k]] * (m - 1) + [gripper[k + 1]])
    return JointTrajectory(np.array(ts), np.array(qs), None if gripper is None else np.array(gs))


def motion_limits(traj: JointTrajectory) -> tuple[np.ndarray, np.ndarray]:
    """Peak |velocity| and |acceleration| per joint of a densely sampled trajectory
    (finite differences between its knots)."""
    if len(traj.t) < 3:
        return np.zeros(traj.q.shape[1]), np.zeros(traj.q.shape[1])
    dt = np.diff(traj.t)
    v = np.diff(traj.q, axis=0) / dt[:, None]
    a = np.diff(v, axis=0) / (0.5 * (dt[1:] + dt[:-1]))[:, None]
    return np.max(np.abs(v), axis=0), np.max(np.abs(a), axis=0)
