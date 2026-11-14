"""(D) action processing: a policy action -> joint position commands every tick and a gripper opening.

The action (in [-1, 1]^8) is applied once per policy step (dt = 0.1 s):

* a[:7] x v_max is the joint velocity the command should reach at the END of the step.  Within the
  step the commanded velocity ramps linearly from the current commanded velocity v to
  v_end = v + clip(target - v, -a_max dt, a_max dt) (per joint), and the commanded positions are
  the ramp's integral, sampled every controller tick (10 ms) and clipped to the joint limits minus
  `margin`.  A joint that ends the step clipped while still moving outwards is stopped (v_end = 0).
* a[7] maps linearly to the commanded WSG opening, -1 -> closed, +1 -> open, held for the step.
"""

from __future__ import annotations

import numpy as np

from dyngrasp.controller import NQ, WSG_CLOSED, WSG_OPEN


def gripper_width(a: float) -> float:
    return WSG_CLOSED + 0.5 * (float(np.clip(a, -1.0, 1.0)) + 1.0) * (WSG_OPEN - WSG_CLOSED)


def is_close_event(width_before: float, width_after: float, threshold: float) -> bool:
    """The commanded opening crosses below `threshold` (the object's width: the fingers start to squeeze it)."""
    return width_before >= threshold > width_after


class ActionProcessor:
    """The commanded joint state (q, v) and opening, advanced one policy step at a time by `start`."""

    def __init__(self, q_lower, q_upper, v_max, a_max, dt: float = 0.1, tick: float = 0.01, margin: float = 0.05):
        self.lo = np.asarray(q_lower, float) + margin
        self.hi = np.asarray(q_upper, float) - margin
        self.v_max = np.asarray(v_max, float)
        self.a_max = np.broadcast_to(np.asarray(a_max, float), (NQ,)).copy()
        self.dt = float(dt)
        self.n = round(dt / tick)
        self.reset(np.zeros(NQ))

    def reset(self, q, wsg: float = WSG_OPEN) -> None:
        self.q = np.asarray(q, float).copy()  # commanded position at the end of the current step
        self.v = np.zeros(NQ)  # commanded velocity at the end of the current step
        self.wsg = float(wsg)
        self.Q = np.tile(self.q, (self.n, 1))  # the current step's positions at ticks 1..n

    def start(self, action) -> None:
        """Begin a step from (self.q, self.v): fills self.Q, then moves (q, v, wsg) to the step's end."""
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        v0 = self.v
        dv = np.clip(a[:NQ] * self.v_max - v0, -self.a_max * self.dt, self.a_max * self.dt)
        v1 = v0 + dv
        tau = self.dt * np.arange(1, self.n + 1)[:, None] / self.n
        Q = self.q + v0 * tau + dv * tau**2 / (2.0 * self.dt)
        self.Q = np.clip(Q, self.lo, self.hi)
        stopped = ((Q[-1] > self.hi) & (v1 > 0)) | ((Q[-1] < self.lo) & (v1 < 0))
        self.q = self.Q[-1].copy()
        self.v = np.where(stopped, 0.0, v1)
        self.wsg = gripper_width(a[NQ])

    def q_at(self, k: int) -> np.ndarray:
        """Commanded position k ticks into the current step (k = 1 .. n; later ticks hold the end)."""
        return self.Q[min(max(k, 1), self.n) - 1]
