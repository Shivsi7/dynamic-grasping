"""Evasive motion primitives, and the post-grasp retreat built from them.

An escape primitive is one smooth curve (a single cubic Hermite segment) from the current
state (q, v) to a target configuration at rest.  The targets are tried fastest first:
brake in place, go home, swing the base / shoulder / elbow away by small and large amounts,
fold the arm in, and random nearby configurations.  A primitive is accepted if it passes
the MotionVerifier checks and holding its target afterwards is collision-free for
`hold_horizon`.  They are much cheaper than the random-tree escape (`st_rrt.plan_to_safe_state`),
which the controller uses only when no primitive works.

The retreat: when a grasp tail ends somewhere that is not safe to stay (an obstacle is on
its way), the arm leaves right after the lift instead of holding still: the tail up to the
end of the lift, then an escape primitive with the object in hand, kept lifted.
"""

from __future__ import annotations

import time

import numpy as np

from dyngrasp.controller import RobotModel
from dyngrasp.planning.collision import SpaceTimeChecker
from dyngrasp.planning.motion import MotionVerifier
from dyngrasp.trajectory import JointTrajectory, hermite

DURATIONS = (0.6, 0.9, 1.2, 1.6)  # primitive durations (s), fastest first


def _offsets(rng: np.random.Generator) -> list[np.ndarray]:
    """Joint offsets of the "move away" targets, relative to the current configuration."""
    out = []
    # single joints: base, shoulder, elbow (the links that meet obstacles), small and large
    for j, mags in ((0, (0.5, 1.0)), (1, (0.4, 0.8, 1.0)), (3, (0.4, 0.8, 1.0))):
        for d in mags:
            for sgn in (1.0, -1.0):
                e = np.zeros(7)
                e[j] = sgn * d
                out.append(e)
    # shoulder back + elbow folded, either way (pulls the whole arm in)
    for s1 in (1.0, -1.0):
        for s3 in (1.0, -1.0):
            e = np.zeros(7)
            e[1], e[3] = s1 * 0.6, s3 * 0.6
            out.append(e)
    return out + [rng.normal(0.0, 0.6, 7) for _ in range(10)]


class EscapePlanner:
    def __init__(self, robot: RobotModel, checker: SpaceTimeChecker, verifier: MotionVerifier,
                 rng: np.random.Generator, q_home: np.ndarray, period: float, joint_margin: float,
                 hold_horizon: float):
        self.robot, self.checker, self.verifier, self.rng = robot, checker, verifier, rng
        self.q_home = np.asarray(q_home, float)
        self.period, self.hold_horizon = period, hold_horizon
        self.lo, self.hi = robot.q_lower + joint_margin, robot.q_upper - joint_margin

    def primitive(self, q_s, v_s, t_s: float, gripper: float, deadline: float, v_lim, a_lim,
                  accept=None) -> JointTrajectory | None:
        """The first escape primitive from (q_s, v_s) at t_s that passes the checks (and
        `accept(path)`, if given), or None if there is none or the wall clock passes `deadline`
        (a time.perf_counter() value)."""
        v_s = np.zeros_like(q_s) if v_s is None else v_s
        targets = [("brake", None), ("home", self.q_home)] + [("offset", q_s + o) for o in _offsets(self.rng)]
        for D in DURATIONS:
            for kind, qt in targets:
                if time.perf_counter() >= deadline:
                    return None
                if kind == "brake":
                    qt = q_s + 0.5 * v_s * D  # constant deceleration to rest
                qt = np.clip(qt, self.lo, self.hi)
                path = hermite([t_s, t_s + D], [q_s, qt], v_s, None, dt=self.period, gripper=[gripper, gripper])
                if self.verifier.verify(path, v_lim, a_lim):
                    continue
                if self.checker.hold_free(qt, t_s + D, self.hold_horizon) and (accept is None or accept(path)):
                    return path
        return None

    def retreat(self, tail: JointTrajectory, t_hold: float, limits: list, budget: float,
                max_drop: float = 0.03) -> JointTrajectory | None:
        """`tail` up to the end of its lift (`t_hold` before its end), then an escape primitive
        to a pose that is safe to hold, keeping the gripper at most `max_drop` m below its height
        at the end of the lift (the object stays lifted).  `limits`: (v_lim, a_lim) pairs, tried
        in order, each with `budget` s of wall time.  None if no retreat is found."""
        t_split = tail.t[-1] - t_hold
        q_s, v_s, g = tail.value(t_split), tail.velocity(t_split), tail.gripper_at(t_split)
        z0 = self.robot.fk(q_s).translation()[2]

        def keeps_lifted(path: JointTrajectory) -> bool:
            idx = np.unique(np.append(np.arange(0, len(path.t), 10), len(path.t) - 1))
            return all(self.robot.fk(path.q[i]).translation()[2] >= z0 - max_drop for i in idx)

        for v_lim, a_lim in limits:
            prim = self.primitive(q_s, v_s, t_split, g, time.perf_counter() + budget, v_lim, a_lim,
                                  accept=keeps_lifted)
            if prim is not None:
                return JointTrajectory.concatenate([tail.slice_until(t_split), prim])
        return None
