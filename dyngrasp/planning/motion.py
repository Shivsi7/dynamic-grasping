"""From planner knots to the trajectory the arm executes, and the checks it must pass.

The RRT returns knots (T, Q) joined by straight space-time edges.  The arm executes a
C1 cubic Hermite curve through the same knots at the same times (`trajectory.hermite`;
space-time plans cannot be retimed, their timing is what avoids the moving obstacles),
sampled every controller period.  That curve deviates a little from the straight edges,
so before it is used it is verified as executed (`MotionVerifier.verify`):

  position limits   the curve can overshoot a knot near a joint limit, and the arm would
                    clip it there (a velocity step)
  velocity          0.95x the iiwa's limits
  acceleration      the given per-joint limits (+5 % for finite differencing)
  torque            inverse dynamics of iiwa + gripper within `torque_margin` of the limits
  collisions        the space-time checker, at its time resolution
"""

from __future__ import annotations

import numpy as np

from dyngrasp.controller import RobotModel
from dyngrasp.planning.collision import SpaceTimeChecker
from dyngrasp.trajectory import JointTrajectory, hermite, motion_limits, strided_indices


class MotionVerifier:
    def __init__(self, robot: RobotModel, checker: SpaceTimeChecker, period: float,
                 smooth: bool = True, torque_check: bool = True, torque_margin: float = 0.9):
        self.robot, self.checker, self.period = robot, checker, period
        self.smooth_on = smooth  # False: execute the polyline (no acceleration limits)
        self.torque_check, self.torque_margin = torque_check, torque_margin

    def within_joint_limits(self, traj: JointTrajectory) -> bool:
        return bool(np.all(traj.q >= self.robot.q_lower - 1e-6) and np.all(traj.q <= self.robot.q_upper + 1e-6))

    def smooth(self, T, Q, v0, vN, gripper: float) -> JointTrajectory:
        """The executed trajectory through the knots: the C1 cubic Hermite curve with boundary
        velocities v0 / vN (None = rest), or the polyline if smoothing is off."""
        g = np.full(len(T), gripper)
        if not self.smooth_on:
            return JointTrajectory(T, Q, g)
        return hermite(T, Q, v0, vN, dt=self.period, gripper=g)

    def smooth_verified(self, T, Q, v0, vN, gripper, v_lim, a_lim, max_seg: float = 0.4):
        """(trajectory, "" or the failed check).  If the curve fails on position limits or
        collisions, retry once with the long segments split (same straight-line path, a knot at
        least every `max_seg` s): a single long cubic that must end at a given velocity (the
        grasp tail's) can bulge tenths of a radian away from its edge; short segments keep the
        deviation local."""
        path = self.smooth(T, Q, v0, vN, gripper)
        why = self.verify(path, v_lim, a_lim)
        if why in ("position", "collision") and self.smooth_on:
            T2, Q2 = subdivide(np.asarray(T, float), np.asarray(Q, float), max_seg)
            if len(T2) > len(T):
                path2 = self.smooth(T2, Q2, v0, vN, gripper)
                if not self.verify(path2, v_lim, a_lim):
                    return path2, ""
        return path, why

    def verify(self, path: JointTrajectory, v_lim, a_lim) -> str:
        """"" if `path` (densely sampled) passes every check, else the name of the first failure."""
        if len(path.t) < 2:
            return ""
        if not self.within_joint_limits(path):
            return "position"
        v, a = motion_limits(path)
        # The curve may exceed the RRT's (conservative) edge speed a little between knots; what
        # matters is the arm's own limit, checked with or without acceleration limits.
        if np.any(v > 0.95 * self.robot.v_limits):
            return "velocity"
        if a_lim is not None and np.any(a > 1.05 * a_lim + 1e-6):
            return "acceleration"
        if self.torque_check and not self.torques_ok(path):
            return "torque"
        idx = strided_indices(len(path.t), round(self.checker.dt_res / max(path.t[1] - path.t[0], 1e-6)))
        if not self.checker.path_free(path.q[idx], path.t[idx]):
            return "collision"
        return ""

    def torques_ok(self, path: JointTrajectory, dt: float = 0.02) -> bool:
        """Every joint torque along the path (sampled every `dt`) within `torque_margin` of
        the iiwa's limits: tau = M(q) qdd + C(q, qd) qd - tau_gravity(q)."""
        ts = np.arange(path.t[0], path.t[-1] + 1e-9, dt)
        if len(ts) < 3:
            return True
        tau = self.robot.joint_torques(np.array([path.value(t) for t in ts]), dt)
        return bool(np.all(np.abs(tau) <= self.torque_margin * self.robot.tau_limits))


def subdivide(T: np.ndarray, Q: np.ndarray, max_seg: float) -> tuple[np.ndarray, np.ndarray]:
    """The same polyline with extra knots, so that no segment is longer than `max_seg` s."""
    Ts, Qs = [T[0]], [Q[0]]
    for i in range(len(T) - 1):
        n = max(1, int(np.ceil((T[i + 1] - T[i]) / max_seg)))
        for s in np.arange(1, n + 1) / n:
            Ts.append(T[i] + s * (T[i + 1] - T[i]))
            Qs.append(Q[i] + s * (Q[i + 1] - Q[i]))
    return np.array(Ts), np.array(Qs)
