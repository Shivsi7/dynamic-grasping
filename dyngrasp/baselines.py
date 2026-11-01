"""Reference controllers.

* `HoldController`     -- stays at q0.  Exercises the harness: obstacles that are
                          scripted to cross the corridor must *not* hit a parked
                          robot (scenario sanity), the target must pass by, etc.
* `NaiveInterceptController` -- differential IK (damped pseudo-inverse) tracking
                          a velocity-matched top-down grasp on the predicted
                          target, with no obstacle awareness at all.  This is the
                          "no avoidance" lower bound that (A) and (B) must beat,
                          and it validates grasp/lift success detection.
"""

from __future__ import annotations

import numpy as np
from pydrake.all import RigidTransform, RotationMatrix

from dyngrasp.controller import NQ, WSG_CLOSED, WSG_OPEN, Command, Controller, RobotModel, RobotState
from dyngrasp.world import WorldModel


class HoldController(Controller):
    name = "hold"
    period = 0.01

    def reset(self, world, robot, state):
        self.q0 = state.q.copy()

    def update(self, state, world):
        return Command(q=self.q0, gripper=WSG_OPEN, info={"phase": "hold"})


def pose_error(X_WG: RigidTransform, X_WG_des: RigidTransform) -> np.ndarray:
    """Spatial error [rotation (axis*angle); translation] in W taking X_WG to X_WG_des."""
    R_err = X_WG_des.rotation() @ X_WG.rotation().inverse()
    aa = R_err.ToAngleAxis()
    return np.concatenate([aa.axis() * aa.angle(), X_WG_des.translation() - X_WG.translation()])


def damped_pinv(J: np.ndarray, damping: float = 0.05) -> np.ndarray:
    return J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(J.shape[0]))


class NaiveInterceptController(Controller):
    """Track the predicted grasp pose with velocity feedforward; close; lift.

    Phases: wait -> pregrasp (hover `hover` above the grasp) -> descend -> close
    -> lift.  Uses the nominal grasp frame `world.object.X_UG_grasp`.  No
    obstacle avoidance, no joint-limit awareness beyond clipping velocities.
    """

    name = "naive"
    period = 0.01

    def __init__(self, hover: float = 0.10, k_p: float = 4.0, k_rot: float = 4.0, v_scale: float = 0.6,
                 lead_time: float = 1.0, close_time: float = 0.4, lift_height: float = 0.15):
        self.hover = hover
        self.k_p = k_p
        self.k_rot = k_rot
        self.v_scale = v_scale  # fraction of the joint velocity limits
        self.lead_time = lead_time  # start moving this long before the target is reachable
        self.close_time = close_time
        self.lift_height = lift_height

    def reset(self, world: WorldModel, robot: RobotModel, state: RobotState):
        self.robot = robot
        self.phase = "wait"
        self.t_close = None
        self.p_lift_ref = None

    def _track(self, state: RobotState, X_WG_des: RigidTransform, v_ff: np.ndarray) -> np.ndarray:
        err = pose_error(state.X_WG, X_WG_des)
        V_des = np.concatenate([self.k_rot * err[:3], self.k_p * err[3:] + v_ff])
        # Limit the Cartesian speed so a far-away target does not saturate the joints.
        speed = np.linalg.norm(V_des[3:])
        if speed > 1.0:
            V_des[3:] *= 1.0 / speed
        J = self.robot.jacobian(state.q)
        v = damped_pinv(J) @ V_des
        vmax = self.v_scale * self.robot.v_limits
        scale = np.max(np.abs(v) / vmax)
        if scale > 1.0:
            v /= scale
        return v

    def update(self, state: RobotState, world: WorldModel) -> Command:
        t = state.t
        X_WG_grasp = world.grasp_pose(t)
        v_target = world.target_velocity(t)
        above = RigidTransform(X_WG_grasp.rotation(), X_WG_grasp.translation() + [0, 0, self.hover])

        if self.phase == "wait":
            if t >= world.t_enter - self.lead_time:
                self.phase = "pregrasp"
            return Command(q=state.q_cmd, gripper=WSG_OPEN, info={"phase": self.phase})

        if self.phase == "pregrasp":
            v = self._track(state, above, v_target)
            err = pose_error(state.X_WG, above)
            if world.target_reachable(t) and np.linalg.norm(err[3:]) < 0.02 and np.linalg.norm(err[:3]) < 0.1:
                self.phase = "descend"
            return Command(v=v, gripper=WSG_OPEN, info={"phase": self.phase})

        if self.phase == "descend":
            v = self._track(state, X_WG_grasp, v_target)
            err = pose_error(state.X_WG, X_WG_grasp)
            if np.linalg.norm(err[3:]) < 0.02:
                self.phase = "close"
                self.t_close = t
            return Command(v=v, gripper=WSG_OPEN, info={"phase": self.phase})

        if self.phase == "close":
            v = self._track(state, X_WG_grasp, v_target)  # keep riding along while closing
            if t - self.t_close > self.close_time:
                self.phase = "lift"
                self.p_lift_ref = state.X_WG.translation().copy()
            return Command(v=v, gripper=WSG_CLOSED, info={"phase": self.phase})

        # lift: go straight up, still matching the cart's velocity horizontally.
        self.p_lift_ref = self.p_lift_ref + v_target * self.period
        p_des = self.p_lift_ref + [0, 0, self.lift_height]
        X_des = RigidTransform(X_WG_grasp.rotation(), p_des)
        v = self._track(state, X_des, v_target)
        return Command(v=v, gripper=WSG_CLOSED, info={"phase": self.phase})
