"""(A) Reactive controller: differential IK as a QP with velocity-damper collision constraints.

Every period (10 ms) it solves, with Drake's MathematicalProgram and OSQP,

    min_{v, s}  |J v - V_des|^2 + lam |v|^2 + mu |v - v_prev|^2 + rho1 sum(s) + rho2 |s|^2
    s.t.        n_i^T J_i v + s_i >= -xi (phi_i - d_s) + n_i^T v_world,i     for every close pair i
                joint velocity limits, damped joint position limits, s >= 0

V_des tracks a velocity-matched gripper pose (P control on the pose error plus
the target's velocity as feedforward).  The pairs i are the robot at its
current q against the world's *predicted* poses at t, t + h/2 and t + h
(h = 0.2 s): the union of those snapshots is each obstacle's swept volume over
the horizon, so the arm gets out of the way of where an obstacle is about to
be, not just where it is.  n_i^T v_world,i is the obstacle's (or the carrier's)
velocity towards the robot.  The slacks keep the QP feasible when constraints
conflict; they are penalised so heavily that they are zero otherwise.

Phases: wait -> approach (hover above the grasp) -> descend -> close -> lift.
The lift is a fixed motion, like (B)'s grasp tail: up `lift_height` in
`lift_time` (minimum jerk) while slowing down from the object's velocity to
rest, then hold that pose.
"""

from __future__ import annotations

import numpy as np
from pydrake.all import MathematicalProgram, OsqpSolver, RigidTransform

from dyngrasp.baselines import pose_error
from dyngrasp.collision import CollisionWorld
from dyngrasp.controller import (
    NQ,
    WSG_CLOSED,
    WSG_OPEN,
    Command,
    Controller,
    RobotModel,
    RobotState,
)
from dyngrasp.world import WorldModel


class ReactiveQPController(Controller):
    name = "reactive"
    period = 0.01

    def __init__(
        self,
        hover: float = 0.10,
        k_p: float = 4.0,
        k_rot: float = 4.0,
        max_speed: float = 0.8,
        v_scale: float = 0.6,
        lead_time: float = 1.0,
        close_time: float = 0.4,
        lift_height: float = 0.10,
        lift_time: float = 0.6,
        d_safe: float = 0.04,
        d_safe_env: float = 0.005,
        d_influence: float = 0.25,
        xi: float = 2.0,
        horizon: float = 0.2,
        n_horizon: int = 3,
        lam: float = 5e-3,
        mu: float = 5e-3,
    ):
        self.hover = hover
        self.k_p, self.k_rot, self.max_speed, self.v_scale = k_p, k_rot, max_speed, v_scale
        self.lead_time, self.close_time, self.lift_height = lead_time, close_time, lift_height
        self.lift_time = lift_time
        self.d_safe, self.d_safe_env, self.d_influence, self.xi = d_safe, d_safe_env, d_influence, xi
        self.horizon, self.n_horizon = horizon, n_horizon
        self.lam, self.mu = lam, mu
        self.solver = OsqpSolver()

    def reset(self, world: WorldModel, robot: RobotModel, state: RobotState, collision: CollisionWorld | None = None):
        self.robot = robot
        self.cw = collision or CollisionWorld(world.spec, world)
        self.cw.rows(state.q, state.t, self.d_influence)  # warm up the geometry engine outside the timed cycles
        self._start("wait")

    def _start(self, phase: str) -> None:
        self.phase = phase
        self.t_close = None
        self.lift_start = None  # (t, gripper position, object velocity) when the lift began
        self.v_prev = np.zeros(NQ)
        self.n_active = 0
        self.slack = 0.0

    def update(self, state: RobotState, world: WorldModel) -> Command:
        v, gripper = self.step(state.t, state.q, state.X_WG, world, self.period)
        info = {"phase": self.phase, "n_active": self.n_active, "slack": self.slack}
        return Command(v=v, gripper=gripper, info=info)

    # ---- phases ---------------------------------------------------------------
    def step(self, t: float, q: np.ndarray, X_WG: RigidTransform, world: WorldModel, dt: float):
        """(joint velocity, gripper opening) at (t, q); advances the phase machine."""
        X_grasp = world.grasp_pose(t)
        v_obj = v_ff = world.target_velocity(t)
        gripper = WSG_OPEN
        if self.phase == "wait":
            if t < world.t_enter - self.lead_time:
                return np.zeros(NQ), gripper
            self.phase = "approach"
        if self.phase == "approach":
            X_des = RigidTransform(X_grasp.rotation(), X_grasp.translation() + [0, 0, self.hover])
            err = pose_error(X_WG, X_des)
            if world.target_reachable(t) and np.linalg.norm(err[3:]) < 0.02 and np.linalg.norm(err[:3]) < 0.1:
                self.phase = "descend"
        if self.phase == "descend":
            X_des = X_grasp
            if np.linalg.norm(pose_error(X_WG, X_des)[3:]) < 0.015:
                self.phase, self.t_close = "close", t
        if self.phase == "close":
            X_des, gripper = X_grasp, WSG_CLOSED
            if t - self.t_close > self.close_time:
                self.phase, self.lift_start = "lift", (t, X_WG.translation().copy(), v_obj.copy())
        if self.phase == "lift":
            p, v_ff = self.lift_reference(t)
            X_des, gripper = RigidTransform(X_grasp.rotation(), p), WSG_CLOSED
        v = self.qp(t, q, X_WG, X_des, v_ff)
        self.v_prev = v
        return v, gripper

    def lift_reference(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Gripper position and velocity during the lift: up `lift_height` while the horizontal
        velocity goes from the object's to zero, both minimum jerk over `lift_time`; then still."""
        t0, p0, v0 = self.lift_start
        T = self.lift_time
        s = float(np.clip((t - t0) / T, 0.0, 1.0))
        sigma, dsigma = 10 * s**3 - 15 * s**4 + 6 * s**5, (30 * s**2 - 60 * s**3 + 30 * s**4) / T
        v_h = np.array([v0[0], v0[1], 0.0])
        p = p0 + v_h * T * (s - (2.5 * s**4 - 3 * s**5 + s**6)) + [0, 0, self.lift_height * sigma]
        return p, v_h * (1 - sigma) + [0, 0, self.lift_height * dsigma]

    # ---- the QP ---------------------------------------------------------------
    def qp(self, t: float, q: np.ndarray, X_WG: RigidTransform, X_des: RigidTransform, v_ff: np.ndarray) -> np.ndarray:
        err = pose_error(X_WG, X_des)
        V = np.concatenate([self.k_rot * err[:3], self.k_p * err[3:] + v_ff])
        speed = np.linalg.norm(V[3:])
        if speed > self.max_speed:
            V[3:] *= self.max_speed / speed
        J = self.robot.jacobian(q)

        prog = MathematicalProgram()
        v = prog.NewContinuousVariables(NQ, "v")
        H = J.T @ J + (self.lam + self.mu) * np.eye(NQ)
        g = J.T @ V + self.mu * self.v_prev
        prog.AddQuadraticCost(2 * H, -2 * g, v, is_convex=True)

        vmax = self.v_scale * self.robot.v_limits
        margin, tau = 0.05, 0.2  # joint-limit damper: approach a limit no faster than (gap / tau)
        lo = np.clip((self.robot.q_lower + margin - q) / tau, -vmax, vmax)
        hi = np.clip((self.robot.q_upper - margin - q) / tau, -vmax, vmax)
        prog.AddBoundingBoxConstraint(np.minimum(lo, hi), np.maximum(lo, hi), v)

        rows = []
        for tau_k in t + np.linspace(0.0, self.horizon, self.n_horizon):
            rows += self.cw.rows(q, tau_k, self.d_influence)
        self.n_active = len(rows)
        if rows:
            m = len(rows)
            s = prog.NewContinuousVariables(m, "s")
            A = np.array([r.a for r in rows])
            d_s = np.array([self.d_safe if r.category == "obstacle" else self.d_safe_env for r in rows])
            lb = -self.xi * (np.array([r.phi for r in rows]) - d_s) + np.array([r.vn for r in rows])
            prog.AddLinearConstraint(np.hstack([A, np.eye(m)]), lb, np.full(m, np.inf), np.concatenate([v, s]))
            prog.AddBoundingBoxConstraint(0.0, np.inf, s)
            prog.AddQuadraticCost(2e3 * np.eye(m), 1e2 * np.ones(m), s, is_convex=True)

        result = self.solver.Solve(prog)
        if not result.is_success():
            self.slack = np.nan
            return np.zeros(NQ)
        self.slack = float(np.sum(result.GetSolution(s))) if rows else 0.0
        return result.GetSolution(v)
