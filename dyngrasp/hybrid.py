"""(C) Hybrid: roll out (A) against the prediction, refine with space-time trajectory optimization.

When (A) would start moving (t_enter - lead_time) the controller

1. rolls out the reactive controller (A) kinematically against the predicted
   target and obstacle motion (`ReactiveQPController.rollout`): a joint path
   from the current configuration to the grasp, reached at t0 + T_roll;
2. refines it with Drake's `KinematicTrajectoryOptimization`: a cubic B-spline
   q(s), s in [0, 1], with free duration T and time t = t0 + s T, that
     * starts at the current state,
     * ends at the grasp pose of the object *at t0 + T*, moving with it,
     * passes the hover pose above the object at the rollout's descent time
       and comes straight down from there,
     * keeps a minimum distance >= d_safe from the obstacles (and d_safe_env
       from the cart, the floor and the target) at sample times t0 + s_k T,
   minimising duration plus path energy.  The constraints are "space-time":
   where the object and the obstacles are at sample k depends on T, and the
   gradients include that dependence (Drake's
   MinimumDistanceLowerBoundConstraint is evaluated on the full kinematic
   plant, whose obstacle/cart joints move with t);
3. tracks the result with position commands, re-solving every `replan_period`
   from the current reference, warm-started with its remainder.  With perfect
   prediction one solve suffices; re-solving only corrects prediction errors.

At t0 + T it closes and lifts with (A).  If the first optimisation fails it
falls back to (A) for the whole episode.
"""

from __future__ import annotations

import time

import numpy as np
from pydrake.all import (
    AutoDiffXd,
    BsplineTrajectory,
    KinematicTrajectoryOptimization,
    MinimumDistanceLowerBoundConstraint,
    SnoptSolver,
)

from dyngrasp.controller import NQ, WSG_OPEN, Command, Controller, RobotModel, RobotState
from dyngrasp.reactive import ReactiveQPController
from dyngrasp.world import WorldModel


def _value(x) -> float:
    return x.value() if isinstance(x, AutoDiffXd) else float(x)


def _vee(R) -> np.ndarray:
    """sin(angle) * axis of a rotation matrix (zero iff R = I near the identity)."""
    return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


class HybridController(Controller):
    name = "hybrid"
    # The station's iiwa driver differentiates the position command every 5 ms (its velocity reference);
    # a command that only changes every 10 ms makes that reference alternate between 2v and 0, and the
    # torque on the shoulder joints chatter between its limits.  Commanding at 5 ms keeps it smooth.
    period = 0.005

    def __init__(
        self,
        n_control_points: int = 12,
        n_collision_samples: int = 30,
        d_safe: float = 0.04,
        d_safe_env: float = 0.005,
        replan_period: float = 0.5,
        energy_weight: float = 0.1,
        accel_limit: float = 4.0,
        **reactive_kwargs,
    ):
        self.reactive = ReactiveQPController(d_safe=d_safe, d_safe_env=d_safe_env, **reactive_kwargs)
        self.reactive.period = self.period  # it runs inside this controller's cycle
        self.n_control_points = n_control_points
        self.n_collision_samples = n_collision_samples
        self.d_safe, self.d_safe_env = d_safe, d_safe_env
        self.replan_period = replan_period
        self.energy_weight = energy_weight
        self.accel_limit = accel_limit
        self.solver = SnoptSolver()

    # ---- episode ------------------------------------------------------------
    def reset(self, world: WorldModel, robot: RobotModel, state: RobotState):
        self.world, self.robot = world, robot
        self.reactive.reset(world, robot, state)
        self.cw = cw = self.reactive.cw
        self.min_distance = [
            MinimumDistanceLowerBoundConstraint(cw.plant, bound, cw.plant_context(cw.context(cats)))
            for cats, bound in [(("obstacle",), self.d_safe), (("cart", "floor", "target"), self.d_safe_env)]
            if any(cw._geoms[c] for c in cats)
        ]
        self.plant_ad = robot.plant.ToAutoDiffXd()
        self.ctx_ad = self.plant_ad.CreateDefaultContext()
        self.G_ad = self.plant_ad.GetFrameByName("body", self.plant_ad.GetModelInstanceByName("wsg"))
        self.phase = "wait"
        self.traj = None  # BsplineTrajectory on [0, T], started at self.t0
        self.solve_log: list[dict] = []

    def update(self, state: RobotState, world: WorldModel) -> Command:
        t = state.t
        info = {}
        if self.phase == "wait":
            if t < world.t_enter - self.reactive.lead_time:
                return Command(q=state.q_cmd, gripper=WSG_OPEN, info={"phase": "wait"})
            ts, qs, t_descend, reached = self.reactive.rollout(world, state.q_cmd, t, t_max=world.t_exit - 2.0 - t)
            self.t_descend = t_descend if t_descend is not None else ts[-1] - 0.3 * (ts[-1] - t)
            info = self._plan(t, state.q_cmd, np.zeros(NQ), ts, qs)
            info["rollout_reached"] = reached
            self.phase = "track" if info["ok"] else "fallback"
            if self.phase == "fallback":
                self.reactive._start("approach")

        if self.phase == "track":
            if t >= self.t0 + self.traj.end_time():
                self.phase = "reactive"
                self.reactive._start("close")
                self.reactive.t_close = t
            else:
                if t >= self.t_replan and t < self.t_descend - 0.2:
                    ts = np.linspace(t, self.t0 + self.traj.end_time(), 40)
                    qs = np.array([self.q_ref(tt) for tt in ts])
                    info = self._plan(t, self.q_ref(t), self.qdot_ref(t), ts, qs)
                return Command(q=self.q_ref(t), gripper=WSG_OPEN, info={"phase": "track", **info})

        cmd = self.reactive.update(state, world)
        cmd.info["phase"] = ("fallback_" if self.phase == "fallback" else "") + cmd.info["phase"]
        cmd.info.update(info)
        return cmd

    def q_ref(self, t: float) -> np.ndarray:
        return self.traj.value(np.clip(t - self.t0, 0.0, self.traj.end_time())).ravel()

    def qdot_ref(self, t: float) -> np.ndarray:
        return self.traj_dot.value(np.clip(t - self.t0, 0.0, self.traj.end_time())).ravel()

    # ---- planning ---------------------------------------------------------------
    def _plan(self, t0, q0, v0, guess_ts, guess_qs) -> dict:
        """Solve from (q0, v0) at t0, seeded with the joint path (guess_ts, guess_qs)."""
        tic = time.perf_counter()
        world = self.world
        T_lo = max(world.t_enter - t0, 0.3)
        T_hi = world.t_exit - 1.5 - t0
        T_guess = float(np.clip(guess_ts[-1] - t0, T_lo, T_hi))
        s_descend = float(np.clip((self.t_descend - t0) / T_guess, 0.3, 0.9))

        N = self.n_control_points
        trajopt = KinematicTrajectoryOptimization(NQ, N, 4)
        prog = trajopt.get_mutable_prog()
        P, T = trajopt.control_points(), trajopt.duration()
        weights = BsplineTrajectory(trajopt.basis(), [np.eye(N)[:, [j]] for j in range(N)])
        dweights = weights.MakeDerivative()
        w = lambda s: weights.value(s).ravel()
        dw = lambda s: dweights.value(s).ravel()

        trajopt.AddDurationCost(1.0)
        trajopt.AddPathEnergyCost(self.energy_weight)
        trajopt.AddDurationConstraint(T_lo, T_hi)
        margin = 0.05
        trajopt.AddPositionBounds(self.robot.q_lower + margin, self.robot.q_upper - margin)
        vmax = self.reactive.v_scale * self.robot.v_limits
        trajopt.AddVelocityBounds(-vmax, vmax)
        trajopt.AddAccelerationBounds(-self.accel_limit * np.ones(NQ), self.accel_limit * np.ones(NQ))
        trajopt.AddPathPositionConstraint(q0, q0, 0.0)
        # q'(0) = v0: r'(0) - v0 T = 0, linear in (P, T).
        prog.AddLinearEqualityConstraint(
            np.hstack([np.kron(dw(0.0), np.eye(NQ)), -v0.reshape(-1, 1)]),
            np.zeros(NQ),
            np.concatenate([P.T.ravel(), [T]]),
        )

        v_obj = world.target_velocity(world.t_enter)
        hover = self.reactive.hover

        def grasp_target(t, dz=0.0):
            """Grasp pose at (AutoDiff) time t: the cart moves at constant velocity."""
            X = world.grasp_pose(_value(t))
            return X.translation() + v_obj * (t - _value(t)) + [0, 0, dz], X.rotation().matrix()

        def at_grasp(q, qd, t):
            X, V = self._ee(q, qd)
            p, R = grasp_target(t)
            return np.concatenate([X.translation() - p, _vee(R.T @ X.rotation().matrix()),
                                   V.rotational(), V.translational() - v_obj])

        def at_hover(q, qd, t):
            X, _ = self._ee(q, qd)
            p, R = grasp_target(t, hover)
            return np.concatenate([X.translation() - p, _vee(R.T @ X.rotation().matrix())])

        def above_grasp(q, qd, t):
            X, _ = self._ee(q, qd)
            p, _ = grasp_target(t)
            return X.translation() - p

        tol_p, tol_r, tol_v = 0.005, 0.02, 0.02
        tol = np.concatenate([[tol_p] * 3, [tol_r] * 3, [tol_r] * 3, [tol_v] * 3])
        self._add(prog, at_grasp, -tol, tol, 1.0, P, T, w, dw, t0)
        self._add(prog, at_hover, -tol[:6] * 2, tol[:6] * 2, s_descend, P, T, w, dw, t0)
        for s in np.linspace(s_descend, 1.0, 5)[1:-1]:  # straight down onto the object
            self._add(prog, above_grasp, [-0.015, -0.015, 0.0], [0.015, 0.015, hover + 0.02], s, P, T, w, dw, t0)
        for s in np.linspace(0.0, 1.0, self.n_collision_samples + 1)[1:]:
            for c in self.min_distance:
                self._add(prog, self._clearance(c), c.lower_bound(), c.upper_bound(), s, P, T, w, dw, t0)

        # Initial guess: least-squares fit of the seed path on [t0, t0 + T_guess].
        s_k = np.clip((guess_ts - t0) / T_guess, 0.0, 1.0)
        M = np.array([w(s) for s in s_k])
        P_guess = np.linalg.lstsq(M, guess_qs, rcond=None)[0].T
        prog.SetInitialGuess(P, P_guess)
        prog.SetInitialGuess(T, T_guess)

        for name, value in [("Major optimality tolerance", 1e-3), ("Major feasibility tolerance", 1e-4)]:
            prog.SetSolverOption(self.solver.solver_id(), name, value)
        result = self.solver.Solve(prog)
        ok = result.is_success()
        info = {"ok": ok, "solve_ms": 1e3 * (time.perf_counter() - tic), "T": float(result.GetSolution(T)),
                "T_guess": T_guess, "t0": t0}
        self.solve_log.append(info)
        if ok:
            self.traj = trajopt.ReconstructTrajectory(result)
            self.traj_dot = self.traj.MakeDerivative()
            self.t0 = t0
            self.t_descend = t0 + s_descend * self.traj.end_time()
        self.t_replan = t0 + self.replan_period
        return info

    @staticmethod
    def _add(prog, f, lb, ub, s, P, T, w, dw, t0):
        """Constraint lb <= f(q(s), qdot(s), t0 + s T) <= ub on the control points active at s and T."""
        ws, dws = w(s), dw(s)
        idx = np.flatnonzero((np.abs(ws) > 1e-12) | (np.abs(dws) > 1e-12))
        ws, dws = ws[idx], dws[idx]

        def g(z):
            Pz, Tz = z[:-1].reshape(len(idx), NQ).T, z[-1]
            return f(Pz @ ws, Pz @ dws / Tz, t0 + s * Tz)

        prog.AddConstraint(g, np.asarray(lb, float), np.asarray(ub, float), np.concatenate([P[:, idx].T.ravel(), [T]]))

    def _clearance(self, constraint):
        """Minimum distance at (q, t): the world's joints are where the prediction puts them at t."""
        cw, s = self.cw, slice(self.cw.iiwa_q, self.cw.iiwa_q + NQ)

        def f(q, qd, t):
            q_full, qdot_full = cw.full_state(_value(t))
            x = q_full + qdot_full * (t - _value(t))
            x[s] = q
            return constraint.Eval(x)

        return f

    def _ee(self, q, qd):
        """Gripper pose and spatial velocity in W (float or AutoDiff, following q)."""
        if q.dtype == object:
            plant, ctx, G = self.plant_ad, self.ctx_ad, self.G_ad
        else:
            plant, ctx, G = self.robot.plant, self.robot.context, self.robot.G
        iiwa = plant.GetModelInstanceByName("iiwa")
        plant.SetPositions(ctx, iiwa, q)
        plant.SetVelocities(ctx, iiwa, qd)
        return plant.CalcRelativeTransform(ctx, plant.world_frame(), G), G.CalcSpatialVelocityInWorld(ctx)
