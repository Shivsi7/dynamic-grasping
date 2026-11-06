"""(C) Hybrid: IK grasp poses, refined with space-time trajectory optimization.

When (A) would start moving (t_enter - lead_time) the controller

1. builds initial guesses (`seed`).  Each ends at a collision-free hover pose
   and the grasp below it, both from inverse kinematics against the predicted
   obstacles, at the earliest grasp time (0.5 s steps) where IK finds them.
   The paths may cut through obstacles: they only have to end at a valid grasp.
   * "ik": straight joint-space lines to the hover pose and the grasp (if that
     fails to optimise, the same at the next grasp times, up to `n_ik_seeds`);
   * "random": the same, but via a random configuration halfway to the hover
     pose (the midpoint of the straight line, each joint moved by up to
     `random_spread` rad), up to `n_random_seeds` of them, in case the
     straight line leads the optimiser into a bad local minimum;
   * "both": each of the above;
2. refines each with Drake's `KinematicTrajectoryOptimization` and keeps the
   cheapest feasible result: a cubic B-spline
   q(s), s in [0, 1], with free duration T and time t = t0 + s T, that
     * starts at the current state,
     * ends at the grasp pose of the object *at t0 + T*, moving with it,
     * passes the hover pose above the object at the seed's descent time
       and comes straight down from there,
     * keeps a minimum distance >= d_safe from the obstacles (and d_safe_env
       from the cart, the floor and the target) at sample times t0 + s_k T,
   minimising duration plus path energy.  The constraints are "space-time":
   where the object and the obstacles are at sample k depends on T, and the
   gradients include that dependence (Drake's
   MinimumDistanceLowerBoundConstraint is evaluated on the full kinematic
   plant, whose obstacle/cart joints move with t);
3. sends the result as a `JointTrajectory`, sampled every 5 ms by the wrapper.

Planning time costs simulated time (the harness's measured latency), so the
plan starts `plan_budget` s after the controller starts computing it: (C)
plans that much before (A) would start moving and holds still until then.  A
plan that takes longer than its budget arrives after it should have started,
the wrapper rejects it, and (C) plans again with a budget of 1.5x that time.
With perfect prediction the plan stays valid, so there is no replanning.

The same trajectory goes on from the grasp at t0 + T: close, lift and hold
(`ReactiveQPController.tail`: (A)'s close and lift, run kinematically against
the prediction and smoothed), like (B)'s grasp tail.  After it, (A) holds that
pose.  If every optimisation fails it falls back to (A) for the rest of the
episode.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
from pydrake.all import (
    AutoDiffXd,
    BsplineTrajectory,
    InverseKinematics,
    KinematicTrajectoryOptimization,
    MinimumDistanceLowerBoundConstraint,
    RigidTransform,
    RotationMatrix,
    SnoptSolver,
    Solve,
)
from pydrake.autodiffutils import ExtractGradient, ExtractValue, InitializeAutoDiff

from dyngrasp.controller import (
    NQ,
    WSG_CLOSED,
    WSG_OPEN,
    Command,
    Controller,
    RobotModel,
    RobotState,
)
from dyngrasp.reactive import ReactiveQPController
from dyngrasp.trajectory import JointTrajectory, hermite
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
        plan_budget: float = 2.0,
        energy_weight: float = 0.1,
        accel_limit: float = 4.0,
        seed: str = "both",
        n_ik_seeds: int = 3,
        n_random_seeds: int = 3,
        random_spread: float = 1.0,
        **reactive_kwargs,
    ):
        assert seed in ("ik", "random", "both")
        self.seed, self.n_ik_seeds = seed, n_ik_seeds
        self.n_random_seeds, self.random_spread = n_random_seeds, random_spread
        self.reactive = ReactiveQPController(d_safe=d_safe, d_safe_env=d_safe_env, **reactive_kwargs)
        self.reactive.period = self.period  # it runs inside this controller's cycle
        self.n_control_points = n_control_points
        self.n_collision_samples = n_collision_samples
        self.d_safe, self.d_safe_env = d_safe, d_safe_env
        self.plan_budget = plan_budget
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
        self.budget = self.plan_budget
        self.traj = None  # BsplineTrajectory on [0, T], started at self.t0
        self.sent = None  # the JointTrajectory sent to the wrapper
        self.solve_log: list[dict] = []

    def update(self, state: RobotState, world: WorldModel) -> Command:
        t = state.t
        info = {}
        if self.phase == "track" and state.trajectory is not self.sent:  # arrived late: rejected
            self.phase = "wait"
        if self.phase == "wait":
            if t < world.t_enter - self.reactive.lead_time - self.budget:
                return Command(q=state.q_cmd, gripper=WSG_OPEN, info={"phase": "wait"})
            tic = time.perf_counter()
            t0 = t + self.budget
            info = self._first_plan(t0, state.q_cmd) if t0 < world.t_exit - 2.0 else {"ok": False}
            info["budget"], self.budget = self.budget, max(self.budget, 1.5 * (time.perf_counter() - tic))
            if info["ok"]:
                self.phase, self.sent = "track", self._joint_trajectory(t, state.q_cmd)
            else:
                self.phase = "fallback"
                self.reactive._start("approach")

        if self.phase == "track":
            if t < self.sent.end_time:
                phase = "track" if t < self.t0 + self.traj.end_time() else "tail"
                return Command(trajectory=self.sent, info={"phase": phase, **info})
            self.phase, self.reactive = "reactive", self.tail_controller  # holding the lifted pose

        cmd = self.reactive.update(state, world)
        cmd.info["phase"] = ("fallback_" if self.phase == "fallback" else "") + cmd.info["phase"]
        cmd.info.update(info)
        return cmd

    def _joint_trajectory(self, t, q0) -> JointTrajectory:
        """Hold q0 from t to t0, then the plan to the grasp, then close, lift and hold."""
        T, dt = self.traj.end_time(), self.period
        s = np.arange(0.0, T, dt)
        q_T, v_T = self.traj.value(T).ravel(), self.traj_dot.value(T).ravel()
        ts = np.concatenate([[t], self.t0 + s])
        approach = JointTrajectory(ts, np.vstack([q0, *(self.traj.value(x).ravel() for x in s)]), np.full(len(ts), WSG_OPEN))
        tail_t, tail_q, self.tail_controller = self.reactive.tail(self.world, q_T, v_T, self.t0 + T)
        tail = hermite(tail_t, tail_q, v_T, np.zeros(NQ), dt, np.full(len(tail_t), WSG_CLOSED))
        return JointTrajectory.concatenate([approach, tail])

    # ---- planning ---------------------------------------------------------------
    def _first_plan(self, t0, q0) -> dict:
        """Solve from each seed; keep the cheapest feasible plan."""
        grasps = self._ik_grasps(t0, q0)
        first = next(grasps, None)
        if first is None:
            return {"ok": False, "ik_found": False}
        rng = np.random.default_rng(0)
        mid = 0.5 * (q0 + first[2])
        lo, hi = self.robot.q_lower + 0.05, self.robot.q_upper - 0.05
        sources = []
        if self.seed in ("ik", "both"):
            straight = itertools.islice(itertools.chain([first], grasps), self.n_ik_seeds)
            sources.append(("ik", (self._path(t0, q0, *g) for g in straight)))
        if self.seed in ("random", "both"):
            vias = (np.clip(mid + rng.uniform(-1, 1, NQ) * self.random_spread, lo, hi) for _ in range(self.n_random_seeds))
            sources.append(("random", (self._path(t0, q0, *first, q_via=v) for v in vias)))
        best, info = None, {}
        for name, seeds in sources:
            tic = time.perf_counter()
            for k, (ts, qs, t_descend) in enumerate(seeds):
                self.t_descend = t_descend
                r = self._plan(t0, q0, ts, qs)
                info[f"{name}_ok"], info[f"{name}_tries"] = r["ok"], k + 1
                if r["ok"]:
                    if best is None or r["cost"] < best[0]["cost"]:
                        best = ({**r, "seed": name}, self.traj, self.traj_dot, self.t_descend)
                    break
            info[f"{name}_ms"] = 1e3 * (time.perf_counter() - tic)
        if best is None:
            return {**info, "ok": False}
        r, self.traj, self.traj_dot, self.t_descend = best
        self.t0 = t0
        return {**info, **r}

    def _ik_grasps(self, t0, q0, n_tries=6, descend_time=0.6):
        """Yields (t_d, t_g, q_h, q_g): IK hover pose at t_d and grasp at t_g, for each grasp time (0.5 s apart) that has both."""
        world, rng = self.world, np.random.default_rng(0)
        v_avg = 0.5 * self.reactive.v_scale * self.robot.v_limits
        for t_g in np.arange(max(world.t_enter, t0 + 1.0), world.t_exit - 2.0, 0.5):
            t_d = t_g - descend_time
            X_h = world.grasp_pose(t_d)
            X_h = RigidTransform(X_h.rotation(), X_h.translation() + [0, 0, self.reactive.hover])
            best = None
            for k in range(n_tries):  # first from q0, then from random configurations; keep the nearest
                q_g = self._ik(t_g, world.grasp_pose(t_g), q0 if k == 0 else rng.uniform(self.robot.q_lower, self.robot.q_upper))
                q_h = None if q_g is None else self._ik(t_d, X_h, q_g)
                if q_h is None:
                    continue
                travel = np.max(np.abs(q_h - q0) / v_avg)
                if t0 + travel <= t_d and (best is None or travel < best[0]):
                    best = (travel, q_h, q_g)
            if best is not None:
                yield t_d, t_g, best[1], best[2]

    @staticmethod
    def _path(t0, q0, t_d, t_g, q_h, q_g, q_via=None):
        """(ts, qs, t_d): straight joint-space lines q0 -> (q_via halfway to t_d) -> q_h at t_d -> q_g at t_g."""
        knots = [(t0, q0), (t_d, q_h), (t_g, q_g)]
        if q_via is not None:
            knots.insert(1, (0.5 * (t0 + t_d), q_via))
        kt, kq = np.array([k[0] for k in knots]), np.array([k[1] for k in knots])
        ts = np.linspace(t0, t_g, 40)
        return ts, np.array([np.interp(ts, kt, kq[:, j]) for j in range(NQ)]).T, t_d

    def _ik(self, t, X, q_guess):
        """iiwa q with the gripper at X and >= d_safe + 1 cm from the obstacles where they are at t (None if not found)."""
        cw, plant = self.cw, self.cw.plant
        ctx = cw.context(("obstacle",))
        cw.set_time(ctx, t, q_guess)
        pc = cw.plant_context(ctx)
        ik = InverseKinematics(plant, pc)
        q, prog = ik.q(), ik.get_mutable_prog()
        x = plant.GetPositions(pc)
        world_joints = np.ones(len(x), bool)
        world_joints[cw.iiwa_q : cw.iiwa_q + NQ] = False
        prog.AddBoundingBoxConstraint(x[world_joints], x[world_joints], q[world_joints])
        G = plant.GetFrameByName("body", cw.kin.wsg)
        ik.AddPositionConstraint(G, np.zeros(3), plant.world_frame(), X.translation() - 0.003, X.translation() + 0.003)
        ik.AddOrientationConstraint(G, RotationMatrix(), plant.world_frame(), X.rotation(), 0.01)
        ik.AddMinimumDistanceLowerBoundConstraint(self.d_safe + 0.01, 0.2)
        result = Solve(prog, x)
        return result.GetSolution(q)[cw.iiwa_q : cw.iiwa_q + NQ] if result.is_success() else None

    def _plan(self, t0, q0, guess_ts, guess_qs) -> dict:
        """Solve from rest at q0 at t0, seeded with the joint path (guess_ts, guess_qs)."""
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
        trajopt.AddPathVelocityConstraint(np.zeros(NQ), np.zeros(NQ), 0.0)

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
                "T_guess": T_guess, "t0": t0, "cost": result.get_optimal_cost()}
        self.solve_log.append(info)
        if ok:
            self.traj = trajopt.ReconstructTrajectory(result)
            self.traj_dot = self.traj.MakeDerivative()
            self.t0 = t0
            self.t_descend = t0 + s_descend * self.traj.end_time()
        return info

    @staticmethod
    def _add(prog, f, lb, ub, s, P, T, w, dw, t0):
        """Constraint lb <= f(q(s), qdot(s), t0 + s T) <= ub on the control points active at s and T.

        q, qdot and t are linear in the control points (and T); their gradients are built with
        numpy and handed to f as AutoDiff, which is much faster than AutoDiff arithmetic in Python."""
        ws, dws = w(s), dw(s)
        idx = np.flatnonzero((np.abs(ws) > 1e-12) | (np.abs(dws) > 1e-12))
        ws, dws = ws[idx], dws[idx]
        Jq = np.hstack([np.kron(ws, np.eye(NQ)), np.zeros((NQ, 1))])  # dq/d(P, T)
        Jt = np.append(np.zeros(len(idx) * NQ), s)  # dt/d(P, T)

        def g(z):
            if z.dtype != object:
                Pz, Tz = z[:-1].reshape(len(idx), NQ).T, z[-1]
                return f(Pz @ ws, Pz @ dws / Tz, t0 + s * Tz)
            zv, G = ExtractValue(z).ravel(), ExtractGradient(z)
            Pz, Tz = zv[:-1].reshape(len(idx), NQ).T, zv[-1]
            qd = Pz @ dws / Tz
            Jqd = np.hstack([np.kron(dws, np.eye(NQ)) / Tz, -qd[:, None] / Tz])
            return f(
                InitializeAutoDiff(Pz @ ws, Jq @ G).ravel(),
                InitializeAutoDiff(qd, Jqd @ G).ravel(),
                InitializeAutoDiff([t0 + s * Tz], (Jt @ G)[None, :])[0, 0],
            )

        prog.AddConstraint(g, np.asarray(lb, float), np.asarray(ub, float), np.concatenate([P[:, idx].T.ravel(), [T]]))

    def _clearance(self, constraint):
        """Minimum distance at (q, t): the world's joints are where the prediction puts them at t."""
        cw, s = self.cw, slice(self.cw.iiwa_q, self.cw.iiwa_q + NQ)

        def f(q, qd, t):
            x, xdot = cw.full_state(_value(t))
            x = x.copy()
            if q.dtype != object:
                x[s] = q
                return constraint.Eval(x)
            x[s] = ExtractValue(q).ravel()
            dx = np.outer(xdot, ExtractGradient(np.array([t]))[0])  # the world moves with t
            dx[s] = ExtractGradient(q)
            return constraint.Eval(InitializeAutoDiff(x, dx).ravel())

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
