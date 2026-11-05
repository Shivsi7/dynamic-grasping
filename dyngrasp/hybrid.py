"""(C) Hybrid: roll out (A) against the prediction, refine with space-time trajectory optimization.

When (A) would start moving (t_enter - lead_time) the controller

1. builds initial guesses (`seed`):
   * "rollout": rolls out the reactive controller (A) kinematically against the
     predicted target and obstacle motion (`ReactiveQPController.rollout`): a
     joint path from the current configuration to the grasp;
   * "ik": a straight joint-space path to a collision-free hover pose and then
     to the grasp below it, both from inverse kinematics against the predicted
     obstacles, at the earliest grasp time (0.5 s steps) where IK finds them
     (if that fails to optimise, the next such times, up to `n_ik_seeds`).
     It may cut through obstacles: it only has to end at a valid grasp, which
     is what (A)'s rollout misses when (A) gets stuck in a local minimum;
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
3. tracks the result with position commands, re-solving every `replan_period`
   from the current reference, warm-started with its remainder.  With perfect
   prediction one solve suffices; re-solving only corrects prediction errors.

At t0 + T it closes and lifts with (A).  If every first optimisation fails it
falls back to (A) for the whole episode.
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
        seed: str = "both",
        n_ik_seeds: int = 3,
        **reactive_kwargs,
    ):
        assert seed in ("rollout", "ik", "both")
        self.seed, self.n_ik_seeds = seed, n_ik_seeds
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
            info = self._first_plan(t, state.q_cmd)
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
    def _first_plan(self, t0, q0) -> dict:
        """Solve from each seed; keep the cheapest feasible plan."""
        world, sources, info = self.world, [], {}
        if self.seed in ("rollout", "both"):
            ts, qs, t_descend, reached = self.reactive.rollout(world, q0, t0, t_max=world.t_exit - 2.0 - t0)
            sources.append(("rollout", [(ts, qs, t_descend if t_descend is not None else ts[-1] - 0.3 * (ts[-1] - t0))]))
            info["rollout_reached"] = reached
        if self.seed in ("ik", "both"):
            sources.append(("ik", itertools.islice(self._ik_seeds(t0, q0), self.n_ik_seeds)))
        best = None
        for name, seeds in sources:
            tic = time.perf_counter()
            for k, (ts, qs, t_descend) in enumerate(seeds):
                self.t_descend = t_descend
                r = self._plan(t0, q0, np.zeros(NQ), ts, qs)
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

    def _ik_seeds(self, t0, q0, n_tries=6, descend_time=0.6):
        """Yields (ts, qs, t_descend): q0 -> IK hover pose -> IK grasp, for each grasp time (0.5 s apart) that has both."""
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
                _, q_h, q_g = best
                ts = np.linspace(t0, t_g, 40)
                qs = np.array([np.interp(ts, [t0, t_d, t_g], [q0[j], q_h[j], q_g[j]]) for j in range(NQ)]).T
                yield ts, qs, t_d

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
                "T_guess": T_guess, "t0": t0, "cost": result.get_optimal_cost()}
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
