"""(C) Hybrid: IK grasp poses, refined with space-time trajectory optimization.

Shortly before the object comes within reach (t_enter - lead_time) the controller

1. finds a collision-free hover pose and the grasp below it by inverse
   kinematics against the predicted obstacles, at the earliest grasp time
   (0.5 s steps) that has both;
2. optimises the approach with Drake's `KinematicTrajectoryOptimization`: a
   cubic B-spline q(s), s in [0, 1], with free duration T and time
   t = t0 + s T, that
     * starts at rest at the current configuration,
     * ends at the grasp pose of the object *at t0 + T*, moving with it,
     * passes the hover pose above the object and comes straight down,
     * keeps a minimum distance >= d_safe from the obstacles (and d_safe_env
       from the cart, the floor and the target) at sample times t0 + s_k T,
   minimising duration plus path energy.  The constraints are "space-time":
   where the object and the obstacles are at sample k depends on T, and the
   gradients include that dependence (Drake's
   MinimumDistanceLowerBoundConstraint is evaluated on the full kinematic
   plant, whose obstacle/cart joints move with t);
3. optimises the close, lift and hold the same way, with a fixed duration
   from the grasp to the end of the scene (`_plan_hold`): start in the
   approach's end state, move with the object while the gripper closes, be at
   least `lift_height` above the grasp from `close_time + lift_time` on, keep
   clear of the obstacles, the cart and the floor, end at rest; minimise path
   energy, with a lower acceleration limit than the approach;
4. sends hold-still + approach + close/lift/hold as one `JointTrajectory`,
   sampled every 5 ms by the wrapper; it holds its last pose afterwards.

The initial guess for the approach is straight joint-space lines to the IK
hover and grasp poses (they may cut through obstacles: they only have to end
at a valid grasp).  If it does not give a plan, the same via a collision-free
configuration halfway to the hover pose: the nearest (IK, obstacles where they
are then) to the straight line's midpoint, then to `n_random_seeds` random
configurations around it (each joint moved by up to `random_spread` rad); then
straight lines to the IK poses of the next grasp times, up to `n_ik_seeds` in
all.  The first seed that gives a feasible approach and hold is kept.  With
`n_workers` > 1 the seeds run in parallel processes and the first to finish
with a plan is kept.

Planning time costs simulated time (the harness's measured latency), so the
plan starts `plan_budget` s after the controller starts computing it: (C)
plans that much ahead and holds still until then.  A plan that takes longer
than its budget arrives after it should have started, the wrapper rejects it,
and (C) plans again with a budget of 1.5x that time.  With perfect prediction
the plan stays valid, so there is no replanning.  If no seed gives a plan, the
arm stays where it is.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import time
from multiprocessing.connection import wait

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
from dyngrasp.trajectory import JointTrajectory
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
        hover: float = 0.10,
        lead_time: float = 1.0,
        v_scale: float = 0.6,
        plan_budget: float = 2.0,
        energy_weight: float = 0.1,
        accel_limit: float = 4.0,
        n_ik_seeds: int = 3,
        n_random_seeds: int = 3,
        random_spread: float = 1.0,
        close_time: float = 0.4,
        lift_height: float = 0.10,
        lift_time: float = 0.6,
        hold_accel_limit: float = 2.0,
        hold_knot_spacing: float = 0.5,
        hold_sample_spacing: float = 0.4,
        obstacle_clearance: str = "fixed",
        track_margin: float = 0.005,
        sample_dt: float = 0.05,
        clearance_pref: float = 0.05,
        clearance_weight: float = 100.0,
        n_workers: int = 1,
    ):
        self._kwargs = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.n_control_points = n_control_points
        self.n_collision_samples = n_collision_samples
        self.d_safe, self.d_safe_env = d_safe, d_safe_env
        self.hover, self.lead_time, self.v_scale = hover, lead_time, v_scale
        self.plan_budget = plan_budget
        self.energy_weight = energy_weight
        self.accel_limit = accel_limit
        self.n_ik_seeds, self.n_random_seeds, self.random_spread = n_ik_seeds, n_random_seeds, random_spread
        self.close_time, self.lift_height, self.lift_time = close_time, lift_height, lift_time
        self.hold_accel_limit = hold_accel_limit
        self.hold_knot_spacing, self.hold_sample_spacing = hold_knot_spacing, hold_sample_spacing
        # "fixed": >= d_safe from the obstacles at the samples.  "speed": every sample_dt, per obstacle,
        # d_i >= track_margin + (v_obs_i + v_arm) sample_dt / 2, plus clearance_weight max(0, clearance_pref - d_i)^2.
        assert obstacle_clearance in ("fixed", "speed")
        self.obstacle_clearance = obstacle_clearance
        self.track_margin, self.sample_dt = track_margin, sample_dt
        self.clearance_pref, self.clearance_weight = clearance_pref, clearance_weight
        self.n_workers = n_workers  # > 1: try the seeds in parallel, in that many processes
        self._workers = []
        self.solver = SnoptSolver()

    # ---- episode ------------------------------------------------------------
    def reset(self, world: WorldModel, robot: RobotModel, state: RobotState):
        self.world, self.robot = world, robot
        self.cw = cw = CollisionWorld(world.spec, world)

        def min_distance(groups):
            return [
                MinimumDistanceLowerBoundConstraint(cw.plant, bound, cw.plant_context(cw.context(cats)))
                for cats, bound in groups
                if any(cw._geoms[c] for c in cats)
            ]

        obstacles = [(("obstacle",), self.d_safe)] if self.obstacle_clearance == "fixed" else []
        self.min_distance = min_distance([*obstacles, (("cart", "floor", "target"), self.d_safe_env)])
        self.min_distance_held = min_distance([*obstacles, (("cart", "floor"), self.d_safe_env)])  # object in hand
        self.obstacle_names = [o.name for o in world.spec.obstacles]
        self.v_obj = world.target_velocity(world.t_enter)
        self.plant_ad = robot.plant.ToAutoDiffXd()
        self.ctx_ad = self.plant_ad.CreateDefaultContext()
        self.G_ad = self.plant_ad.GetFrameByName("body", self.plant_ad.GetModelInstanceByName("wsg"))
        self.phase = "wait"
        self.budget = self.plan_budget
        self.traj = None  # approach: BsplineTrajectory on [0, T], started at self.t0
        self.hold_traj = None  # close, lift, hold: on [0, D], started at self.t0 + T
        self.sent = None  # the JointTrajectory sent to the wrapper
        self.solve_log: list[dict] = []
        self._close_workers()
        if self.n_workers > 1 and not mp.current_process().daemon:  # (a daemon process cannot have children)
            ctx = mp.get_context("spawn")
            for _ in range(self.n_workers):
                conn, child = ctx.Pipe()
                p = ctx.Process(target=_seed_worker, args=(child, world.spec, self._kwargs), daemon=True)
                p.start()
                self._workers.append((p, conn))
            for _, conn in self._workers:  # set up before the episode starts, so it costs no simulated time
                conn.recv()
            self._task = 0

    def _close_workers(self):
        for p, _ in self._workers:
            p.terminate()
        self._workers = []

    def __del__(self):
        self._close_workers()

    def update(self, state: RobotState, world: WorldModel) -> Command:
        t = state.t
        if self.phase == "track" and state.trajectory is not self.sent:  # arrived late: rejected
            self.phase = "wait"
        if self.phase == "wait":
            if t < world.t_enter - self.lead_time - self.budget:
                return Command(q=state.q_cmd, gripper=WSG_OPEN, info={"phase": "wait"})
            tic = time.perf_counter()
            t0 = t + self.budget
            info, plan = self._first_plan(t, t0, state.q_cmd) if t0 < world.t_exit - 2.0 else ({"ok": False}, None)
            info["budget"], self.budget = self.budget, max(self.budget, 1.5 * (time.perf_counter() - tic))
            if plan is None:
                self.phase, self.q_stay = "no_plan", state.q_cmd
                return Command(q=self.q_stay, gripper=WSG_OPEN, info={"phase": "no_plan", **info})
            self.phase, self.sent, self.t0, self.T = "track", plan, t0, info["T"]
            return Command(trajectory=self.sent, info={"phase": "track", **info})
        if self.phase == "no_plan":
            return Command(q=self.q_stay, gripper=WSG_OPEN, info={"phase": "no_plan"})
        phase = "track" if t < self.t0 + self.T else "hold"
        return Command(trajectory=self.sent, info={"phase": phase})

    def _joint_trajectory(self, t, q0) -> JointTrajectory:
        """Hold q0 from t to t0, then the approach (gripper open), then close, lift and hold (closed)."""
        T, D, dt = self.traj.end_time(), self.hold_traj.end_time(), self.period
        s1, s2 = np.arange(0.0, T, dt), np.append(np.arange(0.0, D, dt), D)
        ts = np.concatenate([[t], self.t0 + s1, self.t0 + T + s2])
        qs = np.vstack([q0, *(self.traj.value(x).ravel() for x in s1), *(self.hold_traj.value(x).ravel() for x in s2)])
        gripper = np.concatenate([np.full(1 + len(s1), WSG_OPEN), np.full(len(s2), WSG_CLOSED)])
        return JointTrajectory(ts, qs, gripper)

    # ---- planning ---------------------------------------------------------------
    def _first_plan(self, t, t0, q0):
        """(info, JointTrajectory or None): the first seed that gives a feasible approach and hold."""
        tic = time.perf_counter()
        grasps = self._ik_grasps(t0, q0)
        first = next(grasps, None)
        info, plan = {}, None
        if first is not None:
            info, plan = self._run_seeds(t, t0, q0, self._first_seeds(t0, q0, first))
            if plan is None:  # straight lines to the IK grasps of the next grasp times
                later = (("ik_later", g, None) for g in itertools.islice(grasps, self.n_ik_seeds - 1))
                info, plan = self._run_seeds(t, t0, q0, later)
        return {**info, "ok": plan is not None, "plan_ms": 1e3 * (time.perf_counter() - tic)}, plan

    def _first_seeds(self, t0, q0, grasp):
        """Straight lines to the first IK grasp; then the same via the nearest collision-free configuration,
        halfway to the hover pose, to the straight line's midpoint, and to random configurations around it."""
        t_d, _, q_h, _ = grasp
        yield "ik", grasp, None
        t_mid, mid = 0.5 * (t0 + t_d), 0.5 * (q0 + q_h)
        rng, lo, hi = np.random.default_rng(0), self.robot.q_lower + 0.05, self.robot.q_upper - 0.05
        for k in range(1 + self.n_random_seeds):
            target = mid if k == 0 else np.clip(mid + rng.uniform(-1, 1, NQ) * self.random_spread, lo, hi)
            via = self._ik(t_mid, None, target)
            if via is not None and not (k == 0 and np.allclose(via, mid, atol=1e-3)):  # (midpoint already free)
                yield ("ik_via" if k == 0 else "random_via"), grasp, via

    def _run_seeds(self, t, t0, q0, seeds):
        """(info, JointTrajectory) of the first seed with a feasible plan: in order, or, with workers, the first
        to finish; (info of the last, None) if none."""
        info, k = {}, 0
        if not self._workers:
            for k, seed in enumerate(seeds, 1):
                info, plan = self._try_seed(t, t0, q0, *seed)
                if plan is not None:
                    return {**info, "tries": k}, plan
            return {**info, "tries": k}, None
        self._task += 1
        seeds = list(seeds)
        for k, seed in enumerate(seeds):
            self._workers[k % len(self._workers)][1].send((self._task, t, t0, q0, *seed))
        done = 0
        while done < len(seeds):
            for conn in wait([c for _, c in self._workers]):
                task, info, plan = conn.recv()
                if task != self._task:  # a seed of an earlier plan, still running when that one was found
                    continue
                done += 1
                if plan is not None:
                    return {**info, "tries": done}, plan
        return {**info, "tries": done}, None

    def _try_seed(self, t, t0, q0, name, grasp, via):
        """(info, JointTrajectory or None): approach and hold from one seed."""
        ts, qs, self.t_descend = self._path(t0, q0, *grasp, q_via=via)
        r = self._plan(t0, q0, ts, qs)
        if not r["ok"]:
            return {**r, "seed": name}, None
        h = self._plan_hold()
        return {**r, "seed": name, "ok": h["ok"], "hold_ms": h["solve_ms"]}, self._joint_trajectory(t, q0) if h["ok"] else None

    def _ik_grasps(self, t0, q0, n_tries=6, descend_time=0.6):
        """Yields (t_d, t_g, q_h, q_g): IK hover pose at t_d and grasp at t_g, for each grasp time (0.5 s apart) that has both."""
        world, rng = self.world, np.random.default_rng(0)
        v_avg = 0.5 * self.v_scale * self.robot.v_limits
        for t_g in np.arange(max(world.t_enter, t0 + 1.0), world.t_exit - 2.0, 0.5):
            t_d = t_g - descend_time
            best = None
            for k in range(n_tries):  # first from q0, then from random configurations; keep the nearest
                q_g = self._ik(t_g, world.grasp_pose(t_g), q0 if k == 0 else rng.uniform(self.robot.q_lower, self.robot.q_upper))
                q_h = None if q_g is None else self._ik(t_d, self._pose(t_d, self.hover), q_g)
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

    def _pose(self, t, dz=0.0) -> RigidTransform:
        X = self.world.grasp_pose(t)
        return RigidTransform(X.rotation(), X.translation() + [0, 0, dz])

    def _ik(self, t, X, q_guess):
        """iiwa q with the gripper at X (or, if X is None, nearest to q_guess) and >= d_safe + 1 cm from the
        obstacles where they are at t (None if not found)."""
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
        if X is None:  # the nearest collision-free configuration to q_guess
            prog.AddQuadraticErrorCost(np.eye(NQ), q_guess, q[cw.iiwa_q : cw.iiwa_q + NQ])
        else:
            G = plant.GetFrameByName("body", cw.kin.wsg)
            ik.AddPositionConstraint(G, np.zeros(3), plant.world_frame(), X.translation() - 0.003, X.translation() + 0.003)
            ik.AddOrientationConstraint(G, RotationMatrix(), plant.world_frame(), X.rotation(), 0.01)
        ik.AddMinimumDistanceLowerBoundConstraint(self.d_safe + 0.01, 0.2)
        result = Solve(prog, x)
        return result.GetSolution(q)[cw.iiwa_q : cw.iiwa_q + NQ] if result.is_success() else None

    def _plan(self, t0, q0, guess_ts, guess_qs) -> dict:
        """Approach: from rest at q0 at t0 to the grasp, seeded with the joint path (guess_ts, guess_qs)."""
        tic = time.perf_counter()
        world = self.world
        T_lo = max(world.t_enter - t0, 0.3)
        T_hi = world.t_exit - 1.5 - t0
        T_guess = float(np.clip(guess_ts[-1] - t0, T_lo, T_hi))
        s_descend = float(np.clip((self.t_descend - t0) / T_guess, 0.3, 0.9))

        trajopt, P, T, w, dw = self._trajopt(self.n_control_points, self.accel_limit)
        prog = trajopt.get_mutable_prog()
        trajopt.AddDurationCost(1.0)
        trajopt.AddPathEnergyCost(self.energy_weight)
        trajopt.AddDurationConstraint(T_lo, T_hi)
        trajopt.AddPathPositionConstraint(q0, q0, 0.0)
        trajopt.AddPathVelocityConstraint(np.zeros(NQ), np.zeros(NQ), 0.0)

        def at_grasp(q, qd, t):
            _, V = self._ee(q, qd)
            return np.concatenate([self._pose_error(q, qd, t), V.rotational(), V.translational() - self.v_obj])

        def at_hover(q, qd, t):
            return self._pose_error(q, qd, t, self.hover)

        def above_grasp(q, qd, t):
            X, _ = self._ee(q, qd)
            return X.translation() - self._grasp_target(t)[0]

        tol_p, tol_r, tol_v = 0.005, 0.02, 0.02
        tol = np.array([tol_p] * 3 + [tol_r] * 3 + [tol_r] * 3 + [tol_v] * 3)
        self._add(prog, at_grasp, -tol, tol, 1.0, P, T, w, dw, t0)
        self._add(prog, at_hover, -tol[:6] * 2, tol[:6] * 2, s_descend, P, T, w, dw, t0)
        for s in np.linspace(s_descend, 1.0, 5)[1:-1]:  # straight down onto the object
            self._add(prog, above_grasp, [-0.015, -0.015, 0.0], [0.015, 0.015, self.hover + 0.02], s, P, T, w, dw, t0)
        for s in np.linspace(0.0, 1.0, self.n_collision_samples + 1)[1:]:
            for c in self.min_distance:
                self._add(prog, self._clearance(c), c.lower_bound(), c.upper_bound(), s, P, T, w, dw, t0)
        if self.obstacle_clearance == "speed":
            n = int(np.ceil(T_guess / self.sample_dt))
            for k in range(1, n + 1):
                self._add_obstacle_clearance(prog, k, n, P, T, w, dw, t0)

        self._guess(prog, P, T, w, (guess_ts - t0) / T_guess, guess_qs, T_guess)
        result, info = self._solve(prog, T, tic)
        info.update(t0=t0, T_guess=T_guess)
        if info["ok"]:
            self.traj = trajopt.ReconstructTrajectory(result)
            self.traj_dot = self.traj.MakeDerivative()
            self.t0 = t0
            self.t_descend = t0 + s_descend * self.traj.end_time()
        return info

    def _plan_hold(self) -> dict:
        """Close, lift and hold until the end of the scene, from the approach's end state at t_g = t0 + T."""
        tic = time.perf_counter()
        T = self.traj.end_time()
        t_g = self.t0 + T
        q_T, v_T = self.traj.value(T).ravel(), self.traj_dot.value(T).ravel()
        t_up = self.close_time + self.lift_time
        D = max(self.world.duration - t_g, t_up + 0.5)

        trajopt, P, Td, w, dw = self._trajopt(int(np.ceil(D / self.hold_knot_spacing)) + 3, self.hold_accel_limit)
        prog = trajopt.get_mutable_prog()
        trajopt.AddPathEnergyCost(1.0)
        trajopt.AddDurationConstraint(D, D)
        trajopt.AddPathPositionConstraint(q_T, q_T, 0.0)
        trajopt.AddPathVelocityConstraint(v_T * D, v_T * D, 0.0)  # d/ds = D d/dt
        trajopt.AddPathVelocityConstraint(np.zeros(NQ), np.zeros(NQ), 1.0)

        tol = np.array([0.005] * 3 + [0.02] * 3)
        for tk in np.linspace(0.0, self.close_time, 3)[1:]:  # closing: moving with the object
            self._add(prog, self._pose_error, -tol, tol, tk / D, P, Td, w, dw, t_g)
        z_lift = self.world.grasp_pose(t_g).translation()[2] + self.lift_height

        def height(q, qd, t):
            return self._ee(q, qd)[0].translation()[2:]

        for tk in np.arange(t_up, D + 1e-9, self.hold_sample_spacing):  # lifted, and kept there
            self._add(prog, height, [z_lift], [np.inf], tk / D, P, Td, w, dw, t_g)
        for tk in np.arange(self.close_time, D + 1e-9, self.hold_sample_spacing):
            for c in self.min_distance_held:
                self._add(prog, self._clearance(c), c.lower_bound(), c.upper_bound(), tk / D, P, Td, w, dw, t_g)
        if self.obstacle_clearance == "speed":
            n = int(np.ceil(D / self.sample_dt))
            for k in range(1, n + 1):
                self._add_obstacle_clearance(prog, k, n, P, Td, w, dw, t_g)

        # Initial guess: move with the object, then straight up, then stay (IK poses where they exist).
        t_c = t_g + self.close_time
        q_c = self._ik(t_c, self._pose(t_c), q_T)
        q_c = q_T + v_T * self.close_time if q_c is None else q_c
        q_l = self._ik(t_g + t_up, self._pose(t_c, self.lift_height), q_c)
        q_l = q_c if q_l is None else q_l
        self._guess(prog, P, Td, w, np.array([0.0, self.close_time, t_up, D]) / D, np.array([q_T, q_c, q_l, q_l]), D)
        result, info = self._solve(prog, Td, tic)
        info.update(t0=t_g, hold=True)
        if info["ok"]:
            self.hold_traj = trajopt.ReconstructTrajectory(result)
        return info

    # ---- optimisation helpers ------------------------------------------------------
    def _trajopt(self, n_control_points, accel_limit):
        """A cubic B-spline trajectory optimisation within the joint limits, and its basis functions w(s), w'(s)."""
        N = n_control_points
        trajopt = KinematicTrajectoryOptimization(NQ, N, 4)
        margin = 0.05
        trajopt.AddPositionBounds(self.robot.q_lower + margin, self.robot.q_upper - margin)
        vmax = self.v_scale * self.robot.v_limits
        trajopt.AddVelocityBounds(-vmax, vmax)
        trajopt.AddAccelerationBounds(-accel_limit * np.ones(NQ), accel_limit * np.ones(NQ))
        weights = BsplineTrajectory(trajopt.basis(), [np.eye(N)[:, [j]] for j in range(N)])
        dweights = weights.MakeDerivative()
        return (trajopt, trajopt.control_points(), trajopt.duration(),
                lambda s: weights.value(s).ravel(), lambda s: dweights.value(s).ravel())

    @staticmethod
    def _guess(prog, P, T, w, s_k, qs, T_guess):
        """Initial guess: least-squares fit of the control points to the path qs at s_k (interpolated)."""
        s = np.linspace(0.0, 1.0, 40)
        q = np.array([np.interp(s, np.clip(s_k, 0.0, 1.0), qs[:, j]) for j in range(NQ)]).T
        prog.SetInitialGuess(P, np.linalg.lstsq(np.array([w(x) for x in s]), q, rcond=None)[0].T)
        prog.SetInitialGuess(T, T_guess)

    def _solve(self, prog, T, tic):
        for name, value in [("Major optimality tolerance", 1e-3), ("Major feasibility tolerance", 1e-4)]:
            prog.SetSolverOption(self.solver.solver_id(), name, value)
        result = self.solver.Solve(prog)
        info = {"ok": result.is_success(), "solve_ms": 1e3 * (time.perf_counter() - tic),
                "T": float(result.GetSolution(T)), "cost": result.get_optimal_cost(),
                "status": result.get_solver_details().info}
        self.solve_log.append(info)
        return result, info

    def _grasp_target(self, t, dz=0.0):
        """Grasp pose (position, rotation matrix) at (AutoDiff) time t: the cart moves at constant velocity."""
        X = self.world.grasp_pose(_value(t))
        return X.translation() + self.v_obj * (t - _value(t)) + [0, 0, dz], X.rotation().matrix()

    def _pose_error(self, q, qd, t, dz=0.0):
        X, _ = self._ee(q, qd)
        p, R = self._grasp_target(t, dz)
        return np.concatenate([X.translation() - p, _vee(R.T @ X.rotation().matrix())])

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

    def _add_obstacle_clearance(self, prog, k, n, P, T, w, dw, t0):
        """Sample k of n (s = k / n, spacing h = T / n): per obstacle i, with slack e_i >= 0 costing w e_i^2,
            d_i - (v_obs_i + v_arm) h / 2 >= track_margin   and   d_i + e_i >= clearance_pref.
        d_i: signed distance to obstacle i where it is at t (CollisionWorld.rows); v_arm: speed of the
        robot's witness point under the plan's qdot (its gradient in q through the Jacobian is left out)."""
        s, m = k / n, len(self.obstacle_names)
        ws, dws = w(s), dw(s)
        idx = np.flatnonzero((np.abs(ws) > 1e-12) | (np.abs(dws) > 1e-12))
        ws, dws, nP = ws[idx], dws[idx], len(idx) * NQ
        e = prog.NewContinuousVariables(m, "e")
        prog.AddBoundingBoxConstraint(0.0, np.inf, e)
        prog.AddQuadraticCost(2.0 * self.clearance_weight * np.eye(m), np.zeros(m), e, is_convex=True)
        prog.SetInitialGuess(e, np.zeros(m))
        far = self.clearance_pref + 0.1  # beyond this an obstacle is ignored at this sample

        def g(z):
            zv = ExtractValue(z).ravel() if z.dtype == object else z
            Pz, Tz, ez = zv[:nP].reshape(len(idx), NQ).T, zv[nP], zv[nP + 1 :]
            q, qd, t = Pz @ ws, Pz @ dws / Tz, t0 + s * Tz
            h = Tz / n
            val = np.concatenate([np.full(m, far), np.full(m, far) + ez])
            jac = np.zeros((2 * m, 2 * NQ + 1))  # d val / d (q, qd, t)
            rows = {}
            for r in self.cw.rows(q, t, far):
                if r.category == "obstacle" and (r.other_body not in rows or r.phi < rows[r.other_body].phi):
                    rows[r.other_body] = r
            for i, name in enumerate(self.obstacle_names):
                if name not in rows:
                    continue
                r = rows[name]
                v = r.J @ qd
                v_arm, v_obs = np.linalg.norm(v), np.linalg.norm(r.v_world)
                val[i] = r.phi - (v_obs + v_arm) * h / 2
                val[m + i] = r.phi + ez[i]
                jac[[i, m + i], :NQ] = r.a
                jac[[i, m + i], 2 * NQ] = -r.vn
                jac[i, NQ : 2 * NQ] = -h / 2 * (v @ r.J) / max(v_arm, 1e-9)
                jac[i, 2 * NQ] -= (v_obs + v_arm) / (2 * n) / s  # h = (t - t0) / (s n)
            if z.dtype != object:
                return val
            J_qqdt = np.vstack([  # d (q, qd, t) / d (P, T)
                np.hstack([np.kron(ws, np.eye(NQ)), np.zeros((NQ, 1))]),
                np.hstack([np.kron(dws, np.eye(NQ)) / Tz, -qd[:, None] / Tz]),
                np.append(np.zeros(nP), s)[None, :],
            ])
            J = np.hstack([jac @ J_qqdt, np.vstack([np.zeros((m, m)), np.eye(m)])])
            return InitializeAutoDiff(val, J @ ExtractGradient(z)).ravel()

        lb = np.concatenate([np.full(m, self.track_margin), np.full(m, self.clearance_pref)])
        prog.AddConstraint(g, lb, np.full(2 * m, np.inf), np.concatenate([P[:, idx].T.ravel(), [T], e]))

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


def _seed_worker(conn, spec, kwargs):
    """A process that plans one seed per task (HybridController with n_workers > 1)."""
    c = HybridController(**{**kwargs, "n_workers": 1})
    c.reset(WorldModel(spec), RobotModel(), None)
    conn.send("ready")
    for task, *args in iter(conn.recv, None):
        conn.send((task, *c._try_seed(*args)))
