"""Goal roots for the space-time planner: where the robot must be, and when.

For a candidate intercept time T the goal is a pre-grasp configuration q_g(T)
(the gripper `hover` above the grasp pose at T), and from there a fixed
"tail": descend while matching the target's velocity, close, lift, hold.  The
tail is part of the plan and is collision-checked like everything else, so an
obstacle crossing *during* the grasp rules that (T, q_g) out.

Everything is computed from `WorldModel` (target pose and velocity at any
time) and a grasp frame X_UG, so it applies to any scenario and to grasp
candidates other than the nominal top-down one (pass `grasps`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydrake.all import InverseKinematics, RigidTransform, RotationMatrix, Solve

from dyngrasp.controller import NQ, WSG_CLOSED, WSG_OPEN, RobotModel
from dyngrasp.trajectory import JointTrajectory
from dyngrasp.world import WorldModel


@dataclass
class TailParams:
    hover: float = 0.08  # pre-grasp height above the grasp pose (m)
    t_settle: float = 0.3  # ride along at hover height first, so arrival tracking error dies out
    t_descend: float = 0.6
    t_close: float = 0.4
    t_lift: float = 0.6  # lift while slowing down from the target's velocity to rest
    lift_height: float = 0.15
    t_hold: float = 0.6  # hold still (success needs 0.5 s lifted)
    dt: float = 0.01  # differential-IK integration step
    knot_dt: float = 0.05  # knot spacing of the returned trajectory
    k_p: float = 10.0
    k_rot: float = 10.0
    max_track_err: float = 0.01  # reject a tail whose gripper error exceeds this (m)

    @property
    def duration(self) -> float:
        return self.t_settle + self.t_descend + self.t_close + self.t_lift + self.t_hold


@dataclass
class GoalCandidate:
    T: float  # intercept time (arrival at the pre-grasp)
    q: np.ndarray  # pre-grasp configuration
    tail: JointTrajectory  # T .. T + tail duration, with the gripper schedule
    grasp_index: int
    family: int  # which IK seed family produced it


def _pose_error(X: RigidTransform, X_des: RigidTransform) -> np.ndarray:
    R_err = X_des.rotation() @ X.rotation().inverse()
    aa = R_err.ToAngleAxis()
    return np.concatenate([aa.axis() * aa.angle(), X_des.translation() - X.translation()])


class GoalGenerator:
    def __init__(
        self,
        world: WorldModel,
        robot: RobotModel,
        v_max: np.ndarray,  # joint-velocity limits for the tail (it may use more than the planner)
        tail: TailParams | None = None,
        grasps: list[RigidTransform] | None = None,
        ik_pos_tol: float = 0.003,
        ik_rot_tol: float = 0.02,
    ):
        self.world = world
        self.robot = robot
        self.v_max = np.asarray(v_max, float)
        self.tp = tail or TailParams()
        self.grasps = grasps if grasps else [world.object.X_UG_grasp]
        self.ik_pos_tol, self.ik_rot_tol = ik_pos_tol, ik_rot_tol
        plant = robot.plant
        self._plant = plant
        self._ctx = plant.CreateDefaultContext()
        self._nq_plant = plant.num_positions()
        self._i_iiwa = plant.GetPositionsFromArray(robot.iiwa, np.arange(self._nq_plant, dtype=float)).astype(int)

    # ---- intercept times ---------------------------------------------------------
    def candidate_times(self, t_earliest: float, dT: float = 0.25, max_count: int = 16) -> np.ndarray:
        """Intercept times T to try: from when the target comes within reach until the
        tail would no longer fit in the episode.  Whether the arm can actually follow
        the tail from T is decided by IK and the tail rollout, not here."""
        lo = max(t_earliest, self.world.t_enter - 0.5)
        hi = min(self.world.t_exit, self.world.duration - self.tp.duration)
        if hi < lo:
            return np.zeros(0)
        Ts = np.arange(lo, hi + 1e-9, dT)
        if len(Ts) > max_count:  # spread evenly, always keeping the earliest
            Ts = Ts[np.unique(np.round(np.linspace(0, len(Ts) - 1, max_count)).astype(int))]
        return Ts

    # ---- poses -------------------------------------------------------------------
    def grasp_pose(self, t: float, k: int) -> RigidTransform:
        return self.world.grasp_pose(t, self.grasps[k])

    def hover_pose(self, t: float, k: int) -> RigidTransform:
        X = self.grasp_pose(t, k)
        return RigidTransform(X.rotation(), X.translation() + [0.0, 0.0, self.tp.hover])

    def tail_pose(self, tau: float, T: float, k: int) -> tuple[RigidTransform, np.ndarray]:
        """Desired gripper pose and translational velocity at T + tau."""
        tp = self.tp
        if tau < tp.t_settle:  # ride along at hover height
            t = T + tau
            X = self.grasp_pose(t, k)
            return RigidTransform(X.rotation(), X.translation() + [0.0, 0.0, tp.hover]), self.world.target_velocity(t).copy()
        T, tau = T + tp.t_settle, tau - tp.t_settle  # the rest is timed from the end of the settle phase
        t_grip = tp.t_descend + tp.t_close
        if tau < t_grip:
            t = T + tau
            X = self.grasp_pose(t, k)
            v = self.world.target_velocity(t).copy()
            dz = 0.0
            if tau < tp.t_descend:  # descend from hover, riding along with the target
                dz = tp.hover * (1.0 - tau / tp.t_descend)
                v[2] -= tp.hover / tp.t_descend
            return RigidTransform(X.rotation(), X.translation() + [0.0, 0.0, dz]), v
        # Lift while the horizontal velocity decays linearly from the target's to zero, then hold.
        X0 = self.grasp_pose(T + t_grip, k)
        v0 = self.world.target_velocity(T + t_grip).copy()
        v0[2] = 0.0
        s = min(1.0, (tau - t_grip) / tp.t_lift)
        p = X0.translation() + v0 * tp.t_lift * (s - 0.5 * s * s) + [0.0, 0.0, tp.lift_height * s]
        v = v0 * (1.0 - s)
        if s < 1.0:
            v[2] = tp.lift_height / tp.t_lift
        return RigidTransform(X0.rotation(), p), v

    # ---- IK ----------------------------------------------------------------------
    def solve_ik(self, X_WG: RigidTransform, seed: np.ndarray) -> np.ndarray | None:
        plant = self._plant
        ik = InverseKinematics(plant, self._ctx, with_joint_limits=True)
        q = ik.q()
        p = X_WG.translation()
        tol = self.ik_pos_tol
        ik.AddPositionConstraint(self.robot.G, np.zeros(3), self.robot.W, p - tol, p + tol)
        ik.AddOrientationConstraint(self.robot.W, X_WG.rotation(), self.robot.G, RotationMatrix(), self.ik_rot_tol)
        prog = ik.get_mutable_prog()
        full_seed = np.zeros(self._nq_plant)
        full_seed[self._i_iiwa] = seed
        other = np.setdiff1d(np.arange(self._nq_plant), self._i_iiwa)
        if len(other):
            prog.AddBoundingBoxConstraint(full_seed[other], full_seed[other], q[other])
        prog.AddQuadraticErrorCost(np.eye(NQ) * 0.1, seed, q[self._i_iiwa])
        prog.SetInitialGuess(q, full_seed)
        res = Solve(prog)
        if not res.is_success():
            return None
        return np.asarray(res.GetSolution(q))[self._i_iiwa]

    # ---- tail ----------------------------------------------------------------------
    def tail(self, q_hover: np.ndarray, T: float, k: int) -> JointTrajectory | None:
        """Differential-IK rollout of descend / close / lift / hold from q_hover at T.
        None if it leaves the joint or velocity limits or cannot track the pose."""
        tp, robot = self.tp, self.robot
        self.last_tail_failure = ""
        q = np.asarray(q_hover, float).copy()
        n_steps = round(tp.duration / tp.dt)
        every = max(1, round(tp.knot_dt / tp.dt))
        ts, qs, gs = [T], [q.copy()], [WSG_OPEN]
        for i in range(1, n_steps + 1):
            tau = i * tp.dt
            X_des, v_ff = self.tail_pose(tau, T, k)
            X = robot.fk(q)
            err = _pose_error(X, X_des)
            V = np.concatenate([tp.k_rot * err[:3], tp.k_p * err[3:] + v_ff])
            J = robot.jacobian(q)
            v = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), V)
            if np.any(np.abs(v) > self.v_max):
                self.last_tail_failure = f"velocity limit at tau={tau:.2f}"
                return None
            q = q + v * tp.dt
            if np.any(q < robot.q_lower) or np.any(q > robot.q_upper):
                self.last_tail_failure = f"joint limit at tau={tau:.2f}"
                return None
            if i % every == 0 or i == n_steps:
                if np.linalg.norm(_pose_error(robot.fk(q), X_des)[3:]) > tp.max_track_err:
                    self.last_tail_failure = f"tracking error at tau={tau:.2f}"
                    return None
                ts.append(T + tau)
                qs.append(q.copy())
                gs.append(WSG_OPEN if tau < tp.t_settle + tp.t_descend else WSG_CLOSED)
        return JointTrajectory(np.array(ts), np.array(qs), np.array(gs))

    # ---- redundancy: the elbow's self-motion ----------------------------------------
    def self_motion(self, q: np.ndarray, X_des: RigidTransform, n_steps: int = 6, step: float = 0.2):
        """Configurations with the same gripper pose and a different elbow (iiwa swivel):
        walk along the 1-D null space of the Jacobian in both directions, re-projecting
        onto the pose after every step.  Ordered by distance from q: +1, -1, +2, -2, ..."""
        robot = self.robot
        branches = []
        for sign in (1.0, -1.0):
            out, qv, prev = [], np.asarray(q, float).copy(), None
            for _ in range(n_steps):
                J = robot.jacobian(qv)
                n = np.linalg.svd(J)[2][-1]
                if prev is not None and n @ prev < 0:
                    n = -n
                elif prev is None:
                    n = sign * n
                prev = n
                qv = qv + step * n
                for _ in range(4):  # back onto the pose
                    err = _pose_error(robot.fk(qv), X_des)
                    Jv = robot.jacobian(qv)
                    qv = qv + Jv.T @ np.linalg.solve(Jv @ Jv.T + 1e-6 * np.eye(6), err)
                if np.any(qv < robot.q_lower) or np.any(qv > robot.q_upper):
                    break
                if np.linalg.norm(_pose_error(robot.fk(qv), X_des)[3:]) > 2 * self.ik_pos_tol:
                    break
                out.append(qv.copy())
            branches.append(out)
        merged = []
        for i in range(max(len(b) for b in branches)):
            merged += [b[i] for b in branches if i < len(b)]
        return merged

    # ---- everything ---------------------------------------------------------------
    def start(self, seeds: list[np.ndarray], rng: np.random.Generator, n_random_seeds: int = 1) -> None:
        """Begin an (incremental) goal search; call `generate_for(T, ...)` for each T after."""
        lo, hi = self.robot.q_lower, self.robot.q_upper
        family_seeds = [np.asarray(s, float) for s in seeds] + [rng.uniform(lo, hi) for _ in range(n_random_seeds)]
        self._seeds = {(k, f): s for k in range(len(self.grasps)) for f, s in enumerate(family_seeds)}
        self.stats = {"tried": 0, "ik_fail": 0, "duplicate": 0, "pregrasp_collides": 0, "tail_infeasible": 0,
                      "tail_collides": 0, "rescued_by_swivel": 0, "goals": 0}

    def generate_for(self, T: float, tail_free, state_free, max_per_T: int = 3,
                     swivel_steps: int = 6) -> list[GoalCandidate]:
        """Goal candidates at intercept time T, over grasps and IK families.

        Within a family the IK at T is seeded with the family's solution at the
        previous T, so a family is a consistent branch of the redundant arm.  When a
        family's pre-grasp or tail collides, its elbow self-motion is searched for a
        posture that is free (the obstacles mostly meet the elbow, not the hand)."""
        st, out = self.stats, []
        for (k, f), seed in list(self._seeds.items()):
            if len(out) >= max_per_T:
                break
            st["tried"] += 1
            X_hover = self.hover_pose(T, k)
            q = self.solve_ik(X_hover, seed)
            if q is None:
                st["ik_fail"] += 1
                continue
            self._seeds[(k, f)] = q
            if any(c.grasp_index == k and np.max(np.abs(c.q - q)) < 0.2 for c in out):
                st["duplicate"] += 1
                continue
            reason, cand = None, None
            for i, qc in enumerate([q] + self.self_motion(q, X_hover, swivel_steps) if swivel_steps else [q]):
                if any(c.grasp_index == k and np.max(np.abs(c.q - qc)) < 0.2 for c in out):
                    continue
                if not state_free(qc, T):
                    reason = reason or "pregrasp_collides"
                    continue
                tail = self.tail(qc, T, k)
                if tail is None:
                    reason = reason or "tail_infeasible"
                    continue
                if not tail_free(tail):
                    reason = reason or "tail_collides"
                    continue
                cand = GoalCandidate(T=float(T), q=qc, tail=tail, grasp_index=k, family=f)
                if i > 0:
                    st["rescued_by_swivel"] += 1
                break
            if cand is None:
                st[reason or "duplicate"] += 1
                continue
            out.append(cand)
        st["goals"] += len(out)
        return out

    def generate(
        self,
        Ts: np.ndarray,
        seeds: list[np.ndarray],
        tail_free,  # callable(JointTrajectory) -> bool
        state_free,  # callable(q, t) -> bool
        rng: np.random.Generator,
        n_random_seeds: int = 1,
        **kwargs,
    ) -> list[GoalCandidate]:
        """All goal candidates for the intercept times Ts (see generate_for)."""
        self.start(seeds, rng, n_random_seeds)
        out: list[GoalCandidate] = []
        for T in Ts:
            out += self.generate_for(T, tail_free, state_free, **kwargs)
        return out
