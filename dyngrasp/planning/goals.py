"""Goal roots for the space-time planner: where the robot must be, and when.

For a candidate intercept time T and grasp candidate the goal is a pre-grasp
configuration q_g(T) (the gripper backed off by `hover` along the grasp's
approach direction: above a top grasp, beside a side grasp), and from there a
fixed "tail": ride along, approach while matching the target's velocity,
close, lift, hold.  The
tail is part of the plan and is collision-checked like everything else, so an
obstacle crossing *during* the grasp rules that (T, q_g) out.

Everything is computed from `WorldModel` (target pose and velocity at any
time) and grasp candidates (dyngrasp.grasps, or any X_UG transforms), so it
applies to any scenario and any grasp set.  Per intercept time, candidates are
tried in order of how often they have produced goals so far, then of their
prior score (approach from above / from the robot's side first).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydrake.all import InverseKinematics, RigidTransform, RotationMatrix, Solve

from dyngrasp.controller import NQ, WSG_CLOSED, WSG_OPEN, RobotModel
from dyngrasp.grasps import GraspCandidate, as_candidate, prior_score
from dyngrasp.trajectory import JointTrajectory
from dyngrasp.world import WorldModel


@dataclass
class TailParams:
    hover: float = 0.08  # pre-grasp distance from the grasp pose, back along the approach (m)
    t_settle: float = 0.3  # ride along at the pre-grasp first, so arrival tracking error dies out
    t_descend: float = 0.6  # minimum-jerk approach (slower tails cost success, see docs/planner_B.md)
    t_close: float = 0.4
    t_lift: float = 0.6  # minimum-jerk lift while slowing down from the target's velocity to rest
    lift_height: float = 0.10  # success needs >= 5 cm above the rest height
    # Tried, in order, when the tail with `lift_height` is infeasible or collides: a higher lift
    # gets the object away from obstacles sooner, a lower one needs less acceleration, and which
    # works depends on the scene (scripts/certify.py found scenes for each).
    alt_lift_heights: tuple[float, ...] = (0.15,)
    t_hold: float = 0.6  # hold still (success needs 0.5 s lifted)
    dt: float = 0.01  # differential-IK integration step
    knot_dt: float = 0.05  # spacing of the knots that are collision-checked (the tail itself keeps every step)
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
    grasp_name: str = ""


def _ease(s: float) -> tuple[float, float]:
    """Minimum-jerk profile 10s^3 - 15s^4 + 6s^5 and its derivative: zero velocity *and*
    acceleration at both ends (a smoothstep would jump in acceleration there)."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4


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
        grasps: list[GraspCandidate | RigidTransform] | None = None,
        ik_pos_tol: float = 0.003,
        ik_rot_tol: float = 0.02,
        a_max: np.ndarray | None = None,  # joint-acceleration limits for the tail (None: unchecked)
    ):
        self.world = world
        self.robot = robot
        self.v_max = np.asarray(v_max, float)
        self.a_max = None if a_max is None else np.asarray(a_max, float)
        self.tp = tail or TailParams()
        self.grasps: list[GraspCandidate] = [
            as_candidate(g, i) for i, g in enumerate(grasps if grasps else [world.object.X_UG_grasp])
        ]
        if not grasps:
            self.grasps[0] = GraspCandidate("nominal", "top", world.object.X_UG_grasp, np.nan, np.nan)
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
        return self.world.grasp_pose(t, self.grasps[k].X_UG)

    def approach_W(self, t: float, k: int) -> np.ndarray:
        """Unit approach direction (gripper +y) of grasp k in W at time t."""
        return self.grasp_pose(t, k).rotation().matrix()[:, 1]

    def hover_pose(self, t: float, k: int) -> RigidTransform:
        X = self.grasp_pose(t, k)
        return RigidTransform(X.rotation(), X.translation() - self.tp.hover * self.approach_W(t, k))

    def tail_pose(self, tau: float, T: float, k: int) -> tuple[RigidTransform, np.ndarray]:
        """Desired gripper pose and translational velocity at T + tau."""
        tp = self.tp
        if tau < tp.t_settle:  # ride along at the pre-grasp
            t = T + tau
            return self.hover_pose(t, k), self.world.target_velocity(t).copy()
        T, tau = T + tp.t_settle, tau - tp.t_settle  # the rest is timed from the end of the settle phase
        t_grip = tp.t_descend + tp.t_close
        if tau < t_grip:
            t = T + tau
            X = self.grasp_pose(t, k)
            v = self.world.target_velocity(t).copy()
            off = np.zeros(3)
            if tau < tp.t_descend:  # approach along the grasp's direction (eased), riding along
                a = self.approach_W(t, k)
                e, de = _ease(tau / tp.t_descend)
                off = -tp.hover * (1.0 - e) * a
                v = v + (tp.hover * de / tp.t_descend) * a
            return RigidTransform(X.rotation(), X.translation() + off), v
        # Lift while the horizontal velocity decays linearly from the target's to zero, then hold.
        X0 = self.grasp_pose(T + t_grip, k)
        v0 = self.world.target_velocity(T + t_grip).copy()
        v0[2] = 0.0
        s = min(1.0, (tau - t_grip) / tp.t_lift)
        e, de = _ease(s)
        lift = getattr(self, "_lift", tp.lift_height)
        p = X0.translation() + v0 * tp.t_lift * (s - 0.5 * s * s) + [0.0, 0.0, lift * e]
        v = v0 * (1.0 - s)
        v[2] = lift * de / tp.t_lift
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
        # Polish the IK solution onto the exact pre-grasp pose first, so the rollout's feedback
        # does not start by snapping out the IK tolerance (an acceleration spike).
        X0, _ = self.tail_pose(0.0, T, k)
        for _ in range(6):
            err = _pose_error(robot.fk(q), X0)
            if np.linalg.norm(err) < 1e-6:
                break
            J = robot.jacobian(q)
            q = q + J.T @ np.linalg.solve(J @ J.T + 1e-8 * np.eye(6), err)
        n_steps = round(tp.duration / tp.dt)
        every = max(1, round(tp.knot_dt / tp.dt))
        ts, qs, gs = [T], [q.copy()], [WSG_OPEN]
        v_prev = None
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
            if self.a_max is not None and v_prev is not None and np.any(np.abs(v - v_prev) / tp.dt > self.a_max):
                self.last_tail_failure = f"acceleration limit at tau={tau:.2f}"
                return None
            v_prev = v
            q = q + v * tp.dt
            if np.any(q < robot.q_lower) or np.any(q > robot.q_upper):
                self.last_tail_failure = f"joint limit at tau={tau:.2f}"
                return None
            if (i % every == 0 or i == n_steps) and \
                    np.linalg.norm(_pose_error(robot.fk(q), X_des)[3:]) > tp.max_track_err:
                self.last_tail_failure = f"tracking error at tau={tau:.2f}"
                return None
            # every step is kept: the executed tail is exactly the motion checked against the limits
            ts.append(T + tau)
            qs.append(q.copy())
            gs.append(WSG_OPEN if tau < tp.t_settle + tp.t_descend else WSG_CLOSED)
        return JointTrajectory(np.array(ts), np.array(qs), np.array(gs))

    def check_knots(self, tail: JointTrajectory) -> JointTrajectory:
        """The tail subsampled every `knot_dt` (and its last knot), for collision checking."""
        every = max(1, round(self.tp.knot_dt / self.tp.dt))
        idx = np.unique(np.append(np.arange(0, len(tail.t), every), len(tail.t) - 1))
        return JointTrajectory(tail.t[idx], tail.q[idx], None if tail.gripper is None else tail.gripper[idx])

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
        self._rng = rng
        family_seeds = [np.asarray(s, float) for s in seeds] + [rng.uniform(lo, hi) for _ in range(n_random_seeds)]
        # random seeds only for the first (nominal) grasp; every grasp gets the given seeds
        self._seeds = {(k, f): s for k in range(len(self.grasps)) for f, s in enumerate(family_seeds)
                       if k == 0 or f < len(seeds)}
        self.grasp_goals = np.zeros(len(self.grasps), int)  # goals found per grasp (ordering)
        self.grasp_stats = {g.name: {"tried": 0, "ik_ok": 0, "goals": 0, "restarts": 0} for g in self.grasps}
        self.stats = {"tried": 0, "ik_fail": 0, "duplicate": 0, "pregrasp_collides": 0, "tail_infeasible": 0,
                      "tail_collides": 0, "no_safe_end": 0, "rescued_by_swivel": 0, "goals": 0}

    def _order(self, T: float) -> list[tuple[int, int]]:
        """(grasp, family) keys for time T: grasps that produced goals first, then by prior."""
        X_WU = self.world.target_pose(T)
        p_sh = self._shoulder()
        prio = {k: prior_score(g, X_WU, p_sh) for k, g in enumerate(self.grasps)}
        keys = [key for key in self._seeds if prio[key[0]] > -0.2]  # never from the far side
        return sorted(keys, key=lambda kf: (-self.grasp_goals[kf[0]], kf[0] != 0, -prio[kf[0]], kf[1]))

    def _shoulder(self) -> np.ndarray:
        if not hasattr(self, "_p_shoulder"):
            plant, ctx = self.robot.plant, self.robot.context
            self._p_shoulder = plant.EvalBodyPoseInWorld(
                ctx, plant.GetBodyByName("iiwa_link_2", self.robot.iiwa)).translation().copy()
        return self._p_shoulder

    def generate_for(self, T: float, tail_free, state_free, max_per_T: int = 3,
                     swivel_steps: int = 6, max_ik_per_T: int = 10, ik_restarts: int = 3,
                     max_tails: int = 3, finish=None) -> list[GoalCandidate]:
        """Goal candidates at intercept time T, over grasps and IK families.

        `finish(tail)`, if given, is applied to every collision-free tail: it returns the
        tail to execute (e.g. with a retreat appended) or None to reject it.

        Within a family the IK at T is seeded with the family's solution at the
        previous T, so a family is a consistent branch of the redundant arm.  When a
        family's pre-grasp or tail collides, its elbow self-motion is searched for a
        posture that is free (the obstacles mostly meet the elbow, not the hand)."""
        st, out = self.stats, []
        n_ik = 0
        for k, f in self._order(T):
            if len(out) >= max_per_T or n_ik >= max_ik_per_T:
                break
            seed = self._seeds[(k, f)]
            gs = self.grasp_stats[self.grasps[k].name]
            st["tried"] += 1
            gs["tried"] += 1
            n_ik += 1
            X_hover = self.hover_pose(T, k)
            q = self.solve_ik(X_hover, seed)
            # A grasp that has never had an IK solution (e.g. a side grasp, far from the home
            # wrist) gets a few random restarts; afterwards its family seeds carry over in T.
            # (a budget per grasp, not per T: failing IK is the expensive part)
            restarts = min(ik_restarts - gs["restarts"], max_ik_per_T - n_ik) if gs["ik_ok"] == 0 else 0
            while q is None and restarts > 0:
                restarts -= 1
                n_ik += 1
                gs["restarts"] += 1
                q = self.solve_ik(X_hover, self._rng.uniform(self.robot.q_lower, self.robot.q_upper))
            if q is None:
                st["ik_fail"] += 1
                continue
            gs["ik_ok"] += 1
            self._seeds[(k, f)] = q
            if any(c.grasp_index == k and np.max(np.abs(c.q - q)) < 0.2 for c in out):
                st["duplicate"] += 1
                continue
            reason, cand, n_tails = None, None, 0
            for i, qc in enumerate([q] + self.self_motion(q, X_hover, swivel_steps) if swivel_steps else [q]):
                if n_tails >= max_tails:
                    break
                if any(c.grasp_index == k and np.max(np.abs(c.q - qc)) < 0.2 for c in out):
                    continue
                if not state_free(qc, T):
                    reason = reason or "pregrasp_collides"
                    continue
                n_tails += 1
                tail, why = None, ""
                for lift in (self.tp.lift_height, *self.tp.alt_lift_heights):
                    self._lift = lift
                    tail = self.tail(qc, T, k)
                    if tail is None:
                        why = why or "tail_infeasible"
                    elif not tail_free(self.check_knots(tail)):
                        why, tail = why or "tail_collides", None
                    elif finish is not None and (done := finish(tail)) is None:
                        why, tail = why or "no_safe_end", None
                    else:
                        tail = done if finish is not None else tail
                        break
                self._lift = self.tp.lift_height
                if tail is None:
                    reason = reason or why
                    continue
                cand = GoalCandidate(T=float(T), q=tail.q[0].copy(), tail=tail, grasp_index=k, family=f,
                                     grasp_name=self.grasps[k].name)
                if i > 0:
                    st["rescued_by_swivel"] += 1
                break
            if cand is None:
                st[reason or "duplicate"] += 1
                continue
            out.append(cand)
            self.grasp_goals[k] += 1
            gs["goals"] += 1
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
