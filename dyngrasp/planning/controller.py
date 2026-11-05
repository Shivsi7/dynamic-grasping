"""(B) SpaceTimeRRTController: deliberative space-time planning with replanning.

Every `replan_period` s of sim time (except while committed to a grasp, from `commit_time`
before the intercept until the end of the grasp tail), `_replan` runs one cycle:

1. Start state.  Planning starts from where the *executing* trajectory will be when this
   computation is done: (q_s, t_s) = (traj(t + budget), t + budget), and that trajectory up
   to t_s is kept as the new plan's prefix.  Under the harness's latency model the new plan
   is accepted if planning takes less than `budget` of wall time; a late plan is rejected
   and the old one keeps running (dyngrasp/controller.py).  The budget is short normally,
   longer while waiting safely, and scaled to the time left when a collision is coming.
2. Is the executing plan still valid?  If not, secure a cheap escape primitive first
   (escapes.py): it is the fallback if nothing better is found this cycle.
3. Goals (goals.py): pre-grasp configurations at candidate intercept times T, each with a
   collision-free descend / close / lift / hold "tail".  They are generated incrementally,
   earliest T first, within a share of each cycle's budget; while the executing plan is safe
   and no goal is reachable yet, a whole cycle may be spent on goals ("goal work").
4. Grasp plan: space-time RRT-Connect (st_rrt.py) from (q_s, t_s) to any reachable goal,
   shortcut, smoothed and verified as executed (motion.py).  While a valid grasp plan is
   executing, only an earlier intercept is looked for (`improve`).
5. Otherwise, if the executing plan is unsafe: the escape primitive from step 2, else a
   random-tree escape to a state that is safe to hold.

After the grasp the arm must end somewhere safe: the tail as is, with a planned retreat
(`_resolve_retreat`), or safe long enough to escape; and the object in hand counts in every
collision check from the moment the gripper closes (`SpaceTimeChecker.attach`).

Nothing here is specific to a scenario preset: all inputs are the WorldModel, the scene
geometry and the robot model.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass

import numpy as np

from dyngrasp.controller import WSG_OPEN, Command, Controller, RobotModel, RobotState
from dyngrasp.grasps import generate_grasps
from dyngrasp.planning.collision import SpaceTimeChecker
from dyngrasp.planning.config import IIWA_ACCEL_REF, STRRTConfig, accel_vector, scaled_accel
from dyngrasp.planning.escapes import EscapePlanner
from dyngrasp.planning.goals import GoalCandidate, GoalGenerator
from dyngrasp.planning.motion import MotionVerifier
from dyngrasp.planning.st_rrt import SpaceTimeRRTConnect, min_duration, plan_to_safe_state
from dyngrasp.trajectory import JointTrajectory
from dyngrasp.world import WorldModel

__all__ = ["IIWA_ACCEL_REF", "Plan", "STRRTConfig", "SpaceTimeRRTController", "accel_vector", "scaled_accel"]


@dataclass
class Plan:
    """A trajectory this controller issued, and what it is for."""

    traj: JointTrajectory
    mode: str  # "grasp" | "escape" | "hold"
    goal: GoalCandidate | None = None  # grasp plans: the goal it ends in
    attach: tuple | None = None  # grasp plans: (X_GO, t_closed), the object held from t_closed on


@dataclass
class _Start:
    """Where a replan starts from."""

    t: float  # now
    t_s: float  # start time of the new part of the plan (= t + budget)
    q_s: np.ndarray
    v_s: np.ndarray | None  # velocity at t_s (None without acceleration limits)
    prefix: JointTrajectory  # the executing trajectory up to t_s (or holding still)
    budget: float
    first: bool  # the very first plan (robot at rest, nothing executing)


class SpaceTimeRRTController(Controller):
    name = "st_rrt"
    period = 0.01

    def __init__(self, config: STRRTConfig | None = None, grasps=None):
        self.cfg = config or STRRTConfig()
        self.grasps = grasps  # GraspCandidates or X_UG frames; default: per cfg.grasp_set

    # ---- setup (untimed: scene-independent work and model building only) --------------
    def reset(self, world: WorldModel, robot: RobotModel, state: RobotState):
        cfg = self.cfg
        self.world, self.robot = world, robot
        self.checker = SpaceTimeChecker(world.spec, world, cfg.margin_obstacle, cfg.margin_env)
        n = len(robot.q_lower)
        # motion limits: transfers / escapes, and gentle ones with the object in hand
        self.v_max = cfg.v_scale * robot.v_limits
        self.a_max = accel_vector(cfg.a_max, n)
        self.v_hold = cfg.v_scale_holding * robot.v_limits
        self.a_hold = None if cfg.a_max is None else accel_vector(cfg.a_max_holding, n)

        grasps = self.grasps
        if grasps is None and cfg.grasp_set == "all":
            grasps = [g for g in generate_grasps(world.object) if g.family in cfg.grasp_families]
        self.goalgen = GoalGenerator(world, robot, cfg.tail_v_scale * robot.v_limits, cfg.tail, grasps,
                                     a_max=None if cfg.a_max is None else accel_vector(cfg.tail_a_max, n))
        seed = cfg.seed if cfg.seed is not None else (
            world.spec.seed if world.spec.seed is not None else zlib.crc32(world.spec.name.encode()))
        self.rng = np.random.default_rng(seed)
        lo, hi = robot.q_lower + cfg.joint_margin, robot.q_upper - cfg.joint_margin
        self.rrt = SpaceTimeRRTConnect(lo, hi, self.v_max, self.checker.edge_free, cfg.rrt, self.rng)
        self.rrt.a_max = self.a_max
        self.verifier = MotionVerifier(robot, self.checker, self.period, smooth=cfg.a_max is not None,
                                       torque_check=cfg.torque_check, torque_margin=cfg.torque_margin)
        self.escapes = EscapePlanner(robot, self.checker, self.verifier, self.rng, world.spec.q0, self.period,
                                     cfg.joint_margin, cfg.hold_horizon)

        self.goals: list[GoalCandidate] = []
        self.pending_T: list[float] | None = None  # intercept times not processed yet (None: not started)
        self._cost_per_T = 0.0  # pessimistic running estimate of the wall time to process one T
        self._budget_now = cfg.budget  # budget while waiting without a grasp plan (grows after failures)
        self._needs_retreat: dict = {}  # id(tail) -> tail: goals accepted pending a retreat (lazy)
        self.plans: dict[int, Plan] = {}  # id(trajectory) -> plan, for trajectories we issued
        self.next_replan = 0.0
        self.next_replan_now = False  # set by a goal-work cycle: replan again right away
        self.grasp_done = False  # a grasp tail has been executed (only escapes after that)
        self._held = None  # attachment of the object to the gripper for collision checks
        self.plan_log: list[dict] = []

    # ---- the controller ----------------------------------------------------------------
    def update(self, state: RobotState, world: WorldModel) -> Command:
        t = state.t
        current = self._executing(state)
        # Committed while executing the grasp; afterwards (held, or missed) safety checks resume.
        committed = (current is not None and current.mode == "grasp"
                     and current.goal.T - self.cfg.commit_time <= t < current.traj.end_time)
        self.grasp_done = self.grasp_done or (current is not None and current.mode == "grasp"
                                              and t >= current.traj.end_time)
        if t + 1e-9 >= self.next_replan and not committed:
            self.next_replan_now = False
            new = self._replan(state, current)
            self.next_replan = t if self.next_replan_now else t + self.cfg.replan_period
            if new is not None:
                return Command(trajectory=new.traj, gripper=WSG_OPEN, info={"phase": "replanned", "mode": new.mode})
        if state.trajectory is not None:
            phase = "tail" if committed and t >= current.goal.T else (current.mode if current else "executing")
            return Command(trajectory=state.trajectory, gripper=WSG_OPEN, info={"phase": phase})
        return Command(q=state.q_cmd, gripper=WSG_OPEN, info={"phase": "waiting"})

    def _executing(self, state: RobotState) -> Plan | None:
        return None if state.trajectory is None else self.plans.get(id(state.trajectory))

    # ---- one replanning cycle ------------------------------------------------------------
    def _replan(self, state: RobotState, current: Plan | None) -> Plan | None:
        """One cycle with the configured margin; if the start state already violates it (an
        obstacle came close), with the largest smaller margin it satisfies, rather than freeze."""
        cfg = self.cfg
        # The object counts as held (it must clear the obstacles too) from the moment the
        # executing grasp plan closes the gripper, and for good once a grasp was executed.
        if current is not None and current.mode == "grasp":
            self._held = current.attach
        elif not self.grasp_done:
            self._held = None
        self.checker.attach(self._held if cfg.check_held_object else None)

        margins = [m for m in cfg.margin_fallbacks if m < cfg.margin_obstacle]
        q_probe = state.q_cmd if state.trajectory is None else state.trajectory.value(state.t + cfg.budget)
        self.checker.margin_obstacle = cfg.margin_obstacle
        if not margins or self.checker.state_free(q_probe, state.t + cfg.budget):
            return self._cycle(state, current)
        for m in margins:
            self.checker.margin_obstacle = m
            if self.checker.state_free(q_probe, state.t + cfg.budget):
                break
        try:
            plan = self._cycle(state, current)
            if self.plan_log:
                self.plan_log[-1]["margin"] = self.checker.margin_obstacle
            return plan
        finally:
            self.checker.margin_obstacle = cfg.margin_obstacle

    def _cycle(self, state: RobotState, current: Plan | None) -> Plan | None:
        cfg = self.cfg
        wall0 = time.perf_counter()
        s = self._start(state, current)
        log = {"t": s.t, "t_s": s.t_s, "first": s.first}

        # 2. is the executing plan still valid?  If not, secure an escape primitive first.
        current_valid = current is not None and (not cfg.validate or self._remaining_valid(state.trajectory, s.t_s))
        log["current_valid"] = current_valid
        fallback = None if current_valid else self._escape_primitive(s, wall0, log)

        # 3. goals
        if not self.grasp_done and (self.pending_T is None or self.pending_T):
            # A goal-work cycle (all of it on goals, no plan) only while the executing plan is
            # safe and no goal is reachable yet: otherwise plan to the reachable goal now, before
            # it goes stale, and give goal generation only its share of the budget.
            has_target = not (current_valid and current.mode == "grasp") and any(self._reachable(g, s) for g in self.goals)
            if current_valid and self.pending_T and not has_target:
                self._generate_goals(s.t_s, s.q_s, wall0, cfg.goal_work_cap, share=1.0)
                log.update(n_goals=len(self.goals), pending_T=len(self.pending_T),
                           goal_gen_s=time.perf_counter() - wall0)
                self.next_replan_now = True  # the arm keeps executing meanwhile; no plan can arrive late
                self._log(log, wall0, "goal_work")
                return None
            self._generate_goals(s.t_s, s.q_s, wall0, s.budget)
            log.update(n_goals=len(self.goals), pending_T=len(self.pending_T), goal_gen_s=time.perf_counter() - wall0)

        # 4. grasp plan
        goals = [] if self.grasp_done else [g for g in self.goals if self._reachable(g, s)]
        if current_valid and current.mode == "grasp":
            if not cfg.improve:
                self._log(log, wall0, "keep")
                return None
            goals = [g for g in goals if g.T < current.goal.T - cfg.dT / 2]  # only an earlier intercept
        result = self._plan_grasp(s, goals, wall0, log) if goals else None

        # 5. otherwise, if the executing plan is unsafe: escape
        if result is None and fallback is not None:
            result = fallback
            log.update(escape_ok=True, escape="primitive", holding=self.grasp_done)
        if result is None and not current_valid:
            result = self._escape_rrt(s, wall0, log)

        # budget of the next cycles: short with a grasp plan; doubled while waiting safely without one
        no_grasp = current is None or current.mode != "grasp"
        if result is not None and result.mode == "grasp":
            self._budget_now = cfg.budget
        elif no_grasp and current_valid:
            self._budget_now = min(cfg.max_budget, 2.0 * self._budget_now)
        log["budget"] = s.budget
        self._log(log, wall0, "new" if result else "none")
        if result is not None:
            self.plans[id(result.traj)] = result
        return result

    def _start(self, state: RobotState, current: Plan | None) -> _Start:
        """1. The budget, and the state the new plan starts from."""
        cfg, t = self.cfg, state.t
        first = state.trajectory is None
        if first:
            # The arm holds still while the first plan is computed: if an obstacle would hit
            # it there, leave enough time to get out of the way.
            budget = cfg.initial_budget
            t_hit = self.checker.first_collision_time(state.q_cmd, t, t + budget + cfg.hold_horizon)
            if np.isfinite(t_hit):
                budget = float(np.clip(0.5 * (t_hit - t), 0.05, budget))
            q_s = state.q_cmd.copy()
            prefix = JointTrajectory([t, t + budget], [q_s, q_s], [WSG_OPEN, WSG_OPEN])
            return _Start(t, t + budget, q_s, np.zeros_like(q_s), prefix, budget, True)
        # Without a grasp plan, while waiting safely, a longer budget (planning from further
        # ahead) costs nothing.  If the executing plan (or the hold after it) will collide at
        # t_hit, plan from halfway there: soon -> react now with the short budget; later -> take
        # the time, so that a hard escape (e.g. gentle, object in hand) is found, not timed out.
        no_grasp = current is None or current.mode != "grasp"
        budget = self._budget_now if no_grasp else cfg.budget
        t_hit = self._time_to_collision(state.trajectory, t, t + 2 * cfg.max_budget + cfg.hold_horizon)
        if np.isfinite(t_hit):
            budget = float(np.clip(0.5 * (t_hit - t), cfg.budget, cfg.max_budget))
            if budget == cfg.budget:
                self._budget_now = cfg.budget
        t_s = t + budget
        v_s = state.trajectory.velocity(t_s) if self.a_max is not None else None
        return _Start(t, t_s, state.trajectory.value(t_s), v_s, state.trajectory.slice_until(t_s), budget, False)

    def _plan_grasp(self, s: _Start, goals: list[GoalCandidate], wall0: float, log: dict) -> Plan | None:
        """4. RRT-Connect to any of `goals`, shortcut, smooth, verify; the prefix + path + tail."""
        cfg = self.cfg
        left = self._deadline_left(wall0, 0.85 * s.budget)
        res = self.rrt.plan(
            s.q_s, s.t_s, np.array([g.q for g in goals]), np.array([g.T for g in goals]),
            time_budget=max(0.0, (1.0 - cfg.shortcut_share) * left),
            v_start=s.v_s, goal_v=np.array([self._tail_v0(g) for g in goals]),
        )
        log.update(rrt_ok=res.success, rrt_reason=res.reason, nodes=res.nodes, edges=res.edge_checks, rrt_wall=res.wall)
        if not res.success:
            return None
        g = goals[res.goal_index]
        if not self._resolve_retreat(g):
            log["retreat_failed"] = True
            return None
        v_g = self._tail_v0(g)
        Q, T = self.rrt.shortcut(res.path_q, res.path_t, max(0.0, self._deadline_left(wall0, 0.85 * s.budget)),
                                 v_start=s.v_s, v_end=v_g)
        path, why = self.verifier.smooth_verified(T, Q, s.v_s, v_g, WSG_OPEN, self.v_max, self.a_max)
        log.update(T_goal=g.T, knots=len(T), grasp=g.grasp_name, accel_rejects=res.stats.get("accel_rejects", 0))
        if why:
            log["smooth_reject"] = why  # the smoothed curve failed a check: no plan this cycle
            return None
        # the tail is the full-rate rollout that was checked when the goal was made: executed as is
        return Plan(JointTrajectory.concatenate([s.prefix, path, g.tail]), "grasp", g, attach=self._attachment(g.tail))

    # ---- escapes --------------------------------------------------------------------------
    def _escape_limits(self) -> list:
        """(v, a) limits for escapes, in order: with the object in hand, gentle first, then the
        planner's (shaking the object beats hitting an obstacle); otherwise the planner's."""
        if self.grasp_done and self.a_max is not None:
            gentle = (self.v_hold, self.a_hold)
            return [gentle, (self.v_max, self.a_max)] if self.cfg.escape_fallback else [gentle]
        return [(self.v_max, self.a_max)]

    def _escape_primitive(self, s: _Start, wall0: float, log: dict) -> Plan | None:
        """2. The prefix + an escape primitive, within half the budget (None if there is none)."""
        g = s.prefix.gripper_at(s.t_s) if s.prefix.gripper is not None else WSG_OPEN
        deadline = wall0 + 0.5 * s.budget * self.cfg.wall_scale
        for v_lim, a_lim in self._escape_limits():
            prim = self.escapes.primitive(s.q_s, s.v_s, s.t_s, g, deadline, v_lim, a_lim)
            if prim is not None:
                log["escape_limits"] = "gentle" if a_lim is self.a_hold else "planner"
                return Plan(JointTrajectory.concatenate([s.prefix, prim]), "escape")
        return None

    def _escape_rrt(self, s: _Start, wall0: float, log: dict) -> Plan | None:
        """5. Random-tree escape to a state that is safe to hold (st_rrt.plan_to_safe_state)."""
        v_lim, a_lim = self._escape_limits()[0]
        self.rrt.v_max, self.rrt.a_max = v_lim, a_lim
        try:
            esc = plan_to_safe_state(
                self.rrt, s.q_s, s.t_s, lambda q, tt: self.checker.hold_free(q, tt, self.cfg.hold_horizon),
                t_max=s.t_s + 3.0, time_budget=max(0.0, self._deadline_left(wall0, 0.85 * s.budget)), v_start=s.v_s,
            )
        finally:
            self.rrt.v_max, self.rrt.a_max = self.v_max, self.a_max
        log.update(escape_ok=esc.success, holding=self.grasp_done)
        if not esc.success:
            return None
        if len(esc.path_t) == 1:  # already safe where it is
            return Plan(s.prefix, "hold")
        g = s.prefix.gripper_at(s.t_s) if s.prefix.gripper is not None else WSG_OPEN
        path, why = self.verifier.smooth_verified(esc.path_t, esc.path_q, s.v_s, None, g, v_lim, a_lim)
        if why:
            log["smooth_reject"] = why
            return None
        return Plan(JointTrajectory.concatenate([s.prefix, path]), "escape")

    # ---- goals ----------------------------------------------------------------------------
    def _reachable(self, g: GoalCandidate, s: _Start) -> bool:
        """Velocity limits allow reaching goal g from the start state in time (obstacles aside)."""
        return g.T - s.t_s >= min_duration(s.q_s, g.q, self.v_max) + 1e-6

    def _generate_goals(self, t_s: float, q_s: np.ndarray, wall0: float, budget: float,
                        share: float | None = None) -> None:
        """Process pending intercept times (earliest first) while the goal share of the budget
        lasts.  One T cannot be interrupted, so a T is started only if its expected cost fits
        (in a goal-work cycle, share = 1, at least one T is processed)."""
        cfg = self.cfg
        if self.pending_T is None:
            self.pending_T = list(self.goalgen.candidate_times(t_s, cfg.dT, cfg.max_goal_times))
            self.goalgen.start([q_s, np.asarray(self.world.spec.q0, float)], self.rng, cfg.n_random_ik_seeds)
        share = cfg.goal_share if share is None else share
        first_T = True
        while self.pending_T and (first_T and share >= 1.0
                                  or self._deadline_left(wall0, share * budget) > self._cost_per_T):
            first_T = False
            T = self.pending_T.pop(0)
            if T < t_s:
                continue
            w = time.perf_counter()
            self.goals += self.goalgen.generate_for(
                T, self.tail_free, self.checker.state_free, cfg.max_goals_per_T, cfg.swivel_steps,
                cfg.max_ik_per_T, cfg.ik_restarts, cfg.max_tails, finish=self.finish_tail,
            )
            w = time.perf_counter() - w
            self._cost_per_T = max(w, 0.7 * self._cost_per_T + 0.3 * w)

    def _attachment(self, tail: JointTrajectory):
        """(X_GO, t_closed) for a grasp tail: the object is held from the end of the close on."""
        if not self.cfg.check_held_object:
            return None
        tp = self.cfg.tail
        t_c = tail.t[0] + tp.t_settle + tp.t_descend + tp.t_close
        return self.checker.grasp_attachment(tail.value(t_c), t_c)

    def tail_free(self, tail: JointTrajectory) -> bool:
        """A grasp tail is collision-free, the object included once it is grasped."""
        with self.checker.attached(self._attachment(tail)):
            return self.checker.path_free(tail.q, tail.t)

    def finish_tail(self, tail: JointTrajectory) -> JointTrajectory | None:
        """Make sure a grasp tail ends somewhere safe (see STRRTConfig, "after the grasp"):
        returns the tail as it is, the tail with a retreat, or None (reject the goal).  With
        `lazy_retreat` an unsafe end is accepted for now and resolved by `_resolve_retreat`."""
        cfg, ck = self.cfg, self.checker
        if not self.verifier.within_joint_limits(tail):
            return None
        with ck.attached(self._attachment(tail)):
            if cfg.post_grasp_hold <= 0 or ck.hold_free(tail.q[-1], tail.t[-1], cfg.hold_horizon):
                return tail
            if cfg.retreat and cfg.lazy_retreat:
                self._needs_retreat[id(tail)] = tail
                return tail
            if cfg.retreat:
                r = self._plan_retreat(tail)
                if r is not None:
                    return r
            return tail if ck.hold_free(tail.q[-1], tail.t[-1], cfg.post_grasp_hold) else None

    def _resolve_retreat(self, g: GoalCandidate) -> bool:
        """For a goal accepted pending a retreat: plan the retreat now and append it to the
        goal's tail; else keep the tail if its end is safe for `post_grasp_hold`; else drop the
        goal.  True if the goal is usable."""
        if self._needs_retreat.pop(id(g.tail), None) is None:
            return True
        tail = g.tail
        with self.checker.attached(self._attachment(tail)):
            r = self._plan_retreat(tail)
            if r is not None:
                g.tail = r
                return True
            if self.checker.hold_free(tail.q[-1], tail.t[-1], self.cfg.post_grasp_hold):
                return True
        self.goalgen.stats["no_safe_end"] += 1
        self.goals = [x for x in self.goals if x is not g]
        return False

    def _plan_retreat(self, tail: JointTrajectory) -> JointTrajectory | None:
        """The tail with a retreat after the lift: gentle limits first, then the planner's."""
        r = self.escapes.retreat(tail, self.cfg.tail.t_hold, [(self.v_hold, self.a_hold), (self.v_max, self.a_max)],
                                 budget=self.cfg.retreat_budget * self.cfg.wall_scale)
        if r is not None:
            self.goalgen.stats["retreats"] = self.goalgen.stats.get("retreats", 0) + 1
        return r

    @staticmethod
    def _tail_v0(g: GoalCandidate) -> np.ndarray:
        """Joint velocity the grasp tail starts with (riding along with the target)."""
        return (g.tail.q[1] - g.tail.q[0]) / (g.tail.t[1] - g.tail.t[0])

    # ---- checks of the executing plan ------------------------------------------------------
    def _remaining_valid(self, traj: JointTrajectory, t_s: float) -> bool:
        """The executing trajectory from t_s on, and holding its last configuration for
        `hold_horizon` after it ends, are collision-free."""
        keep = traj.t > t_s + 1e-9
        t_end = max(t_s, traj.end_time)
        if np.any(keep):
            Q = np.vstack([traj.value(t_s), traj.q[keep]])
            T = np.concatenate([[t_s], traj.t[keep]])
            # dense (10 ms) trajectories: check at the checker's time resolution
            step = max(1, round(self.checker.dt_res / max(float(np.median(np.diff(T))), 1e-6)))
            idx = np.unique(np.append(np.arange(0, len(T), step), len(T) - 1))
            if not self.checker.path_free(Q[idx], T[idx]):
                return False
        return self.checker.hold_free(traj.q[-1], t_end, self.cfg.hold_horizon)

    def _time_to_collision(self, traj: JointTrajectory, t0: float, t1: float) -> float:
        """First time in [t0, t1] at which following `traj` (and then holding its end) is not
        collision-free, at the checker's time resolution (inf if none)."""
        for tt in np.arange(t0, t1 + 1e-9, self.checker.dt_res):
            q = traj.value(tt) if tt < traj.end_time else traj.q[-1]
            if not self.checker.state_free(q, float(tt)):
                return float(tt)
        return np.inf

    # ---- bookkeeping -------------------------------------------------------------------------
    def _deadline_left(self, wall0: float, budget: float) -> float:
        """Wall time left of `budget` (sim-time seconds, scaled by cfg.wall_scale)."""
        return budget * self.cfg.wall_scale - (time.perf_counter() - wall0)

    def _log(self, log: dict, wall0: float, outcome: str) -> None:
        log["outcome"] = outcome
        log["wall"] = time.perf_counter() - wall0
        self.plan_log.append(log)
