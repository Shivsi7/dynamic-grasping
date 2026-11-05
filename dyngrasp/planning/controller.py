"""(B) SpaceTimeRRTController: deliberative space-time planning with replanning.

Cycle (every `replan_period` s of sim time, except while committed to a grasp,
i.e. from `commit_time` before the intercept until the end of the grasp tail):

1.  Planning starts from where the *executing* trajectory will be when this
    computation is finished: (q_s, t_s) = (traj(t + budget), t + budget).  The
    executing trajectory up to t_s is kept as the prefix, so under the
    harness's latency model the new plan is accepted as long as planning takes
    less than `budget` of wall time; a late plan is rejected by the wrapper
    and the old one keeps running (see controller.py).
2.  Goals: pre-grasp configurations at candidate intercept times T, each with a
    collision-free descend/close/lift tail (goals.py).  They are generated
    incrementally, earliest T first, with a share of every replan's budget
    (`goal_share`), so a hard scene never blows the budget; with exact
    prediction they do not go stale.
3.  If the current plan is still valid, only look for an earlier intercept
    (`improve`); otherwise plan to any reachable goal.
4.  If no grasp plan can be found and the current plan is no longer valid,
    plan an escape to a state that is safe to hold, and retry later.
5.  Within `commit_time` of T, and during the tail, stop replanning.

Nothing here is specific to a scenario preset: all inputs are the WorldModel,
the scene geometry and the robot model.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field

import numpy as np

from dyngrasp.controller import WSG_OPEN, Command, Controller, RobotModel, RobotState
from dyngrasp.grasps import generate_grasps
from dyngrasp.planning.collision import SpaceTimeChecker
from dyngrasp.planning.goals import GoalCandidate, GoalGenerator, TailParams
from dyngrasp.planning.st_rrt import (
    SpaceTimeRRTConnect,
    STRRTParams,
    min_duration,
    plan_to_safe_state,
)
from dyngrasp.trajectory import JointTrajectory, hermite, motion_limits
from dyngrasp.world import WorldModel


# Reference joint-acceleration limits of the iiwa (rad/s^2).  KUKA publishes none; these are
# the per-joint limits used for an iiwa in arXiv 2407.13423 (cf. the joint torques: 176 N m at
# joints 1-2, 110 at 3-5, 40 at 6-7; joint 2 lifts the whole arm).  The planner uses fractions.
IIWA_ACCEL_REF = (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0)


def _scaled(f: float) -> tuple[float, ...]:
    return tuple(round(f * a, 3) for a in IIWA_ACCEL_REF)


def accel_vector(a, n: int = 7) -> np.ndarray | None:
    """A scalar or per-joint acceleration limit as a per-joint array (None stays None)."""
    return None if a is None else np.broadcast_to(np.asarray(a, float), (n,)).copy()


@dataclass
class STRRTConfig:
    v_scale: float = 0.5  # planner joint-velocity limits, as a fraction of the iiwa's
    tail_v_scale: float = 0.9  # the grasp tail may move faster (short, smooth motion)
    # Joint-acceleration limits (rad/s^2), per joint or one number for all joints: fractions of
    # IIWA_ACCEL_REF (None disables the acceleration-aware planning and smoothing).
    a_max: float | tuple | None = _scaled(0.6)  # planner (transfer and escape motions)
    tail_a_max: float | tuple = _scaled(0.8)  # the grasp tail
    # While the object may be in the hand (after the grasp), move gently so it is not flung;
    # if no gentle escape exists, the planner limits are used rather than collide.
    v_scale_holding: float = 0.25
    a_max_holding: float | tuple = _scaled(0.2)
    escape_fallback: bool = True  # holding: fall back to the planner limits if no gentle escape exists
    torque_check: bool = True  # executed motions must also stay within the joint torque limits
    torque_margin: float = 0.9  # ... with this fraction of them (payload, model error)
    budget: float = 0.25  # wall-clock budget of a replan = how far ahead it plans from (s)
    initial_budget: float = 1.5  # first plan, robot at rest (includes goal generation)
    replan_period: float = 0.5  # sim time between replans
    commit_time: float = 0.4  # no replanning within this long of the intercept time
    improve: bool = True  # replans look for an earlier intercept while the current plan is valid
    validate: bool = True  # re-check the remaining plan against the world model at each replan
    hold_horizon: float = 1.5  # escape target: holding still for this long is collision-free
    # After the grasp tail, holding its end pose must be collision-free for this long: time for
    # a replan (budget) plus a gentle escape.  Without it the arm could finish the grasp in an
    # obstacle's path, with too little time to get out at the gentle holding limits.
    post_grasp_hold: float = 1.0
    shortcut_share: float = 0.3  # share of the remaining budget spent on shortcutting
    dT: float = 0.25  # spacing of candidate intercept times
    max_goal_times: int = 16
    n_random_ik_seeds: int = 1
    swivel_steps: int = 6  # elbow self-motion search when a pre-grasp / tail collides
    max_goals_per_T: int = 3
    max_ik_per_T: int = 10  # IK attempts per intercept time
    ik_restarts: int = 3  # random IK restarts per grasp until it first succeeds
    max_tails: int = 3  # tail rollouts per IK attempt (elbow self-motion search)
    goal_share: float = 0.6  # max share of a replan's budget spent generating goals
    max_budget: float = 1.0  # while holding safely without a grasp plan, failed replans double the budget up to this
    goal_work_cap: float = 0.5  # wall time of a goal-work cycle (only while the current plan is safe)
    margin_obstacle: float = 0.03
    margin_fallbacks: tuple[float, ...] = (0.01, 0.0)  # used only when the start state violates the margin
    margin_env: float = 0.005
    joint_margin: float = 0.05  # keep sampled configurations this far inside the joint limits
    rrt: STRRTParams = field(default_factory=STRRTParams)
    tail: TailParams = field(default_factory=TailParams)
    grasp_set: str = "all"  # "all" (dyngrasp.grasps candidates) | "nominal" (objects.X_UG_grasp only)
    # Families used with grasp_set="all".  Side grasps are off by default: on this setup they
    # are rarely reachable and mostly fail physically (scripts/grasp_feasibility.py --execute).
    grasp_families: tuple[str, ...] = ("top", "top_tilt")
    seed: int | None = None
    # Wall-clock compute per unit of planning budget.  1 = real time.  An offline "oracle"
    # (scripts/certify.py) uses e.g. 20 with latency="none": same planner, 20x the compute.
    wall_scale: float = 1.0


@dataclass
class _Plan:
    traj: JointTrajectory
    mode: str  # "grasp" | "escape" | "hold"
    goal: GoalCandidate | None = None


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
        self.v_max = cfg.v_scale * robot.v_limits
        grasps = self.grasps
        if grasps is None and cfg.grasp_set == "all":
            grasps = [g for g in generate_grasps(world.object) if g.family in cfg.grasp_families]
        self.goalgen = GoalGenerator(world, robot, cfg.tail_v_scale * robot.v_limits, cfg.tail, grasps,
                                     a_max=None if cfg.a_max is None else accel_vector(cfg.tail_a_max))
        seed = cfg.seed if cfg.seed is not None else (
            world.spec.seed if world.spec.seed is not None else zlib.crc32(world.spec.name.encode())
        )
        self.rng = np.random.default_rng(seed)
        lo = robot.q_lower + cfg.joint_margin
        hi = robot.q_upper - cfg.joint_margin
        self.rrt = SpaceTimeRRTConnect(lo, hi, self.v_max, self.checker.edge_free, cfg.rrt, self.rng)
        self.a_max = accel_vector(cfg.a_max, len(lo))
        self.a_hold = None if cfg.a_max is None else accel_vector(cfg.a_max_holding, len(lo))
        self.rrt.a_max = self.a_max
        self.goals: list[GoalCandidate] = []
        self.pending_T: list[float] | None = None  # intercept times not yet processed
        self._cost_per_T = 0.0
        self._budget_now = self.cfg.budget  # running estimate of the wall time to process one intercept time
        self.plans: dict[int, _Plan] = {}  # id(trajectory) -> plan, for trajectories we issued
        self.next_replan = 0.0
        self.grasp_done = False  # a grasp tail has been executed (only escapes after that)
        self.plan_log: list[dict] = []

    # ---- helpers ----------------------------------------------------------------------
    def _executing(self, state: RobotState) -> _Plan | None:
        if state.trajectory is None:
            return None
        return self.plans.get(id(state.trajectory))

    def _deadline_left(self, wall0: float, budget: float) -> float:
        """Wall time left of `budget` (sim-time seconds, scaled by cfg.wall_scale)."""
        return budget * self.cfg.wall_scale - (time.perf_counter() - wall0)

    def _remaining_valid(self, traj: JointTrajectory, t_s: float) -> bool:
        """The executing trajectory from t_s on, and holding its last configuration
        for `hold_horizon` after it ends, are collision-free."""
        keep = traj.t > t_s + 1e-9
        t_end = max(t_s, traj.end_time)
        if np.any(keep):
            Q = np.vstack([traj.value(t_s), traj.q[keep]])
            T = np.concatenate([[t_s], traj.t[keep]])
            # dense (10 ms) trajectories: check at the checker's time resolution
            step = max(1, int(round(self.checker.dt_res / max(float(np.median(np.diff(T))), 1e-6))))
            idx = np.unique(np.append(np.arange(0, len(T), step), len(T) - 1))
            if not self.checker.path_free(Q[idx], T[idx]):
                return False
        return self.checker.hold_free(traj.q[-1], t_end, self.cfg.hold_horizon)

    def _generate_goals(self, t_s: float, q_s: np.ndarray, wall0: float, budget: float, share: float | None = None) -> None:
        """Process pending intercept times (earliest first) until the goal share of the budget is used."""
        cfg = self.cfg
        if self.pending_T is None:
            self.pending_T = list(self.goalgen.candidate_times(t_s, cfg.dT, cfg.max_goal_times))
            self.goalgen.start([q_s, np.asarray(self.world.spec.q0, float)], self.rng, cfg.n_random_ik_seeds)
        # A single T cannot be interrupted, so only start one if its expected cost fits.
        share = cfg.goal_share if share is None else share
        first_T = True
        while self.pending_T and (first_T and share >= 1.0 or self._deadline_left(wall0, share * budget) > self._cost_per_T):
            first_T = False
            T = self.pending_T.pop(0)
            if T < t_s:
                continue
            w = time.perf_counter()
            self.goals += self.goalgen.generate_for(
                T, self._tail_free, self.checker.state_free,
                cfg.max_goals_per_T, cfg.swivel_steps, cfg.max_ik_per_T, cfg.ik_restarts, cfg.max_tails,
            )
            w = time.perf_counter() - w
            self._cost_per_T = max(w, 0.7 * self._cost_per_T + 0.3 * w)  # pessimistic running estimate

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

    def _replan(self, state: RobotState, current: _Plan | None) -> _Plan | None:
        """Plan with the configured margin; if the start state already violates it (an
        obstacle came close), plan with smaller margins rather than freezing in place."""
        cfg = self.cfg
        margins = [m for m in cfg.margin_fallbacks if m < cfg.margin_obstacle]
        q_probe = state.q_cmd if state.trajectory is None else state.trajectory.value(state.t + cfg.budget)
        self.checker.margin_obstacle = cfg.margin_obstacle
        if not margins or self.checker.state_free(q_probe, state.t + cfg.budget):
            return self._replan_with_margin(state, current)
        for m in margins:
            self.checker.margin_obstacle = m
            if self.checker.state_free(q_probe, state.t + cfg.budget):
                break
        try:
            plan = self._replan_with_margin(state, current)
            if self.plan_log:
                self.plan_log[-1]["margin"] = self.checker.margin_obstacle
            return plan
        finally:
            self.checker.margin_obstacle = cfg.margin_obstacle

    def _replan_with_margin(self, state: RobotState, current: _Plan | None) -> _Plan | None:
        cfg = self.cfg
        wall0 = time.perf_counter()
        t = state.t
        first = state.trajectory is None
        # Without a grasp plan, while the executing plan is a safe hold, a longer budget
        # (planning from further ahead) costs nothing: grow it after every failure.
        no_grasp = current is None or current.mode != "grasp"
        budget = cfg.initial_budget if first else (self._budget_now if no_grasp else cfg.budget)
        if not first:
            # The executing plan (or the hold after it) will collide at t_hit: plan from halfway
            # there.  Soon -> react now with the short budget; later -> take the time, so a hard
            # escape (e.g. gentle, with the object in hand) is found rather than timed out.
            t_hit = self._time_to_collision(state.trajectory, t, t + 2 * cfg.max_budget + cfg.hold_horizon)
            if np.isfinite(t_hit):
                budget = float(np.clip(0.5 * (t_hit - t), cfg.budget, cfg.max_budget))
                if budget == cfg.budget:
                    self._budget_now = cfg.budget
        if first:
            # The arm holds still while the first plan is computed: if an obstacle would hit
            # it there, leave enough time to get out of the way.
            t_hit = self.checker.first_collision_time(state.q_cmd, t, t + budget + cfg.hold_horizon)
            if np.isfinite(t_hit):
                budget = float(np.clip(0.5 * (t_hit - t), 0.05, budget))
        t_s = t + budget
        if first:
            q_s = state.q_cmd.copy()
            v_s = np.zeros_like(q_s)
            prefix = JointTrajectory([t, t_s], [q_s, q_s], [WSG_OPEN, WSG_OPEN])
        else:
            q_s = state.trajectory.value(t_s)
            v_s = state.trajectory.velocity(t_s) if self.a_max is not None else None
            prefix = state.trajectory.slice_until(t_s)
        log = {"t": t, "t_s": t_s, "first": first}

        current_valid = current is not None and (
            not cfg.validate or self._remaining_valid(state.trajectory, t_s)
        )
        log["current_valid"] = current_valid

        # The current plan is unsafe: secure a cheap structured escape first (it is the fallback
        # if no grasp plan is found in what is left of the budget).
        fallback = None
        if not current_valid:
            g_esc = prefix.gripper_at(t_s) if prefix.gripper is not None else WSG_OPEN
            for v_lim, a_lim in self._escape_limits():
                prim = self._escape_primitive(q_s, v_s, t_s, g_esc, wall0, 0.5 * budget, v_lim, a_lim)
                if prim is not None:
                    fallback = _Plan(JointTrajectory.concatenate([prefix, prim]), "escape")
                    log["escape_limits"] = "gentle" if a_lim is self.a_hold else "planner"
                    break

        if not self.grasp_done and (self.pending_T is None or self.pending_T):
            # A goal the arm can still reach, and no grasp plan yet: plan to it now (goal
            # generation then only gets its budget share) instead of spending whole cycles on
            # later intercept times while this goal goes stale.
            has_target = not (current_valid and current.mode == "grasp") and any(
                g.T - t_s >= min_duration(q_s, g.q, self.v_max) + 1e-6 for g in self.goals)
            if current_valid and self.pending_T and not has_target:
                # Goal-work cycle: the executing plan is safe, so this cycle may run over
                # its budget (the arm keeps executing meanwhile); it returns no plan.
                self._generate_goals(t_s, q_s, wall0, cfg.goal_work_cap, share=1.0)
                log.update(n_goals=len(self.goals), pending_T=len(self.pending_T),
                           goal_gen_s=time.perf_counter() - wall0)
                self.next_replan_now = True
                self._log(log, wall0, "goal_work")
                return None
            self._generate_goals(t_s, q_s, wall0, budget)
            log.update(n_goals=len(self.goals), pending_T=len(self.pending_T), goal_gen_s=time.perf_counter() - wall0)

        goals = [g for g in self.goals if g.T - t_s >= min_duration(q_s, g.q, self.v_max) + 1e-6]
        if self.grasp_done:
            goals = []  # holding (or missed) the object: only stay safe
        if current_valid and current.mode == "grasp":
            if not cfg.improve:
                self._log(log, wall0, "keep")
                return None
            goals = [g for g in goals if g.T < current.goal.T - cfg.dT / 2]

        result = None
        if goals:
            left = self._deadline_left(wall0, 0.85 * budget)
            res = self.rrt.plan(
                q_s, t_s, np.array([g.q for g in goals]), np.array([g.T for g in goals]),
                time_budget=max(0.0, (1.0 - cfg.shortcut_share) * left),
                v_start=v_s, goal_v=np.array([self._tail_v0(g) for g in goals]),
            )
            log.update(rrt_ok=res.success, rrt_reason=res.reason, nodes=res.nodes, edges=res.edge_checks,
                       rrt_wall=res.wall)
            if res.success:
                g = goals[res.goal_index]
                v_g = self._tail_v0(g)
                Q, T = self.rrt.shortcut(res.path_q, res.path_t, max(0.0, self._deadline_left(wall0, 0.85 * budget)),
                                         v_start=v_s, v_end=v_g)
                path, tail = self._smooth(T, Q, v_s, v_g, WSG_OPEN), self._smooth_tail(g)
                why = self._verify(path, self.v_max, self.a_max)
                log.update(T_goal=g.T, knots=len(T), grasp=g.grasp_name, accel_rejects=res.stats.get("accel_rejects", 0))
                if why:
                    log["smooth_reject"] = why  # the smoothed curve failed a check: no plan this cycle
                else:
                    result = _Plan(JointTrajectory.concatenate([prefix, path, tail]), "grasp", g)

        if result is None and fallback is not None:
            result = fallback
            log.update(escape_ok=True, escape="primitive", holding=self.grasp_done)
        if result is None and not current_valid:
            # No grasp plan and the current one is unsafe (or there is none): get out of the way.
            left = self._deadline_left(wall0, 0.85 * budget)
            # with the object (possibly) in hand: slower and gentler, so it is not flung
            v_lim, a_lim = self._escape_limits()[0]
            self.rrt.v_max, self.rrt.a_max = v_lim, a_lim
            try:
                esc = plan_to_safe_state(
                    self.rrt, q_s, t_s,
                    lambda q, tt: self.checker.hold_free(q, tt, cfg.hold_horizon),
                    t_max=t_s + 3.0, time_budget=max(0.0, left), v_start=v_s,
                )
            finally:
                self.rrt.v_max, self.rrt.a_max = self.v_max, self.a_max
            log.update(escape_ok=esc.success, holding=self.grasp_done)
            if esc.success:
                parts = [prefix]
                if len(esc.path_t) > 1:
                    g = prefix.gripper_at(t_s) if prefix.gripper is not None else WSG_OPEN
                    path = self._smooth(esc.path_t, esc.path_q, v_s, None, g)
                    why = self._verify(path, v_lim, a_lim)
                    if why:
                        log["smooth_reject"] = why
                        parts = None
                    else:
                        parts.append(path)
                if parts is not None:
                    result = _Plan(JointTrajectory.concatenate(parts), "escape" if len(esc.path_t) > 1 else "hold")

        if result is not None and result.mode == "grasp":
            self._budget_now = cfg.budget
        elif no_grasp and current_valid:
            self._budget_now = min(cfg.max_budget, 2.0 * self._budget_now)
        log["budget"] = budget
        self._log(log, wall0, "new" if result else "none")
        if result is not None:
            self.plans[id(result.traj)] = result
        return result

    def _time_to_collision(self, traj: JointTrajectory, t0: float, t1: float) -> float:
        """First time in [t0, t1] at which following `traj` (and then holding its end) is not
        collision-free, at the checker's time resolution (inf if none)."""
        q_end = traj.q[-1]
        for tt in np.arange(t0, t1 + 1e-9, self.checker.dt_res):
            q = traj.value(tt) if tt < traj.end_time else q_end
            if not self.checker.state_free(q, float(tt)):
                return float(tt)
        return np.inf

    def _escape_limits(self) -> list:
        """(v, a) limits for escapes: with the object in hand, gentle first, then the planner's
        (shaking the object beats hitting an obstacle); otherwise the planner's."""
        if self.grasp_done and self.a_max is not None:
            gentle = (self.cfg.v_scale_holding * self.robot.v_limits, self.a_hold)
            return [gentle, (self.v_max, self.a_max)] if self.cfg.escape_fallback else [gentle]
        return [(self.v_max, self.a_max)]

    def _tail_free(self, tail: JointTrajectory) -> bool:
        """The grasp tail is collision-free, and so is holding its end pose for `post_grasp_hold`:
        the arm ends the grasp with the object in hand somewhere it can stay while it plans
        its next (gentle) move, not in an obstacle's path."""
        return (self.checker.path_free(tail.q, tail.t)
                and (self.cfg.post_grasp_hold <= 0
                     or self.checker.hold_free(tail.q[-1], tail.t[-1], self.cfg.post_grasp_hold)))

    # ---- evasive motion primitives -----------------------------------------------------
    def _escape_primitive(self, q_s, v_s, t_s, gripper, wall0, budget, v_lim, a_lim):
        """Structured escapes, tried before the random-tree escape: a single smooth curve from
        (q_s, v_s) to a target at rest (brake in place, go home, swing the base / shoulder /
        elbow away, random nearby targets), fastest first.  Accepted if it respects the limits,
        is collision-free, and holding the target afterwards is collision-free."""
        cfg, robot = self.cfg, self.robot
        v_s = np.zeros_like(q_s) if v_s is None else v_s
        lo, hi = robot.q_lower + cfg.joint_margin, robot.q_upper - cfg.joint_margin
        offsets = []
        # single joints: base, shoulder, elbow (the links that meet obstacles), small and large
        for j, mags in ((0, (0.5, 1.0)), (1, (0.4, 0.8, 1.0)), (3, (0.4, 0.8, 1.0))):
            for d in mags:
                for sgn in (1.0, -1.0):
                    e = np.zeros(7)
                    e[j] = sgn * d
                    offsets.append(e)
        # shoulder back + elbow folded, either way (pulls the whole arm in)
        for s1 in (1.0, -1.0):
            for s3 in (1.0, -1.0):
                e = np.zeros(7)
                e[1], e[3] = s1 * 0.6, s3 * 0.6
                offsets.append(e)
        offsets += [self.rng.normal(0.0, 0.6, 7) for _ in range(10)]
        targets = [("brake", None), ("home", np.asarray(self.world.spec.q0, float))] + \
                  [("offset", q_s + o) for o in offsets]
        for D in (0.6, 0.9, 1.2, 1.6):
            for kind, qt in targets:
                if self._deadline_left(wall0, budget) <= 0:
                    return None
                if kind == "brake":
                    qt = q_s + 0.5 * v_s * D  # constant deceleration to rest
                qt = np.clip(qt, lo, hi)
                path = hermite([t_s, t_s + D], [q_s, qt], v_s, None, dt=self.period,
                               gripper=[gripper, gripper])
                if self._verify(path, v_lim, a_lim):
                    continue
                if self.checker.hold_free(qt, t_s + D, cfg.hold_horizon):
                    return path
        return None

    # ---- smoothing --------------------------------------------------------------------
    @staticmethod
    def _tail_v0(g: GoalCandidate) -> np.ndarray:
        """Joint velocity the grasp tail starts with (riding along with the target)."""
        return (g.tail.q[1] - g.tail.q[0]) / (g.tail.t[1] - g.tail.t[0])

    def _smooth(self, T, Q, v0, vN, gripper) -> JointTrajectory:
        """The executed trajectory through the planner's knots: C1 cubic Hermite (same knot
        positions and times) when acceleration limits are on, else the polyline."""
        g = np.full(len(T), gripper)
        if self.a_max is None:
            return JointTrajectory(T, Q, g)
        return hermite(T, Q, v0, vN, dt=self.period, gripper=g)

    def _smooth_tail(self, g: GoalCandidate) -> JointTrajectory:
        """The tail is already the full-rate (10 ms) differential-IK rollout whose velocities and
        accelerations were checked when it was generated: execute it as it is."""
        return g.tail

    def _verify(self, path: JointTrajectory, v_lim, a_lim) -> str:
        """Check the curve actually executed: collision-free (it deviates slightly from the
        planner's straight edges) and within the velocity / acceleration limits."""
        if len(path.t) < 2:
            return ""
        v, a = motion_limits(path)
        # the curve may exceed the planner's (conservative) edge speed a little between knots;
        # what matters is the arm's own limit -- checked with or without acceleration limits
        if np.any(v > 0.95 * self.robot.v_limits):
            return "velocity"
        if a_lim is not None and np.any(a > 1.05 * a_lim + 1e-6):
            return "acceleration"
        if self.cfg.torque_check and not self._torques_ok(path):
            return "torque"
        step = max(1, int(round(self.checker.dt_res / max(path.t[1] - path.t[0], 1e-6))))
        idx = np.unique(np.append(np.arange(0, len(path.t), step), len(path.t) - 1))
        if not self.checker.path_free(path.q[idx], path.t[idx]):
            return "collision"
        return ""

    def _torques_ok(self, path: JointTrajectory, dt: float = 0.02) -> bool:
        """Every joint torque along the path (inverse dynamics of iiwa + gripper, every `dt`)
        within `torque_margin` of its limit."""
        ts = np.arange(path.t[0], path.t[-1] + 1e-9, dt)
        if len(ts) < 3:
            return True
        tau = self.robot.joint_torques(np.array([path.value(t) for t in ts]), dt)
        return bool(np.all(np.abs(tau) <= self.cfg.torque_margin * self.robot.tau_limits))

    def _log(self, log: dict, wall0: float, outcome: str) -> None:
        log["outcome"] = outcome
        log["wall"] = time.perf_counter() - wall0
        self.plan_log.append(log)
