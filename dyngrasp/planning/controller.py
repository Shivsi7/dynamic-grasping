"""(B) SpaceTimeRRTController: deliberative space-time planning with replanning.

Cycle (every `replan_period` s of sim time, until committed to a grasp):

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
from dyngrasp.planning.collision import SpaceTimeChecker
from dyngrasp.planning.goals import GoalCandidate, GoalGenerator, TailParams
from dyngrasp.planning.st_rrt import (
    SpaceTimeRRTConnect,
    STRRTParams,
    min_duration,
    plan_to_safe_state,
)
from dyngrasp.trajectory import JointTrajectory
from dyngrasp.world import WorldModel


@dataclass
class STRRTConfig:
    v_scale: float = 0.5  # planner joint-velocity limits, as a fraction of the iiwa's
    tail_v_scale: float = 0.9  # the grasp tail may move faster (short, smooth motion)
    budget: float = 0.25  # wall-clock budget of a replan = how far ahead it plans from (s)
    initial_budget: float = 1.5  # first plan, robot at rest (includes goal generation)
    replan_period: float = 0.5  # sim time between replans
    commit_time: float = 0.4  # no replanning within this long of the intercept time
    improve: bool = True  # replans look for an earlier intercept while the current plan is valid
    validate: bool = True  # re-check the remaining plan against the world model at each replan
    hold_horizon: float = 1.5  # escape target: holding still for this long is collision-free
    shortcut_share: float = 0.3  # share of the remaining budget spent on shortcutting
    dT: float = 0.25  # spacing of candidate intercept times
    max_goal_times: int = 16
    n_random_ik_seeds: int = 1
    swivel_steps: int = 6  # elbow self-motion search when a pre-grasp / tail collides
    max_goals_per_T: int = 3
    goal_share: float = 0.6  # max share of a replan's budget spent generating goals
    max_budget: float = 1.0  # while holding safely without a grasp plan, failed replans double the budget up to this
    goal_work_cap: float = 0.5  # wall time of a goal-work cycle (only while the current plan is safe)
    margin_obstacle: float = 0.03
    margin_fallbacks: tuple[float, ...] = (0.01, 0.0)  # used only when the start state violates the margin
    margin_env: float = 0.005
    joint_margin: float = 0.05  # keep sampled configurations this far inside the joint limits
    rrt: STRRTParams = field(default_factory=STRRTParams)
    tail: TailParams = field(default_factory=TailParams)
    seed: int | None = None


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
        self.grasps = grasps  # list of X_UG grasp frames (default: the object's nominal grasp)

    # ---- setup (untimed: scene-independent work and model building only) --------------
    def reset(self, world: WorldModel, robot: RobotModel, state: RobotState):
        cfg = self.cfg
        self.world, self.robot = world, robot
        self.checker = SpaceTimeChecker(world.spec, world, cfg.margin_obstacle, cfg.margin_env)
        self.v_max = cfg.v_scale * robot.v_limits
        self.goalgen = GoalGenerator(world, robot, cfg.tail_v_scale * robot.v_limits, cfg.tail, self.grasps)
        seed = cfg.seed if cfg.seed is not None else (
            world.spec.seed if world.spec.seed is not None else zlib.crc32(world.spec.name.encode())
        )
        self.rng = np.random.default_rng(seed)
        lo = robot.q_lower + cfg.joint_margin
        hi = robot.q_upper - cfg.joint_margin
        self.rrt = SpaceTimeRRTConnect(lo, hi, self.v_max, self.checker.edge_free, cfg.rrt, self.rng)
        self.goals: list[GoalCandidate] = []
        self.pending_T: list[float] | None = None  # intercept times not yet processed
        self._cost_per_T = 0.0
        self._budget_now = self.cfg.budget  # running estimate of the wall time to process one intercept time
        self.plans: dict[int, _Plan] = {}  # id(trajectory) -> plan, for trajectories we issued
        self.next_replan = 0.0
        self.plan_log: list[dict] = []

    # ---- helpers ----------------------------------------------------------------------
    def _executing(self, state: RobotState) -> _Plan | None:
        if state.trajectory is None:
            return None
        return self.plans.get(id(state.trajectory))

    def _deadline_left(self, wall0: float, budget: float) -> float:
        return budget - (time.perf_counter() - wall0)

    def _remaining_valid(self, traj: JointTrajectory, t_s: float) -> bool:
        """The executing trajectory from t_s on, and holding its last configuration
        for `hold_horizon` after it ends, are collision-free."""
        keep = traj.t > t_s + 1e-9
        t_end = max(t_s, traj.end_time)
        if np.any(keep):
            Q = np.vstack([traj.value(t_s), traj.q[keep]])
            T = np.concatenate([[t_s], traj.t[keep]])
            if not self.checker.path_free(Q, T):
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
                T, lambda tr: self.checker.path_free(tr.q, tr.t), self.checker.state_free,
                cfg.max_goals_per_T, cfg.swivel_steps,
            )
            w = time.perf_counter() - w
            self._cost_per_T = max(w, 0.7 * self._cost_per_T + 0.3 * w)  # pessimistic running estimate

    # ---- the controller ----------------------------------------------------------------
    def update(self, state: RobotState, world: WorldModel) -> Command:
        t = state.t
        current = self._executing(state)
        committed = current is not None and current.mode == "grasp" and t >= current.goal.T - self.cfg.commit_time
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
        if first:
            # The arm holds still while the first plan is computed: if an obstacle would hit
            # it there, leave enough time to get out of the way.
            t_hit = self.checker.first_collision_time(state.q_cmd, t, t + budget + cfg.hold_horizon)
            if np.isfinite(t_hit):
                budget = float(np.clip(0.5 * (t_hit - t), 0.05, budget))
        t_s = t + budget
        if first:
            q_s = state.q_cmd.copy()
            prefix = JointTrajectory([t, t_s], [q_s, q_s], [WSG_OPEN, WSG_OPEN])
        else:
            q_s = state.trajectory.value(t_s)
            prefix = state.trajectory.slice_until(t_s)
        log = {"t": t, "t_s": t_s, "first": first}

        current_valid = current is not None and (
            not cfg.validate or self._remaining_valid(state.trajectory, t_s)
        )
        log["current_valid"] = current_valid

        if self.pending_T is None or self.pending_T:
            if current_valid and self.pending_T:
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
            )
            log.update(rrt_ok=res.success, rrt_reason=res.reason, nodes=res.nodes, edges=res.edge_checks,
                       rrt_wall=res.wall)
            if res.success:
                Q, T = self.rrt.shortcut(res.path_q, res.path_t, max(0.0, self._deadline_left(wall0, 0.85 * budget)))
                g = goals[res.goal_index]
                path = JointTrajectory(T, Q, np.full(len(T), WSG_OPEN))
                traj = JointTrajectory.concatenate([prefix, path, g.tail])
                result = _Plan(traj, "grasp", g)
                log.update(T_goal=g.T, knots=len(T))

        if result is None and not current_valid:
            # No grasp plan and the current one is unsafe (or there is none): get out of the way.
            left = self._deadline_left(wall0, 0.85 * budget)
            esc = plan_to_safe_state(
                self.rrt, q_s, t_s,
                lambda q, tt: self.checker.hold_free(q, tt, cfg.hold_horizon),
                t_max=t_s + 3.0, time_budget=max(0.0, left),
            )
            log.update(escape_ok=esc.success)
            if esc.success:
                parts = [prefix]
                if len(esc.path_t) > 1:
                    parts.append(JointTrajectory(esc.path_t, esc.path_q, np.full(len(esc.path_t), WSG_OPEN)))
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

    def _log(self, log: dict, wall0: float, outcome: str) -> None:
        log["outcome"] = outcome
        log["wall"] = time.perf_counter() - wall0
        self.plan_log.append(log)
