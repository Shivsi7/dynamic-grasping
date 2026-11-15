"""(D) demonstrations from (C): HybridController episodes as (D) observations and actions.

    demo = run_demo(spec, level, cfg)      # (C) on spec; obs (K, 84), actions (K, 8) if it succeeded
    result, obs = replay(spec, demo["actions"], cfg["env"])   # the actions open-loop through RLController

(C) runs as in evaluation (same harness and grading) but with n_workers=1 (episodes run in parallel instead)
and ideal compute latency (latency="none"): a plan takes effect when it is computed, so the demonstrations
do not contain (C)'s planning delay (the cloned policy is fast).  (C) still holds still until it starts its
approach (t_enter - lead_time), which is its design, not a delay.  A plan that finds nothing is retried at
the next 5 ms tick at no simulated cost, so an episode can take long in wall time: after `max_wall_s` it is
abandoned (outcome "aborted").

Conversion (`to_actions`), per policy step k (t_k = k dt), from (C)'s commanded joint positions q_C and
opening g_C at its 5 ms ticks.  RLController commands, at t_k + j x 10 ms (j = 1..10), the ramp of the
action taken at t_k (actions.py), and the step's opening from t_k + 10 ms to t_{k+1} + 10 ms:
* joints: the ramp's end velocity v1 = v_C + e / dt, v_C = (C)'s commanded velocity at t_{k+1} (central
  difference), e = q_C(t_{k+1}) - (q + (v + v_C) dt / 2) the end-position error the ramp would have from the
  commanded (q, v) without the correction.  The correction removes half of e in this step and the rest in
  the next (deadbeat: errors do not accumulate).  The action is v1 / v_max clipped to [-1, 1], applied to a
  shadow ActionProcessor (the same acceleration clip and joint limits), whose state the next step starts from;
* gripper: (C)'s mean commanded opening over the step's command window, mapped linearly to [-1, 1]
  (the inverse of actions.gripper_width).
The observation at t_k is Observer(state, v) on the (C) episode's RobotState at t_k, with the commanded
velocity and opening of the shadow processor: what RLController computes when it replays the actions.

DAgger / DART (`run_takeover`, `Takeover`): a policy drives until a switch time, then (C) (ExpertC: its plan
starts as soon as it is computed, from the robot's moving state, instead of 2 s later from rest) takes over through
(D)'s actions, converted online the same way; its actions label the steps from the switch on.  From the start with
noise on the executed actions (labelled clean): DART.
"""

from __future__ import annotations

import copy
import dataclasses
import time

import numpy as np
from pydrake.all import RigidTransform

from dyngrasp.controller import (
    NQ,
    WSG_CLOSED,
    WSG_OPEN,
    Command,
    Controller,
    RobotModel,
    RobotState,
)
from dyngrasp.harness import run_episode, shared_robot_model
from dyngrasp.hybrid import HybridController
from dyngrasp.planning.config import IIWA_ACCEL_REF
from dyngrasp.rl.actions import ActionProcessor
from dyngrasp.rl.controller import RLController
from dyngrasp.rl.observation import Observer
from dyngrasp.rl.scenes import fingerprint
from dyngrasp.scenario import ScenarioSpec
from dyngrasp.trajectory import JointTrajectory
from dyngrasp.world import WorldModel

C_TICK = HybridController.period  # 5 ms


class Aborted(RuntimeError):
    pass


def make_processor(robot: RobotModel, env_cfg: dict) -> ActionProcessor:
    """The ActionProcessor RLController builds (same limits)."""
    a_max = env_cfg["accel_fraction"] * np.asarray(IIWA_ACCEL_REF)
    return ActionProcessor(robot.q_lower, robot.q_upper, robot.v_limits, a_max, env_cfg["dt"], RLController.period,
                           env_cfg["joint_margin"])


def gripper_action(width) -> np.ndarray:
    """The inverse of actions.gripper_width: opening -> [-1, 1]."""
    return 2.0 * (np.asarray(width, float) - WSG_CLOSED) / (WSG_OPEN - WSG_CLOSED) - 1.0


class Recorder(Controller):
    """A controller as is, recording the commanded (q, opening) of every tick and the RobotState of every policy
    tick.  The state a tick sees holds the command of the tick before; `commands()` lines them up."""

    def __init__(self, inner: Controller, dt: float, max_wall_s: float = np.inf):
        self.inner, self.name, self.period = inner, inner.name, inner.period
        self.every = round(dt / inner.period)
        self.max_wall_s = max_wall_s
        self.seen: list[tuple[np.ndarray, float]] = []
        self.states: list[RobotState] = []

    def reset(self, world, robot, state):
        self.t_start = time.perf_counter()
        self.inner.reset(world, robot, state)

    def update(self, state: RobotState, world) -> Command:
        i = round(state.t / self.period)
        assert i == len(self.seen) and abs(state.t - i * self.period) < 1e-9, (state.t, len(self.seen))
        self.seen.append((state.q_cmd.copy(), float(state.wsg_cmd)))
        if i % self.every == 0:
            self.states.append(dataclasses.replace(state, q=state.q.copy(), v=state.v.copy(), X_WG=RigidTransform(state.X_WG),
                                                   q_cmd=state.q_cmd.copy(), trajectory=None))
        if time.perf_counter() - self.t_start > self.max_wall_s:
            raise Aborted(f"(C) episode over {self.max_wall_s} s of wall time")
        return self.inner.update(state, world)

    def close(self):
        self.inner.close()

    def commands(self) -> tuple[np.ndarray, np.ndarray]:
        """(Q, G): the commanded positions and opening in effect from tick i on (i = 0 .. n_ticks - 2)."""
        Q = np.array([q for q, _ in self.seen[1:]])
        G = np.array([g for _, g in self.seen[1:]])
        return Q, G


def ramp_action(proc: ActionProcessor, q_end, v_end, opening: float) -> np.ndarray:
    """The action whose ramp from (proc.q, proc.v) ends the step at velocity v_end + e / dt, e the end-position error
    from q_end without the correction (see the module doc); the gripper at `opening`."""
    e = q_end - (proc.q + 0.5 * (proc.v + v_end) * proc.dt)
    a = np.empty(NQ + 1)
    a[:NQ] = np.clip((v_end + e / proc.dt) / proc.v_max, -1.0, 1.0)
    a[NQ] = np.clip(gripper_action(opening), -1.0, 1.0)
    return a


def to_actions(Q: np.ndarray, G: np.ndarray, q0, n_steps: int, proc: ActionProcessor, tick: float = C_TICK):
    """(D) actions (n_steps, 8) from commanded positions Q and openings G at ticks of `tick` s (see the module doc).

    Returns (actions, v_before, wsg_before, err): the shadow processor's commanded velocity and opening at each
    step's start (what the observation sees) and the max |ramp - Q| per step over its 10 command ticks (rad)."""
    m, r = round(proc.dt / tick), round(RLController.period / tick)  # ticks per step, per (D) command tick
    pad = (n_steps + 2) * m - len(Q)
    if pad > 0:  # (C)'s last command holds after the episode
        Q, G = np.vstack([Q, np.repeat(Q[-1:], pad, 0)]), np.concatenate([G, np.repeat(G[-1:], pad)])
    proc.reset(q0, WSG_OPEN)
    acts, v_before, wsg_before, err = [], [], [], []
    for k in range(n_steps):
        v_before.append(proc.v.copy())
        wsg_before.append(proc.wsg)
        end = (k + 1) * m
        a = ramp_action(proc, Q[end], (Q[end + 1] - Q[end - 1]) / (2 * tick), G[k * m + r: end + r].mean())
        proc.start(a)
        err.append(float(np.max(np.abs(proc.Q - Q[k * m + r: end + 1: r]))))
        acts.append(a)
    return np.array(acts), np.array(v_before), np.array(wsg_before), np.array(err)


def run_demo(spec: ScenarioSpec, level: int, cfg: dict, max_wall_s: float | None = None) -> dict:
    """(C) on `spec`; for a success also the (D) observations and actions of its policy steps."""
    env_cfg = cfg["env"]
    max_wall_s = cfg["demos"]["max_wall_s"] if max_wall_s is None else max_wall_s
    rec = Recorder(HybridController(n_workers=1), env_cfg["dt"], max_wall_s)
    out = {"scene": spec.name, "level": level, "seed": spec.seed, "fingerprint": fingerprint(spec),
           "spec": spec.to_dict()}
    t0 = time.perf_counter()
    try:
        r = run_episode(spec, rec, keep_log=False, latency="none")
    except Aborted:
        rec.close()
        return {**out, "outcome": "aborted", "wall_s": time.perf_counter() - t0}
    out.update(outcome=r.outcome, c_result=r.row(), wall_s=time.perf_counter() - t0)
    if r.outcome != "success":
        return out
    robot = shared_robot_model()
    Q, G = rec.commands()
    states = rec.states
    acts, v_before, wsg_before, err = to_actions(Q, G, spec.q0, len(states), make_processor(robot, env_cfg))
    observer = Observer(WorldModel(spec), robot)
    obs = np.array([observer(dataclasses.replace(s, wsg_cmd=w), v) for s, v, w in zip(states, v_before, wsg_before)])
    out.update(obs=obs.astype(np.float32), actions=acts.astype(np.float32), track_err_max=float(err.max()),
               action_clipped=int(np.sum(np.abs(acts[:, :NQ]) >= 1.0)))
    return out


def replay(spec: ScenarioSpec, actions: np.ndarray, env_cfg: dict, latency=None):
    """The actions open-loop through RLController (after the last: hold still, same opening).  Fixed one-tick
    latency by default, as in training.  Returns (EpisodeResult, the observations it computed)."""
    seen = []
    hold = np.concatenate([np.zeros(NQ), actions[-1, NQ:]])

    def actor(obs):
        seen.append(obs)
        return actions[len(seen) - 1] if len(seen) <= len(actions) else hold

    r = run_episode(spec, RLController(env_cfg=env_cfg, actor=actor), keep_log=False,
                    latency=env_cfg["latency"] if latency is None else latency)
    return r, np.array(seen)


# ---- (C) as the expert of DAgger / DART: (C) through (D)'s action interface -------------------------------------------


class ExpertC(HybridController):
    """(C) as a takeover's expert: unchanged, except when a grasp plan starts and how long it may compute.  (C) starts
    a plan plan_budget (2 s) after computing it; at zero latency that budget only caps the planner's wall time, and (C)
    taking over a moving robot would hold it still for it.  Here a plan starts at t0 = max(t + one tick,
    t_enter - lead_time), from the state the robot is in then: a robot moving at the switch (the takeover's brake, its
    command if (C) sends nothing) is planned from at its velocity, as (C) plans from a dodge (hybrid.py's v0), not
    stopped first.  In an episode from the start the first plan starts as in (C) (t_plan + budget = t_enter -
    lead_time); retries start at once instead of a budget later.  A plan (all its seeds, run one after the other)
    stops trying further seeds once it has taken plan_cap_s of wall time (config demos.plan_cap_s)."""

    def __init__(self, plan_cap_s: float = np.inf, **kwargs):
        super().__init__(**kwargs)
        self.plan_cap_s, self.cap_end = plan_cap_s, np.inf

    def _run_seeds(self, t, t0, q0, v0, seeds):  # (C)'s sequential seed loop, stopped at the wall-time cap
        def capped():
            for seed in seeds:
                if time.perf_counter() > self.cap_end:
                    return
                yield seed

        return super()._run_seeds(t, t0, q0, v0, capped())

    def update(self, state: RobotState, world) -> Command:  # HybridController.update with the start time above
        t = state.t
        if self.phase == "track" and state.trajectory is not self.sent:
            self.phase = "wait"
        if self.phase in ("track", "hold"):
            self.phase = "track" if t < self.t0 + self.T else "hold"
            return Command(trajectory=self.sent, info={"phase": self.phase})
        moving = state.trajectory
        t_plan = world.t_enter - self.lead_time - self.budget
        if moving is None:
            t_go = max(t, t_plan) + self.budget if self.phase == "wait" else np.inf
            t_hit = self._parked_hit(state.q_cmd, t)
            if t_hit is not None and t + 0.1 < t_hit <= t_go:
                budget = min(self.plan_budget, 0.5 * (t_hit - t))
                self.deadline = time.time() + budget - 0.05
                info, dodge = self._dodge(t, t + budget, state.q_cmd)
                if dodge is not None:
                    return Command(trajectory=dodge, info={"phase": "dodge", "t_hit": t_hit, **info})
                self._hit = (self._hit[0], None)
        stay = Command(q=state.q_cmd, gripper=WSG_OPEN) if moving is None else Command(trajectory=moving)
        stay.info = {"phase": self.phase}
        if self.phase == "no_plan" or t < t_plan:
            return stay
        tic = time.perf_counter()
        self.cap_end = tic + self.plan_cap_s
        t0 = max(t + self.period, world.t_enter - self.lead_time)  # (C): t + self.budget
        if self.phase == "retry" and t0 >= world.t_exit - self.exit_margin:
            self.phase = stay.info["phase"] = "no_plan"
            return stay
        self.deadline = time.time() + self.budget - 0.05
        q0, v0 = (state.q_cmd, np.zeros(NQ)) if moving is None else (moving.value(t0), moving.velocity(t0))
        self._used, self._fine = set(), False
        info, plan = self._first_plan(t, t0, q0, v0, moving) if t0 < world.t_exit - self.exit_margin else ({"ok": False}, None)
        info.update(budget=self.budget, attempt=self.attempt, fine=self._fine)
        if plan is None:
            self.phase, self.attempt = "retry", self.attempt + 1
            self.tried |= self._used
            stay.info = {"phase": "retry", **info}
            return stay
        self.budget = max(self.budget, 1.5 * (time.perf_counter() - tic))
        self.phase, self.sent, self.t0, self.T = "track", plan, t0, info["T"]
        return Command(trajectory=self.sent, info={"phase": "track", **info})


class Takeover(Controller):
    """A policy (obs -> action) until t_switch, then (C) (ExpertC) executed through (D)'s action interface.

    (C) runs on what it commands itself (its q_cmd and trajectory, as the harness would keep them); the robot runs on
    (D) actions: at every policy tick the action whose ramp follows (C)'s active command over the step (ramp_action,
    as to_actions; the command is assumed to hold for the step).  At the switch (a policy tick) the robot may be
    moving: (C) is handed a brake to rest (the (D) ramp of zero joint actions) as if it were its own dodge, i.e. what
    the robot does while (C) has no plan; (C) plans from the brake's state at the plan's start (ExpertC): one tick
    after the switch, from the robot's moving (q, v), once (C)'s approach would have started (t_enter - lead_time);
    before that, as (C) holds still until then, from where the brake has taken it then (at rest unless it is still
    braking).  Planning takes no simulated time (zero latency).  t_switch = 0: (C) from the start; inf: the policy
    alone.
    `steps`: every policy step's (t, obs, executed action, expert action or None).  noise_std > 0 (DART): (C)'s joint
    actions are executed with Gaussian noise (action units) and recorded clean."""

    period = C_TICK

    def __init__(self, policy, env_cfg: dict, t_switch: float, noise_std: float = 0.0, seed: int = 0,
                 max_wall_s: float = np.inf, plan_cap_s: float = np.inf):
        self.name, self.policy, self.t_switch = "takeover", policy, t_switch
        self.noise_std, self.rng, self.max_wall_s = noise_std, np.random.default_rng(seed), max_wall_s
        self.rl = RLController(env_cfg=env_cfg, actor=self._act)
        self.c = ExpertC(plan_cap_s, n_workers=1)
        self.steps: list[tuple] = []

    def reset(self, world, robot, state):
        self.t_start = time.perf_counter()
        self.world, self.robot = world, robot
        self.rl.reset(world, robot, state)
        self.cmd = self.brake = None  # (C)'s active command; the takeover's brake
        self.c_q, self.c_wsg, self.info, self.last = None, None, {}, None

    def close(self):
        self.c.close()

    def _q_at(self, t: float) -> np.ndarray:
        c = self.cmd
        q = c.trajectory.value(t) if c.trajectory is not None else (c.q if c.q is not None else self.c_q)
        return np.clip(q, self.robot.q_lower, self.robot.q_upper)

    def _g_at(self, t: float) -> float:
        g = self.cmd.trajectory.gripper_at(t) if self.cmd.trajectory is not None else None
        return float(np.clip(self.cmd.gripper if g is None else g, WSG_CLOSED, WSG_OPEN))

    def _c_state(self, state: RobotState) -> RobotState:
        if self.brake is not None and self.cmd.trajectory is self.brake and state.t >= self.brake.end_time:
            self.cmd = Command(q=self.brake.q[-1].copy())  # the brake is over: parked (so (C) can dodge)
        return dataclasses.replace(state, q_cmd=self.c_q.copy(), wsg_cmd=self.c_wsg, trajectory=self.cmd.trajectory)

    def _take_over(self, state: RobotState) -> None:
        p = copy.deepcopy(self.rl.proc)  # the commanded state now (a policy tick: the end of the last step)
        T, Q = [state.t], [p.q.copy()]
        while np.any(p.v != 0.0):
            p.start(np.concatenate([np.zeros(NQ), [gripper_action(p.wsg)]]))
            T += list(T[-1] + RLController.period * np.arange(1, p.n + 1))
            Q += list(p.Q)
        self.brake = JointTrajectory(np.array(T), np.array(Q), np.full(len(T), WSG_OPEN)) if len(T) > 1 else None
        self.cmd = Command(trajectory=self.brake) if self.brake is not None else Command(q=Q[0])
        self.c_q, self.c_wsg = state.q_cmd.copy(), state.wsg_cmd
        self.c.reset(self.world, self.robot, self._c_state(state))

    def update(self, state: RobotState, world) -> Command:
        t = state.t
        if time.perf_counter() - self.t_start > self.max_wall_s:
            raise Aborted(f"takeover episode over {self.max_wall_s} s of wall time")
        if self.cmd is None and t >= self.t_switch - 1e-9 and t >= self.rl._t_next - 1e-6:
            self._take_over(state)
        if self.cmd is not None:
            cmd = self.c.update(self._c_state(state), world)
            self.cmd, self.info = cmd, cmd.info
            self.c_q, self.c_wsg = self._q_at(t), self._g_at(t)
        if round(t / self.period) % round(RLController.period / self.period) == 0:
            self.last = self.rl.update(state, world)
        return Command(q=self.last.q, gripper=self.last.gripper, info=self.info)

    def _expert_action(self, t: float) -> np.ndarray:
        proc, h = self.rl.proc, C_TICK
        q = [self._q_at(t + proc.dt + d) for d in (-h, 0.0, h)]
        g = np.mean([self._g_at(t + RLController.period + h * j) for j in range(round(proc.dt / h))])
        return ramp_action(proc, q[1], (q[2] - q[0]) / (2 * h), g)

    def _act(self, obs: np.ndarray) -> np.ndarray:
        t = self.rl._t_step
        if self.cmd is None:
            e, a = None, np.asarray(self.policy(obs), float)
        else:
            e = self._expert_action(t)
            a = e.copy()
            if self.noise_std > 0:
                a[:NQ] = np.clip(a[:NQ] + self.rng.normal(0.0, self.noise_std, NQ), -1.0, 1.0)
        self.steps.append((t, obs, a, e))
        return a


def run_takeover(spec: ScenarioSpec, level: int, cfg: dict, policy, t_switch: float, noise_std: float = 0.0,
                 seed: int = 0, max_wall_s: float | None = None) -> dict:
    """A Takeover episode on `spec` (zero latency, as run_demo); for a success also the observations and (C)'s actions
    of the steps from the switch on (obs, actions), and of all steps the executed actions."""
    max_wall_s = cfg["demos"]["max_wall_s"] if max_wall_s is None else max_wall_s
    tk = Takeover(policy, cfg["env"], t_switch, noise_std, seed, max_wall_s, cfg["demos"]["plan_cap_s"])
    out = {"scene": spec.name, "level": level, "seed": spec.seed, "fingerprint": fingerprint(spec), "t_switch": t_switch,
           "noise_std": noise_std}
    t0 = time.perf_counter()
    try:
        r = run_episode(spec, tk, keep_log=False, latency="none")
    except Aborted:
        tk.close()
        return {**out, "outcome": "aborted", "wall_s": time.perf_counter() - t0}
    lab = [s for s in tk.steps if s[3] is not None]
    out.update(outcome=r.outcome, c_result=r.row(), wall_s=time.perf_counter() - t0, n_policy_steps=len(tk.steps) - len(lab))
    if r.outcome == "success" and lab:
        acts = np.array([s[3] for s in lab], np.float32)
        out.update(obs=np.array([s[1] for s in lab], np.float32), actions=acts,
                   executed=np.array([s[2] for s in tk.steps], np.float32),
                   action_clipped=int(np.sum(np.abs(acts[:, :NQ]) >= 1.0)))
    return out
