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
"""

from __future__ import annotations

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
        v_c = (Q[end + 1] - Q[end - 1]) / (2 * tick)
        e = Q[end] - (proc.q + 0.5 * (proc.v + v_c) * proc.dt)
        a = np.empty(NQ + 1)
        a[:NQ] = np.clip((v_c + e / proc.dt) / proc.v_max, -1.0, 1.0)
        a[NQ] = np.clip(gripper_action(G[k * m + r: end + r].mean()), -1.0, 1.0)
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
