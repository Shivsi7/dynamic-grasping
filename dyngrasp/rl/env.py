"""(D) Gymnasium environment: the evaluation's Drake simulation, one policy step (0.1 s) per `step`.

Each reset builds the scene with harness.build_episode (same HardwareStation, 1 ms physics, same
EpisodeMonitor at 100 Hz deciding collisions, drops and success) around an RLController that
executes the actions (actions.py).  `step` runs the simulation just past the next policy tick: the
observation is the one that tick computed (as in evaluation, see controller.py) and the action
completes that tick's command.  Differences from an evaluation episode:
* the episode ends at the first step that ends in success (object lifted and held for HOLD_TIME),
  collision or drop; otherwise at the first step end at or after the scene's duration (timeout).
  All four are terminations (no truncation: the observation has the time left before the object
  leaves reach, and the scene ends a fixed 1.5 s after that).  With env.post_success the episode goes
  on after success to the scene's end, graded as the harness grades it: a collision after success,
  or the object not in hand and lifted at the end, makes it a failure (outcome collision or dropped,
  reward collision/dropped - success, so the episode's terminal rewards add up to the failure's);
  the success reward is paid at success, and nothing more at a successful end;
* a fixed 1 ms command latency (one tick, as "measured" latency gives a policy faster than 10 ms).

Rewards: rewards.py.  Scenes: the curriculum (scenes.py; `set_level`) or a fixed list cycled in order
(`scenes=`, for validation).  The next curriculum scene is generated in a background thread while the
current episode runs (the generator can take seconds on rare seeds).
"""

from __future__ import annotations

import functools
import multiprocessing
import signal
from concurrent.futures import Future, ThreadPoolExecutor

import gymnasium as gym
import numpy as np

from dyngrasp.harness import build_episode, shared_robot_model
from dyngrasp.rl.actions import gripper_width, is_close_event
from dyngrasp.rl.config import load_config
from dyngrasp.rl.controller import RLController
from dyngrasp.rl.curriculum import sample_level
from dyngrasp.rl.observation import OBS_DIM
from dyngrasp.rl.rewards import clearance_potential, close_reward, grasp_errors, progress_potential
from dyngrasp.rl.scenes import sample_training_scene
from dyngrasp.scenario import ScenarioSpec

TERMS = ("progress", "clearance", "close", "terminal")
PAST_TICK = 0.5e-3  # AdvanceTo(tick time + this): Drake runs that time's updates (physics, controller)


class GraspEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (8,), np.float32)

    def __init__(self, cfg: dict | None = None, seed: int = 0, scenes: list[ScenarioSpec] | None = None):
        self.cfg = cfg or load_config()
        self.rng = np.random.default_rng(seed)
        self.scenes, self._k = scenes, 0
        self.level = 0
        self.robot = shared_robot_model()
        self._pool = ThreadPoolExecutor(1) if scenes is None else None
        self._next: Future | None = None
        self.h = None

    # ---- curriculum / checkpoint hooks (called through VecEnv.env_method) ----------------------------
    def set_level(self, level: int) -> None:
        self.level = int(level)

    def get_rng_state(self) -> dict:
        return self.rng.bit_generator.state

    def set_rng_state(self, state: dict) -> None:
        self.rng.bit_generator.state = state
        self._next = None

    def restart_list(self) -> None:
        """Fixed scene list: the next reset starts again from its first scene."""
        self._k = 0

    # ---- episodes -------------------------------------------------------------------------------------
    def _prefetch(self) -> Future:
        level = sample_level(self.level, self.rng, self.cfg["curriculum"]["current_share"])
        rng = np.random.default_rng(int(self.rng.integers(2**63)))
        return self._pool.submit(lambda: (sample_training_scene(level, rng), level))

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng, self._next = np.random.default_rng(seed), None
        if options and "spec" in options:
            spec, level = options["spec"], options.get("level", -1)
        elif self.scenes is not None:
            spec, level = self.scenes[self._k % len(self.scenes)], -1
            self._k += 1
        else:
            if self._next is None:
                self._next = self._prefetch()
            spec, level = self._next.result()
            self._next = self._prefetch()
        return self._start(spec, level), {}

    def _start(self, spec: ScenarioSpec, level: int) -> np.ndarray:
        c = self.cfg["env"]
        assert c["latency"] > 0, "set_action completes a command after its tick: it must take effect later"
        self.spec, self.scene_level = spec, level
        self.ctrl = RLController(env_cfg=c)
        # the monitor samples at the controller's tick times (k x 10 ms, as in evaluation)
        self.h = build_episode(spec, self.ctrl, None, stop_on_collision=True, monitor_hz=1.0 / RLController.period,
                               latency=c["latency"])
        self.sim, self.world = self.h.simulator, self.h.world
        self.ticks_per_step = round(c["dt"] / RLController.period)
        self.sim.Initialize()
        self.n_steps = 0
        state = self._to_policy_tick()
        lvl = min(self.level, len(self.cfg["curriculum"]["tol_pos"]) - 1)
        self.tol_pos = self.cfg["curriculum"]["tol_pos"][lvl]
        self.tol_rot = np.deg2rad(self.cfg["curriculum"]["tol_rot_deg"][lvl])
        self.close_paid, self.n_close = False, {"good": 0, "bad": 0}
        self.succeeded = False  # post_success: success paid, the episode going on
        self.ep_terms = dict.fromkeys(TERMS, 0.0)
        self.phi_p, self.phi_c = self._potentials(state)
        return self.ctrl.obs

    def _to_policy_tick(self):
        """Simulate through policy tick n_steps (t = n_steps x dt); its RobotState (also self.ctrl.state)."""
        t = self.n_steps * self.ticks_per_step * RLController.period  # Drake's time of that tick
        self.sim.AdvanceTo(t + PAST_TICK)
        state = self.ctrl.state
        assert state is not None and abs(state.t - t) < 1e-9, (t, state and state.t)
        return state

    def _potentials(self, state) -> tuple[float, float]:
        """Progress (state) and clearance (the monitor's sample at state.t) potentials."""
        r, log = self.cfg["reward"], self.h.monitor.log
        assert abs(log.t[-1] - state.t) < 1e-9, (log.t[-1], state.t)
        d_min, in_hand = float(log.clearance_obstacles[-1]), bool(log.in_hand[-1])
        return (progress_potential(state.X_WG, self.world.grasp_pose(state.t), in_hand, r["rot_weight"]),
                clearance_potential(d_min, r["d_clear"], r["clearance_weight"]))

    def _speeds_apart(self) -> float:
        """|v_gripper - v_object| in the simulation."""
        sc = self.h.scene
        pc = sc.plant.GetMyContextFromRoot(self.sim.get_context())
        v = lambda body: sc.plant.EvalBodySpatialVelocityInWorld(pc, body).translational()
        return float(np.linalg.norm(v(sc.gripper_body) - v(sc.target_body)))

    def step(self, action):
        rc, gamma = self.cfg["reward"], self.cfg["gamma"]
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        width_before = self.ctrl.proc.wsg
        self.ctrl.set_action(a)
        self.n_steps += 1
        state = self._to_policy_tick()
        phi_p, phi_c = self._potentials(state)
        r = {
            "progress": rc["k_p"] * (gamma * phi_p - self.phi_p),
            "clearance": gamma * phi_c - self.phi_c,
            "close": 0.0,
            "terminal": 0.0,
        }
        self.phi_p, self.phi_c = phi_p, phi_c
        if is_close_event(width_before, gripper_width(a[7]), self.world.object.grasp_width):
            e_p, e_r = grasp_errors(state.X_WG, self.world.grasp_pose(state.t))
            good = e_p <= self.tol_pos and e_r <= self.tol_rot
            r["close"] = close_reward(good, not self.close_paid, self._speeds_apart(), rc)
            self.close_paid |= good
            self.n_close["good" if good else "bad"] += 1
        m = self.h.monitor
        outcome = None
        end = state.t >= self.spec.duration - 1e-9
        if self.cfg["env"]["post_success"] and self.succeeded:
            if m.collisions:
                outcome = "collision"
            elif end and not m.held_at_end:
                outcome = "dropped"
            elif end:
                outcome = "success"
            if outcome is not None and outcome != "success":
                r["terminal"] = rc[outcome] - rc["success"]
        else:
            if m.collisions:
                outcome = "collision"
            elif m.t_success is not None:
                outcome = "success"
            elif m.dropped:
                outcome = "dropped"
            elif end:
                outcome = "timeout"
            if outcome == "success" and self.cfg["env"]["post_success"] and not end:
                self.succeeded, outcome = True, None  # paid now, graded at the end
                r["terminal"] = rc["success"]
            elif outcome is not None:
                r["terminal"] = rc[outcome]
        for k, v in r.items():
            self.ep_terms[k] += v
        info = {"t": state.t, "terms": r}
        if outcome is not None:
            c = m.collisions[0] if m.collisions else None
            info.update(outcome=outcome, is_success=outcome == "success", level=self.scene_level, scene=self.spec.name,
                        ep_terms=dict(self.ep_terms), ep_len=self.n_steps, closes=dict(self.n_close),
                        collision=f"{c.category}:{c.robot_body}" if c else "")
        return self.ctrl.obs, float(sum(r.values())), outcome is not None, False, info

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)


def make_env(cfg: dict, seed: int, scenes: list[ScenarioSpec] | None = None):
    """A picklable factory for a (Subproc)VecEnv: GraspEnv in SB3's Monitor (episode return and length)."""
    return functools.partial(_monitored_env, cfg, seed, scenes)


def _monitored_env(cfg, seed, scenes):
    from stable_baselines3.common.monitor import Monitor

    if multiprocessing.parent_process() is not None:  # a SubprocVecEnv worker: the learner checkpoints on
        for s in (signal.SIGUSR1, signal.SIGTERM):  # these (Slurm sends SIGTERM to every process) and
            signal.signal(s, signal.SIG_IGN)  # needs the workers alive for that
    return Monitor(GraspEnv(cfg, seed, scenes))
