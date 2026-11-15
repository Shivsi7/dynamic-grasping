"""(D) the SAC policy as a Controller (same interface as (A)/(B)/(C)).

Every 10 ms tick it sends the joint positions of the current policy step's velocity ramp (see
actions.py); every `dt` (0.1 s) a policy tick observes (`obs`) and starts a new step from an action.
In evaluation the action is the deterministic policy (tanh of the Gaussian mean) on `obs`, computed
in numpy from the actor weights a checkpoint saves (actor.npz).  In training (`actor=None`) the
policy tick returns a command that the environment completes with `set_action` before the
simulation goes on; the harness applies a command one tick after it is computed (latency > 0), so
training and evaluation run the same ticks on the same observations.

    ctrl = RLController("runs/sac0/checkpoints/latest")   # a checkpoint directory
    run_episode(spec, ctrl)                                # harness default: measured latency
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dyngrasp.controller import Command, Controller, RobotModel, RobotState
from dyngrasp.planning.config import IIWA_ACCEL_REF
from dyngrasp.rl.actions import ActionProcessor
from dyngrasp.rl.config import DEFAULTS
from dyngrasp.rl.observation import Observer
from dyngrasp.world import WorldModel


class NumpyActor:
    """Deterministic action of an sbx SquashedGaussianActor: tanh(mean), ReLU hidden layers.  With res_* arrays and
    res_scale (agent.py's residual mode): tanh(mean + res_scale x tanh(the residual network's mean))."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        n = sum(k.startswith("W") and k[1:].isdigit() for k in arrays)
        self.hidden = [(np.asarray(arrays[f"W{i}"], np.float64), np.asarray(arrays[f"b{i}"], np.float64)) for i in range(n)]
        self.mean = (np.asarray(arrays["Wm"], np.float64), np.asarray(arrays["bm"], np.float64))
        self.residual = None
        if "res_scale" in arrays:
            res = NumpyActor({k[4:]: v for k, v in arrays.items() if k.startswith("res_") and k != "res_scale"})
            self.residual = (res, float(arrays["res_scale"]))

    @staticmethod
    def arrays_from_sbx(actor_params) -> dict[str, np.ndarray]:
        """sbx actor params ({"params": {"Dense_0": {kernel, bias}, ...}}): hidden layers, mean, log_std (unused)."""
        p = actor_params["params"]
        n = len(p) - 2
        out = {}
        for i in range(n):
            out[f"W{i}"], out[f"b{i}"] = np.asarray(p[f"Dense_{i}"]["kernel"]), np.asarray(p[f"Dense_{i}"]["bias"])
        out["Wm"], out["bm"] = np.asarray(p[f"Dense_{n}"]["kernel"]), np.asarray(p[f"Dense_{n}"]["bias"])
        return out

    def pre_tanh(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, np.float64)
        for W, b in self.hidden:
            x = np.maximum(x @ W + b, 0.0)
        W, b = self.mean
        return x @ W + b

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        m = self.pre_tanh(obs)
        if self.residual is not None:
            res, scale = self.residual
            m = m + scale * np.tanh(res.pre_tanh(obs))
        return np.tanh(m)


class RLController(Controller):
    name = "rl_sac"
    period = 0.01

    def __init__(self, checkpoint: str | Path | None = None, env_cfg: dict | None = None,
                 actor: NumpyActor | None = None):
        if checkpoint is not None:
            ckpt = Path(checkpoint)
            env_cfg = yaml.safe_load((ckpt / "config.yaml").read_text())["env"]
            actor = NumpyActor(dict(np.load(ckpt / "actor.npz")))
        self.env_cfg = env_cfg or DEFAULTS["env"]
        self.actor = actor
        self.observer: Observer | None = None
        self.proc: ActionProcessor | None = None
        self.state: RobotState | None = None  # the last policy tick's state and observation
        self.obs: np.ndarray | None = None
        self._cmd: Command | None = None  # training: the last policy tick's command, until set_action

    def set_action(self, action: np.ndarray) -> None:
        """Training: the action for `obs`; completes the policy tick's command before it takes effect."""
        assert self._cmd is not None, "no policy tick is waiting for an action"
        self.proc.start(action)
        self._cmd.q, self._cmd.gripper = self.proc.q_at(1).copy(), self.proc.wsg
        self._cmd = None

    def reset(self, world: WorldModel, robot: RobotModel, state: RobotState) -> None:
        c = self.env_cfg
        if self.observer is None or self.observer.world is not world:
            self.observer = Observer(world, robot)
            self.observer(state, np.zeros(7))  # warm up the distance query outside the timed cycles
        a_max = c["accel_fraction"] * np.asarray(IIWA_ACCEL_REF)
        self.proc = ActionProcessor(robot.q_lower, robot.q_upper, robot.v_limits, a_max, c["dt"], self.period,
                                    c["joint_margin"])
        self.proc.reset(state.q_cmd, state.wsg_cmd)
        self._t_step = self._t_next = state.t
        self.state = self.obs = self._cmd = None

    def update(self, state: RobotState, world: WorldModel) -> Command:
        t = state.t
        if t >= self._t_next - 1e-6:
            assert self._cmd is None, "the last policy tick got no action"
            self._t_step, self._t_next = t, t + self.env_cfg["dt"]
            self.state, self.obs = state, self.observer(state, self.proc.v)
            if self.actor is None:
                self._cmd = Command(q=self.proc.q.copy(), gripper=self.proc.wsg)  # completed by set_action
                return self._cmd
            self.proc.start(self.actor(self.obs))
        k = round((t - self._t_step) / self.period) + 1
        return Command(q=self.proc.q_at(k).copy(), gripper=self.proc.wsg)
