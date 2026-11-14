"""(D) configuration: every number of the environment, the reward, the curriculum, SAC and behaviour
cloning, in one dict.

`load_config(path, overrides)` = DEFAULTS, updated by a YAML file, then by "section.key=value"
overrides whose values are parsed as YAML (`sac.batch_size=1024`, `train.n_envs=64`).
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "gamma": 0.99,  # SAC's discount, and the potentials' gamma
    "env": {
        "dt": 0.1,  # policy step (s); the controller ticks every 10 ms within it
        # Joint acceleration limit = this fraction of planning.config.IIWA_ACCEL_REF (15, 7.5, 10, 12.5, 15,
        # 20, 20 rad/s^2): 0.6 is (B)'s planning limit (9, 4.5, 6, 7.5, 9, 12, 12).
        "accel_fraction": 0.6,
        "joint_margin": 0.05,  # rad inside the joint limits (as (A)'s joint-limit damper)
        "latency": 0.001,  # training: fixed command delay (s); < 1 tick = one tick, as "measured" for a fast policy
    },
    "reward": {
        # 1 / mean start (t = 0) position error to the grasp pose over 200 generated L5 training scenes:
        # 1.0677 m (scenes.estimate_start_error) -> 0.9366; used 0.937.
        "k_p": 0.937,
        "rot_weight": 0.1,  # Phi_p = -(position error + rot_weight x orientation error in rad)
        "clearance_weight": 0.5,  # Phi_c = -clearance_weight x max(0, d_clear - d_min) / d_clear
        "d_clear": 0.04,
        "close_bonus": 0.5,  # first good close: close_bonus x exp(-|v_gripper - v_object| / close_v_scale)
        "close_v_scale": 0.05,
        "bad_close": -0.1,
        "success": 1.0,
        "collision": -1.0,
        "dropped": -1.0,
        "timeout": 0.0,
    },
    "curriculum": {
        "window": 100,  # rolling window (episodes at the current level)
        "advance_at": 0.5,  # success rate over a full window that moves to the next level
        "current_share": 0.7,  # episodes at the current level; the rest uniform over earlier levels
        "tol_pos": [0.03, 0.02, 0.01, 0.01, 0.01, 0.01],  # good close: position error (m) per level
        "tol_rot_deg": [30.0, 20.0, 10.0, 10.0, 10.0, 10.0],  # and orientation error (deg)
    },
    "validation": {
        "every_steps": 200_000,  # env steps between validations
        "n_scenes": 50,  # the first n of scenarios/rl_validation.yaml
        "n_envs": 10,  # processes for them (each runs n_scenes / n_envs)
        "patience": 3,  # final level: stop after this many validations without improvement
    },
    "sac": {
        "learning_rate": 3.0e-4,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "train_freq": 1,  # train every this many vector-env steps (each collects n_envs transitions) ...
        "updates_per_env_step": 1.0,  # ... with this many gradient steps per collected transition
        "learning_starts": 10_000,  # transitions of uniform random actions before learning
        "tau": 0.005,
        "net_arch": [512, 512, 512],  # actor and each of the 2 critics
        "layer_norm": True,  # in the critics (sbx's critic LayerNorm)
    },
    "train": {
        "n_envs": 16,
        "seed": 0,
        "max_steps": 20_000_000,  # hard cap (env steps)
        "checkpoint_minutes": 10.0,
        "log_seconds": 60.0,
        "keep_checkpoints": 2,
    },
    "demos": {  # behaviour-cloning demonstrations from (C) (demos.py, scripts/make_bc_demos.py)
        "level_weights": [0.1, 0.1, 0.15, 0.2, 0.2, 0.25],  # share of training scenes per curriculum level L0-L5
        # A (C) episode is abandoned after this much wall time (it retries a failed plan every 5 ms at zero latency).
        # Pilot (144 scenes, ORCD): successes took <= 46 s, the 5 episodes stopped at 300 s were all still unsolved.
        "max_wall_s": 120.0,
    },
}


def _merge(base: dict, upd: dict, path: str = "") -> None:
    for k, v in upd.items():
        if k not in base:
            raise KeyError(f"unknown config key {path}{k}")
        if isinstance(base[k], dict):
            _merge(base[k], v, f"{path}{k}.")
        else:
            base[k] = v


def load_config(path: str | Path | None = None, overrides: list[str] | tuple = ()) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    for o in overrides:
        key, _, value = o.partition("=")
        upd = node = {}
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
        _merge(cfg, upd)
    return cfg
