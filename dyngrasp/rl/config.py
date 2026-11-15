"""(D) configuration: every number of the environment, the reward, the curriculum, SAC, behaviour cloning, SAC
fine-tuning and PPO, in one dict.

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
        "post_success": False,  # episodes go on after success to the scene's end, graded as the harness (env.py)
    },
    "reward": {
        # 1 / mean start (t = 0) position error to the grasp pose over 200 generated L5 training scenes:
        # 1.0677 m (scenes.estimate_start_error) -> 0.9366; used 0.937.
        "k_p": 0.937,
        "rot_weight": 0.1,  # Phi_p = -(position error + rot_weight x orientation error in rad)
        "clearance_weight": 0.5,  # Phi_c = -clearance_weight x max(0, d_clear - d_min) / d_clear
        "d_clear": 0.04,
        # Phi_e (cart and floor, rewards.py): within 2 cm for the iiwa links, 5 mm ((C)'s d_safe_env) for the WSG, which
        # has to come close to the cart top to grasp; half the obstacle term's weight
        "env_clearance_weight": 0.25,
        "d_clear_env_arm": 0.02,
        "d_clear_env_hand": 0.005,
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
        "plan_cap_s": 10.0,  # takeovers (DAgger): wall time after which a (C) plan tries no further seeds (ExpertC)
    },
    "bc": {  # behaviour cloning (bc.py): the SAC actor (sac.net_arch), MSE on tanh(mean), NLL for log_std
        "learning_rate": 3.0e-4,  # Adam
        "batch_size": 512,
        "max_epochs": 300,
        "patience": 15,  # stop after this many epochs without a validation-loss improvement
        "keep": "best_val_mse",  # the policy kept (best/): lowest validation MSE, or "last" (max_epochs, no early stop)
        "val_fraction": 0.1,  # episodes held out for the validation loss
        "nll_weight": 1.0,  # log_std head: Gaussian NLL of atanh(action), its own head (trunk gradient stopped)
        "action_eps": 1.0e-3,  # actions clipped to +-(1 - eps) for atanh
        "seed": 0,
        "ema": 0.0,  # > 0: the policy is the exponential moving average of the weights (this decay per minibatch)
        "obs_noise": 0.0,  # std of Gaussian noise added to the training observations (their fixed scales)
        "closed_loop_every": 10,  # epochs between closed-loop runs on the validation scenes (and at the end)
        "closed_loop_workers": 15,  # processes for them
    },
    "ft": {  # SAC fine-tuning of the BC policy (finetune.py, agent.py, scripts/finetune_rl.py)
        "bc_checkpoint": None,  # directory with the BC actor.npz: the initial actor and the bc_policy_weight reference
        "demos": None,  # transitions of (C) demonstrations (scripts/make_rl_demos.py; a file or a list), or None
        "demo_fraction": 0.5,  # of every batch (RLPD); the rest from the online replay buffer
        "n_envs": 56,  # asynchronous environment processes on this machine ...
        "remote_envs": 0,  # ... and slots for environments on others (scripts/env_workers.py; vecenv.py)
        "seed": 0,
        "start_level": 5,  # curriculum level at the start (5: the final mix)
        "max_steps": 20_000_000,  # environment steps
        "buffer_size": 1_000_000,
        "learning_starts": 5_000,  # online rows before the first update
        "warmup_steps": 50_000,  # environment steps before the actor and alpha update (critics only until then)
        "batch_size": 256,  # rows per update (online + demonstrations)
        "utd": 1.0,  # updates per environment step
        "updates_per_call": 16,  # updates per jitted call
        "max_chunks_per_poll": 4,  # update calls between two rounds of actions at most
        "lr_actor": 1.0e-4,
        "lr_critic": 3.0e-4,
        "lr_alpha": 3.0e-4,
        "tau": 0.005,
        "n_critics": 2,  # RLPD: 10
        "critic_subset": 2,  # target = min over this many randomly chosen critics
        "critic_arch": [512, 512, 512],
        "layer_norm": True,
        "n_step": 1,
        "ent_coef": "auto",  # or a fixed value
        "init_ent_coef": 0.01,
        "target_entropy": None,  # None: -8 (minus the action dimension) ...
        "target_entropy_bc_offset": None,  # ... unless set: the initial actor's entropy on demonstration states + this
        "backup_entropy": False,  # entropy term in the critic target (agent.py: off, episodes end by termination)
        "actor_delay": 1,  # actor (and alpha) update every this many critic updates
        "bc_weight": 0.0,  # x MSE(policy mean action, demonstration action) on the demonstration rows ...
        "bc_q_filter": False,  # ... only where the critics value the demonstrated action above the policy's
        "bc_policy_weight": 0.0,  # x MSE(policy mean action, initial BC policy's) on every row
        "cql_alpha": 0.0,  # > 0: CQL(H) term in the critic loss (agent.py) ...
        "cql_n": 4,  # ... with this many uniform, policy (s) and policy (s') actions per row
        "cql_is": False,  # ... importance-weighted by the sampling densities (CQL(H); degenerate with a narrow policy)
        "calql": False,  # ... and Cal-QL's lower bound (the Monte-Carlo return) on the policy actions' values
        "awr": False,  # with awac_beta: state-value critics and n-step TD advantages (AWR; agent.py)
        "q_norm": False,  # SAC actor term / mean |Q| (TD3+BC; agent.py)
        "awac_beta": 0.0,  # > 0: AWAC actor (advantage-weighted regression, temperature) instead of SAC's
        "actor_obs_noise": 0.0,  # std of Gaussian noise on the actor's observations (BC's bc.obs_noise; agent.py)
        "actor_ema": 0.0,  # > 0: act, validate and bootstrap with this EMA of the actor's parameters (BC's bc.ema)
        "init_log_std": None,  # set: the actor's log_std head starts as this constant (the BC head can be wild)
        "residual_scale": 0.0,  # > 0: frozen BC actor + this x tanh(a trained residual) before the tanh (agent.py)
        "val_every": 250_000,  # environment steps between validations (and one at step 0)
        "val_workers": 8,
        "checkpoint_minutes": 15.0,
        "log_seconds": 60.0,
    },
    "ppo": {  # PPO with an imitation term from the BC policy (ppo.py, rollout.py, scripts/ppo_rl.py)
        "bc_checkpoint": None,  # directory with the BC actor.npz: the initial actor
        "demos": None,  # (C)'s demonstrations for the imitation term (a make_bc_demos.py merge .npz: obs, act)
        "dagger_init": None,  # DAgger labels made before the run (scripts/ppo_offline.py merge *_labels.npz)
        "value_init": None,  # rollouts made before the run (ppo_offline.py *_rollouts.npz): V regressed on their returns
        "value_init_epochs": 10,  # ... for this many epochs before the first batch
        "seed": 0,
        "max_steps": 300_000_000,  # environment steps
        "batch_steps": 32768,  # environment steps (complete episodes) per PPO batch
        "window": 1,  # PPO trains on the latest this many batches (behaviour log-probabilities in the ratio)
        "value_window": 2,  # V trains on the latest this many batches' GAE returns
        "epochs": 4,  # PPO epochs per batch (stopped early past 1.5 target_kl once w > 0)
        "value_epochs": 4,
        "minibatch": 4096,  # PPO and value minibatch
        "imit_batch": 4096,  # imitation rows per actor step ...
        "dagger_fraction": 0.5,  # ... this share from the DAgger labels (once there are min_dagger), the rest demonstrations
        "min_dagger": 5000,
        "lr_actor": 5.0e-5,
        "lr_log_std": 1.0e-3,
        "lr_value": 3.0e-4,
        "max_grad_norm": 1.0,
        "clip": 0.2,
        "is_cap": 2.0,  # decoupled PPO: the prox / behaviour importance weight truncated here (ppo.py)
        "target_kl": 0.02,
        "gae_lambda": 0.95,
        "value_arch": [512, 512, 512],  # LayerNorm, ReLU
        "init_log_std": [-3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -4.0],  # joints sigma 0.030 (BC: ~0.014), gripper 0.018
        # log_std's lower (per dimension) and upper bounds; the gripper's sigma (action units, ppo.dist_mean) at least
        # 0.018 (~1 mm of WSG width): its imitation residual is ~0 wherever both are saturated, which drives it to the
        # bound, and a tiny sigma there makes PPO's ratios on the few closing steps explode
        "log_std_bounds": [[-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -4.0], -1.5],
        "ent_coef": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01],  # c_i: the entropy bonus per dimension ...
        "ent_coef_final": [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001],  # ... going to this over the ramp
        "imit_weight": 0.04,  # lam0 (lam = lam0 until the ramp, then down to 0)
        "ppo_weight": 1.0,  # w at the end of the ramp
        "nll_clip": 3.0,  # the imitation noise likelihood's residuals clipped at this many sigma
        "obs_noise": 0.02,  # BC's observation noise, on the imitation rows
        "action_eps": 1.0e-3,  # expert actions clipped to +-(1 - eps) for atanh
        "ema": 0.999,  # the exported (validated) policy: this EMA of the trained one per actor step
        "gate_ev": 0.5,  # PPO ramps in once V's explained variance of the returns of new batches is >= this ...
        "gate_batches": 3,  # ... on this many batches in a row ...
        "min_pretrain": 100,  # ... and at least this many batches have passed (BC + DAgger, V learning)
        "stop_at_gate": False,  # checkpoint and stop (DONE) once the gate opens: runs forked from there (ppo.py)
        "ramp_batches": 300,  # batches over which w 0 -> ppo_weight, lam imit_weight -> 0, label_fraction -> 0
        "label_fraction": 0.25,  # share of episodes followed by a DAgger takeover before the ramp
        "label_max_wall_s": 60.0,  # a takeover is abandoned after this much wall time
        "val_every": 25,  # batches between validations (val50, the EMA policy)
        "val_workers": 16,
        "checkpoint_minutes": 15.0,
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
