# (D) TD3+BC fine-tuning

Off-policy fine-tuning of the learned policy, the first item of the RL report's future work. It reuses the
fine-tuning stack (`dyngrasp/rl/agent.py`, `finetune.py`, `vecenv.py`, `scripts/finetune_rl.py`,
`scripts/orcd/ft.sbatch`) with `ft.td3=True`; everything else (environments, demos, validation, checkpoints,
remote workers, evaluation) is unchanged.

## Why TD3+BC, and why not TD3 from scratch

TD3 from scratch cannot work here: terminal rewards, episodes of 50–150 policy steps, and a 1 ms physics
simulation at 50–150 ms per environment step. What the project has is a strong policy (PPO b=1300, 193/214) and
4.4M expert transitions from (C). The question an off-policy method can answer is how many environment steps it
needs to improve on that policy, against PPO's 42.7M steps. The method for that regime is TD3+BC (Fujimoto &
Gu 2021: a deterministic actor regularised towards the data, the Q term normalised by its scale) with RLPD-style
batches (Ball et al. 2023: half demonstrations, half online, LayerNorm critics).

Lessons from the SAC fine-tuning attempt (commit history) baked into the defaults:

* 1-step critics were flat against Monte-Carlo returns (correlation 0.31; 0.93 with 3-step rows): `ft.n_step=3`.
* The BC actor's learned noise head reached std e^2 on some states (stochastic BC 5/48 vs deterministic 41/48):
  TD3 ignores the head and explores with a fixed, decaying, clipped Gaussian in action space.
* Entropy in the target made ending episodes look good: TD3 has no entropy term at all.
* The evaluation policy is deterministic (u = mean) for every (D) variant, so TD3's actor is what is evaluated.

## The update (`agent.py`, `td3=True`)

* Critic: `y = r + gamma^n (1 - done) min_{i in subset} Q'_i(s', a')`, `a' = clip(tanh(mean(s')) + clip(N(0, 0.2), ±0.5), ±1)`
  (target policy smoothing). Two LayerNorm critics (`n_critics=2`, `critic_subset=2`), Polyak `tau`.
* Actor (every `actor_delay` critic updates, with the target networks): `-mean_i Q_i(s, tanh(mean(s))) / mean|Q|`
  (`q_norm`) `+ bc_policy_weight MSE(tanh(mean(s)), tanh(mean_0(s)))` on every row `+ bc_weight MSE(tanh(mean(s)), a_demo)` on
  demonstration rows. `mean_0` is the frozen starting policy.
* Acting: `clip(tanh(mean(s)) + N(0, sigma_t), ±1)`, `sigma_t` from `td3_explore_std` to `td3_explore_final` over
  `td3_explore_steps` environment steps.
* Warm-up: the first `warmup_steps` are collected by the starting policy (plus noise) with the critics learning alone.

## Recommended configuration

```
ft.td3=True ft.actor_delay=2 ft.q_norm=True ft.n_step=3 ft.n_critics=2 ft.critic_subset=2
ft.bc_policy_weight=0.5 ft.bc_weight=0.5 ft.demo_fraction=0.5
ft.td3_explore_std=0.1 ft.td3_explore_final=0.03 ft.td3_explore_steps=2000000
ft.lr_actor=3e-5 ft.lr_critic=3e-4 ft.tau=0.005 ft.warmup_steps=100000 ft.utd=0.5
ft.bc_checkpoint=<the PPO b=1300 policy directory> ft.demos=<make_rl_demos.py transitions>
```

* Start from the PPO policy (its `actor.npz` has the same arrays; its log_std head is a constant, unused here). A run
  from the BC policy is the cleaner comparison with the SAC attempt; do both if there is time.
* `bc_policy_weight` is the knob that trades improvement against safety: too high and nothing moves, zero and the
  critics' extrapolation drives the policy off the data (the TD3+BC paper's whole point). Start at 0.5 and read
  `dev_bc` (mean squared drift from the starting policy) in `metrics.jsonl`; the validation success tells whether
  the drift pays.
* Actor learning rate low (3e-5): the starting policy is already good; most of the first million steps are the
  critics catching up. Watch `q_mean` / `target_mean` for divergence and `critic_loss` for a plateau.
* `utd` 0.5 and `updates_per_call` 16 keep the GPU learner ahead of ~50 CPU environments; raise `utd` if
  `updates_per_s` is far above `steps_per_s`.

## Running on ORCD

One line per run in a plan file (`scripts/orcd/td3.plan` is a template), then as for SAC fine-tuning:

```
CODE_DIR=<checkout> sbatch -c 36 --gres=gpu:a100:1 scripts/orcd/ft.sbatch scripts/orcd/td3.plan
sbatch -c 50 scripts/orcd/env_workers.sbatch ~/dyngrasp_rl/rl2/td3_ppo:48      # remote environments, as needed
```

Validation runs every `val_every` steps on the 50 validation scenes (deterministic policy, measured latency); the
best lands in `~/dyngrasp_rl/rl2/<run>/best/`. Final numbers like every other method: `scripts/eval_rl.py` on the
214 scenes with the rerun-failures protocol, plus `scripts/bc_bigeval.py` for the Train-500 check used to pick PPO's
checkpoint. Report: success / collision / timeout / dropped per set, compute per decision, and environment steps to
the best validation, next to PPO's 42.7M.

## Ablations worth one run each

1. Starting policy: PPO b=1300 vs BC.
2. `bc_policy_weight` 0 / 0.5 / 2 (does the anchor matter, and how much is it holding back?).
3. `n_step` 1 vs 3 (the flat-critic finding, now under TD3).
4. `demo_fraction` 0 vs 0.5 (are the (C) transitions still useful once the policy is better than BC?).
