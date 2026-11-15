"""(D) SAC agent for fine-tuning the behaviour-cloned policy (finetune.py), in JAX.

Actor: sbx's SquashedGaussianActor parameters (ReLU hidden layers, mean and log_std heads; bc.py and
RLController use the same arrays), so it starts from a BC actor.npz.  Critics: an ensemble of MLPs with
LayerNorm on (obs, action).  One update (RLPD-style when the batch mixes online and demonstration rows):

  critic   y = r + disc (1 - done) (min over a random subset of the target critics Q'(s', a') [- alpha log pi(a'|s')])
           disc = gamma^n per row (n-step rows), loss = sum over critics of the mean squared TD error.  The
           entropy term in the target (backup_entropy) is off by default: with every episode end a termination,
           a low-entropy policy's -alpha log pi < 0 would be a per-step cost that makes ending episodes (by a
           collision too) look good.
  actor    alpha log pi(a|s) - mean over critics Q(s, a), a ~ pi(.|s), plus optional regularisers:
           bc_weight x MSE(tanh(mean(s)), a_demo) on the demonstration rows (bc_q_filter: only those where the
           critics value the demonstrated action above the policy's mean action),
           bc_policy_weight x MSE(tanh(mean(s)), tanh(mean_BC(s))) on every row (BC = the frozen initial actor)
  alpha    automatic (target entropy, settable: set_target_entropy) or fixed
  targets  Polyak averaging (tau)
Residual mode (residual_scale > 0): the BC actor stays frozen and the trained network (a copy of it with a zero mean
head) adds residual_scale x tanh(its mean) to the BC mean before the tanh; its log_std head gives the noise.
`export_arrays` writes either kind in the actor.npz format RLController reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from dyngrasp.rl.observation import OBS_DIM

ACTION_DIM = 8
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


# ---- networks --------------------------------------------------------------------------------------------------


def policy(actor_params, bc_params, x, res_scale: float = 0.0):
    """(pre-tanh mean, log_std): the actor's, or with res_scale > 0 the BC mean + res_scale x tanh(the actor's mean)."""
    mean, log_std = actor_forward(actor_params, x)
    if res_scale > 0:
        mean = jax.lax.stop_gradient(actor_forward(bc_params, x)[0]) + res_scale * jnp.tanh(mean)
    return mean, log_std


def actor_forward(params, x):
    """(mean, log_std) of sbx's SquashedGaussianActor."""
    p = params["params"]
    n = len(p) - 2
    for i in range(n):
        x = jax.nn.relu(x @ p[f"Dense_{i}"]["kernel"] + p[f"Dense_{i}"]["bias"])
    mean = x @ p[f"Dense_{n}"]["kernel"] + p[f"Dense_{n}"]["bias"]
    log_std = x @ p[f"Dense_{n + 1}"]["kernel"] + p[f"Dense_{n + 1}"]["bias"]
    return mean, jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)


def sample_action(mean, log_std, key):
    """a = tanh(u), u ~ N(mean, std); log pi(a) with the tanh correction."""
    u = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
    logp = jnp.sum(-0.5 * ((u - mean) * jnp.exp(-log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), -1)
    logp -= jnp.sum(2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u)), -1)
    return jnp.tanh(u), logp


class Critic(nn.Module):
    arch: Sequence[int]
    layer_norm: bool

    @nn.compact
    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], -1)
        for h in self.arch:
            x = nn.Dense(h)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
        return nn.Dense(1)(x)[..., 0]


class Critics(nn.Module):
    """n independent critics: (n, batch) values."""

    n: int
    arch: Sequence[int]
    layer_norm: bool

    @nn.compact
    def __call__(self, obs, act):
        ens = nn.vmap(Critic, variable_axes={"params": 0}, split_rngs={"params": True}, in_axes=None, out_axes=0,
                      axis_size=self.n)
        return ens(self.arch, self.layer_norm)(obs, act)


# ---- the agent ---------------------------------------------------------------------------------------------------


class Agent:
    """Parameters and optimiser states in `self.state` (a dict of arrays: what checkpoints save)."""

    def __init__(self, cfg: dict, actor_params, seed: int = 0):
        f = cfg["ft"]
        self.f, self.gamma = f, cfg["gamma"]
        self.critic = Critics(f["n_critics"], tuple(f["critic_arch"]), f["layer_norm"])
        self.opt_actor = optax.adam(float(f["lr_actor"]))
        self.opt_critic = optax.adam(float(f["lr_critic"]))
        self.opt_alpha = optax.adam(float(f["lr_alpha"]))
        self.auto_alpha = f["ent_coef"] == "auto"
        key = jax.random.PRNGKey(seed)
        key, k = jax.random.split(key)
        critic_params = self.critic.init(k, jnp.zeros((1, OBS_DIM)), jnp.zeros((1, ACTION_DIM)))
        actor_params = jax.tree_util.tree_map(jnp.asarray, actor_params)
        self.res_scale = float(f["residual_scale"])
        bc_params = actor_params
        if self.res_scale > 0:  # the residual network: BC's weights, mean head zeroed (no change at the start)
            p = dict(actor_params["params"])
            n = len(p) - 2
            p[f"Dense_{n}"] = jax.tree_util.tree_map(jnp.zeros_like, p[f"Dense_{n}"])
            actor_params = {"params": p}
        log_alpha = jnp.log(float(f["init_ent_coef"] if self.auto_alpha else f["ent_coef"]))
        copy = partial(jax.tree_util.tree_map, lambda x: jnp.array(x, copy=True))  # distinct buffers (donated)
        self.state = {
            "actor": actor_params, "bc": copy(bc_params),
            "critic": critic_params, "target": copy(critic_params),
            "log_alpha": log_alpha,
            "opt_actor": self.opt_actor.init(actor_params),
            "opt_critic": self.opt_critic.init(critic_params),
            "opt_alpha": self.opt_alpha.init(log_alpha),
            "key": key,
            "n_updates": jnp.zeros((), jnp.int32),
            "target_entropy": jnp.asarray(float(-ACTION_DIM if f["target_entropy"] is None else f["target_entropy"])),
        }
        self._act = jax.jit(self._act_impl, static_argnames="deterministic")
        self.pi = partial(policy, res_scale=self.res_scale)
        self._chunk = jax.jit(self._chunk_impl, static_argnames="train_actor", donate_argnums=0)

    # ---- acting -------------------------------------------------------------------------------------------------
    def _act_impl(self, actor_params, bc_params, obs, key, deterministic: bool):
        mean, log_std = self.pi(actor_params, bc_params, obs)
        return jnp.tanh(mean) if deterministic else sample_action(mean, log_std, key)[0]

    def act(self, obs: np.ndarray, key, deterministic: bool = False) -> np.ndarray:
        return np.asarray(self._act(self.state["actor"], self.state["bc"], jnp.asarray(obs), key, deterministic))

    def entropy(self, obs: np.ndarray, key) -> float:
        """-mean log pi(a|s), a ~ pi(.|s), of the current actor on these observations."""
        mean, log_std = self.pi(self.state["actor"], self.state["bc"], jnp.asarray(obs))
        return -float(sample_action(mean, log_std, key)[1].mean())

    def export_arrays(self) -> dict:
        """actor.npz arrays (bc.to_arrays format; residual mode: the BC actor's, plus res_* and res_scale)."""
        from dyngrasp.rl.bc import to_arrays

        actor = to_arrays(jax.device_get(self.state["actor"]))
        if self.res_scale <= 0:
            return actor
        return {**to_arrays(jax.device_get(self.state["bc"])), **{f"res_{k}": v for k, v in actor.items()},
                "res_scale": np.asarray(self.res_scale)}

    def set_target_entropy(self, value: float) -> None:
        self.state = {**self.state, "target_entropy": jnp.asarray(float(value))}

    # ---- learning -----------------------------------------------------------------------------------------------
    def update(self, batches: dict, train_actor: bool = True) -> dict:
        """len(batches["obs"]) updates, one per minibatch (arrays (K, B, ...)); returns the last one's metrics."""
        self.state, metrics = self._chunk(self.state, batches, train_actor=train_actor)
        return {k: float(v) for k, v in metrics.items()}

    def _chunk_impl(self, state, batches, train_actor: bool):
        def body(s, b):
            return self._update(s, b, train_actor)

        state, metrics = jax.lax.scan(body, state, batches)
        return state, jax.tree_util.tree_map(lambda x: x[-1], metrics)

    def _update(self, s, b, train_actor: bool):
        f = self.f
        key, k_next, k_sub, k_pi = jax.random.split(s["key"], 4)
        alpha = jnp.exp(s["log_alpha"])
        obs, act = b["obs"], b["act"]

        # critic
        mean_n, log_std_n = self.pi(s["actor"], s["bc"], b["next_obs"])
        a_n, logp_n = sample_action(mean_n, log_std_n, k_next)
        q_next = self.critic.apply(s["target"], b["next_obs"], a_n)
        sub = jax.random.choice(k_sub, f["n_critics"], (f["critic_subset"],), replace=False)
        v_next = q_next[sub].min(0) - (alpha * logp_n if f["backup_entropy"] else 0.0)
        y = jax.lax.stop_gradient(b["rew"] + b["disc"] * (1.0 - b["done"]) * v_next)

        def critic_loss(cp):
            q = self.critic.apply(cp, obs, act)
            return jnp.sum(jnp.mean((q - y) ** 2, axis=1)), q

        (lc, q), g = jax.value_and_grad(critic_loss, has_aux=True)(s["critic"])
        upd, opt_c = self.opt_critic.update(g, s["opt_critic"], s["critic"])
        critic = optax.apply_updates(s["critic"], upd)
        target = optax.incremental_update(critic, s["target"], float(f["tau"]))
        s = {**s, "critic": critic, "target": target, "opt_critic": opt_c, "key": key, "n_updates": s["n_updates"] + 1}
        metrics = {"critic_loss": lc, "q_mean": q.mean(), "q_max": q.max(), "target_mean": y.mean()}

        # actor and temperature
        mean_bc = jax.lax.stop_gradient(jnp.tanh(actor_forward(s["bc"], obs)[0]))
        demo = b["is_demo"]
        if f["bc_q_filter"]:  # BC only where the critics prefer the demonstrated action (Nair et al. 2018)
            mean_now = jnp.tanh(self.pi(s["actor"], s["bc"], obs)[0])
            better = self.critic.apply(critic, obs, act).mean(0) > self.critic.apply(critic, obs, mean_now).mean(0)
            demo = demo * jax.lax.stop_gradient(better)

        def actor_loss(ap):
            mean, log_std = self.pi(ap, s["bc"], obs)
            a, logp = sample_action(mean, log_std, k_pi)
            qa = self.critic.apply(critic, obs, a).mean(0)
            dev = jnp.mean((jnp.tanh(mean) - mean_bc) ** 2, -1)
            bc_demo = jnp.sum(demo * jnp.mean((jnp.tanh(mean) - act) ** 2, -1)) / jnp.maximum(demo.sum(), 1.0)
            loss = jnp.mean(alpha * logp - qa) + float(f["bc_weight"]) * bc_demo + float(f["bc_policy_weight"]) * dev.mean()
            return loss, {"logp": logp.mean(), "dev_bc": dev.mean(), "mse_demo": bc_demo, "log_std": log_std.mean(),
                          "q_pi": qa.mean()}

        (la, aux), g = jax.value_and_grad(actor_loss, has_aux=True)(s["actor"])
        metrics.update(actor_loss=la, alpha=alpha, **aux)
        if not train_actor:
            return s, metrics
        do = (s["n_updates"] % f["actor_delay"]) == 0
        upd, opt_a = self.opt_actor.update(g, s["opt_actor"], s["actor"])
        upd = jax.tree_util.tree_map(lambda u: jnp.where(do, u, 0.0), upd)
        opt_a = jax.tree_util.tree_map(lambda new, old: jnp.where(do, new, old), opt_a, s["opt_actor"])
        s = {**s, "actor": optax.apply_updates(s["actor"], upd), "opt_actor": opt_a}
        if self.auto_alpha:
            ent_gap = jax.lax.stop_gradient(aux["logp"] + s["target_entropy"])
            ga = jax.grad(lambda la_: -la_ * ent_gap)(s["log_alpha"])
            upd, opt_al = self.opt_alpha.update(ga, s["opt_alpha"], s["log_alpha"])
            upd = jnp.where(do, upd, 0.0)
            s = {**s, "log_alpha": optax.apply_updates(s["log_alpha"], upd),
                 "opt_alpha": jax.tree_util.tree_map(lambda new, old: jnp.where(do, new, old), opt_al, s["opt_alpha"])}
        return s, metrics

    # ---- persistence -----------------------------------------------------------------------------------------------
    def state_numpy(self) -> dict:
        return jax.tree_util.tree_map(np.asarray, jax.device_get(self.state))

    def load_state_numpy(self, st: dict) -> None:
        self.state = jax.tree_util.tree_map(jnp.asarray, st)
