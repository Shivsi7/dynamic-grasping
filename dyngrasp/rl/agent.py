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
           bc_policy_weight x MSE(tanh(mean(s)), tanh(mean_BC(s))) on every row (BC = the frozen initial actor);
           q_norm: the SAC term divided by mean |Q| (TD3+BC's normalisation, so the regularisers' weights are relative);
           awac_beta > 0: instead of the SAC term, AWAC's weighted regression on the batch actions,
           MSE(tanh(mean(s)), a) x exp(A/awac_beta) (clipped at 100, normalised to mean 1), A = Q(s, a) - Q(s, tanh(mean(s)))
           (the log_std head then gets no gradient: the exploration noise stays as it starts);
           awr: the critics are state values V(s) (zero action input, TD on the batch's n-step rows) and
           A = r^(n) + gamma^n V'(s_n) - V(s) (AWR, Peng et al. 2019): no extrapolation in the action
           cql_alpha > 0: + cql_alpha x (log-sum-exp of Q over cql_n uniform and 2 cql_n policy actions at s and s',
           importance-weighted if cql_is, minus Q(s, a)) per critic (CQL(H)); calql: the policy actions' Q bounded below by the
           row's Monte-Carlo return (Cal-QL, Nakamoto et al. 2023)
           actor_obs_noise: the actor's terms on observations + this x N(0, 1) (BC's obs_noise)
  actor_ema > 0: the policy that acts, is exported and gives the critic targets' actions is this EMA of the actor (as BC's)
  alpha    automatic (target entropy, settable: set_target_entropy) or fixed
  targets  Polyak averaging (tau)
TD3 mode (td3, "TD3+BC"): the actor is deterministic, a = tanh(mean) (the log_std head is unused), there is no
entropy term, the critic target uses the (delayed) actor's smoothed target action td3_target_action, the actor loss is
-mean over critics Q(s, tanh(mean(s))) (q_norm: / mean |Q|) plus the same regularisers, and the actor AND the target
critics move only every actor_delay critic updates.  Acting adds N(0, explore_std) to the action (clipped to [-1, 1]);
finetune.py decays explore_std over td3_explore_steps.  The remaining options (demo mixing, bc_* anchors, n-step
rows, actor_ema, actor_obs_noise, residual mode) apply unchanged.
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


def td3_target_action(mean, key, noise_std: float, noise_clip: float):
    """TD3's smoothed target action: tanh(mean) + clip(N(0, noise_std), +-noise_clip), clipped to [-1, 1]."""
    eps = jnp.clip(noise_std * jax.random.normal(key, mean.shape), -noise_clip, noise_clip)
    return jnp.clip(jnp.tanh(mean) + eps, -1.0, 1.0)


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
        self.td3 = bool(f["td3"])
        if self.td3:
            self.auto_alpha = False  # a deterministic actor: no entropy term, no temperature
        self.explore_std = float(f["td3_explore_std"])  # TD3's exploration noise while acting (finetune.py decays it)
        key = jax.random.PRNGKey(seed)
        key, k = jax.random.split(key)
        critic_params = self.critic.init(k, jnp.zeros((1, OBS_DIM)), jnp.zeros((1, ACTION_DIM)))
        actor_params = jax.tree_util.tree_map(jnp.asarray, actor_params)
        self.res_scale = float(f["residual_scale"])
        if f["init_log_std"] is not None:  # a constant exploration noise instead of the BC log_std head
            p = dict(actor_params["params"])
            n = len(p) - 2
            p[f"Dense_{n + 1}"] = {"kernel": jnp.zeros_like(p[f"Dense_{n + 1}"]["kernel"]),
                                   "bias": jnp.full_like(p[f"Dense_{n + 1}"]["bias"], float(f["init_log_std"]))}
            actor_params = {"params": p}
        bc_params = actor_params
        if self.res_scale > 0:  # the residual network: BC's weights, mean head zeroed (no change at the start)
            p = dict(actor_params["params"])
            n = len(p) - 2
            p[f"Dense_{n}"] = jax.tree_util.tree_map(jnp.zeros_like, p[f"Dense_{n}"])
            actor_params = {"params": p}
        # (TD3: alpha is unused; it just needs a number)
        log_alpha = jnp.log(float(f["init_ent_coef"] if (self.auto_alpha or f["ent_coef"] == "auto") else f["ent_coef"]))
        copy = partial(jax.tree_util.tree_map, lambda x: jnp.array(x, copy=True))  # distinct buffers (donated)
        self.state = {
            "actor": actor_params, "bc": copy(bc_params),
            **({"ema": copy(actor_params)} if f["actor_ema"] > 0 else {}),
            "critic": critic_params, "target": copy(critic_params),
            "log_alpha": log_alpha,
            "opt_actor": self.opt_actor.init(actor_params),
            "opt_critic": self.opt_critic.init(critic_params),
            "opt_alpha": self.opt_alpha.init(log_alpha),
            "key": key,
            "n_updates": jnp.zeros((), jnp.int32),
            "target_entropy": jnp.asarray(float(-ACTION_DIM if f["target_entropy"] is None else f["target_entropy"])),
        }
        self.pol = "ema" if f["actor_ema"] > 0 else "actor"  # the acting (and exported, and target) policy's parameters
        self._act = jax.jit(self._act_impl, static_argnames="deterministic")
        self.pi = partial(policy, res_scale=self.res_scale)
        self._chunk = jax.jit(self._chunk_impl, static_argnames="train_actor", donate_argnums=0)

    # ---- acting -------------------------------------------------------------------------------------------------
    def _act_impl(self, actor_params, bc_params, obs, key, explore_std, deterministic: bool):
        mean, log_std = self.pi(actor_params, bc_params, obs)
        if self.td3:
            a = jnp.tanh(mean)
            return a if deterministic else jnp.clip(a + explore_std * jax.random.normal(key, a.shape), -1.0, 1.0)
        return jnp.tanh(mean) if deterministic else sample_action(mean, log_std, key)[0]

    def act(self, obs: np.ndarray, key, deterministic: bool = False) -> np.ndarray:
        return np.asarray(self._act(self.state[self.pol], self.state["bc"], jnp.asarray(obs), key,
                                    jnp.float32(self.explore_std), deterministic))

    def entropy(self, obs: np.ndarray, key) -> float:
        """-mean log pi(a|s), a ~ pi(.|s), of the current actor on these observations."""
        mean, log_std = self.pi(self.state[self.pol], self.state["bc"], jnp.asarray(obs))
        return -float(sample_action(mean, log_std, key)[1].mean())

    def export_arrays(self) -> dict:
        """actor.npz arrays (bc.to_arrays format; residual mode: the BC actor's, plus res_* and res_scale)."""
        from dyngrasp.rl.bc import to_arrays

        actor = to_arrays(jax.device_get(self.state[self.pol]))
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
        key, k_next, k_sub, k_pi, k_cql, k_obs = jax.random.split(s["key"], 6)
        alpha = jnp.exp(s["log_alpha"])
        obs, act = b["obs"], b["act"]
        awr = f["awr"]
        if awr:  # state values: the critics see a zero action
            act = jnp.zeros_like(act)

        # critic
        mean_n, log_std_n = self.pi(s[self.pol], s["bc"], b["next_obs"])
        if self.td3:  # target policy smoothing; no entropy
            a_n, logp_n = td3_target_action(mean_n, k_next, float(f["td3_target_noise"]), float(f["td3_noise_clip"])), 0.0
        else:
            a_n, logp_n = sample_action(mean_n, log_std_n, k_next)
        q_next = self.critic.apply(s["target"], b["next_obs"], jnp.zeros_like(a_n) if awr else a_n)
        sub = jax.random.choice(k_sub, f["n_critics"], (f["critic_subset"],), replace=False)
        v_next = q_next[sub].min(0) - (alpha * logp_n if f["backup_entropy"] else 0.0)
        y = jax.lax.stop_gradient(b["rew"] + b["disc"] * (1.0 - b["done"]) * v_next)

        cql = f["cql_alpha"] > 0
        if cql:  # N uniform actions, N policy actions at s and N at s' (CQL(H)'s importance-sampled log-sum-exp)
            N, nb = int(f["cql_n"]), obs.shape[0]
            k_r, k_c, k_n = jax.random.split(k_cql, 3)
            mean_c, log_std_c = self.pi(s["actor"], s["bc"], obs)
            rep = lambda x: jnp.broadcast_to(x, (N, *x.shape))
            a_c, lp_c = sample_action(rep(mean_c), rep(log_std_c), k_c)
            a_nn, lp_nn = sample_action(rep(mean_n), rep(log_std_n), k_n)
            ood_a = jax.lax.stop_gradient(jnp.concatenate([jax.random.uniform(k_r, (N, nb, ACTION_DIM), minval=-1.0, maxval=1.0),
                                                           a_nn, a_c]))
            # importance weights (CQL(H)'s default) only with cql_is: with a narrow policy (log pi ~ +30) they make the
            # policy actions' terms vanish, and with them Cal-QL's bound
            ood_lp = (jax.lax.stop_gradient(jnp.concatenate([jnp.full((N, nb), ACTION_DIM * jnp.log(0.5)), lp_nn, lp_c]))
                      if f["cql_is"] else 0.0)
            ood_obs = jnp.broadcast_to(obs, (3 * N, *obs.shape))
            # Cal-QL: the policy actions' values are bounded below by the data's Monte-Carlo return
            lower = jnp.concatenate([jnp.full((N, nb), -jnp.inf), jnp.broadcast_to(b["mc"], (2 * N, nb))]) if f["calql"] else None

        def critic_loss(cp):
            q = self.critic.apply(cp, obs, act)
            loss, gap = jnp.sum(jnp.mean((q - y) ** 2, axis=1)), jnp.zeros(())
            if cql:
                qo = self.critic.apply(cp, ood_obs, ood_a)  # (critics, 3N, batch)
                if lower is not None:
                    qo = jnp.maximum(qo, lower)
                gap = jnp.mean(jax.nn.logsumexp(qo - ood_lp, axis=1) - q, axis=1).sum()
                loss = loss + float(f["cql_alpha"]) * gap
            return loss, (q, gap)

        (lc, (q, gap)), g = jax.value_and_grad(critic_loss, has_aux=True)(s["critic"])
        upd, opt_c = self.opt_critic.update(g, s["opt_critic"], s["critic"])
        critic = optax.apply_updates(s["critic"], upd)
        n_new = s["n_updates"] + 1
        do = (n_new % f["actor_delay"]) == 0  # this update also moves the actor (and, in TD3 mode, the targets)
        target = optax.incremental_update(critic, s["target"], float(f["tau"]))
        if self.td3:
            target = jax.tree_util.tree_map(lambda new, old: jnp.where(do, new, old), target, s["target"])
        s = {**s, "critic": critic, "target": target, "opt_critic": opt_c, "key": key, "n_updates": n_new}
        metrics = {"critic_loss": lc, "q_mean": q.mean(), "q_max": q.max(), "target_mean": y.mean(), "cql_gap": gap}

        # actor and temperature
        mean_bc = jax.lax.stop_gradient(jnp.tanh(actor_forward(s["bc"], obs)[0]))
        demo = b["is_demo"]
        if f["bc_q_filter"]:  # BC only where the critics prefer the demonstrated action (Nair et al. 2018)
            mean_now = jnp.tanh(self.pi(s["actor"], s["bc"], obs)[0])
            better = self.critic.apply(critic, obs, act).mean(0) > self.critic.apply(critic, obs, mean_now).mean(0)
            demo = demo * jax.lax.stop_gradient(better)

        if f["awac_beta"] > 0:  # advantage weights of the batch actions over the policy's mean action
            if awr:  # (AWR: the row's n-step return + bootstrapped value - V(s))
                adv = y - self.critic.apply(critic, obs, act).mean(0)
            else:
                mean_now = jnp.tanh(self.pi(s["actor"], s["bc"], obs)[0])
                adv = self.critic.apply(critic, obs, act).mean(0) - self.critic.apply(critic, obs, mean_now).mean(0)
            w = jnp.minimum(jnp.exp(adv / float(f["awac_beta"])), 100.0)
            w = jax.lax.stop_gradient(w / w.mean())
            metrics.update(adv_mean=adv.mean(), adv_std=adv.std(), w_max=w.max())

        noise = float(f["actor_obs_noise"])  # (as BC's bc.obs_noise: the actor sees perturbed observations)
        obs_a = obs + noise * jax.random.normal(k_obs, obs.shape) if noise > 0 else obs

        act = b["act"]

        def actor_loss(ap):
            mean, log_std = self.pi(ap, s["bc"], obs_a)
            if self.td3:  # deterministic actor
                a, logp = jnp.tanh(mean), jnp.zeros(mean.shape[:-1])
            else:
                a, logp = sample_action(mean, log_std, k_pi)
            qa = self.critic.apply(critic, obs, a).mean(0)
            dev = jnp.mean((jnp.tanh(mean) - mean_bc) ** 2, -1)
            bc_demo = jnp.sum(demo * jnp.mean((jnp.tanh(mean) - act) ** 2, -1)) / jnp.maximum(demo.sum(), 1.0)
            if f["awac_beta"] > 0:
                rl = jnp.mean(w * jnp.mean((jnp.tanh(mean) - act) ** 2, -1))
            else:
                scale = jax.lax.stop_gradient(jnp.abs(qa).mean()) if f["q_norm"] else 1.0
                rl = jnp.mean(alpha * logp - qa) / scale
            loss = rl + float(f["bc_weight"]) * bc_demo + float(f["bc_policy_weight"]) * dev.mean()
            return loss, {"logp": logp.mean(), "dev_bc": dev.mean(), "mse_demo": bc_demo, "log_std": log_std.mean(),
                          "q_pi": qa.mean()}

        (la, aux), g = jax.value_and_grad(actor_loss, has_aux=True)(s["actor"])
        metrics.update(actor_loss=la, alpha=alpha, **aux)
        if not train_actor:
            return s, metrics
        upd, opt_a = self.opt_actor.update(g, s["opt_actor"], s["actor"])
        upd = jax.tree_util.tree_map(lambda u: jnp.where(do, u, 0.0), upd)
        opt_a = jax.tree_util.tree_map(lambda new, old: jnp.where(do, new, old), opt_a, s["opt_actor"])
        s = {**s, "actor": optax.apply_updates(s["actor"], upd), "opt_actor": opt_a}
        if self.pol == "ema":
            d = jnp.where(do, float(f["actor_ema"]), 1.0)
            s = {**s, "ema": jax.tree_util.tree_map(lambda e, p: d * e + (1.0 - d) * p, s["ema"], s["actor"])}
        if self.auto_alpha and f["awac_beta"] <= 0:  # (AWAC: no entropy term)
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
