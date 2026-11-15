"""RL: PPO with an imitation term (BC demonstrations + DAgger labels from (C)), on asynchronous episode workers.

    ppo(run_dir, cfg)   # resumes from run_dir/ckpt if there is one (with the run's saved config.yaml)

One loss for the actor (initialised from the BC actor, ppo.bc_checkpoint) and its state-independent per-dimension
log_std (initialised at ppo.init_log_std, learned):

    L = w(t) L_PPO + lam(t) L_imit - sum_i c_i log sigma_i

  L_PPO   decoupled PPO (Hilton et al. 2021) on the latest `window` batches of complete episodes:
            mean[ min(prox / mu, is_cap) x min(r A, clip(r, 1 +- clip) A) ],  r = pi(u|s) / prox(u|s)
          prox the policy at the start of this update, mu the behaviour probability the worker stored when it acted
          (versions older: asynchronous workers, window); the KL early stop is against prox too.  u: pre-tanh for the
          joints (the tanh Jacobian cancels), action units for the gripper (dist_mean).  GAE advantages (gamma,
          gae_lambda) normalised per update batch.
  L_imit  on rows of (C)'s demonstrations (ppo.demos, BC's dataset) and of the aggregated DAgger labels:
            sum_i (tanh mu_i(s~) - a*_i)^2 / (2 sbar^2)  +  sum_i (r_i^2 / (2 sigma_i^2) + log sigma_i)
          s~ = s + obs_noise x N(0, 1) (BC's trick); sbar^2 = the joints' mean sigma^2, gradient stopped: the first term
          is the expert action's Gaussian log-likelihood at the policy's noise scale (commensurate with PPO's
          log-likelihood ratio), its dimensions weighted equally as in BC's MSE; r_i = atanh(a*_i) - mu_i (both clipped to
          atanh(1 - action_eps); the gripper's a*_i - tanh(mu_i)), mu's gradient stopped; r_i^2 at most (nll_clip sigma_i)^2, so that the gripper's few
          steps where the expert has closed and the policy not yet do not set its noise): the noise's own likelihood.
          While w = 0, sigma_i settles where E[min(r_i^2 / sigma_i^2, nll_clip^2)] = 1 - c_i / lam: the entropy bonus
          keeps it above the imitation residual.  Over the ramp c_i goes to ent_coef_final: with lam below c the
          entropy bonus alone grew sigma ~0.4% per batch (0.020 -> 0.041 by halfway) as the stochastic success fell.
  entropy c_i per dimension (ent_coef: smaller for the gripper).
V(s): its own network and optimiser, regressed on the GAE returns (TD(lambda) targets) of the latest value_window
batches.

Schedule, in batches: first w = 0, lam = lam0 (BC + DAgger only) while V learns the policy's own returns.  V is checked
on every new batch before it trains on it (explained variance, correlation and bias against the Monte-Carlo returns,
mean V on successful and failed episodes: metrics.jsonl).  Once its explained variance has been >= gate_ev on
gate_batches batches in a row (and min_pretrain batches have passed), over ramp_batches: w 0 -> 1, lam lam0 -> 0 and
the share of episodes that get a DAgger takeover label_fraction -> 0 (labels already collected stay).  The policy that
acts is the trained one (PPO's ratios need it); the exported and validated one is the EMA of its parameters (ema).

Workers (rollout.serve, scripts/env_workers.py --ppo): whole episodes of GraspEnv at the final level (L5, no curriculum),
never one of the L5 check scenes (scripts/bc_bigeval.py's), with the latest policy the learner exported
(run_dir/policy/latest.json, read at each episode's start): u ~ N(dist_mean(mu(s)), sigma), action tanh(u) for the
joints and clip(u, -1, 1) for the gripper (dist_mean: its Gaussian is in action units); each episode goes to
the learner (obs, u, log mu(u), rewards, outcome).  With probability label_fraction a worker then reruns the scene with
the episode's executed actions up to a switch time drawn uniformly before its failure (collision time; else its grasp
or end) and (C) from there (demos.run_takeover: (C) plans from the robot's moving state, the simulation paused), and
sends (C)'s actions on the steps from the switch on if (C) succeeded: the DAgger labels.

SIGUSR1 / SIGTERM: checkpoint and return without DONE (run again to resume).  max_steps: checkpoint, DONE.  stop_at_gate:
the same once the gate opens (after that batch's update): copies of the run directory, without DONE and with their own
config.yaml (window, ...), go on from there as separate runs.
Run directory: config.yaml, metrics.jsonl (one line per batch), labels.jsonl (every takeover), val.jsonl,
vals/<batch>/ (EMA actor; raw/: the trained one), best/ (best val50), policy/ (what the workers load), ckpt/.
"""

from __future__ import annotations

import json
import os
import pickle
import queue
import secrets
import shutil
import signal
import socket
import threading
import time
from collections import deque
from multiprocessing.connection import Listener, wait
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from dyngrasp.rl.config import load_config
from dyngrasp.rl.observation import OBS_DIM
from dyngrasp.rl.rollout import check_seeds

ACTION_DIM = 8
GRIP = 7  # the gripper's action dimension
LOG_2PI = float(np.log(2.0 * np.pi))
STOP_SIGNALS = (signal.SIGUSR1, signal.SIGTERM)
OUTCOMES = ("success", "collision", "timeout", "dropped")


# ---- policy -------------------------------------------------------------------------------------------------------


def actor_from_arrays(arrays: dict) -> dict:
    """actor.npz (BC's or ours) -> {"hidden": [(W, b), ...], "mean": (W, b)} (jnp)."""
    n = sum(k.startswith("W") and k[1:].isdigit() for k in arrays)
    f = lambda x: jnp.asarray(x, jnp.float32)
    return {"hidden": [(f(arrays[f"W{i}"]), f(arrays[f"b{i}"])) for i in range(n)], "mean": (f(arrays["Wm"]), f(arrays["bm"]))}


def export_arrays(net: dict, log_std) -> dict[str, np.ndarray]:
    """What RLController's NumpyActor and the workers read: hidden layers, mean head, and the state-independent
    log_std as a log_std head with zero weights (Wls, bls)."""
    out = {}
    for i, (W, b) in enumerate(net["hidden"]):
        out[f"W{i}"], out[f"b{i}"] = np.asarray(W), np.asarray(b)
    out["Wm"], out["bm"] = np.asarray(net["mean"][0]), np.asarray(net["mean"][1])
    out["Wls"] = np.zeros_like(out["Wm"])
    out["bls"] = np.asarray(log_std, np.float32)
    return out


def mean_fn(net: dict, x):
    for W, b in net["hidden"]:
        x = jax.nn.relu(x @ W + b)
    W, b = net["mean"]
    return x @ W + b


def dist_mean(mu):
    """The policy Gaussian's mean from the network's output mu: pre-tanh for the joints (action tanh(u)), tanh(mu) for
    the gripper (its Gaussian is in action units, the action clipped to [-1, 1]: BC's gripper mu sits at |mu| ~ 4-9,
    deep in tanh's saturation, where it drifted ~10-20 sigma per batch without changing the action, swamping PPO's
    ratios; rollout.NumpyPolicy is the same)."""
    return jnp.concatenate([mu[..., :GRIP], jnp.tanh(mu[..., GRIP:])], -1)


def gauss_logp(u, mean, log_std):
    """log N(u; mean, exp(log_std)) summed over the action dimensions (mean: dist_mean)."""
    return jnp.sum(-0.5 * ((u - mean) * jnp.exp(-log_std)) ** 2 - log_std - 0.5 * LOG_2PI, -1)


class Value(nn.Module):
    arch: tuple

    @nn.compact
    def __call__(self, x):
        for h in self.arch:
            x = nn.relu(nn.LayerNorm()(nn.Dense(h)(x)))
        return nn.Dense(1)(x)[..., 0]


# ---- returns ------------------------------------------------------------------------------------------------------


def gae(rew, val, ep_len, gamma: float, lam: float) -> np.ndarray:
    """GAE advantages of concatenated episodes, each ending in a termination (V = 0 after its last step)."""
    adv = np.zeros(len(rew), np.float64)
    s = 0
    for n in ep_len:
        last = 0.0
        for t in range(s + n - 1, s - 1, -1):
            nv = val[t + 1] if t + 1 < s + n else 0.0
            last = rew[t] + gamma * nv - val[t] + gamma * lam * last
            adv[t] = last
        s += n
    return adv


def mc_returns(rew, ep_len, gamma: float) -> np.ndarray:
    return gae(rew, np.zeros(len(rew)), ep_len, gamma, 1.0)


def value_report(v, g, ep_len, success) -> dict:
    """V against the Monte-Carlo returns g on a batch it has not trained on."""
    starts = np.concatenate([[0], np.cumsum(ep_len)[:-1]])
    per_step_success = np.repeat(success, ep_len)
    out = {"v_ev": float(1.0 - np.var(g - v) / max(np.var(g), 1e-8)), "v_corr": float(np.corrcoef(v, g)[0, 1]),
           "v_bias": float(np.mean(v - g)), "g_mean": float(np.mean(g)), "v_mean": float(np.mean(v)),
           "v_start_ev": float(1.0 - np.var(g[starts] - v[starts]) / max(np.var(g[starts]), 1e-8))}
    for name, m, ms in (("succ", success, per_step_success), ("fail", ~success, ~per_step_success)):
        if m.any():
            out.update({f"v0_{name}": float(np.mean(v[starts][m])), f"g0_{name}": float(np.mean(g[starts][m])),
                        f"v_{name}": float(np.mean(v[ms])), f"g_{name}": float(np.mean(g[ms]))})
    return out


# ---- the learner's update ---------------------------------------------------------------------------------------------


class Learner:
    """Actor (net, log_std, EMA), value function and their optimisers in `self.s` (what checkpoints save)."""

    def __init__(self, cfg: dict, actor_arrays: dict, seed: int = 0):
        p = self.p = cfg["ppo"]
        self.gamma = cfg["gamma"]
        net = actor_from_arrays(actor_arrays)
        log_std = jnp.asarray(p["init_log_std"], jnp.float32)
        assert log_std.shape == (ACTION_DIM,)
        self.value = Value(tuple(p["value_arch"]))
        vparams = self.value.init(jax.random.PRNGKey(seed), jnp.zeros((1, OBS_DIM)))
        self.opt_net = optax.chain(optax.clip_by_global_norm(p["max_grad_norm"]), optax.adam(float(p["lr_actor"])))
        self.opt_ls = optax.adam(float(p["lr_log_std"]))
        self.opt_v = optax.chain(optax.clip_by_global_norm(p["max_grad_norm"]), optax.adam(float(p["lr_value"])))
        self.s = {"net": net, "ema": net, "log_std": log_std, "v": vparams, "os_net": self.opt_net.init(net),
                  "os_ls": self.opt_ls.init(log_std), "os_v": self.opt_v.init(vparams)}
        self.lo = jnp.asarray(p["log_std_bounds"][0], jnp.float32)  # a number or one per dimension
        self.hi = jnp.asarray(p["log_std_bounds"][1], jnp.float32)
        self.u_max = float(np.arctanh(1.0 - p["action_eps"]))
        self._actor_step = jax.jit(self._actor_step_impl)
        self._value_step = jax.jit(self._value_step_impl)
        self._v = jax.jit(lambda vp, x: self.value.apply(vp, x))
        self._mu = jax.jit(mean_fn)

    # -- numpy in / out
    def values(self, obs: np.ndarray, chunk: int = 65536) -> np.ndarray:
        return np.concatenate([np.asarray(self._v(self.s["v"], obs[i: i + chunk])) for i in range(0, len(obs), chunk)])

    def logp(self, obs, u, net=None) -> np.ndarray:
        mu = dist_mean(self._mu(self.s["net"] if net is None else net, obs))
        return np.asarray(gauss_logp(jnp.asarray(u), mu, jnp.clip(self.s["log_std"], self.lo, self.hi)))

    def export(self, ema: bool = True) -> dict:
        return export_arrays(self.s["ema" if ema else "net"], jnp.clip(self.s["log_std"], self.lo, self.hi))

    def state_numpy(self) -> dict:
        return jax.tree_util.tree_map(np.asarray, self.s)

    def load_state_numpy(self, st: dict) -> None:
        self.s = jax.tree_util.tree_map(jnp.asarray, st)

    # -- one update of the actor on a PPO minibatch and an imitation minibatch
    def _actor_step_impl(self, s, mb, im, coefs, key):
        p = self.p
        w, lam, c = coefs["w"], coefs["lam"], coefs["c"]

        def loss(net, log_std):
            log_std = jnp.clip(log_std, self.lo, self.hi)
            mu = mean_fn(net, mb["obs"])
            logp = gauss_logp(mb["u"], dist_mean(mu), log_std)
            # decoupled PPO: the clipped ratio against the policy at the start of this update (prox), weighted by
            # prox / behaviour (the workers' policy, versions older), truncated at is_cap
            ratio = jnp.exp(jnp.clip(logp - mb["logp_prox"], -20.0, 20.0))
            w_is = jnp.minimum(jnp.exp(jnp.clip(mb["logp_prox"] - mb["logp"], -20.0, 20.0)), p["is_cap"])
            a = mb["adv"]
            l_ppo = -jnp.mean(w_is * jnp.minimum(ratio * a, jnp.clip(ratio, 1.0 - p["clip"], 1.0 + p["clip"]) * a))
            x = im["obs"] + p["obs_noise"] * jax.random.normal(key, im["obs"].shape)
            mu_i = mean_fn(net, x)
            var = jnp.exp(2.0 * log_std)
            sbar2 = jax.lax.stop_gradient(jnp.mean(var[:7]))
            l_mean = jnp.mean(jnp.sum((jnp.tanh(mu_i) - im["act"]) ** 2, -1)) / (2.0 * sbar2)
            mu_sg = jax.lax.stop_gradient(mu_i)  # the residual in each dimension's Gaussian's units (dist_mean)
            r = jnp.concatenate([jnp.arctanh(jnp.clip(im["act"][:, :GRIP], -1.0 + p["action_eps"], 1.0 - p["action_eps"]))
                                 - jnp.clip(mu_sg[:, :GRIP], -self.u_max, self.u_max),
                                 im["act"][:, GRIP:] - jnp.tanh(mu_sg[:, GRIP:])], -1)
            l_std = jnp.mean(jnp.sum(jnp.minimum(r**2, p["nll_clip"] ** 2 * var) / (2.0 * var) + log_std, -1))
            l_ent = -jnp.sum(c * log_std)
            total = w * l_ppo + lam * (l_mean + l_std) + l_ent
            aux = {"l_ppo": l_ppo, "l_mean": l_mean, "l_std": l_std, "approx_kl": jnp.mean(mb["logp_prox"] - logp),
                   "is_mean": jnp.mean(w_is), "is_capped": jnp.mean(w_is >= p["is_cap"]),
                   "clipfrac": jnp.mean(jnp.abs(ratio - 1.0) > p["clip"]), "imit_mse": jnp.mean((jnp.tanh(mu_i) - im["act"]) ** 2),
                   "imit_resid_rms": jnp.sqrt(jnp.mean(r**2, 0))}
            return total, aux

        (_, aux), (g_net, g_ls) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(s["net"], s["log_std"])
        upd, os_net = self.opt_net.update(g_net, s["os_net"], s["net"])
        net = optax.apply_updates(s["net"], upd)
        upd, os_ls = self.opt_ls.update(g_ls, s["os_ls"], s["log_std"])
        log_std = jnp.clip(optax.apply_updates(s["log_std"], upd), self.lo, self.hi)
        d = p["ema"]
        ema = jax.tree_util.tree_map(lambda e, x: d * e + (1.0 - d) * x, s["ema"], net) if d > 0 else net
        aux["grad_norm"] = optax.global_norm(g_net)
        return {**s, "net": net, "ema": ema, "log_std": log_std, "os_net": os_net, "os_ls": os_ls}, aux

    def _value_step_impl(self, s, obs, ret):
        def loss(vp):
            return jnp.mean((self.value.apply(vp, obs) - ret) ** 2)

        l, g = jax.value_and_grad(loss)(s["v"])
        upd, os_v = self.opt_v.update(g, s["os_v"], s["v"])
        return {**s, "v": optax.apply_updates(s["v"], upd), "os_v": os_v}, l

    def update(self, ppo_batch: dict, value_batch: dict, imitation, coefs: dict, rng: np.random.Generator, key) -> dict:
        """PPO epochs (early stop at 1.5 target_kl) with an imitation minibatch per step, then the value epochs."""
        p = self.p
        n, M = len(ppo_batch["adv"]), p["minibatch"]
        ppo_batch = {**ppo_batch, "logp_prox": self.logp(ppo_batch["obs"], ppo_batch["u"]).astype(np.float32)}
        jcoefs = {"w": jnp.float32(coefs["w"]), "lam": jnp.float32(coefs["lam"]), "c": jnp.asarray(coefs["c"], jnp.float32)}
        dev = {k: jnp.asarray(v) for k, v in ppo_batch.items()}
        logs, epochs = [], 0
        for _ in range(p["epochs"]):
            perm = rng.permutation(n)
            kls = []
            for i in range(max(1, n // M)):
                idx = jnp.asarray(perm[i * M: (i + 1) * M])
                mb = {k: v[idx] for k, v in dev.items()}
                key, k = jax.random.split(key)
                self.s, aux = self._actor_step(self.s, mb, imitation(rng), jcoefs, k)
                logs.append(aux)
                kls.append(aux["approx_kl"])
            epochs += 1
            if coefs["w"] > 0 and float(jnp.mean(jnp.stack(kls))) > 1.5 * p["target_kl"]:
                break
        vl = []
        nv = len(value_batch["ret"])
        vdev = {k: jnp.asarray(v) for k, v in value_batch.items()}
        for _ in range(p["value_epochs"]):
            perm = rng.permutation(nv)
            for i in range(max(1, nv // M)):
                idx = jnp.asarray(perm[i * M: (i + 1) * M])
                self.s, l = self._value_step(self.s, vdev["obs"][idx], vdev["ret"][idx])
                vl.append(l)
        out = {k: float(np.mean([np.asarray(a[k]) for a in logs])) for k in logs[0] if k != "imit_resid_rms"}
        out["imit_resid_rms"] = np.mean([np.asarray(a["imit_resid_rms"]) for a in logs], 0).round(4).tolist()
        out.update(epochs=epochs, n_actor_steps=len(logs), value_loss=float(np.mean(jnp.stack(vl))))
        return out


# ---- data from the workers ----------------------------------------------------------------------------------------


class Receiver:
    """Listens for workers (address in addr_file, key in addr_file.key); sends each its ("init", payload(i)) and puts
    every message it sends into `self.q` (a background thread per role)."""

    def __init__(self, addr_file: Path, payload):
        self.payload, self.q = payload, queue.Queue()
        self.conns: set = set()
        self.lock = threading.Lock()
        self.n_conn = 0
        key = secrets.token_bytes(32)
        self.listener = Listener(("0.0.0.0", 0), authkey=key)
        tmp = addr_file.with_name(f".{addr_file.name}.tmp")
        Path(str(tmp) + ".key").write_bytes(key)
        os.chmod(str(tmp) + ".key", 0o600)
        os.replace(str(tmp) + ".key", str(addr_file) + ".key")
        tmp.write_text(f"{socket.gethostname()} {self.listener.address[1]}\n")
        os.replace(tmp, addr_file)
        self.closed = False
        threading.Thread(target=self._accept, daemon=True).start()
        threading.Thread(target=self._recv, daemon=True).start()

    @property
    def live(self) -> int:
        return len(self.conns)

    def _accept(self) -> None:
        while not self.closed:
            try:
                c = self.listener.accept()
                self.n_conn += 1
                c.send(("init", self.payload(self.n_conn)))
            except Exception:  # noqa: BLE001, S112  (closed, or a client that failed the handshake)
                continue
            with self.lock:
                self.conns.add(c)

    def _recv(self) -> None:
        while not self.closed:
            with self.lock:
                cs = list(self.conns)
            if not cs:
                time.sleep(0.2)
                continue
            for c in wait(cs, 0.5):
                try:
                    self.q.put(c.recv())
                except (EOFError, OSError):
                    with self.lock:
                        self.conns.discard(c)
                    c.close()

    def close(self) -> None:
        self.closed = True
        self.listener.close()
        with self.lock:
            for c in self.conns:
                try:
                    c.send(("close", None))
                except OSError:
                    pass


class Imitation:
    """Imitation minibatches: (C)'s demonstrations (on the device) and the aggregated DAgger labels (host, growing)."""

    def __init__(self, demos_path, batch: int, dagger_fraction: float, min_dagger: int, dagger_init=None):
        z = np.load(demos_path)
        self.demo_obs, self.demo_act = jnp.asarray(z["obs"]), jnp.asarray(z["act"])
        self.n_demo = len(z["act"])
        self.batch, self.frac, self.min_dagger = batch, dagger_fraction, min_dagger
        self.obs = np.zeros((0, OBS_DIM), np.float32)
        self.act = np.zeros((0, ACTION_DIM), np.float32)
        if dagger_init:  # labels made before the run (scripts/ppo_offline.py): reloaded at each start, never saved
            d = np.load(dagger_init)
            self.obs, self.act = d["obs"].astype(np.float32), d["act"].astype(np.float32)
        self.n_init = len(self.act)
        self._chunks: list = []

    def add(self, obs, act) -> None:
        self._chunks.append((np.asarray(obs, np.float32), np.asarray(act, np.float32)))

    @property
    def n_dagger(self) -> int:
        return len(self.act) + sum(len(a) for _, a in self._chunks)

    def _flush(self) -> None:
        if self._chunks:
            self.obs = np.concatenate([self.obs, *(o for o, _ in self._chunks)])
            self.act = np.concatenate([self.act, *(a for _, a in self._chunks)])
            self._chunks = []

    def __call__(self, rng: np.random.Generator) -> dict:
        self._flush()
        nd = round(self.batch * self.frac) if len(self.act) >= self.min_dagger else 0
        i = jnp.asarray(rng.integers(0, self.n_demo, self.batch - nd))
        obs, act = self.demo_obs[i], self.demo_act[i]
        if nd:
            j = rng.integers(0, len(self.act), nd)
            obs, act = jnp.concatenate([obs, jnp.asarray(self.obs[j])]), jnp.concatenate([act, jnp.asarray(self.act[j])])
        return {"obs": obs, "act": act}

    def save(self, path: Path) -> None:
        self._flush()
        tmp = path.with_name(f".{path.name}.tmp.npz")
        np.savez(tmp, obs=self.obs[self.n_init:], act=self.act[self.n_init:])
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        z = np.load(path)
        self.obs = np.concatenate([self.obs[: self.n_init], z["obs"]])
        self.act = np.concatenate([self.act[: self.n_init], z["act"]])
        self._chunks = []


def pretrain_value(learner: Learner, path, gamma: float, epochs: int, rng: np.random.Generator) -> dict:
    """V regressed on the discounted returns of rollouts made before the run (scripts/ppo_offline.py merge), 10% of the
    episodes held out: value_report on them before and after."""
    z = np.load(path)
    ep_len, success = z["ep_len"], z["success"]
    g = mc_returns(z["rew"], ep_len, gamma).astype(np.float32)
    ep = np.repeat(np.arange(len(ep_len)), ep_len)
    held = rng.random(len(ep_len)) < 0.1
    tr, te = ~held[ep], held[ep]
    obs = z["obs"].astype(np.float32)

    def report():
        return value_report(learner.values(obs[te]), g[te], ep_len[held], success[held])

    before = report()
    o, r = jnp.asarray(obs[tr]), jnp.asarray(g[tr])
    M = learner.p["minibatch"]
    for _ in range(epochs):
        perm = rng.permutation(len(r))
        for i in range(len(r) // M):
            idx = jnp.asarray(perm[i * M: (i + 1) * M])
            learner.s, _ = learner._value_step(learner.s, o[idx], r[idx])
    return {"episodes": len(ep_len), "rows": len(g), "success": float(success.mean()), "before": before, "after": report()}


def schedule(p: dict, batch: int, ramp_start: int | None) -> dict:
    """w, lam, label fraction and ramp progress x at this batch."""
    x = 0.0 if ramp_start is None else float(np.clip((batch - ramp_start) / max(p["ramp_batches"], 1), 0.0, 1.0))
    c = [(1.0 - x) * c0 + x * c1 for c0, c1 in zip(p["ent_coef"], p["ent_coef_final"])]
    return {"x": x, "w": x * p["ppo_weight"], "lam": (1.0 - x) * p["imit_weight"],
            "label_fraction": (1.0 - x) * p["label_fraction"], "c": c}


def assemble(episodes: list[dict]) -> dict:
    cat = lambda k: np.concatenate([e[k] for e in episodes])
    return {"obs": cat("obs").astype(np.float32), "u": cat("u").astype(np.float32), "logp": cat("logp").astype(np.float32),
            "rew": cat("rew").astype(np.float64), "ep_len": np.array([len(e["rew"]) for e in episodes]),
            "success": np.array([e["outcome"] == "success" for e in episodes]),
            "version": np.array([e["version"] for e in episodes])}


# ---- the run ------------------------------------------------------------------------------------------------------


def ppo(run_dir, cfg: dict) -> str:
    from dyngrasp.rl.finetune import Validator

    run_dir = Path(run_dir)
    if (run_dir / "DONE").exists():
        return (run_dir / "DONE").read_text().strip()
    signals: list[str] = []
    prev = {s: signal.signal(s, lambda signum, frame: signals.append(signal.Signals(signum).name)) for s in STOP_SIGNALS}
    rx = validator = None
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt, pol_dir = run_dir / "ckpt", run_dir / "policy"
        if (ckpt / "state.json").exists():
            cfg = load_config(run_dir / "config.yaml")  # (defaults for options added since the run started)
            st = json.loads((ckpt / "state.json").read_text())
        else:
            (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            st = {"batch": 0, "steps": 0, "episodes": 0, "starts": 0, "ramp_start": None, "gate_run": 0, "best": None,
                  "version": 0, "takeovers": 0, "labels": 0}
        st["starts"] += 1
        p = cfg["ppo"]
        rng = np.random.default_rng([p["seed"], st["starts"]])
        key = jax.random.PRNGKey(int(rng.integers(2**31)))
        learner = Learner(cfg, dict(np.load(Path(p["bc_checkpoint"]) / "actor.npz")), seed=p["seed"])
        imitation = Imitation(p["demos"], p["imit_batch"], p["dagger_fraction"], p["min_dagger"], p["dagger_init"])
        if (ckpt / "learner.pkl").exists():
            learner.load_state_numpy(pickle.loads((ckpt / "learner.pkl").read_bytes()))
        elif p["value_init"]:
            rep = pretrain_value(learner, p["value_init"], cfg["gamma"], p["value_init_epochs"], rng)
            (run_dir / "value_init.json").write_text(json.dumps(rep, indent=1))
            print(f"V pretrained on {rep['episodes']} rollouts: held-out explained variance {rep['before']['v_ev']:.3f} -> "
                  f"{rep['after']['v_ev']:.3f}", flush=True)
        if (ckpt / "dagger.npz").exists():
            imitation.load(ckpt / "dagger.npz")
        seeds_file = run_dir / "check_seeds.json"
        if not seeds_file.exists():
            seeds_file.write_text(json.dumps(check_seeds()))
        excluded = json.loads(seeds_file.read_text())

        def publish(version: int, label_fraction: float) -> None:
            pol_dir.mkdir(exist_ok=True)
            np.savez(pol_dir / f".w{version}.npz", **learner.export(ema=False))
            os.replace(pol_dir / f".w{version}.npz", pol_dir / f"w{version}.npz")
            tmp = pol_dir / ".latest.json"
            tmp.write_text(json.dumps({"version": version, "file": f"w{version}.npz", "label_fraction": label_fraction}))
            os.replace(tmp, pol_dir / "latest.json")
            for old in pol_dir.glob("w*.npz"):
                if int(old.stem[1:]) < version - 3:
                    old.unlink(missing_ok=True)

        sch = schedule(p, st["batch"], st["ramp_start"])
        publish(st["version"], sch["label_fraction"])
        worker_cfg = {**cfg, "curriculum": {**cfg["curriculum"], "current_share": 1.0},
                      "demos": {**cfg["demos"], "max_wall_s": p["label_max_wall_s"]}}
        rx = Receiver(run_dir / "workers.addr",
                      lambda i: (worker_cfg, p["seed"] * 1_000_000 + st["starts"] * 100_000 + i, str(run_dir), excluded))
        validator = Validator(cfg, p["val_workers"])
        print(f"start {st['starts']}: batch {st['batch']}, {st['steps']} steps, {imitation.n_dagger} DAgger labels, "
              f"device {jax.devices()[0]}", flush=True)

        def save_policy(out: Path) -> Path:
            out.mkdir(parents=True, exist_ok=True)
            np.savez(out / "actor.npz", **learner.export(ema=True))
            (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            (out / "raw").mkdir(exist_ok=True)
            np.savez(out / "raw" / "actor.npz", **learner.export(ema=False))
            (out / "raw" / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            return out

        def checkpoint():
            tmp = ckpt.with_name(".ckpt_tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
            (tmp / "learner.pkl").write_bytes(pickle.dumps(learner.state_numpy()))
            imitation.save(tmp / "dagger.npz")
            (tmp / "state.json").write_text(json.dumps(st, indent=1))
            ckpt.mkdir(exist_ok=True)
            for name in ("learner.pkl", "dagger.npz", "state.json"):
                os.replace(tmp / name, ckpt / name)
            shutil.rmtree(tmp, ignore_errors=True)

        pending: list[dict] = []
        n_pending = 0
        window: deque = deque(maxlen=max(p["window"], p["value_window"]))
        recent = deque(maxlen=400)  # episodes, for the rolling outcome rates
        labels_log = (run_dir / "labels.jsonl").open("a")
        t_ckpt = t_start = time.time()
        t_batch = time.time()
        reason = None
        while True:
            # what the workers sent
            try:
                while True:
                    kind, d = rx.q.get(timeout=0.05)
                    if kind == "episode":
                        pending.append(d)
                        n_pending += len(d["rew"])
                        recent.append(d)
                    elif kind == "labels":
                        st["takeovers"] += 1
                        if "obs" in d:
                            imitation.add(d.pop("obs"), d.pop("act"))
                            st["labels"] += d["n_labels"]
                        labels_log.write(json.dumps(d, default=float) + "\n")
                    if n_pending >= p["batch_steps"]:
                        break
            except queue.Empty:
                pass
            if n_pending >= p["batch_steps"]:
                b = assemble(pending)
                pending, n_pending = [], 0
                t_wait = time.time() - t_batch
                tic = time.time()
                # V on the new batch before training on it: the tracking numbers and the gate
                v = learner.values(b["obs"])
                g = mc_returns(b["rew"], b["ep_len"], cfg["gamma"])
                rep = value_report(v, g, b["ep_len"], b["success"])
                b["adv"] = gae(b["rew"], v, b["ep_len"], cfg["gamma"], p["gae_lambda"])
                b["ret"] = b["adv"] + v
                window.append(b)
                if st["ramp_start"] is None:
                    st["gate_run"] = st["gate_run"] + 1 if rep["v_ev"] >= p["gate_ev"] else 0
                    if st["gate_run"] >= p["gate_batches"] and st["batch"] + 1 >= p["min_pretrain"]:
                        st["ramp_start"] = st["batch"] + 1
                        print(f"gate open at batch {st['batch']}: V explained variance {rep['v_ev']:.3f} "
                              f"({p['gate_batches']} batches >= {p['gate_ev']}); PPO ramps in from batch {st['ramp_start']}",
                              flush=True)
                        if p["stop_at_gate"]:
                            reason = (f"gate open at batch {st['batch']} ({st['steps'] + int(b['ep_len'].sum())} steps): "
                                      f"stopped before PPO ramps in (ppo.stop_at_gate)")
                sch = schedule(p, st["batch"], st["ramp_start"])
                # PPO batch: the latest `window` batches, advantages recomputed with the current V
                wins = list(window)[-p["window"]:]
                for wb in wins[:-1]:
                    wv = learner.values(wb["obs"])
                    wb["adv"] = gae(wb["rew"], wv, wb["ep_len"], cfg["gamma"], p["gae_lambda"])
                    wb["ret"] = wb["adv"] + wv
                cat = lambda k, bs: np.concatenate([x[k] for x in bs])
                adv = cat("adv", wins)
                ppo_batch = {"obs": cat("obs", wins), "u": cat("u", wins), "logp": cat("logp", wins),
                             "adv": ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32)}
                vb = list(window)[-p["value_window"]:]
                value_batch = {"obs": cat("obs", vb), "ret": cat("ret", vb).astype(np.float32)}
                lr0 = np.clip(learner.logp(ppo_batch["obs"], ppo_batch["u"]) - ppo_batch["logp"], -20.0, 20.0)
                key, k = jax.random.split(key)
                m = learner.update(ppo_batch, value_batch, imitation, sch, rng, k)
                st["batch"] += 1
                st["version"] += 1
                st["steps"] += int(b["ep_len"].sum())
                st["episodes"] += len(b["ep_len"])
                publish(st["version"], schedule(p, st["batch"], st["ramp_start"])["label_fraction"])
                eps = list(recent)
                row = {"batch": st["batch"], "steps": st["steps"], "episodes": st["episodes"], "hours": (time.time() - t_start) / 3600,
                       "workers": rx.live, "wait_s": t_wait, "update_s": time.time() - tic,
                       "steps_per_s": float(b["ep_len"].sum()) / max(t_wait + time.time() - tic, 1e-6),
                       "batch_episodes": len(b["ep_len"]), "batch_success": float(b["success"].mean()),
                       "version_lag": float(np.mean(st["version"] - 1 - b["version"])), "ratio0_mean": float(np.exp(lr0).mean()),
                       "logratio0_std": float(lr0.std()),
                       "ramp_x": sch["x"], "w": sch["w"], "lam": sch["lam"], "label_fraction": sch["label_fraction"],
                       "gate_run": st["gate_run"], "takeovers": st["takeovers"], "dagger_labels": imitation.n_dagger,
                       "log_std": np.asarray(learner.s["log_std"]).round(3).tolist(), **rep, **m}
                for o in OUTCOMES:
                    row[o] = float(np.mean([e["outcome"] == o for e in eps]))
                cats = [e["collision"].split(":")[0] for e in eps if e["outcome"] == "collision"]
                row["coll_obstacle"] = cats.count("obstacle") / len(eps)
                row["coll_cart_floor"] = (cats.count("cart") + cats.count("floor")) / len(eps)
                row["coll_target"] = cats.count("target") / len(eps)
                for term in eps[0]["ep_terms"]:
                    row[f"term_{term}"] = float(np.mean([e["ep_terms"][term] for e in eps]))
                row["return"] = float(np.mean([e["rew"].sum() for e in eps]))
                row["ep_len"] = float(np.mean([len(e["rew"]) for e in eps]))
                with (run_dir / "metrics.jsonl").open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
                t_batch = time.time()
                if st["batch"] % p["val_every"] == 0 and not validator.busy:
                    validator.start(st["batch"], save_policy(run_dir / "vals" / f"{st['batch']:06d}"))
                labels_log.flush()
            vr = validator.result()
            if vr is not None:
                with (run_dir / "val.jsonl").open("a") as fh:
                    fh.write(json.dumps(vr) + "\n")
                bst = st["best"]
                if bst is None or (vr["success"], -vr["collision"]) > (bst["success"], -bst["collision"]):
                    st["best"] = vr
                    shutil.rmtree(run_dir / "best", ignore_errors=True)
                    shutil.copytree(vr["dir"], run_dir / "best")
                print(f"validation at batch {vr['steps']}: success {vr['success']:.2f} collision {vr['collision']:.2f} "
                      f"dropped {vr['dropped']:.2f} timeout {vr['timeout']:.2f} ({vr['wall_s']:.0f} s)", flush=True)
            now = time.time()
            if st["steps"] >= p["max_steps"]:
                reason = f"max_steps ({p['max_steps']}) reached"
            if reason or signals or now - t_ckpt >= 60 * p["checkpoint_minutes"]:
                checkpoint()
                t_ckpt = time.time()
                print(f"checkpoint at batch {st['batch']}", flush=True)
            if reason:
                save_policy(run_dir / "final")
                (run_dir / "DONE").write_text(reason + "\n")
                return reason
            if signals:
                return f"interrupted by {signals[0]} at batch {st['batch']} (checkpointed; run again to resume)"
    finally:
        for s, h in prev.items():
            signal.signal(s, h)
        if validator is not None:
            validator.close()
        if rx is not None:
            rx.close()
