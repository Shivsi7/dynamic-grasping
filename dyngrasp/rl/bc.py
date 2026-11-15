"""(D) behaviour cloning of (C): the SAC actor trained on (C)'s demonstrations (scripts/make_bc_demos.py).

Model: sbx's SquashedGaussianActor (hidden layers config sac.net_arch, ReLU; mean and log_std heads), the
actor a SAC fine-tuning can start from (`sbx_params`).  Same observation as RL (fixed scales).  Loss:

    MSE(tanh(mean), a) + nll_weight x NLL(atanh(a) | stop_gradient(mean), exp(log_std))

with the log_std head on the trunk's features with their gradient stopped (it does not move the policy).
Adam on minibatches (optionally: Gaussian noise on the observations, bc.obs_noise; the policy an
exponential moving average of the weights, bc.ema).  After every epoch the MSE on held-out episodes
(val_fraction, split by episode); training stops after `patience` epochs without a lower one and keeps
the lowest (best/).  Every closed_loop_every
epochs and on best/, the deterministic policy tanh(mean) runs on the validation scenes in the harness
(config env latency, graded as in evaluation).

Run directory:
  config.yaml   the configuration      run.json   dataset, split, timings
  log.csv       per epoch: train loss / MSE, validation MSE / NLL, closed-loop outcomes
  epochs/<n>/   the policies run in closed loop     best/   lowest validation MSE (+ closed_loop.json)
  Each policy directory has actor.npz (with the log_std head) and config.yaml: what RLController and
  scripts/eval_rl.py load.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import yaml

from dyngrasp.rl.controller import NumpyActor
from dyngrasp.rl.curriculum import OUTCOMES
from dyngrasp.rl.observation import OBS_DIM
from dyngrasp.rl.scenes import validation_scenes

ACTION_DIM = 8


# ---- data ---------------------------------------------------------------------------------------------------


def load_dataset(path, val_fraction: float, seed: int) -> dict:
    """The dataset's arrays and a split of its episodes: train / val transition indices."""
    z = np.load(path)
    obs, act, start, length = z["obs"], z["act"], z["ep_start"], z["ep_len"]
    assert obs.shape == (len(act), OBS_DIM) and act.shape[1] == ACTION_DIM and length.sum() == len(act)
    perm = np.random.default_rng(seed).permutation(len(start))
    val_eps = np.sort(perm[: round(val_fraction * len(start))])
    is_val = np.zeros(len(act), bool)
    for e in val_eps:
        is_val[start[e]: start[e] + length[e]] = True
    return {"obs": obs, "act": act, "train": np.flatnonzero(~is_val), "val": np.flatnonzero(is_val),
            "val_episodes": val_eps, "n_episodes": len(start)}


# ---- model ----------------------------------------------------------------------------------------------------


def init_params(net_arch, seed: int):
    """sbx SAC actor parameters ({"params": {"Dense_0": {kernel, bias}, ...}}), sbx's initialisation."""
    from sbx.common.policies import SquashedGaussianActor

    actor = SquashedGaussianActor(net_arch=list(net_arch), action_dim=ACTION_DIM)
    return jax.tree_util.tree_map(jnp.asarray, actor.init(jax.random.PRNGKey(seed), jnp.zeros((1, OBS_DIM))))


def forward(params, x):
    """(mean, log_std) as sbx's SquashedGaussianActor computes them; log_std from stop_gradient(features)."""
    p = params["params"]
    n = len(p) - 2
    for i in range(n):
        x = jax.nn.relu(x @ p[f"Dense_{i}"]["kernel"] + p[f"Dense_{i}"]["bias"])
    mean = x @ p[f"Dense_{n}"]["kernel"] + p[f"Dense_{n}"]["bias"]
    log_std = jax.lax.stop_gradient(x) @ p[f"Dense_{n + 1}"]["kernel"] + p[f"Dense_{n + 1}"]["bias"]
    return mean, jnp.clip(log_std, -20.0, 2.0)


def losses(params, x, a, eps: float):
    """(MSE of tanh(mean), Gaussian NLL of atanh(a) per dimension with the mean held fixed)."""
    mean, log_std = forward(params, x)
    mse = jnp.mean((jnp.tanh(mean) - a) ** 2)
    u = jnp.arctanh(jnp.clip(a, -1.0 + eps, 1.0 - eps))
    z = (u - jax.lax.stop_gradient(mean)) * jnp.exp(-log_std)
    nll = jnp.mean(0.5 * z**2 + log_std + 0.5 * jnp.log(2 * jnp.pi))
    return mse, nll


def to_arrays(params) -> dict[str, np.ndarray]:
    """actor.npz: NumpyActor's arrays (hidden layers, mean) and the log_std head (Wls, bls)."""
    p = params["params"]
    n = len(p) - 2
    out = NumpyActor.arrays_from_sbx(jax.tree_util.tree_map(np.asarray, params))
    out["Wls"], out["bls"] = np.asarray(p[f"Dense_{n + 1}"]["kernel"]), np.asarray(p[f"Dense_{n + 1}"]["bias"])
    return out


def sbx_params(arrays: dict[str, np.ndarray]):
    """actor.npz -> sbx actor parameters (for model.policy.actor_state.replace(params=...))."""
    n = sum(k.startswith("W") and k[1:].isdigit() for k in arrays)
    layers = [(arrays[f"W{i}"], arrays[f"b{i}"]) for i in range(n)] + [(arrays["Wm"], arrays["bm"]), (arrays["Wls"], arrays["bls"])]
    return {"params": {f"Dense_{i}": {"kernel": jnp.asarray(W), "bias": jnp.asarray(b)} for i, (W, b) in enumerate(layers)}}


def save_policy(params, cfg: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "actor.npz", **to_arrays(params))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


# ---- closed loop ------------------------------------------------------------------------------------------------


def _run_scene(checkpoint: str, spec_dict: dict, latency) -> dict:
    from dyngrasp.harness import run_episode
    from dyngrasp.rl.controller import RLController
    from dyngrasp.scenario import ScenarioSpec

    return run_episode(ScenarioSpec.from_dict(spec_dict), RLController(checkpoint), keep_log=False, latency=latency).row()


def closed_loop(checkpoint: Path, specs, workers: int, latency) -> tuple[dict, list[dict]]:
    """The policy at `checkpoint` on each scene (harness, graded as in evaluation): outcome rates and the rows."""
    with ProcessPoolExecutor(min(workers, len(specs)), mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(_run_scene, [str(checkpoint)] * len(specs), [s.to_dict() for s in specs], [latency] * len(specs)))
    out = {o: float(np.mean([r["outcome"] == o for r in rows])) for o in OUTCOMES}
    t = [r["t_grasp"] for r in rows if r["outcome"] == "success"]
    out.update(n=len(rows), t_grasp_median=float(np.median(t)) if t else None)
    return out, rows


# ---- training ----------------------------------------------------------------------------------------------------


def train(run_dir, cfg: dict, data_path) -> str:
    """Train on the dataset at `data_path` in run_dir (a fresh run); returns why it stopped."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    c = cfg["bc"]
    t_start = time.time()
    data = load_dataset(data_path, c["val_fraction"], c["seed"])
    obs, act = jax.device_put(data["obs"]), jax.device_put(data["act"])
    tr, va = data["train"], jnp.asarray(data["val"])
    info = {"data": str(Path(data_path).resolve()), "n_episodes": data["n_episodes"], "n_train": len(tr),
            "n_val": len(va), "val_episodes": data["val_episodes"].tolist(), "device": str(jax.devices()[0])}
    print({k: v for k, v in info.items() if k != "val_episodes"}, flush=True)

    params = init_params(cfg["sac"]["net_arch"], c["seed"])
    opt = optax.adam(float(c["learning_rate"]))  # (YAML reads "1e-3" as a string)
    opt_state = opt.init(params)
    eps, w = float(c["action_eps"]), float(c["nll_weight"])

    decay, noise = float(c["ema"]), float(c["obs_noise"])

    @jax.jit
    def step(params, ema, opt_state, obs, act, idx, key):
        x = obs[idx]
        if noise > 0:
            x = x + noise * jax.random.normal(key, x.shape)

        def loss(p):
            mse, nll = losses(p, x, act[idx], eps)
            return mse + w * nll, (mse, nll)

        (l, aux), g = jax.value_and_grad(loss, has_aux=True)(params)
        updates, opt_state = opt.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema = jax.tree_util.tree_map(lambda e, p: decay * e + (1.0 - decay) * p, ema, params) if decay > 0 else params
        return params, ema, opt_state, l, aux[0]

    @jax.jit
    def evaluate(params, obs, act, idx):
        return losses(params, obs[idx], act[idx], eps)

    val_specs = validation_scenes()[: cfg["validation"]["n_scenes"]]
    latency = cfg["env"]["latency"]
    rng = np.random.default_rng(c["seed"])
    key = jax.random.PRNGKey(c["seed"])
    ema = params
    B = c["batch_size"]
    best, best_epoch, rows, reason = np.inf, 0, [], f"max_epochs ({c['max_epochs']})"
    for epoch in range(1, c["max_epochs"] + 1):
        t = time.time()
        perm = rng.permutation(tr)
        ls, ms = [], []
        for i in range(len(perm) // B):
            key, k = jax.random.split(key)
            params, ema, opt_state, l, m = step(params, ema, opt_state, obs, act, jnp.asarray(perm[i * B: (i + 1) * B]), k)
            ls.append(l)
            ms.append(m)
        v_mse, v_nll = (float(x) for x in evaluate(ema, obs, act, va))
        row = {"epoch": epoch, "train_loss": float(np.mean(jnp.stack(ls))), "train_mse": float(np.mean(jnp.stack(ms))),
               "val_mse": v_mse, "val_nll": v_nll, "epoch_s": time.time() - t}
        if v_mse < best:
            best, best_epoch = v_mse, epoch
            save_policy(ema, cfg, run_dir / "best")
        if c["closed_loop_every"] and epoch % c["closed_loop_every"] == 0:
            r, _ = closed_loop(save_policy(ema, cfg, run_dir / "epochs" / f"{epoch:04d}"), val_specs,
                               c["closed_loop_workers"], latency)
            row.update({f"cl_{k}": v for k, v in r.items() if k in (*OUTCOMES, "t_grasp_median")})
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "log.csv", index=False)
        print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if epoch - best_epoch >= c["patience"]:
            reason = f"no validation improvement in {c['patience']} epochs (best epoch {best_epoch})"
            break
    r, cl_rows = closed_loop(run_dir / "best", val_specs, c["closed_loop_workers"], latency)
    (run_dir / "best" / "closed_loop.json").write_text(json.dumps({"epoch": best_epoch, "val_mse": best, **r,
                                                                   "rows": cl_rows}, indent=1, default=str))
    info.update(stop_reason=reason, best_epoch=best_epoch, best_val_mse=best, best_closed_loop=r,
                wall_min=(time.time() - t_start) / 60)
    (run_dir / "run.json").write_text(json.dumps(info, indent=1))
    print(f"best epoch {best_epoch}: val_mse {best:.5g}, validation scenes {r}", flush=True)
    return reason
