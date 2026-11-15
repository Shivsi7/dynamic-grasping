"""(D) fine-tuning diagnostic: a run's critics against Monte-Carlo returns, on demonstration rows and, when its buffer
holds them (ft.calql runs), on its online rows.

  python scripts/rl_diag_q.py DEMOS.npz RUN_DIR [RUN_DIR ...]
"""
import pickle
import sys
from pathlib import Path

import jax
import numpy as np

from dyngrasp.rl.agent import Agent
from dyngrasp.rl.config import load_config
from dyngrasp.rl.finetune import load_demos


def report(name, agent, d, rng):
    idx = rng.choice(len(d["rew"]), min(20000, len(d["rew"])), replace=False)
    obs, act, mc = d["obs"][idx], d["act"][idx], d["mc"][idx]
    q = np.asarray(agent.critic.apply(agent.state["critic"], obs, act)).mean(0)
    a_pi = agent.act(obs, jax.random.PRNGKey(0), deterministic=True)
    q_pi = np.asarray(agent.critic.apply(agent.state["critic"], obs, a_pi)).mean(0)
    print(f"  {name}: n {len(idx)} mc mean {mc.mean():.3f} sd {mc.std():.3f} | Q(s,a) mean {q.mean():.3f} corr {np.corrcoef(q, mc)[0, 1]:.3f} "
          f"bias {np.mean(q - mc):.3f} rmse {np.sqrt(np.mean((q - mc) ** 2)):.3f} | Q(s,pi) - Q(s,a) {np.mean(q_pi - q):.3f}", flush=True)
    for lo, hi in ((-9, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 9)):
        m = (mc >= lo) & (mc < hi)
        if m.any():
            print(f"    mc in [{lo},{hi}): {m.mean():.2f} of rows, Q mean {q[m].mean():.3f}, mc mean {mc[m].mean():.3f}")


rng = np.random.default_rng(0)
for run in sys.argv[2:]:
    run = Path(run)
    cfg = load_config(run / "config.yaml")
    agent = Agent(cfg, pickle.loads((run / "ckpt" / "agent.pkl").read_bytes())["actor"])
    agent.load_state_numpy(pickle.loads((run / "ckpt" / "agent.pkl").read_bytes()))
    print(f"== {run.name} ({int(agent.state['n_updates'])} updates, k_p {cfg['reward']['k_p']})", flush=True)
    report("demos", agent, load_demos(sys.argv[1], 1, cfg["gamma"]), rng)
    z = np.load(run / "ckpt" / "buffer.npz")
    if "mc" in z and np.any(z["mc"] != 0):
        report("online", agent, {k: z[k] for k in ("obs", "act", "mc", "rew")}, rng)
