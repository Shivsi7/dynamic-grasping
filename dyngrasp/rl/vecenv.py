"""(D) asynchronous environments: one GraspEnv per worker process, stepped independently.

The learner sends an action to each environment waiting for one and takes the results as they arrive
(`poll`), so a slow step (an episode reset builds a new simulation, ~0.4 s, vs ~50-80 ms per step) delays
only its own environment, not all of them as in a synchronous vector environment.  An environment resets
itself at the end of an episode: a step result is (obs to act on, reward, done, the step's own next
observation, episode info if done).  Workers ignore SIGUSR1/SIGTERM (the learner checkpoints on them).
"""

from __future__ import annotations

import multiprocessing as mp
import signal
from multiprocessing.connection import wait

import numpy as np

EPISODE_KEYS = ("outcome", "level", "scene", "ep_terms", "ep_len", "closes")


def _worker(conn, cfg: dict, seed: int, level: int) -> None:
    for s in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(s, signal.SIG_IGN)
    from dyngrasp.rl.env import GraspEnv

    env = GraspEnv(cfg, seed)
    env.set_level(level)
    ret = 0.0
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                obs, r, term, trunc, info = env.step(data)
                ret += r
                if term or trunc:
                    ep = {k: info[k] for k in EPISODE_KEYS}
                    ep["return"] = ret
                    ret = 0.0
                    obs0, _ = env.reset()
                    conn.send((obs0, r, True, obs, ep))
                else:
                    conn.send((obs, r, False, obs, None))
            elif cmd == "reset":
                ret = 0.0
                conn.send(env.reset()[0])
            elif cmd == "set_level":
                env.set_level(data)
            elif cmd == "close":
                break
    finally:
        env.close()
        conn.close()


class AsyncEnvs:
    def __init__(self, cfg: dict, n: int, seed: int, level: int = 0):
        ctx = mp.get_context("spawn")
        self.n = n
        self.conns, self.procs = [], []
        for i in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, cfg, seed * 10_000 + i, level), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self._index = {c: i for i, c in enumerate(self.conns)}
        self.busy = np.zeros(n, bool)

    def reset(self) -> np.ndarray:
        """Start an episode in every environment (none may be stepping); their observations."""
        assert not self.busy.any()
        for c in self.conns:
            c.send(("reset", None))
        return np.stack([c.recv() for c in self.conns])

    def send(self, idx, actions) -> None:
        for i, a in zip(idx, actions):
            assert not self.busy[i]
            self.conns[i].send(("step", np.asarray(a, np.float64)))
            self.busy[i] = True

    def poll(self, timeout: float | None = None) -> list[tuple[int, tuple]]:
        """Results of the steps that finished (waits up to `timeout` for the first one)."""
        busy = [self.conns[i] for i in np.flatnonzero(self.busy)]
        out = []
        for c in wait(busy, timeout):
            i = self._index[c]
            out.append((i, c.recv()))
            self.busy[i] = False
        return out

    def drain(self) -> list[tuple[int, tuple]]:
        """Wait for every step in flight."""
        out = []
        while self.busy.any():
            out += self.poll()
        return out

    def set_level(self, level: int) -> None:
        for c in self.conns:
            c.send(("set_level", int(level)))

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
