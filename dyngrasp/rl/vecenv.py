"""(D) asynchronous environments: one GraspEnv per worker process, stepped independently.

The learner sends an action to each environment waiting for one and takes the results as they arrive
(`poll`), so a slow step (an episode reset builds a new simulation, ~0.4 s, vs ~50-150 ms per step) delays
only its own environment, not all of them as in a synchronous vector environment.  An environment resets
itself at the end of an episode.

Workers are local processes (slots 0 .. n_local-1) and, with `remote` > 0, processes on other machines
(scripts/env_workers.py) that connect over TCP to a listener whose address and key the learner writes to
`addr_file` (and its .key); they fill the remote slots as they connect and free them when their connection
drops (preemption), and keep retrying to connect while the run goes on.  Messages from a worker:
("start", obs) when it has started its first episode (after "init"), ("step", (obs to act on, reward, done,
the step's own next observation, episode info if done)).  `poll` also reports a dropped connection as
(slot, "lost", None).  Workers ignore SIGUSR1/SIGTERM (the learner checkpoints on them).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import secrets
import signal
import socket
import threading
import time
from multiprocessing.connection import Client, Listener, wait
from pathlib import Path

import numpy as np

EPISODE_KEYS = ("outcome", "level", "scene", "ep_terms", "ep_len", "closes", "collision")


def serve(conn, max_episodes: int | None = None) -> bool:
    """A worker's loop on one connection, until "close", the connection drops or (True) max_episodes episodes end."""
    from dyngrasp.rl.env import GraspEnv

    env, ret, episodes = None, 0.0, 0
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "init":
                cfg, seed, level = data
                if env is not None:
                    env.close()
                env = GraspEnv(cfg, seed)
                env.set_level(level)
                ret = 0.0
                conn.send(("start", env.reset()[0]))
            elif cmd == "step":
                try:
                    obs, r, term, trunc, info = env.step(data)
                except Exception as exc:  # noqa: BLE001  (a simulation error: drop the episode, start another)
                    print(f"GraspEnv.step failed ({exc!r}): new episode", flush=True)
                    ret = 0.0
                    conn.send(("start", env.reset()[0]))
                    continue
                ret += r
                if term or trunc:
                    ep = {k: info[k] for k in EPISODE_KEYS}
                    ep["return"] = ret
                    ret = 0.0
                    episodes += 1
                    if max_episodes and episodes >= max_episodes:  # (the learner sees the slot drop after this episode)
                        conn.send(("step", (obs, r, True, obs, ep)))
                        return True
                    obs0, _ = env.reset()
                    conn.send(("step", (obs0, r, True, obs, ep)))
                else:
                    conn.send(("step", (obs, r, False, obs, None)))
            elif cmd == "set_level":
                env.set_level(data)
            elif cmd == "close":
                break
    except (EOFError, OSError):
        pass
    finally:
        if env is not None:
            env.close()
    return False


def _ignore_signals() -> None:
    for s in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(s, signal.SIG_IGN)


def _local_worker(conn) -> None:
    _ignore_signals()
    try:
        serve(conn)
    finally:
        conn.close()


def remote_worker(addr_file: str, stop_file: str, max_episodes: int | None = None, ppo: bool = False) -> None:
    """Connect to the learner at the address in addr_file (and its .key), serve, reconnect; until stop_file exists or
    (max_episodes) this process has served that many episodes on one connection (a fresh process then takes over:
    GraspEnv's memory grows with every episode's new simulation).  ppo: serve whole episodes with the learner's
    latest policy (rollout.serve) instead of steps."""
    _ignore_signals()
    fn = serve
    if ppo:
        from dyngrasp.rl.rollout import serve as fn
    while not os.path.exists(stop_file):
        try:
            host, port = Path(addr_file).read_text().split()
            key = Path(addr_file + ".key").read_bytes()
            conn = Client((host, int(port)), authkey=key)
        except (OSError, ValueError, EOFError):
            time.sleep(5.0)
            continue
        try:
            if fn(conn, max_episodes):
                return
        except Exception as exc:  # noqa: BLE001  (the learner sees a dropped connection)
            print(f"worker error: {exc!r}", flush=True)
        finally:
            conn.close()
        time.sleep(1.0)


class AsyncEnvs:
    def __init__(self, cfg: dict, n: int, seed: int, level: int = 0, remote: int = 0, addr_file: str | Path | None = None):
        self.cfg, self.seed, self.level = cfg, seed, level
        self.n_local, self.n = n, n + remote
        self.conns: list = [None] * self.n
        self.busy = np.zeros(self.n, bool)
        self._slot: dict = {}
        self._inits = 0
        self._lost: list[int] = []  # slots whose connection dropped, reported by the next poll
        ctx = mp.get_context("spawn")
        self.procs = []
        for i in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_local_worker, args=(child,), daemon=True)
            p.start()
            child.close()
            self.procs.append(p)
            self._attach(i, parent)
        self._new: queue.Queue = queue.Queue()
        self.listener = None
        if remote:
            key = secrets.token_bytes(32)
            self.listener = Listener(("0.0.0.0", 0), authkey=key)
            addr_file = Path(addr_file)
            tmp = addr_file.with_name(f".{addr_file.name}.tmp")
            (Path(str(tmp) + ".key")).write_bytes(key)
            os.chmod(str(tmp) + ".key", 0o600)
            os.replace(str(tmp) + ".key", str(addr_file) + ".key")
            tmp.write_text(f"{socket.gethostname()} {self.listener.address[1]}\n")
            os.replace(tmp, addr_file)
            threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                self._new.put(self.listener.accept())
            except Exception:  # noqa: BLE001  (closed, or a client that failed the handshake)
                if self.listener is None:
                    return

    def _attach(self, i: int, conn) -> None:
        self.conns[i] = conn
        self._slot[conn] = i
        self._inits += 1
        conn.send(("init", (self.cfg, self.seed * 10_000 + self._inits, self.level)))
        self.busy[i] = True  # its "start" is coming

    @property
    def live(self) -> int:
        return sum(c is not None for c in self.conns)

    def send(self, idx, actions) -> None:
        for i, a in zip(idx, actions):
            assert not self.busy[i] and self.conns[i] is not None
            try:
                self.conns[i].send(("step", np.asarray(a, np.float64)))
                self.busy[i] = True
            except OSError:
                self._drop(i)  # reported as lost by the next poll

    def _drop(self, i: int) -> None:
        c = self.conns[i]
        self.conns[i], self.busy[i] = None, False
        self._slot.pop(c, None)
        self._lost.append(i)
        try:
            c.close()
        except OSError:
            pass

    def poll(self, timeout: float | None = None) -> list[tuple[int, str, object]]:
        """(slot, "start" | "step" | "lost", payload) for what arrived (waits up to `timeout` for the first)."""
        while not self._new.empty():
            c = self._new.get()
            free = [i for i in range(self.n_local, self.n) if self.conns[i] is None]
            if not free:
                c.close()
                continue
            try:
                self._attach(free[0], c)
            except OSError:
                self.conns[free[0]] = None
                self._slot.pop(c, None)
                self.busy[free[0]] = False
        busy = [self.conns[i] for i in np.flatnonzero(self.busy)]
        out = []
        for c in wait(busy, timeout) if busy else []:
            i = self._slot[c]
            try:
                kind, payload = c.recv()
            except (EOFError, OSError):
                self._drop(i)
                continue
            self.busy[i] = False
            out.append((i, kind, payload))
        lost, self._lost = self._lost, []
        return out + [(i, "lost", None) for i in lost]

    def drain(self, timeout: float = 30.0) -> list:
        """Wait for every step in flight (at most `timeout`)."""
        out, t_end = [], time.time() + timeout
        while self.busy.any() and time.time() < t_end:
            out += self.poll(1.0)
        return out

    def set_level(self, level: int) -> None:
        self.level = int(level)
        for i, c in enumerate(self.conns):
            if c is not None:
                try:
                    c.send(("set_level", self.level))
                except OSError:
                    self._drop(i)

    def close(self) -> None:
        listener, self.listener = self.listener, None
        if listener is not None:
            listener.close()
        for c in self.conns:
            if c is not None:
                try:
                    c.send(("close", None))
                except OSError:
                    pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
