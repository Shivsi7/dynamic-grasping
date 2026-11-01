"""Scripted motion primitives for the cart and the obstacles.

The simulation drives bodies along these paths (through PD-controlled prismatic
joints, see scene.py) and the controllers "predict" with the very same objects
(see world.py).  With ground-truth state the prediction is therefore exact up to
the small tracking error of the joint controllers; the RGB-D extension would
replace `WorldModel` with an estimator that fits one of these to observations.

Every path is parked (zero velocity) outside its active interval, so an obstacle
waits at its start point and stays at its end point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(3)


@dataclass
class Path:
    """Base class: position/velocity of a point as a function of time."""

    def position(self, t: float) -> np.ndarray:
        raise NotImplementedError

    def velocity(self, t: float) -> np.ndarray:
        raise NotImplementedError

    @property
    def t_start(self) -> float:
        return -np.inf

    @property
    def t_end(self) -> float:
        return np.inf

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Path":
        kind = d["type"]
        if kind == "stationary":
            return StationaryPath(_arr(d["p"]))
        if kind == "linear":
            return LinearPath(_arr(d["p_start"]), _arr(d["p_end"]), d["t_start"], d["t_end"])
        if kind == "parabolic":
            return ParabolicPath(
                _arr(d["p0"]), _arr(d["v0"]), _arr(d["accel"]), d["t0"], d["t_start"], d["t_end"]
            )
        raise ValueError(f"unknown path type {kind!r}")


@dataclass
class StationaryPath(Path):
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.p = _arr(self.p)

    def position(self, t):
        return self.p.copy()

    def velocity(self, t):
        return np.zeros(3)

    def to_dict(self):
        return {"type": "stationary", "p": self.p.tolist()}


@dataclass
class LinearPath(Path):
    """Constant velocity from p_start (at t_start) to p_end (at t_end)."""

    p_start: np.ndarray
    p_end: np.ndarray
    t_start_: float
    t_end_: float

    def __post_init__(self):
        self.p_start = _arr(self.p_start)
        self.p_end = _arr(self.p_end)
        assert self.t_end_ > self.t_start_

    @property
    def t_start(self):
        return self.t_start_

    @property
    def t_end(self):
        return self.t_end_

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.p_end - self.p_start) / (self.t_end_ - self.t_start_))

    @property
    def direction(self) -> np.ndarray:
        d = self.p_end - self.p_start
        n = np.linalg.norm(d)
        return d / n if n > 0 else d

    def position(self, t):
        s = np.clip((t - self.t_start_) / (self.t_end_ - self.t_start_), 0.0, 1.0)
        return self.p_start + s * (self.p_end - self.p_start)

    def velocity(self, t):
        if self.t_start_ <= t < self.t_end_:
            return (self.p_end - self.p_start) / (self.t_end_ - self.t_start_)
        return np.zeros(3)

    def time_at(self, axis: int, value: float) -> float:
        """Time at which coordinate `axis` equals `value` (inf if never)."""
        v = self.velocity(0.5 * (self.t_start_ + self.t_end_))[axis]
        if abs(v) < 1e-12:
            return np.inf
        t = self.t_start_ + (value - self.p_start[axis]) / v
        return t if self.t_start_ <= t <= self.t_end_ else np.inf

    def to_dict(self):
        return {
            "type": "linear",
            "p_start": self.p_start.tolist(),
            "p_end": self.p_end.tolist(),
            "t_start": float(self.t_start_),
            "t_end": float(self.t_end_),
        }


@dataclass
class ParabolicPath(Path):
    """p(t) = p0 + v0 (t - t0) + 1/2 accel (t - t0)^2, active on [t_start, t_end]."""

    p0: np.ndarray
    v0: np.ndarray
    accel: np.ndarray
    t0: float
    t_start_: float
    t_end_: float

    def __post_init__(self):
        self.p0 = _arr(self.p0)
        self.v0 = _arr(self.v0)
        self.accel = _arr(self.accel)
        assert self.t_end_ > self.t_start_

    @property
    def t_start(self):
        return self.t_start_

    @property
    def t_end(self):
        return self.t_end_

    def position(self, t):
        tau = np.clip(t, self.t_start_, self.t_end_) - self.t0
        return self.p0 + self.v0 * tau + 0.5 * self.accel * tau**2

    def velocity(self, t):
        if self.t_start_ <= t < self.t_end_:
            return self.v0 + self.accel * (t - self.t0)
        return np.zeros(3)

    def to_dict(self):
        return {
            "type": "parabolic",
            "p0": self.p0.tolist(),
            "v0": self.v0.tolist(),
            "accel": self.accel.tolist(),
            "t0": float(self.t0),
            "t_start": float(self.t_start_),
            "t_end": float(self.t_end_),
        }


def sample_path(path: Path, times) -> np.ndarray:
    """Positions along `path` at the given times, shape (N, 3)."""
    return np.array([path.position(t) for t in times])
