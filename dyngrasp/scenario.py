"""Scenario specification: what the world looks like and how it moves.

A `ScenarioSpec` is a plain, serialisable description (YAML) of one episode:
the robot's start configuration, the cart and its object, and the obstacles
with their scripted paths.  `sample_scenario` draws a random one from a seeded
generator so that every controller can be run on *identical* scenario sets.

Workspace layout (world frame, metres)
--------------------------------------
* iiwa base at the origin, home configuration faces +x.
* The cart runs along y in a lane at x = `CartSpec.x` (~0.60), its top surface
  at z = `CartSpec.top_z` (0.20).  It starts out of reach and enters the robot's
  reach (|y| < REACH_Y) at t = WARMUP seconds whatever its speed.
* Obstacles cross the corridor between the robot base and the lane while the
  target is reachable, through a crossing point in CROSSING_BOX, moving mostly
  along y.  A sampled path is rejected if it would hit the cart (and the column
  above it where the object rides), the fixed robot base, or the floor: the
  scenario must be solvable by a robot that gets out of the way.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path as FilePath
from typing import Any

import numpy as np
import yaml

from dyngrasp.objects import OBJECTS
from dyngrasp.paths import LinearPath, ParabolicPath, Path

# Joint configuration the robot starts from (gripper above the base, facing +x).
Q_HOME = np.array([0.0, 0.1, 0.0, -1.2, 0.0, 1.6, 0.0])

# The target is "reachable" while the cart is within this |y| of the base.
REACH_Y = 0.5
# The cart reaches |y| = REACH_Y this many seconds after the start.
WARMUP = 3.0
# Where obstacle paths cross the corridor (x, y, z bounds).
CROSSING_BOX = np.array([[0.20, 0.42], [-0.30, 0.30], [0.35, 0.70]])
# Half-length of an obstacle's path on each side of the crossing point.
PATH_HALF_LENGTH = 1.0
# Obstacle direction: angle from the y axis, in the horizontal plane.
MAX_HEADING_DEG = 25.0
# Keep-out volumes for obstacle paths (see path_is_admissible).
BASE_RADIUS = 0.16  # cylinder around the z axis ...
BASE_HEIGHT = 0.40  # ... up to this height (iiwa links 0-1 cannot move away)
OBJECT_COLUMN = 0.45  # height above the cart top reserved for the object + grasp
LANE_X = (0.56, 0.62)

TARGET_SPEED_RANGE = (0.05, 0.30)
OBSTACLE_SPEED_RANGE = (0.10, 0.50)
SPHERE_RADIUS_RANGE = (0.05, 0.10)
BOX_HALF_RANGE = (0.05, 0.10)

OBSTACLE_COLORS = [
    (0.85, 0.15, 0.15, 1.0),
    (0.95, 0.55, 0.10, 1.0),
    (0.60, 0.15, 0.60, 1.0),
    (0.15, 0.55, 0.85, 1.0),
]


@dataclass
class TargetSpec:
    object: str = "soup_can"
    yaw: float = 0.0  # rotation of the object about z on the cart

    def __post_init__(self):
        if self.object not in OBJECTS:
            raise ValueError(f"unknown object {self.object!r}; known: {sorted(OBJECTS)}")


@dataclass
class CartSpec:
    x: float = 0.60
    y_start: float = -1.0
    y_end: float = 1.0
    speed: float = 0.1
    t_start: float = 0.0
    top_z: float = 0.20
    size: tuple[float, float, float] = (0.30, 0.50, 0.20)  # box below the top surface (x, y, z)

    @property
    def path(self) -> LinearPath:
        """Path of the cart-top centre C (constant velocity along y)."""
        dur = abs(self.y_end - self.y_start) / self.speed
        return LinearPath(
            [self.x, self.y_start, self.top_z],
            [self.x, self.y_end, self.top_z],
            self.t_start,
            self.t_start + dur,
        )

    @property
    def direction(self) -> float:
        return 1.0 if self.y_end > self.y_start else -1.0

    def time_at_y(self, y: float) -> float:
        return self.path.time_at(1, y)

    @property
    def t_enter(self) -> float:
        return self.time_at_y(-self.direction * REACH_Y)

    @property
    def t_exit(self) -> float:
        return self.time_at_y(self.direction * REACH_Y)

    @property
    def t_closest(self) -> float:
        return self.time_at_y(0.0)


@dataclass
class ObstacleSpec:
    name: str
    shape: str  # "sphere" | "box"
    dims: list[float]  # sphere: [radius]; box: [sx, sy, sz] (full extents)
    path: dict[str, Any]  # serialised Path (see paths.Path.from_dict)
    color: tuple[float, float, float, float] = (0.85, 0.15, 0.15, 1.0)

    def get_path(self) -> Path:
        return Path.from_dict(self.path)

    @property
    def bounding_radius(self) -> float:
        if self.shape == "sphere":
            return float(self.dims[0])
        return float(0.5 * np.linalg.norm(self.dims))


@dataclass
class ScenarioSpec:
    name: str = "scenario"
    seed: int | None = None
    q0: list[float] = field(default_factory=lambda: Q_HOME.tolist())
    target: TargetSpec = field(default_factory=TargetSpec)
    cart: CartSpec = field(default_factory=CartSpec)
    obstacles: list[ObstacleSpec] = field(default_factory=list)
    duration: float = 10.0  # episode time limit (s)
    time_step: float = 0.001  # plant time step (s)
    notes: str = ""

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cart"]["size"] = list(self.cart.size)
        for o in d["obstacles"]:
            o["color"] = list(o["color"])
        return _plain(d)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ScenarioSpec":
        d = copy.deepcopy(d)
        d["target"] = TargetSpec(**d.get("target", {}))
        cart = d.get("cart", {})
        if "size" in cart:
            cart["size"] = tuple(cart["size"])
        d["cart"] = CartSpec(**cart)
        obs = []
        for o in d.get("obstacles", []):
            o["color"] = tuple(o.get("color", OBSTACLE_COLORS[0]))
            obs.append(ObstacleSpec(**o))
        d["obstacles"] = obs
        return ScenarioSpec(**d)

    def save(self, path: str | FilePath) -> None:
        FilePath(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @staticmethod
    def load(path: str | FilePath) -> "ScenarioSpec":
        return ScenarioSpec.from_dict(yaml.safe_load(FilePath(path).read_text()))

    # ---- summary numbers used as columns in the results table ------------
    def summary(self) -> dict[str, Any]:
        speeds = [_path_speed(o.get_path()) for o in self.obstacles]
        return {
            "scenario": self.name,
            "seed": self.seed,
            "object": self.target.object,
            "target_speed": self.cart.speed,
            "n_obstacles": len(self.obstacles),
            "obstacle_speed_mean": float(np.mean(speeds)) if speeds else 0.0,
            "obstacle_speed_max": float(np.max(speeds)) if speeds else 0.0,
            "duration": self.duration,
        }


def _plain(x):
    """numpy scalars/arrays -> Python floats/lists, so YAML can represent them."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def _path_speed(p: Path) -> float:
    if isinstance(p, LinearPath):
        return p.speed
    if isinstance(p, ParabolicPath):
        return float(np.linalg.norm(p.v0))
    return 0.0


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def make_cart(speed: float, direction: float = 1.0, x: float = 0.60, top_z: float = 0.20) -> CartSpec:
    """A cart that enters reach at t = WARMUP and leaves the lane at |y| = 1.0 or later."""
    y_start = -direction * (REACH_Y + WARMUP * speed)
    y_end = direction * max(1.0, REACH_Y + 1.0 * speed + 0.2)
    return CartSpec(x=x, y_start=y_start, y_end=y_end, speed=speed, top_z=top_z)


def episode_duration(cart: CartSpec, margin: float = 1.5) -> float:
    return float(np.clip(cart.t_exit + margin, 5.0, 45.0))


def linear_crossing(
    p_cross, heading_deg: float, speed: float, t_cross: float, direction: float = 1.0
) -> dict:
    """Straight path through p_cross at t_cross, moving mostly along +-y."""
    th = np.deg2rad(heading_deg)
    d = np.array([np.sin(th), np.cos(th), 0.0]) * direction
    p_cross = np.asarray(p_cross, float)
    L = PATH_HALF_LENGTH
    return LinearPath(
        p_cross - d * L, p_cross + d * L, t_cross - L / speed, t_cross + L / speed
    ).to_dict()


def parabolic_crossing(
    p_cross, heading_deg: float, speed: float, t_cross: float, direction: float = 1.0
) -> dict:
    """A lob: apex at p_cross at t_cross, ends near the floor on both sides."""
    th = np.deg2rad(heading_deg)
    d = np.array([np.sin(th), np.cos(th), 0.0]) * direction
    p_cross = np.asarray(p_cross, float)
    L = PATH_HALF_LENGTH
    tau = L / speed
    g_eff = 2.0 * (p_cross[2] - 0.05) / tau**2
    return ParabolicPath(
        p_cross, d * speed, [0.0, 0.0, -g_eff], t_cross, t_cross - tau, t_cross + tau
    ).to_dict()


def make_obstacle(
    name: str,
    shape: str,
    dims,
    p_cross,
    heading_deg: float,
    speed: float,
    t_cross: float,
    direction: float = 1.0,
    kind: str = "linear",
    color=None,
) -> ObstacleSpec:
    f = linear_crossing if kind == "linear" else parabolic_crossing
    return ObstacleSpec(
        name=name,
        shape=shape,
        dims=[float(x) for x in np.atleast_1d(dims)],
        path=f(p_cross, heading_deg, speed, t_cross, direction),
        color=color or OBSTACLE_COLORS[0],
    )


def path_is_admissible(path: Path, radius: float, cart: CartSpec, n_samples: int = 200) -> bool:
    """False if the obstacle's bounding sphere would hit the cart column, the base or the floor."""
    t0 = path.t_start if np.isfinite(path.t_start) else 0.0
    t1 = path.t_end if np.isfinite(path.t_end) else t0 + 1.0
    sx, sy, _ = cart.size
    for t in np.linspace(t0, t1, n_samples):
        p = path.position(t)
        if p[2] - radius < 0.02:
            return False  # floor
        if np.hypot(p[0], p[1]) - radius < BASE_RADIUS and p[2] - radius < BASE_HEIGHT:
            return False  # fixed base
        c = cart.path.position(t)
        dx = max(abs(p[0] - c[0]) - sx / 2, 0.0)
        dy = max(abs(p[1] - c[1]) - sy / 2, 0.0)
        dz = max((c[2] - cart.size[2]) - p[2], p[2] - (c[2] + OBJECT_COLUMN), 0.0)
        if np.sqrt(dx**2 + dy**2 + dz**2) < radius:
            return False  # cart (and the object column above it)
    return True


# ---------------------------------------------------------------------------
# Random scenarios
# ---------------------------------------------------------------------------


def sample_scenario(
    rng: np.random.Generator,
    n_obstacles: int | None = None,
    target_speed: float | None = None,
    obstacle_speed_range: tuple[float, float] = OBSTACLE_SPEED_RANGE,
    objects: tuple[str, ...] = ("soup_can",),
    parabolic_fraction: float = 0.3,
    name: str = "random",
    seed: int | None = None,
) -> ScenarioSpec:
    """One random scenario.  Obstacles cross while the target is reachable."""
    if target_speed is None:
        target_speed = float(rng.uniform(*TARGET_SPEED_RANGE))
    if n_obstacles is None:
        n_obstacles = int(rng.integers(1, 5))
    direction = float(rng.choice([-1.0, 1.0]))
    cart = make_cart(target_speed, direction, x=float(rng.uniform(*LANE_X)))
    target = TargetSpec(object=str(rng.choice(list(objects))), yaw=float(rng.uniform(-0.3, 0.3)))

    obstacles = []
    for i in range(n_obstacles):
        for _attempt in range(200):
            shape = "sphere" if rng.random() < 0.5 else "box"
            if shape == "sphere":
                dims = [float(rng.uniform(*SPHERE_RADIUS_RANGE))]
            else:
                dims = list(2 * rng.uniform(*BOX_HALF_RANGE, size=3))
            p_cross = rng.uniform(CROSSING_BOX[:, 0], CROSSING_BOX[:, 1])
            heading = float(rng.uniform(-MAX_HEADING_DEG, MAX_HEADING_DEG))
            speed = float(rng.uniform(*obstacle_speed_range))
            t_cross = float(rng.uniform(cart.t_enter, cart.t_exit))
            kind = "parabolic" if rng.random() < parabolic_fraction else "linear"
            o = make_obstacle(
                f"obstacle_{i}",
                shape,
                dims,
                p_cross,
                heading,
                speed,
                t_cross,
                float(rng.choice([-1.0, 1.0])),
                kind,
                OBSTACLE_COLORS[i % len(OBSTACLE_COLORS)],
            )
            if path_is_admissible(o.get_path(), o.bounding_radius, cart):
                obstacles.append(o)
                break
        else:
            raise RuntimeError("could not sample an admissible obstacle path")
    return ScenarioSpec(
        name=name,
        seed=seed,
        target=target,
        cart=cart,
        obstacles=obstacles,
        duration=episode_duration(cart),
    )


def generate_scenario_set(seed: int, n: int, prefix: str = "rand", **kwargs) -> list[ScenarioSpec]:
    """`n` reproducible scenarios; scenario i uses seed `seed + i`."""
    specs = []
    for i in range(n):
        s = seed + i
        specs.append(sample_scenario(np.random.default_rng(s), name=f"{prefix}_{i:03d}", seed=s, **kwargs))
    return specs


def save_scenario_set(specs: list[ScenarioSpec], path: str | FilePath) -> None:
    FilePath(path).write_text(yaml.safe_dump([s.to_dict() for s in specs], sort_keys=False))


def load_scenario_set(path: str | FilePath) -> list[ScenarioSpec]:
    return [ScenarioSpec.from_dict(d) for d in yaml.safe_load(FilePath(path).read_text())]


# ---------------------------------------------------------------------------
# Named presets
# ---------------------------------------------------------------------------


def no_obstacles(target_speed: float = 0.1, object: str = "soup_can", direction: float = 1.0) -> ScenarioSpec:
    cart = make_cart(target_speed, direction)
    return ScenarioSpec(
        name=f"no_obstacles_v{target_speed:.2f}",
        target=TargetSpec(object=object),
        cart=cart,
        duration=episode_duration(cart),
    )


def _blocker_scenario(name, shape_dims, x, speed, t_cross, target_speed, object, color=None):
    """A box standing on the floor, crossing the corridor along -y through x."""
    cart = make_cart(target_speed, 1.0)
    if t_cross is None:
        # One second after the target becomes reachable: in the way of a direct grasp.
        t_cross = cart.t_enter + 1.0
    blocker = make_obstacle(
        "blocker", "box", shape_dims, [x, 0.0, 0.5 * shape_dims[2] + 0.01], 0.0, speed, t_cross, -1.0, "linear",
        color or OBSTACLE_COLORS[0],
    )
    return ScenarioSpec(
        name=f"{name}_v{target_speed:.2f}_o{speed:.2f}_t{t_cross:.1f}",
        target=TargetSpec(object=object),
        cart=cart,
        obstacles=[blocker],
        duration=episode_duration(cart),
    )


def tall_blocker(
    target_speed: float = 0.1,
    obstacle_speed: float = 0.2,
    t_cross: float | None = None,
    object: str = "soup_can",
) -> ScenarioSpec:
    """The initial test from the proposal: a tall post (0.12 x 0.12 x 0.8 m, standing
    on the floor) sliding across the corridor between the robot and the cart.

    By default it crosses the robot's line to the cart one second after the target
    becomes reachable, i.e. exactly when a direct grasp would be happening, so a
    controller that goes straight for the object hits it.  It is thin, so the arm
    can reach around it on either side (or over it, top at z = 0.81 m): a
    collision-free grasp configuration exists wherever it stands.
    """
    return _blocker_scenario("tall_blocker", [0.12, 0.12, 0.80], 0.28, obstacle_speed, t_cross, target_speed, object)


def plank_blocker(
    target_speed: float = 0.1,
    obstacle_speed: float = 0.2,
    t_cross: float | None = None,
    object: str = "soup_can",
) -> ScenarioSpec:
    """Like `tall_blocker` but a thin plank: 0.20 m wide (x), 0.04 m thick, 0.9 m tall.

    Still passable on either side or over the top; it just sweeps more of the
    corridor, so the detour has to be larger.
    """
    return _blocker_scenario(
        "plank_blocker", [0.20, 0.04, 0.90], 0.30, obstacle_speed, t_cross, target_speed, object, OBSTACLE_COLORS[2]
    )
