"""(D) training and validation scenes: curriculum levels built from the existing generators.

Levels (easy -> hard; `make_scene(level, seed)`):
  L0  no obstacles, soup can, slow cart (0.05-0.10 m/s)
  L1  no obstacles, any object, any cart speed (0.05-0.30 m/s)
  L2  1-2 slow obstacles (0.10-0.25 m/s) on rigs
  L3  1-4 obstacles at any speed (0.10-0.50 m/s), a quarter of the scenes free-flying ("float")
  L4  blockers: tall post 40 %, plank 40 %, pillar + post 20 %, continuous random parameters
  L5  the full mix: 60 % like the held-out generator (0-4 obstacles, 25 % float, any object), 40 % L4

Disjointness from the 214 evaluation scenes (dev 20, held-out mix 80, pillar 18, plank 48, tall 48):
* seeds: the random evaluation sets use seeds 0-19 (dev_v0) and 5000-5079 (heldout_mixed_s5000);
  validation uses VAL_SEED + i (i < 50) and training draws seeds in [TRAIN_SEED_MIN, 2**31);
* content: `sample_training_scene` rejects any scene whose `fingerprint` (the spec's content, without
  names, seeds and colours) is one of the evaluation or validation scenes'.  The preset grids have no
  seed; their training versions draw speeds and crossing times from continuous ranges, so a repeat is
  impossible in practice, and the fingerprint check makes it impossible by construction.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import numpy as np

from dyngrasp import scenario as S
from dyngrasp.objects import OBJECTS
from dyngrasp.scenario import ScenarioSpec

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "scenarios"
EVAL_SETS = ("dev_v0.yaml", "heldout_mixed_s5000.yaml", "preset_grids.yaml")
TRAIN_SEED_MIN = 10_000_000
VAL_SEED = 9_000_000
N_VALIDATION = 50
VALIDATION_SET = "rl_validation.yaml"
ALL_OBJECTS = tuple(OBJECTS)


def _random(rng, n_obstacles, objects=ALL_OBJECTS, target_speed=None, obstacle_speed_range=S.OBSTACLE_SPEED_RANGE,
            mount="cart") -> ScenarioSpec:
    return S.sample_scenario(rng, n_obstacles=n_obstacles, target_speed=target_speed,
                             obstacle_speed_range=obstacle_speed_range, objects=objects, mount=mount)


def _blocker(rng) -> ScenarioSpec:
    kind = str(rng.choice(["tall", "plank", "pillar"], p=[0.4, 0.4, 0.2]))
    obj = str(rng.choice(ALL_OBJECTS))
    if kind == "pillar":  # evaluation grid: target 0.05-0.15, post 0.2/0.4, pillar 0.05, offset 0.05-0.12
        v, o, p, off = rng.uniform(0.05, 0.15), rng.uniform(0.15, 0.45), rng.uniform(0.03, 0.07), rng.uniform(0.05, 0.13)
        return S.pillar_and_blocker(float(v), float(o), float(p), float(off), obj)
    preset = S.tall_blocker if kind == "tall" else S.plank_blocker  # grid: 0.05-0.3 x 0.1-0.5 x t_enter + 0.5-2
    v, o, dt = rng.uniform(0.05, 0.30), rng.uniform(0.10, 0.50), rng.uniform(0.3, 2.5)
    return preset(float(v), float(o), float(S.WARMUP + dt), obj)


def _mixed(rng) -> ScenarioSpec:
    """Like scripts/stress_planner.mixed_set (the held-out generator), with mount and count drawn at random."""
    n = int(rng.integers(0, 5))
    return _random(rng, n, mount="float" if rng.random() < 0.25 else "cart")


LEVELS = (
    lambda rng: _random(rng, 0, ("soup_can",), target_speed=float(rng.uniform(0.05, 0.10))),
    lambda rng: _random(rng, 0),
    lambda rng: _random(rng, int(rng.integers(1, 3)), obstacle_speed_range=(0.10, 0.25)),
    lambda rng: _random(rng, int(rng.integers(1, 5)), mount="float" if rng.random() < 0.25 else "cart"),
    _blocker,
    lambda rng: _mixed(rng) if rng.random() < 0.6 else _blocker(rng),
)
N_LEVELS = len(LEVELS)


def make_scene(level: int, seed: int) -> ScenarioSpec:
    """The level's scene for this seed (deterministic)."""
    spec = LEVELS[level](np.random.default_rng(seed))
    spec.name, spec.seed = f"L{level}_s{seed}", int(seed)
    return spec


def fingerprint(spec: ScenarioSpec) -> str:
    """Hash of what the scene is (names, seeds, notes and colours left out; floats to 1e-6)."""
    d = spec.to_dict()
    for k in ("name", "seed", "notes"):
        d.pop(k)
    for o in d["obstacles"]:
        o.pop("name")
        o.pop("color")
    return hashlib.sha1(json.dumps(_rounded(d), sort_keys=True).encode()).hexdigest()


def _rounded(x):
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rounded(v) for v in x]
    return round(x, 6) if isinstance(x, float) else x


def eval_scenes() -> list[ScenarioSpec]:
    """The 214 evaluation scenes."""
    return [s for f in EVAL_SETS for s in S.load_scenario_set(SCENARIO_DIR / f)]


def validation_scenes() -> list[ScenarioSpec]:
    """The 50 validation scenes (scenarios/rl_validation.yaml, written by `validation_scene`)."""
    return S.load_scenario_set(SCENARIO_DIR / VALIDATION_SET)


def validation_scene(i: int) -> ScenarioSpec:
    """Validation scene i: the final level's distribution, seed VAL_SEED + i."""
    seed = VAL_SEED + i
    while True:  # the generator gives up on rare seeds; skip those deterministically
        try:
            spec = make_scene(N_LEVELS - 1, seed)
            spec.name = f"val_{i:03d}"
            return spec
        except RuntimeError:
            seed += 1_000


@functools.cache
def reserved_fingerprints() -> frozenset[str]:
    """Scenes training must never use: the evaluation scenes and the validation scenes."""
    return frozenset(fingerprint(s) for s in [*eval_scenes(), *validation_scenes()])


def sample_training_scene(level: int, rng: np.random.Generator) -> ScenarioSpec:
    """A fresh scene of `level`, with a seed from the training range, never an evaluation or validation scene."""
    reserved = reserved_fingerprints()
    while True:
        seed = int(rng.integers(TRAIN_SEED_MIN, 2**31 - 1))
        try:
            spec = make_scene(level, seed)
        except RuntimeError:  # no admissible obstacle paths for this seed
            continue
        if fingerprint(spec) not in reserved:
            return spec


def estimate_start_error(n: int = 200, level: int = N_LEVELS - 1) -> float:
    """Mean distance at t = 0 from the gripper (at q0) to the grasp pose, over n training scenes (for k_p)."""
    from dyngrasp.controller import RobotModel
    from dyngrasp.world import WorldModel

    robot = RobotModel()
    errs = [
        np.linalg.norm(WorldModel(spec).grasp_pose(0.0).translation() - robot.fk(spec.q0).translation())
        for spec in (make_scene(level, TRAIN_SEED_MIN + i) for i in range(n))
    ]
    return float(np.mean(errs))
