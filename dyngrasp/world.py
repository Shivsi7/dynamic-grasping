"""WorldModel: the shared "predicted state" object.

The core comparison uses ground-truth poses, so prediction is exact: the world
model evaluates the scenario's scripted paths at any time, past or future.  The
controllers ask it where the target and the obstacles *will be*; the harness
measures where they actually are.

Everything is expressed in the world frame W.  See objects.py for the frames
U (upright object) and G (gripper).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydrake.all import RigidTransform

from dyngrasp.objects import ObjectInfo, get_object
from dyngrasp.paths import Path
from dyngrasp.scenario import ObstacleSpec, ScenarioSpec
from dyngrasp.scene import Scene, set_scripted_positions, target_upright_pose


@dataclass
class ObstacleState:
    name: str
    shape: str  # "sphere" | "box"
    dims: np.ndarray  # sphere: [r]; box: full extents (axis-aligned, never rotates)
    p: np.ndarray  # centre position
    v: np.ndarray  # centre velocity
    radius: float  # bounding-sphere radius (conservative for boxes)


@dataclass
class WorldState:
    """Snapshot of the (predicted) world at time t."""

    t: float
    X_WU: RigidTransform  # upright-object frame
    v_target: np.ndarray  # translational velocity of the target (= cart)
    X_WG_grasp: RigidTransform  # nominal top-down grasp pose for the target
    obstacles: list[ObstacleState]
    target_reachable: bool


class WorldModel:
    def __init__(self, spec: ScenarioSpec):
        self.spec = spec
        self.object: ObjectInfo = get_object(spec.target.object)
        self.cart_path: Path = spec.cart.path
        self._obstacles: list[tuple[ObstacleSpec, Path]] = [(o, o.get_path()) for o in spec.obstacles]

    # ---- timing -------------------------------------------------------------
    @property
    def t_enter(self) -> float:
        """When the target comes within reach (|y| < REACH_Y)."""
        return self.spec.cart.t_enter

    @property
    def t_exit(self) -> float:
        return self.spec.cart.t_exit

    @property
    def t_closest(self) -> float:
        """When the cart passes y = 0, closest to the robot base."""
        return self.spec.cart.t_closest

    @property
    def duration(self) -> float:
        return self.spec.duration

    def target_reachable(self, t: float) -> bool:
        return self.t_enter <= t <= self.t_exit

    # ---- target -------------------------------------------------------------
    def cart_pose(self, t: float) -> RigidTransform:
        """Frame C: centre of the cart's top surface."""
        return RigidTransform(self.cart_path.position(t))

    def target_pose(self, t: float) -> RigidTransform:
        """X_WU(t): upright-object frame (footprint centre on the cart surface, z up)."""
        return target_upright_pose(self.spec, t)

    def target_body_pose(self, t: float) -> RigidTransform:
        """X_WO(t): the object's own body frame, as the plant reports it."""
        return self.target_pose(t) @ self.object.X_UO

    def target_centre(self, t: float) -> np.ndarray:
        return self.target_pose(t) @ self.object.p_UO_centre

    def target_velocity(self, t: float) -> np.ndarray:
        return self.cart_path.velocity(t)

    def grasp_pose(self, t: float, X_UG: RigidTransform | None = None) -> RigidTransform:
        """Gripper pose that grasps the target at time t (default: nominal top-down)."""
        return self.target_pose(t) @ (X_UG if X_UG is not None else self.object.X_UG_grasp)

    # ---- obstacles ------------------------------------------------------------
    @property
    def obstacle_specs(self) -> list[ObstacleSpec]:
        return [o for o, _ in self._obstacles]

    def obstacle_states(self, t: float) -> list[ObstacleState]:
        return [
            ObstacleState(
                name=o.name,
                shape=o.shape,
                dims=np.asarray(o.dims, float),
                p=path.position(t),
                v=path.velocity(t),
                radius=o.bounding_radius,
            )
            for o, path in self._obstacles
        ]

    def obstacle_positions(self, t: float) -> np.ndarray:
        """(N, 3) obstacle centres at time t."""
        return np.array([path.position(t) for _, path in self._obstacles]).reshape(-1, 3)

    def swept_capsules(self, t0: float, t1: float) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """(p(t0), p(t1), radius) per obstacle: the swept volume over [t0, t1] as a capsule.

        For a linear path this is exact up to the bounding sphere; for a parabolic
        path it is the chord, which is adequate for short horizons.
        """
        return [(path.position(t0), path.position(t1), o.bounding_radius) for o, path in self._obstacles]

    # ---- snapshot ---------------------------------------------------------------
    def snapshot(self, t: float) -> WorldState:
        return WorldState(
            t=t,
            X_WU=self.target_pose(t),
            v_target=self.target_velocity(t),
            X_WG_grasp=self.grasp_pose(t),
            obstacles=self.obstacle_states(t),
            target_reachable=self.target_reachable(t),
        )

    # ---- kinematic plant support ------------------------------------------------
    def set_kinematic_state(self, kin: Scene, root_context, t: float, q: np.ndarray | None = None) -> None:
        """Pose a kinematic diagram (scene.build_kinematic_diagram) at time t.

        Sets the cart, the obstacles and the target to their scripted poses and,
        if given, the iiwa to q.  Collision / distance queries on its SceneGraph
        then answer "would configuration q collide at time t".
        """
        ctx = kin.plant.GetMyMutableContextFromRoot(root_context)
        set_scripted_positions(kin, ctx, t)
        kin.plant.SetFreeBodyPose(ctx, kin.target_body, self.target_body_pose(t))
        if q is not None:
            kin.plant.SetPositions(ctx, kin.iiwa, np.asarray(q, float))
