"""Robot <-> world signed distances at a hypothetical (q, t), for (A) and (C).

`CollisionWorld` wraps the kinematic diagram (scene.build_kinematic_diagram)
posed by the WorldModel's prediction.  Only robot <-> world pairs are kept
(robot = iiwa links 1-7 and the WSG; world = obstacles, cart, floor, target);
robot self-pairs and world <-> world pairs are filtered out of every context,
so Drake's `MinimumDistanceLowerBoundConstraint` can be used directly on them.

* `rows(q, t)`: per close pair, the signed distance, the normal and the
  Jacobian row n^T J (for the reactive QP's velocity dampers), plus the normal
  velocity of the world point (moving obstacles, the carrier).
* `context(categories)`: a diagram context in which only robot <-> {categories}
  pairs remain (for minimum-distance constraints with per-category bounds).
* `full_state(t)`: the plant's full position vector at time t (cart, obstacles,
  target on their scripts) and its time derivative, for space-time constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydrake.all import (
    BodyIndex,
    CollisionFilterDeclaration,
    GeometrySet,
    JacobianWrtVariable,
    SpatialVelocity,
)

from dyngrasp.controller import NQ, WSG_OPEN
from dyngrasp.scenario import ScenarioSpec
from dyngrasp.scene import BodyRoles, build_kinematic_diagram
from dyngrasp.world import WorldModel


@dataclass
class DistanceRow:
    phi: float  # signed distance robot <-> world geometry
    a: np.ndarray  # n^T J_iiwa (7,): rate of change of phi per unit joint velocity
    vn: float  # n^T v_world: normal velocity of the world witness point (towards the robot > 0)
    category: str  # obstacle | cart | floor | target
    robot_body: str
    other_body: str


class CollisionWorld:
    def __init__(self, spec: ScenarioSpec, world: WorldModel):
        self.world = world
        self.diagram, self.kin = build_kinematic_diagram(spec)
        self.plant = self.kin.plant
        self.scene_graph = self.kin.scene_graph
        self.roles = BodyRoles(self.kin)
        self.inspector = self.scene_graph.model_inspector()
        self.W = self.plant.world_frame()
        self.iiwa_q = self.plant.GetJointByName("iiwa_joint_1", self.kin.iiwa).position_start()
        self.iiwa_v = self.plant.GetJointByName("iiwa_joint_1", self.kin.iiwa).velocity_start()

        robot_bodies = set(self.roles.robot)
        self._geoms = {"robot": [], "base": [], "obstacle": [], "cart": [], "floor": [], "target": []}
        for g, b in self.roles.geom_body.items():
            if b in robot_bodies:
                self._geoms["robot"].append(g)
            elif b in self.roles.category:
                self._geoms[self.roles.category[b]].append(g)
            else:
                self._geoms["base"].append(g)  # iiwa_link_0: fixed, never a collision
        self._ctx_cache: dict[tuple, object] = {}
        self._state_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self._ctx = self.context(("obstacle", "cart", "floor", "target"))
        # full_state: the scripted joints' indices in the plant's positions, and the rest (iiwa at zero, WSG open)
        n = self.plant.num_positions()
        index = lambda m: self.plant.GetPositionsFromArray(m, np.arange(n, dtype=float)).astype(int)
        self._paths = [(index(self.kin.cart), spec.cart.path)]
        self._paths += [(index(self.kin.obstacles[o.name]), o.get_path()) for o in spec.obstacles]
        s = self.kin.target_body.floating_positions_start()
        self._target_q = np.arange(s, s + 7)  # quaternion (w, x, y, z), position
        ctx = self.diagram.CreateDefaultContext()
        self.set_time(ctx, 0.0, np.zeros(NQ))
        self._q0 = self.plant.GetPositions(self.plant_context(ctx))

    # ---- contexts -----------------------------------------------------------
    def context(self, categories: tuple[str, ...]):
        """A diagram context where only robot <-> `categories` pairs are candidates."""
        key = tuple(sorted(categories))
        if key in self._ctx_cache:
            return self._ctx_cache[key]
        ctx = self.diagram.CreateDefaultContext()
        sg_ctx = self.scene_graph.GetMyMutableContextFromRoot(ctx)
        g = self._geoms
        world = GeometrySet(g["base"] + g["obstacle"] + g["cart"] + g["floor"] + g["target"])
        robot = GeometrySet(g["robot"] + g["base"])
        decl = CollisionFilterDeclaration().ExcludeWithin(robot).ExcludeWithin(world)
        for cat in ("obstacle", "cart", "floor", "target"):
            if cat not in key and g[cat]:
                decl.ExcludeBetween(GeometrySet(g["robot"]), GeometrySet(g[cat]))
        self.scene_graph.collision_filter_manager(sg_ctx).Apply(decl)
        self._ctx_cache[key] = ctx
        return ctx

    def plant_context(self, ctx):
        return self.plant.GetMyMutableContextFromRoot(ctx)

    def set_time(self, ctx, t: float, q: np.ndarray | None = None) -> None:
        """Cart, obstacles and target where (and moving as) the prediction says at t; iiwa at q, WSG open."""
        self.world.set_kinematic_state(self.kin, ctx, t, q)
        self.plant.SetPositions(self.plant_context(ctx), self.kin.wsg, [-WSG_OPEN / 2, WSG_OPEN / 2])
        self.plant.SetFreeBodySpatialVelocity(
            self.plant_context(ctx), self.kin.target_body, SpatialVelocity(np.zeros(3), self.world.target_velocity(t))
        )

    def full_state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(q_full(t), dq_full/dt) of the kinematic plant with the iiwa at zero (cached per t): what set_time puts
        in a context, written directly (about twice as fast as through a context)."""
        key = round(float(t), 9)
        if key not in self._state_cache:
            if len(self._state_cache) > 4096:
                self._state_cache.clear()
            q, qdot = self._q0.copy(), np.zeros(len(self._q0))
            for i, path in self._paths:  # cart (x, y), obstacles (x, y, z): prismatic joints
                q[i], qdot[i] = path.position(t)[: len(i)], path.velocity(t)[: len(i)]
            X = self.world.target_body_pose(t)  # the target: a free body, translating with the cart
            q[self._target_q] = np.concatenate([X.rotation().ToQuaternion().wxyz(), X.translation()])
            qdot[self._target_q[4:]] = self.world.target_velocity(t)
            self._state_cache[key] = (q, qdot)
        return self._state_cache[key]

    # ---- distance rows for velocity dampers ------------------------------------
    def rows(self, q: np.ndarray, t: float, max_distance: float) -> list[DistanceRow]:
        """Every robot <-> world geometry pair closer than `max_distance` at (q, t)."""
        ctx = self._ctx
        self.set_time(ctx, t, q)
        pc = self.plant_context(ctx)
        qo = self.scene_graph.get_query_output_port().Eval(self.scene_graph.GetMyContextFromRoot(ctx))
        v_full = self.plant.GetVelocities(pc)
        out = []
        for p in qo.ComputeSignedDistancePairwiseClosestPoints(max_distance):
            c = self.roles.classify_geometry_pair(p.id_A, p.id_B)
            if c is None:
                continue
            robot_b, other_b, cat = c
            if self.roles.geom_body[p.id_A] == robot_b:
                g_r, p_r, g_o, p_o, n = p.id_A, p.p_ACa, p.id_B, p.p_BCb, p.nhat_BA_W
            else:
                g_r, p_r, g_o, p_o, n = p.id_B, p.p_BCb, p.id_A, p.p_ACa, -p.nhat_BA_W
            J_r = self._point_jacobian(pc, robot_b, self.inspector.GetPoseInFrame(g_r) @ p_r)
            J_o = self._point_jacobian(pc, other_b, self.inspector.GetPoseInFrame(g_o) @ p_o)
            out.append(
                DistanceRow(
                    phi=float(p.distance),
                    a=n @ J_r[:, self.iiwa_v : self.iiwa_v + NQ],
                    vn=float(n @ (J_o @ v_full)),
                    category=cat,
                    robot_body=self.roles.body_name(robot_b),
                    other_body=self.roles.body_name(other_b),
                )
            )
        return out

    def _point_jacobian(self, pc, body_index: int, p_BC: np.ndarray) -> np.ndarray:
        frame = self.plant.get_body(BodyIndex(body_index)).body_frame()
        return self.plant.CalcJacobianTranslationalVelocity(pc, JacobianWrtVariable.kV, frame, p_BC, self.W, self.W)
