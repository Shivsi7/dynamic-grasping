"""Collision checks at a hypothetical (q, t), on the scene's kinematic diagram.

Uses only the scenario geometry (scene.build_kinematic_diagram) and the
predicted world state (WorldModel), so it works for any scenario the generator
can produce: any number and shape of obstacles, rigs and masts included, both
cart directions, every object.

* A state (q, t) is free when no robot body (iiwa links, WSG) is within
  `margin_obstacle` of an obstacle, within `margin_env` of the cart / conveyor /
  floor, penetrating the target (gripper excepted, as in the monitor), or
  penetrating another robot body (self-collision; WSG-internal pairs ignored).
* An edge (q_a, t_a) -> (q_b, t_b) is checked at samples no more than `dq_res`
  apart in every joint and `dt_res` apart in time, midpoint-first so that
  collisions are found early.  The margins absorb the motion between samples
  and the tracking lag of the position-controlled arm.
"""

from __future__ import annotations

import numpy as np
from pydrake.all import BodyIndex

from dyngrasp.scenario import ScenarioSpec
from dyngrasp.scene import BodyRoles, build_kinematic_diagram
from dyngrasp.world import WorldModel

WSG_FINGER_OPEN = 0.0535  # finger joint position for a fully open gripper (+-)


def _midpoint_order(n: int) -> np.ndarray:
    """Indices 0..n-1 in bisection order (ends first, then midpoints, ...)."""
    order, seen = [0, n - 1], {0, n - 1}
    queue = [(0, n - 1)]
    while queue:
        a, b = queue.pop(0)
        if b - a < 2:
            continue
        m = (a + b) // 2
        if m not in seen:
            order.append(m)
            seen.add(m)
        queue += [(a, m), (m, b)]
    return np.array([i for i in order if i < n])


class SpaceTimeChecker:
    def __init__(
        self,
        spec: ScenarioSpec,
        world: WorldModel,
        margin_obstacle: float = 0.03,
        margin_env: float = 0.005,
        dq_res: float = 0.03,
        dt_res: float = 0.04,
    ):
        self.spec, self.world = spec, world
        self.margin_obstacle = margin_obstacle
        self.margin_env = margin_env
        self.dq_res, self.dt_res = dq_res, dt_res
        self.diagram, self.kin = build_kinematic_diagram(spec)
        self.roles = BodyRoles(self.kin)
        self.ctx = self.diagram.CreateDefaultContext()
        plant = self.kin.plant
        self.plant = plant
        self.pctx = plant.GetMyMutableContextFromRoot(self.ctx)
        self.sgctx = self.kin.scene_graph.GetMyContextFromRoot(self.ctx)
        self.port = self.kin.scene_graph.get_query_output_port()
        self._max_dist = max(margin_obstacle, margin_env, 1e-4)

        # Index maps into the full position vector (positions are not contiguous per model).
        nq = plant.num_positions()
        idx = np.arange(nq, dtype=float)
        get = lambda m: plant.GetPositionsFromArray(m, idx).astype(int)
        self.i_iiwa = get(self.kin.iiwa)
        self.i_wsg = get(self.kin.wsg)
        self.i_cart = get(self.kin.cart)
        self.i_target = get(self.kin.target)
        self.i_obst = [(o.get_path(), get(self.kin.obstacles[o.name])) for o in spec.obstacles]
        self._q_full = np.array(plant.GetPositions(self.pctx))
        wsg_open = np.array([-WSG_FINGER_OPEN, WSG_FINGER_OPEN])
        self._q_full[self.i_wsg] = np.clip(
            wsg_open, plant.GetPositionLowerLimits()[self.i_wsg], plant.GetPositionUpperLimits()[self.i_wsg]
        )
        self._cart_path = spec.cart.path
        self._t_env: float | None = None

        # Pair classification, precomputed per geometry pair on first sight.
        self._robot = self.roles.robot
        self._gripper = self.roles.gripper
        self._pair_cache: dict = {}
        self.n_state_checks = 0
        self.n_edge_checks = 0

    # ---- state ------------------------------------------------------------------
    def _set_env(self, t: float) -> None:
        if self._t_env is not None and abs(t - self._t_env) < 1e-12:
            return
        q = self._q_full
        q[self.i_cart] = self._cart_path.position(t)[: len(self.i_cart)]
        for path, ii in self.i_obst:
            q[ii] = path.position(t)[: len(ii)]
        X = self.world.target_body_pose(t)
        quat = X.rotation().ToQuaternion()
        q[self.i_target] = np.concatenate([[quat.w(), quat.x(), quat.y(), quat.z()], X.translation()])
        self._t_env = t

    def set_state(self, q: np.ndarray, t: float) -> None:
        self._set_env(t)
        self._q_full[self.i_iiwa] = q
        self.plant.SetPositions(self.pctx, self._q_full)

    def _pair_category(self, id_A, id_B) -> str | None:
        """'obstacle' | 'env' | 'contact' (penetration only), or None if the pair never counts."""
        key = (id_A, id_B)
        if key in self._pair_cache:
            return self._pair_cache[key]
        gb = self.roles.geom_body
        a, b = gb.get(id_A), gb.get(id_B)
        cat = None
        if a is not None and b is not None:
            if a in self._robot and b in self._robot:
                if not (a in self._gripper and b in self._gripper):
                    cat = "contact"  # self-collision
            else:
                c = self.roles.classify(a, b)
                if c is not None:
                    cat = {"obstacle": "obstacle", "target": "contact"}.get(c[2], "env")
        self._pair_cache[key] = cat
        return cat

    def _pair_margin(self, id_A, id_B) -> float | None:
        """Required clearance for a geometry pair (margins may be changed at run time)."""
        cat = self._pair_category(id_A, id_B)
        if cat is None:
            return None
        return {"obstacle": self.margin_obstacle, "env": self.margin_env}.get(cat, 0.0)

    def state_free(self, q: np.ndarray, t: float) -> bool:
        self.n_state_checks += 1
        self.set_state(q, t)
        qo = self.port.Eval(self.sgctx)
        for p in qo.ComputeSignedDistancePairwiseClosestPoints(self._max_dist):
            m = self._pair_margin(p.id_A, p.id_B)
            if m is not None and p.distance < m:
                return False
        return True

    def clearance(self, q: np.ndarray, t: float, max_distance: float = 1.0) -> tuple[float, float]:
        """(distance to obstacles, distance to cart/floor) at (q, t), as the monitor measures it."""
        self.set_state(q, t)
        return self.roles.robot_clearance(self.port.Eval(self.sgctx), max_distance)

    def collisions(self, q: np.ndarray, t: float) -> list[tuple[str, str, float]]:
        """Every pair that makes (q, t) not free: (body A, body B, distance).  For debugging."""
        self.set_state(q, t)
        qo = self.port.Eval(self.sgctx)
        out = []
        for p in qo.ComputeSignedDistancePairwiseClosestPoints(self._max_dist):
            m = self._pair_margin(p.id_A, p.id_B)
            if m is not None and p.distance < m:
                na = self.plant.get_body(BodyIndex(self.roles.geom_body[p.id_A])).name()
                nb = self.plant.get_body(BodyIndex(self.roles.geom_body[p.id_B])).name()
                out.append((na, nb, float(p.distance)))
        return out

    # ---- edges ------------------------------------------------------------------
    def n_samples(self, qa, ta, qb, tb) -> int:
        n_q = float(np.max(np.abs(np.asarray(qb) - np.asarray(qa)))) / self.dq_res
        n_t = abs(tb - ta) / self.dt_res
        return int(np.ceil(max(n_q, n_t, 1.0))) + 1

    def edge_free(self, qa: np.ndarray, ta: float, qb: np.ndarray, tb: float) -> bool:
        self.n_edge_checks += 1
        n = self.n_samples(qa, ta, qb, tb)
        dq, dt = qb - qa, tb - ta
        for i in _midpoint_order(n):
            s = i / (n - 1)
            if not self.state_free(qa + s * dq, ta + s * dt):
                return False
        return True

    def path_free(self, Q: np.ndarray, T: np.ndarray) -> bool:
        return all(self.edge_free(Q[i], T[i], Q[i + 1], T[i + 1]) for i in range(len(T) - 1))

    def hold_free(self, q: np.ndarray, t: float, horizon: float) -> bool:
        """Can the robot stay at q from t to t + horizon?"""
        return self.edge_free(q, t, q, t + horizon)

    def first_collision_time(self, q: np.ndarray, t0: float, t1: float) -> float:
        """Earliest t in [t0, t1] at which holding q is not free (inf if never), to dt_res."""
        for t in np.arange(t0, t1 + 1e-9, self.dt_res):
            if not self.state_free(q, float(t)):
                return float(t)
        return np.inf
