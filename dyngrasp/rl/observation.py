"""(D) observation: one flat float32 vector from what every controller sees (RobotState + WorldModel).

Fixed scales (no running statistics), so nothing needs saving with the model beyond this code:

  q                       7   (q - mid) / half range of the joint limits
  commanded joint vel.    7   / v_max
  WSG opening             2   commanded, measured: 2 w / WSG_OPEN - 1
  gripper pose            9   (p_WG - [0.5, 0, 0.4]) / 0.5, and R_WG's first two columns
  grasp-pose error in G   9   position / 0.5, position / 0.05 clipped to +-1, rotation (axis x angle, rad)
  object velocity         3   / 0.3 (it translates with the cart; no rotation)
  object width            1   / 0.1 (the close-event threshold, see env.py)
  4 nearest obstacles    44   per obstacle: position - p_WG (/ 0.5), velocity (/ 0.5), half extents (/ 0.25; a pillar 3.2),
                              min distance to the robot (clipped to [-0.1, 1] m, / 0.25), present (0/1);
                              nearest first by that distance, zero-padded
  timing                  2   (t_enter - t) clipped to [0, 5] / 5, (t_exit - t) clipped to [-2, 10] / 10

The obstacle distances are the controllers' own query (collision.CollisionWorld posed at the
WorldModel's prediction, robot at the measured q), robot <-> obstacle geometry only.
"""

from __future__ import annotations

import numpy as np

from dyngrasp.collision import CollisionWorld
from dyngrasp.controller import WSG_OPEN, RobotModel, RobotState
from dyngrasp.world import WorldModel

N_OBSTACLES = 4
OBS_DIM = 7 + 7 + 2 + 9 + 9 + 3 + 1 + 11 * N_OBSTACLES + 2  # 84
D_MAX = 1.0  # obstacle distances are queried up to this (m)
P_CENTRE = np.array([0.5, 0.0, 0.4])


def grasp_error(X_WG, X_WGgrasp) -> tuple[np.ndarray, np.ndarray]:
    """(position, rotation as axis x angle) of the grasp pose relative to the gripper, in the gripper frame G."""
    R_GW = X_WG.rotation().inverse()
    aa = (R_GW @ X_WGgrasp.rotation()).ToAngleAxis()
    return R_GW @ (X_WGgrasp.translation() - X_WG.translation()), aa.axis() * aa.angle()


class Observer:
    def __init__(self, world: WorldModel, robot: RobotModel, collision: CollisionWorld | None = None):
        self.world, self.robot = world, robot
        self.cw = collision or CollisionWorld(world.spec, world)
        self._ctx = self.cw.context(("obstacle",))
        self._qo_port = self.cw.scene_graph.get_query_output_port()
        self.q_mid = 0.5 * (robot.q_upper + robot.q_lower)
        self.q_half = 0.5 * (robot.q_upper - robot.q_lower)
        self.half_extents = {
            o.name: np.full(3, o.dims[0]) if o.shape == "sphere" else 0.5 * np.asarray(o.dims, float)
            for o in world.obstacle_specs
        }

    def obstacle_distances(self, q: np.ndarray, t: float) -> dict[str, float]:
        """Min signed distance robot <-> each obstacle (D_MAX if farther), at the predicted time t."""
        d = {name: D_MAX for name in self.half_extents}
        if not d:
            return d
        self.cw.set_time(self._ctx, t, q)
        qo = self._qo_port.Eval(self.cw.scene_graph.GetMyContextFromRoot(self._ctx))
        roles = self.cw.roles
        for p in qo.ComputeSignedDistancePairwiseClosestPoints(D_MAX):
            c = roles.classify_geometry_pair(p.id_A, p.id_B)
            if c is not None and c[2] == "obstacle":
                name = roles.body_name(c[1])
                d[name] = min(d[name], float(p.distance))
        return d

    def __call__(self, state: RobotState, v_cmd: np.ndarray) -> np.ndarray:
        w, t = self.world, state.t
        X_WG = state.X_WG
        p_G = X_WG.translation()
        e_p, e_r = grasp_error(X_WG, w.grasp_pose(t))
        R = X_WG.rotation().matrix()
        parts = [
            (state.q - self.q_mid) / self.q_half,
            np.asarray(v_cmd) / self.robot.v_limits,
            [2.0 * state.wsg_cmd / WSG_OPEN - 1.0, 2.0 * state.wsg_width / WSG_OPEN - 1.0],
            (p_G - P_CENTRE) / 0.5,
            R[:, 0],
            R[:, 1],
            e_p / 0.5,
            np.clip(e_p / 0.05, -1.0, 1.0),
            e_r,
            w.target_velocity(t) / 0.3,
            [w.object.grasp_width / 0.1],
        ]
        dist = self.obstacle_distances(state.q, t)
        obstacles = sorted(w.obstacle_states(t), key=lambda o: (dist[o.name], np.linalg.norm(o.p - p_G)))
        for k in range(N_OBSTACLES):
            if k < len(obstacles):
                o = obstacles[k]
                parts += [(o.p - p_G) / 0.5, o.v / 0.5, self.half_extents[o.name] / 0.25,
                          [np.clip(dist[o.name], -0.1, D_MAX) / 0.25, 1.0]]
            else:
                parts.append(np.zeros(11))
        parts.append([np.clip(w.t_enter - t, 0.0, 5.0) / 5.0, np.clip(w.t_exit - t, -2.0, 10.0) / 10.0])
        obs = np.concatenate([np.asarray(p, float).ravel() for p in parts]).astype(np.float32)
        assert obs.shape == (OBS_DIM,)
        return obs
