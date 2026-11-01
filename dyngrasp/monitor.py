"""EpisodeMonitor: watches one episode and records the metrics.

Collisions   at `contact_hz` (500 Hz), from `contact_results`: any contact between a
             robot body (iiwa links 1-7, WSG body, fingers) and an obstacle, the
             cart, the floor or the target -- except gripper <-> target, which
             is grasping.  (Self-collisions are not counted.)
Clearance    at `rate_hz`, from `query_object`: minimum signed distance between
             robot geometry and obstacle geometry, and separately to the cart/floor.
Grasp        at `rate_hz`, from `body_poses` and the WSG state: the object is
             "in hand" when the finger-pad point lies inside the object's
             bounding box (grown by IN_HAND_MARGIN, so the object may slip in
             the grasp) and the gripper is not fully open; "lifted" when its
             centre is LIFT_HEIGHT above its rest height; success after
             HOLD_TIME seconds of in-hand & lifted.
Trajectory   at `rate_hz`: q, gripper position, object position, clearance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydrake.all import AbstractValue, ContactResults, LeafSystem, QueryObject, RigidTransform

from dyngrasp.controller import NQ, WSG_OPEN
from dyngrasp.objects import PAD_Y
from dyngrasp.scene import BodyRoles, Scene

LIFT_HEIGHT = 0.05
HOLD_TIME = 0.5
IN_HAND_MARGIN = 0.02  # how far outside the object's bounding box the pad point may be
CLEARANCE_QUERY_DISTANCE = 1.0


@dataclass
class CollisionEvent:
    t: float
    robot_body: str
    other_body: str
    category: str  # obstacle | cart | floor | target


@dataclass
class EpisodeLog:
    t: list[float] = field(default_factory=list)
    q: list[np.ndarray] = field(default_factory=list)
    p_WG: list[np.ndarray] = field(default_factory=list)
    p_WO: list[np.ndarray] = field(default_factory=list)
    wsg: list[float] = field(default_factory=list)
    clearance_obstacles: list[float] = field(default_factory=list)
    clearance_environment: list[float] = field(default_factory=list)
    in_hand: list[bool] = field(default_factory=list)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {k: np.array(v) for k, v in self.__dict__.items()}


class EpisodeMonitor(LeafSystem):
    def __init__(self, scene: Scene, rate_hz: float = 100.0, stop_on_collision: bool = True, contact_hz: float = 500.0):
        super().__init__()
        self.scene = scene
        self.plant = scene.plant
        self.stop_on_collision = stop_on_collision
        self.roles = BodyRoles(scene)
        self._gripper_index = int(scene.gripper_body.index())
        self._target_index = int(scene.target_body.index())
        self._cart_index = int(scene.cart_top_body.index())
        self._obj = scene.object_info

        # results
        self.collisions: list[CollisionEvent] = []
        self.collision_pairs: set[tuple[str, str]] = set()
        self.min_clearance_obstacles = np.inf
        self.min_clearance_obstacles_t = np.nan
        self.min_clearance_environment = np.inf
        self.t_grasp: float | None = None  # first time the object is in hand with the gripper closing
        self.t_lift: float | None = None  # first time it is in hand and lifted
        self.t_success: float | None = None
        self.dropped = False
        self.log = EpisodeLog()
        self._lifted_since: float | None = None
        self._was_lifted = False
        self.done = False
        self.done_reason: str | None = None

        self._contacts_in = self.DeclareAbstractInputPort("contact_results", AbstractValue.Make(ContactResults()))
        self._poses_in = self.DeclareAbstractInputPort("body_poses", AbstractValue.Make([RigidTransform()]))
        self._query_in = self.DeclareAbstractInputPort("query_object", AbstractValue.Make(QueryObject()))
        self._wsg_in = self.DeclareVectorInputPort("wsg.state_measured", 2)
        self._q_in = self.DeclareVectorInputPort("iiwa.position_measured", NQ)
        self.DeclarePeriodicPublishEvent(1.0 / contact_hz, 0.0, self._check_contacts)
        self.DeclarePeriodicPublishEvent(1.0 / rate_hz, 0.0, self._sample)

    # ---- collisions (at contact_hz) ---------------------------------------------
    def _check_contacts(self, context):
        cr = self._contacts_in.Eval(context)
        pairs = []
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            pairs.append((int(info.bodyA_index()), int(info.bodyB_index())))
        for i in range(cr.num_hydroelastic_contacts()):
            info = cr.hydroelastic_contact_info(i)
            s = info.contact_surface()
            pairs.append((self.roles.geom_body[s.id_M()], self.roles.geom_body[s.id_N()]))
        t = context.get_time()
        for a, b in pairs:
            c = self.roles.classify(a, b)
            if c is None:
                continue
            robot, other, cat = c
            key = (self.roles.body_name(robot), self.roles.body_name(other))
            if key not in self.collision_pairs:
                self.collision_pairs.add(key)
                self.collisions.append(CollisionEvent(t, key[0], key[1], cat))
            if self.stop_on_collision and not self.done:
                self.done, self.done_reason = True, "collision"

    # ---- clearance, grasp, log (periodic) ---------------------------------------
    def _clearances(self, context) -> tuple[float, float]:
        return self.roles.robot_clearance(self._query_in.Eval(context), CLEARANCE_QUERY_DISTANCE)

    def _sample(self, context):
        t = context.get_time()
        poses = self._poses_in.Eval(context)
        X_WG = poses[self._gripper_index]
        X_WO = poses[self._target_index]
        X_WC = poses[self._cart_index]
        wsg = float(self._wsg_in.Eval(context)[0])
        q = np.array(self._q_in.Eval(context))

        # Is the finger-pad point inside the object's (upright-frame) bounding box?
        X_WU = X_WO @ self._obj.X_UO.inverse()
        p_WOc = X_WU @ self._obj.p_UO_centre
        p_Upad = X_WU.inverse() @ (X_WG @ np.array([0.0, PAD_Y, 0.0]))
        fx, fy = self._obj.footprint
        m = IN_HAND_MARGIN
        in_hand = (
            abs(p_Upad[0]) < fx / 2 + m
            and abs(p_Upad[1]) < fy / 2 + m
            and -m < p_Upad[2] < self._obj.height + m
            and wsg < WSG_OPEN - 0.01
        )
        rest_height = X_WC.translation()[2] + self._obj.p_UO_centre[2]
        lifted = (p_WOc[2] - rest_height) > LIFT_HEIGHT

        if in_hand and self.t_grasp is None:
            self.t_grasp = t
        if in_hand and lifted:
            if self.t_lift is None:
                self.t_lift = t
            self._was_lifted = True
            if self._lifted_since is None:
                self._lifted_since = t
            elif self.t_success is None and t - self._lifted_since >= HOLD_TIME:
                self.t_success = t
                if not self.done:
                    self.done, self.done_reason = True, "success"
        else:
            self._lifted_since = None
            if self._was_lifted and not in_hand and self.t_success is None:
                self.dropped = True

        d_obs, d_env = self._clearances(context)
        if d_obs < self.min_clearance_obstacles:
            self.min_clearance_obstacles, self.min_clearance_obstacles_t = d_obs, t
        self.min_clearance_environment = min(self.min_clearance_environment, d_env)

        L = self.log
        L.t.append(t)
        L.q.append(q)
        L.p_WG.append(X_WG.translation().copy())
        L.p_WO.append(p_WOc.copy())
        L.wsg.append(wsg)
        L.clearance_obstacles.append(d_obs)
        L.clearance_environment.append(d_env)
        L.in_hand.append(in_hand)

