"""Grasp candidates for the known YCB objects, from their bounding boxes.

A candidate is a gripper pose X_UG in the object's upright frame U (see
objects.py), plus its approach direction (the gripper's +y: palm towards
fingertips) and closing axis (the gripper's x).  Families:

    top        approach straight down, closing across U's y or x, and each one
               turned by 180 deg about the vertical (same contact, other wrist)
    top_tilt   a top grasp pitched by +-`tilt` about its closing axis (reaches the
               near / far edge of the workspace without a fully vertical wrist)
    side       approach horizontally onto each face of the box, both wrist flips;
               objects that are round in plan view also get the diagonals

Each candidate is placed as deep as possible (pads at most at the object's centre
along the approach) such that the WSG palm stays clear of the object, the
fingertips stay above the cart, and the pads straddle the object.  These are
the geometric checks the planner's collision checker cannot do: it ignores
gripper-target contact, because a grasp *is* gripper-target contact.

Whether a candidate is reachable, collision-free and trackable at a given time
is for the planner to find out (goals.py).  This module is meant to be the
interface to grasp selection: anything that produces `GraspCandidate`s (or plain
X_UG transforms) can be handed to the planner instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydrake.all import RigidTransform, RotationMatrix

from dyngrasp.objects import PAD_Y, ObjectInfo

# WSG collision geometry in the gripper frame G (wsg::collision and the finger tips).
PALM_HALF = np.array([0.073, 0.03625, 0.025])  # palm box half extents, centred on G
TIP_CENTRE_X = 0.0575  # finger tip boxes at x = +-0.0575 (open), y = PAD_Y
TIP_HALF = np.array([0.0015, 0.01, 0.01])
FINGER_HALF = np.array([0.006, 0.041, 0.01])  # finger boxes at x = +-0.065, y = 0.0771
FINGER_CENTRE = np.array([0.065, 0.0771, 0.0])
# Families that fail *physically* for an object although they pass the box-geometry checks,
# measured with `scripts/grasp_feasibility.py --execute` (each candidate executed alone,
# obstacle-free, cart at 0.10 and 0.25 m/s).  The mustard bottle is not a box: tilted pads
# land on the tapering shoulder and slip (0 of 4 tilted executions held it, 4/4 top did).
# A real grasp selector (contact model, antipodal check on the mesh) should replace this.
EXCLUDED_FAMILIES: dict[str, set[str]] = {"mustard": {"top_tilt"}}
MAX_WIDTH = 0.095  # widest object the open gripper (gap ~0.112 between tips) closes on safely
MIN_WIDTH = 0.01


@dataclass(frozen=True)
class GraspCandidate:
    name: str
    family: str  # "top" | "top_tilt" | "side"
    X_UG: RigidTransform
    width: float  # object extent across the closing axis
    depth: float  # pad centre distance from the near face, along the approach

    @property
    def approach_U(self) -> np.ndarray:
        return self.X_UG.rotation().matrix()[:, 1]

    @property
    def close_U(self) -> np.ndarray:
        return self.X_UG.rotation().matrix()[:, 0]

    def approach_W(self, X_WU: RigidTransform) -> np.ndarray:
        return X_WU.rotation().matrix() @ self.approach_U


def as_candidate(g, i: int = 0) -> GraspCandidate:
    """Wrap a plain X_UG transform (e.g. from another grasp selector) as a candidate."""
    if isinstance(g, GraspCandidate):
        return g
    return GraspCandidate(f"custom_{i}", "custom", g, width=np.nan, depth=np.nan)


# ---- geometry ------------------------------------------------------------------------
def _box_corners(half: np.ndarray, X: RigidTransform, centre=np.zeros(3)) -> np.ndarray:
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return (X.rotation().matrix() @ (centre + s * half).T).T + X.translation()


def _boxes_overlap(half_a, X_a: RigidTransform, half_b, X_b: RigidTransform) -> bool:
    """Separating-axis test for two oriented boxes (centred at their frames)."""
    Ra, Rb = X_a.rotation().matrix(), X_b.rotation().matrix()
    d = X_b.translation() - X_a.translation()
    axes = [Ra[:, i] for i in range(3)] + [Rb[:, i] for i in range(3)]
    axes += [np.cross(Ra[:, i], Rb[:, j]) for i in range(3) for j in range(3)]
    for L in axes:
        n = np.linalg.norm(L)
        if n < 1e-9:
            continue
        L = L / n
        ra = np.sum(np.abs(Ra.T @ L) * half_a)
        rb = np.sum(np.abs(Rb.T @ L) * half_b)
        if abs(d @ L) > ra + rb:
            return False
    return True


def grasp_is_valid(obj: ObjectInfo, X_UG: RigidTransform, palm_clearance: float = 0.005,
                   cart_clearance: float = 0.004) -> bool:
    """Palm clear of the object, gripper above the cart surface, pads straddling the object."""
    half_obj = np.array([obj.footprint[0] / 2, obj.footprint[1] / 2, obj.height / 2])
    X_UO_box = RigidTransform([0.0, 0.0, obj.height / 2])
    if _boxes_overlap(PALM_HALF + palm_clearance, X_UG, half_obj, X_UO_box):
        return False
    # everything (palm, fingers, tips) above the cart top (z = 0 in U)
    pts = [_box_corners(PALM_HALF, X_UG)]
    for sx in (-1, 1):
        pts.append(_box_corners(FINGER_HALF, X_UG, FINGER_CENTRE * [sx, 1, 1]))
        pts.append(_box_corners(TIP_HALF, X_UG, np.array([sx * TIP_CENTRE_X, PAD_Y, 0.0])))
    if np.min(np.vstack(pts)[:, 2]) < cart_clearance:
        return False
    # the fingers do not hit the object while open: tips outside it along the closing axis
    for sx in (-1, 1):
        X_tip = X_UG @ RigidTransform([sx * TIP_CENTRE_X, PAD_Y, 0.0])
        if _boxes_overlap(TIP_HALF, X_tip, half_obj, X_UO_box):
            return False
    # the pad point (monitor's "in hand" test) inside the object, with a few mm to spare
    p = X_UG @ np.array([0.0, PAD_Y, 0.0])
    inner = half_obj - 0.003
    return bool(np.all(np.abs(p - X_UO_box.translation()) <= inner))


def _frame(approach: np.ndarray, close: np.ndarray) -> RotationMatrix:
    a = approach / np.linalg.norm(approach)
    c = close - (close @ a) * a
    c /= np.linalg.norm(c)
    return RotationMatrix(np.column_stack([c, a, np.cross(c, a)]))


def _extent_along(obj: ObjectInfo, u: np.ndarray) -> float:
    """Width of the object's box along direction u (in U)."""
    half = np.array([obj.footprint[0] / 2, obj.footprint[1] / 2, obj.height / 2])
    return float(2 * np.sum(np.abs(u) * half))


def _place(obj: ObjectInfo, R: RotationMatrix, entry: np.ndarray, max_depth: float, name: str,
           family: str, z_pad: float | None = None, n: int = 40) -> GraspCandidate | None:
    """Deepest valid pad position along the approach, entering the object at `entry` (U)."""
    a = R.matrix()[:, 1]
    width = _extent_along(obj, R.matrix()[:, 0])
    if not (MIN_WIDTH <= width <= MAX_WIDTH + 1e-9):
        return None
    for depth in np.linspace(max_depth, 0.008, n):
        p_pad = entry + depth * a
        if z_pad is not None:
            p_pad = p_pad.copy()
            p_pad[2] = z_pad
        X_UG = RigidTransform(R, p_pad - PAD_Y * a)
        if grasp_is_valid(obj, X_UG):
            return GraspCandidate(name, family, X_UG, width, float(depth))
    return None


def generate_grasps(obj: ObjectInfo, tilt_deg: float = 35.0, side: bool = True, tilted: bool = True,
                    round_tol: float = 0.006, exclusions: bool = True) -> list[GraspCandidate]:
    """All valid candidates for `obj`, the nominal top-down grasp first (minus the families in
    EXCLUDED_FAMILIES for this object, unless exclusions=False)."""
    if exclusions:
        tilted = tilted and "top_tilt" not in EXCLUDED_FAMILIES.get(obj.name, set())
        side = side and "side" not in EXCLUDED_FAMILIES.get(obj.name, set())
    h = obj.height
    fx, fy = obj.footprint
    centre = np.array([0.0, 0.0, h / 2])
    ez = np.array([0.0, 0.0, 1.0])
    out: list[GraspCandidate] = []

    # top-down: closing across y (nominal, as objects.X_UG_grasp) or x, both wrist flips
    tops = []
    for axis_name, c in (("y", np.array([0.0, 1.0, 0.0])), ("x", np.array([1.0, 0.0, 0.0]))):
        for flip in (1.0, -1.0):
            tops.append((f"top_{axis_name}{'+' if flip > 0 else '-'}", _frame(-ez, flip * c)))
    for name, R in tops:
        g = _place(obj, R, np.array([0.0, 0.0, h]), h / 2, name, "top")
        if g is not None:
            out.append(g)

    # tilted top: pitch about the closing axis, entering through the top centre
    if tilted:
        for name, R in tops:
            c = R.matrix()[:, 0]
            for sign in (1.0, -1.0):
                Rt = RotationMatrix(_axis_angle(c, sign * np.radians(tilt_deg))) @ R
                g = _place(obj, Rt, np.array([0.0, 0.0, h]), h / 2,
                           f"{name}_tilt{'+' if sign > 0 else '-'}{int(tilt_deg)}", "top_tilt")
                if g is not None:
                    out.append(g)

    # side grasps: horizontal approach onto each face (and diagonals if round), both wrist flips
    if side:
        az = [0, 90, 180, 270]
        if abs(fx - fy) < round_tol:
            az += [45, 135, 225, 315]
        for deg in az:
            phi = np.radians(deg)
            inward = -np.array([np.cos(phi), np.sin(phi), 0.0])  # approach towards the centre
            half_along = 0.5 * _extent_along(obj, inward)
            entry = centre - half_along * inward  # on the face, at mid height
            z_pad = float(np.clip(h / 2, 0.03, h - 0.012))
            for flip in (1.0, -1.0):
                c = flip * np.cross(ez, inward)
                g = _place(obj, _frame(inward, c), entry, half_along, f"side_{deg}{'+' if flip > 0 else '-'}",
                           "side", z_pad=z_pad)
                if g is not None:
                    out.append(g)
    return out


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def prior_score(g: GraspCandidate, X_WU: RigidTransform, p_W_shoulder: np.ndarray) -> float:
    """How natural the approach is for an arm at the shoulder, in [-1, 1] (higher first):
    straight down 1, horizontal from the robot's side 1, from the left/right 0, from the
    far side -1; tilted approaches in between.  The planner skips candidates below -0.2."""
    a = g.approach_W(X_WU)
    d = X_WU.translation() - p_W_shoulder
    d_h = np.array([d[0], d[1], 0.0])
    d_h /= max(np.linalg.norm(d_h), 1e-9)
    down = max(-a[2], 0.0)
    return float(down + (1.0 - down) * (a @ d_h))
