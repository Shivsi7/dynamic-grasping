"""Known YCB objects: how they rest on the cart and a nominal top-down grasp.

Frames
------
C  cart top: origin at the centre of the cart's top surface, z up (see scene.py)
U  "upright object" frame: origin at the centre of the object's footprint, on
   the cart surface, z up.  X_CU = RotZ(yaw).  This is the frame the world model
   predicts and that grasp selection should reason in.
O  the object's own body frame from its SDF (the YCB files have +Y pointing to
   the bottom of the object and the origin at the centre; the mustard bottle
   has z up with the origin at its base).
G  gripper frame = WSG `body` frame: fingers extend along +y, close along x.
   The finger pads are centred at y = 0.108 (p_GgraspO = [0, 0.11, 0] in the
   course notes).

`X_UG_grasp` is a conservative top-down grasp for the baseline controller and
for success checking; grasp selection (member 2) can and should override it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydrake.all import RigidTransform, RotationMatrix

# Finger pad centre along G's y axis, and the pads' extent beyond it.
PAD_Y = 0.108
FINGER_TIP_Y = 0.118
# WSG body box extends +-0.036 along G's y; the object top must stay below it.
BODY_HALF_Y = 0.036

# Rotation of the gripper for a top-down grasp closing along U's y axis:
# x_G = y_U, y_G = -z_U, z_G = x_G x y_G = -x_U.
R_UG_TOPDOWN = RotationMatrix(np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
assert abs(np.linalg.det(R_UG_TOPDOWN.matrix()) - 1) < 1e-9


@dataclass(frozen=True)
class ObjectInfo:
    name: str
    url: str  # package:// URL of the SDF
    body_name: str
    X_UO: RigidTransform  # body frame expressed in the upright frame
    height: float  # along U's z
    footprint: tuple[float, float]  # extent along U's x and y
    grasp_width: float  # opening needed across U's y
    mass: float

    @property
    def pad_height(self) -> float:
        """Height above the cart surface at which the finger pads should sit.

        Mid-height, unless the WSG body (which starts 0.072 above the pads) would
        then be below the object's top: for tall objects the pads sit 5 cm below
        the top, leaving the body ~2 cm above it.
        """
        return max(0.02, 0.5 * self.height, self.height - 0.05)

    @property
    def X_UG_grasp(self) -> RigidTransform:
        return RigidTransform(R_UG_TOPDOWN, [0.0, 0.0, self.pad_height + PAD_Y])

    @property
    def p_UO_centre(self) -> np.ndarray:
        """Centre of the object's bounding box in U."""
        return np.array([0.0, 0.0, 0.5 * self.height])


def _centred(name, file, body, size_xyz, mass):
    # YCB convention: +Y towards bottom, origin at the centre.  Upright: +Y -> -z.
    sx, sy, sz = size_xyz
    return ObjectInfo(
        name=name,
        url=f"package://manipulation/hydro/{file}",
        body_name=body,
        X_UO=RigidTransform(RotationMatrix.MakeXRotation(-np.pi / 2), [0.0, 0.0, 0.5 * sy]),
        height=sy,
        footprint=(sx, sz),
        grasp_width=sz,
        mass=mass,
    )


OBJECTS: dict[str, ObjectInfo] = {
    o.name: o
    for o in [
        _centred("soup_can", "005_tomato_soup_can.sdf", "base_link_soup", (0.064, 0.100, 0.064), 0.349),
        _centred("sugar_box", "004_sugar_box.sdf", "base_link_sugar", (0.090, 0.170, 0.039), 0.514),
        _centred("cracker_box", "003_cracker_box.sdf", "base_link_cracker", (0.158, 0.207, 0.066), 0.411),
        _centred("gelatin_box", "009_gelatin_box.sdf", "base_link_gelatin", (0.087, 0.070, 0.026), 0.097),
        _centred("potted_meat_can", "010_potted_meat_can.sdf", "base_link_meat", (0.095, 0.082, 0.053), 0.370),
        ObjectInfo(
            name="mustard",
            url="package://manipulation/hydro/006_mustard_bottle.sdf",
            body_name="base_link_mustard",
            # Collision hull spans x[-0.064, 0.034], y[-0.057, 0.010], z[-0.003, 0.188].
            X_UO=RigidTransform([0.0153, 0.0236, 0.0033]),
            height=0.192,
            footprint=(0.098, 0.067),
            grasp_width=0.067,
            mass=0.435,
        ),
    ]
}


def get_object(name: str) -> ObjectInfo:
    try:
        return OBJECTS[name]
    except KeyError:
        raise KeyError(f"unknown object {name!r}; known: {sorted(OBJECTS)}") from None
