"""Build the Drake scene for a ScenarioSpec.

The simulation is a course-style `HardwareStation` (iiwa driven in position
mode with an inverse-dynamics controller, WSG driven by the Schunk driver) plus:

* a conveyor: a static frame along the lane, and a carrier plate on two
  PD-controlled prismatic joints (x, y) whose top surface is the frame C; the
  target YCB object rests on it as a free body;
* obstacles: three PD-controlled prismatic joints (x, y, z) each, carrying a
  sphere or a box with collision geometry -- on a mast on a wheeled rig
  (`mount: cart`), or free-flying (`mount: float`);
* a floor.

Each PD joint chain is fed [q(t); v(t)] from the scenario's scripted path by a
`ScriptedJointSource`, so the bodies follow their script to ~1e-4 m while still
being ordinary dynamic bodies that the robot can collide with.

`build_station` returns a `Scene` holding the station diagram and the handles
(model instances, bodies, frames) the rest of the package needs.  The station's
ports are the usual ones: inputs `iiwa.position`, `wsg.position`; outputs
`iiwa.position_measured`, `iiwa.velocity_estimated`, `wsg.state_measured`,
`body_poses`, `contact_results`, `query_object`, ...

`build_kinematic_diagram` builds the same geometry with no drivers, for
planners and reactive controllers that need distance / collision queries at a
hypothetical (q, t): see `WorldModel.set_kinematic_state`.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path as FilePath

import numpy as np
from manipulation.station import LoadScenario, MakeHardwareStation
from manipulation.utils import ConfigureParser
from pydrake.all import (
    BasicVector,
    BodyIndex,
    CollisionFilterDeclaration,
    GeometrySet,
    Diagram,
    DiagramBuilder,
    LeafSystem,
    Meshcat,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    ModelDirectives,
    ProcessModelDirectives,
    QueryObject,
    RigidBody,
    RigidTransform,
    RobotDiagram,
    RobotDiagramBuilder,
    RollPitchYaw,
    RotationMatrix,
    SceneGraph,
    SpatialVelocity,
)

from dyngrasp.objects import ObjectInfo, get_object
from dyngrasp.paths import Path
from dyngrasp.scenario import RIG_PLATE, RIG_TOP, CartSpec, ObstacleSpec, ScenarioSpec

IIWA_URL = "package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf"
WSG_URL = "package://manipulation/hydro/schunk_wsg_50_with_tip.sdf"

# PD gains for the scripted joints (SAP integrates them implicitly, so they can be stiff).
CART_GAINS = (8000.0, 400.0)
OBSTACLE_GAINS = (3000.0, 150.0)

ASSET_DIR = FilePath(tempfile.gettempdir()) / "dyngrasp_assets"


# ---------------------------------------------------------------------------
# SDF generation
# ---------------------------------------------------------------------------


def _write_asset(name: str, text: str) -> FilePath:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(text.encode()).hexdigest()[:10]
    p = ASSET_DIR / f"{name}_{h}.sdf"
    if not p.exists():
        p.write_text(text)
    return p


def _inertial(mass: float, i: float = 1e-3) -> str:
    return (
        f"<inertial><mass>{mass}</mass><inertia><ixx>{i}</ixx><iyy>{i}</iyy><izz>{i}</izz>"
        f"<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>"
    )


def _prismatic(name: str, parent: str, child: str, axis: str, gains) -> str:
    p, d = gains
    return (
        f'<joint name="{name}" type="prismatic"><parent>{parent}</parent><child>{child}</child>'
        f"<axis><xyz>{axis}</xyz><limit><effort>1e5</effort></limit>"
        f"<dynamics><damping>0.0</damping></dynamics></axis>"
        f'<drake:controller_gains p="{p}" d="{d}"/></joint>'
    )


def cart_sdf(cart: CartSpec) -> str:
    """Conveyor frame (fixed to the world) + carrier plate on x/y prismatic joints."""
    sx, sy, sz = cart.size
    frame_h = cart.top_z - sz - 0.005  # the carrier plate rides 5 mm above the frame
    L = cart.conveyor_length
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
<model name="cart">
  <link name="conveyor">
    <pose>{cart.x} 0 0 0 0 0</pose>
    {_inertial(50.0, 1.0)}
    <visual name="frame"><pose>0 0 {frame_h / 2} 0 0 0</pose>
      <geometry><box><size>{sx} {L} {frame_h}</size></box></geometry>
      <material><diffuse>0.80 0.80 0.80 1</diffuse></material></visual>
    <visual name="belt"><pose>0 0 {frame_h + 0.001} 0 0 0</pose>
      <geometry><box><size>{sx - 0.02} {L} 0.002</size></box></geometry>
      <material><diffuse>0.70 0.70 0.70 1</diffuse></material></visual>
    <collision name="frame_collision"><pose>0 0 {frame_h / 2} 0 0 0</pose>
      <geometry><box><size>{sx} {L} {frame_h}</size></box></geometry></collision>
  </link>
  <joint name="conveyor_weld" type="fixed"><parent>world</parent><child>conveyor</child></joint>
  <link name="x_slider">{_inertial(0.1)}</link>
  {_prismatic("cart_x", "world", "x_slider", "1 0 0", CART_GAINS)}
  <link name="top">
    <pose>0 0 {cart.top_z} 0 0 0</pose>
    {_inertial(8.0, 0.2)}
    <visual name="top_visual"><pose>0 0 {-sz / 2} 0 0 0</pose>
      <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      <material><diffuse>0.55 0.56 0.58 1</diffuse></material></visual>
    <collision name="top_collision"><pose>0 0 {-sz / 2} 0 0 0</pose>
      <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      <drake:proximity_properties>
        <drake:mu_static>0.9</drake:mu_static><drake:mu_dynamic>0.8</drake:mu_dynamic>
      </drake:proximity_properties></collision>
  </link>
  {_prismatic("cart_y", "x_slider", "top", "0 1 0", CART_GAINS)}
</model>
</sdf>
"""


def _rig_elements(o: ObstacleSpec, z_centre: float) -> str:
    """Wheeled rig under a shape whose centre is z_centre above the floor (link origin)."""
    plate_t = 0.03
    z_plate = RIG_TOP - plate_t / 2 - z_centre
    z_wheel = 0.035 - z_centre
    mast_len = max(z_centre - o.half_height - RIG_TOP, 0.0)
    z_mast = RIG_TOP + mast_len / 2 - z_centre
    dark = "<material><diffuse>0.22 0.22 0.24 1</diffuse></material>"
    half = RIG_PLATE / 2 - 0.04
    wheels = "".join(
        f'<visual name="wheel{i}"><pose>{sx * half} {sy * half} {z_wheel} 0 0 0</pose>'
        f"<geometry><sphere><radius>0.035</radius></sphere></geometry>{dark}</visual>"
        for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)])
    )
    mast = ""
    if mast_len > 0.005:
        mast = (
            f'<visual name="mast"><pose>0 0 {z_mast} 0 0 0</pose>'
            f"<geometry><cylinder><radius>0.02</radius><length>{mast_len}</length></cylinder></geometry>{dark}</visual>"
            f'<collision name="mast_collision"><pose>0 0 {z_mast} 0 0 0</pose>'
            f"<geometry><cylinder><radius>0.02</radius><length>{mast_len}</length></cylinder></geometry></collision>"
        )
    return (
        f'<visual name="plate"><pose>0 0 {z_plate} 0 0 0</pose>'
        f"<geometry><box><size>{RIG_PLATE} {RIG_PLATE} {plate_t}</size></box></geometry>{dark}</visual>"
        f'<collision name="plate_collision"><pose>0 0 {z_plate} 0 0 0</pose>'
        f"<geometry><box><size>{RIG_PLATE} {RIG_PLATE} {plate_t}</size></box></geometry></collision>"
        + wheels
        + mast
    )


def obstacle_sdf(o: ObstacleSpec) -> str:
    if o.shape == "sphere":
        geom = f"<sphere><radius>{o.dims[0]}</radius></sphere>"
    elif o.shape == "box":
        geom = f"<box><size>{o.dims[0]} {o.dims[1]} {o.dims[2]}</size></box>"
    else:
        raise ValueError(f"unknown obstacle shape {o.shape!r}")
    r, g, b, a = o.color
    rig = _rig_elements(o, float(o.get_path().position(o.get_path().t_start)[2])) if o.mount == "cart" else ""
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
<model name="{o.name}">
  <link name="x_slider">{_inertial(0.05)}</link>
  {_prismatic(f"{o.name}_x", "world", "x_slider", "1 0 0", OBSTACLE_GAINS)}
  <link name="y_slider">{_inertial(0.05)}</link>
  {_prismatic(f"{o.name}_y", "x_slider", "y_slider", "0 1 0", OBSTACLE_GAINS)}
  <link name="{o.name}">
    {_inertial(1.0, 0.01)}
    <visual name="visual"><geometry>{geom}</geometry>
      <material><diffuse>{r} {g} {b} {a}</diffuse></material></visual>
    <collision name="collision"><geometry>{geom}</geometry></collision>
    {rig}
  </link>
  {_prismatic(f"{o.name}_z", "y_slider", o.name, "0 0 1", OBSTACLE_GAINS)}
</model>
</sdf>
"""


FLOOR_SDF = """<?xml version="1.0"?>
<sdf version="1.7">
<model name="floor">
  <link name="floor_link">
    <visual name="visual"><pose>0 0 -0.03 0 0 0</pose>
      <geometry><box><size>5 5 0.04</size></box></geometry>
      <material><diffuse>0.96 0.96 0.96 1</diffuse></material></visual>
    <collision name="collision"><pose>0 0 -0.03 0 0 0</pose>
      <geometry><box><size>4 4 0.04</size></box></geometry></collision>
  </link>
</model>
</sdf>
"""


# ---------------------------------------------------------------------------
# Directives
# ---------------------------------------------------------------------------


def robot_directives() -> str:
    """iiwa (with collision boxes) + WSG, as in the course's iiwa_and_wsg.dmd.yaml."""
    return f"""
- add_model:
    name: iiwa
    file: {IIWA_URL}
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
- add_model:
    name: wsg
    file: {WSG_URL}
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90]}}
"""


def scene_directives(spec: ScenarioSpec) -> str:
    obj = get_object(spec.target.object)
    cart_file = _write_asset("cart", cart_sdf(spec.cart))
    floor_file = _write_asset("floor", FLOOR_SDF)
    lines = [robot_directives()]
    lines.append(f"""
- add_model:
    name: floor
    file: file://{floor_file}
- add_weld:
    parent: world
    child: floor::floor_link
- add_model:
    name: cart
    file: file://{cart_file}
- add_model:
    name: target
    file: {obj.url}
""")
    for o in spec.obstacles:
        f = _write_asset(o.name, obstacle_sdf(o))
        lines.append(f"""
- add_model:
    name: {o.name}
    file: file://{f}
""")
    return "directives:" + "".join(lines)


def look_at_camera(name: str, p_WC, p_WT, width: int = 640, height: int = 480, fov_y_deg: float = 50.0) -> dict:
    """A CameraConfig (as a dict for the scenario YAML) at p_WC looking at p_WT, image up = world up."""
    p_WC, p_WT = np.asarray(p_WC, float), np.asarray(p_WT, float)
    z = p_WT - p_WC
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, up)) > 0.999:  # looking straight down: image up = +x
        up = np.array([1.0, 0.0, 0.0])
    y = -(up - np.dot(up, z) * z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = RotationMatrix(np.column_stack([x, y, z]))
    rpy_deg = np.rad2deg(RollPitchYaw(R).vector()).tolist()
    return {
        "name": name,
        "width": width,
        "height": height,
        "fov_y_deg": float(fov_y_deg),
        "depth": False,
        "background": [0.97, 0.97, 0.97, 1.0],
        "X_PB": {"base_frame": "world", "translation": p_WC.tolist(), "rotation": {"_tag": "!Rpy", "deg": rpy_deg}},
    }


def _cameras_yaml(cameras: dict[str, dict] | None) -> str:
    if not cameras:
        return ""
    lines = ["cameras:"]
    for name, c in cameras.items():
        X = c["X_PB"]
        lines += [
            f"    {name}:",
            f"        name: {c['name']}",
            f"        width: {c['width']}",
            f"        height: {c['height']}",
            f"        focal: !FovDegrees {{ y: {c['fov_y_deg']} }}",
            f"        depth: {str(c['depth']).lower()}",
            f"        background: {{ rgba: {list(c['background'])} }}",
            "        renderer_class: !RenderEngineVtkParams",
            f"            default_clear_color: {list(c['background'][:3])}",
            "            lights:",
            "            - { type: directional, frame: world, direction: [0.4, 0.3, -1.0], intensity: 1.3 }",
            "            - { type: directional, frame: world, direction: [-0.5, -0.6, -0.4], intensity: 0.7 }",
            "        X_PB:",
            f"            base_frame: {X['base_frame']}",
            f"            translation: {X['translation']}",
            f"            rotation: !Rpy {{ deg: {X['rotation']['deg']} }}",
        ]
    return "\n".join(lines) + "\n"


def scenario_yaml(spec: ScenarioSpec, cameras: dict[str, dict] | None = None) -> str:
    return (
        scene_directives(spec)
        + "\n"
        + _cameras_yaml(cameras)
        + f"""
plant_config:
    time_step: {spec.time_step}
    contact_model: hydroelastic_with_fallback
    discrete_contact_approximation: sap
model_drivers:
    iiwa: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg
    wsg: !SchunkWsgDriver {{}}
"""
    )


# ---------------------------------------------------------------------------
# Scripted joint sources
# ---------------------------------------------------------------------------


class ScriptedJointSource(LeafSystem):
    """Outputs [q(t); v(t)] for a chain of prismatic joints following `path`.

    `axes` selects which coordinates of the path drive the joints (the cart has
    (x, y), obstacles have (x, y, z)).
    """

    def __init__(self, path: Path, axes=(0, 1, 2)):
        super().__init__()
        self._path = path
        self._axes = list(axes)
        n = len(self._axes)
        self.DeclareVectorOutputPort("desired_state", BasicVector(2 * n), self._calc)

    def _calc(self, context, output):
        t = context.get_time()
        q = self._path.position(t)[self._axes]
        v = self._path.velocity(t)[self._axes]
        output.SetFromVector(np.concatenate([q, v]))


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@dataclass
class Scene:
    spec: ScenarioSpec
    station: Diagram
    plant: MultibodyPlant
    scene_graph: SceneGraph
    object_info: ObjectInfo
    iiwa: ModelInstanceIndex
    wsg: ModelInstanceIndex
    cart: ModelInstanceIndex
    target: ModelInstanceIndex
    floor: ModelInstanceIndex
    obstacles: dict[str, ModelInstanceIndex] = field(default_factory=dict)

    # ---- handles ---------------------------------------------------------
    @property
    def gripper_body(self) -> RigidBody:
        return self.plant.GetBodyByName("body", self.wsg)

    @property
    def gripper_frame(self):
        return self.gripper_body.body_frame()

    @property
    def target_body(self) -> RigidBody:
        return self.plant.GetBodyByName(self.object_info.body_name, self.target)

    @property
    def cart_top_body(self) -> RigidBody:
        return self.plant.GetBodyByName("top", self.cart)

    def obstacle_body(self, name: str) -> RigidBody:
        return self.plant.GetBodyByName(name, self.obstacles[name])

    def body_sets(self) -> dict[str, set[BodyIndex]]:
        """Bodies grouped by role, for the monitor's collision bookkeeping."""
        p = self.plant
        robot = set(p.GetBodyIndices(self.iiwa)) | set(p.GetBodyIndices(self.wsg))
        robot.discard(p.GetBodyByName("iiwa_link_0", self.iiwa).index())  # fixed base
        fingers = {
            p.GetBodyByName("left_finger", self.wsg).index(),
            p.GetBodyByName("right_finger", self.wsg).index(),
        }
        return {
            "robot": robot,
            "fingers": fingers,
            "gripper": set(p.GetBodyIndices(self.wsg)),
            "target": {self.target_body.index()},
            "cart": set(p.GetBodyIndices(self.cart)),
            "floor": set(p.GetBodyIndices(self.floor)),
            "obstacle": {self.obstacle_body(n).index() for n in self.obstacles},
        }

    # ---- initial conditions -----------------------------------------------
    def set_initial_state(self, root_context, t: float = 0.0) -> None:
        """Robot at q0, cart/obstacles on their scripts, object resting on the cart."""
        p = self.plant
        ctx = p.GetMyMutableContextFromRoot(root_context)
        set_scripted_positions(self, ctx, t)
        p.SetPositions(ctx, self.iiwa, np.asarray(self.spec.q0, float))
        p.SetVelocities(ctx, self.iiwa, np.zeros(7))
        # Object: at its rest pose on the cart, already moving with the cart
        # (otherwise it would have to be dragged up to speed by friction).
        X_WU = target_upright_pose(self.spec, t)
        X_WO = X_WU @ self.object_info.X_UO
        X_WO = RigidTransform(X_WO.rotation(), X_WO.translation() + [0, 0, 0.001])
        p.SetFreeBodyPose(ctx, self.target_body, X_WO)
        v_cart = self.spec.cart.path.velocity(t)
        p.SetFreeBodySpatialVelocity(ctx, self.target_body, SpatialVelocity(np.zeros(3), v_cart))


class BodyRoles:
    """Which body plays which role, and what counts as a collision.

    A *collision* is any contact between a robot body (iiwa links 1-7, the WSG
    body, the fingers) and an obstacle, the cart, the floor or the target --
    except gripper (WSG body or fingers) <-> target, which is grasping.
    Self-contact is ignored.
    """

    def __init__(self, scene: Scene):
        self.plant = scene.plant
        sets = scene.body_sets()
        self.robot = {int(b) for b in sets["robot"]}
        self.fingers = {int(b) for b in sets["fingers"]}
        self.gripper = {int(b) for b in sets["gripper"]}
        self.category: dict[int, str] = {}
        for cat in ("obstacle", "cart", "floor", "target"):
            for b in sets[cat]:
                self.category[int(b)] = cat
        self.geom_body: dict = {}  # GeometryId -> body index (int)
        for b in range(self.plant.num_bodies()):
            body = self.plant.get_body(BodyIndex(b))
            for g in self.plant.GetCollisionGeometriesForBody(body):
                self.geom_body[g] = b

    def body_name(self, b: int) -> str:
        return self.plant.get_body(BodyIndex(b)).name()

    def classify(self, a: int, b: int) -> tuple[int, int, str] | None:
        """(robot body, other body, category) if the pair counts, else None."""
        if a in self.robot and b not in self.robot:
            robot, other = a, b
        elif b in self.robot and a not in self.robot:
            robot, other = b, a
        else:
            return None
        cat = self.category.get(other)
        if cat is None or (cat == "target" and robot in self.gripper):
            return None
        return robot, other, cat

    def classify_geometry_pair(self, id_A, id_B):
        a, b = self.geom_body.get(id_A), self.geom_body.get(id_B)
        return None if a is None or b is None else self.classify(a, b)

    def robot_collisions(self, query_object: QueryObject) -> list[tuple[str, str, str]]:
        """(robot body, other body, category) for every penetrating pair that counts."""
        out = []
        for pp in query_object.ComputePointPairPenetration():
            c = self.classify_geometry_pair(pp.id_A, pp.id_B)
            if c is not None:
                out.append((self.body_name(c[0]), self.body_name(c[1]), c[2]))
        return out

    def robot_clearance(self, query_object: QueryObject, max_distance: float = 1.0) -> tuple[float, float]:
        """(min distance robot <-> obstacles, min distance robot <-> cart/floor)."""
        d_obs, d_env = np.inf, np.inf
        for pair in query_object.ComputeSignedDistancePairwiseClosestPoints(max_distance):
            c = self.classify_geometry_pair(pair.id_A, pair.id_B)
            if c is None:
                continue
            if c[2] == "obstacle":
                d_obs = min(d_obs, pair.distance)
            elif c[2] in ("cart", "floor"):
                d_env = min(d_env, pair.distance)
        return d_obs, d_env


def set_scripted_positions(scene: Scene, plant_context, t: float) -> None:
    """Put the cart and the obstacles where their scripts say they are at time t."""
    p = scene.plant
    spec = scene.spec
    cart_q = spec.cart.path.position(t)[:2]
    cart_v = spec.cart.path.velocity(t)[:2]
    p.SetPositions(plant_context, scene.cart, cart_q)
    p.SetVelocities(plant_context, scene.cart, cart_v)
    for o in spec.obstacles:
        path = o.get_path()
        p.SetPositions(plant_context, scene.obstacles[o.name], path.position(t))
        p.SetVelocities(plant_context, scene.obstacles[o.name], path.velocity(t))


def target_upright_pose(spec: ScenarioSpec, t: float) -> RigidTransform:
    """Scripted pose of the upright-object frame U at time t (object rides the cart)."""
    p_WC = spec.cart.path.position(t)
    return RigidTransform(RotationMatrix.MakeZRotation(spec.target.yaw), p_WC)


def _set_defaults(plant: MultibodyPlant, spec: ScenarioSpec, cart: ModelInstanceIndex, obstacles) -> None:
    """Default joint positions = script at t=0 (so contexts start consistent)."""
    q = spec.cart.path.position(0.0)
    plant.GetJointByName("cart_x", cart).set_default_translation(q[0])
    plant.GetJointByName("cart_y", cart).set_default_translation(q[1])
    for o in spec.obstacles:
        p0 = o.get_path().position(0.0)
        for k, ax in enumerate("xyz"):
            plant.GetJointByName(f"{o.name}_{ax}", obstacles[o.name]).set_default_translation(p0[k])


def build_station(spec: ScenarioSpec, meshcat: Meshcat | None = None, cameras: dict[str, dict] | None = None) -> Scene:
    """The simulated HardwareStation for `spec` (see the module docstring).

    `cameras` (name -> look_at_camera(...)) adds RGB cameras; the station then has
    output ports `<name>.rgb_image` (see scripts/render.py).
    """
    scenario = LoadScenario(data=scenario_yaml(spec, cameras))
    handles: dict = {}

    def prefinalize(parser: Parser) -> None:
        plant = parser.plant()
        handles["cart"] = plant.GetModelInstanceByName("cart")
        handles["obstacles"] = {o.name: plant.GetModelInstanceByName(o.name) for o in spec.obstacles}
        _set_defaults(plant, spec, handles["cart"], handles["obstacles"])

    def prebuild(builder: DiagramBuilder) -> None:
        plant = builder.GetSubsystemByName("plant")
        src = builder.AddNamedSystem("cart_script", ScriptedJointSource(spec.cart.path, axes=(0, 1)))
        builder.Connect(src.get_output_port(), plant.get_desired_state_input_port(handles["cart"]))
        for o in spec.obstacles:
            src = builder.AddNamedSystem(f"{o.name}_script", ScriptedJointSource(o.get_path()))
            builder.Connect(
                src.get_output_port(), plant.get_desired_state_input_port(handles["obstacles"][o.name])
            )

    station = MakeHardwareStation(
        scenario, meshcat, parser_prefinalize_callback=prefinalize, prebuild_callback=prebuild
    )
    plant = station.GetSubsystemByName("plant")
    return Scene(
        spec=spec,
        station=station,
        plant=plant,
        scene_graph=station.GetSubsystemByName("scene_graph"),
        object_info=get_object(spec.target.object),
        iiwa=plant.GetModelInstanceByName("iiwa"),
        wsg=plant.GetModelInstanceByName("wsg"),
        cart=handles["cart"],
        target=plant.GetModelInstanceByName("target"),
        floor=plant.GetModelInstanceByName("floor"),
        obstacles=handles["obstacles"],
    )


def build_kinematic_diagram(spec: ScenarioSpec, time_step: float = 0.0) -> tuple[RobotDiagram, Scene]:
    """Same bodies and geometry, no drivers: for collision checks at a chosen (q, t).

    Returns (diagram, scene); `scene.station` is the diagram.  Typical use:

        diagram, kin = build_kinematic_diagram(spec)
        ctx = diagram.CreateDefaultContext()
        world.set_kinematic_state(kin, ctx, t, q)        # see world.py
        qo = kin.scene_graph.get_query_output_port().Eval(kin.scene_graph.GetMyContextFromRoot(ctx))
        roles = BodyRoles(kin)
        roles.robot_collisions(qo)      # [] if q is collision-free at time t
        roles.robot_clearance(qo)       # (distance to obstacles, distance to cart/floor)
    """
    rb = RobotDiagramBuilder(time_step=time_step)
    plant, scene_graph = rb.plant(), rb.scene_graph()
    parser = rb.parser()
    ConfigureParser(parser)
    scenario = LoadScenario(data=scene_directives(spec))
    ProcessModelDirectives(ModelDirectives(directives=scenario.directives), parser)
    cart = plant.GetModelInstanceByName("cart")
    obstacles = {o.name: plant.GetModelInstanceByName(o.name) for o in spec.obstacles}
    _set_defaults(plant, spec, cart, obstacles)
    plant.Finalize()
    # The object rests on the cart, the cart rides just above the floor, and the
    # gripper has to touch the object to grasp it: none of those pairs is a
    # collision a planner should care about (same rule as the monitor).
    target_model = plant.GetModelInstanceByName("target")
    floor_model = plant.GetModelInstanceByName("floor")
    wsg_model = plant.GetModelInstanceByName("wsg")
    iiwa_model = plant.GetModelInstanceByName("iiwa")
    # The iiwa's box collision model leaves links 5 and 7 ~1 cm apart at ordinary
    # wrist angles (link 6 sits between them), which would defeat any minimum
    # distance constraint; the real links cannot touch there.
    link = lambda name: GeometrySet(list(plant.GetCollisionGeometriesForBody(plant.GetBodyByName(name, iiwa_model))))

    def geoms(model):
        ids = []
        for b in plant.GetBodyIndices(model):
            ids += list(plant.GetCollisionGeometriesForBody(plant.get_body(b)))
        return GeometrySet(ids)

    scene_graph.collision_filter_manager().Apply(
        CollisionFilterDeclaration()
        .ExcludeBetween(geoms(cart), geoms(target_model))
        .ExcludeBetween(geoms(cart), geoms(floor_model))
        .ExcludeBetween(geoms(wsg_model), geoms(target_model))
        .ExcludeBetween(link("iiwa_link_5"), link("iiwa_link_7"))
    )
    diagram = rb.Build()
    scene = Scene(
        spec=spec,
        station=diagram,
        plant=plant,
        scene_graph=scene_graph,
        object_info=get_object(spec.target.object),
        iiwa=plant.GetModelInstanceByName("iiwa"),
        wsg=plant.GetModelInstanceByName("wsg"),
        cart=cart,
        target=plant.GetModelInstanceByName("target"),
        floor=plant.GetModelInstanceByName("floor"),
        obstacles=obstacles,
    )
    return diagram, scene


def build_robot_diagram(time_step: float = 0.0) -> RobotDiagram:
    """iiwa + WSG only (collision geometry included): the controllers' kinematic model."""
    rb = RobotDiagramBuilder(time_step=time_step)
    parser = rb.parser()
    ConfigureParser(parser)
    scenario = LoadScenario(data="directives:" + robot_directives())
    ProcessModelDirectives(ModelDirectives(directives=scenario.directives), parser)
    rb.plant().Finalize()
    return rb.Build()
