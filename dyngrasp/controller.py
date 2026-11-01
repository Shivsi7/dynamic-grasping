"""The common controller API, and its wrapper as a Drake system.

A controller is a plain Python object:

    class MyController(Controller):
        name = "mine"
        period = 0.01                      # seconds between update() calls

        def reset(self, world, robot, state): ...
        def update(self, state: RobotState, world: WorldModel) -> Command: ...

`update` returns joint *positions* (Command.q) or joint *velocities*
(Command.v, integrated by the wrapper at the controller's period) for the
iiwa, plus a WSG opening.  The wrapper feeds the station's `iiwa.position` and
`wsg.position` ports and records the wall-clock time of every `update` call
("computation per cycle").

`RobotModel` is a kinematic model of iiwa + WSG for Jacobians, forward
kinematics and limits (controllers should not touch the simulation plant).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydrake.all import (
    AbstractValue,
    BasicVector,
    JacobianWrtVariable,
    LeafSystem,
    RigidTransform,
    RobotDiagram,
)

from dyngrasp.scene import build_robot_diagram
from dyngrasp.world import WorldModel

WSG_OPEN = 0.107
WSG_CLOSED = 0.0
NQ = 7


@dataclass
class RobotState:
    t: float
    q: np.ndarray  # measured joint positions (7)
    v: np.ndarray  # estimated joint velocities (7)
    X_WG: RigidTransform  # gripper (WSG body) pose, from the simulation
    wsg_width: float  # measured opening
    q_cmd: np.ndarray  # the last commanded joint positions (7)
    wsg_cmd: float  # the last commanded opening


@dataclass
class Command:
    q: np.ndarray | None = None  # desired joint positions
    v: np.ndarray | None = None  # or desired joint velocities (integrated at `period`)
    gripper: float = WSG_OPEN  # commanded WSG opening (WSG_OPEN .. WSG_CLOSED)
    info: dict[str, Any] = field(default_factory=dict)  # anything to log (phase, ...)


class Controller(ABC):
    name: str = "controller"
    period: float = 0.01

    def reset(self, world: WorldModel, robot: "RobotModel", state: RobotState) -> None:
        """Called once before the episode starts."""

    @abstractmethod
    def update(self, state: RobotState, world: WorldModel) -> Command: ...


class RobotModel:
    """Kinematics of iiwa + WSG (the controllers' own plant, with collision geometry)."""

    def __init__(self):
        self.diagram: RobotDiagram = build_robot_diagram()
        self.plant = self.diagram.plant()
        self.scene_graph = self.diagram.scene_graph()
        self.root_context = self.diagram.CreateDefaultContext()
        self.context = self.plant.GetMyMutableContextFromRoot(self.root_context)
        self.iiwa = self.plant.GetModelInstanceByName("iiwa")
        self.wsg = self.plant.GetModelInstanceByName("wsg")
        self.G = self.plant.GetFrameByName("body", self.wsg)
        self.W = self.plant.world_frame()
        self.q_lower = self.plant.GetPositionLowerLimits()[:NQ]
        self.q_upper = self.plant.GetPositionUpperLimits()[:NQ]
        self.v_limits = self.plant.GetVelocityUpperLimits()[:NQ]
        self._iiwa_v_start = self.plant.GetJointByName("iiwa_joint_1", self.iiwa).velocity_start()

    def set_q(self, q: np.ndarray) -> None:
        self.plant.SetPositions(self.context, self.iiwa, np.asarray(q, float))

    def fk(self, q: np.ndarray) -> RigidTransform:
        """X_WG at q."""
        self.set_q(q)
        return self.plant.CalcRelativeTransform(self.context, self.W, self.G)

    def jacobian(self, q: np.ndarray, p_GP=np.zeros(3)) -> np.ndarray:
        """6x7 spatial-velocity Jacobian of point P (in G) w.r.t. iiwa joint velocities, in W."""
        self.set_q(q)
        J = self.plant.CalcJacobianSpatialVelocity(
            self.context, JacobianWrtVariable.kV, self.G, np.asarray(p_GP, float), self.W, self.W
        )
        s = self._iiwa_v_start
        return J[:, s : s + NQ]

    def body_poses(self, q: np.ndarray) -> list[RigidTransform]:
        self.set_q(q)
        return [self.plant.EvalBodyPoseInWorld(self.context, self.plant.get_body(i)) for i in range(self.plant.num_bodies())]


class ControllerSystem(LeafSystem):
    """Runs a `Controller` at its period and drives the station's command ports.

    Inputs:  iiwa.position_measured (7), iiwa.velocity_estimated (7),
             wsg.state_measured (2), body_poses (abstract)
    Outputs: iiwa.position (7), wsg.position (1)
    """

    def __init__(self, controller: Controller, world: WorldModel, robot: RobotModel, gripper_body_index: int, q0):
        super().__init__()
        self.controller = controller
        self.world = world
        self.robot = robot
        self._gripper_body_index = int(gripper_body_index)
        self.q0 = np.asarray(q0, float)
        self.cycle_times: list[tuple[float, float]] = []  # (sim time, wall seconds)
        self.log: list[dict[str, Any]] = []  # per-cycle Command.info
        self._reset_done = False

        self._q_in = self.DeclareVectorInputPort("iiwa.position_measured", NQ)
        self._v_in = self.DeclareVectorInputPort("iiwa.velocity_estimated", NQ)
        self._wsg_in = self.DeclareVectorInputPort("wsg.state_measured", 2)
        self._poses_in = self.DeclareAbstractInputPort("body_poses", AbstractValue.Make([RigidTransform()]))

        # Discrete state: [q_cmd (7), wsg_cmd (1)].
        self._state = self.DeclareDiscreteState(np.concatenate([self.q0, [WSG_OPEN]]))
        self.DeclareVectorOutputPort("iiwa.position", BasicVector(NQ), self._out_q)
        self.DeclareVectorOutputPort("wsg.position", BasicVector(1), self._out_wsg)
        self.DeclarePeriodicDiscreteUpdateEvent(controller.period, 0.0, self._update)

    def _read_state(self, context) -> RobotState:
        xd = context.get_discrete_state(self._state).value()
        poses = self._poses_in.Eval(context)
        return RobotState(
            t=context.get_time(),
            q=np.array(self._q_in.Eval(context)),
            v=np.array(self._v_in.Eval(context)),
            X_WG=poses[self._gripper_body_index],
            wsg_width=float(self._wsg_in.Eval(context)[0]),
            q_cmd=xd[:NQ].copy(),
            wsg_cmd=float(xd[NQ]),
        )

    def _update(self, context, discrete_state):
        state = self._read_state(context)
        if not self._reset_done:
            self.controller.reset(self.world, self.robot, state)
            self._reset_done = True
        t0 = time.perf_counter()
        cmd = self.controller.update(state, self.world)
        self.cycle_times.append((state.t, time.perf_counter() - t0))

        if cmd.q is not None:
            q_cmd = np.asarray(cmd.q, float)
        elif cmd.v is not None:
            q_cmd = state.q_cmd + np.asarray(cmd.v, float) * self.controller.period
        else:
            q_cmd = state.q_cmd
        q_cmd = np.clip(q_cmd, self.robot.q_lower, self.robot.q_upper)
        wsg = float(np.clip(cmd.gripper, WSG_CLOSED, WSG_OPEN))
        if cmd.info:
            self.log.append({"t": state.t, **cmd.info})
        discrete_state.get_mutable_vector(self._state).SetFromVector(np.concatenate([q_cmd, [wsg]]))

    def _out_q(self, context, output):
        output.SetFromVector(context.get_discrete_state(self._state).value()[:NQ])

    def _out_wsg(self, context, output):
        output.SetFromVector(context.get_discrete_state(self._state).value()[NQ:])

    # ---- statistics -------------------------------------------------------------
    def compute_stats(self) -> dict[str, float]:
        if not self.cycle_times:
            return {"n_cycles": 0, "compute_mean_ms": np.nan, "compute_max_ms": np.nan, "compute_p95_ms": np.nan}
        s = np.array([c[1] for c in self.cycle_times]) * 1e3
        return {
            "n_cycles": int(len(s)),
            "compute_mean_ms": float(s.mean()),
            "compute_max_ms": float(s.max()),
            "compute_p95_ms": float(np.percentile(s, 95)),
        }
