"""The common controller API, and its wrapper as a Drake system.

A controller is a plain Python object:

    class MyController(Controller):
        name = "mine"
        period = 0.01                      # seconds between update() calls

        def reset(self, world, robot, state): ...
        def update(self, state: RobotState, world: WorldModel) -> Command: ...

`update` returns joint *positions* (Command.q), joint *velocities*
(Command.v, integrated by the wrapper at the controller's period) or a whole
time-indexed `JointTrajectory` (Command.trajectory, sampled by the wrapper at
every tick) for the iiwa, plus a WSG opening.  The wrapper feeds the station's
`iiwa.position` and `wsg.position` ports and records the wall-clock time of
every `update` call ("computation per cycle").

Latency model
-------------
The simulator waits for `update`, so without a latency model a controller
that thinks for 500 ms acts as if it had answered instantly.  With
`latency="measured"` (the harness default) a command computed at sim time t
in w seconds of wall time takes effect at t + w * compute_scale, and the
controller is not called again before then (it is busy).  Meanwhile the
previously accepted command keeps running: a trajectory keeps being sampled,
a velocity keeps being integrated, a position is held.  A fast controller
(w < period) therefore sees one period of delay; a planner sees its real
planning time.  `latency=<float>` uses a fixed delay instead (deterministic,
machine-independent); `latency="none"` is the old zero-delay behaviour.

A trajectory that arrives late, i.e. that does not agree with what the robot
is already executing at the moment it takes effect (max joint difference >
`stitch_tol`, or a joint velocity difference > `stitch_vtol`: splicing in a
plan that has already started moving is a velocity step, i.e. an unbounded
acceleration), is rejected and the previous command keeps running.  Planners
should therefore plan from where the current trajectory will be once they are
done (t + planning budget) and keep the current trajectory as the prefix.

`reset` is not timed: use it for scene-independent setup (building models),
not for planning.

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
from dyngrasp.trajectory import JointTrajectory
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
    trajectory: JointTrajectory | None = None  # the trajectory being executed, if any
    n_rejected: int = 0  # trajectories rejected so far for arriving late


@dataclass
class Command:
    q: np.ndarray | None = None  # desired joint positions
    v: np.ndarray | None = None  # or desired joint velocities (integrated at `period`)
    trajectory: JointTrajectory | None = None  # or a time-indexed plan, sampled every tick
    gripper: float = WSG_OPEN  # commanded WSG opening (WSG_OPEN .. WSG_CLOSED)
    info: dict[str, Any] = field(default_factory=dict)  # anything to log (phase, ...)


class Controller(ABC):
    name: str = "controller"
    period: float = 0.01

    def reset(self, world: WorldModel, robot: "RobotModel", state: RobotState) -> None:
        """Called once before the episode starts."""

    @abstractmethod
    def update(self, state: RobotState, world: WorldModel) -> Command: ...

    def close(self) -> None:
        """Called once after the episode ends (frees what reset set up, e.g. worker processes)."""


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
        self.tau_limits = self.plant.GetEffortUpperLimits()[:NQ]  # iiwa datasheet joint torques (N m)
        self._id_forces = None

    def joint_torques(self, Q: np.ndarray, dt: float) -> np.ndarray:
        """Joint torques the iiwa needs to follow positions Q (sampled every dt, finite
        differences) with the gripper attached (no payload), by inverse dynamics:
        tau = M(q) qdd + C(q, qd) qd - tau_gravity(q).  One row per interior sample."""
        from pydrake.all import MultibodyForces

        if self._id_forces is None:
            self._id_forces = MultibodyForces(self.plant)
        Q = np.asarray(Q, float)
        if len(Q) < 3:
            return np.zeros((0, NQ))
        V = np.diff(Q, axis=0) / dt
        A = np.diff(V, axis=0) / dt
        nv, s = self.plant.num_velocities(), self._iiwa_v_start
        out = np.empty((len(A), NQ))
        for k, (q, v, a) in enumerate(zip(Q[1:-1], 0.5 * (V[:-1] + V[1:]), A)):
            self.set_q(q)
            vv, vd = np.zeros(nv), np.zeros(nv)
            vv[s:s + NQ], vd[s:s + NQ] = v, a
            self.plant.SetVelocities(self.context, vv)
            self._id_forces.SetZero()
            self.plant.CalcForceElementsContribution(self.context, self._id_forces)
            out[k] = self.plant.CalcInverseDynamics(self.context, vd, self._id_forces)[s:s + NQ]
        return out

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

    def __init__(
        self,
        controller: Controller,
        world: WorldModel,
        robot: RobotModel,
        gripper_body_index: int,
        q0,
        latency: str | float = "none",
        compute_scale: float = 1.0,
        stitch_tol: float = 0.02,
        stitch_vtol: float = 0.03,  # rad/s; a 0.03 rad/s step over one 10 ms period is 3 rad/s^2
    ):
        super().__init__()
        self.controller = controller
        self.world = world
        self.robot = robot
        self._gripper_body_index = int(gripper_body_index)
        self.q0 = np.asarray(q0, float)
        if not (latency in ("none", "measured") or isinstance(latency, (int, float))):
            raise ValueError(f"latency must be 'none', 'measured' or seconds, got {latency!r}")
        self.latency = latency
        self.compute_scale = float(compute_scale)
        self.stitch_tol = float(stitch_tol)
        self.stitch_vtol = float(stitch_vtol)
        self.cycle_times: list[tuple[float, float]] = []  # (sim time, wall seconds)
        self.log: list[dict[str, Any]] = []  # per-cycle Command.info
        self._reset_done = False
        # Command bookkeeping (Python-side; the outputs are in the discrete state).
        self._active: Command | None = None  # command currently being executed
        self._pending: tuple[float, Command] | None = None  # (apply_at, command) still "in flight"
        self._busy_until = -np.inf
        self.n_rejected = 0
        self.delays: list[float] = []  # sim-time delay of every command
        self.q_cmd_log: list[np.ndarray] = []  # commanded positions, one per tick (motion statistics)

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
            trajectory=self._active.trajectory if self._active is not None else None,
            n_rejected=self.n_rejected,
        )

    def _delay(self, wall: float) -> float:
        if self.latency == "none":
            return 0.0
        if self.latency == "measured":
            return wall * self.compute_scale
        return float(self.latency)

    def _activate(self, t: float, cmd: Command, q_now: np.ndarray) -> None:
        """Make `cmd` the active command at time t, unless it is a trajectory that arrives too late."""
        if cmd.trajectory is not None and self.latency != "none":
            gap = float(np.max(np.abs(cmd.trajectory.value(t) - q_now)))
            active = self._active.trajectory if self._active is not None else None
            v_now = active.velocity(t) if active is not None else np.zeros_like(q_now)
            vgap = float(np.max(np.abs(cmd.trajectory.velocity(t) - v_now)))
            if gap > self.stitch_tol or vgap > self.stitch_vtol:
                self.n_rejected += 1
                self.log.append({"t": t, "event": "trajectory_rejected", "gap": gap, "vgap": vgap})
                return
        self._active = cmd

    def _update(self, context, discrete_state):
        state = self._read_state(context)
        t, p = state.t, self.controller.period
        if not self._reset_done:
            self.controller.reset(self.world, self.robot, state)
            self._reset_done = True

        # The q that the active command would output now (for the stitch check).
        q_now = self._q_from(self._active, t, state.q_cmd, p) if self._active is not None else state.q_cmd

        if self._pending is not None and t >= self._pending[0] - 1e-9:
            self._activate(t, self._pending[1], q_now)
            self._pending = None
            # the controller sees what is executing from now on
            state.trajectory = self._active.trajectory if self._active is not None else None
            state.n_rejected = self.n_rejected

        if t >= self._busy_until - 1e-9:
            t0 = time.perf_counter()
            cmd = self.controller.update(state, self.world)
            wall = time.perf_counter() - t0
            self.cycle_times.append((t, wall))
            delay = self._delay(wall)
            self.delays.append(delay)
            if cmd.info:
                self.log.append({"t": t, **cmd.info})
            if delay <= 0.0:
                self._activate(t, cmd, q_now)
            else:
                self._pending = (t + delay, cmd)
                self._busy_until = t + delay

        if self._active is None:
            q_cmd, wsg = state.q_cmd, state.wsg_cmd
        else:
            q_cmd = self._q_from(self._active, t, state.q_cmd, p)
            g = self._active.trajectory.gripper_at(t) if self._active.trajectory is not None else None
            wsg = self._active.gripper if g is None else g
        q_cmd = np.clip(q_cmd, self.robot.q_lower, self.robot.q_upper)
        self.q_cmd_log.append(q_cmd.copy())
        wsg = float(np.clip(wsg, WSG_CLOSED, WSG_OPEN))
        discrete_state.get_mutable_vector(self._state).SetFromVector(np.concatenate([q_cmd, [wsg]]))

    @staticmethod
    def _q_from(cmd: Command, t: float, q_prev: np.ndarray, period: float) -> np.ndarray:
        if cmd.trajectory is not None:
            return cmd.trajectory.value(t)
        if cmd.q is not None:
            return np.asarray(cmd.q, float)
        if cmd.v is not None:
            return q_prev + np.asarray(cmd.v, float) * period
        return q_prev

    def _out_q(self, context, output):
        output.SetFromVector(context.get_discrete_state(self._state).value()[:NQ])

    def _out_wsg(self, context, output):
        output.SetFromVector(context.get_discrete_state(self._state).value()[NQ:])

    # ---- statistics -------------------------------------------------------------
    def motion_stats(self) -> dict[str, float]:
        """Commanded joint motion: peak speed as a fraction of the iiwa's velocity limits,
        and peak acceleration (rad/s^2), by finite differences of the commanded positions."""
        if len(self.q_cmd_log) < 3:
            return {"cmd_speed_ratio_max": 0.0, "cmd_accel_max": 0.0, "cmd_accel_p99": 0.0, "cmd_torque_ratio_max": 0.0}
        q = np.array(self.q_cmd_log)
        dt = self.controller.period
        v = np.diff(q, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        amax = np.max(np.abs(a), axis=1)
        # torques: at 20 ms (finite-difference noise of the 10 ms log is not a torque)
        tau = self.robot.joint_torques(q[::2], 2 * dt)
        return {
            "cmd_torque_ratio_max": float(np.max(np.abs(tau) / self.robot.tau_limits)) if len(tau) else 0.0,
            "cmd_speed_ratio_max": float(np.max(np.abs(v) / self.robot.v_limits)),
            "cmd_accel_max": float(np.max(amax)),
            "cmd_accel_p99": float(np.percentile(amax, 99)),
        }


    def compute_stats(self) -> dict[str, float]:
        lat = {
            **self.motion_stats(),
            "latency_mode": str(self.latency),
            "n_rejected_commands": int(self.n_rejected),
            "delay_max_ms": float(max(self.delays) * 1e3) if self.delays else 0.0,
        }
        if not self.cycle_times:
            return {"n_cycles": 0, "compute_mean_ms": np.nan, "compute_max_ms": np.nan, "compute_p95_ms": np.nan, **lat}
        s = np.array([c[1] for c in self.cycle_times]) * 1e3
        return {
            "n_cycles": int(len(s)),
            "compute_mean_ms": float(s.mean()),
            "compute_max_ms": float(s.max()),
            "compute_p95_ms": float(np.percentile(s, 95)),
            **lat,
        }
