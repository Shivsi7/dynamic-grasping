"""Run episodes and batches; collect `EpisodeResult` rows.

    result = run_episode(spec, NaiveInterceptController(), meshcat=get_meshcat())
    df = run_batch(specs, {"naive": NaiveInterceptController, "hold": HoldController}, out_dir="results/dev")

Outcome of an episode (in order of precedence):
    collision         a robot body touched an obstacle / the cart / the floor / the target (non-finger)
    success           object lifted and held for HOLD_TIME without any collision
    dropped           lifted, then lost
    timeout           nothing of the above by `spec.duration`
`success` and `collided` are also reported separately, so "success despite a
collision" is visible (outcome == "collision" and success == True).

Latency: by default (`latency="measured"`) a controller's commands take effect
only after its measured compute time has elapsed in simulated time, so slow
planning has consequences (see controller.py).  Use `latency=<seconds>` for a
deterministic fixed delay, `compute_scale` to model faster/slower hardware, and
`latency="none"` for the old instantaneous behaviour.  Run batches one process
per core at most: measured latency is wall time.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path as FilePath
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydrake.all import DiagramBuilder, Meshcat, Simulator

from dyngrasp.controller import Controller, ControllerSystem, RobotModel
from dyngrasp.monitor import EpisodeMonitor
from dyngrasp.scenario import ScenarioSpec
from dyngrasp.scene import Scene, build_station
from dyngrasp.world import WorldModel

ControllerFactory = Callable[[], Controller]


@dataclass
class EpisodeResult:
    scenario: str
    controller: str
    outcome: str
    success: bool
    collided: bool
    t_grasp: float | None
    t_lift: float | None
    t_success: float | None
    t_first_collision: float | None
    collision_with: str  # "obstacle_0,cart"
    collision_category: str  # "obstacle" | "cart" | ...
    min_clearance_obstacles: float
    min_clearance_obstacles_t: float
    min_clearance_environment: float
    dropped: bool
    t_end: float
    sim_wall_s: float
    n_cycles: int
    compute_mean_ms: float
    compute_max_ms: float
    compute_p95_ms: float
    cmd_speed_ratio_max: float = 0.0
    cmd_accel_max: float = 0.0
    cmd_accel_p99: float = 0.0
    cmd_torque_ratio_max: float = 0.0  # peak |joint torque| / torque limit of the commanded motion
    latency_mode: str = "none"
    n_rejected_commands: int = 0
    delay_max_ms: float = 0.0
    held_at_end: bool = False
    dropped_after_success: bool = False
    obj_speed_max_after_grasp: float = 0.0
    slip_max: float = 0.0
    scenario_summary: dict[str, Any] = field(default_factory=dict)
    log: dict[str, np.ndarray] | None = None
    controller_log: list[dict[str, Any]] | None = None
    collisions: list[dict[str, Any]] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k not in ("log", "controller_log", "collisions", "scenario_summary")}
        d.update(self.scenario_summary)
        return d


@dataclass
class EpisodeHandles:
    """Everything built for one episode, for notebooks / debugging."""

    scene: Scene
    world: WorldModel
    robot: RobotModel
    controller_system: ControllerSystem
    monitor: EpisodeMonitor
    diagram: Any
    simulator: Simulator


_robot_model: RobotModel | None = None


def shared_robot_model() -> RobotModel:
    global _robot_model
    if _robot_model is None:
        _robot_model = RobotModel()
    return _robot_model


def build_episode(
    spec: ScenarioSpec,
    controller: Controller,
    meshcat: Meshcat | None = None,
    stop_on_collision: bool = True,
    monitor_hz: float = 100.0,
    cameras: dict | None = None,
    latency: str | float = "measured",
    compute_scale: float = 1.0,
    stitch_vtol: float = 0.03,
) -> EpisodeHandles:
    scene = build_station(spec, meshcat, cameras)
    world = WorldModel(spec)
    robot = shared_robot_model()

    builder = DiagramBuilder()
    station = builder.AddNamedSystem("station", scene.station)
    ctrl = builder.AddNamedSystem(
        "controller", ControllerSystem(
            controller, world, robot, scene.gripper_body.index(), spec.q0, latency=latency, compute_scale=compute_scale,
            stitch_vtol=stitch_vtol,
        ),
    )
    builder.Connect(station.GetOutputPort("iiwa.position_measured"), ctrl.GetInputPort("iiwa.position_measured"))
    builder.Connect(station.GetOutputPort("iiwa.velocity_estimated"), ctrl.GetInputPort("iiwa.velocity_estimated"))
    builder.Connect(station.GetOutputPort("wsg.state_measured"), ctrl.GetInputPort("wsg.state_measured"))
    builder.Connect(station.GetOutputPort("body_poses"), ctrl.GetInputPort("body_poses"))
    builder.Connect(ctrl.GetOutputPort("iiwa.position"), station.GetInputPort("iiwa.position"))
    builder.Connect(ctrl.GetOutputPort("wsg.position"), station.GetInputPort("wsg.position"))

    monitor = builder.AddNamedSystem("monitor", EpisodeMonitor(scene, monitor_hz, stop_on_collision))
    builder.Connect(station.GetOutputPort("contact_results"), monitor.GetInputPort("contact_results"))
    builder.Connect(station.GetOutputPort("body_poses"), monitor.GetInputPort("body_poses"))
    builder.Connect(station.GetOutputPort("query_object"), monitor.GetInputPort("query_object"))
    builder.Connect(station.GetOutputPort("wsg.state_measured"), monitor.GetInputPort("wsg.state_measured"))
    builder.Connect(station.GetOutputPort("iiwa.position_measured"), monitor.GetInputPort("iiwa.position_measured"))

    diagram = builder.Build()
    simulator = Simulator(diagram)
    root = simulator.get_mutable_context()
    scene.set_initial_state(root, 0.0)
    return EpisodeHandles(scene, world, robot, ctrl, monitor, diagram, simulator)


def advance(h: EpisodeHandles, t_final: float, chunk: float = 0.05) -> float:
    """AdvanceTo(t_final) in chunks, stopping early when the monitor says the episode is over."""
    sim = h.simulator
    while True:
        t = sim.get_context().get_time()
        if h.monitor.done or t >= t_final - 1e-9:
            return t
        sim.AdvanceTo(min(t + chunk, t_final))


def run_episode(
    spec: ScenarioSpec,
    controller: Controller,
    meshcat: Meshcat | None = None,
    realtime: bool | None = None,
    record: bool = False,
    stop_on_collision: bool = True,
    keep_log: bool = True,
    monitor_hz: float = 100.0,
    settle_after_success: float = 0.0,
    latency: str | float = "measured",
    compute_scale: float = 1.0,
) -> EpisodeResult:
    """Simulate one episode of `controller` on `spec` and score it."""
    h = build_episode(
        spec, controller, meshcat, stop_on_collision, monitor_hz, latency=latency, compute_scale=compute_scale
    )
    sim = h.simulator
    if realtime is None:
        realtime = meshcat is not None
    sim.set_target_realtime_rate(1.0 if realtime else 0.0)
    if meshcat is not None and record:
        meshcat.StartRecording(set_visualizations_while_recording=False)
    t0 = time.perf_counter()
    sim.Initialize()
    t_end = advance(h, spec.duration)
    if h.monitor.done_reason == "success" and settle_after_success > 0 and t_end < spec.duration:
        h.monitor.done = False
        t_end = advance(h, min(spec.duration, t_end + settle_after_success))
    wall = time.perf_counter() - t0
    if meshcat is not None and record:
        meshcat.StopRecording()
        meshcat.PublishRecording()
    return summarize(spec, controller, h, t_end, wall, keep_log)


def summarize(spec, controller, h: EpisodeHandles, t_end: float, wall: float, keep_log: bool) -> EpisodeResult:
    m = h.monitor
    collided = len(m.collisions) > 0
    success = m.t_success is not None
    if collided:
        outcome = "collision"
    elif success:
        outcome = "success"
    elif m.dropped:
        outcome = "dropped"
    else:
        outcome = "timeout"
    first = m.collisions[0] if collided else None
    stats = h.controller_system.compute_stats()
    return EpisodeResult(
        scenario=spec.name,
        controller=controller.name,
        outcome=outcome,
        success=success,
        collided=collided,
        t_grasp=m.t_grasp,
        t_lift=m.t_lift,
        t_success=m.t_success,
        t_first_collision=first.t if first else None,
        collision_with=",".join(sorted({c.other_body for c in m.collisions})),
        collision_category=first.category if first else "",
        min_clearance_obstacles=float(m.min_clearance_obstacles),
        min_clearance_obstacles_t=float(m.min_clearance_obstacles_t),
        min_clearance_environment=float(m.min_clearance_environment),
        dropped=m.dropped,
        held_at_end=m.held_at_end,
        dropped_after_success=m.dropped_after_success,
        obj_speed_max_after_grasp=m.obj_speed_max_after_grasp,
        slip_max=m.slip_max,
        t_end=float(t_end),
        sim_wall_s=float(wall),
        scenario_summary=spec.summary(),
        log=m.log.as_arrays() if keep_log else None,
        controller_log=h.controller_system.log if keep_log else None,
        collisions=[asdict(c) for c in m.collisions],
        **stats,
    )


def run_batch(
    specs: list[ScenarioSpec],
    controllers: dict[str, ControllerFactory],
    out_dir: str | FilePath | None = None,
    stop_on_collision: bool = True,
    verbose: bool = True,
    save_logs: bool = False,
    latency: str | float = "measured",
    compute_scale: float = 1.0,
) -> pd.DataFrame:
    """Every controller on every scenario (fresh controller instance per episode).

    Writes `results.csv` (one row per episode) incrementally to `out_dir`, plus
    `scenarios.yaml`; with `save_logs`, also `logs/<scenario>__<controller>.npz`.
    """
    from dyngrasp.scenario import save_scenario_set

    rows = []
    out = FilePath(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        save_scenario_set(specs, out / "scenarios.yaml")
        if save_logs:
            (out / "logs").mkdir(exist_ok=True)
    total = len(specs) * len(controllers)
    k = 0
    for spec in specs:
        for cname, factory in controllers.items():
            k += 1
            ctrl = factory()
            res = run_episode(
                spec, ctrl, meshcat=None, stop_on_collision=stop_on_collision, keep_log=save_logs,
                latency=latency, compute_scale=compute_scale,
            )
            rows.append(res.row())
            if verbose:
                print(
                    f"[{k}/{total}] {spec.name:>14s} {cname:>8s}: {res.outcome:9s}"
                    f" t_grasp={_fmt(res.t_grasp)} clearance={res.min_clearance_obstacles:6.3f}"
                    f" compute={res.compute_mean_ms:5.2f}ms  ({res.sim_wall_s:4.1f}s wall)"
                )
            if out:
                pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
                if save_logs and res.log is not None:
                    np.savez_compressed(out / "logs" / f"{spec.name}__{cname}.npz", **res.log)
                    (out / "logs" / f"{spec.name}__{cname}.json").write_text(
                        json.dumps({"collisions": res.collisions, "controller_log": res.controller_log}, default=float)
                    )
    return pd.DataFrame(rows)


def _fmt(x):
    return "  -  " if x is None else f"{x:5.2f}"


def aggregate(df: pd.DataFrame, by=("controller",)) -> pd.DataFrame:
    """Success / collision rates and timing per group."""
    df = df.assign(_clearance=df["min_clearance_obstacles"].replace(np.inf, np.nan))
    g = df.groupby(list(by))
    return pd.DataFrame(
        {
            "episodes": g.size(),
            "success_rate": g["success"].mean(),
            "collision_rate": g["collided"].mean(),
            "clean_success_rate": g.apply(lambda d: ((d["success"]) & (~d["collided"])).mean(), include_groups=False),
            "t_grasp_mean": g["t_grasp"].mean(),
            "min_clearance_mean": g["_clearance"].mean(),
            "compute_mean_ms": g["compute_mean_ms"].mean(),
            "compute_p95_ms": g["compute_p95_ms"].mean(),
        }
    )
