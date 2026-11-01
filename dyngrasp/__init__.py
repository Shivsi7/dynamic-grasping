"""dyngrasp: grasping a moving object among moving obstacles (6.4212 project).

Modules
-------
paths       scripted motion primitives (linear / parabolic), shared by sim and prediction
objects     YCB object registry: rest pose on the cart, nominal top-down grasp frame
scenario    ScenarioSpec dataclasses, random generator, named presets, YAML I/O
scene       builds the Drake HardwareStation for a ScenarioSpec (iiwa + WSG, cart, obstacles)
world       WorldModel: ground-truth "predicted" state of target and obstacles at any time
controller  the common controller API (RobotState -> Command) and its Drake wrapper
baselines   reference controllers: hold still, naive intercept (no avoidance)
monitor     EpisodeMonitor: collisions, clearance, grasp/lift detection
harness     run_episode / run_batch -> EpisodeResult rows (pandas)
viz         Meshcat helper (HEADLESS=1 disables it)
"""

from dyngrasp.controller import Command, Controller, RobotState, WSG_CLOSED, WSG_OPEN
from dyngrasp.harness import EpisodeResult, run_batch, run_episode
from dyngrasp.scenario import ScenarioSpec, generate_scenario_set, sample_scenario
from dyngrasp.world import WorldModel

__all__ = [
    "Command",
    "Controller",
    "RobotState",
    "WSG_CLOSED",
    "WSG_OPEN",
    "EpisodeResult",
    "run_batch",
    "run_episode",
    "ScenarioSpec",
    "generate_scenario_set",
    "sample_scenario",
    "WorldModel",
]
