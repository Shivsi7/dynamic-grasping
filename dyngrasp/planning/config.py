"""Configuration of the (B) space-time RRT controller, and the iiwa's acceleration limits.

All tunable numbers of the planner live in `STRRTConfig`, grouped as:
  motion limits | replanning cycle and budgets | safety margins | after the grasp |
  goal search | other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dyngrasp.planning.goals import TailParams
from dyngrasp.planning.st_rrt import STRRTParams

# Reference joint-acceleration limits of the iiwa (rad/s^2).  KUKA publishes none; these are the
# per-joint limits used for an iiwa in arXiv 2407.13423.  They follow the joint torques (176 N m at
# joints 1-2, 110 at 3-5, 40 at 6-7; joint 2 lifts the whole arm).  The planner uses fractions.
IIWA_ACCEL_REF = (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0)


def scaled_accel(fraction: float) -> tuple[float, ...]:
    """`fraction` x IIWA_ACCEL_REF, per joint."""
    return tuple(round(fraction * a, 3) for a in IIWA_ACCEL_REF)


def accel_vector(a, n: int = 7) -> np.ndarray | None:
    """A scalar or per-joint acceleration limit as a per-joint array (None stays None)."""
    return None if a is None else np.broadcast_to(np.asarray(a, float), (n,)).copy()


@dataclass
class STRRTConfig:
    # ---- motion limits ---------------------------------------------------------------
    # Joint velocities as fractions of the iiwa's datasheet limits.
    v_scale: float = 0.5  # transfer and escape motions (the RRT)
    tail_v_scale: float = 0.9  # the grasp tail (short and smooth, may move faster)
    # Joint accelerations (rad/s^2), per joint or one number for all joints.
    # a_max=None disables the acceleration-aware planning and smoothing altogether, including
    # the tail's limit (tail_a_max is then ignored) and the gentle holding limits.
    a_max: float | tuple | None = scaled_accel(0.6)  # 9, 4.5, 6, 7.5, 9, 12, 12
    tail_a_max: float | tuple = scaled_accel(0.8)  # 12, 6, 8, 10, 12, 16, 16
    # With the object (possibly) in hand, move gently so it is not flung.
    v_scale_holding: float = 0.25
    a_max_holding: float | tuple = scaled_accel(0.2)  # 3, 1.5, 2, 2.5, 3, 4, 4
    escape_fallback: bool = True  # holding: use the planner limits if no gentle escape exists
    torque_check: bool = True  # every smoothed plan must also be within the joint torques ...
    torque_margin: float = 0.9  # ... times this (payload, model error)

    # ---- replanning cycle and budgets (seconds of sim time) -----------------------------
    replan_period: float = 0.5  # time between replans
    budget: float = 0.25  # a replan's wall-clock budget = how far ahead it plans from
    initial_budget: float = 1.5  # the first plan (robot at rest; includes goal generation)
    max_budget: float = 1.0  # cap of the budget grown while safely waiting without a grasp plan
    commit_time: float = 0.4  # no replanning from this long before the intercept to the tail's end
    improve: bool = True  # while the current grasp plan is valid, look for an earlier intercept
    validate: bool = True  # re-check the remaining plan against the world model at each replan
    shortcut_share: float = 0.3  # share of the RRT's remaining budget spent on shortcutting
    # Wall-clock compute per unit of budget.  1 = real time.  The offline "oracle"
    # (scripts/certify.py) uses 20 with latency="none": the same planner with 20x the compute.
    wall_scale: float = 1.0

    # ---- safety margins ----------------------------------------------------------------
    margin_obstacle: float = 0.03  # clearance from obstacles (m)
    margin_env: float = 0.005  # clearance from the cart / conveyor / floor (m)
    # Smaller obstacle margins, used only when the start state already violates the margin.
    margin_fallbacks: tuple[float, ...] = (0.01, 0.0)
    joint_margin: float = 0.05  # sampled configurations stay this far inside the joint limits
    hold_horizon: float = 1.5  # a "safe" state: holding still there is free for this long

    # ---- after the grasp ---------------------------------------------------------------
    # The grasp must end somewhere safe.  If holding the tail's end pose is free for
    # `hold_horizon`, fine.  Otherwise a gentle *retreat* (object in hand, kept lifted) to a
    # safe pose is appended to the tail; failing that, the end pose must be free for at least
    # `post_grasp_hold` (time to replan and escape); otherwise the goal is rejected.
    retreat: bool = True
    retreat_budget: float = 0.1  # wall time (x wall_scale) per retreat search
    lazy_retreat: bool = True  # plan the retreat only for the goal the RRT picks (False: every goal)
    post_grasp_hold: float = 1.0
    check_held_object: bool = True  # once grasped, the object must clear the obstacles too

    # ---- goal search (goals.py) -----------------------------------------------------------
    grasp_set: str = "all"  # "all" (dyngrasp.grasps candidates) | "nominal" (objects.X_UG_grasp)
    # Grasp families used with grasp_set="all".  Side grasps are off: on this setup they are
    # rarely reachable and mostly fail physically (scripts/grasp_feasibility.py --execute).
    grasp_families: tuple[str, ...] = ("top", "top_tilt")
    dT: float = 0.25  # spacing of candidate intercept times
    max_goal_times: int = 16
    max_goals_per_T: int = 3
    max_ik_per_T: int = 10  # IK attempts per intercept time
    ik_restarts: int = 3  # random IK restarts per grasp until it first succeeds
    n_random_ik_seeds: int = 1
    swivel_steps: int = 6  # elbow self-motion search when a pre-grasp / tail collides
    max_tails: int = 3  # tail rollouts per IK solution (over elbow postures)
    goal_share: float = 0.6  # max share of a replan's budget spent generating goals
    goal_work_cap: float = 0.5  # wall time of a goal-only cycle (only while the current plan is safe)

    # ---- other -------------------------------------------------------------------------------
    rrt: STRRTParams = field(default_factory=STRRTParams)
    tail: TailParams = field(default_factory=TailParams)
    seed: int | None = None  # default: the scenario's seed
