# (C) Hybrid: space-time trajectory optimisation seeded by IK

Branch `planner-a-c-reactive-hybrid`. Controller name in the scripts: `hybrid`.

## 1. The idea in one paragraph

(C) plans the whole episode once, as two space-time trajectory optimisations (Drake's
`KinematicTrajectoryOptimization`). The approach has a free duration, ends at the grasp pose
of the object at the end time, and keeps clear of the obstacles where they are at each
sample time; its initial guess is a straight joint-space path to a collision-aware IK grasp
(if that fails, the same via a random configuration halfway). The second optimisation runs
from the grasp to the end of the scene: move with the object while the gripper closes, lift
10 cm, and hold, still keeping clear of the obstacles. Both are sent as one joint trajectory.

## 2. Code map and reading order

| # | file | what it does | start with |
|---|---|---|---|
| 1 | `dyngrasp/hybrid.py` | initial guesses, the two trajectory optimisations, execution | `update`, `_first_plan`, `_ik_grasps`, `_path`, `_plan`, `_plan_hold`, `_joint_trajectory` |
| 2 | `dyngrasp/collision.py` | per-category contexts and the world's full state at t for the clearance constraints | `CollisionWorld.context`, `full_state` |
| - | `dyngrasp/scenario.py` | `pillar_and_blocker`: a scene that needs a way around the pillar | `pillar_and_blocker` |

Tests: `tests/test_controllers.py` (in Drake, together with (A)).

## 3. One planning cycle (`HybridController.update`)

1. Wait at rest until `plan_budget` (2 s) before `t_enter - lead_time` (1 s before the
   object is reachable).
2. `_first_plan` for the start time t0 = now + budget: for each initial guess in turn,
   optimise the approach (`_plan`), then the close, lift and hold (`_plan_hold`); keep the
   first that gives both.
3. Send hold (until t0) + approach + close, lift and hold as one `JointTrajectory`, sampled
   by the wrapper every 5 ms. It holds its last pose afterwards.
4. If the plan arrived late (the wrapper rejected it), plan again with 1.5x the time the
   last plan took. If no plan is found, the arm stays where it is.

## 4. Key ideas

### 4.1 Space-time trajectory optimisation (`_plan`)

* A cubic B-spline q(s), s in [0, 1], with free duration T: time is t0 + s T.
* Constraints: start at rest at q0; pass the hover pose (10 cm above the grasp) and come
  straight down; end at the grasp pose *of time t0 + T*, moving with the object; clearance
  (4 cm to obstacles, 5 mm to the cart, floor and target) at 30 sample times.
* The clearance constraint (`MinimumDistanceLowerBoundConstraint`) is evaluated on the full
  kinematic plant, whose obstacle and cart joints are set to their predicted state at
  t0 + s T; their velocities give the derivative with respect to T, so moving the grasp later
  also moves every obstacle sample.
* Cost: duration plus path energy. Velocity bounds (0.6 x datasheet) and acceleration bounds
  (4 rad/s^2) on the plan. Solver: SNOPT.

### 4.2 Initial guesses (`_first_plan`, `_ik_grasps`, `_path`)

* All end at a hover pose and a grasp pose from collision-aware IK at the earliest grasp
  time (0.5 s steps) that has both. The paths may cut through obstacles; they only need to
  end at a valid grasp.
* Tried in turn, until one gives a feasible approach and hold:
  1. straight joint-space lines to the hover pose and down to the grasp;
  2. the same via a collision-free configuration halfway to the hover pose: the nearest
     (IK against the obstacles at that time) to the straight line's midpoint, then to 3
     random configurations around it (each joint moved by up to 1 rad), in case the
     straight line leads the optimiser into a bad local minimum;
  3. straight lines to the IK poses of the next grasp times, up to 2.
* `n_workers` > 1 runs the seeds in parallel processes (started in `reset`, so they cost no
  simulated time) and keeps the first to finish with a plan. Off by default: with four
  SNOPT processes at once each solve was 30-60 % slower, and the first seed usually works.

### 4.3 Close, lift and hold (`_plan_hold`)

* A second B-spline from the grasp at t0 + T to the end of the scene, fixed duration, knots
  every 0.5 s, starting in the approach's end state (position and velocity).
* Constraints: at the grasp pose of the moving object at 0.2 and 0.4 s (closing); the gripper
  at least 10 cm above the grasp height from 1.0 s on; clearance to the obstacles (4 cm) and
  the cart and floor (5 mm) every 0.4 s; at rest at the end. The target is left out of the
  clearance: it is in the hand.
* Cost: path energy, so the arm stays still unless an obstacle comes. Acceleration bound
  2 rad/s^2 (gentler than the approach).
* The object in the hand is not part of the collision model.

### 4.4 Planning time costs time (`update`, `_joint_trajectory`)

* With `latency="measured"` a command takes effect after the wall time it took to compute.
  (C) therefore plans ahead (`plan_budget`) and sends a time-indexed `JointTrajectory`, so
  its hold-then-move plan is valid whenever it arrives within the budget.
* The constraints' inputs q(s), qdot(s) and t are linear in the control points and the
  duration; their gradients are built with numpy and passed to Drake as AutoDiff
  (`InitializeAutoDiff`), which made one solve 3-4x faster than AutoDiff arithmetic in
  Python.
* No replanning: with perfect prediction the plan stays valid, and re-solving from the
  plan's own reference does not change it.

### 4.5 Simulation details that mattered

* (C) commands every 5 ms: the iiwa driver differentiates the position command every 5 ms,
  and a 10 ms command made its velocity reference alternate between 2v and 0.
* The box collision model's links 5 and 7 overlap past |q6| of about 1.75 rad (the joint
  goes to 2.09); the pair is filtered in the simulation as in the planning model.

## 5. Tools

| command | question it answers |
|---|---|
| `scripts/run_eval.py --scenarios scenarios/dev_v0.yaml --controllers hybrid` | success / collisions on a set, scored to the end of each scene |
| `scripts/render.py --preset pillar_and_blocker --controller hybrid --video --views close34 top` | an MP4 of one episode (needs `ffmpeg`) |
| `scripts/demo_scene.py --controller hybrid` | watch it in Meshcat |

## 6. Current results

**These numbers are from the version before the planned close/lift/hold and the new
initial guesses (rollout of (A) + IK, (A) closing and lifting); to be rerun.** On the four
report scenes (measured latency), the current version grasps and holds in `tall_blocker`,
`mix_0077` and `mix_0061`, and is hit while parked in `pillar_and_blocker` (7.7 s plan).

All runs scored to the end of each scene, 8 episodes in parallel on a 10-core machine, one
run per scene, with compute time costing simulated time (`measured`) and with planning time
free (`none`). Sets as in `planner_A.md`.

| set | scenes | success (measured / free) | collision | dropped | timeout | plans rejected (late) |
|---|---|---|---|---|---|---|
| dev | 20 | 11 / 13 | 5 / 4 | 3 / 3 | 1 / 0 | 2 |
| post grid | 48 | 36 / 34 | 0 / 0 | 2 / 4 | 10 / 10 | 33 |
| plank grid | 48 | 33 / 36 | 0 / 1 | 4 / 3 | 11 / 8 | 27 |
| pillar grid | 18 | 1 / 8 | 16 / 6 | 0 / 0 | 1 / 4 | 0 |
| held-out mixed | 80 | 29 / 34 | 9 / 2 | 33 / 38 | 9 / 6 | 18 |

Median first-plan compute 1-2.5 s (worst about 20 s, in the pillar scenes). With measured
latency, a plan that took longer than its 2 s budget is redone later, so the median grasp in
the grids is 1.7-2.4 s later than with planning time free. The plan itself stays near
3 rad/s^2 (99th percentile); the hand-off to (A) peaks at 200-240 rad/s^2 (torque 3.6-4x).

## 7. Limitations and open questions

1. **Compute**: around a pillar the approach needs hundreds of SNOPT iterations (several
   seconds); the hold adds 0.1-2 s. With measured latency the arm stays parked while (C)
   computes, and an obstacle can hit it there; without a plan it stays parked.
2. **The object in the hand** is not in the collision model of the hold.
3. **One plan**: no replanning, which is fine with perfect prediction; with prediction errors
   the plan would go stale.
