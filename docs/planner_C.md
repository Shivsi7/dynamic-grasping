# (C) Hybrid: space-time trajectory optimisation seeded by (A) and IK

Branch `planner-a-c-reactive-hybrid`. Controller name in the scripts: `hybrid`.

## 1. The idea in one paragraph

(C) plans the whole approach to the grasp once, as a space-time trajectory optimisation
(Drake's `KinematicTrajectoryOptimization`): free duration, the grasp pose of the object at
the end time, and clearance to the obstacles where they are at each sample time. The
optimisation starts from two initial guesses, a rollout of the reactive controller (A)
against the prediction and a straight path to a collision-aware IK grasp, and (C) executes
the cheapest feasible plan. At the end of the plan, (A) closes the gripper and lifts.

## 2. Code map and reading order

| # | file | what it does | start with |
|---|---|---|---|
| 1 | `dyngrasp/hybrid.py` | initial guesses, the trajectory optimisation, execution | `update`, `_first_plan`, `_ik_seeds`, `_plan`, `_joint_trajectory` |
| 2 | `dyngrasp/reactive.py` | (A): its `rollout` is one initial guess; it closes and lifts | `rollout`, `step` |
| 3 | `dyngrasp/collision.py` | per-category contexts and the world's full state at t for the clearance constraints | `CollisionWorld.context`, `full_state` |
| - | `dyngrasp/scenario.py` | `pillar_and_blocker`: a scene where (A) gets stuck | `pillar_and_blocker` |

Tests: `tests/test_controllers.py` (in Drake, together with (A)).

## 3. One planning cycle (`HybridController.update`)

1. Wait at rest until `plan_budget` (2 s) before (A) would start moving.
2. `_first_plan` for the start time t0 = now + budget: build the initial guesses, optimise
   each, keep the cheapest feasible plan.
3. Send hold (until t0) + plan + 0.3 s at the plan's final joint velocity as one
   `JointTrajectory`, sampled by the wrapper every 5 ms.
4. If the plan arrived late (the wrapper rejected it), plan again with 1.5x the time the
   last plan took. If no plan is found, (A) runs the rest of the episode.
5. At t0 + T, (A) takes over in its close phase.

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

### 4.2 Two initial guesses (`_first_plan`, `_ik_seeds`)

* **rollout**: (A) run kinematically against the prediction. Good when (A) succeeds; when
  it gets stuck (behind the pillar) the optimiser cannot recover from it.
* **ik**: a straight joint-space path to a hover pose and a grasp pose from collision-aware
  IK at the earliest grasp time (0.5 s steps) that has both. The path may cut through
  obstacles; it only needs to end at a valid grasp. If it does not optimise, the next grasp
  times, up to 3.
* Both are optimised and the cheaper plan is kept (`seed="rollout"` / `"ik"` keeps one).
  Planning time free, success rate rollout only -> both: dev set 80 -> 85 %, post grid
  81 -> 83 %, plank grid 79 -> 83 %, pillar grid 11 -> 61 %.

### 4.3 Planning time costs time (`update`, `_joint_trajectory`)

* With `latency="measured"` a command takes effect after the wall time it took to compute.
  (C) therefore plans ahead (`plan_budget`) and sends a time-indexed `JointTrajectory`, so
  its hold-then-move plan is valid whenever it arrives within the budget.
* No replanning: with perfect prediction the plan stays valid, and re-solving from the
  plan's own reference does not change it.

### 4.4 Simulation details that mattered

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

1. **Compute**: one SNOPT solve takes 0.3 s to 13 s; around a pillar it needs hundreds of
   iterations. With measured latency the arm stays parked while (C) computes, and in 16 of 18
   pillar scenes an obstacle hits it there. Planning time free, 8 of 18 succeed.
2. **The close and lift are (A)'s**: not planned, not collision-checked against the future,
   and the lift follows the cart forever (see `planner_A.md`): 33 of the 80 held-out scenes
   end "dropped".
3. **Hand-off to (A)**: a velocity step at the end of the plan (200-240 rad/s^2).
4. **One plan**: no replanning, which is fine with perfect prediction; with prediction errors
   the plan would go stale.
