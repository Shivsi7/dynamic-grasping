# (A) Reactive QP controller

Branch `planner-a-c-reactive-hybrid`. Controller name in the scripts: `reactive`.

## 1. The idea in one paragraph

(A) never plans. Every 10 ms it solves a small QP for the joint velocities that move the
gripper towards a velocity-matched grasp pose, with one constraint per nearby robot-world
pair that limits how fast the arm may approach it (a velocity damper). The pairs are taken
against where the obstacles *will be* over the next 0.2 s, so the arm keeps out of their
swept volume, not just their current position. It is fast (about one control period per
cycle) but local: it can get stuck behind an obstacle it would have to go around.

## 2. Code map and reading order

| # | file | what it does | start with |
|---|---|---|---|
| 1 | `dyngrasp/collision.py` | robot vs. world signed distances, witness points and Jacobians at a predicted (q, t) | `CollisionWorld.rows` |
| 2 | `dyngrasp/reactive.py` | the phase machine, the QP, and a kinematic rollout of the controller | `step`, `qp`, `rollout` |

Tests: `tests/test_controllers.py` (in Drake, together with (C)).

## 3. One control cycle (`ReactiveQPController.step`)

1. Phase: wait (until 1 s before the object is reachable) -> approach (10 cm above the
   grasp) -> descend -> close (0.4 s) -> lift (15 cm, riding along with the cart).
2. Desired gripper twist: P control on the pose error plus the object's velocity as
   feedforward, capped at 0.8 m/s.
3. `CollisionWorld.rows`: every robot-world pair closer than 25 cm, against the predicted
   world at t, t + 0.1 and t + 0.2 s.
4. QP (Drake `MathematicalProgram` + OSQP): track the twist, one slacked damper per pair,
   joint velocity limits (0.6 x datasheet) and damped joint position limits. The result is
   sent as joint velocities (`Command.v`).

## 4. Key ideas

### 4.1 Velocity dampers against the swept volume

* Per pair, `n^T J v >= -xi (phi - d_s) + n^T v_world`: the closer the pair, the slower it may
  approach; at `d_s` (4 cm to obstacles, 5 mm to the cart, floor and target) it may not.
* `n^T v_world` is the obstacle's (or the carrier's) velocity towards the robot, so an
  obstacle coming at the arm pushes it away before it is close.
* Three snapshots of the predicted world (now, +0.1 s, +0.2 s) turn the constraint set into
  the swept volume; with one snapshot the arm reacts too late to fast obstacles.

### 4.2 Always feasible

* Every damper has a slack, penalised (linear + quadratic) so heavily that it is zero unless
  constraints conflict. The QP always has a solution, and the slack used is logged.

### 4.3 Latency

* A cycle takes about one control period, so with `latency="measured"` its commands take
  effect about 10 ms late; the previous velocity keeps being integrated meanwhile.

### 4.4 Rollout

* `rollout` runs the same controller kinematically against the prediction (no simulation).
  (C) uses it as one of its initial guesses.

## 5. Tools

| command | question it answers |
|---|---|
| `scripts/run_eval.py --scenarios scenarios/dev_v0.yaml --controllers reactive` | success / collisions on a set, scored to the end of each scene |
| `scripts/render.py --preset tall_blocker --controller reactive --video` | an MP4 of one episode (needs `ffmpeg`) |
| `scripts/demo_scene.py --controller reactive` | watch it in Meshcat |

## 6. Current results

All runs: compute time costs simulated time (`latency="measured"`), scored to the end of each
scene, 8 episodes in parallel on a 10-core machine, one run per scene. Scenario sets in
`scenarios/`: `dev_v0.yaml` (20), `preset_grids.yaml` (48 post, 48 plank, 18 pillar scenes),
`heldout_mixed_s5000.yaml` ((B)'s 80 held-out scenes).

| set | scenes | success | collision | dropped | timeout |
|---|---|---|---|---|---|
| dev | 20 | 8 | 2 | 4 | 6 |
| post grid | 48 | 30 | 0 | 4 | 14 |
| plank grid | 48 | 28 | 0 | 4 | 16 |
| pillar grid | 18 | 2 | 0 | 0 | 16 |
| held-out mixed | 80 | 28 | 7 | 27 | 18 |

"dropped": lifted, then lost, or not lifted (5 cm) any more when the scene ends. Median peak
commanded joint acceleration 150-190 rad/s^2; median peak torque 2.0-2.3x the joint limits.
Compute per cycle about 2 ms on average; the slowest cycle of an episode about 20 ms (median).

## 7. Limitations and open questions

1. **Local minima**: behind an obstacle it has to go around (the pillar) it stops and waits;
   it grasps in 2 of the 18 pillar scenes.
2. **The lift never ends**: after the grasp it keeps lifting towards 15 cm above the grasp
   point while following the cart. As the cart leaves, the arm stretches out and the object
   sinks below 5 cm, or is shaken loose: 27 of the 80 held-out scenes end "dropped".
3. **No acceleration limits**: only velocities are limited. Phase switches (wait -> approach,
   close -> lift) are velocity steps: peaks of 100-300 rad/s^2, above the joint torques.
4. Velocity dampers keep a margin to obstacles but do not guarantee it (slacks).
