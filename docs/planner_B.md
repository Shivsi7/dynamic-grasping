# (B) Space-time RRT-Connect planner

Owner: Eli. Branch `planner-b-st-rrt`. Controller name in the scripts: `st_rrt`.
How we got here (measurements, bugs, fixes) is in `planner_B_history.md`.

## 1. The idea in one paragraph

The object rides past on a cart and obstacles move through the workspace, all on known
paths (`WorldModel` predicts every pose at any time t). So we plan in **space-time**: a
state is (q, t), joint angles at a time. A goal is "be at the pre-grasp configuration
q_g at the intercept time T, then run a fixed grasp motion (the *tail*)". RRT-Connect
searches (q, t) for a path from where the arm will be to any such goal that never
collides with anything *at the time it is there*. The arm executes a smooth curve
through the path, the tail grasps and lifts the object, and the arm ends somewhere safe
with it. The plan is redone every 0.5 s, and planning time costs simulated time.

## 2. Code map and reading order

Read in this order. Each file has a module docstring that explains it in more detail.

| # | file | what it does | start with |
|---|---|---|---|
| 1 | `dyngrasp/planning/st_rrt.py` | generic (q, t) RRT-Connect; no Drake, no robot | `SpaceTimeRRTConnect.plan`, `_extend`, `_shot`, `shortcut`, `plan_to_safe_state` |
| 2 | `dyngrasp/planning/collision.py` | is the robot (and a held object) clear at (q, t)? | `state_free`, `edge_free`, `hold_free`, `attach` |
| 3 | `dyngrasp/planning/goals.py` | intercept times, pre-grasp IK, the grasp tail | `generate_for`, `tail`, `tail_pose`, `self_motion` |
| 4 | `dyngrasp/trajectory.py` | the trajectory type the arm executes; Hermite smoothing | `JointTrajectory`, `hermite`, `knot_velocities` |
| 5 | `dyngrasp/planning/motion.py` | knots -> executed curve, and the checks it must pass | `MotionVerifier.smooth_verified`, `verify` |
| 6 | `dyngrasp/planning/escapes.py` | evasive moves; the post-grasp retreat | `EscapePlanner.primitive`, `retreat` |
| 7 | `dyngrasp/planning/config.py` | every tunable number, grouped; iiwa acceleration limits | `STRRTConfig`, `IIWA_ACCEL_REF` |
| 8 | `dyngrasp/planning/controller.py` | the replanning cycle that ties it all together | `update`, `_cycle` (section 3 below) |
| - | `dyngrasp/planning/variants.py` | named configurations used by the scripts | `VARIANTS` |
| - | `dyngrasp/grasps.py` | grasp candidates from the object's bounding box | `generate_grasps`, `prior_score` |

Shared harness pieces (B) relies on, in `dyngrasp/controller.py` (the controller wrapper
and its latency model), `dyngrasp/harness.py` (`run_episode`) and `dyngrasp/monitor.py`
(scoring). Tests: `tests/test_st_rrt.py` runs the RRT on 1-D / 2-D toy problems;
`tests/test_planner.py` runs the whole controller in Drake.

## 3. One replanning cycle (`SpaceTimeRRTController._cycle`)

```
update(state)                                   every 10 ms
  committed to a grasp?  (from commit_time before T to the end of the tail)  -> keep executing
  every replan_period:   _replan -> _cycle

_cycle
 1. _start            budget; start state (q_s, v_s) = executing trajectory at t_s = t + budget;
                      prefix = executing trajectory up to t_s
 2. _remaining_valid  is the executing plan still collision-free?  if not: _escape_primitive
                      (a cheap fallback, secured first)
 3. _generate_goals   more goals, earliest intercept time first, within a share of the budget
                      (or a whole "goal-work" cycle while waiting safely with no reachable goal)
 4. _plan_grasp       RRT-Connect to any reachable goal -> shortcut -> smooth + verify
                      -> Plan(prefix + path + tail)
 5. if no grasp plan and the executing one is unsafe: the fallback, else _escape_rrt
```

What the pieces are for:

* **Why start at t + budget.** The harness charges planning time (section 4.4). The new
  plan keeps the executing trajectory up to t_s, so it agrees with what the arm is doing
  when it arrives, as long as planning finished within the budget.
* **The budget** is 0.25 s normally, 1.5 s for the first plan, grows (up to 1 s) while
  waiting safely without a grasp plan, and is set to half the time left when a collision is
  predicted (`_start`). Then a hard escape gets the time it needs, instead of being cut off.
* **Margins.** `_replan` wraps `_cycle`: if the start state already violates the 3 cm
  obstacle margin, it plans with the largest smaller margin (1 cm, 0) it satisfies, rather
  than freeze.
* **Improve.** While a valid grasp plan executes, only goals with an earlier T are tried.

## 4. Key ideas

### 4.1 Space-time RRT-Connect (`st_rrt.py`)

* An edge (q_a, t_a) -> (q_b, t_b) is admissible if t_b > t_a and |q_b - q_a|_j <=
  v_max_j (t_b - t_a) for every joint, and it is collision-free along the way.
* The start tree grows forward in time; the goal tree grows backward from several goal
  roots (one per intercept time and arm posture), so one search considers all of them.
* Samples are drawn only in the space-time "double cone": states reachable from the start in
  time *and* from which some goal is still reachable in time (ST-RRT*'s sampling trick).
* `shortcut` replaces sub-paths with straight edges but keeps the knot times, so the arrival
  time at the goal does not change.
* `plan_to_safe_state`: single tree, grows until it reaches a state from which holding still
  is free (an escape).

### 4.2 Goals and the grasp tail (`goals.py`, `grasps.py`)

* **Intercept times** T span the window in which the object is within reach and the tail
  still fits in the episode (`candidate_times`).
* **Pre-grasp**: IK for the gripper 8 cm back along the grasp's approach direction at T.
  IK "families" are seeded with their own solution at the previous T, so each family is a
  consistent arm posture. If the pre-grasp or tail collides, the elbow's **self-motion**
  (the 1-D null space of the Jacobian: same gripper pose, different elbow) is searched,
  since obstacles mostly meet the elbow, not the hand.
* **The tail** (`tail`, `tail_pose`): ride along 0.3 s, descend 0.6 s while matching the
  object's velocity, close 0.4 s, lift 0.6 s while slowing to rest, hold 0.6 s. Minimum-jerk
  profiles (zero velocity and acceleration at the ends). It is a differential-IK rollout at
  10 ms, rejected if it breaks joint, velocity or acceleration limits or loses track of the
  pose, and executed exactly as rolled out. Two lift heights are tried (10 cm, 15 cm).
* **Grasp candidates** (`grasps.py`): top-down along both horizontal axes (and turned 180
  deg), the same tilted +-35 deg, and side grasps; placed as deep as the WSG geometry allows.
  Side grasps are off by default (rarely reachable, mostly fail physically); tilted grasps are
  off for the mustard bottle. Candidates are tried in order of past success, then a prior
  (approach from above or the robot's side first).

### 4.3 Collision checking (`collision.py`)

* Drake signed-distance queries on a kinematic copy of the scene, with every obstacle, the
  cart and the target placed at their predicted poses at time t.
* Margins: 3 cm from obstacles (absorbs the gaps between edge samples and the arm's tracking
  lag), 5 mm from the cart / floor, no penetration of the target (except by the gripper) or
  of the robot itself.
* Edges are sampled at least every 0.03 rad and 0.04 s, midpoint first.
* **Held object** (`attach`): from the moment the gripper closes, the target is moved with
  the gripper at its grasp pose, and it must clear the obstacles by 3 cm too. Without this,
  an obstacle swept through a box hanging below the gripper and knocked it out.

### 4.4 Planning time costs time (`dyngrasp/controller.py`, shared)

* `latency="measured"` (the default): a command computed in w s of wall time takes effect
  w s of sim time later; the previous command keeps running meanwhile.
* A late trajectory is rejected if, at the moment it takes effect, it disagrees with what
  the arm is doing, in position (`stitch_tol`, 0.02 rad) **or velocity** (`stitch_vtol`,
  0.03 rad/s). The velocity rule matters: splicing in a plan that already started moving is
  a velocity step, which was the source of 20-95 rad/s^2 acceleration spikes.
* `latency=<seconds>` gives a fixed, machine-independent delay; `"none"` makes planning free
  (used by the certification oracle).

### 4.5 Smooth motion within the iiwa's limits (`motion.py`, `trajectory.py`, `st_rrt.py`)

* **Executed curve.** A C1 cubic Hermite curve through the RRT's knots at the same times.
  Space-time plans cannot be retimed (e.g. by TOPP-RA): their timing is what avoids the
  obstacles.
* **Acceleration-aware RRT.** At every knot, |v_out - v_in|_j <= a_max_j min(h_in, h_out) / 6
  keeps the cubic's acceleration within a_max. Extensions *project* their velocity into that
  window (instead of rejecting most samples), and the trees are joined by exact single-edge
  "shots" that satisfy it at both ends. The start velocity is the executing trajectory's,
  and a goal's is the velocity the tail starts with, so there is no jump at either end.
* **Verification** (`MotionVerifier.verify`), before any plan is used: joint position limits,
  0.95x the velocity limits, the acceleration limits, joint torques (inverse dynamics of
  iiwa + gripper within 0.9x the datasheet torques), collisions. If position or collision
  fails, the curve is retried with long segments split (a long cubic that must end at a given
  velocity can bulge past a joint limit).
* **Limits.** KUKA publishes no acceleration limits, so they are fractions of a per-joint
  reference set from the literature (`IIWA_ACCEL_REF`, arXiv 2407.13423). The joints really
  differ: 176 N m of torque at joints 1-2, 110 at 3-5, 40 at 6-7, and joint 2 lifts the arm.

  | motion | velocity | acceleration (rad/s^2, joints 1-7) |
  |---|---|---|
  | transfer / escape (`a_max`, 0.6x ref.) | 0.5x datasheet | 9, 4.5, 6, 7.5, 9, 12, 12 |
  | grasp tail (`tail_a_max`, 0.8x ref.) | 0.9x | 12, 6, 8, 10, 12, 16, 16 |
  | holding the object (`a_max_holding`, 0.2x ref.) | 0.25x | 3, 1.5, 2, 2.5, 3, 4, 4 |

### 4.6 Staying safe (`escapes.py`, `controller.py`)

* **Escape primitives**: one smooth curve to a target at rest (brake, go home, swing the
  base / shoulder / elbow, fold the arm, random nearby targets), fastest first, accepted if
  it passes verification and holding the target is free for 1.5 s. Tried before goal work
  when the executing plan becomes unsafe; the random-tree escape is the fallback.
* **After the grasp** the arm must end somewhere safe (`finish_tail`, `_resolve_retreat`):
  1. holding the tail's end pose is free for 1.5 s: fine;
  2. otherwise a **planned retreat**: right after the lift, a gentle escape primitive with the
     object in hand, kept lifted, to a safe pose, appended to the tail and checked with it.
     It is planned only for the goal the RRT picks (planning it for every candidate made the
     first plan late);
  3. otherwise the end pose must be free for at least 1 s (time to replan and escape);
  4. otherwise the goal is dropped.
* **With the object in hand**, escapes use the gentle limits first, then the planner's
  (shaking the object beats hitting an obstacle).

## 5. Tools

| script | question it answers |
|---|---|
| `scripts/stress_planner.py --scenarios <set> --controllers st_rrt` | how does (B) do on many scenes? (simulated to the end of each scene, so post-grasp drops / collisions count) |
| `scripts/certify.py --results <results.csv> --scenarios <set>` | for each scene (B) did not grasp: was abstaining correct? exhaustive goal search, a time-optimal bound, and an "oracle" (same planner, 20x compute, planning free) at several acceleration limits |
| `scripts/compare_videos.py --scenarios <set> --names mix_0061 --controllers st_rrt_ckpt st_rrt` | side-by-side videos with live status, joint acceleration and object speed |
| `scripts/grasp_feasibility.py` | which grasp candidates are kinematically possible / hold the object |

The monitor (`dyngrasp/monitor.py`) scores `success` (lifted 5 cm, held 0.5 s, no collision,
still held at the end of the scene) and, after the grasp, `held_at_end`, `dropped_after_success`,
`obj_speed_max_after_grasp`, `slip_max`.
The wrapper reports `cmd_accel_max` and `cmd_torque_ratio_max` for every controller.

## 6. Current results

80 held-out mixed scenes (`results/stress_v3_heldout/scenarios.yaml`, never used for tuning),
measured latency, 4 workers on an M1 Pro, simulated to the end of each scene:

| | clean success, held at end | collisions | dropped after success | commanded accel (median / p90 / worst) | peak torque / limit |
|---|---|---|---|---|---|
| naive baseline (no avoidance) | 35 (of 80, at the first success) | 40 | - | - | - |
| **(B)** | **63** | **0** | **0** | 4.4 / 7.2 / 8.5 rad/s^2 | 0.46 |

Run-to-run variation from measured latency is about +-2 scenes (the run after the code cleanup:
62, the difference being mix_0061, which succeeds in about 2 of 3 runs).

The 17 scenes (B) does not grasp, by the last certification (before the latest changes):

| verdict | scenes |
|---|---|
| no feasible grasp at any acceleration | mix_0007, 0014, 0028, 0038, 0059, 0068 |
| grasps exist, no plan even without acceleration limits (oracle) | mix_0019, 0043, 0063, 0067, 0079 |
| possible only with more acceleration (one oracle success each; not reliable) | mix_0024, 0034, 0039, 0064, 0069, 0072 |

So at most 69 of 80 scenes are possible at all, and (B) grasps 63.

## 7. Limitations and open questions

1. **Goal coverage** is the bottleneck: most misses have no feasible goal. More grasp types
   (from grasp selection), a shorter tail at high cart speeds, collision-aware IK.
2. **Speed**: about 70 us per state check. A sphere-based checker or VAMP would allow more
   replans and bigger trees.
3. **Exact prediction**: with perfect knowledge of the future, the first valid plan stays
   valid and replanning only finds earlier intercepts. A prediction-error option in
   `WorldModel` would make "plans go stale" a measurable effect.
4. **Held object model**: rigid at its grasp pose (no slip); object vs. cart / floor is not
   checked.
5. **Velocity continuity** makes the bidirectional search harder (trees must meet in
   velocity too); much tighter acceleration limits fail often.
6. **To agree with the team**: compare (A) and (B) under `latency="measured"` with at most one
   episode per physical core; `reset()` is untimed (may build models, must not plan); the
   acceleration limits. (Scoring to the end of the scene is now the harness default.)
