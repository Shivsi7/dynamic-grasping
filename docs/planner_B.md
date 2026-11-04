# (B) Space-time RRT-Connect: design notes

Owner: Eli. Branch `planner-b-st-rrt`. Controller name in `run_eval.py`: `st_rrt`.

## What it does

```
every replan (0.5 s of sim time), until committed to a grasp:
  (q_s, t_s) = executing trajectory at t + budget        # plan from where we will be
  goals      += pre-grasp (q_g, T) for more intercept times T      (anytime, earliest first)
  if current plan still collision-free:  look only for an earlier T  (improve)
  else:                                  plan to any reachable goal
  RRT-Connect in (q, t)  ->  time-preserving shortcuts  ->  prefix + path + grasp tail
  no plan and current one unsafe  ->  escape to a state that is safe to hold
```

* **State space** (`planning/st_rrt.py`): nodes (q, t). An edge is admissible
  if time moves forward and |dq_j| <= v_max_j dt. The start tree grows forward
  in time, the goal tree backward from several goal roots (one per intercept
  time / arm posture). Samples are drawn inside the space-time double cone
  (reachable from the start in time, and can still reach some goal in time),
  which is ST-RRT*'s main sampling trick. The module is pure numpy and knows
  nothing about Drake; `tests/test_st_rrt.py` runs it on 1-D and 2-D toy
  problems (waiting for a gate to open, dodging a moving disc).
* **Collision checks** (`planning/collision.py`): Drake signed-distance
  queries on the scene's kinematic diagram at a hypothetical (q, t), with a 3
  cm margin to obstacles (it absorbs sampling between edge checks and the arm's
  tracking lag), 5 mm to the cart and floor, and self-collision. Edges are
  sampled every 0.03 rad / 0.04 s, midpoint first. About 70 µs per state on an
  M1 Pro.
* **Goals** (`planning/goals.py`): for each candidate T, IK for the hover pose
  (8 cm above the grasp at T), then a differential-IK rollout of the **tail**:
  descend while matching the cart's velocity, close, lift while slowing to a
  stop, hold. The tail is collision-checked in space-time like everything
  else, so an obstacle that crosses *during* the grasp rules that (T, q_g) out.
  When the pre-grasp or tail collides, the elbow's **self-motion** (the 1-D
  null space of the 6x7 Jacobian) is searched for a posture that is free. That
  is what most failures needed: obstacles crossing the corridor meet the
  elbow, not the hand.
* **Execution**: the controller returns a `JointTrajectory`, and the wrapper
  plays it at 100 Hz, including the gripper schedule. Between replans the
  controller just returns the executing trajectory.

## Planning time is not free (team decision, needs agreement)

The simulator waits for `update()`, so before this branch a 500 ms plan looked
instantaneous and only showed up in `compute_p95_ms`. `ControllerSystem` now
has a latency model, which is the harness default:

* `latency="measured"`: a command computed in w s of wall time takes effect w
  s of sim time later, and the controller is busy until then. The previous
  command keeps running meanwhile. A controller faster than its period gets
  exactly one period of delay (this changes the baselines and (A) by 10 ms).
* A trajectory that would make the arm jump when it finally takes effect is
  rejected (`n_rejected_commands`). For a planner this is the real cost of
  overrunning its budget.
* `latency=<seconds>` gives a fixed, deterministic delay; `compute_scale`
  models faster or slower hardware; `latency="none"` is the old behaviour.

Things to agree on with Shivam and Victor:

1. Run the A/B comparison under `measured`, with at most one episode per
   physical core, since wall time is the measurement.
2. `reset()` is untimed. It may build models (the planner builds its collision
   diagram there) but must not plan.
3. Report `compute_scale` alongside results, or report results at a fixed
   latency as well, if we want numbers that don't depend on the machine.

## What "general" means here, and how it is checked

No part of (B) reads a preset's geometry. Its inputs are the `WorldModel`
(target and obstacle poses at any time), the scene's collision geometry
(rigs and masts included), and the robot model. The things that would break
on unseen scenes were each replaced by something scene-independent:

| assumption that would not generalise | replaced by |
|---|---|
| a fixed intercept time | candidate T over the whole window; IK + tail feasibility decide |
| "the target is reachable while abs(y) < 0.5" | kinematic feasibility of the tail (velocity, joint limits, tracking) |
| one IK solution per T | IK families seeded across T + elbow self-motion search |
| one grasp | `grasps=[X_UG, ...]` (any number; grasp selection plugs in here) |
| plans always succeed | escape plan to a safe-to-hold state; retry at the next replan |
| fixed compute per scene | anytime goal generation within each replan's budget share |
| fixed RNG | seeded per scenario (`spec.seed`), so runs are reproducible |

`scripts/stress_planner.py` runs N generated scenes in parallel (mixing cart
and float mounts, all six objects, 0 to 4 obstacles, both cart directions) and
writes a failure table with goal-generation statistics per scene:

```
python scripts/stress_planner.py --n 200 --workers 4 --controllers st_rrt naive --out results/stress
```

## Results so far

80 **held-out** mixed scenes (`--seed 5000`, never used for tuning), measured
latency, 4 workers on an M1 Pro:

| | success | collision | timeout |
|---|---|---|---|
| naive (no avoidance) | 35 | 40 | 5 |
| **st_rrt** | **57** | **0** | 23 |

Success rate by cart speed (0-0.15 / 0.15-0.22 / 0.22-0.31 m/s): st_rrt 0.86 /
0.70 / 0.50, naive 0.56 / 0.45 / 0.25. By number of obstacles (0-4): st_rrt
1.00 / 0.88 / 0.75 / 0.62 / 0.31. No plan was rejected for arriving late; the
minimum obstacle clearance was 2.2 cm.

What the failures look like:

* When (B) fails, it **times out** (abstains); it does not collide. That
  includes scenes where an obstacle drives into the parked home pose (the hold
  baseline is hit there). (B) checks how long it can stay parked, shrinks the
  first planning budget to fit, and plans its way out with a reduced margin if
  the start state already violates the 3 cm margin
  (`test_planner_gets_out_of_the_way_when_parking_would_be_hit`).
* 20 of the 23 timeouts are scenes where **no goal candidate survived**, so
  the RRT is rarely what fails. Typical causes: fast carts (0.25 to 0.3 m/s
  leave only a few intercept times before the arm reaches a singularity
  during the lift), and 3 to 4 obstacles crossing the corridor where the elbow
  must be.
* Some scenes are infeasible for a top-down grasp. Example: a slow floating
  sphere riding along the shoulder-wrist line for the whole window. Only
  other grasps can solve those, which is the interface to Victor's grasp
  selection (`grasps=`).
* The remaining 3 have goals but no path within the budget: narrow passages
  in space-time. Persistent trees across replans (rather than replanning from
  scratch) would help.

For Shivam: the scenario generator can produce obstacles that hit a robot
parked at `q0` (for example seed 5034 with 4 obstacles, or float mounts).
"Solvable by a robot that gets out of the way" still holds, but the
`HoldController` sanity check only covers the presets.

## Known limitations / next steps

1. **Goal coverage** is the bottleneck: grasp candidates from grasp selection,
   a shorter tail at high cart speeds, and collision-aware IK.
   Goal generation is anytime: each replan spends at most `goal_share` of its
   budget on it, and "goal-work" cycles may overrun only while the executing
   plan is safe (they return no plan, so nothing can arrive late).
2. **Speed**: 70 µs per state check. A sphere-based numpy checker or VAMP
   would allow more replans and bigger trees. The Drake checker stays the
   reference for testing a faster one.
3. **Smoothness**: piecewise-linear q(t) has velocity jumps at knots; the
   position-controlled arm smooths them but lags (minimum clearance dips to
   about 2 cm against a 3 cm margin). Time-parameterising with acceleration
   limits would close the gap.
4. **With exact prediction, replanning only improves T.** The first valid
   plan stays valid. To make "deliberative plans go stale" a measurable
   effect, add a prediction-error option to `WorldModel` (e.g. predict
   parabolic paths as constant velocity). Then `validate=True` triggers real
   replans.
5. During the lift, the object is not attached to the hand in the collision
   model (object vs. obstacles is not checked).
