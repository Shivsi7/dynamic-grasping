# Grasping a Moving Object Among Moving Obstacles

6.4212 (Robotic Manipulation, Fall 2026) final project: a 7-DoF KUKA iiwa with a
WSG gripper picks a known YCB object off a moving cart while scripted obstacles
cross its workspace. We compare a **reactive** controller (A, differential IK as
a QP with collision constraints) against a **deliberative** space–time planner
(B, RRT-Connect in (q, t)) on identical randomized scenarios using ground-truth
state. The proposal (with milestones and the division of work) is on Gradescope;
drop a copy in `docs/` when convenient.

This repository holds the shared infrastructure: the Drake scene, the
`WorldModel` (predicted state), the controller API, the evaluation harness and
two reference controllers. (A) and (B) plug into the `Controller` API below.

## Setup

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd dynamic-grasping
uv sync --extra dev            # creates .venv with drake 1.57, manipulation, pandas, ...
source .venv/bin/activate
pytest                         # ~1 min, 34 tests
python scripts/demo_scene.py   # opens Meshcat: tall blocker vs. the naive controller
```

`HEADLESS=1` disables Meshcat (no browser wait, no real-time pacing); use it for
batch runs and for checking that something works. `python scripts/render.py
--gif` writes annotated snapshots and GIFs of the presets to `results/renders/`
from fixed cameras, with scripted vs. simulated positions printed under each row.

## Layout

```
dyngrasp/
  paths.py       LinearPath / ParabolicPath: p(t), v(t), parked outside [t_start, t_end]
  objects.py     YCB registry: rest pose on the cart, nominal top-down grasp X_UG, sizes
  scenario.py    ScenarioSpec (YAML-serialisable), random generator, presets
  scene.py       builds the HardwareStation for a spec; kinematic diagram for planners
  world.py       WorldModel: predicted target / obstacle state at any time t
  controller.py  Controller API, RobotState, Command, RobotModel, Drake wrapper (+ latency model)
  trajectory.py  JointTrajectory: time-indexed plans the wrapper executes tick by tick
  baselines.py   HoldController, NaiveInterceptController (no avoidance)
  planning/      approach (B): space-time RRT-Connect (see docs/planner_B.md)
    controller.py  SpaceTimeRRTController (registered as `st_rrt`): the replanning cycle
    config.py      STRRTConfig: every tunable number; the iiwa's acceleration limits
    st_rrt.py      generic (q, t) RRT-Connect, time-preserving shortcuts, escape planning
    goals.py       intercept times, pre-grasp IK (+ elbow self-motion), grasp tails
    collision.py   SpaceTimeChecker: is (q, t) / an edge free (held object included)
    motion.py      smoothing planner knots into the executed curve, and verifying it
    escapes.py     evasive motion primitives and the post-grasp retreat
    variants.py    named (B) configurations used by the scripts
  grasps.py      grasp candidates (top / tilted / side) from object boxes, geometric validity
  monitor.py     EpisodeMonitor: collisions, clearance, grasp / lift / success
  harness.py     run_episode, run_batch (-> pandas), aggregate
  viz.py         Meshcat helper
scripts/
  demo_scene.py  watch one scenario in Meshcat
  run_eval.py    batch evaluation -> results/<name>/results.csv + scenarios.yaml
  summarize.py   tables (markdown) and figures from results.csv
  render.py      headless PNG contact sheets / GIFs of scenarios (no browser needed)
  stress_planner.py  many generated scenarios in parallel, failure tables
  grasp_feasibility.py  which grasp candidates are reachable / collision-free / physically hold
  compare_videos.py  side-by-side MP4s of one scene under several controllers
scenarios/       shared scenario sets (YAML), so everyone runs the same episodes
tests/           pytest
```

## The world

World frame W: iiwa base at the origin, home configuration faces +x, floor at z = 0.

* **Conveyor** ("treadmill"): a static frame along y in a lane at x ≈ 0.63 m,
  with a carrier plate moving along it, top surface at z = 0.20 m. Speed
  0.05–0.30 m/s. The carrier always enters the robot's reach (|y| < 0.5) at
  t = 3 s and the target is reachable for 1/speed seconds.
* **Target**: one of `soup_can`, `sugar_box`, `gelatin_box`, `potted_meat_can`,
  `cracker_box`, `mustard`, upright on the carrier, riding along (free body, friction).
* **Obstacles**: 0–4 spheres (r 5–10 cm) or boxes (10–20 cm), each on a mast on
  a small wheeled rig (`mount: cart`) at 0.1–0.5 m/s. Each one either drives
  along y through the corridor between the robot's pedestal and the conveyor
  (straight `linear` or curving `parabolic` ground path) or along x in a band
  beside the robot, from behind it to just short of the conveyor or back, as in
  the proposal figure. All are timed to cross while the target is reachable and
  never run into each other. `mount: float` gives a free-flying shape on a 3D
  path (lobs) instead. Sampled paths never hit the conveyor, the carrier, the
  fixed robot base or the floor, so every scenario is solvable by a robot that
  gets out of the way. Obstacles are dynamic bodies driven by stiff PD joints: they follow
  their script to ~mm but a collision is physical.
* **Robot**: `iiwa7_with_box_collision` + `schunk_wsg_50_with_tip`, driven as in
  the course's HardwareStation (position-controlled iiwa, Schunk WSG driver).

Frames (see `objects.py`): **U** = upright object frame (footprint centre on the
cart surface, z up), **G** = WSG body frame (fingers along +y, close along x,
pads at y = 0.108). `WorldModel.grasp_pose(t)` = `X_WU(t) @ X_UG_grasp`.

Named presets: `scenario.no_obstacles(speed)`, `scenario.tall_blocker(speed,
obstacle_speed, t_cross)` (the "initial test" from the proposal: a 0.12 x 0.12
x 0.4 m post on a wheeled rig, top at 0.5 m) and `scenario.plank_blocker(...)`
(a thin 0.20 x 0.04 x 0.4 m plank). Both drive across the corridor between the
robot and the conveyor and, at `t_cross` (default: one second after the target
becomes reachable, when a direct grasp would happen), sit exactly on the
straight line from the base to the target. They cannot be ignored (a
straight-in grasp collides) but can be passed on either side or over the top:
collision-aware IK finds grasp configurations of both kinds, and over-the-top
stops being possible above a top height of about 0.5 m. Random sets:
`generate_scenario_set(seed, n)`; save/load with `save_scenario_set` /
`load_scenario_set`. `scenarios/dev_v0.yaml` is the shared development set.

## Writing a controller

```python
from dyngrasp import Controller, Command, RobotState, WorldModel, WSG_OPEN, WSG_CLOSED

class MyController(Controller):
    name = "reactive_qp"
    period = 0.01                                   # update() runs at 100 Hz

    def reset(self, world: WorldModel, robot, state: RobotState):
        self.robot = robot                          # RobotModel: fk(q), jacobian(q), limits

    def update(self, state: RobotState, world: WorldModel) -> Command:
        t = state.t
        X_WG_des = world.grasp_pose(t + 0.5)        # where the grasp will be in 0.5 s
        v_obj = world.target_velocity(t)
        for o in world.obstacle_states(t + 0.2):    # predicted obstacles (p, v, radius, shape, dims)
            ...
        caps = world.swept_capsules(t, t + 0.2)     # (p0, p1, radius) per obstacle
        v = ...                                     # your QP here
        return Command(v=v, gripper=WSG_OPEN, info={"phase": "approach"})
```

`RobotState` carries `q`, `v`, `X_WG` (measured), `wsg_width`, the last
command and the `trajectory` being executed (if any). `Command.q` (positions),
`Command.v` (velocities, integrated at `period` and clipped to joint limits) or
`Command.trajectory` (a `JointTrajectory`, sampled at every tick, with an
optional gripper schedule) plus a gripper opening. `info` is logged.

### Computation time costs simulated time

The simulator waits for `update()`, so by default a slow controller would look
instantaneous. The harness therefore runs with `latency="measured"`: a command
computed in w seconds of wall time takes effect w seconds of *sim* time later,
and the controller is not called again until then. Meanwhile the previous
command keeps running (a trajectory keeps playing, a velocity keeps being
integrated). A controller faster than its period sees one period of delay. A
trajectory that would make the arm jump when it finally takes effect (it
disagrees with what is executing by more than 2 cm in any joint) is rejected
and counted in `n_rejected_commands`. Planners should plan from `t + budget`
and keep the executing trajectory as the prefix.

Options (`run_episode`, `run_batch`, `run_eval.py --latency/--compute-scale`):
`"measured"` (default), a fixed delay in seconds (deterministic), `"none"`
(the old behaviour), and `compute_scale` to model faster or slower hardware.
Measured latency is wall time: run at most one episode per physical core. Use
`reset()` only for scene-independent setup (it is not timed).

For planners that need collision checks at a hypothetical (q, t):

```python
from dyngrasp.scene import build_kinematic_diagram, BodyRoles
diagram, kin = build_kinematic_diagram(spec)
roles, ctx = BodyRoles(kin), diagram.CreateDefaultContext()
world.set_kinematic_state(kin, ctx, t, q)
qo = kin.scene_graph.get_query_output_port().Eval(kin.scene_graph.GetMyContextFromRoot(ctx))
roles.robot_collisions(qo)   # [] if free;  roles.robot_clearance(qo) -> (d_obstacles, d_cart_floor)
```

The kinematic diagram filters the pairs that are not collisions (object on
cart, cart over floor, gripper vs. target, and iiwa links 5/7 whose collision
boxes nearly touch at ordinary wrist angles), so `qo.HasCollisions()` and a
`MinimumDistanceLowerBoundConstraint` in `InverseKinematics` can be used directly.

Register the controller in `scripts/run_eval.py`'s `CONTROLLERS` dict.

## Evaluation

```bash
python scripts/run_eval.py --n 20 --seed 0 --out results/dev           # random set, all controllers
python scripts/run_eval.py --scenarios scenarios/dev_v0.yaml --controllers naive --out results/naive
python scripts/run_eval.py --sweep --seeds-per-cell 5 --out results/sweep   # speed x obstacle grid
python scripts/run_eval.py --tall-blocker --out results/post       # post: target speed x obstacle speed x crossing time
python scripts/run_eval.py --plank-blocker --out results/plank     # same grid, thin plank
python scripts/summarize.py results/sweep                               # summary.md + png figures
```

One row per episode in `results.csv`:

| column | meaning |
|---|---|
| `outcome` | `collision` / `success` / `dropped` / `timeout` (collision takes precedence) |
| `success`, `collided` | separately, so "grasped despite a collision" is visible |
| `t_grasp`, `t_lift`, `t_success` | first in-hand, first in-hand-and-lifted, held 0.5 s |
| `t_first_collision`, `collision_with`, `collision_category` | obstacle / cart (carrier or conveyor) / floor / target |
| `min_clearance_obstacles` (+ time), `min_clearance_environment` | signed distance, robot to obstacles / cart+floor |
| `n_cycles`, `compute_mean_ms`, `compute_max_ms`, `compute_p95_ms` | wall time of `update()` per cycle |
| `latency_mode`, `delay_max_ms`, `n_rejected_commands` | latency model used, longest command delay, late trajectories rejected |
| `target_speed`, `n_obstacles`, `obstacle_speed_mean/max`, `object`, `seed` | scenario parameters |

Definitions (see `monitor.py`): a **collision** is any contact between a robot
body (iiwa links 1–7, WSG) and an obstacle, the cart, the floor, or the target,
except the gripper touching the target. The object is **in hand** when the
finger-pad point lies inside its bounding box and the gripper is not open;
**success** is in-hand and ≥ 5 cm above the rest height for 0.5 s. By default an
episode stops at the first collision (`--keep-going` to continue).

Baselines on `no_obstacles`: `naive` grasps the soup can, gelatin box, potted
meat can and sugar box; it is too slow to converge for the cracker box and the
mustard bottle (grasp selection is the job of (A)/(B), not of this baseline).
Simulation runs at roughly 2–3x real time headless.
