# (C) Hybrid: space-time trajectory optimisation seeded by IK

Branch `planner-a-c-reactive-hybrid`. Controller name in the scripts: `hybrid`.

## 1. The idea in one paragraph

(C) plans the whole episode once, as two space-time trajectory optimisations (Drake's
`KinematicTrajectoryOptimization`, solved with SNOPT). The first, the approach, has a free
duration, ends at the grasp pose of the object at its end time, and keeps clear of the
obstacles where they are at each sample time. Its initial guess is a straight joint-space
path to a hover pose and a grasp pose from collision-aware IK; if that gives no plan, the
same through a collision-free configuration halfway, then straight paths to the IK poses of
later grasp times. The second runs from the grasp to the end of the scene: move with the
object while the gripper closes, lift 10 cm and hold, still keeping clear of the obstacles.
Both are sent as one joint trajectory; (A) is not used. If staying parked would be hit before
a grasp plan moves the arm, (C) first solves a third, clearance-only optimisation (a dodge)
and plans the grasp from it. The exact problems are in section 8.

## 2. Code map and reading order

| # | file | what it does | start with |
|---|---|---|---|
| 1 | `dyngrasp/hybrid.py` | initial guesses, the two trajectory optimisations, execution | `update`, `_first_plan`, `_ik_grasps`, `_path`, `_plan`, `_plan_hold`, `_joint_trajectory` |
| 2 | `dyngrasp/collision.py` | per-category contexts and the world's full state at t for the clearance constraints | `CollisionWorld.context`, `full_state` |
| - | `dyngrasp/scenario.py` | `pillar_and_blocker`: a scene that needs a way around the pillar | `pillar_and_blocker` |

Tests: `tests/test_controllers.py` (in Drake, together with (A)).

## 3. One planning cycle (`HybridController.update`)

1. Wait at rest until `plan_budget` (2 s) before `t_enter - lead_time` (1 s before the
   object is reachable). While parked, check whether holding still gets too close to
   anything before a grasp plan could move the arm (`_parked_hit`); if so, dodge (4.6).
2. `_first_plan` for the start time t0 = now + budget, from where the arm is then (parked,
   or on the dodge): for each initial guess in turn, optimise the approach (`_plan`), then
   the close, lift and hold (`_plan_hold`); keep the first that gives both.
3. Send hold (until t0) + approach + close, lift and hold as one `JointTrajectory`, sampled
   by the wrapper every 5 ms. It holds its last pose afterwards.
4. If the plan arrived late (the wrapper rejected it), plan again with 1.5x the time the
   last plan took. If no plan is found, the arm stays where it is (dodging first if it
   would be hit there) or keeps dodging.

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
  (4 rad/s^2) on the plan. Solver: SNOPT, major optimality tolerance 1e-1: most of a tighter solve
  polishes a plan that is already feasible (`pillar_and_blocker`: 3.3 s at 1e-3, 0.45 s at
  1e-1, nearly the same path).
* Refinement (`_solve`): if it likely fits in the time left (3x the first solve), SNOPT runs
  again at 1e-2 from the 1e-1 solution, which is kept if it succeeds with no higher cost.
  A loose solve can stop at a feasible but poor plan: in `mix_0009` the hold at 1e-1 swung
  the arm around (cost 19.2, 38.8 rad of joint travel) and, between two clearance samples,
  through an obstacle; refined, cost 0.2 in 0.57 s more (from scratch at 1e-2: 0.54 in 2 s).
  The approach leaves 0.5 s for the hold. The same applies to the hold and the dodge.

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
* The seeds run in `n_workers` (4) parallel processes, started in `reset` so they cost no
  simulated time, and the first to finish with a plan is kept (`n_workers=1`: in turn).
  Inside a daemon process (`multiprocessing.Pool`) they run in turn. Four SNOPT processes at
  once make each solve 30-60 % slower, so this pays off when the first seed fails.

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

### 4.6 Dodging (`_parked_hit`, `_dodge`)

* While parked, (C) checks every 0.05 s of the predicted scene whether holding still breaks
  the clearance (4 cm to obstacles, 5 mm to the cart, floor and target). This takes about
  20 ms, once per parked pose.
* If that happens before a grasp plan could move the arm (or at any time, once no grasp plan
  was found), it optimises a dodge: from rest at the parked pose to the end of the scene,
  clearance every 0.1 s, minimum path energy, so the arm moves only as much as it has to.
  Its budget is half the time left before the hit (at most 2 s).
* The grasp plan then starts from the dodge's state at t0, moving: the approach's start
  velocity is the dodge's, and the dodge is kept before t0.
* `mix_0023`: an obstacle reaches the parked arm at 1.48 s; the dodge took 0.17 s and the
  arm kept 4.3 cm away (there is no IK grasp in this scene, so it then just keeps dodging).

## 5. Tools

| command | question it answers |
|---|---|
| `scripts/run_eval.py --scenarios scenarios/dev_v0.yaml --controllers hybrid` | success / collisions on a set, scored to the end of each scene |
| `scripts/render.py --preset pillar_and_blocker --controller hybrid --video --views close34 top` | an MP4 of one episode (needs `ffmpeg`) |
| `scripts/demo_scene.py --controller hybrid` | watch it in Meshcat |

## 6. Current results

Commit `76c07b2`. All runs: compute time costs simulated time (`measured`), scored to the
end of each scene, one run per scene; sets as in `planner_A.md`. First 8 episodes in parallel
(seeds then run in turn, see 4.2); every failure was then run again alone, with the default
4 seed processes, and that result counts (13 of 67 failures became successes).

| set | scenes | success | collision | dropped | timeout | plans rejected (late) |
|---|---|---|---|---|---|---|
| dev | 20 | 16 | 1 | 0 | 3 | 1 |
| post grid | 48 | 37 | 0 | 0 | 11 | 9 |
| plank grid | 48 | 38 | 0 | 0 | 10 | 4 |
| pillar grid | 18 | 6 | 10 | 0 | 2 | 6 |
| held-out mixed | 80 | 63 | 2 | 1 | 14 | 7 |

* Most failures are scenes where no grasp plan is found (in an earlier check, most of them
  had no collision-free IK grasp, 8.3). The arm then waits, dodging if needed (4.6): a
  timeout. Before dodging there were 24 collisions (12 in the held-out set), now 13.
* 10 of the 13 collisions are in the pillar grid, mostly with a first plan that took 2-5.7 s
  (late); the others are `dev_001`, `mix_0009` and `mix_0034`.
* "plans rejected (late)": the first plan took longer than its 2 s budget and was redone
  later; 18 of these 27 episodes still succeed.
* Median first-plan compute 0.65-1.2 s per set. Median grasp 0.8-1.8 s after the object
  becomes reachable (pillar grid: 3.7 s).
* Peak commanded acceleration 4 rad/s^2, except one episode at 10.5
  (`plank_blocker_v0.05_o0.20_t5.0`, not looked into yet); peak torque at most 0.52x the
  limits.

On the four report scenes (one episode at a time) (C) grasps and holds in `tall_blocker`,
`mix_0077` and `mix_0061`. In `pillar_and_blocker` its first plan took 1.6-1.9 s in three runs
(success) and 2.8 s in another: late, the pillar hit the parked arm.

## 7. Limitations and open questions

1. **Compute**: the first plan takes about 1-2 s (`pillar_and_blocker`: 1.8 s); a seed
   that leads SNOPT into a bad local minimum can take several seconds. With measured
   latency the arm stays parked while (C) computes, and an obstacle can hit it there;
   without a plan it stays parked.
2. **One plan**: no replanning, which is fine with perfect prediction; with prediction errors
   the plan would go stale.

## 8. The optimisation problems, exactly

Defaults are the `HybridController` constructor arguments (in brackets).

**Notation.** All in the world frame.

* A plan is a cubic B-spline (clamped, uniform knots) $q(s) = \sum_{j=1}^{N} b_j(s)\,P_j$, $s \in [0, 1]$,
  with control points $P_j \in \mathbb{R}^7$, executed as $t = t_0 + sT$. So $\dot q = q'(s)/T$ and
  $\ddot q = q''(s)/T^2$. Drake's B-spline derivative control points are $r'_j$ (for $q'$) and $r''_j$ (for $q''$).
* $p_G, R_G$: position and orientation of the WSG body frame at $q$. $\omega_G, v_G$: its angular and linear
  velocity at $(q, \dot q)$.
* $p^*(t), R^*$: the grasp pose of the object at time $t$ (`WorldModel.grasp_pose`, nominal top-down).
  $v_{obj}$: the carrier's velocity, constant while the object is in reach, so $p^*(t) = p^*(\bar t) + v_{obj}(t - \bar t)$.
* $e_R = \tfrac12 (R - R^\top)^\vee$ with $R = R^{*\top} R_G$: the rotation error, $\sin\theta$ times the axis.
* $d_{\mathcal C}(q, t)$: the smallest signed distance between the robot (iiwa links 1-7 and the WSG, fingers
  open) at $q$ and the world geometries of the categories $\mathcal C$, posed where the prediction puts them at
  $t$. The categories are obstacles $O$, cart (conveyor and carrier) $C$, floor $F$ and target $Tg$.
* $d_{\mathcal C}(q, t) \ge lb$ is Drake's `MinimumDistanceLowerBoundConstraint`:
  $\mathrm{SmoothOverMax}_i\ \varphi\big((d_i - d_{inf})/(d_{inf} - lb)\big)/\varphi(-1) \le 1$ over the robot-world
  pairs $i$, with $d_{inf} = lb + 0.01$ m and $\varphi$ the quadratically smoothed hinge. The world's
  configuration at $t = t_0 + sT$ is linearised as $x_w(\bar t) + \dot x_w(\bar t)(t - \bar t)$, so the
  gradient with respect to $T$ includes the obstacles' motion.
* Inequalities between vectors hold per component. "$\forall s$" bounds are imposed on the control points of
  $q$, $q'$ or $q''$, so by the convex-hull property they hold for all $s$ (conservatively).

### 8.1 Approach (`_plan`)

Variables: $P_1, \dots, P_N$ ($N$ = 12) and $T$. $t = t_0 + sT$ throughout.

$$
\begin{aligned}
\min_{P,\,T}\quad & T + \lambda \textstyle\sum_j \lVert r'_j \rVert^2 \qquad (\lambda = 0.1;\ \text{an upper bound on } \textstyle\int_0^1 \lVert q'(s) \rVert^2 ds) \\
\text{s.t.}\quad & \max(t_{enter} - t_0,\ 0.3) \le T \le t_{exit} - 1.5 - t_0 \\
& q(0) = q_0, \quad \dot q(0) = v_0 \\
& q_{lo} + 0.05 \le q(s) \le q_{hi} - 0.05, \quad |\dot q(s)| \le 0.6\, v_{max}, \quad |\ddot q(s)| \le 4 \qquad \forall s \\
& \textbf{grasp } (s = 1):\ |p_G - p^*(t)| \le 0.005, \ |e_R| \le 0.02, \ |\omega_G| \le 0.02, \ |v_G - v_{obj}| \le 0.02 \\
& \textbf{hover } (s = s_d):\ |p_G - p^*(t) - h\,\hat z| \le 0.01, \ |e_R| \le 0.04 \\
& \textbf{descent } (s = s_d + \tfrac{i}{4}(1 - s_d),\ i = 1, 2, 3):\ p_G - p^*(t) \in [-0.015, 0.015]^2 \times [0,\ h + 0.02] \\
& \textbf{clearance } (s = \tfrac{k}{30},\ k = 1, \dots, 30):\ d_O(q, t) \ge 0.04, \quad d_{C \cup F \cup Tg}(q, t) \ge 0.005
\end{aligned}
$$

* $q_0, v_0$: where the arm will be at $t_0$ = now + `plan_budget`: parked ($v_0 = 0$, imposed as $q'(0) = 0$), or
  on the dodge.
* $v_{max}$: the iiwa datasheet joint speeds, scaled by `v_scale` (0.6). $|\ddot q| \le$ `accel_limit` (4 rad/s²).
* $h$ = `hover` (0.10 m).
* $s_d = \mathrm{clip}\big((t_d - t_0)/T_{guess},\ 0.3,\ 0.9\big)$ is fixed, where $t_d$ is the initial
  guess's descent time.
* The obstacle and environment bounds are `d_safe` (0.04 m) and `d_safe_env` (0.005 m), checked at
  `n_collision_samples` (30) times.
* Initial guess: $T_{guess}$ = the guess's grasp time $- t_0$ (clipped to the bounds on $T$), and the control
  points fitted by least squares to the guess path at 40 points.
* Solver: SNOPT, major optimality tolerance `optimality_tol` (0.1), major feasibility tolerance 1e-4. A plan is
  accepted if SNOPT reports success. Then, if now + 3x that solve's time is before the deadline (the plan's budget,
  minus 0.5 s left for the hold), SNOPT again at `refine_tol` (0.01) from that solution; kept if it succeeds with
  a cost no higher.

### 8.2 Close, lift and hold (`_plan_hold`)

* It starts at the grasp, $t_g = t_0 + T^*$ (the approach's solved duration), in the approach's end state
  $q_T, \dot q_T$.
* Its duration is fixed: $D = \max(t_{end} - t_g,\ t_{up} + 0.5)$, with $t_{up} = 0.4 + 0.6 = 1.0$ s
  (`close_time` + `lift_time`) and $t_{end}$ the end of the scene.
* $N_h = \lceil D/0.5 \rceil + 3$ control points (`hold_knot_spacing`).
* $t = t_g + sD$, and $\tau = t - t_g$ is the time since the grasp.

$$
\begin{aligned}
\min_{P}\quad & \textstyle\sum_j \lVert r'_j \rVert^2 \\
\text{s.t.}\quad & q(0) = q_T, \quad q'(0) = D\,\dot q_T, \quad q'(1) = 0 \\
& q_{lo} + 0.05 \le q(s) \le q_{hi} - 0.05, \quad |\dot q(s)| \le 0.6\, v_{max}, \quad |\ddot q(s)| \le 2 \qquad \forall s \\
& \textbf{closing } (\tau = 0.2,\ 0.4):\ |p_G - p^*(t)| \le 0.005, \ |e_R| \le 0.02 \\
& \textbf{lifted } (\tau = 1.0,\ 1.4,\ 1.8, \dots \le D):\ z_G \ge z^*(t_g) + 0.10 \\
& \textbf{clearance } (\tau = 0.4,\ 0.8,\ 1.2, \dots \le D):\ d_O(q, t) \ge 0.04, \quad d_{C \cup F}(q, t) \ge 0.005
\end{aligned}
$$

* $|\ddot q| \le$ `hold_accel_limit` (2 rad/s²). The lift height is `lift_height` (0.10 m). Samples are every
  `hold_sample_spacing` (0.4 s).
* The target is left out of the clearance because it is in the hand. The object is not collision-checked
  against the obstacles.
* The gripper is commanded closed from $t_g$ on.
* Initial guess: $q_T$, then $q_c$ at $\tau = 0.4$, then $q_l$ from $\tau = 1.0$ to $D$, interpolated in
  joint space.
  * $q_c$: IK of the grasp pose at $t_g + 0.4$. If IK fails, $q_T + 0.4\,\dot q_T$.
  * $q_l$: IK of the same pose raised by 0.10 m. If IK fails, $q_c$.
* Same solver settings as 8.1.

### 8.3 IK for the initial guesses (`_ik_grasps`, `_first_seeds`, `_ik`)

Drake's `InverseKinematics`, with the obstacles fixed where they are at time $t$, and $d_O(q, t) \ge 0.05$
(`d_safe` + 1 cm, influence offset 0.2 m).

* **Grasp and hover poses:** find $q$ with $|p_G - p_X| \le 0.003$ and $\angle(R_G, R_X) \le 0.01$ rad (no
  cost; the initial guess as below).
  * Grasp times $t_g$ step by 0.5 s from $\max(t_{enter}, t_0 + 1)$ to $t_{exit} - 2$. For each:
    * the grasp $q_g$ at $t_g$;
    * the hover $q_h$ at $t_d = t_g - 0.6$: the grasp pose raised by $h$, IK started from $q_g$.
  * Up to 6 tries per grasp time: first from $q_0$, then from random configurations. Keep the pair whose
    hover is nearest to $q_0$ in $\max_j |q_{h,j} - q_{0,j}|/v_j$ ($v_j$: datasheet speed of joint $j$).
* **Halfway configuration:** $\min \lVert q - q_{target} \rVert^2$ subject to the clearance, at
  $t = (t_0 + t_d)/2$.
  * $q_{target}$ is first the midpoint $(q_0 + q_h)/2$, then that midpoint with each joint moved by up to
    `random_spread` (1 rad) at random (`n_random_seeds`: 3).
  * If the midpoint is already collision-free, it adds nothing to the straight line and is skipped.
* **Paths (`_path`):** straight lines in joint space, $q_0 \to$ (halfway configuration at $(t_0 + t_d)/2$)
  $\to q_h$ at $t_d \to q_g$ at $t_g$.

### 8.4 Dodge (`_parked_hit`, `_dodge`)

* $t_{hit}$: the first $t \ge$ now (0.05 s steps) at which holding the parked $q_p$ breaks
  $d_O(q_p, t) \ge 0.04$ or $d_{C \cup F \cup Tg}(q_p, t) \ge 0.005$.
* Dodge if now $+ 0.1 < t_{hit} \le t_{go}$. $t_{go}$, when a grasp plan could move the arm: $\max(\text{now},\ t_{plan})$
  $+$ `plan_budget` while waiting for one ($t_{plan}$ = when the grasp plan is computed), $\infty$ once none was found.
* It starts at $t_s$ = now $+ \min(2,\ (t_{hit} - \text{now})/2)$ and lasts to the end of the scene, $D = t_{end} - t_s$, with
  $N_d = \lceil D/0.5 \rceil + 3$ control points. $t = t_s + sD$, $\tau = t - t_s$.

$$
\begin{aligned}
\min_{P}\quad & \textstyle\sum_j \lVert r'_j \rVert^2 \\
\text{s.t.}\quad & q(0) = q_p, \quad q'(0) = q'(1) = 0 \\
& q_{lo} + 0.05 \le q(s) \le q_{hi} - 0.05, \quad |\dot q(s)| \le 0.6\, v_{max}, \quad |\ddot q(s)| \le 4 \qquad \forall s \\
& \textbf{clearance } (\tau = 0.1,\ 0.2, \dots \le D):\ d_O(q, t) \ge 0.04, \quad d_{C \cup F \cup Tg}(q, t) \ge 0.005
\end{aligned}
$$

* Samples every `dodge_sample_spacing` (0.1 s). Initial guess: staying at $q_p$. Same solver settings as 8.1.
* If it finds no dodge, (C) carries on as before (parked).
