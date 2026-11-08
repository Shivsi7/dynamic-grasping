# (C) Hybrid: space-time trajectory optimisation seeded by IK

Controller `hybrid`, in `dyngrasp/hybrid.py` (start with `update`). Tests: `tests/test_controllers.py`.

## Idea

(C) plans the episode with Drake's `KinematicTrajectoryOptimization` (cubic B-splines, SNOPT), as
two problems sent as one joint trajectory, replanning only until it has one. It does not use (A) or (B).

1. **Approach:** from the arm's state at t0 to the object's grasp pose at t0 + T (T free), through
   a hover pose 10 cm above it, clear of the obstacles where they will be at each time.
2. **Hold:** from the grasp to the end of the scene: close while moving with the object, lift
   10 cm, stay clear of the obstacles.

If the parked arm would be hit before a plan moves it, a third problem, a **dodge**, moves it
out of the way first; the grasp plan then starts from the dodge.

## One cycle (`update`)

1. Park until $t_{enter}$ − 1 s (`lead_time`) − 2 s (`plan_budget`); $t_{enter}$: the object comes into reach.
2. Plan for t0 = now + 2 s. Compute time costs simulated time, so the plan holds still until t0.
   * **Grasp time:** 0.5 s steps outward from the object's closest approach to the robot (earlier
     first), at least 1.2 s (`exit_margin`) before it leaves reach; the first with a
     collision-free IK grasp and hover pose (6 IK starts each). If none: 0.1 and 0.2 s steps
     around the two nearest misses (reach error + clearance shortfall).
   * **Seeds:** straight joint-space lines to those poses, and to those of the earliest grasp time
     that works (a short guess); the same via a collision-free halfway configuration (nearest to
     the midpoint, then to 2 random ones, 3 without the earliest); straight lines to the next 2
     grasp times.
   * **Workers:** 4 processes (started in `reset`, stopped by `close` when the episode ends) solve
     the seeds in parallel. Each sends its plan at tolerance 0.1, then refines it at 0.01 from that
     solution until the deadline; (C) keeps the lowest-cost plan. Inside a daemon process the
     seeds run in turn, and a refine runs only if it likely fits (3x the first solve).
3. A late plan is rejected; (C) replans with 1.5x the time it took. No plan: it stays parked
   (dodging if needed) and plans again on the next update while t0 < $t_{exit}$ − 1.2 s: fresh
   random vias and IK starts, untried grasp times first, the vias to the earliest grasp time.

## The problems

Defaults are `HybridController` arguments (in backticks). World frame throughout.

**Notation**

* $q(s) = \sum_{j=1}^{N} b_j(s)\,P_j$, $s \in [0,1]$: clamped cubic B-spline with uniform knots and
  control points $P_j \in \mathbb{R}^7$, executed at $t = t_0 + sT$; $\dot q = q'/T$, $\ddot q = q''/T^2$.
* $E(P) = \sum_{j=1}^{N-1} \lVert P_{j+1} - P_j \rVert^2$ (Drake's `AddPathEnergyCost`).
* $\mathcal L_a$: $q_{lo} + 0.05 \le q \le q_{hi} - 0.05$, $|\dot q| \le 0.6\,v_{max}$ (`v_scale`),
  $|\ddot q| \le a$, imposed on the control points of $q$, $\dot q$, $\ddot q$, so for all $s$.
  $v_{max}$: the iiwa datasheet speeds. Vector inequalities hold per component.
* $p_G, R_G$: pose of the WSG body frame at $q$; $\omega_G, v_G$: its angular and linear velocity.
* $p^*(t), R^*$: the object's grasp pose at $t$ (`WorldModel.grasp_pose`, top-down), with
  $\partial p^*/\partial t = v_{obj}$, the cart's velocity at $t_{enter}$. $e_R = \tfrac12 (R - R^\top)^\vee$ with
  $R = R^{*\top} R_G$. $h$ = `hover` (0.10 m); $\hat z$: up.
* $d_{\mathcal C}(q, t)$: the smallest signed distance from the robot at $q$ (iiwa links 1-7, WSG body
  and open fingers) to the categories $\mathcal C \subseteq \{O, C, F, Tg\}$ (obstacles; conveyor and
  carrier; floor; target), posed where the prediction puts them at $t$. WSG-target pairs are
  filtered, so $d_{Tg}$ counts links 1-7 only.
* $d_{\mathcal C} \ge lb$ is Drake's `MinimumDistanceLowerBoundConstraint`:
  $\mathrm{SmoothOverMax}_i\,\varphi\big((d_i - d_{inf})/(d_{inf} - lb)\big)/\varphi(-1) \le 1$ over the pairs $i$,
  $\varphi$ the quadratically smoothed hinge, $d_{inf} = lb + 0.01$. The world's configuration enters as
  $x_w(\bar t) + \dot x_w(\bar t)(t - \bar t)$, so the gradient w.r.t. $T$ includes its motion.
* A constraint at $s$ is on $(q(s), \dot q(s), t_0 + sT)$; their gradients w.r.t. $(P, T)$ are formed in
  numpy and passed to Drake as AutoDiff.

**Solver** (every problem): SNOPT, major feasibility tolerance 1e-4, major optimality tolerance
`optimality_tol` (0.1); a solution is accepted if SNOPT reports success. Refining re-solves the same
problem at `refine_tol` (0.01) from the loose solution.

* Seed worker: sends the loose approach and hold; then, if before the deadline, the refined hold;
  then, if still before it, the refined approach with a hold refined from the last one (each sent
  if feasible). (C) keeps the received plan with the lowest $J_{approach} + J_{hold}$: at the deadline
  once it has one, else when every seed is done.
* Main process (seeds in turn, and the dodge): right after a successful loose solve, if now + 3x
  its solve time is before the deadline minus 0.5 s (approach) or 0 (others); kept if it succeeds
  with no higher cost.
* Deadline: wall time at the start of planning + budget − 0.05 s.

### IK for the seeds (`_ik`, `_ik_grasps`, `_first_seeds`, `_path`)

Drake's `InverseKinematics` with the world fixed where the prediction puts it at $t$ (WSG open):
$q_{lo} \le q \le q_{hi}$ and $d_O(q, t) \ge 0.05$ (`d_safe` + 0.01; $d_{inf} = lb + 0.2$), solved from
$q_{guess}$.

* $\mathrm{IK}(t, X, q_{guess})$: find $q$ with $|p_G - p_X| \le 0.003$ and $\angle(R_G, R_X) \le 0.01$ rad (no cost).
* $\mathrm{IK}(t, q_{target})$: $\min_q \lVert q - q_{target} \rVert^2$, from $q_{guess} = q_{target}$.
* Miss of a failed IK, at the solver's last iterate: $|p_G - p_X| + 0.1\,\angle(R_G, R_X) + \max(0,\ 0.05 - d_O)$
  (the first two terms only with $X$).

**Grasp and hover:** $t_c = \mathrm{clip}(t_{closest}, lo, hi)$ with $lo = \max(t_{enter}, t_0 + 1)$,
$hi = t_{exit} - 1.2$ ($t_{closest}$: the cart passes $y = 0$, nearest the robot base). Candidates
$t_g = t_c + 0.5k \in [lo, hi]$, $k \in \mathbb{Z}$, sorted by (tried, $|k|$, $t_g$); tried: by an earlier plan that
found nothing. For each, 6 tries, $q_{guess} = q_0$ then 5 uniform random in $[q_{lo}, q_{hi}]$ (new on each retry):

$$
q_g = \mathrm{IK}\big(t_g,\ (p^*(t_g), R^*),\ q_{guess}\big), \qquad
q_h = \mathrm{IK}\big(t_d,\ (p^*(t_d) + h\,\hat z,\ R^*),\ q_g\big), \qquad t_d = t_g - 0.6,
$$

keeping the pair with the least $\max_j |q_{h,j} - q_{0,j}|/v_{max,j}$. The first $t_g$ with a pair
gives the first seeds; the next 2 (`n_ik_seeds` − 1) the later ones.

* **Earliest** (`ik_early`): the first pair with candidates sorted by ($t_g > t_0 + 5$, tried, $t_g$).
* **Fine search**, if no candidate has a pair: $t \pm 0.1$, $t \pm 0.2$ (in $[lo, hi]$) around the two
  candidates $t$ with the lowest miss (least over their tries), lowest first.

**Halfway configurations**, at $t_m = (t_0 + t_d)/2$ with $q_m = (q_0 + q_h)/2$, $(t_d, q_h)$ of the first
grasp (on a retry, of the earliest): $\mathrm{IK}(t_m, q_m)$ (dropped if it returns $q_m$ within 1e-3 rad), then
$\mathrm{IK}\big(t_m,\ \mathrm{clip}(q_m + u_i,\ q_{lo} + 0.05,\ q_{hi} - 0.05)\big)$, $u_i \sim U[-1, 1]^7$
(`random_spread` 1 rad, new on each retry), $i = 1, \dots, n$: $n = 3$ (`n_random_seeds`), 2 with an earliest seed.

**Seed paths** (`_path`): piecewise linear in joint space through $(t_0, q_0)$, $(t_m, q_{via})$ (if
any), $(t_d, q_h)$, $(t_g, q_g)$, sampled at 40 times on $[t_0, t_g]$. Order: no via; no via to the
earliest grasp (if at another time); each halfway configuration found; if none gives a plan, no via
to each later grasp time.

### Approach (`_plan`)

$t_0$ = now + budget, planned from $t_{plan} = t_{enter} -$ `lead_time` (1 s) $-$ budget, and only if
$t_0 < t_{exit} - 1.2$ (`exit_margin`); while none is found, again at every update. Budget: `plan_budget`
(2 s), raised to 1.5x a found plan's compute time when that is larger. $q_0, v_0$: the arm's state at $t_0$, parked ($v_0 = 0$) or on the dodge.
$N = 12$ (`n_control_points`).

$$
\begin{aligned}
\min_{P,\,T}\quad & T + 0.1\,E(P) \\
\text{s.t.}\quad & \max(t_{enter} - t_0,\ 0.3) \le T \le t_{exit} - 1.2 - t_0, \quad q(0) = q_0, \quad \dot q(0) = v_0, \quad \mathcal L_4 \\
& \textbf{grasp } (s = 1):\ |p_G - p^*(t)| \le 0.005,\ |e_R| \le 0.02,\ |\omega_G| \le 0.02,\ |v_G - v_{obj}| \le 0.02 \\
& \textbf{hover } (s = s_d):\ |p_G - p^*(t) - h\,\hat z| \le 0.01,\ |e_R| \le 0.04 \\
& \textbf{descent } (s = s_d + \tfrac{i}{4}(1 - s_d),\ i = 1, 2, 3):\ p_G - p^*(t) \in [-0.015, 0.015]^2 \times [0,\ h + 0.02] \\
& \textbf{clearance } (s = \tfrac{k}{30},\ k = 1, \dots, 30):\ d_O(q, t) \ge 0.04,\ d_{C \cup F \cup Tg}(q, t) \ge 0.005
\end{aligned}
$$

* 0.1 = `energy_weight`; 4 = `accel_limit` (rad/s²); 0.04 = `d_safe`; 0.005 = `d_safe_env`;
  30 = `n_collision_samples`. $v_0 = 0$ is imposed as $q'(0) = 0$.
* $s_d = \mathrm{clip}\big((t_d - t_0)/T_{guess},\ 0.3,\ 0.9\big)$, fixed; $t_d$: the seed's hover time.
* Initial guess: $T_{guess} = t_g - t_0$ (clipped to the bounds on $T$), $t_g$ the seed's grasp time;
  $P$ the least-squares fit of $q(s_k)$ to the seed path at $t_0 + s_k T_{guess}$, $s_k = k/39$,
  $k = 0, \dots, 39$. Refining: the loose $(P, T)$, same $s_d$.

### Close, lift and hold (`_plan_hold`)

From the grasp, $t_g = t_0 + T^*$, in the approach's end state $(q_T, \dot q_T)$, for the fixed
$D = \max(t_{end} - t_g,\ 1.5)$ ($t_{end}$: end of the scene; 1.5 = `close_time` + `lift_time` + 0.5);
$N = \lceil D/0.5 \rceil + 3$ (`hold_knot_spacing`); $t = t_g + sD$, $\tau = t - t_g$. The gripper is
commanded closed from $t_g$.

$$
\begin{aligned}
\min_{P}\quad & E(P) \\
\text{s.t.}\quad & T = D, \quad q(0) = q_T, \quad q'(0) = D\,\dot q_T, \quad q'(1) = 0, \quad \mathcal L_2 \\
& \textbf{closing } (\tau = 0.2,\ 0.4):\ |p_G - p^*(t)| \le 0.005,\ |e_R| \le 0.02 \\
& \textbf{lifted } (\tau = 1.0,\ 1.4,\ \dots \le D):\ z_G \ge z^*(t_g) + 0.10 \\
& \textbf{clearance } (\tau = 0.4,\ 0.8,\ \dots \le D):\ d_O(q, t) \ge 0.04,\ d_{C \cup F}(q, t) \ge 0.005
\end{aligned}
$$

* 2 = `hold_accel_limit`; closing at `close_time` (0.4 s) and half of it; lifted by `lift_height`
  (0.10 m) from `close_time` + `lift_time` (1.0 s); samples every `hold_sample_spacing` (0.4 s).
* Initial guess: piecewise linear through $q_T$ ($\tau = 0$), $q_c$ (0.4), $q_l$ (1.0), $q_l$ ($D$),
  fitted as above, $T = D$; IK as above:
  * $q_c = \mathrm{IK}\big(t_g + 0.4,\ (p^*(t_g + 0.4), R^*),\ q_T\big)$, else $q_T + 0.4\,\dot q_T$;
  * $q_l = \mathrm{IK}\big(t_g + 1.0,\ (p^*(t_g + 0.4) + 0.10\,\hat z,\ R^*),\ q_c\big)$, else $q_c$.
* Refining: from the last hold, re-timed to $D$.

### Dodge (`_parked_hit`, `_dodge`)

While parked at $q_p$: $t_{hit}$ = the first $t \in \{\text{now}, \text{now} + 0.05, \dots\}$, $t < t_{end}$,
at which $q_p$ breaks $d_O(q_p, t) \ge 0.04$ or $d_{C \cup F \cup Tg}(q_p, t) \ge 0.005$ (computed once per
$q_p$). Dodge if now $+ 0.1 < t_{hit} \le t_{go}$, with $t_{go} = \max(\text{now}, t_{plan}) +$ budget while
the grasp plan is pending, $\infty$ once none was found. Start $t_s$ = now $+ \min(2,\ (t_{hit} - \text{now})/2)$
(`plan_budget`; also its compute budget), holding $q_p$ until then; $D = t_{end} - t_s$,
$N = \lceil D/0.5 \rceil + 3$, $t = t_s + sD$, $\tau = t - t_s$.

$$
\begin{aligned}
\min_{P}\quad & E(P) \\
\text{s.t.}\quad & T = D, \quad q(0) = q_p, \quad q'(0) = q'(1) = 0, \quad \mathcal L_4 \\
& \textbf{clearance } (\tau = 0.1,\ 0.2,\ \dots \le D):\ d_O(q, t) \ge 0.04,\ d_{C \cup F \cup Tg}(q, t) \ge 0.005
\end{aligned}
$$

* Samples every `dodge_sample_spacing` (0.1 s). Initial guess $q \equiv q_p$, $T = D$. No solution:
  (C) stays parked and no longer dodges from $q_p$.
* A grasp plan then starts from the dodge's $(q, \dot q)$ at $t_0$, with the dodge kept before $t_0$.

**Simulation details:** commands every 5 ms (the iiwa driver differentiates them every 5 ms);
links 5 and 7 of the box model overlap past |q6| ~ 1.75 rad, so that pair is filtered.

## Results

Measured latency, scored to the end of each scene; 8 episodes in parallel, then every non-success
rerun alone (that result counts). (C) at `5424aac` (reruns at `6246476`, same planning); (A), (B)
at `1c10b7a`. No (C) episode dropped the object.

| set | scenes | success | collision | timeout | (A) / (B) success |
|---|---|---|---|---|---|
| dev | 20 | 17 | 1 | 2 | 11 / 15 |
| post grid | 48 | 48 | 0 | 0 | 34 / 45 |
| plank grid | 48 | 48 | 0 | 0 | 33 / 44 |
| pillar grid | 18 | 10 | 7 | 1 | 2 / 6 |
| held-out mixed | 80 | 72 | 1 | 7 | 52 / 63 |
| all | 214 | 195 | 9 | 10 | 132 / 173 |

(C) fails where (A) or (B) succeeds in 2 scenes only (pillar grid, grasped by (A)).

## Limitations

1. **No plan** (the 10 timeouts: `dev_004`, `dev_007`, one pillar scene, `mix_0007`, `0014`, `0038`,
   `0059`, `0063`, `0067`, `0068`): none found before the object leaves reach. `mix_0007`: grasps
   only at the edge of reach, the hover out of reach; `mix_0038`: a grasp needs ~2 cm clearance
   (IK asks 5 cm); `mix_0068`: its one plan took 4.2 s and arrived late.
2. **Blind while computing** (the 9 collisions): the arm cannot react during a solve, which has no
   time limit. 7 pillar-grid scenes, whose dodges take 1.4-20 s; `mix_0079` and `dev_001`, hit while
   parked as a first plan of 4.9 s / 2.6 s arrives too late.
3. **No replanning once a plan is sent:** fine with perfect prediction.

## Tools

* `scripts/run_eval.py --scenarios scenarios/dev_v0.yaml --controllers hybrid`: success on a set.
* `scripts/render.py --preset pillar_and_blocker --controller hybrid --video`: an MP4.
* `scripts/demo_scene.py --controller hybrid`: Meshcat.

## Iterative upgrades

What I changed along the way, and why:

- **No more (A) as the starting point.** (C) depended too much on (A) solving the task, so it wasn't much of an improvement. Now the initial guess is just a straight line from the start to an IK grasp pose.
- **More initial guesses.** A straight line can be a bad start too, so we add 3 paths through a random configuration halfway (the straight-line midpoint with every joint moved by up to 1 rad).
- **Speed.** It was still too slow (planning took about 4 s, and the arm got hit before it finished), so:
  - the initial guesses are solved in parallel;
  - the halfway points are made collision-free, which gives better starts;
  - constraint gradients are built in numpy and handed to Drake as AutoDiff: 3–4× faster solves;
  - the optimality tolerance went from 1e-3 to 0.1, the biggest gain. That sometimes gives really bad plans, so while there's time left each plan is re-solved at a tighter tolerance, starting from that solution, and we keep the lowest-cost plan across all guesses.
- **Dodging and holding.** Obstacles coming at the parked arm would still hit it, so a small `_dodge` optimisation moves it out of the way first. At the other end, `_plan_hold` closes, lifts and holds the target until the end of the scene.
- **Better grasp times.** Grasp times are now tried starting from when the target passes closest to the robot, instead of from the earliest one.
  - **Back to an early grasp.** That broke slow targets (their late grasps gave long guesses SNOPT couldn't solve), so one guess now always grasps at the earliest time that works.
  - **Finer grasp times.** If no grid time gives a grasp, we try 0.1 s steps around the two nearest misses (reach error + clearance shortfall).
- **No giving up.** After a failed plan (C) keeps replanning until the target can't be grasped any more, each time with new random vias, new grasp times and at least 2 guesses grasping within 5 s. Together these fixed 28 of the 45 failed scenes, and none of the 17 control scenes broke.
