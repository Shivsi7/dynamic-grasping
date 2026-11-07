# (C) Hybrid: space-time trajectory optimisation seeded by IK

Controller `hybrid`, in `dyngrasp/hybrid.py` (start with `update`). Tests: `tests/test_controllers.py`.

## Idea

(C) plans the episode once with Drake's `KinematicTrajectoryOptimization` (cubic B-splines, SNOPT),
as two problems sent as one joint trajectory. It does not use (A) or (B).

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
     collision-free IK grasp and hover pose (6 IK starts each).
   * **Seeds:** straight joint-space lines to those poses; the same via a collision-free halfway
     configuration (nearest to the midpoint, then to 3 random ones); straight lines to the
     next 2 grasp times.
   * **Workers:** 4 processes (started in `reset`) solve the seeds in parallel. Each sends its
     plan at tolerance 0.1, then refines it at 0.01 from that solution until the deadline;
     (C) keeps the lowest-cost plan. Inside a daemon process the seeds run in turn, and a
     refine runs only if it likely fits (3x the first solve).
3. A late plan is rejected; (C) replans with 1.5x the time it took. No plan: it stays parked
   (dodging if needed).

## The problems

$q(s) = \sum_j b_j(s) P_j$, $s \in [0,1]$, $t = t_0 + sT$, $\dot q = q'/T$. In all three, on the
control points (so for all $s$): joint limits minus 0.05, $|\dot q| \le 0.6\,v_{max}$,
$|\ddot q| \le a$. Clearance: $d_O \ge 4$ cm to obstacles, 5 mm to cart, floor and target, with the
world posed where the prediction puts it at $t$ (its motion enters the gradient w.r.t. $T$).

**Approach** (12 control points, $a = 4$):

$$
\begin{aligned}
\min_{P,T}\quad & T + 0.1 \textstyle\sum_j \lVert r'_j \rVert^2 \\
\text{s.t.}\quad & q(0) = q_0,\ \dot q(0) = v_0, \quad \max(t_{enter} - t_0, 0.3) \le T \le t_{exit} - 1.2 - t_0 \\
& \text{grasp } (s=1):\ \text{pose} \pm 5\,\text{mm} / 0.02\,\text{rad},\ \text{gripper velocity} = v_{obj} \pm 0.02 \\
& \text{hover } (s = s_d):\ 10\,\text{cm above the grasp pose} \pm 1\,\text{cm} / 0.04\,\text{rad} \\
& \text{descent } (3\ \text{points in } (s_d, 1)):\ \pm 1.5\,\text{cm horizontally},\ 0\text{-}12\,\text{cm above} \\
& \text{clearance at 30 samples}
\end{aligned}
$$

$s_d$ is fixed by the seed (0.6 s before its grasp). The grasp target moves with the object, so
moving the grasp later changes where it is.

**Hold** (fixed duration to the scene end, knots every 0.5 s, $a = 2$): $\min \sum_j \lVert r'_j \rVert^2$,
starting in the approach's end state and ending at rest; at the moving grasp pose at 0.2 and 0.4 s
(closing); 10 cm above the grasp height from 1.0 s; clearance every 0.4 s (not to the target).

**Dodge** (from rest at the parked pose to rest at the scene end, $a = 4$): $\min \sum_j \lVert r'_j \rVert^2$,
clearance every 0.1 s. Its budget is half the time to the hit (at most 2 s).

**Solver:** SNOPT, optimality tolerance 0.1, then 0.01 to refine; feasibility 1e-4. Constraint
gradients are built with numpy and handed to Drake as AutoDiff (3-4x faster). A loose solve can
be feasible but poor (`mix_0009`: the hold swung the arm around and through an obstacle between
samples; refined, it does not).

**Simulation details:** commands every 5 ms (the iiwa driver differentiates them every 5 ms);
links 5 and 7 of the box model overlap past |q6| ~ 1.75 rad, so that pair is filtered.

## Results

Measured latency, scored to the end of each scene; 8 episodes in parallel, then every failure
rerun alone (that result counts). Held-out at `dbf03c2`, the rest at `76c07b2`; neither includes
the closest-approach grasp times (`965444e`).

| set | scenes | success | collision | dropped | timeout |
|---|---|---|---|---|---|
| dev | 20 | 16 | 1 | 0 | 3 |
| post grid | 48 | 37 | 0 | 0 | 11 |
| plank grid | 48 | 38 | 0 | 0 | 10 |
| pillar grid | 18 | 6 | 10 | 0 | 2 |
| held-out mixed | 80 | 66 | 1 | 0 | 13 |

Held-out: (A) 52, (B) 63. (C) fails where another succeeds only in `mix_0044` and `mix_0049`.

## Limitations

1. **No grasp found:** most timeouts have no collision-free IK grasp or hover pose (an obstacle
   at the grasp, or the hover out of reach).
2. **Compute:** while (C) computes, the parked arm cannot dodge; a failing seed can take several
   seconds (most pillar-grid collisions: a first plan of 2-5.7 s).
3. **One plan:** no replanning, fine with perfect prediction.

## Tools

* `scripts/run_eval.py --scenarios scenarios/dev_v0.yaml --controllers hybrid`: success on a set.
* `scripts/render.py --preset pillar_and_blocker --controller hybrid --video`: an MP4.
* `scripts/demo_scene.py --controller hybrid`: Meshcat.
