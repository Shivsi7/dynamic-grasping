# (A) Reactive QP controller

Controller `reactive`, `ReactiveQPController` in `dyngrasp/reactive.py` (start with `step` and `qp`);
distance rows from `CollisionWorld.rows` in `dyngrasp/collision.py`. Tests: `tests/test_controllers.py`.

## Idea

(A) never plans. Every 10 ms it solves one QP for the joint velocities that track a velocity-matched
gripper pose, with a velocity damper per robot-world pair near the world's predicted poses over the
next 0.2 s (its swept volume). Fast (about 3 ms per cycle) but local: it can stall behind an obstacle.

## One cycle (`update` → `step` → `qp`)

1. Phase: wait → approach → descend → close → lift (table below; lift is final).
2. Desired gripper twist $V$: P control on the error to the phase's target pose, plus a feed-forward.
3. Distance rows: the robot at its current $q$ against the predicted world at $t$, $t + 0.1$, $t + 0.2$ s.
4. Solve the QP; send $\dot q$ and the gripper opening.

## The problem

Defaults are `ReactiveQPController` arguments (in backticks); the other numbers are hard-coded. World
frame throughout; vector inequalities hold per component.

**Notation**

* $q$, $X_G = (R_G, p_G)$: measured joint positions and WSG body pose at $t$; $J(q) \in \mathbb{R}^{6 \times 7}$:
  the WSG body's spatial-velocity Jacobian, rows [angular; linear] (`RobotModel.jacobian`).
* $X^*(t) = (R^*, p^*(t))$: the object's top-down grasp pose (`WorldModel.grasp_pose`); $v_{obj}(t)$: the
  cart's velocity; $[t_{enter}, t_{exit}]$: when the object is in reach; $\hat z$: up.
* Error to a target pose $(R_d, p_d)$ (`pose_error`): $e_R = \theta k$, the axis-angle of $R_d R_G^\top$;
  $e_p = p_d - p_G$.

**Phases** ($h$ = `hover`, 0.10 m; gripper open = 0.107 m, closed = 0; $t_{close}$: when close began)

| phase | target $(R_d, p_d)$ | $v_{ff}$ | gripper | next phase when |
|---|---|---|---|---|
| wait | $X_{rest}$ ($X_G$ at `reset`) | 0 | open | $t \ge t_{enter} - 1$ (`lead_time`) |
| approach | $(R^*,\ p^*(t) + h\hat z)$ | $v_{obj}(t)$ | open | $t_{enter} \le t \le t_{exit}$, $\lVert e_p \rVert < 0.02$, $\lVert e_R \rVert < 0.1$ |
| descend | $X^*(t)$ | $v_{obj}(t)$ | open | $\lVert e_p \rVert < 0.015$ |
| close | $X^*(t)$ | $v_{obj}(t)$ | closed | $t - t_{close} > 0.4$ (`close_time`) |
| lift | $(R^*,\ p_L(t))$ | $\dot p_L(t)$ | closed | final |

**Lift reference** (`lift_reference`): from the lift's start $t_L$, with $p_0 = p_G(t_L)$, $v_h$ the horizontal
part of $v_{obj}(t_L)$, $s = \mathrm{clip}\big((t - t_L)/T_L,\ 0,\ 1\big)$ and $\sigma(s) = 10s^3 - 15s^4 + 6s^5$:

$$
p_L = p_0 + T_L\, v_h \big(s - 2.5s^4 + 3s^5 - s^6\big) + H\,\sigma(s)\,\hat z, \qquad
\dot p_L = \big(1 - \sigma(s)\big)\, v_h + \frac{H}{T_L}\,\sigma'(s)\,\hat z
$$

$T_L$ = `lift_time` (0.6 s), $H$ = `lift_height` (0.10 m); after $T_L$ it holds $p_0 + 0.3\,v_h + H\hat z$.

**Desired twist**

$$
V = \begin{bmatrix} k_{rot}\, e_R \\ \mathrm{sat}\big(k_p\, e_p + v_{ff}\big) \end{bmatrix}, \qquad
\mathrm{sat}(x) = x \min\big(1,\ 0.8 / \lVert x \rVert\big)
$$

$k_{rot}$ = `k_rot` (4 1/s), $k_p$ = `k_p` (4 1/s), 0.8 m/s = `max_speed` (the angular part is not capped).

**Distance rows** (`CollisionWorld.rows`): for each $\tau \in \{t,\ t + 0.1,\ t + 0.2\}$ (`horizon` 0.2 s,
`n_horizon` 3), the world posed and moving as predicted at $\tau$ and the robot at $q$ (iiwa links 1-7, WSG
body, fingers modelled open), one row per robot-world pair $i$ with signed distance $\phi_i \le 0.25$ m
(`d_influence`); world = obstacles, cart, floor, target (WSG-target pairs excluded). $m$ rows in all.

* $n_i$: unit normal from the world witness point towards the robot one.
* $a_i = n_i^\top J_{r,i}(q) \in \mathbb{R}^{1 \times 7}$, $J_{r,i}$: translational Jacobian of the robot witness point.
* $b_i = n_i^\top v_{w,i}$, $v_{w,i}$: predicted velocity of the world witness point at $\tau$ (> 0: towards the robot).
* $d_i$ = `d_safe` (0.04 m) for obstacles, `d_safe_env` (0.005 m) for the cart, floor and target.

Since $\dot\phi_i = a_i \dot q - b_i$, the damper below lets a pair close in at most at $\xi(\phi_i - d_i)$, and
pushes it apart inside $d_i$.

**The QP** (`qp`), over $\dot q \in \mathbb{R}^7$ and slacks $s \in \mathbb{R}^m$:

$$
\begin{aligned}
\min_{\dot q,\ s}\quad & \lVert W (J \dot q - V) \rVert^2 + \lambda \lVert \dot q \rVert^2 + \mu \lVert \dot q - \dot q_{prev} \rVert^2
  + 100 \textstyle\sum_i s_i + 1000 \lVert s \rVert^2 \\
\text{s.t.}\quad & a_i \dot q + s_i \ge -\xi (\phi_i - d_i) + b_i, \quad s_i \ge 0, \qquad i = 1, \dots, m \\
& \mathrm{clip}\Big(\frac{q_{lo} + 0.05 - q}{0.2},\ -\bar v,\ \bar v\Big) \le \dot q \le
  \mathrm{clip}\Big(\frac{q_{hi} - 0.05 - q}{0.2},\ -\bar v,\ \bar v\Big)
\end{aligned}
$$

* $W = I_6$; in lift $W = \mathrm{diag}(1, 1, 1, 1, 1, 10)$ (`lift_z_weight`): the held arm gives way sideways, not down.
* $\lambda$ = `lam` (5e-3), $\mu$ = `mu` (5e-3), $\xi$ = `xi` (2 1/s); $\dot q_{prev}$: the previous cycle's $\dot q$
  (0 after `reset` or a failed solve).
* Joint limits $q_{lo}, q_{hi}$: the iiwa's, 0.05 rad margin; $\bar v = 0.6\, v_{max}$ (`v_scale`), $v_{max}$: datasheet speeds.
* Slacks: zero unless dampers conflict, so the QP is always feasible; none if $m = 0$.
* Solver: OSQP (default settings) on a Drake `MathematicalProgram` built every cycle; if it fails, $\dot q = 0$.

**Sent to the robot** (`Command(v, gripper)`): every 10 ms the wrapper sets
$q_{cmd} \leftarrow \mathrm{clip}(q_{cmd} + 0.01\, \dot q,\ q_{lo},\ q_{hi})$ and the gripper opening. A command computed in
$w$ seconds takes effect at the first tick at or after $t + w$ (measured latency); until then the previous $\dot q$ is
integrated and (A) is not called.

## Results

Commit `76c07b2` ((A)'s code unchanged since). Measured latency, scored to the end of each scene; 8
episodes in parallel on a 10-core machine, one run per scene. Sets in `scenarios/`: `dev_v0.yaml`,
`preset_grids.yaml` (post, plank, pillar), `heldout_mixed_s5000.yaml`.

| set | scenes | success | collision | dropped | timeout |
|---|---|---|---|---|---|
| dev | 20 | 11 | 3 | 1 | 5 |
| post grid | 48 | 34 | 0 | 0 | 14 |
| plank grid | 48 | 33 | 0 | 0 | 15 |
| pillar grid | 18 | 2 | 0 | 0 | 16 |
| held-out mixed | 80 | 52 | 4 | 4 | 20 |

"Dropped": lifted, then lost, or no longer lifted (5 cm) when the scene ends. Median peak commanded
joint acceleration 100-190 rad/s²; median peak torque 1.3-3.2x the joint limits; slowest cycle of an
episode 15-25 ms (median).

## Limitations

1. **Local minima:** behind an obstacle it must go around (the pillar) it stops and waits: 2 of 18 grasped.
2. **Holding:** the dampers push the held arm away from an obstacle and it swings back; that can shake
   the object loose (`mix_0061`).
3. **No acceleration limits:** phase switches are velocity steps: peaks of 100-300 rad/s², above the torque limits.
4. **No guaranteed margin:** the slacks let dampers give way when they conflict.

## Tools

* `scripts/run_eval.py --scenarios scenarios/dev_v0.yaml --controllers reactive`: success on a set.
* `scripts/render.py --preset tall_blocker --controller reactive --video`: an MP4 (needs `ffmpeg`).
* `scripts/demo_scene.py --controller reactive`: Meshcat.
