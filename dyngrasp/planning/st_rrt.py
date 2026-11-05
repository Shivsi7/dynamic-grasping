"""Space-time RRT-Connect (informed by ST-RRT*, Grothe et al. 2022), generic in the dimension.

States are (q, t).  An edge (q_a, t_a) -> (q_b, t_b) is admissible iff
    t_b > t_a                                   (time only moves forward)
    |q_b - q_a|_j <= v_max_j (t_b - t_a)  for all j   (joint-velocity limits)
    edge_free(q_a, t_a, q_b, t_b)               (collision check, supplied by the caller)
Linear interpolation keeps |dq|/dt constant along an edge, so steering along
the segment towards a reachable sample never violates the velocity limits.

Bidirectional: the start tree grows forward in time from (q_start, t_start);
the goal tree grows backward in time from one or more goal roots (q_g, T_g)
(e.g. pre-grasp configurations at candidate intercept times).  Samples are
drawn only inside the space-time "double cone" of states that can be reached
from the start in time *and* can still reach some goal in time, which is
where all the useful samples are (ST-RRT*'s main sampling trick).

Acceleration (optional, `a_max`): the executed trajectory is the C1 cubic
Hermite curve through the knots (dyngrasp.trajectory.hermite), whose knot
velocities lie between the adjacent segment velocities.  Its acceleration on a
segment is then at most (4|dv_in| + 2|dv_out|) / h, so requiring at every knot

    |v_out - v_in|_j <= a_max_j * min(h_in, h_out) / 6          (per joint)

keeps it within a_max.  The check is local (the node's own velocity and segment
duration are stored in the tree), so it is applied while extending, connecting
and shortcutting.  Rejecting extensions that violate it would waste almost all
samples (a random sample rarely matches the parent's velocity), so extensions
*project* the edge velocity into the admissible window around the parent's
velocity instead (kinodynamic-style steering), and the trees are joined by
exact single-edge "shots" that must satisfy the bound at both ends.  The start
knot's v_in is the velocity of the trajectory being executed and a goal root's
v_out is the velocity the grasp tail starts with, so the plan joins both without
a velocity jump; an escape must end at rest.  (The bound limits acceleration,
not position: a long segment that must end at a given velocity can bulge away
from its straight edge, so the smoothed curve is re-verified; see motion.py.)

Nothing here knows about Drake, the iiwa or the scenario: the planner only
sees joint limits, velocity (and acceleration) limits, goal roots and a
collision callback.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

EdgeFree = Callable[[np.ndarray, float, np.ndarray, float], bool]


def min_duration(qa: np.ndarray, qb: np.ndarray, v_max: np.ndarray) -> np.ndarray | float:
    """Shortest time in which the joints can move from qa to qb (vectorised over leading axes)."""
    return np.max(np.abs(qb - qa) / v_max, axis=-1)


@dataclass
class STRRTParams:
    step_q: float = 0.35  # max joint change (inf-norm, rad) per extension
    step_t: float = 0.6  # max time change per extension (s)
    goal_bias: float = 0.1  # probability of sampling a goal root (start tree) / the start (goal tree)
    time_weight: float = 0.25  # weight of the time gap in the nearest-neighbour metric
    max_nodes: int = 20000
    max_connect_steps: int = 50


@dataclass
class PlanResult:
    success: bool
    path_q: np.ndarray | None = None  # (N, n)
    path_t: np.ndarray | None = None  # (N,)
    goal_index: int | None = None  # which goal root the path ends at
    iterations: int = 0
    nodes: int = 0
    edge_checks: int = 0
    rejected_samples: int = 0
    wall: float = 0.0
    reason: str = ""
    stats: dict = field(default_factory=dict)


class _Tree:
    """Nodes (q, t) with parents; direction +1 grows forward in time, -1 backward."""

    def __init__(self, n: int, cap: int, direction: int):
        self.Q = np.empty((cap, n))
        self.T = np.empty(cap)
        self.parent = np.empty(cap, dtype=int)
        self.root = np.empty(cap, dtype=int)  # index of the root this node descends from
        # velocity of the node's edge to its parent (forward in time), and that edge's duration;
        # for a root: the boundary velocity, and an infinite duration
        self.U = np.empty((cap, n))
        self.H = np.empty(cap)
        self.size = 0
        self.direction = direction

    def add(self, q, t, parent: int, u=0.0, h=np.inf) -> int:
        i = self.size
        if i >= len(self.T):
            raise OverflowError
        self.Q[i], self.T[i], self.parent[i] = q, t, parent
        self.U[i], self.H[i] = u, h
        self.root[i] = i if parent < 0 else self.root[parent]
        self.size += 1
        return i

    def nearest_reachable(self, q, t, v_max, time_weight) -> int:
        """Closest node from which (q, t) is reachable in this tree's direction; -1 if none."""
        Q, T = self.Q[: self.size], self.T[: self.size]
        dt = self.direction * (t - T)
        need = min_duration(Q, q, v_max)
        ok = (dt > 1e-9) & (need <= dt + 1e-12)
        if not np.any(ok):
            return -1
        cost = np.where(ok, need + time_weight * dt, np.inf)
        return int(np.argmin(cost))

    def branch(self, i: int) -> list[int]:
        out = []
        while i >= 0:
            out.append(i)
            i = self.parent[i]
        return out


class SpaceTimeRRTConnect:
    def __init__(
        self,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
        v_max: np.ndarray,
        edge_free: EdgeFree,
        params: STRRTParams | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.lo = np.asarray(q_lower, float)
        self.hi = np.asarray(q_upper, float)
        self.v_max = np.asarray(v_max, float)
        self._edge_free_fn = edge_free
        self.p = params or STRRTParams()
        self.rng = rng or np.random.default_rng(0)
        self.edge_checks = 0
        self.a_max: np.ndarray | None = None  # joint accelerations; None = velocity limits only
        self.n_accel_rejects = 0

    # ---- helpers ------------------------------------------------------------------
    def _edge_free(self, qa, ta, qb, tb) -> bool:
        """Always called with ta < tb."""
        self.edge_checks += 1
        return self._edge_free_fn(qa, ta, qb, tb)

    def admissible(self, qa, ta, qb, tb) -> bool:
        return tb > ta and min_duration(qa, qb, self.v_max) <= tb - ta + 1e-12

    def jump_ok(self, dv, h_in: float, h_out: float) -> bool:
        """Velocity change at a knot between segments of durations h_in and h_out (see module doc)."""
        if self.a_max is None:
            return True
        ok = bool(np.all(np.abs(dv) <= self.a_max * min(h_in, h_out) / 6.0 + 1e-9))
        if not ok:
            self.n_accel_rejects += 1
        return ok

    def _steer(self, qa, ta, qb, tb):
        """Point on the segment (qa, ta) -> (qb, tb) at most one step away, and whether it is qb."""
        dq, dt = qb - qa, tb - ta
        f = 1.0
        mq = float(np.max(np.abs(dq)))
        if mq > self.p.step_q:
            f = min(f, self.p.step_q / mq)
        if abs(dt) > self.p.step_t:
            f = min(f, self.p.step_t / abs(dt))
        return qa + f * dq, ta + f * dt, f >= 1.0

    def _ordered_edge_free(self, tree: _Tree, i: int, q, t) -> bool:
        qa, ta = tree.Q[i], tree.T[i]
        return self._edge_free(qa, ta, q, t) if ta < t else self._edge_free(q, t, qa, ta)

    def _extend(self, tree: _Tree, q, t):
        """Grow `tree` one step towards (q, t).  Returns (status, node) with status in
        'trapped' (no reachable node or blocked), 'advanced', 'reached'."""
        i = tree.nearest_reachable(q, t, self.v_max, self.p.time_weight)
        if i < 0:
            return "trapped", -1
        qn, tn, reached = self._steer(tree.Q[i], tree.T[i], q, t)
        h = abs(tn - tree.T[i])
        w = (qn - tree.Q[i]) / (tn - tree.T[i])  # velocity of the new edge, forward in time
        if self.a_max is not None:
            # project the velocity into the admissible window around the node's velocity
            lim = self.a_max * min(tree.H[i], h) / 6.0
            w_p = np.clip(np.clip(w, tree.U[i] - lim, tree.U[i] + lim), -self.v_max, self.v_max)
            if np.any(np.abs(w_p - w) > 1e-12):
                reached = False
                w = w_p
                qn = tree.Q[i] + w * (tn - tree.T[i])
        if not (self.lo - 1e-9 <= qn).all() or not (qn <= self.hi + 1e-9).all():
            return "trapped", -1
        if not self._ordered_edge_free(tree, i, qn, tn):
            return "trapped", -1
        j = tree.add(qn, tn, i, w, h)
        return ("reached" if reached else "advanced"), j

    def _shot(self, start: _Tree, i_s: int, goal: _Tree, i_g: int) -> bool:
        """Can start node i_s be joined to goal node i_g by one straight edge?"""
        qa, ta, qb, tb = start.Q[i_s], start.T[i_s], goal.Q[i_g], goal.T[i_g]
        if not self.admissible(qa, ta, qb, tb):
            return False
        h = tb - ta
        w = (qb - qa) / h
        return (self.jump_ok(w - start.U[i_s], start.H[i_s], h) and self.jump_ok(goal.U[i_g] - w, h, goal.H[i_g])
                and self._edge_free(qa, ta, qb, tb))

    def _connect(self, a: _Tree, j: int, b: _Tree, start: _Tree):
        """Grow `b` towards node j of `a`, trying to join them with a shot after every step.
        Returns (start node, goal node) of a successful join, or None."""
        q, t = a.Q[j], a.T[j]

        def join(k):
            return (k, j) if b is start else (j, k)

        k = b.nearest_reachable(q, t, self.v_max, self.p.time_weight)
        for _ in range(self.p.max_connect_steps):
            if k < 0:
                return None
            i_s, i_g = join(k)
            goal = a if b is start else b
            if self._shot(start, i_s, goal, i_g):
                return i_s, i_g
            status, k = self._extend(b, q, t)
            if status == "trapped":
                return None
        return None

    def _sample(self, q_start, t_start, goal_q, goal_t, t_max):
        """Uniform q, then t uniform in the reachable window; None if the window is empty."""
        q = self.rng.uniform(self.lo, self.hi)
        t_lo = t_start + min_duration(q_start, q, self.v_max)
        if len(goal_t):
            t_hi = float(np.max(goal_t - min_duration(goal_q, q, self.v_max)))
        else:
            t_hi = t_max
        t_hi = min(t_hi, t_max)
        if t_lo >= t_hi:
            return None
        return q, float(self.rng.uniform(t_lo, t_hi))

    # ---- planning -----------------------------------------------------------------
    def plan(
        self,
        q_start: np.ndarray,
        t_start: float,
        goal_q: np.ndarray,
        goal_t: np.ndarray,
        time_budget: float,
        t_max: float | None = None,
        v_start: np.ndarray | None = None,
        goal_v: np.ndarray | None = None,
    ) -> PlanResult:
        """Plan from (q_start, t_start) to any goal root (goal_q[k], goal_t[k]) within `time_budget` s.
        v_start / goal_v: boundary velocities (default rest), used with `a_max`."""
        wall0 = time.perf_counter()
        self.edge_checks = 0
        q_start = np.asarray(q_start, float)
        goal_q = np.asarray(goal_q, float).reshape(-1, len(self.lo))
        goal_t = np.asarray(goal_t, float).reshape(-1)
        t_max = float(np.max(goal_t)) if t_max is None else t_max
        n = len(self.lo)
        res = PlanResult(success=False)
        if len(goal_t) == 0:
            res.reason = "no goals"
            return res

        v_start = np.zeros(n) if v_start is None else np.asarray(v_start, float)
        goal_v = np.zeros((len(goal_t), n)) if goal_v is None else np.asarray(goal_v, float).reshape(-1, n)
        self.n_accel_rejects = 0
        start = _Tree(n, self.p.max_nodes, +1)
        start.add(q_start, t_start, -1, v_start, np.inf)
        goal = _Tree(n, self.p.max_nodes, -1)
        for qg, tg, vg in zip(goal_q, goal_t, goal_v):
            goal.add(qg, tg, -1, vg, np.inf)

        # A direct connection to some goal (cheap, and common when nothing is in the way).
        for k in np.argsort(goal_t):
            if self._shot(start, 0, goal, int(k)):
                return self._finish(res, start, 0, goal, int(k), wall0, 0)

        trees = (start, goal)
        it = 0
        try:
            while time.perf_counter() - wall0 < time_budget:
                it += 1
                a, b = trees
                if self.rng.random() < self.p.goal_bias:
                    # bias towards the other tree's roots
                    if a is start:
                        k = int(self.rng.integers(len(goal_t)))
                        s = (goal_q[k], goal_t[k])
                    else:
                        s = (q_start, t_start)
                else:
                    s = self._sample(q_start, t_start, goal_q, goal_t, t_max)
                    if s is None:
                        res.rejected_samples += 1
                        continue
                status, j = self._extend(a, *s)
                if status != "trapped":
                    joined = self._connect(a, j, b, start)
                    if joined is not None:
                        return self._finish(res, start, joined[0], goal, joined[1], wall0, it)
                trees = (b, a)
        except OverflowError:
            res.reason = "tree full"
        res.reason = res.reason or "time budget exhausted"
        res.iterations, res.nodes, res.edge_checks = it, start.size + goal.size, self.edge_checks
        res.wall = time.perf_counter() - wall0
        res.stats["accel_rejects"] = self.n_accel_rejects
        return res

    def _finish(self, res, start, i_start, goal, i_goal, wall0, it) -> PlanResult:
        """Path = start root .. i_start, then i_goal .. goal root (a meeting node in both is kept once)."""
        fwd = start.branch(i_start)[::-1]
        Qs = [start.Q[i] for i in fwd]
        Ts = [start.T[i] for i in fwd]
        bwd = goal.branch(i_goal)
        Qs += [goal.Q[i] for i in bwd]
        Ts += [goal.T[i] for i in bwd]
        Q, T = np.array(Qs), np.array(Ts)
        keep = np.concatenate([[True], np.diff(T) > 1e-9])  # drop duplicated meeting nodes
        res.success = True
        res.path_q, res.path_t = Q[keep], T[keep]
        res.goal_index = int(goal.root[bwd[-1]]) if bwd else None
        res.iterations, res.nodes, res.edge_checks = it, start.size + goal.size, self.edge_checks
        res.wall = time.perf_counter() - wall0
        res.reason = "ok"
        res.stats["accel_rejects"] = self.n_accel_rejects
        return res

    # ---- post-processing ----------------------------------------------------------
    def shortcut(self, Q: np.ndarray, T: np.ndarray, time_budget: float, max_tries: int = 200,
                 v_start: np.ndarray | None = None, v_end: np.ndarray | None = None):
        """Time-preserving shortcutting: replace i..j by a straight (q, t) edge when it is
        admissible, respects the knot acceleration bound at i and j, and is collision-free.
        Knot times are kept, so the arrival time at the goal does not change."""
        Q, T = Q.copy(), T.copy()
        n = Q.shape[1]
        v_start = np.zeros(n) if v_start is None else v_start
        v_end = np.zeros(n) if v_end is None else v_end

        def seg(k):  # velocity and duration of segment k -> k+1 (boundary values outside)
            if k < 0:
                return v_start, np.inf
            if k >= len(T) - 1:
                return v_end, np.inf
            return (Q[k + 1] - Q[k]) / (T[k + 1] - T[k]), T[k + 1] - T[k]

        wall0 = time.perf_counter()
        for _ in range(max_tries):
            if len(T) <= 2 or time.perf_counter() - wall0 > time_budget:
                break
            i, j = sorted(self.rng.choice(len(T), size=2, replace=False))
            if j - i < 2:
                continue
            h = T[j] - T[i]
            w = (Q[j] - Q[i]) / h
            (u_in, h_in), (u_out, h_out) = seg(i - 1), seg(j)
            if (self.admissible(Q[i], T[i], Q[j], T[j]) and self.jump_ok(w - u_in, h_in, h)
                    and self.jump_ok(u_out - w, h, h_out) and self._edge_free(Q[i], T[i], Q[j], T[j])):
                Q = np.vstack([Q[: i + 1], Q[j:]])
                T = np.concatenate([T[: i + 1], T[j:]])
        return Q, T


def plan_to_safe_state(
    planner: SpaceTimeRRTConnect,
    q_start: np.ndarray,
    t_start: float,
    is_safe: Callable[[np.ndarray, float], bool],
    t_max: float,
    time_budget: float,
    v_start: np.ndarray | None = None,
) -> PlanResult:
    """Single-tree escape: grow forward from the start until a node is reached from which
    staying still is collision-free (is_safe), e.g. to get out of the way of an obstacle.
    With `a_max`, the node must also be one the arm can stop at (velocity -> 0)."""
    wall0 = time.perf_counter()
    planner.edge_checks = 0
    n = len(planner.lo)
    v_start = np.zeros(n) if v_start is None else np.asarray(v_start, float)
    tree = _Tree(n, planner.p.max_nodes, +1)
    tree.add(np.asarray(q_start, float), t_start, -1, v_start, np.inf)
    res = PlanResult(success=False)
    at_rest = planner.a_max is None or float(np.max(np.abs(v_start))) < 1e-6
    if at_rest and is_safe(q_start, t_start):
        res.success, res.path_q, res.path_t, res.reason = True, np.array([q_start]), np.array([t_start]), "ok"
        return res
    it = 0
    try:
        while time.perf_counter() - wall0 < time_budget:
            it += 1
            s = planner._sample(q_start, t_start, np.zeros((0, n)), np.zeros(0), t_max)
            if s is None:
                res.rejected_samples += 1
                continue
            status, j = planner._extend(tree, *s)
            if (status != "trapped" and planner.jump_ok(-tree.U[j], tree.H[j], np.inf)
                    and is_safe(tree.Q[j], tree.T[j])):
                idx = tree.branch(j)[::-1]
                res.success, res.reason = True, "ok"
                res.path_q, res.path_t = tree.Q[idx].copy(), tree.T[idx].copy()
                break
    except OverflowError:
        res.reason = "tree full"
    res.reason = res.reason or "time budget exhausted"
    res.iterations, res.nodes, res.edge_checks = it, tree.size, planner.edge_checks
    res.wall = time.perf_counter() - wall0
    return res
