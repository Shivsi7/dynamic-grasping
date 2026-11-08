import multiprocessing as mp

import numpy as np

from dyngrasp import scenario as S
from dyngrasp.collision import CollisionWorld
from dyngrasp.controller import RobotModel
from dyngrasp.harness import run_episode
from dyngrasp.hybrid import HybridController
from dyngrasp.reactive import ReactiveQPController
from dyngrasp.world import WorldModel


def test_collision_rows_see_the_blocker_coming():
    spec = S.tall_blocker(0.1, 0.2)
    cw = CollisionWorld(spec, WorldModel(spec))
    rows = [r for r in cw.rows(np.asarray(spec.q0), 3.0, 0.3) if r.category == "obstacle"]
    assert rows and all(r.other_body == "blocker" for r in rows)
    assert min(r.phi for r in rows) > 0.05
    # the blocker drives along -y past the arm: some witness point sees it approaching
    assert max(r.vn for r in rows) > 0.0


def test_reactive_and_hybrid_grasp_past_the_tall_blocker():
    for controller in (ReactiveQPController(), HybridController()):
        r = run_episode(S.tall_blocker(0.1, 0.2), controller)
        assert r.outcome == "success" and not r.collided, (controller.name, r.outcome)
        assert r.min_clearance_obstacles > 0.02, controller.name


def test_hybrid_gets_around_the_pillar():
    # planning time free: this optimisation takes ~10 s, so with measured latency (C) misses the pillar scene
    r = run_episode(S.pillar_and_blocker(), HybridController(), latency="none")
    assert r.outcome == "success" and not r.collided and r.min_clearance_obstacles > 0.02


def test_hybrid_plan_ends_at_the_moving_grasp_pose():
    spec = S.tall_blocker(0.1, 0.2)
    r = run_episode(spec, HybridController())
    plan = next(e for e in r.controller_log if "ok" in e)
    assert plan["ok"]
    t_end = plan["t0"] + plan["T"]
    assert WorldModel(spec).t_enter <= t_end and r.t_grasp is not None and abs(r.t_grasp - t_end) < 0.3


def test_hybrid_seed_workers_stop_when_the_episode_ends():
    spec = S.no_obstacles(0.3)
    spec.duration = 1.0
    c, seen = HybridController(n_workers=2), []
    reset = c.reset
    c.reset = lambda *a: (reset(*a), seen.extend(p for p, _ in c._workers))
    run_episode(spec, c)
    assert len(seen) == 2 and not any(p.is_alive() for p in seen) and not mp.active_children()
    # a script that never closes its controller: the next one's reset stops the workers it left
    a, b = HybridController(n_workers=2), HybridController(n_workers=2)
    a.reset(WorldModel(spec), RobotModel(), None)
    left = [p for p, _ in a._workers]
    b.reset(WorldModel(spec), RobotModel(), None)
    assert not any(p.is_alive() for p in left) and len(mp.active_children()) == 2
    b.close()
    assert not mp.active_children()
