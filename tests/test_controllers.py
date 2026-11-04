import numpy as np

from dyngrasp import scenario as S
from dyngrasp.collision import CollisionWorld
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


def test_hybrid_plan_ends_at_the_moving_grasp_pose():
    spec = S.tall_blocker(0.1, 0.2)
    r = run_episode(spec, HybridController(replan_period=100.0))
    plan = next(e for e in r.controller_log if "ok" in e)
    assert plan["ok"] and plan["rollout_reached"]
    t_end = plan["t0"] + plan["T"]
    assert WorldModel(spec).t_enter <= t_end and r.t_grasp is not None and abs(r.t_grasp - t_end) < 0.3
