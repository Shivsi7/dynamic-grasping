"""(B) the space-time planner on the Drake scene."""

import numpy as np
import pytest

from dyngrasp import scenario as S
from dyngrasp.controller import RobotModel
from dyngrasp.harness import run_episode
from dyngrasp.planning.collision import SpaceTimeChecker
from dyngrasp.planning.controller import SpaceTimeRRTController
from dyngrasp.planning.goals import GoalGenerator
from dyngrasp.world import WorldModel


@pytest.fixture(scope="module")
def robot():
    return RobotModel()


def test_checker_is_conservative_with_respect_to_the_monitor_definition():
    # every configuration the course's collision definition calls a collision is not free
    for spec in [S.tall_blocker(0.1, 0.2)] + S.generate_scenario_set(3, 2, n_obstacles=4):
        world = WorldModel(spec)
        ck = SpaceTimeChecker(spec, world)
        rng = np.random.default_rng(0)
        lo, hi = ck.plant.GetPositionLowerLimits()[ck.i_iiwa], ck.plant.GetPositionUpperLimits()[ck.i_iiwa]
        for _ in range(200):
            q, t = rng.uniform(lo, hi), rng.uniform(0, spec.duration)
            ck.set_state(q, t)
            truth = ck.roles.robot_collisions(ck.port.Eval(ck.sgctx))
            if truth:
                assert not ck.state_free(q, t)


def test_goals_have_feasible_velocity_matched_tails(robot):
    spec = S.no_obstacles(0.2)
    world = WorldModel(spec)
    ck = SpaceTimeChecker(spec, world)
    gg = GoalGenerator(world, robot, 0.9 * robot.v_limits)
    goals = gg.generate(gg.candidate_times(0.0), [np.array(spec.q0)], lambda tr: ck.path_free(tr.q, tr.t),
                        ck.state_free, np.random.default_rng(0))
    assert len(goals) >= 3
    for g in goals:
        assert g.tail.start_time == g.T and np.allclose(g.tail.q[0], g.q)
        assert g.tail.max_joint_speed_ratio(0.9 * robot.v_limits) <= 1.0 + 1e-6
        # the gripper ends on the grasp pose at the end of the descent, moving with the target
        t_g = g.T + gg.tp.t_settle + gg.tp.t_descend
        p = robot.fk(g.tail.value(t_g)).translation()
        assert np.linalg.norm(p - gg.grasp_pose(t_g, 0).translation()) < 0.015
        assert g.tail.gripper_at(g.T) > g.tail.gripper_at(g.tail.end_time)  # opens, then closes


@pytest.mark.parametrize("preset", [S.tall_blocker, S.plank_blocker])
def test_planner_grasps_past_the_blockers_with_latency(preset):
    # a fixed delay keeps the test deterministic in sim time; stress runs use "measured"
    c = SpaceTimeRRTController()
    r = run_episode(preset(0.1, 0.2), c, latency=0.1)
    assert r.outcome == "success" and not r.collided, (r.outcome, r.collision_with)
    assert r.n_rejected_commands == 0
    assert r.min_clearance_obstacles > 0.01
    # replans that returned a plan stayed within their budget (goal-work cycles may overrun:
    # they return nothing and the arm keeps executing a safe plan meanwhile)
    planned = [l for l in c.plan_log[1:] if l["outcome"] == "new"]
    assert all(l["wall"] < l["budget"] for l in planned)


def test_planner_grasps_a_fast_target_without_obstacles():
    r = run_episode(S.no_obstacles(0.3), SpaceTimeRRTController(), latency=0.1)
    assert r.outcome == "success"


def test_planner_gets_out_of_the_way_when_parking_would_be_hit():
    # Held-out scenario (seed 5034) in which an obstacle drives into the home pose: the
    # hold baseline is hit; the planner must move away even though no grasp is found.
    from dyngrasp.baselines import HoldController
    from dyngrasp.objects import OBJECTS

    spec = S.sample_scenario(np.random.default_rng(5034), n_obstacles=4, objects=tuple(OBJECTS), seed=5034)
    assert run_episode(spec, HoldController()).collided
    r = run_episode(spec, SpaceTimeRRTController(), latency=0.1)
    assert not r.collided, (r.t_first_collision, r.collision_with)


def test_executed_motion_respects_the_acceleration_limits():
    c = SpaceTimeRRTController()
    r = run_episode(S.tall_blocker(0.1, 0.2), c, latency=0.1)
    assert r.outcome == "success"
    # commanded joint accelerations within the (per-joint) planner and tail limits (+ 10 % for
    # finite differencing); without the smoothing this was > 100 rad/s^2
    assert r.cmd_accel_max < 1.1 * max(c.cfg.tail_a_max), r.cmd_accel_max
    assert r.cmd_speed_ratio_max < 0.95
    assert r.cmd_torque_ratio_max < 1.0, r.cmd_torque_ratio_max


def test_a_held_object_must_clear_the_obstacles_too():
    spec = S.tall_blocker(0.1, 0.2)
    world = WorldModel(spec)
    ck = SpaceTimeChecker(spec, world)
    q, t = np.asarray(spec.q0, float), 1.0
    assert ck.state_free(q, t)

    # hold the object "inside" the first obstacle: free while not held, not free while held
    ck.set_state(q, t)
    X_WG = ck.plant.EvalBodyPoseInWorld(ck.pctx, ck._gripper_body)
    bodies = [ck.plant.get_body(i) for i in ck.plant.GetBodyIndices(ck.kin.obstacles[spec.obstacles[0].name])]
    body = next(b for b in bodies if ck.plant.GetCollisionGeometriesForBody(b))
    X_GO = X_WG.inverse() @ ck.plant.EvalBodyPoseInWorld(ck.pctx, body)
    with ck.attached((X_GO, t + 0.5)):
        assert ck.state_free(q, t)  # not yet grasped
        assert not ck.state_free(q, t + 0.5)
    assert ck.state_free(q, t)
    # and the attachment from a real state reproduces the predicted object pose
    X_GO2, tc = ck.grasp_attachment(q, t)
    with ck.attached((X_GO2, tc)):
        ck.set_state(q, t)
        X = ck.plant.EvalBodyPoseInWorld(ck.pctx, ck.kin.target_body)
        np.testing.assert_allclose(X.translation(), world.target_body_pose(t).translation(), atol=1e-9)
