"""Grasp candidates: geometry and how the planner uses them."""

import numpy as np
import pytest
from pydrake.all import RigidTransform

from dyngrasp import scenario as S
from dyngrasp.controller import RobotModel
from dyngrasp.grasps import MAX_WIDTH, PAD_Y, generate_grasps, grasp_is_valid, prior_score
from dyngrasp.objects import OBJECTS
from dyngrasp.planning.goals import GoalGenerator
from dyngrasp.world import WorldModel


@pytest.mark.parametrize("name", sorted(OBJECTS))
def test_candidates_are_geometrically_valid(name):
    obj = OBJECTS[name]
    gs = generate_grasps(obj)
    assert gs and gs[0].name == "top_y+" and len({g.name for g in gs}) == len(gs)
    half = np.array([obj.footprint[0] / 2, obj.footprint[1] / 2, obj.height / 2])
    for g in gs:
        assert grasp_is_valid(obj, g.X_UG)
        assert g.width <= MAX_WIDTH + 1e-9
        R = g.X_UG.rotation().matrix()
        assert abs(np.linalg.det(R) - 1) < 1e-9 and abs(R[:, 0] @ R[:, 1]) < 1e-9
        # pad point within the object's box
        p = g.X_UG @ np.array([0.0, PAD_Y, 0.0]) - [0, 0, obj.height / 2]
        assert np.all(np.abs(p) < half)
    # the nominal candidate has the same orientation as the hand-made objects.X_UG_grasp
    np.testing.assert_allclose(gs[0].X_UG.rotation().matrix(), obj.X_UG_grasp.rotation().matrix(), atol=1e-9)


def test_too_wide_faces_are_not_grasped():
    gs = {g.name for g in generate_grasps(OBJECTS["cracker_box"])}  # 0.158 m along U's x
    assert "top_y+" in gs and not any(n.startswith("top_x") for n in gs)
    # a palm inside the object is rejected
    obj = OBJECTS["soup_can"]
    assert not grasp_is_valid(obj, RigidTransform(obj.X_UG_grasp.rotation(), [0, 0, 0.05]))


def test_prior_prefers_natural_approaches():
    obj = OBJECTS["soup_can"]
    by = {g.name: g for g in generate_grasps(obj)}
    X_WU = RigidTransform([0.63, 0.0, 0.2])
    sh = np.array([0.0, 0.0, 0.36])
    assert prior_score(by["top_y+"], X_WU, sh) == pytest.approx(1.0)
    assert prior_score(by["side_180+"], X_WU, sh) == pytest.approx(1.0)  # from the robot's side
    assert prior_score(by["side_0+"], X_WU, sh) == pytest.approx(-1.0)  # from the far side
    assert prior_score(by["side_90+"], X_WU, sh) == pytest.approx(0.0, abs=1e-9)
    assert 0.3 < prior_score(by["top_y+_tilt+35"], X_WU, sh) < 1.0 or 0.3 < prior_score(by["top_y+_tilt-35"], X_WU, sh) < 1.0


def test_pregrasp_backs_off_along_the_approach():
    robot = RobotModel()
    world = WorldModel(S.no_obstacles(0.1))
    gs = generate_grasps(world.object)
    gg = GoalGenerator(world, robot, robot.v_limits, grasps=gs)
    T = world.t_closest
    for k, g in enumerate(gs):
        d = gg.grasp_pose(T, k).translation() - gg.hover_pose(T, k).translation()
        np.testing.assert_allclose(d, gg.tp.hover * gg.approach_W(T, k), atol=1e-12)
    k_side = next(k for k, g in enumerate(gs) if g.family == "side")
    assert abs(gg.approach_W(T, k_side)[2]) < 1e-9  # side grasps approach horizontally
