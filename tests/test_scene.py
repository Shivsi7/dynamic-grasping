import numpy as np
from pydrake.all import Simulator

from dyngrasp import scenario as S
from dyngrasp.objects import OBJECTS
from dyngrasp.scene import BodyRoles, build_kinematic_diagram, build_station
from dyngrasp.world import WorldModel


def test_scripted_bodies_track_their_paths():
    spec = S.tall_blocker(0.2, 0.3)
    scene = build_station(spec)
    sim = Simulator(scene.station)
    ctx = sim.get_mutable_context()
    scene.set_initial_state(ctx)
    pc = scene.plant.GetMyContextFromRoot(ctx)
    scene.station.GetInputPort("iiwa.position").FixValue(scene.station.GetMyContextFromRoot(ctx), np.array(spec.q0))
    scene.station.GetInputPort("wsg.position").FixValue(scene.station.GetMyContextFromRoot(ctx), [0.1])
    world = WorldModel(spec)
    # (only before the blocker reaches the parked robot, which it cannot push through)
    for t in [1.0, 2.0, 3.0]:
        sim.AdvanceTo(t)
        cart = scene.plant.EvalBodyPoseInWorld(pc, scene.cart_top_body).translation()
        assert np.allclose(cart, world.cart_pose(t).translation(), atol=2e-3), (t, cart)
        obj = scene.plant.EvalBodyPoseInWorld(pc, scene.target_body).translation()
        assert np.allclose(obj[:2], world.target_body_pose(t).translation()[:2], atol=5e-3), (t, obj)
        blocker = scene.plant.EvalBodyPoseInWorld(pc, scene.obstacle_body("blocker")).translation()
        assert np.allclose(blocker, world.obstacle_positions(t)[0], atol=5e-3), (t, blocker)


def test_every_object_rests_upright_on_the_cart():
    for name, info in OBJECTS.items():
        spec = S.no_obstacles(0.1, object=name)
        scene = build_station(spec)
        sim = Simulator(scene.station)
        ctx = sim.get_mutable_context()
        scene.set_initial_state(ctx)
        st = scene.station.GetMyContextFromRoot(ctx)
        scene.station.GetInputPort("iiwa.position").FixValue(st, np.array(spec.q0))
        scene.station.GetInputPort("wsg.position").FixValue(st, [0.1])
        sim.AdvanceTo(1.0)
        pc = scene.plant.GetMyContextFromRoot(ctx)
        X_WO = scene.plant.EvalBodyPoseInWorld(pc, scene.target_body)
        X_WU = X_WO @ info.X_UO.inverse()
        z_up = X_WU.rotation().matrix()[:, 2]
        assert z_up[2] > 0.99, (name, z_up)
        assert abs(X_WU.translation()[2] - spec.cart.top_z) < 0.01, (name, X_WU.translation())


def test_kinematic_diagram_reports_robot_collisions_only():
    spec = S.tall_blocker(0.1, 0.2)
    diagram, kin = build_kinematic_diagram(spec)
    world = WorldModel(spec)
    roles = BodyRoles(kin)
    ctx = diagram.CreateDefaultContext()
    sg_ctx = kin.scene_graph.GetMyContextFromRoot(ctx)
    world.set_kinematic_state(kin, ctx, 0.0, np.array(spec.q0))
    qo = kin.scene_graph.get_query_output_port().Eval(sg_ctx)
    assert roles.robot_collisions(qo) == []
    assert not qo.HasCollisions()  # object-on-cart and cart-over-floor are filtered
    world.set_kinematic_state(kin, ctx, spec.cart.t_enter + 1.0, np.array(spec.q0))  # blocker crossing time
    qo = kin.scene_graph.get_query_output_port().Eval(sg_ctx)
    hits = roles.robot_collisions(qo)
    assert hits and all(h[2] == "obstacle" for h in hits)
    d_obs, d_env = roles.robot_clearance(qo)
    assert d_obs < 0 < d_env
