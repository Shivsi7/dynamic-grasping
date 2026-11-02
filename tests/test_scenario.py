import numpy as np
import pytest

from dyngrasp import scenario as S


def test_cart_timing_is_speed_independent():
    for v in [0.05, 0.1, 0.3]:
        cart = S.make_cart(v)
        assert np.isclose(cart.t_enter, S.WARMUP)
        assert np.isclose(cart.t_exit, S.WARMUP + 2 * S.REACH_Y / v)
        assert cart.t_enter < cart.t_closest < cart.t_exit


def test_random_scenarios_are_reproducible_and_serialisable(tmp_path):
    a = S.generate_scenario_set(7, 3)
    b = S.generate_scenario_set(7, 3)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]
    S.save_scenario_set(a, tmp_path / "set.yaml")
    c = S.load_scenario_set(tmp_path / "set.yaml")
    assert [s.to_dict() for s in a] == [s.to_dict() for s in c]


@pytest.mark.parametrize("seed", range(20))
def test_sampled_obstacle_paths_are_admissible_and_cross_in_time(seed):
    spec = S.sample_scenario(np.random.default_rng(seed), seed=seed)
    for o in spec.obstacles:
        p = o.get_path()
        assert S.path_is_admissible(p, o.bounding_radius, spec.cart, mount=o.mount)
        assert np.ptp([p.position(t)[2] for t in np.linspace(p.t_start, p.t_end, 50)]) < 1e-9  # ground path
        # the obstacle is somewhere in the crossing box while the target is reachable
        ts = np.linspace(spec.cart.t_enter, spec.cart.t_exit, 200)
        pts = np.array([p.position(t) for t in ts])
        inside = np.all((pts >= S.CROSSING_BOX[:, 0] - 1e-9) & (pts <= S.CROSSING_BOX[:, 1] + 1e-9), axis=1)
        assert inside.any()


def test_path_rejects_cart_column_and_base():
    cart = S.make_cart(0.1)
    from dyngrasp.paths import LinearPath

    through_cart = LinearPath([cart.x, -2, 0.3], [cart.x, 2, 0.3], 0, 40)
    assert not S.path_is_admissible(through_cart, 0.05, cart)
    through_base = LinearPath([0.0, -1, 0.4], [0.0, 1, 0.4], 0, 10)
    assert not S.path_is_admissible(through_base, 0.05, cart)
    high_above = LinearPath([cart.x, -1, 1.2], [cart.x, 1, 1.2], 0, 10)
    assert S.path_is_admissible(high_above, 0.05, cart, mount="float")
    assert not S.path_is_admissible(high_above, 0.05, cart, mount="cart")  # the rig would drive over the conveyor
    corridor = LinearPath([0.28, -1, 0.5], [0.28, 1, 0.5], 0, 10)
    assert S.path_is_admissible(corridor, 0.1, cart, mount="cart")


def test_unknown_object_rejected():
    with pytest.raises(ValueError):
        S.TargetSpec(object="banana")
