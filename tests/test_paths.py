import numpy as np

from dyngrasp.paths import LinearPath, ParabolicPath, Path, StationaryPath


def test_linear_path_clamps_and_velocity():
    p = LinearPath([0, 0, 0], [1, 0, 0], 2.0, 4.0)
    assert np.allclose(p.position(0.0), [0, 0, 0])
    assert np.allclose(p.position(3.0), [0.5, 0, 0])
    assert np.allclose(p.position(10.0), [1, 0, 0])
    assert np.allclose(p.velocity(3.0), [0.5, 0, 0])
    assert np.allclose(p.velocity(1.0), 0) and np.allclose(p.velocity(5.0), 0)
    assert p.speed == 0.5
    assert np.isclose(p.time_at(0, 0.25), 2.5)
    assert np.isinf(p.time_at(1, 0.25))


def test_parabolic_path_is_consistent_with_finite_differences():
    p = ParabolicPath([0, 0, 1], [1, 0, 0], [0, 0, -2], 5.0, 4.0, 6.0)
    assert np.allclose(p.position(5.0), [0, 0, 1])
    h = 1e-5
    v_fd = (p.position(5.5 + h) - p.position(5.5 - h)) / (2 * h)
    assert np.allclose(v_fd, p.velocity(5.5), atol=1e-5)


def test_round_trip_serialisation():
    for p in [
        StationaryPath([1, 2, 3]),
        LinearPath([0, 0, 0], [1, 1, 1], 0.0, 1.0),
        ParabolicPath([0, 0, 1], [1, 0, 0], [0, 0, -2], 5.0, 4.0, 6.0),
    ]:
        q = Path.from_dict(p.to_dict())
        for t in [-1.0, 0.3, 4.5, 7.0]:
            assert np.allclose(p.position(t), q.position(t))
            assert np.allclose(p.velocity(t), q.velocity(t))
