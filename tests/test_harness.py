import numpy as np

from dyngrasp import scenario as S
from dyngrasp.baselines import HoldController, NaiveInterceptController
from dyngrasp.controller import Command, Controller, WSG_OPEN
from dyngrasp.harness import aggregate, run_batch, run_episode


def test_naive_controller_grasps_without_obstacles():
    r = run_episode(S.no_obstacles(0.15), NaiveInterceptController())
    assert r.outcome == "success" and r.success and not r.collided
    assert r.t_grasp is not None and r.t_grasp < r.t_lift < r.t_success
    assert r.n_cycles > 100 and r.compute_mean_ms > 0
    assert np.isinf(r.min_clearance_obstacles)  # no obstacles
    assert r.log is not None and len(r.log["t"]) > 100


def test_blockers_cannot_be_ignored_by_a_straight_in_grasp():
    for preset in (S.tall_blocker, S.plank_blocker):
        r = run_episode(preset(0.1, 0.2), NaiveInterceptController())
        assert r.outcome == "collision" and r.collided and not r.success, (preset.__name__, r.outcome)
        assert r.collision_category == "obstacle" and "blocker" in r.collision_with
        assert r.t_first_collision is not None and r.t_end < r.scenario_summary["duration"]
        assert r.min_clearance_obstacles <= 0.005
        # a parked robot is not hit: the blocker is low enough to pass under the home pose
        r = run_episode(preset(0.1, 0.2), HoldController())
        assert r.outcome == "timeout" and not r.collided, (preset.__name__, r.outcome)


def test_velocity_commands_are_integrated_and_clipped():
    class Spin(Controller):
        name = "spin"
        period = 0.02

        def update(self, state, world):
            return Command(v=np.array([10.0, 0, 0, 0, 0, 0, 0]), gripper=WSG_OPEN)

    spec = S.no_obstacles(0.3)
    spec.duration = 1.0
    r = run_episode(spec, Spin())
    q1 = r.log["q"][-1][0]
    assert 1.5 < q1 <= 2.97  # 10 rad/s commanded, iiwa limit ~1.7 rad/s, joint limit 2.967


def test_run_batch_writes_results(tmp_path):
    specs = [S.no_obstacles(0.3), S.tall_blocker(0.3, 0.5)]
    df = run_batch(specs, {"hold": HoldController}, out_dir=tmp_path, verbose=False)
    assert len(df) == 2 and (tmp_path / "results.csv").exists() and (tmp_path / "scenarios.yaml").exists()
    agg = aggregate(df)
    assert agg.loc["hold", "episodes"] == 2
    assert set(df["outcome"]) <= {"success", "collision", "dropped", "timeout"}
