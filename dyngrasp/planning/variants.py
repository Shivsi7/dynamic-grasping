"""Named configurations of (B), shared by the scripts (stress tests, evaluation, videos).

    VARIANTS[name] = Variant(label, factory, episode_kw)

`factory()` builds a fresh controller; `episode_kw` are extra `run_episode` / `build_episode`
arguments the variant needs (e.g. the old stitching rule).  The certification oracle
(scripts/certify.py) is defined there, next to how it is used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from dyngrasp.planning.config import STRRTConfig
from dyngrasp.planning.controller import SpaceTimeRRTController
from dyngrasp.planning.goals import TailParams


@dataclass
class Variant:
    label: str
    factory: Callable[[], SpaceTimeRRTController]
    episode_kw: dict = field(default_factory=dict)


def _cfg(**kw) -> Callable[[], SpaceTimeRRTController]:
    return lambda: SpaceTimeRRTController(STRRTConfig(**kw))


VARIANTS: dict[str, Variant] = {
    "st_rrt": Variant("(B) now", SpaceTimeRRTController),
    "st_rrt_nominal": Variant("(B) nominal grasp only", _cfg(grasp_set="nominal")),
    "st_rrt_noaccel": Variant("(B) no accel. limits", _cfg(a_max=None)),
    "st_rrt_slowtail": Variant("(B) slower grasp tail", _cfg(tail=TailParams(t_descend=0.7, t_lift=0.8))),
    # Commit 72b3097 (tag ckpt-safe-hold): an unsafe grasp end is rejected (no planned retreat),
    # and the held object is not collision-checked.
    "st_rrt_ckpt": Variant("(B) checkpoint: no retreat, object unchecked",
                           _cfg(retreat=False, check_held_object=False)),
    # As of commit bde69c7: one acceleration limit for all joints (4 / 6 / 1.5 rad/s^2), no
    # post-grasp safety, no torque check, and late plans that were already moving spliced in
    # (no velocity check at the stitch).  The goal-work scheduling fix (mix_0073) is not undone.
    "st_rrt_before": Variant("(B) before: 4/6 rad/s2 all joints",
                             _cfg(a_max=4.0, tail_a_max=6.0, a_max_holding=1.5, post_grasp_hold=0.0,
                                  escape_fallback=False, torque_check=False, retreat=False,
                                  check_held_object=False),
                             {"stitch_vtol": np.inf}),
}
