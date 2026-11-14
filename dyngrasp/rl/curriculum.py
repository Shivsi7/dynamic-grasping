"""(D) curriculum: which level the training is at, and the per-level outcome statistics.

Lives in the training process; the environments get the level with `set_level` and draw each
episode's scene level with `sample_level`: the current level with probability `current_share`,
else uniformly one of the earlier levels.  The level goes up once a full window of episodes at the
current level (episodes whose scene was of that level, finished since the level was reached) has a
success rate of at least `advance_at`.
"""

from __future__ import annotations

from collections import deque

import numpy as np

OUTCOMES = ("success", "collision", "dropped", "timeout")


def sample_level(current: int, rng: np.random.Generator, current_share: float) -> int:
    if current == 0 or rng.random() < current_share:
        return current
    return int(rng.integers(0, current))


class Curriculum:
    def __init__(self, n_levels: int, window: int = 100, advance_at: float = 0.5):
        self.n_levels, self.window, self.advance_at = n_levels, window, advance_at
        self.level = 0
        self.at_level: deque[bool] = deque(maxlen=window)  # successes at the current level, since reaching it
        self.recent = [deque(maxlen=window) for _ in range(n_levels)]  # last outcomes per scene level
        self.episodes = [0] * n_levels

    @property
    def final(self) -> bool:
        return self.level == self.n_levels - 1

    def record(self, scene_level: int, outcome: str) -> bool:
        """Count one finished episode; True if that moved the curriculum up a level."""
        assert outcome in OUTCOMES, outcome
        self.recent[scene_level].append(outcome)
        self.episodes[scene_level] += 1
        if scene_level != self.level:
            return False
        self.at_level.append(outcome == "success")
        if not self.final and len(self.at_level) == self.window and np.mean(self.at_level) >= self.advance_at:
            self.level += 1
            self.at_level.clear()
            return True
        return False

    def rates(self, level: int) -> dict[str, float]:
        r = self.recent[level]
        return {o: sum(x == o for x in r) / len(r) for o in OUTCOMES} if r else {}

    def state_dict(self) -> dict:
        return {"level": self.level, "at_level": list(self.at_level), "recent": [list(r) for r in self.recent],
                "episodes": list(self.episodes)}

    def load_state_dict(self, d: dict) -> None:
        self.level = d["level"]
        self.at_level = deque(d["at_level"], maxlen=self.window)
        self.recent = [deque(r, maxlen=self.window) for r in d["recent"]]
        self.episodes = list(d["episodes"])
