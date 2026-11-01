"""One Meshcat server per process.  `HEADLESS=1` disables visualisation."""

from __future__ import annotations

import os
import time

from pydrake.all import Meshcat, StartMeshcat

HEADLESS = bool(os.environ.get("HEADLESS"))
BROWSER_WAIT = float(os.environ.get("MESHCAT_BROWSER_WAIT", "10"))

_meshcat: Meshcat | None = None


def get_meshcat() -> Meshcat | None:
    """The process's Meshcat, started on first use; None when HEADLESS."""
    global _meshcat
    if HEADLESS:
        return None
    if _meshcat is None:
        _meshcat = StartMeshcat()
        print(f"Meshcat: {_meshcat.web_url()}")
        deadline = time.time() + BROWSER_WAIT
        while time.time() < deadline and _meshcat.GetNumActiveConnections() == 0:
            time.sleep(0.1)
    return _meshcat


def keep_open(message: str = "Press Enter to exit") -> None:
    if not HEADLESS:
        input(message)
