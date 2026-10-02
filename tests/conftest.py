"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Repository root (used by several test modules)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]


def _sumo_available() -> bool:
    """Return True if SUMO is reachable on this machine."""
    # 1. SUMO_HOME
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home and os.path.isdir(sumo_home):
        return True
    # 2. System PATH
    if shutil.which("sumo") or shutil.which("sumo-gui"):
        return True
    # 3. sumolib (pip-installed eclipse-sumo)
    try:
        import sumolib  # type: ignore[import-untyped]

        sumolib.checkBinary("sumo")
        return True
    except Exception:
        pass
    return False


# Marker: skip integration tests when SUMO is not installed
requires_sumo = pytest.mark.skipif(
    not _sumo_available(),
    reason="SUMO is not installed or not on PATH / SUMO_HOME.",
)
