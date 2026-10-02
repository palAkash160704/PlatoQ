"""
Shared helper functions.

Small, stateless utility functions used across multiple modules.
"""

from __future__ import annotations

import math


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the Euclidean distance between two 2-D points."""
    return math.hypot(x2 - x1, y2 - y1)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the interval [lo, hi]."""
    return max(lo, min(hi, value))
