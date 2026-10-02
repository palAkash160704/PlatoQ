"""
Platooning rules and constraints.

Defines configurable rules governing platoon eligibility, maximum size,
minimum gap, speed compatibility, and similar constraints.

.. todo:: Phase 3 — implement rule evaluation engine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlatooningRules:
    """Configurable platooning constraints.

    Attributes
    ----------
    minimum_platoon_size : int
        Minimum number of vehicles to form a platoon.
    maximum_platoon_size : int
        Upper limit on platoon membership.
    desired_time_headway : float
        Target time-headway between consecutive platoon members (s).
    minimum_gap : float
        Hard minimum inter-vehicle gap (m).
    max_speed_difference : float
        Maximum allowable speed difference between a candidate vehicle
        and the platoon leader (m/s) for join eligibility.
    """

    minimum_platoon_size: int = 2
    maximum_platoon_size: int = 10
    desired_time_headway: float = 1.5
    minimum_gap: float = 5.0
    max_speed_difference: float = 5.0
