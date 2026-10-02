"""
Platoon formation logic.

Contains functions/classes that determine *how* vehicles should be grouped
into platoons — e.g. destination-based clustering, proximity scoring.

.. todo:: Phase 3 — implement formation algorithms.
"""

from __future__ import annotations

from platooning.models.vehicle import VehicleState
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_candidates(states: list[VehicleState]) -> list[list[str]]:
    """Identify groups of vehicles that are eligible to form a platoon.

    Parameters
    ----------
    states : list[VehicleState]
        Current states of all vehicles in the simulation.

    Returns
    -------
    list[list[str]]
        A list of candidate vehicle-ID groups.

    .. todo:: Phase 3 — implement destination / proximity heuristics.
    """
    # TODO — Phase 3
    raise NotImplementedError("Formation candidate evaluation not yet implemented.")
