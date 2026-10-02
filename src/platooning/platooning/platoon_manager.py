"""
Platoon manager — central coordinator for platoon lifecycle operations.

The ``PlatoonManager`` receives vehicle states, maintains the current set
of platoons, delegates to the optimisation layer when reconfiguration is
needed, and issues control directives back to the cooperative controller.

.. todo:: Phase 3 — implement formation, merging, splitting, and
   reconfiguration logic.
"""

from __future__ import annotations

from typing import Any

from platooning.models.platoon import Platoon
from platooning.models.vehicle import VehicleState
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


class PlatoonManager:
    """High-level platoon lifecycle manager.

    Parameters
    ----------
    config : dict
        The ``platooning`` section of the project configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._min_size: int = int(config.get("minimum_platoon_size", 2))
        self._max_size: int = int(config.get("maximum_platoon_size", 10))
        self._platoons: dict[str, Platoon] = {}

    # ------------------------------------------------------------------
    # Vehicle state ingestion
    # ------------------------------------------------------------------
    def update_vehicle_states(self, states: list[VehicleState]) -> None:
        """Receive the latest vehicle states from the simulation layer.

        .. todo:: Phase 3 — evaluate whether platoon reconfiguration is
           needed based on updated positions, speeds, and destinations.
        """
        # TODO — Phase 3
        logger.debug("Received %d vehicle states.", len(states))

    # ------------------------------------------------------------------
    # Platoon queries
    # ------------------------------------------------------------------
    def get_platoon(self, platoon_id: str) -> Platoon | None:
        """Return the platoon with the given ID, or *None*."""
        return self._platoons.get(platoon_id)

    def get_all_platoons(self) -> list[Platoon]:
        """Return a list of all active platoons."""
        return list(self._platoons.values())

    # ------------------------------------------------------------------
    # Platoon lifecycle (stubs)
    # ------------------------------------------------------------------
    def form_platoon(self, vehicle_ids: list[str], leader_id: str) -> Platoon | None:
        """Create a new platoon from the given vehicles.

        .. todo:: Phase 3 — validate eligibility, assign platoon ID,
           and register the new platoon.
        """
        # TODO — Phase 3
        raise NotImplementedError("PlatoonManager.form_platoon() not yet implemented.")

    def dissolve_platoon(self, platoon_id: str) -> None:
        """Remove a platoon and release its member vehicles.

        .. todo:: Phase 3 — notify vehicles and update internal state.
        """
        # TODO — Phase 3
        raise NotImplementedError(
            "PlatoonManager.dissolve_platoon() not yet implemented."
        )

    def request_optimization(self, states: list[VehicleState]) -> Any:
        """Delegate platoon assignment to the optimisation layer.

        .. todo:: Phase 3 — invoke the configured optimizer and return
           the recommended platoon configuration.
        """
        # TODO — Phase 3
        raise NotImplementedError(
            "PlatoonManager.request_optimization() not yet implemented."
        )

    def apply_configuration(self, solution: Any) -> None:
        """Apply an optimiser solution to update platoon assignments.

        .. todo:: Phase 3 — translate optimizer output into platoon
           create / merge / split / dissolve operations.
        """
        # TODO — Phase 3
        raise NotImplementedError(
            "PlatoonManager.apply_configuration() not yet implemented."
        )
