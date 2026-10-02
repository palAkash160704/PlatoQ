"""
Cooperative vehicle-following controller.

Implements the low-level longitudinal (and optionally lateral) control
that keeps platoon members at the desired headway from their predecessor.

This layer is intentionally separated from the optimisation layer:
the optimiser decides *which* vehicles platoon together, while the
controller decides *how* each vehicle follows the vehicle ahead.

.. todo:: Phase 7 — implement CACC / cooperative ACC controller.
"""

from __future__ import annotations

from platooning.models.vehicle import VehicleState
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


class CooperativeController:
    """Cooperative Adaptive Cruise Control (CACC) placeholder.

    .. todo:: Phase 7 — implement headway-tracking control law.
    """

    def compute_acceleration(
        self,
        ego: VehicleState,
        predecessor: VehicleState | None,
        desired_gap: float = 5.0,
    ) -> float:
        """Compute the desired acceleration for the ego vehicle.

        Parameters
        ----------
        ego : VehicleState
            Current state of the controlled vehicle.
        predecessor : VehicleState or None
            State of the vehicle directly ahead (None if leader).
        desired_gap : float
            Target inter-vehicle gap (metres).

        Returns
        -------
        float
            Desired acceleration (m/s²).

        .. todo:: Phase 7 — replace stub with real control law.
        """
        # TODO — Phase 7
        raise NotImplementedError(
            "CooperativeController.compute_acceleration() not yet implemented."
        )
