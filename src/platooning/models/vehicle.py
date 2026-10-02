"""
Vehicle state data model.

Represents the instantaneous state of a single vehicle as extracted from the
traffic simulator (SUMO via TraCI) or synthesised for testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VehicleState:
    """Snapshot of a vehicle's state at a given simulation time-step.

    Attributes
    ----------
    vehicle_id : str
        Unique identifier for the vehicle (matches SUMO vehicle ID).
    timestamp : float
        Simulation time (seconds) at which this state was captured.
    position_x : float
        Longitudinal position on the road network (metres).
    position_y : float
        Lateral position on the road network (metres).
    speed : float
        Current speed (m/s).
    acceleration : float
        Current acceleration (m/s²).
    lane_id : str
        Identifier of the lane the vehicle currently occupies.
    route_id : str
        Identifier of the vehicle's assigned route.
    destination : str
        Target edge / junction identifier.
    vehicle_type : str
        Vehicle type label (e.g. ``"passenger"``, ``"truck"``).
    platoon_id : Optional[str]
        Identifier of the platoon this vehicle belongs to, or *None* if
        the vehicle is not platooned.
    """

    vehicle_id: str
    timestamp: float = 0.0
    position_x: float = 0.0
    position_y: float = 0.0
    speed: float = 0.0
    acceleration: float = 0.0
    lane_id: str = ""
    route_id: str = ""
    destination: str = ""
    vehicle_type: str = "passenger"
    platoon_id: str | None = field(default=None)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def is_platooned(self) -> bool:
        """Return *True* if the vehicle is currently assigned to a platoon."""
        return self.platoon_id is not None
