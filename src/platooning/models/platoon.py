"""
Platoon data model.

Represents a cooperative vehicle platoon — a group of connected vehicles
travelling together under coordinated management.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Platoon:
    """A cooperative vehicle platoon.

    Attributes
    ----------
    platoon_id : str
        Unique identifier for this platoon.
    vehicle_ids : list[str]
        Ordered list of vehicle IDs in the platoon.  The first entry is
        conventionally the leader.
    leader_id : str
        Vehicle ID of the current platoon leader.
    destination : str
        Common destination edge / junction for the platoon.
    target_speed : float
        Desired cruising speed for the platoon (m/s).
    """

    platoon_id: str
    vehicle_ids: list[str] = field(default_factory=list)
    leader_id: str = ""
    route_id: str = ""
    lane_id: str = ""
    formation_time: float = 0.0
    target_speed_mps: float = 0.0
    status: str = "FORMED"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        """Return the number of vehicles currently in the platoon."""
        return len(self.vehicle_ids)

    def contains(self, vehicle_id: str) -> bool:
        """Return *True* if *vehicle_id* is a member of this platoon."""
        return vehicle_id in self.vehicle_ids
