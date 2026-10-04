"""
Platoon formation logic.

Contains functions and classes that determine *how* vehicles should be grouped
into platoons based on explicit compatibility criteria, as well as the
deterministic classical baseline for platoon formation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from platooning.models.platoon import Platoon
from platooning.models.vehicle import VehicleState
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


# ======================================================================
# Compatibility Model
# ======================================================================
@dataclass
class VehiclePairCompatibility:
    """Represents the platooning compatibility between two vehicles.

    Attributes
    ----------
    vehicle_a_id : str
        ID of the first vehicle.
    vehicle_b_id : str
        ID of the second vehicle.
    compatible : bool
        True if the vehicles satisfy all platooning constraints.
    distance_m : float
        Euclidean distance between the vehicles.
    speed_difference_mps : float
        Absolute speed difference in m/s.
    same_route : bool
        True if the vehicles share the same route ID.
    same_lane : bool
        True if the vehicles share the same lane ID.
    communication_fresh : bool
        True if the state age is within the stale threshold.
    reason : str
        Explains why the vehicles are compatible or incompatible.
    """

    vehicle_a_id: str
    vehicle_b_id: str
    compatible: bool
    distance_m: float
    speed_difference_mps: float
    same_route: bool
    same_lane: bool
    communication_fresh: bool
    reason: str


def evaluate_compatibility(
    a: VehicleState,
    b: VehicleState,
    config: dict[str, Any],
    current_time: float,
) -> VehiclePairCompatibility:
    """Evaluate whether two vehicles are eligible to form a platoon.

    Constraints evaluated:
    - Communication freshness
    - Spatial distance
    - Speed difference
    - Route compatibility
    - Lane compatibility
    """
    # Parameters
    plat_conf = config.get("platooning", {})
    comm_conf = config.get("communication", {})

    max_dist = float(plat_conf.get("max_formation_distance_m", 50.0))
    max_speed_diff = float(plat_conf.get("max_speed_difference_mps", 3.0))
    stale_thresh_ms = float(comm_conf.get("stale_threshold_ms", 200.0))

    dx = a.position_x - b.position_x
    dy = a.position_y - b.position_y
    dist = math.sqrt(dx * dx + dy * dy)

    speed_diff = abs(a.speed - b.speed)

    same_route = a.route_id == b.route_id
    same_lane = a.lane_id == b.lane_id

    # Check communication freshness
    # Use the older of the two states (worst-case age)
    age_a = current_time - a.timestamp
    age_b = current_time - b.timestamp
    max_age = max(age_a, age_b)
    comm_fresh = max_age <= (stale_thresh_ms / 1000.0)

    # Evaluate compatibility
    compatible = True
    reason = "compatible"

    if not comm_fresh:
        compatible = False
        reason = "stale_state"
    elif dist > max_dist:
        compatible = False
        reason = "distance_exceeded"
    elif speed_diff > max_speed_diff:
        compatible = False
        reason = "speed_difference_exceeded"
    elif not same_route:
        compatible = False
        reason = "different_route"
    elif not same_lane:
        compatible = False
        reason = "different_lane"

    return VehiclePairCompatibility(
        vehicle_a_id=a.vehicle_id,
        vehicle_b_id=b.vehicle_id,
        compatible=compatible,
        distance_m=dist,
        speed_difference_mps=speed_diff,
        same_route=same_route,
        same_lane=same_lane,
        communication_fresh=comm_fresh,
        reason=reason,
    )


def build_compatibility_graph(
    states: list[VehicleState],
    config: dict[str, Any],
    current_time: float,
) -> dict[str, list[VehiclePairCompatibility]]:
    """Build a graph representing pairwise vehicle compatibilities.

    Returns
    -------
    dict[str, list[VehiclePairCompatibility]]
        Adjacency list mapping a vehicle ID to its compatibility relations.
    """
    graph: dict[str, list[VehiclePairCompatibility]] = {
        s.vehicle_id: [] for s in states
    }

    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            a = states[i]
            b = states[j]
            comp = evaluate_compatibility(a, b, config, current_time)

            graph[a.vehicle_id].append(comp)
            # Create the symmetric record for b
            comp_sym = VehiclePairCompatibility(
                vehicle_a_id=b.vehicle_id,
                vehicle_b_id=a.vehicle_id,
                compatible=comp.compatible,
                distance_m=comp.distance_m,
                speed_difference_mps=comp.speed_difference_mps,
                same_route=comp.same_route,
                same_lane=comp.same_lane,
                communication_fresh=comp.communication_fresh,
                reason=comp.reason,
            )
            graph[b.vehicle_id].append(comp_sym)

    return graph


# ======================================================================
# Classical Baseline Formation
# ======================================================================
def select_leader(vehicles: list[VehicleState]) -> str:
    """Select a platoon leader deterministically.

    Rule: The vehicle furthest ahead along the route (highest x-coord)
    is the leader. Breaks ties using vehicle_id.
    """
    if not vehicles:
        raise ValueError("Cannot select leader of an empty list.")

    # Sort descending by x-coord, then ascending by vehicle_id
    sorted_vs = sorted(vehicles, key=lambda v: (-v.position_x, v.vehicle_id))
    return sorted_vs[0].vehicle_id


def form_platoons(
    states: list[VehicleState],
    config: dict[str, Any],
    current_time: float,
) -> list[Platoon]:
    """Deterministically form platoons using a classical greedy approach.

    Algorithm:
    1. Sort vehicles deterministically (route_id, -position_x, vehicle_id).
    2. Pick first unassigned vehicle as a candidate platoon.
    3. Add compatible unassigned vehicles until max_platoon_size is reached.
    4. Ensure all members of a platoon are mutually compatible with each other.
    5. Repeat until all vehicles are processed.
    6. Return formed platoons (size >= minimum_platoon_size).
    """
    plat_conf = config.get("platooning", {})
    min_size = int(plat_conf.get("minimum_platoon_size", 2))
    max_size = int(plat_conf.get("maximum_platoon_size", 5))

    # 1. Build compatibility graph
    graph = build_compatibility_graph(states, config, current_time)

    # 2. Sort vehicles deterministically
    sorted_states = sorted(
        states, key=lambda v: (v.route_id, -v.position_x, v.vehicle_id)
    )

    unassigned = {s.vehicle_id for s in states}
    platoons: list[Platoon] = []
    platoon_counter = 1

    for base_v in sorted_states:
        if base_v.vehicle_id not in unassigned:
            continue

        # Start a new candidate group
        candidate_ids = [base_v.vehicle_id]
        unassigned.remove(base_v.vehicle_id)

        # Find compatible vehicles to add
        for other_v in sorted_states:
            if other_v.vehicle_id not in unassigned:
                continue

            if len(candidate_ids) >= max_size:
                break

            # Check if other_v is compatible with ALL current members
            is_mutually_compatible = True
            for member_id in candidate_ids:
                # Find the relation in graph
                comp_record = next(
                    (
                        c
                        for c in graph[other_v.vehicle_id]
                        if c.vehicle_b_id == member_id
                    ),
                    None,
                )
                if comp_record is None or not comp_record.compatible:
                    is_mutually_compatible = False
                    break

            if is_mutually_compatible:
                candidate_ids.append(other_v.vehicle_id)
                unassigned.remove(other_v.vehicle_id)

        # Check if we meet minimum size requirements
        if len(candidate_ids) >= min_size:
            # Get the actual state objects to select a leader
            member_states = [s for s in states if s.vehicle_id in candidate_ids]
            leader_id = select_leader(member_states)

            p = Platoon(
                platoon_id=f"P{platoon_counter}",
                vehicle_ids=candidate_ids,
                leader_id=leader_id,
                route_id=base_v.route_id,
                lane_id=base_v.lane_id,
                formation_time=current_time,
                target_speed_mps=base_v.speed,  # simplified
                status="FORMED",
            )
            platoons.append(p)
            platoon_counter += 1
        else:
            # For size < min_size, they remain unassigned (or we can just ignore them,
            # but since we removed them from unassigned, they are effectively solo).
            # The classical algorithm doesn't track 1-vehicle platoons.
            pass

    return platoons
