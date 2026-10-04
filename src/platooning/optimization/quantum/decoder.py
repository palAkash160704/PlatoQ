"""
Decoder and validator for QAOA solutions — Phase 5.

Converts a raw QAOA binary bitstring back into physical platoon
configurations using the Phase 4 variable mapping, and validates
the decoded solution against all hard platooning constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from platooning.models.platoon import Platoon
from platooning.models.vehicle import VehicleState
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PlatoonConfiguration:
    """Decoded platoon configuration from a QUBO/QAOA solution.

    Attributes
    ----------
    platoons : list[Platoon]
        Formed platoons.
    ungrouped : list[str]
        Vehicle IDs not assigned to any platoon.
    """

    platoons: list[Platoon] = field(default_factory=list)
    ungrouped: list[str] = field(default_factory=list)


@dataclass
class ConstraintValidationResult:
    """Result of validating a decoded solution against platooning constraints.

    Attributes
    ----------
    feasible : bool
        True if all hard constraints are satisfied.
    violations : list[str]
        Human-readable descriptions of each constraint violation.
    violation_count : int
        Total number of constraint violations.
    details : dict[str, Any]
        Structured violation information by constraint type.
    """

    feasible: bool = True
    violations: list[str] = field(default_factory=list)
    violation_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)


def build_variable_mapping(
    n_veh: int,
    config: dict[str, Any],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int], int]:
    """Reconstruct the Phase 4 QUBO variable mapping.

    Returns
    -------
    x_indices : dict
        Mapping (j, i) → variable index for assignment variables.
    s_indices : dict
        Mapping (i, v) → variable index for size slack variables.
    num_vars : int
        Total number of binary variables.
    """
    plat_conf = config.get("platooning", {})
    min_size = int(plat_conf.get("minimum_platoon_size", 2))
    max_size = int(plat_conf.get("maximum_platoon_size", 5))

    x_indices: dict[tuple[int, int], int] = {}
    idx = 0
    for i in range(n_veh):
        for j in range(i, n_veh):
            x_indices[(j, i)] = idx
            idx += 1

    s_indices: dict[tuple[int, int], int] = {}
    for i in range(n_veh):
        max_possible_size = min(max_size, n_veh - i)
        if max_possible_size >= min_size:
            for v in range(min_size, max_possible_size + 1):
                s_indices[(i, v)] = idx
                idx += 1

    return x_indices, s_indices, idx


def decode_solution(
    bitstring: list[int],
    states: list[VehicleState],
    config: dict[str, Any],
) -> PlatoonConfiguration:
    """Decode a QUBO/QAOA binary solution into a platoon configuration.

    Uses the exact Phase 4 variable mapping to correctly interpret
    which vehicles are assigned to which platoons.

    Parameters
    ----------
    bitstring : list[int]
        Binary solution vector.
    states : list[VehicleState]
        Vehicle states (will be sorted using Phase 4 ordering).
    config : dict
        Project configuration.

    Returns
    -------
    PlatoonConfiguration
        Decoded platoon assignments including ungrouped vehicles.
    """
    n_veh = len(states)
    if n_veh == 0 or len(bitstring) == 0:
        return PlatoonConfiguration()

    # Use exactly the same sorting as Phase 4
    sorted_states = sorted(
        states, key=lambda v: (v.route_id, -v.position_x, v.vehicle_id)
    )

    x_indices, _, _ = build_variable_mapping(n_veh, config)
    min_size = int(config.get("platooning", {}).get("minimum_platoon_size", 2))

    platoons: list[Platoon] = []
    assigned_vehicles: set[str] = set()
    pid_counter = 1

    for i in range(n_veh):
        idx_ii = x_indices[(i, i)]
        if idx_ii >= len(bitstring):
            continue

        if bitstring[idx_ii] == 1:
            leader = sorted_states[i]
            members = [leader]

            for j in range(i + 1, n_veh):
                idx_ji = x_indices[(j, i)]
                if idx_ji < len(bitstring) and bitstring[idx_ji] == 1:
                    members.append(sorted_states[j])

            if len(members) >= min_size:
                member_ids = [v.vehicle_id for v in members]
                p = Platoon(
                    platoon_id=f"QAOA_P{pid_counter}",
                    vehicle_ids=member_ids,
                    leader_id=leader.vehicle_id,
                )
                p.route_id = leader.route_id
                p.lane_id = leader.lane_id
                p.status = "FORMED"
                platoons.append(p)
                assigned_vehicles.update(member_ids)
                pid_counter += 1

    ungrouped = [
        s.vehicle_id for s in sorted_states if s.vehicle_id not in assigned_vehicles
    ]

    logger.info("[DECODE] Platoons:")
    for p in platoons:
        logger.info(
            "  %s: leader=%s members=[%s]",
            p.platoon_id,
            p.leader_id,
            ", ".join(p.vehicle_ids),
        )
    if ungrouped:
        logger.info("  Ungrouped: [%s]", ", ".join(ungrouped))

    return PlatoonConfiguration(platoons=platoons, ungrouped=ungrouped)


def validate_solution(
    bitstring: list[int],
    states: list[VehicleState],
    config: dict[str, Any],
    current_time: float,
) -> ConstraintValidationResult:
    """Validate a decoded QAOA solution against all hard platooning constraints.

    Checks:
    1. Vehicle assignment uniqueness
    2. Leader activation
    3. Compatibility (route, lane, distance, speed, communication)
    4. Minimum / maximum platoon size

    Parameters
    ----------
    bitstring : list[int]
        Binary solution vector.
    states : list[VehicleState]
        Vehicle states.
    config : dict
        Project configuration.
    current_time : float
        Simulation time for communication freshness check.

    Returns
    -------
    ConstraintValidationResult
        Feasibility status and list of any violations.
    """
    result = ConstraintValidationResult()
    result.details = {
        "assignment": [],
        "leader": [],
        "compatibility": [],
        "size": [],
        "route": [],
        "lane": [],
        "distance": [],
        "speed": [],
        "communication": [],
    }

    n_veh = len(states)
    if n_veh == 0 or len(bitstring) == 0:
        return result

    sorted_states = sorted(
        states, key=lambda v: (v.route_id, -v.position_x, v.vehicle_id)
    )

    x_indices, _, _ = build_variable_mapping(n_veh, config)
    plat_conf = config.get("platooning", {})
    comm_conf = config.get("communication", {})

    min_size = int(plat_conf.get("minimum_platoon_size", 2))
    max_size = int(plat_conf.get("maximum_platoon_size", 5))
    max_dist = float(plat_conf.get("max_formation_distance_m", 50.0))
    max_speed_diff = float(plat_conf.get("max_speed_difference_mps", 3.0))
    stale_thresh_ms = float(comm_conf.get("stale_threshold_ms", 200.0))

    # Compatibility is checked directly in the pairwise loop below

    # 1. Check assignment uniqueness
    assigned_count: dict[int, int] = {j: 0 for j in range(n_veh)}
    platoon_members: dict[int, list[int]] = {i: [] for i in range(n_veh)}

    for i in range(n_veh):
        for j in range(i, n_veh):
            key = (j, i)
            if key in x_indices:
                idx = x_indices[key]
                if idx < len(bitstring) and bitstring[idx] == 1:
                    assigned_count[j] += 1
                    platoon_members[i].append(j)

    for j, count in assigned_count.items():
        if count > 1:
            vid = sorted_states[j].vehicle_id
            msg = f"Vehicle {vid} assigned to {count} platoons"
            result.feasible = False
            result.violations.append(msg)
            result.details["assignment"].append(msg)

    # 2. Check leader activation and platoon constraints
    for i, members in platoon_members.items():
        size = len(members)
        if size == 0:
            continue

        leader_vid = sorted_states[i].vehicle_id

        # 2a. Leader must be in its own platoon
        if i not in members:
            msg = f"Platoon led by {leader_vid}: leader not self-assigned"
            result.feasible = False
            result.violations.append(msg)
            result.details["leader"].append(msg)

        # 2b. Size constraints
        if size < min_size:
            msg = f"Platoon led by {leader_vid}: size {size} < min {min_size}"
            result.feasible = False
            result.violations.append(msg)
            result.details["size"].append(msg)

        if size > max_size:
            msg = f"Platoon led by {leader_vid}: size {size} > max {max_size}"
            result.feasible = False
            result.violations.append(msg)
            result.details["size"].append(msg)

        # 2c. Pairwise compatibility checks
        for u_idx_in_members in range(len(members)):
            for v_idx_in_members in range(u_idx_in_members + 1, len(members)):
                u = members[u_idx_in_members]
                v = members[v_idx_in_members]
                v1 = sorted_states[u]
                v2 = sorted_states[v]

                # Route compatibility
                if v1.route_id != v2.route_id:
                    msg = f"Incompatible routes: {v1.vehicle_id} ({v1.route_id}) and {v2.vehicle_id} ({v2.route_id})"
                    result.feasible = False
                    result.violations.append(msg)
                    result.details["route"].append(msg)

                # Lane compatibility
                if v1.lane_id != v2.lane_id:
                    msg = f"Incompatible lanes: {v1.vehicle_id} ({v1.lane_id}) and {v2.vehicle_id} ({v2.lane_id})"
                    result.feasible = False
                    result.violations.append(msg)
                    result.details["lane"].append(msg)

                # Distance check
                dist = math.sqrt(
                    (v1.position_x - v2.position_x) ** 2
                    + (v1.position_y - v2.position_y) ** 2
                )
                if dist > max_dist:
                    msg = f"Distance exceeded: {v1.vehicle_id}-{v2.vehicle_id} = {dist:.1f}m > {max_dist:.1f}m"
                    result.feasible = False
                    result.violations.append(msg)
                    result.details["distance"].append(msg)

                # Speed difference check
                speed_diff = abs(v1.speed - v2.speed)
                if speed_diff > max_speed_diff:
                    msg = f"Speed diff exceeded: {v1.vehicle_id}-{v2.vehicle_id} = {speed_diff:.1f}m/s > {max_speed_diff:.1f}m/s"
                    result.feasible = False
                    result.violations.append(msg)
                    result.details["speed"].append(msg)

                # Communication freshness
                age_1 = current_time - v1.timestamp
                age_2 = current_time - v2.timestamp
                max_age = max(age_1, age_2)
                if max_age > stale_thresh_ms / 1000.0:
                    msg = f"Stale comms: {v1.vehicle_id}-{v2.vehicle_id}, age={max_age:.3f}s"
                    result.feasible = False
                    result.violations.append(msg)
                    result.details["communication"].append(msg)

    result.violation_count = len(result.violations)
    return result


def analyze_samples(
    counts: dict[str, int],
    q_matrix: dict[tuple[int, int], float],
    states: list[VehicleState],
    config: dict[str, Any],
    current_time: float,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Analyse the top-k most frequent QAOA measurement outcomes.

    For each candidate bitstring:
    - Calculates QUBO energy
    - Decodes the platoon configuration
    - Validates constraints

    Parameters
    ----------
    counts : dict
        Measurement counts {bitstring: count}.
    q_matrix : dict
        QUBO matrix.
    states : list[VehicleState]
        Vehicle states.
    config : dict
        Project configuration.
    current_time : float
        Simulation time.
    top_k : int
        Number of top candidates to analyse.

    Returns
    -------
    list[dict]
        Analysis for each top candidate.
    """
    # Sort by frequency (descending)
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:top_k]

    total_shots = sum(counts.values())
    analyses: list[dict[str, Any]] = []

    for bitstr, count in sorted_counts:
        bits = [int(b) for b in reversed(bitstr)]

        # Energy
        energy = 0.0
        for (i, j), coef in q_matrix.items():
            if bits[i] == 1 and bits[j] == 1:
                energy += coef

        # Decode
        decoded = decode_solution(bits, states, config)

        # Validate
        validation = validate_solution(bits, states, config, current_time)

        analyses.append({
            "bitstring": bitstr,
            "bits": bits,
            "count": count,
            "frequency": count / total_shots,
            "energy": energy,
            "feasible": validation.feasible,
            "violation_count": validation.violation_count,
            "violations": validation.violations,
            "num_platoons": len(decoded.platoons),
            "platoon_sizes": [p.size for p in decoded.platoons],
            "ungrouped_count": len(decoded.ungrouped),
        })

    return analyses


def select_best_feasible(
    counts: dict[str, int],
    q_matrix: dict[tuple[int, int], float],
    states: list[VehicleState],
    config: dict[str, Any],
    current_time: float,
) -> tuple[list[int] | None, float, bool]:
    """Select the best feasible solution from measurement counts.

    Iterates over all unique measured bitstrings. For each, repairs the
    slack variables (s_indices) to perfectly match the physical platoon
    sizes encoded in the assignment variables (x_indices), which eliminates
    artificial size-penalty offsets (e.g., the +1000 discrepancy).
    Then evaluates QUBO energy, validates constraints, and returns the
    lowest-energy feasible solution.

    If no feasible solution exists, returns the lowest-energy solution
    overall with feasible=False.

    Returns
    -------
    tuple
        (best_bitstring, best_energy, is_feasible)
    """
    n_veh = len(states)
    x_indices, s_indices, _ = build_variable_mapping(n_veh, config)

    best_feasible_bits: list[int] | None = None
    best_feasible_energy = float("inf")

    best_any_bits: list[int] | None = None
    best_any_energy = float("inf")

    for bitstr in counts:
        # 1. Parse raw bitstring
        raw_bits = [int(b) for b in reversed(bitstr)]
        
        # 2. Repair slack variables to match physical assignment
        repaired_bits = list(raw_bits)
        
        # Count physical size of each platoon i
        platoon_sizes: dict[int, int] = {i: 0 for i in range(n_veh)}
        for i in range(n_veh):
            for j in range(i, n_veh):
                key = (j, i)
                if key in x_indices:
                    idx = x_indices[key]
                    if idx < len(raw_bits) and raw_bits[idx] == 1:
                        platoon_sizes[i] += 1
                        
        # Enforce exactly the correct slack variable
        for i, size in platoon_sizes.items():
            # Zero out all slack variables for platoon i
            for (i_idx, v), s_idx in s_indices.items():
                if i_idx == i and s_idx < len(repaired_bits):
                    repaired_bits[s_idx] = 0
            
            # Set the correct slack variable if it exists
            if size > 0:
                key = (i, size)
                if key in s_indices:
                    s_idx = s_indices[key]
                    if s_idx < len(repaired_bits):
                        repaired_bits[s_idx] = 1

        # 3. Evaluate energy with repaired bitstring
        energy = 0.0
        for (i, j), coef in q_matrix.items():
            if repaired_bits[i] == 1 and repaired_bits[j] == 1:
                energy += coef

        if energy < best_any_energy:
            best_any_energy = energy
            best_any_bits = repaired_bits

        # 4. Validate physical constraints
        validation = validate_solution(repaired_bits, states, config, current_time)
        if validation.feasible and energy < best_feasible_energy:
            best_feasible_energy = energy
            best_feasible_bits = repaired_bits

    if best_feasible_bits is not None:
        return best_feasible_bits, best_feasible_energy, True
    else:
        return best_any_bits, best_any_energy, False
