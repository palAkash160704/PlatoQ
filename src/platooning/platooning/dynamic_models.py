"""
Phase 6 — Dynamic Platoon Management Models.

Defines reconfiguration triggers, platoon health status, reconfiguration
events, and dynamic metrics used by the platoon monitor and reconfiguration
engine.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


# ======================================================================
# Reconfiguration Triggers
# ======================================================================
class ReconfigTrigger(enum.Enum):
    """Reasons that can trigger a platoon reconfiguration."""

    MEMBER_TOO_FAR = "MEMBER_TOO_FAR"
    SPEED_MISMATCH = "SPEED_MISMATCH"
    STALE_STATE = "STALE_STATE"
    COMMUNICATION_LOSS = "COMMUNICATION_LOSS"
    ROUTE_MISMATCH = "ROUTE_MISMATCH"
    LANE_MISMATCH = "LANE_MISMATCH"
    MEMBER_LEFT = "MEMBER_LEFT"
    NEW_COMPATIBLE_VEHICLE = "NEW_COMPATIBLE_VEHICLE"
    PERIODIC_REOPTIMIZATION = "PERIODIC_REOPTIMIZATION"
    SIZE_VIOLATION = "SIZE_VIOLATION"


# ======================================================================
# Platoon Status
# ======================================================================
class PlatoonStatus(enum.Enum):
    """Lifecycle status of a platoon."""

    ACTIVE = "ACTIVE"
    INVALID = "INVALID"
    RECONFIGURING = "RECONFIGURING"
    DISSOLVED = "DISSOLVED"


# ======================================================================
# Violation Severity
# ======================================================================
class ViolationSeverity(enum.Enum):
    """Severity level of a platoon health violation."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ======================================================================
# Platoon Health
# ======================================================================
@dataclass
class PlatoonHealth:
    """Health assessment result for a single platoon.

    Attributes
    ----------
    platoon_id : str
        ID of the inspected platoon.
    valid : bool
        True if the platoon satisfies all constraints.
    violations : list[ReconfigTrigger]
        List of triggers that indicate problems.
    violation_details : list[str]
        Human-readable descriptions of each violation.
    affected_members : list[str]
        Vehicle IDs that are involved in the violations.
    severity : ViolationSeverity
        Overall severity of the health check result.
    """

    platoon_id: str = ""
    valid: bool = True
    violations: list[ReconfigTrigger] = field(default_factory=list)
    violation_details: list[str] = field(default_factory=list)
    affected_members: list[str] = field(default_factory=list)
    severity: ViolationSeverity = ViolationSeverity.LOW


# ======================================================================
# Reconfiguration Event
# ======================================================================
@dataclass
class ReconfigurationEvent:
    """Structured record of a single reconfiguration action.

    Attributes
    ----------
    timestamp : float
        Simulation time when the event occurred.
    trigger : ReconfigTrigger
        Primary trigger that caused reconfiguration.
    old_configuration : dict[str, list[str]]
        Previous platoon assignments {platoon_id: [vehicle_ids]}.
    new_configuration : dict[str, list[str]]
        New platoon assignments after reconfiguration.
    affected_vehicles : list[str]
        Vehicles whose membership changed.
    affected_platoons : list[str]
        Platoon IDs that were modified.
    optimizer : str
        Optimiser used ("classical", "exact_qubo", "qaoa").
    old_objective : float
        Objective value before reconfiguration.
    new_objective : float
        Objective value after reconfiguration.
    objective_improvement : float
        new_objective - old_objective (positive = improvement).
    old_energy : float
        QUBO energy before (if applicable).
    new_energy : float
        QUBO energy after (if applicable).
    runtime_ms : float
        Reconfiguration latency in milliseconds.
    feasible : bool
        True if the new configuration satisfies all constraints.
    violations : list[str]
        Any remaining violations in the new configuration.
    event_type : str
        Classification: "join", "leave", "split", "merge", "reform".
    """

    timestamp: float = 0.0
    trigger: ReconfigTrigger = ReconfigTrigger.PERIODIC_REOPTIMIZATION
    old_configuration: dict[str, list[str]] = field(default_factory=dict)
    new_configuration: dict[str, list[str]] = field(default_factory=dict)
    affected_vehicles: list[str] = field(default_factory=list)
    affected_platoons: list[str] = field(default_factory=list)
    optimizer: str = "classical"
    old_objective: float = 0.0
    new_objective: float = 0.0
    objective_improvement: float = 0.0
    old_energy: float = float("inf")
    new_energy: float = float("inf")
    runtime_ms: float = 0.0
    feasible: bool = True
    violations: list[str] = field(default_factory=list)
    event_type: str = "reform"


# ======================================================================
# Dynamic Metrics
# ======================================================================
@dataclass
class DynamicMetrics:
    """Aggregated metrics for a dynamic platoon management run.

    Attributes
    ----------
    total_reconfigurations : int
        Total reconfiguration events.
    join_events : int
        Vehicles joining existing platoons.
    leave_events : int
        Vehicles leaving platoons.
    merge_events : int
        Platoons merging together.
    split_events : int
        Platoons splitting.
    stale_triggered : int
        Reconfigurations triggered by stale V2V state.
    communication_triggered : int
        Reconfigurations triggered by communication loss.
    configuration_changes : int
        Number of times the platoon config actually changed.
    total_simulation_time : float
        Duration of the simulation run.
    churn_rate : float
        configuration_changes / total_simulation_time.
    average_reconfiguration_latency_ms : float
        Mean time from trigger to applied configuration.
    average_num_platoons : float
        Mean number of active platoons over time.
    average_platoon_size : float
        Mean platoon size over time.
    average_objective : float
        Mean objective value over time.
    """

    total_reconfigurations: int = 0
    join_events: int = 0
    leave_events: int = 0
    merge_events: int = 0
    split_events: int = 0
    stale_triggered: int = 0
    communication_triggered: int = 0
    configuration_changes: int = 0
    total_simulation_time: float = 0.0
    churn_rate: float = 0.0
    average_reconfiguration_latency_ms: float = 0.0
    average_num_platoons: float = 0.0
    average_platoon_size: float = 0.0
    average_objective: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        from dataclasses import asdict

        return asdict(self)
