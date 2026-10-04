"""
Phase 6 — Platoon Monitor.

Inspects every active platoon and determines whether it remains valid.
The monitor does NOT modify any platoon — it returns structured
PlatoonHealth results that the reconfiguration engine acts upon.
"""

from __future__ import annotations

import math
from typing import Any

from platooning.models.platoon import Platoon
from platooning.models.vehicle import VehicleState
from platooning.platooning.dynamic_models import (
    PlatoonHealth,
    ReconfigTrigger,
    ViolationSeverity,
)
from platooning.platooning.formation import evaluate_compatibility
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


class PlatoonMonitor:
    """Monitor active platoons for constraint violations.

    The monitor is a read-only inspector.  It receives the current vehicle
    states and V2V freshness information, checks every active platoon
    against the established Phase 3 compatibility constraints, and
    returns a list of ``PlatoonHealth`` assessments.

    Parameters
    ----------
    config : dict
        The full project configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._plat_conf = config.get("platooning", {})
        self._comm_conf = config.get("communication", {})

    def check_all(
        self,
        platoons: list[Platoon],
        states: list[VehicleState],
        current_time: float,
    ) -> list[PlatoonHealth]:
        """Check the health of all active platoons.

        Parameters
        ----------
        platoons : list[Platoon]
            Currently active platoons.
        states : list[VehicleState]
            Latest known vehicle states.
        current_time : float
            Current simulation time.

        Returns
        -------
        list[PlatoonHealth]
            One health report per platoon.
        """
        state_map = {s.vehicle_id: s for s in states}
        results: list[PlatoonHealth] = []

        for platoon in platoons:
            health = self.check_platoon(platoon, state_map, current_time)
            results.append(health)

        return results

    def check_platoon(
        self,
        platoon: Platoon,
        state_map: dict[str, VehicleState],
        current_time: float,
    ) -> PlatoonHealth:
        """Perform a comprehensive health check on a single platoon.

        Checks (in order):
        1. Vehicle state availability
        2. Communication freshness
        3. Route consistency
        4. Lane consistency
        5. Pairwise distance
        6. Pairwise speed difference
        7. Minimum platoon size
        8. Maximum platoon size
        9. Leader validity

        Parameters
        ----------
        platoon : Platoon
            The platoon to check.
        state_map : dict[str, VehicleState]
            Latest vehicle states keyed by vehicle ID.
        current_time : float
            Current simulation time.

        Returns
        -------
        PlatoonHealth
            Structured health assessment.
        """
        health = PlatoonHealth(platoon_id=platoon.platoon_id)
        health.severity = ViolationSeverity.LOW

        max_dist = float(self._plat_conf.get("max_formation_distance_m", 50.0))
        max_speed_diff = float(
            self._plat_conf.get("max_speed_difference_mps", 3.0)
        )
        min_size = int(self._plat_conf.get("minimum_platoon_size", 2))
        max_size = int(self._plat_conf.get("maximum_platoon_size", 5))
        stale_thresh_s = (
            float(self._comm_conf.get("stale_threshold_ms", 200.0)) / 1000.0
        )

        # ------------------------------------------------------------------
        # 1. Vehicle state availability
        # ------------------------------------------------------------------
        missing = []
        present_states: list[VehicleState] = []
        for vid in platoon.vehicle_ids:
            vs = state_map.get(vid)
            if vs is None:
                missing.append(vid)
            else:
                present_states.append(vs)

        if missing:
            health.valid = False
            health.violations.append(ReconfigTrigger.COMMUNICATION_LOSS)
            health.violation_details.append(
                f"Missing state data for: {', '.join(missing)}"
            )
            health.affected_members.extend(missing)
            health.severity = ViolationSeverity.CRITICAL

        # ------------------------------------------------------------------
        # 2. Communication freshness
        # ------------------------------------------------------------------
        for vs in present_states:
            age = current_time - vs.timestamp
            if age > stale_thresh_s:
                health.valid = False
                if ReconfigTrigger.STALE_STATE not in health.violations:
                    health.violations.append(ReconfigTrigger.STALE_STATE)
                health.violation_details.append(
                    f"{vs.vehicle_id}: stale (age={age:.3f}s > {stale_thresh_s:.3f}s)"
                )
                if vs.vehicle_id not in health.affected_members:
                    health.affected_members.append(vs.vehicle_id)
                health.severity = ViolationSeverity.CRITICAL

        # ------------------------------------------------------------------
        # 3-6. Pairwise checks among present members
        # ------------------------------------------------------------------
        for i in range(len(present_states)):
            for j in range(i + 1, len(present_states)):
                va = present_states[i]
                vb = present_states[j]

                # Route
                if va.route_id != vb.route_id:
                    health.valid = False
                    if ReconfigTrigger.ROUTE_MISMATCH not in health.violations:
                        health.violations.append(ReconfigTrigger.ROUTE_MISMATCH)
                    health.violation_details.append(
                        f"{va.vehicle_id}-{vb.vehicle_id}: route mismatch "
                        f"({va.route_id} vs {vb.route_id})"
                    )
                    for v in (va, vb):
                        if v.vehicle_id not in health.affected_members:
                            health.affected_members.append(v.vehicle_id)
                    health.severity = max(
                        health.severity, ViolationSeverity.CRITICAL,
                        key=lambda s: list(ViolationSeverity).index(s),
                    )

                # Lane
                if va.lane_id != vb.lane_id:
                    health.valid = False
                    if ReconfigTrigger.LANE_MISMATCH not in health.violations:
                        health.violations.append(ReconfigTrigger.LANE_MISMATCH)
                    health.violation_details.append(
                        f"{va.vehicle_id}-{vb.vehicle_id}: lane mismatch "
                        f"({va.lane_id} vs {vb.lane_id})"
                    )
                    for v in (va, vb):
                        if v.vehicle_id not in health.affected_members:
                            health.affected_members.append(v.vehicle_id)
                    health.severity = max(
                        health.severity, ViolationSeverity.HIGH,
                        key=lambda s: list(ViolationSeverity).index(s),
                    )

                # Distance
                dist = math.sqrt(
                    (va.position_x - vb.position_x) ** 2
                    + (va.position_y - vb.position_y) ** 2
                )
                if dist > max_dist:
                    health.valid = False
                    if ReconfigTrigger.MEMBER_TOO_FAR not in health.violations:
                        health.violations.append(ReconfigTrigger.MEMBER_TOO_FAR)
                    health.violation_details.append(
                        f"{va.vehicle_id}-{vb.vehicle_id}: distance "
                        f"{dist:.1f}m > {max_dist:.1f}m"
                    )
                    for v in (va, vb):
                        if v.vehicle_id not in health.affected_members:
                            health.affected_members.append(v.vehicle_id)
                    health.severity = max(
                        health.severity, ViolationSeverity.MEDIUM,
                        key=lambda s: list(ViolationSeverity).index(s),
                    )

                # Speed
                speed_diff = abs(va.speed - vb.speed)
                if speed_diff > max_speed_diff:
                    health.valid = False
                    if ReconfigTrigger.SPEED_MISMATCH not in health.violations:
                        health.violations.append(ReconfigTrigger.SPEED_MISMATCH)
                    health.violation_details.append(
                        f"{va.vehicle_id}-{vb.vehicle_id}: speed diff "
                        f"{speed_diff:.1f}m/s > {max_speed_diff:.1f}m/s"
                    )
                    for v in (va, vb):
                        if v.vehicle_id not in health.affected_members:
                            health.affected_members.append(v.vehicle_id)
                    health.severity = max(
                        health.severity, ViolationSeverity.MEDIUM,
                        key=lambda s: list(ViolationSeverity).index(s),
                    )

        # ------------------------------------------------------------------
        # 7. Minimum platoon size
        # ------------------------------------------------------------------
        effective_size = len(present_states)
        if effective_size < min_size:
            health.valid = False
            if ReconfigTrigger.SIZE_VIOLATION not in health.violations:
                health.violations.append(ReconfigTrigger.SIZE_VIOLATION)
            health.violation_details.append(
                f"Size {effective_size} < min {min_size}"
            )
            health.severity = max(
                health.severity, ViolationSeverity.HIGH,
                key=lambda s: list(ViolationSeverity).index(s),
            )

        # ------------------------------------------------------------------
        # 8. Maximum platoon size
        # ------------------------------------------------------------------
        if effective_size > max_size:
            health.valid = False
            if ReconfigTrigger.SIZE_VIOLATION not in health.violations:
                health.violations.append(ReconfigTrigger.SIZE_VIOLATION)
            health.violation_details.append(
                f"Size {effective_size} > max {max_size}"
            )
            health.severity = max(
                health.severity, ViolationSeverity.MEDIUM,
                key=lambda s: list(ViolationSeverity).index(s),
            )

        # ------------------------------------------------------------------
        # 9. Leader validity
        # ------------------------------------------------------------------
        if platoon.leader_id not in platoon.vehicle_ids:
            health.valid = False
            health.violation_details.append(
                f"Leader {platoon.leader_id} not in member list"
            )
            health.severity = ViolationSeverity.CRITICAL

        if platoon.leader_id not in state_map:
            health.valid = False
            health.violation_details.append(
                f"Leader {platoon.leader_id} state data missing"
            )
            if platoon.leader_id not in health.affected_members:
                health.affected_members.append(platoon.leader_id)
            health.severity = ViolationSeverity.CRITICAL

        return health

    def detect_new_compatible_vehicles(
        self,
        platoons: list[Platoon],
        states: list[VehicleState],
        current_time: float,
    ) -> list[tuple[str, str]]:
        """Find ungrouped vehicles that could join existing platoons.

        Returns
        -------
        list[tuple[str, str]]
            List of (vehicle_id, platoon_id) join candidates.
        """
        grouped = set()
        for p in platoons:
            grouped.update(p.vehicle_ids)

        ungrouped_states = [s for s in states if s.vehicle_id not in grouped]
        if not ungrouped_states:
            return []

        max_size = int(self._plat_conf.get("maximum_platoon_size", 5))
        state_map = {s.vehicle_id: s for s in states}
        candidates: list[tuple[str, str]] = []

        for p in platoons:
            if p.size >= max_size:
                continue

            for u_state in ungrouped_states:
                all_compatible = True
                for member_id in p.vehicle_ids:
                    m_state = state_map.get(member_id)
                    if m_state is None:
                        all_compatible = False
                        break
                    comp = evaluate_compatibility(
                        u_state, m_state, self._config, current_time
                    )
                    if not comp.compatible:
                        all_compatible = False
                        break

                if all_compatible:
                    candidates.append((u_state.vehicle_id, p.platoon_id))

        return candidates
