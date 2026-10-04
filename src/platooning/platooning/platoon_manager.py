"""
Platoon manager — central coordinator for platoon lifecycle operations.

The ``PlatoonManager`` receives latest_known_states from V2V, validates them,
delegates to formation logic to compute classical platoon configurations,
and manages platoon lifecycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from platooning.models.platoon import Platoon
from platooning.models.vehicle import VehicleState
from platooning.platooning.formation import build_compatibility_graph, form_platoons
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FormationMetrics:
    """Metrics tracking platoon formation outcomes."""

    number_of_vehicles: int = 0
    number_of_platoons: int = 0
    average_platoon_size: float = 0.0
    largest_platoon_size: int = 0
    vehicles_in_platoons: int = 0
    vehicles_not_in_platoons: int = 0
    average_inter_vehicle_distance: float = 0.0
    average_speed_difference: float = 0.0
    compatible_pairs: int = 0
    incompatible_pairs: int = 0
    objective_value: float = 0.0
    formation_time_ms: float = 0.0


class PlatoonManager:
    """High-level platoon lifecycle manager.

    Parameters
    ----------
    config : dict
        The full project configuration (accesses ``platooning``).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._plat_config = config.get("platooning", {})
        self._enabled: bool = self._plat_config.get("enabled", True)
        self._formation_interval: float = (
            float(self._plat_config.get("formation_interval_ms", 500)) / 1000.0
        )

        self._platoons: dict[str, Platoon] = {}

        self._last_formation_time: float = -float("inf")
        self._latest_metrics: FormationMetrics = FormationMetrics()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def get_platoon(self, platoon_id: str) -> Platoon | None:
        """Return the platoon with the given ID, or *None*."""
        return self._platoons.get(platoon_id)

    def get_all_platoons(self) -> list[Platoon]:
        """Return a list of all active platoons."""
        return list(self._platoons.values())

    def get_formation_metrics(self) -> FormationMetrics:
        """Return the latest calculated formation metrics."""
        return self._latest_metrics

    # ------------------------------------------------------------------
    # Vehicle state ingestion and Lifecycle
    # ------------------------------------------------------------------
    def update_vehicle_states(
        self,
        latest_known_states: dict[str, dict[str, VehicleState]],
        current_time: float,
    ) -> bool:
        """Evaluate and potentially reconfigure platoons based on new V2V data.

        This uses the *latest known states* provided by the V2V network,
        not absolute truth, ensuring formation adheres to communication limits.

        Parameters
        ----------
        latest_known_states : dict
            V2V mapping of receiver_id -> {sender_id -> VehicleState}
        current_time : float
            Current SUMO simulation time.

        Returns
        -------
        bool
            True if a new formation was generated, False otherwise.
        """
        if not self._enabled:
            return False

        if (current_time - self._last_formation_time) < (
            self._formation_interval - 1e-9
        ):
            return False

        self._last_formation_time = current_time

        # For the classical baseline where a centralized manager forms platoons,
        # the manager acts as a node that receives V2V broadcasts.
        # Alternatively, we aggregate the most recent state known to *any* node,
        # but the strictest interpretation is picking one manager perspective.
        # Here we flatten all known latest states across the network to represent
        # the manager's global (but V2V-constrained) view.
        global_states: dict[str, VehicleState] = {}
        for _receiver_id, senders in latest_known_states.items():
            for sender_id, state in senders.items():
                existing = global_states.get(sender_id)
                if existing is None or state.timestamp > existing.timestamp:
                    global_states[sender_id] = state

        valid_states = list(global_states.values())

        # Form platoons
        new_platoons = form_platoons(valid_states, self._config, current_time)

        # Check if configuration changed (simple equivalence check)
        changed = self._detect_changes(new_platoons)

        # Apply the new configuration
        self._platoons = {p.platoon_id: p for p in new_platoons}

        # Calculate metrics
        self._calculate_metrics(valid_states, new_platoons, current_time)

        if changed or len(new_platoons) > 0:
            self._log_platoons(current_time)

        return changed

    def _detect_changes(self, new_platoons: list[Platoon]) -> bool:
        """Return True if the new configuration differs from the current one."""
        if len(new_platoons) != len(self._platoons):
            return True

        current_sets = {frozenset(p.vehicle_ids) for p in self._platoons.values()}
        new_sets = {frozenset(p.vehicle_ids) for p in new_platoons}

        return current_sets != new_sets

    def _calculate_metrics(
        self,
        states: list[VehicleState],
        platoons: list[Platoon],
        current_time: float,
    ) -> None:
        """Calculate and store objective metrics for the current configuration."""
        metrics = FormationMetrics()
        metrics.number_of_vehicles = len(states)
        metrics.number_of_platoons = len(platoons)

        if platoons:
            metrics.average_platoon_size = sum(p.size for p in platoons) / len(platoons)
            metrics.largest_platoon_size = max(p.size for p in platoons)
            metrics.vehicles_in_platoons = sum(p.size for p in platoons)

        metrics.vehicles_not_in_platoons = (
            metrics.number_of_vehicles - metrics.vehicles_in_platoons
        )

        # Calculate graph metrics
        graph = build_compatibility_graph(states, self._config, current_time)
        compatible_count = 0
        incompatible_count = 0
        for comps in graph.values():
            for c in comps:
                if c.compatible:
                    compatible_count += 1
                else:
                    incompatible_count += 1

        # Graph edges are bidirectional, halve them for undirected pairs
        metrics.compatible_pairs = compatible_count // 2
        metrics.incompatible_pairs = incompatible_count // 2

        # Objective calculation weights
        w_membership = float(self._plat_config.get("weight_membership", 10.0))
        w_dist = float(self._plat_config.get("weight_distance_penalty", 0.5))
        w_speed = float(self._plat_config.get("weight_speed_penalty", 1.0))
        w_ungrouped = float(self._plat_config.get("weight_ungrouped_penalty", 5.0))

        objective = 0.0
        objective += metrics.vehicles_in_platoons * w_membership
        objective -= metrics.vehicles_not_in_platoons * w_ungrouped

        state_dict = {s.vehicle_id: s for s in states}
        total_dist = 0.0
        total_speed_diff = 0.0
        pairs_counted = 0

        for p in platoons:
            leader_state = state_dict[p.leader_id]
            for i in range(len(p.vehicle_ids) - 1):
                v1 = state_dict[p.vehicle_ids[i]]
                v2 = state_dict[p.vehicle_ids[i + 1]]
                dist = (
                    (v1.position_x - v2.position_x) ** 2
                    + (v1.position_y - v2.position_y) ** 2
                ) ** 0.5
                total_dist += dist
                pairs_counted += 1
                objective -= dist * w_dist

            for vid in p.vehicle_ids:
                v = state_dict[vid]
                speed_diff = abs(leader_state.speed - v.speed)
                total_speed_diff += speed_diff
                objective -= speed_diff * w_speed

        if pairs_counted > 0:
            metrics.average_inter_vehicle_distance = total_dist / pairs_counted
        if metrics.vehicles_in_platoons > 0:
            metrics.average_speed_difference = (
                total_speed_diff / metrics.vehicles_in_platoons
            )

        metrics.objective_value = objective
        self._latest_metrics = metrics

    def _log_platoons(self, current_time: float) -> None:
        """Log the current platoon configuration."""
        logger.info("=" * 50)
        logger.info("PLATOON FORMATION")
        logger.info("Simulation Time: %.1f s", current_time)
        logger.info("=" * 50)

        for p in self._platoons.values():
            logger.info("Platoon %s", p.platoon_id)
            logger.info("  Leader: %s", p.leader_id)
            logger.info("  Members: %s", ", ".join(p.vehicle_ids))
            logger.info("  Size: %d", p.size)
            logger.info("  Route: %s", p.route_id)
            logger.info("  Lane: %s\n", p.lane_id)

        m = self._latest_metrics
        logger.info("Metrics:")
        logger.info("  Vehicles: %d", m.number_of_vehicles)
        logger.info("  Platoons: %d", m.number_of_platoons)
        logger.info("  Average Size: %.1f", m.average_platoon_size)
        logger.info("  Compatible Pairs: %d", m.compatible_pairs)
        logger.info("  Formation Objective: %.2f", m.objective_value)
        logger.info("=" * 50)

    # ------------------------------------------------------------------
    # Platoon lifecycle (stubs to satisfy tests that still use them)
    # ------------------------------------------------------------------
    def form_platoon(self, vehicle_ids: list[str], leader_id: str) -> Platoon | None:
        """Legacy direct assignment. Use update_vehicle_states instead."""
        raise NotImplementedError("Use update_vehicle_states for Phase 3+.")
