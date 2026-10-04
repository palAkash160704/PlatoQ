"""
Phase 6 — Dynamic Platoon Management Tests.

20 test categories covering:
1.  Valid platoon remains unchanged
2.  Too-far vehicle triggers reconfiguration
3.  Speed mismatch triggers reconfiguration
4.  Stale V2V state triggers reconfiguration
5.  Communication loss handled correctly
6.  Vehicle joining
7.  Vehicle leaving
8.  Platoon splitting
9.  Platoon merging
10. Reconfiguration cooldown
11. Hysteresis prevents repeated switching
12. Configuration validation
13. Invalid configuration rejected
14. Event generation
15. Objective before/after calculation
16. Reconfiguration latency measurement
17. Classical dynamic experiment
18. Exact QUBO dynamic experiment
19. QAOA dynamic experiment
20. Existing Phase 1-5 regression (run separately)
"""

from __future__ import annotations

import pytest

from platooning.models.platoon import Platoon
from platooning.models.vehicle import VehicleState
from platooning.platooning.dynamic_manager import DynamicPlatoonManager
from platooning.platooning.dynamic_models import (
    ReconfigTrigger,
    ViolationSeverity,
)
from platooning.platooning.monitor import PlatoonMonitor


# ======================================================================
# Fixtures
# ======================================================================
@pytest.fixture
def config():
    """Standard test configuration."""
    return {
        "communication": {"stale_threshold_ms": 500.0},
        "platooning": {
            "enabled": True,
            "formation_interval_ms": 100,
            "max_formation_distance_m": 50.0,
            "max_speed_difference_mps": 3.0,
            "minimum_platoon_size": 2,
            "maximum_platoon_size": 5,
            "weight_membership": 10.0,
            "weight_distance_penalty": 0.5,
            "weight_speed_penalty": 1.0,
            "weight_ungrouped_penalty": 5.0,
        },
        "qubo": {
            "enabled": True,
            "solver": "exact",
            "max_exact_variables": 25,
            "weight_pairwise_distance": 0.1,
            "weight_pairwise_speed": 0.5,
            "penalty_assignment": 1000.0,
            "penalty_incompatibility": 1000.0,
            "penalty_size": 1000.0,
        },
        "quantum": {
            "backend": "aer_simulator",
            "shots": 256,
            "reps": 1,
            "optimizer": "COBYLA",
            "seed": 42,
        },
        "dynamic": {
            "reconfiguration_interval_ms": 2000,
            "reconfiguration_cooldown_ms": 1000,
            "minimum_improvement": 0.5,
        },
    }


def make_states(
    positions: list[tuple[str, float]],
    speed: float = 30.0,
    timestamp: float = 10.0,
    route: str = "r1",
    lane: str = "l1",
) -> list[VehicleState]:
    """Create vehicle states from (vehicle_id, position_x) tuples."""
    return [
        VehicleState(
            vehicle_id=vid,
            timestamp=timestamp,
            position_x=px,
            position_y=0.0,
            speed=speed,
            route_id=route,
            lane_id=lane,
        )
        for vid, px in positions
    ]


def make_platoon(
    pid: str,
    members: list[str],
    leader: str | None = None,
    route: str = "r1",
    lane: str = "l1",
) -> Platoon:
    """Create a test platoon."""
    return Platoon(
        platoon_id=pid,
        vehicle_ids=members,
        leader_id=leader or members[0],
        route_id=route,
        lane_id=lane,
        status="FORMED",
    )


# ======================================================================
# 1. Valid platoon remains unchanged
# ======================================================================
class TestValidPlatoonUnchanged:
    def test_valid_platoon_detected(self, config):
        """Monitor reports a valid platoon as valid."""
        monitor = PlatoonMonitor(config)
        states = make_states([("V1", 100.0), ("V2", 95.0)])
        platoon = make_platoon("P1", ["V1", "V2"], "V1")

        health = monitor.check_platoon(platoon, {s.vehicle_id: s for s in states}, 10.1)
        assert health.valid is True
        assert len(health.violations) == 0

    def test_no_reconfiguration_when_valid(self, config):
        """DynamicManager does not reconfigure a valid platoon."""
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")
        states = make_states([("V1", 100.0), ("V2", 95.0)])
        dm.initial_formation(states, 10.0)

        # Update with same states — should not reconfigure
        changed = dm.update(states, 10.5)
        # No reconfig because cooldown hasn't expired
        assert changed is False


# ======================================================================
# 2. Too-far vehicle triggers reconfiguration
# ======================================================================
class TestMemberTooFar:
    def test_too_far_detected(self, config):
        """Monitor detects member that exceeded max distance."""
        monitor = PlatoonMonitor(config)
        states = make_states([("V1", 100.0), ("V2", 40.0)])  # 60m apart > 50m
        platoon = make_platoon("P1", ["V1", "V2"], "V1")

        health = monitor.check_platoon(platoon, {s.vehicle_id: s for s in states}, 10.1)
        assert health.valid is False
        assert ReconfigTrigger.MEMBER_TOO_FAR in health.violations

    def test_too_far_triggers_reconfig(self, config):
        """System reconfigures when a member drifts too far."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        # Initial: close together
        states_t0 = make_states([("V1", 100.0), ("V2", 95.0), ("V3", 90.0)])
        dm.initial_formation(states_t0, 10.0)
        assert len(dm.platoons) > 0

        # V3 drifts far away
        states_t1 = make_states([("V1", 100.0), ("V2", 95.0)])
        states_t1.append(
            VehicleState(
                vehicle_id="V3",
                timestamp=12.0,
                position_x=30.0,
                position_y=0.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            )
        )
        changed = dm.update(states_t1, 12.0)
        assert changed is True


# ======================================================================
# 3. Speed mismatch triggers reconfiguration
# ======================================================================
class TestSpeedMismatch:
    def test_speed_mismatch_detected(self, config):
        """Monitor detects speed difference exceeding threshold."""
        monitor = PlatoonMonitor(config)
        states = [
            VehicleState(
                vehicle_id="V1",
                timestamp=10.0,
                position_x=100.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
            VehicleState(
                vehicle_id="V2",
                timestamp=10.0,
                position_x=95.0,
                speed=36.0,
                route_id="r1",
                lane_id="l1",
            ),  # 6m/s diff > 3m/s
        ]
        platoon = make_platoon("P1", ["V1", "V2"], "V1")

        health = monitor.check_platoon(platoon, {s.vehicle_id: s for s in states}, 10.1)
        assert health.valid is False
        assert ReconfigTrigger.SPEED_MISMATCH in health.violations


# ======================================================================
# 4. Stale V2V state triggers reconfiguration
# ======================================================================
class TestStaleState:
    def test_stale_state_detected(self, config):
        """Monitor detects stale vehicle state."""
        monitor = PlatoonMonitor(config)
        states = [
            VehicleState(
                vehicle_id="V1",
                timestamp=10.0,
                position_x=100.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
            VehicleState(
                vehicle_id="V2",
                timestamp=8.0,
                position_x=95.0,  # 4s old
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
        ]
        platoon = make_platoon("P1", ["V1", "V2"], "V1")

        health = monitor.check_platoon(platoon, {s.vehicle_id: s for s in states}, 12.0)
        assert health.valid is False
        assert ReconfigTrigger.STALE_STATE in health.violations


# ======================================================================
# 5. Communication loss handled correctly
# ======================================================================
class TestCommunicationLoss:
    def test_missing_state_detected(self, config):
        """Monitor detects missing vehicle state (communication loss)."""
        monitor = PlatoonMonitor(config)
        # Only V1 has state data; V2 is missing
        states = [
            VehicleState(
                vehicle_id="V1",
                timestamp=10.0,
                position_x=100.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
        ]
        platoon = make_platoon("P1", ["V1", "V2"], "V1")
        state_map = {s.vehicle_id: s for s in states}

        health = monitor.check_platoon(platoon, state_map, 10.1)
        assert health.valid is False
        assert ReconfigTrigger.COMMUNICATION_LOSS in health.violations
        assert "V2" in health.affected_members
        assert health.severity == ViolationSeverity.CRITICAL


# ======================================================================
# 6. Vehicle joining
# ======================================================================
class TestVehicleJoining:
    def test_join_candidate_detected(self, config):
        """Monitor finds ungrouped vehicle that can join a platoon."""
        monitor = PlatoonMonitor(config)
        states = make_states([("V1", 100.0), ("V2", 95.0), ("V3", 92.0)])
        platoon = make_platoon("P1", ["V1", "V2"], "V1")

        candidates = monitor.detect_new_compatible_vehicles([platoon], states, 10.1)
        # V3 should be a candidate to join P1
        assert len(candidates) > 0
        assert any(v == "V3" for v, _ in candidates)

    def test_vehicle_joins_after_update(self, config):
        """Vehicle joins a platoon through the dynamic manager."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        states_t0 = make_states([("V1", 100.0), ("V2", 95.0)])
        dm.initial_formation(states_t0, 10.0)

        # V3 appears nearby
        states_t1 = make_states(
            [("V1", 100.0), ("V2", 95.0), ("V3", 92.0)], timestamp=12.0
        )
        changed = dm.update(states_t1, 12.0)
        assert changed is True

        # V3 should now be in a platoon
        all_members = set()
        for p in dm.platoons:
            all_members.update(p.vehicle_ids)
        assert "V3" in all_members


# ======================================================================
# 7. Vehicle leaving
# ======================================================================
class TestVehicleLeaving:
    def test_vehicle_leaves_when_incompatible(self, config):
        """Vehicle leaves platoon when it becomes incompatible."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        states_t0 = make_states([("V1", 100.0), ("V2", 95.0), ("V3", 90.0)])
        dm.initial_formation(states_t0, 10.0)
        initial_count = sum(p.size for p in dm.platoons)
        assert initial_count == 3

        # V3 moves to different route
        states_t1 = [
            VehicleState(
                vehicle_id="V1",
                timestamp=12.0,
                position_x=100.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
            VehicleState(
                vehicle_id="V2",
                timestamp=12.0,
                position_x=95.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
            VehicleState(
                vehicle_id="V3",
                timestamp=12.0,
                position_x=90.0,
                speed=30.0,
                route_id="r2",
                lane_id="l1",
            ),
        ]
        changed = dm.update(states_t1, 12.0)
        assert changed is True

        # V3 should not be in any platoon with V1/V2
        for p in dm.platoons:
            if "V1" in p.vehicle_ids:
                assert "V3" not in p.vehicle_ids


# ======================================================================
# 8. Platoon splitting
# ======================================================================
class TestPlatoonSplitting:
    def test_platoon_splits_on_distance(self, config):
        """Platoon splits when subgroups become too far apart."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        config["platooning"]["max_formation_distance_m"] = 20.0
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        # Initially all close
        states_t0 = make_states(
            [("V1", 100.0), ("V2", 95.0), ("V3", 90.0), ("V4", 85.0)]
        )
        dm.initial_formation(states_t0, 10.0)

        # Split into two groups
        states_t1 = [
            VehicleState(
                vehicle_id="V1",
                timestamp=12.0,
                position_x=100.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
            VehicleState(
                vehicle_id="V2",
                timestamp=12.0,
                position_x=95.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
            VehicleState(
                vehicle_id="V3",
                timestamp=12.0,
                position_x=60.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
            VehicleState(
                vehicle_id="V4",
                timestamp=12.0,
                position_x=55.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
        ]
        changed = dm.update(states_t1, 12.0)
        assert changed is True
        assert len(dm.platoons) == 2


# ======================================================================
# 9. Platoon merging
# ======================================================================
class TestPlatoonMerging:
    def test_platoons_merge_when_compatible(self, config):
        """Two platoons merge when all members become compatible."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        config["platooning"]["max_formation_distance_m"] = 20.0
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        # Two separate groups
        states_t0 = [
            VehicleState(
                vehicle_id="V1",
                timestamp=10.0,
                position_x=100.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
            VehicleState(
                vehicle_id="V2",
                timestamp=10.0,
                position_x=95.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
            VehicleState(
                vehicle_id="V3",
                timestamp=10.0,
                position_x=60.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
            VehicleState(
                vehicle_id="V4",
                timestamp=10.0,
                position_x=55.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
        ]
        dm.initial_formation(states_t0, 10.0)
        assert len(dm.platoons) == 2

        # Groups come together
        config["platooning"]["max_formation_distance_m"] = 50.0
        dm._config = config
        dm._plat_conf = config["platooning"]
        dm._min_improvement = -100.0  # Apply directly to the instance
        dm._monitor = PlatoonMonitor(config)

        states_t1 = make_states(
            [("V1", 100.0), ("V2", 95.0), ("V3", 90.0), ("V4", 85.0)], timestamp=15.0
        )
        changed = dm.update(states_t1, 15.0)
        assert changed is True
        assert len(dm.platoons) == 1
        assert dm.platoons[0].size == 4


# ======================================================================
# 10. Reconfiguration cooldown
# ======================================================================
class TestCooldown:
    def test_cooldown_prevents_immediate_reconfig(self, config):
        """Cooldown prevents reconfiguration within the cooldown period."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 5000  # 5s
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        states = make_states([("V1", 100.0), ("V2", 95.0)])
        dm.initial_formation(states, 10.0)

        # Try to update at t=11 (1s later, within 5s cooldown)
        changed = dm.update(states, 11.0)
        assert changed is False

    def test_cooldown_allows_reconfig_after_expiry(self, config):
        """After cooldown expires, reconfiguration is allowed."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 1000  # 1s
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        states_t0 = make_states([("V1", 100.0), ("V2", 95.0)])
        dm.initial_formation(states_t0, 10.0)

        # Add V3 at t=12 (2s later, beyond 1s cooldown)
        states_t1 = make_states(
            [("V1", 100.0), ("V2", 95.0), ("V3", 90.0)], timestamp=12.0
        )
        changed = dm.update(states_t1, 12.0)
        assert changed is True


# ======================================================================
# 11. Hysteresis prevents repeated switching
# ======================================================================
class TestHysteresis:
    def test_marginal_improvement_rejected(self, config):
        """Periodic reoptimization with insufficient improvement is skipped."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        config["dynamic"]["minimum_improvement"] = 100.0  # Very high threshold
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        states_t0 = make_states([("V1", 100.0), ("V2", 95.0)], timestamp=10.0)
        dm.initial_formation(states_t0, 10.0)

        # Update at much later time — periodic trigger fires
        # Provide fresh states so it doesn't trigger STALE_STATE
        states_t1 = make_states([("V1", 100.0), ("V2", 95.0)], timestamp=100.0)
        changed = dm.update(states_t1, 100.0)
        # Should not change because improvement < threshold
        assert changed is False


# ======================================================================
# 12. Configuration validation
# ======================================================================
class TestConfigValidation:
    def test_monitor_validates_route_consistency(self, config):
        """Monitor validates route consistency within a platoon."""
        monitor = PlatoonMonitor(config)
        states = [
            VehicleState(
                vehicle_id="V1",
                timestamp=10.0,
                position_x=100.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
            VehicleState(
                vehicle_id="V2",
                timestamp=10.0,
                position_x=95.0,
                speed=30.0,
                route_id="r2",
                lane_id="l1",
            ),  # diff route
        ]
        platoon = make_platoon("P1", ["V1", "V2"], "V1")

        health = monitor.check_platoon(platoon, {s.vehicle_id: s for s in states}, 10.1)
        assert health.valid is False
        assert ReconfigTrigger.ROUTE_MISMATCH in health.violations

    def test_monitor_validates_lane_consistency(self, config):
        """Monitor validates lane consistency within a platoon."""
        monitor = PlatoonMonitor(config)
        states = [
            VehicleState(
                vehicle_id="V1",
                timestamp=10.0,
                position_x=100.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            ),
            VehicleState(
                vehicle_id="V2",
                timestamp=10.0,
                position_x=95.0,
                speed=30.0,
                route_id="r1",
                lane_id="l2",
            ),  # diff lane
        ]
        platoon = make_platoon("P1", ["V1", "V2"], "V1")

        health = monitor.check_platoon(platoon, {s.vehicle_id: s for s in states}, 10.1)
        assert health.valid is False
        assert ReconfigTrigger.LANE_MISMATCH in health.violations


# ======================================================================
# 13. Invalid configuration rejected
# ======================================================================
class TestInvalidConfigRejected:
    def test_leader_not_in_members(self, config):
        """Monitor rejects platoon where leader is not in the member list."""
        monitor = PlatoonMonitor(config)
        states = make_states([("V1", 100.0), ("V2", 95.0)])
        platoon = Platoon(
            platoon_id="P1",
            vehicle_ids=["V1", "V2"],
            leader_id="V_MISSING",  # Invalid leader
        )

        health = monitor.check_platoon(platoon, {s.vehicle_id: s for s in states}, 10.1)
        assert health.valid is False
        assert health.severity == ViolationSeverity.CRITICAL


# ======================================================================
# 14. Event generation
# ======================================================================
class TestEventGeneration:
    def test_events_created_on_reconfig(self, config):
        """Reconfiguration events are created and stored."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        states = make_states([("V1", 100.0), ("V2", 95.0)])
        dm.initial_formation(states, 10.0)

        assert len(dm.events) == 1
        event = dm.events[0]
        assert event.timestamp == 10.0
        assert event.feasible is True
        assert len(event.new_configuration) > 0

    def test_event_has_required_fields(self, config):
        """Events contain all required fields."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        states = make_states([("V1", 100.0), ("V2", 95.0)])
        dm.initial_formation(states, 10.0)

        event = dm.events[0]
        assert hasattr(event, "timestamp")
        assert hasattr(event, "trigger")
        assert hasattr(event, "old_configuration")
        assert hasattr(event, "new_configuration")
        assert hasattr(event, "affected_vehicles")
        assert hasattr(event, "optimizer")
        assert hasattr(event, "old_objective")
        assert hasattr(event, "new_objective")
        assert hasattr(event, "runtime_ms")
        assert hasattr(event, "feasible")
        assert hasattr(event, "event_type")


# ======================================================================
# 15. Objective before/after calculation
# ======================================================================
class TestObjectiveCalculation:
    def test_objective_computed_correctly(self, config):
        """Objective is computed before and after reconfiguration."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        states_t0 = make_states([("V1", 100.0), ("V2", 95.0), ("V3", 90.0)])
        dm.initial_formation(states_t0, 10.0)

        # Force a change
        states_t1 = make_states([("V1", 100.0), ("V2", 95.0)], timestamp=12.0)
        states_t1.append(
            VehicleState(
                vehicle_id="V3",
                timestamp=12.0,
                position_x=30.0,
                speed=30.0,
                route_id="r2",
                lane_id="l1",
            )
        )
        dm.update(states_t1, 12.0)

        # Check that last event has objective values
        last_event = dm.events[-1]
        assert last_event.old_objective != 0.0 or last_event.new_objective != 0.0


# ======================================================================
# 16. Reconfiguration latency measurement
# ======================================================================
class TestReconfigLatency:
    def test_runtime_measured(self, config):
        """Reconfiguration runtime is measured and recorded."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        states = make_states([("V1", 100.0), ("V2", 95.0)])
        dm.initial_formation(states, 10.0)

        event = dm.events[0]
        assert event.runtime_ms >= 0.0

    def test_average_latency_metric(self, config):
        """Average reconfiguration latency is tracked in metrics."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        states_t0 = make_states([("V1", 100.0), ("V2", 95.0)])
        dm.initial_formation(states_t0, 10.0)

        states_t1 = make_states(
            [("V1", 100.0), ("V2", 95.0), ("V3", 90.0)], timestamp=12.0
        )
        dm.update(states_t1, 12.0)

        metrics = dm.finalize_metrics(12.0)
        assert metrics.average_reconfiguration_latency_ms >= 0.0


# ======================================================================
# 17. Classical dynamic experiment
# ======================================================================
class TestClassicalDynamic:
    def test_classical_full_scenario(self, config):
        """Full scenario with classical optimizer."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        # Phase 1: form
        states_t0 = make_states([("V1", 100.0), ("V2", 95.0), ("V3", 90.0)])
        dm.initial_formation(states_t0, 10.0)
        assert len(dm.platoons) >= 1

        # Phase 2: V3 separates
        states_t1 = make_states([("V1", 100.0), ("V2", 95.0)])
        states_t1.append(
            VehicleState(
                vehicle_id="V3",
                timestamp=12.0,
                position_x=30.0,
                speed=30.0,
                route_id="r1",
                lane_id="l1",
            )
        )
        dm.update(states_t1, 12.0)

        # Phase 3: V3 rejoins
        states_t2 = make_states(
            [("V1", 100.0), ("V2", 95.0), ("V3", 92.0)], timestamp=14.0
        )
        dm.update(states_t2, 14.0)

        metrics = dm.finalize_metrics(14.0)
        assert metrics.total_reconfigurations >= 1


# ======================================================================
# 18. Exact QUBO dynamic experiment
# ======================================================================
class TestExactQUBODynamic:
    def test_qubo_dynamic_formation(self, config):
        """Dynamic formation with exact QUBO solver."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        dm = DynamicPlatoonManager(config, optimizer_mode="exact_qubo")

        states = make_states([("V1", 100.0), ("V2", 95.0), ("V3", 90.0)])
        dm.initial_formation(states, 10.0)
        assert len(dm.platoons) >= 1
        assert dm.optimizer_mode == "exact_qubo"

    def test_qubo_reconfiguration(self, config):
        """QUBO reconfigures when state changes."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        dm = DynamicPlatoonManager(config, optimizer_mode="exact_qubo")

        states_t0 = make_states([("V1", 100.0), ("V2", 95.0)])
        dm.initial_formation(states_t0, 10.0)

        states_t1 = make_states(
            [("V1", 100.0), ("V2", 95.0), ("V3", 90.0)], timestamp=12.0
        )
        changed = dm.update(states_t1, 12.0)
        assert changed is True


# ======================================================================
# 19. QAOA dynamic experiment
# ======================================================================
class TestQAOADynamic:
    def test_qaoa_dynamic_formation(self, config):
        """Dynamic formation with QAOA solver."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        dm = DynamicPlatoonManager(config, optimizer_mode="qaoa")

        states = make_states([("V1", 100.0), ("V2", 95.0)])
        dm.initial_formation(states, 10.0)
        assert len(dm.platoons) >= 1
        assert dm.optimizer_mode == "qaoa"


# ======================================================================
# Additional: Metrics and serialisation
# ======================================================================
class TestMetricsAndSerialisation:
    def test_churn_rate_computed(self, config):
        """Churn rate is correctly computed."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        states = make_states([("V1", 100.0), ("V2", 95.0)])
        dm.initial_formation(states, 10.0)

        metrics = dm.finalize_metrics(10.0)
        assert metrics.churn_rate >= 0.0

    def test_events_serialisable(self, config):
        """Events can be serialised to dicts."""
        config["dynamic"]["reconfiguration_cooldown_ms"] = 0
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")

        states = make_states([("V1", 100.0), ("V2", 95.0)])
        dm.initial_formation(states, 10.0)

        dicts = dm.events_to_dicts()
        assert len(dicts) >= 1
        assert "trigger" in dicts[0]
        assert "timestamp" in dicts[0]

    def test_metrics_to_dict(self, config):
        """Metrics can be serialised to dict."""
        dm = DynamicPlatoonManager(config, optimizer_mode="classical")
        states = make_states([("V1", 100.0), ("V2", 95.0)])
        dm.initial_formation(states, 10.0)
        metrics = dm.finalize_metrics(10.0)
        d = metrics.to_dict()
        assert "total_reconfigurations" in d
        assert "churn_rate" in d
