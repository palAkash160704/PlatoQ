"""
Tests for PlatoonManager and classical platoon formation logic.
"""

import pytest

from platooning.models.vehicle import VehicleState
from platooning.platooning.formation import (
    evaluate_compatibility,
    form_platoons,
    select_leader,
)
from platooning.platooning.platoon_manager import PlatoonManager


@pytest.fixture
def config():
    return {
        "communication": {"stale_threshold_ms": 200.0},
        "platooning": {
            "enabled": True,
            "formation_interval_ms": 500,
            "max_formation_distance_m": 50.0,
            "max_speed_difference_mps": 3.0,
            "minimum_platoon_size": 2,
            "maximum_platoon_size": 5,
        },
    }


def test_evaluate_compatibility_success(config):
    v1 = VehicleState(
        "v1",
        timestamp=10.0,
        position_x=10.0,
        position_y=0.0,
        speed=20.0,
        route_id="r1",
        lane_id="l1",
    )
    v2 = VehicleState(
        "v2",
        timestamp=10.0,
        position_x=20.0,
        position_y=0.0,
        speed=21.0,
        route_id="r1",
        lane_id="l1",
    )

    comp = evaluate_compatibility(v1, v2, config, current_time=10.1)

    assert comp.compatible is True
    assert comp.reason == "compatible"
    assert comp.distance_m == pytest.approx(10.0)
    assert comp.speed_difference_mps == pytest.approx(1.0)
    assert comp.communication_fresh is True


def test_evaluate_compatibility_distance_exceeded(config):
    v1 = VehicleState(
        "v1",
        timestamp=10.0,
        position_x=10.0,
        position_y=0.0,
        speed=20.0,
        route_id="r1",
        lane_id="l1",
    )
    v2 = VehicleState(
        "v2",
        timestamp=10.0,
        position_x=80.0,
        position_y=0.0,
        speed=20.0,
        route_id="r1",
        lane_id="l1",
    )

    comp = evaluate_compatibility(v1, v2, config, current_time=10.1)
    assert comp.compatible is False
    assert comp.reason == "distance_exceeded"


def test_evaluate_compatibility_speed_difference(config):
    v1 = VehicleState(
        "v1",
        timestamp=10.0,
        position_x=10.0,
        position_y=0.0,
        speed=20.0,
        route_id="r1",
        lane_id="l1",
    )
    v2 = VehicleState(
        "v2",
        timestamp=10.0,
        position_x=20.0,
        position_y=0.0,
        speed=25.0,
        route_id="r1",
        lane_id="l1",
    )

    comp = evaluate_compatibility(v1, v2, config, current_time=10.1)
    assert comp.compatible is False
    assert comp.reason == "speed_difference_exceeded"


def test_evaluate_compatibility_different_route(config):
    v1 = VehicleState(
        "v1",
        timestamp=10.0,
        position_x=10.0,
        position_y=0.0,
        speed=20.0,
        route_id="r1",
        lane_id="l1",
    )
    v2 = VehicleState(
        "v2",
        timestamp=10.0,
        position_x=20.0,
        position_y=0.0,
        speed=20.0,
        route_id="r2",
        lane_id="l1",
    )

    comp = evaluate_compatibility(v1, v2, config, current_time=10.1)
    assert comp.compatible is False
    assert comp.reason == "different_route"


def test_evaluate_compatibility_different_lane(config):
    v1 = VehicleState(
        "v1",
        timestamp=10.0,
        position_x=10.0,
        position_y=0.0,
        speed=20.0,
        route_id="r1",
        lane_id="l1",
    )
    v2 = VehicleState(
        "v2",
        timestamp=10.0,
        position_x=20.0,
        position_y=0.0,
        speed=20.0,
        route_id="r1",
        lane_id="l2",
    )

    comp = evaluate_compatibility(v1, v2, config, current_time=10.1)
    assert comp.compatible is False
    assert comp.reason == "different_lane"


def test_evaluate_compatibility_stale_state(config):
    v1 = VehicleState(
        "v1",
        timestamp=9.0,
        position_x=10.0,
        position_y=0.0,
        speed=20.0,
        route_id="r1",
        lane_id="l1",
    )
    v2 = VehicleState(
        "v2",
        timestamp=10.0,
        position_x=20.0,
        position_y=0.0,
        speed=20.0,
        route_id="r1",
        lane_id="l1",
    )

    comp = evaluate_compatibility(
        v1, v2, config, current_time=10.5
    )  # Age is 1.5s for v1 > 0.2s
    assert comp.compatible is False
    assert comp.reason == "stale_state"


def test_select_leader():
    v1 = VehicleState("v1", position_x=10.0)
    v2 = VehicleState("v2", position_x=20.0)
    assert select_leader([v1, v2]) == "v2"

    v3 = VehicleState("v3", position_x=20.0)  # Tie
    assert select_leader([v2, v3]) == "v2"  # By ID


def test_form_platoons_two_vehicles(config):
    v1 = VehicleState(
        "v1", timestamp=10.0, position_x=20.0, route_id="r1", lane_id="l1"
    )
    v2 = VehicleState(
        "v2", timestamp=10.0, position_x=10.0, route_id="r1", lane_id="l1"
    )

    platoons = form_platoons([v1, v2], config, 10.0)
    assert len(platoons) == 1
    assert platoons[0].size == 2
    assert platoons[0].leader_id == "v1"


def test_form_platoons_three_vehicles(config):
    v1 = VehicleState(
        "v1", timestamp=10.0, position_x=30.0, route_id="r1", lane_id="l1"
    )
    v2 = VehicleState(
        "v2", timestamp=10.0, position_x=20.0, route_id="r1", lane_id="l1"
    )
    v3 = VehicleState(
        "v3", timestamp=10.0, position_x=10.0, route_id="r1", lane_id="l1"
    )

    platoons = form_platoons([v1, v2, v3], config, 10.0)
    assert len(platoons) == 1
    assert platoons[0].size == 3
    assert platoons[0].leader_id == "v1"


def test_max_platoon_size(config):
    config["platooning"]["maximum_platoon_size"] = 3
    # 4 vehicles close together
    states = [
        VehicleState(
            f"v{i}", timestamp=10.0, position_x=10.0 + i, route_id="r1", lane_id="l1"
        )
        for i in range(4)
    ]
    platoons = form_platoons(states, config, 10.0)

    # Expected: one platoon of 3, one vehicle unassigned (since min=2)
    assert len(platoons) == 1
    assert platoons[0].size == 3


def test_no_compatible_vehicles(config):
    v1 = VehicleState(
        "v1", timestamp=10.0, position_x=10.0, route_id="r1", lane_id="l1"
    )
    v2 = VehicleState(
        "v2", timestamp=10.0, position_x=80.0, route_id="r1", lane_id="l1"
    )  # Too far

    platoons = form_platoons([v1, v2], config, 10.0)
    assert len(platoons) == 0


def test_determinism(config):
    states = [
        VehicleState(
            f"v{i}", timestamp=10.0, position_x=10.0 + i, route_id="r1", lane_id="l1"
        )
        for i in range(5)
    ]
    platoons1 = form_platoons(states, config, 10.0)
    platoons2 = form_platoons(states, config, 10.0)

    assert len(platoons1) == len(platoons2)
    for p1, p2 in zip(platoons1, platoons2, strict=True):
        assert set(p1.vehicle_ids) == set(p2.vehicle_ids)
        assert p1.leader_id == p2.leader_id


def test_manager_lifecycle(config):
    pm = PlatoonManager(config)
    v1 = VehicleState(
        "v1", timestamp=10.0, position_x=20.0, route_id="r1", lane_id="l1"
    )
    v2 = VehicleState(
        "v2", timestamp=10.0, position_x=10.0, route_id="r1", lane_id="l1"
    )

    latest_known = {"v1": {"v2": v2}, "v2": {"v1": v1}}

    changed = pm.update_vehicle_states(latest_known, 10.0)
    assert changed is True
    assert len(pm.get_all_platoons()) == 1

    metrics = pm.get_formation_metrics()
    assert metrics.number_of_platoons == 1
    assert metrics.average_platoon_size == 2.0
    assert metrics.vehicles_in_platoons == 2
    assert metrics.vehicles_not_in_platoons == 0
    assert metrics.compatible_pairs == 1


def test_manager_vehicle_leaves(config):
    pm = PlatoonManager(config)
    v1 = VehicleState(
        "v1", timestamp=10.0, position_x=20.0, route_id="r1", lane_id="l1"
    )
    v2 = VehicleState(
        "v2", timestamp=10.0, position_x=10.0, route_id="r1", lane_id="l1"
    )

    latest_known = {"v1": {"v2": v2}, "v2": {"v1": v1}}
    pm.update_vehicle_states(latest_known, 10.0)

    # Vehicle 2 leaves
    latest_known_new = {"v1": {}}
    changed = pm.update_vehicle_states(latest_known_new, 10.5)  # Interval is 500ms
    assert changed is True
    assert len(pm.get_all_platoons()) == 0


def test_manager_vehicle_becomes_stale(config):
    pm = PlatoonManager(config)
    v1 = VehicleState(
        "v1", timestamp=10.0, position_x=20.0, route_id="r1", lane_id="l1"
    )
    v2 = VehicleState(
        "v2", timestamp=10.0, position_x=10.0, route_id="r1", lane_id="l1"
    )

    latest_known = {"v1": {"v2": v2}, "v2": {"v1": v1}}
    pm.update_vehicle_states(latest_known, 10.0)

    # At t=10.5, v1's states are still timestamp=10.0, which means age=0.5s > 0.2s stale threshold
    changed = pm.update_vehicle_states(latest_known, 10.5)
    assert changed is True
    assert len(pm.get_all_platoons()) == 0
