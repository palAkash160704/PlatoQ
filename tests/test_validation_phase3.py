"""
Phase 3 Validation Tests.
"""

import pytest

from platooning.models.vehicle import VehicleState
from platooning.platooning.formation import form_platoons
from platooning.platooning.platoon_manager import PlatoonManager


@pytest.fixture
def config():
    return {
        "communication": {"stale_threshold_ms": 500.0},
        "platooning": {
            "enabled": True,
            "formation_interval_ms": 100,
            "max_formation_distance_m": 500.0,
            "max_speed_difference_mps": 10.0,
            "minimum_platoon_size": 2,
            "maximum_platoon_size": 5,
            "weight_membership": 10.0,
            "weight_distance_penalty": 0.5,
            "weight_speed_penalty": 1.0,
            "weight_ungrouped_penalty": 5.0,
        },
    }


def test_validation_maximum_platoon_size(config):
    """
    Test with 6 fully compatible vehicles and maximum_platoon_size = 5.
    Confirms that no platoon can contain more than 5 vehicles.
    """
    # Create 6 identical vehicles with just slightly different positions
    states = [
        VehicleState(
            vehicle_id=f"v{i}",
            timestamp=10.0,
            position_x=100.0 - i * 5.0,  # 5m gap between each
            position_y=0.0,
            speed=30.0,
            route_id="main_route",
            lane_id="lane_0",
        )
        for i in range(1, 7)  # v1 to v6
    ]

    # Verify we have exactly 6 vehicles
    assert len(states) == 6

    # Form platoons
    platoons = form_platoons(states, config, current_time=10.1)

    # We expect 1 platoon of 5 (the max size limit), and 1 ungrouped vehicle
    # (Since minimum size is 2, the remaining 1 vehicle cannot form a platoon)
    assert len(platoons) == 1
    assert platoons[0].size == 5

    for p in platoons:
        assert p.size <= 5


def test_validation_determinism(config):
    """
    Run the same formation input multiple times.
    Confirm that platoon membership, leader selection, and objective value are identical.
    """
    states = [
        VehicleState(
            vehicle_id=f"v{i}",
            timestamp=10.0,
            position_x=100.0 - i * 5.0,
            position_y=0.0,
            speed=30.0,
            route_id="main_route",
            lane_id="lane_0",
        )
        for i in range(1, 7)
    ]

    # V2V perspective
    latest_known = {
        s.vehicle_id: {other.vehicle_id: other for other in states} for s in states
    }

    # Manager 1 run
    pm1 = PlatoonManager(config)
    pm1.update_vehicle_states(latest_known, 10.1)

    # Manager 2 run
    pm2 = PlatoonManager(config)
    pm2.update_vehicle_states(latest_known, 10.1)

    platoons1 = pm1.get_all_platoons()
    platoons2 = pm2.get_all_platoons()

    assert len(platoons1) == len(platoons2)

    for p1, p2 in zip(platoons1, platoons2, strict=True):
        # Membership identical
        assert set(p1.vehicle_ids) == set(p2.vehicle_ids)
        # Leader identical
        assert p1.leader_id == p2.leader_id

    m1 = pm1.get_formation_metrics()
    m2 = pm2.get_formation_metrics()

    # Objective value identical
    assert m1.objective_value == m2.objective_value
