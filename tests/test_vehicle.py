"""Tests for the VehicleState data model."""

from platooning.models.vehicle import VehicleState


class TestVehicleStateCreation:
    """Verify that VehicleState can be instantiated with valid data."""

    def test_minimal_creation(self) -> None:
        """A VehicleState with only the required field should be valid."""
        vs = VehicleState(vehicle_id="v1")
        assert vs.vehicle_id == "v1"
        assert vs.speed == 0.0
        assert vs.platoon_id is None

    def test_full_creation(self) -> None:
        """A VehicleState with all fields populated should preserve values."""
        vs = VehicleState(
            vehicle_id="v42",
            timestamp=10.5,
            position_x=100.0,
            position_y=3.5,
            speed=25.0,
            acceleration=1.2,
            lane_id="edge_0_0",
            route_id="route_0",
            destination="junction_5",
            vehicle_type="truck",
            platoon_id="p1",
        )
        assert vs.vehicle_id == "v42"
        assert vs.timestamp == 10.5
        assert vs.position_x == 100.0
        assert vs.speed == 25.0
        assert vs.acceleration == 1.2
        assert vs.lane_id == "edge_0_0"
        assert vs.route_id == "route_0"
        assert vs.destination == "junction_5"
        assert vs.vehicle_type == "truck"
        assert vs.platoon_id == "p1"

    def test_is_platooned_true(self) -> None:
        """is_platooned returns True when a platoon_id is assigned."""
        vs = VehicleState(vehicle_id="v1", platoon_id="p1")
        assert vs.is_platooned is True

    def test_is_platooned_false(self) -> None:
        """is_platooned returns False when no platoon_id is set."""
        vs = VehicleState(vehicle_id="v1")
        assert vs.is_platooned is False

    def test_default_vehicle_type(self) -> None:
        """Default vehicle_type should be 'passenger'."""
        vs = VehicleState(vehicle_id="v1")
        assert vs.vehicle_type == "passenger"
