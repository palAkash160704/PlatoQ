"""Tests for the Platoon data model."""

from platooning.models.platoon import Platoon


class TestPlatoonCreation:
    """Verify that Platoon can be instantiated and queried."""

    def test_minimal_creation(self) -> None:
        """A Platoon with only a platoon_id should be valid."""
        p = Platoon(platoon_id="p1")
        assert p.platoon_id == "p1"
        assert p.vehicle_ids == []
        assert p.leader_id == ""
        assert p.size == 0

    def test_full_creation(self) -> None:
        """A Platoon with all fields populated should preserve values."""
        p = Platoon(
            platoon_id="p1",
            vehicle_ids=["v1", "v2", "v3"],
            leader_id="v1",
            destination="junction_5",
            target_speed=30.0,
        )
        assert p.platoon_id == "p1"
        assert p.size == 3
        assert p.leader_id == "v1"
        assert p.target_speed == 30.0

    def test_contains_member(self) -> None:
        """contains() returns True for a vehicle in the platoon."""
        p = Platoon(platoon_id="p1", vehicle_ids=["v1", "v2"])
        assert p.contains("v1") is True
        assert p.contains("v2") is True

    def test_contains_non_member(self) -> None:
        """contains() returns False for a vehicle not in the platoon."""
        p = Platoon(platoon_id="p1", vehicle_ids=["v1"])
        assert p.contains("v99") is False

    def test_size_reflects_vehicle_count(self) -> None:
        """The size property matches the length of vehicle_ids."""
        p = Platoon(platoon_id="p1", vehicle_ids=["v1", "v2", "v3", "v4"])
        assert p.size == 4
