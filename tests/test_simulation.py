"""
Unit and integration tests for the SUMO simulation layer.

Unit tests do NOT require SUMO to be installed.
Integration tests are marked with ``@requires_sumo`` and are skipped
automatically when SUMO is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from platooning.models.vehicle import VehicleState
from platooning.simulation.simulator import SUMOSimulator
from platooning.simulation.traci_manager import TraCIManager
from tests.conftest import REPO_ROOT, requires_sumo

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SUMO_CFG = REPO_ROOT / "simulation" / "sumo" / "configs" / "basic_platooning.sumocfg"
_SUMO_NET = REPO_ROOT / "simulation" / "sumo" / "networks" / "basic_platooning.net.xml"
_SUMO_ROU = REPO_ROOT / "simulation" / "sumo" / "routes" / "basic_platooning.rou.xml"


# ======================================================================
# SUMO file existence (unit tests — no SUMO binary required)
# ======================================================================
class TestSUMOFiles:
    """Verify that the required SUMO simulation files exist."""

    def test_network_file_exists(self) -> None:
        assert _SUMO_NET.is_file(), f"Missing network file: {_SUMO_NET}"

    def test_route_file_exists(self) -> None:
        assert _SUMO_ROU.is_file(), f"Missing route file: {_SUMO_ROU}"

    def test_config_file_exists(self) -> None:
        assert _SUMO_CFG.is_file(), f"Missing SUMO config: {_SUMO_CFG}"


# ======================================================================
# SUMOSimulator unit tests (no SUMO binary required)
# ======================================================================
class TestSUMOSimulatorUnit:
    """Unit tests for SUMOSimulator that do NOT require a running SUMO."""

    def _make_config(self, **overrides) -> dict:
        base = {
            "simulator": "sumo",
            "config_file": "simulation/sumo/configs/basic_platooning.sumocfg",
            "step_length": 0.1,
            "end_time": 10,
            "gui": False,
            "seed": 42,
            "log_interval": 50,
        }
        base.update(overrides)
        return base

    def test_initialization(self) -> None:
        """SUMOSimulator should initialise without connecting to SUMO."""
        sim = SUMOSimulator(self._make_config(), project_root=REPO_ROOT)
        assert sim.is_running() is False
        assert sim.step_count == 0

    def test_step_without_start_raises(self) -> None:
        """Stepping before start() should raise RuntimeError."""
        sim = SUMOSimulator(self._make_config(), project_root=REPO_ROOT)
        with pytest.raises(RuntimeError, match="not running"):
            sim.step()

    def test_get_vehicle_states_when_not_running(self) -> None:
        """get_vehicle_states() should return [] when not running."""
        sim = SUMOSimulator(self._make_config(), project_root=REPO_ROOT)
        assert sim.get_vehicle_states() == []

    def test_missing_config_file_raises(self) -> None:
        """start() with a non-existent config_file should raise FileNotFoundError."""
        sim = SUMOSimulator(
            self._make_config(config_file="nonexistent.sumocfg"),
            project_root=REPO_ROOT,
        )
        with pytest.raises(FileNotFoundError):
            sim.start()

    def test_empty_config_file_raises(self) -> None:
        """start() with no config_file key should raise FileNotFoundError."""
        sim = SUMOSimulator(
            self._make_config(config_file=""),
            project_root=REPO_ROOT,
        )
        with pytest.raises(FileNotFoundError):
            sim.start()

    def test_check_availability_with_sumo_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """check_availability should return True when SUMO_HOME is valid."""
        monkeypatch.setenv("SUMO_HOME", str(tmp_path))
        sim = SUMOSimulator(self._make_config())
        assert sim.check_availability() is True

    def test_check_availability_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """check_availability should return False when SUMO is absent."""
        monkeypatch.delenv("SUMO_HOME", raising=False)
        monkeypatch.setattr("shutil.which", lambda _name: None)
        # Also block the sumolib fallback
        import builtins

        _real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "sumolib":
                raise ImportError("mocked")
            return _real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        sim = SUMOSimulator(self._make_config())
        assert sim.check_availability() is False

    def test_stop_when_not_running_is_safe(self) -> None:
        """Calling stop() before start() should not raise."""
        sim = SUMOSimulator(self._make_config(), project_root=REPO_ROOT)
        sim.stop()  # should not raise

    def test_log_interval_from_config(self) -> None:
        """log_interval property should reflect the configuration."""
        sim = SUMOSimulator(self._make_config(log_interval=100))
        assert sim.log_interval == 100


# ======================================================================
# TraCIManager unit tests
# ======================================================================
class TestTraCIManagerUnit:
    """Unit tests for TraCIManager that do NOT require a running SUMO."""

    def test_initialization(self) -> None:
        mgr = TraCIManager({"step_length": 0.1})
        assert mgr.is_connected is False

    def test_disconnect_when_not_connected(self) -> None:
        """disconnect() should be safe when not connected."""
        mgr = TraCIManager({"step_length": 0.1})
        mgr.disconnect()  # should not raise
        assert mgr.is_connected is False

    def test_build_sumo_cmd_headless(self) -> None:
        """build_sumo_cmd with gui=False should use the sumo binary."""
        mgr = TraCIManager({})
        with patch.object(TraCIManager, "_find_sumo_binary", return_value="sumo"):
            cmd = mgr.build_sumo_cmd("test.sumocfg")
        assert cmd[0] == "sumo"
        assert "-c" in cmd
        assert "test.sumocfg" in cmd

    def test_build_sumo_cmd_with_options(self) -> None:
        """build_sumo_cmd should include optional overrides."""
        mgr = TraCIManager({})
        with patch.object(TraCIManager, "_find_sumo_binary", return_value="sumo"):
            cmd = mgr.build_sumo_cmd(
                "test.sumocfg",
                step_length=0.05,
                seed=99,
                end_time=120.0,
            )
        assert "--step-length" in cmd
        assert "0.05" in cmd
        assert "--seed" in cmd
        assert "99" in cmd
        assert "--end" in cmd
        assert "120.0" in cmd


# ======================================================================
# VehicleState mapping (unit test — no SUMO)
# ======================================================================
class TestVehicleStateMapping:
    """Verify that VehicleState can be correctly populated from mock TraCI data."""

    def test_mapping_from_traci_values(self) -> None:
        """Simulate the mapping that SUMOSimulator.get_vehicle_states() performs."""
        state = VehicleState(
            vehicle_id="V1",
            timestamp=10.5,
            position_x=125.4,
            position_y=-1.6,
            speed=13.2,
            acceleration=0.5,
            lane_id="main_road_0",
            route_id="main_route",
            vehicle_type="coop_vehicle",
        )
        assert state.vehicle_id == "V1"
        assert state.timestamp == 10.5
        assert state.position_x == 125.4
        assert state.speed == 13.2
        assert state.acceleration == 0.5
        assert state.lane_id == "main_road_0"
        assert state.route_id == "main_route"
        assert state.vehicle_type == "coop_vehicle"
        assert state.is_platooned is False

    def test_timestamp_is_simulation_time(self) -> None:
        """Timestamps should be SUMO simulation time, not wall-clock."""
        state = VehicleState(vehicle_id="V1", timestamp=0.1)
        assert state.timestamp == 0.1
        state2 = VehicleState(vehicle_id="V1", timestamp=59.9)
        assert state2.timestamp == 59.9


# ======================================================================
# Integration tests (require SUMO)
# ======================================================================
@requires_sumo
class TestSUMOIntegration:
    """Integration tests that start a real SUMO simulation.

    These tests are automatically skipped when SUMO is not installed.
    """

    def _make_config(self) -> dict:
        return {
            "simulator": "sumo",
            "config_file": "simulation/sumo/configs/basic_platooning.sumocfg",
            "step_length": 0.1,
            "end_time": 30,
            "gui": False,
            "seed": 42,
            "log_interval": 0,
        }

    def test_start_step_stop(self) -> None:
        """SUMO should start, step, and stop without errors."""
        sim = SUMOSimulator(self._make_config(), project_root=REPO_ROOT)
        try:
            sim.start()
            assert sim.is_running() is True

            # Step a few times
            for _ in range(100):
                sim.step()

            assert sim.step_count == 100
            assert sim.simulation_time > 0.0
        finally:
            sim.stop()
            assert sim.is_running() is False

    def test_vehicle_states_populated(self) -> None:
        """After enough steps, vehicles should appear and have populated states."""
        sim = SUMOSimulator(self._make_config(), project_root=REPO_ROOT)
        try:
            sim.start()

            # Step until vehicles appear (V1 departs at t=0)
            for _ in range(10):
                sim.step()

            states = sim.get_vehicle_states()
            assert len(states) >= 1, "Expected at least one vehicle in simulation"

            # Verify the first vehicle's state
            v1 = states[0]
            assert isinstance(v1, VehicleState)
            assert v1.vehicle_id != ""
            assert v1.timestamp > 0.0
            assert v1.lane_id != ""
            assert v1.route_id != ""
            assert v1.vehicle_type != ""

        finally:
            sim.stop()

    def test_five_vehicles_appear(self) -> None:
        """All 5 defined vehicles should appear by t=10s (100 steps at 0.1s)."""
        sim = SUMOSimulator(self._make_config(), project_root=REPO_ROOT)
        try:
            sim.start()

            # V5 departs at t=8.0s → step to t=10.0s
            for _ in range(100):
                sim.step()

            states = sim.get_vehicle_states()
            vehicle_ids = {s.vehicle_id for s in states}
            assert (
                len(states) >= 5
            ), f"Expected 5 vehicles, got {len(states)}: {vehicle_ids}"

            # All vehicles should have valid position and speed
            for s in states:
                assert s.position_x >= 0.0
                assert s.speed >= 0.0
        finally:
            sim.stop()

    def test_vehicle_position_advances(self) -> None:
        """Vehicle positions should increase over time on a straight road."""
        sim = SUMOSimulator(self._make_config(), project_root=REPO_ROOT)
        try:
            sim.start()

            # Step to t=5.0s — V1 should have moved
            for _ in range(50):
                sim.step()

            states = sim.get_vehicle_states()
            v1_states = [s for s in states if s.vehicle_id == "V1"]
            assert len(v1_states) == 1
            pos_early = v1_states[0].position_x

            # Step more
            for _ in range(50):
                sim.step()

            states = sim.get_vehicle_states()
            v1_states = [s for s in states if s.vehicle_id == "V1"]
            if v1_states:
                # Vehicle still in simulation — position should have advanced
                assert v1_states[0].position_x > pos_early
        finally:
            sim.stop()

    def test_simulation_uses_sumo_time(self) -> None:
        """Vehicle timestamps should use SUMO simulation time."""
        sim = SUMOSimulator(self._make_config(), project_root=REPO_ROOT)
        try:
            sim.start()

            for _ in range(10):
                sim.step()

            states = sim.get_vehicle_states()
            if states:
                # Timestamp should be ~1.0s (10 steps × 0.1s)
                assert states[0].timestamp == pytest.approx(1.0, abs=0.2)
        finally:
            sim.stop()

    def test_no_orphan_sumo_process(self) -> None:
        """After stop(), the SUMO process should be terminated."""
        sim = SUMOSimulator(self._make_config(), project_root=REPO_ROOT)
        try:
            sim.start()
            for _ in range(10):
                sim.step()
        finally:
            sim.stop()

        assert sim.is_running() is False
