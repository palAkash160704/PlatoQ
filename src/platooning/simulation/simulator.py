"""
Simulator abstraction and SUMO implementation.

Defines a ``BaseSimulator`` interface and a concrete ``SUMOSimulator`` that
connects to SUMO via TraCI for live vehicle state extraction.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from platooning.models.vehicle import VehicleState
from platooning.simulation.traci_manager import TraCIManager
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


class BaseSimulator(ABC):
    """Abstract interface for a traffic simulator backend.

    Any new simulator backend (e.g. CARLA in a future extension) should
    sub-class this interface so that the rest of the system remains
    simulator-agnostic.
    """

    @abstractmethod
    def start(self) -> None:
        """Launch the simulation."""

    @abstractmethod
    def stop(self) -> None:
        """Shut down the simulation cleanly."""

    @abstractmethod
    def step(self) -> None:
        """Advance the simulation by one time-step."""

    @abstractmethod
    def get_vehicle_states(self) -> list[VehicleState]:
        """Return the current state of every vehicle in the simulation."""

    @abstractmethod
    def is_running(self) -> bool:
        """Return *True* if the simulation is currently active."""


class SUMOSimulator(BaseSimulator):
    """SUMO traffic simulator backend (TraCI).

    Parameters
    ----------
    config : dict
        The ``simulation`` section of the project configuration.
    project_root : str or Path, optional
        Repository root used to resolve relative ``config_file`` paths.
        Defaults to the repo root detected from the package layout.
    """

    def __init__(
        self,
        config: dict[str, Any],
        project_root: str | Path | None = None,
    ) -> None:
        self._config = config
        self._running = False
        self._step_length: float = config.get("step_length", 0.1)
        self._seed: int = config.get("seed", 42)
        self._end_time: float = config.get("end_time", 60)
        self._gui: bool = config.get("gui", False)
        self._log_interval: int = config.get("log_interval", 50)

        # Resolve project root for relative paths
        if project_root is not None:
            self._project_root = Path(project_root).resolve()
        else:
            # Default: 3 levels up from src/platooning/simulation/simulator.py
            self._project_root = Path(__file__).resolve().parents[3]

        self._traci = TraCIManager(config)
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # BaseSimulator interface
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Launch SUMO via TraCI.

        Raises
        ------
        FileNotFoundError
            If the SUMO binary or configuration file cannot be found.
        RuntimeError
            If the TraCI connection fails.
        """
        config_file_rel = self._config.get("config_file", "")
        if not config_file_rel:
            raise FileNotFoundError(
                "No simulation.config_file specified in configuration."
            )

        config_file = self._project_root / config_file_rel
        if not config_file.is_file():
            raise FileNotFoundError(f"SUMO config file not found: {config_file}")

        sumo_cmd = self._traci.build_sumo_cmd(
            str(config_file),
            gui=self._gui,
            step_length=self._step_length,
            seed=self._seed,
            end_time=self._end_time,
        )

        self._traci.connect(sumo_cmd)
        self._running = True
        self._step_count = 0
        logger.info("SUMO simulation started (gui=%s).", self._gui)

    def stop(self) -> None:
        """Close the TraCI connection and shut down SUMO."""
        if not self._running and not self._traci.is_connected:
            return
        try:
            self._traci.disconnect()
        finally:
            self._running = False
            logger.info("SUMO simulation stopped after %d steps.", self._step_count)

    def step(self) -> None:
        """Advance SUMO by one simulation step.

        Raises
        ------
        RuntimeError
            If the simulation is not running.
        """
        if not self._running:
            raise RuntimeError("Cannot step — simulation is not running.")
        self._traci.step()
        self._step_count += 1

    def get_vehicle_states(self) -> list[VehicleState]:
        """Extract every active vehicle's state from SUMO via TraCI.

        Returns
        -------
        list[VehicleState]
            One ``VehicleState`` per vehicle currently in the simulation.
        """
        if not self._running:
            return []

        sim_time = self._traci.get_simulation_time()
        vehicle_ids = self._traci.get_vehicle_ids()
        states: list[VehicleState] = []

        for vid in vehicle_ids:
            try:
                pos = self._traci.get_vehicle_position(vid)
                state = VehicleState(
                    vehicle_id=vid,
                    timestamp=sim_time,
                    position_x=pos[0],
                    position_y=pos[1],
                    speed=self._traci.get_vehicle_speed(vid),
                    acceleration=self._traci.get_vehicle_acceleration(vid),
                    lane_id=self._traci.get_vehicle_lane_id(vid),
                    route_id=self._traci.get_vehicle_route_id(vid),
                    vehicle_type=self._traci.get_vehicle_type(vid),
                )
                states.append(state)
            except Exception as exc:
                logger.warning("Failed to extract state for vehicle '%s': %s", vid, exc)

        return states

    def is_running(self) -> bool:
        """Return whether the SUMO simulation is currently active."""
        return self._running

    def has_vehicles_pending(self) -> bool:
        """Return *True* if more vehicles are expected to enter the simulation.

        This combines the count of currently active vehicles and vehicles
        that are scheduled to depart in the future.
        """
        if not self._running:
            return False
        return self._traci.get_min_expected_vehicles() > 0

    @property
    def simulation_time(self) -> float:
        """Return the current SUMO simulation time (seconds)."""
        if not self._running:
            return 0.0
        return self._traci.get_simulation_time()

    @property
    def step_count(self) -> int:
        """Return the number of simulation steps executed so far."""
        return self._step_count

    @property
    def log_interval(self) -> int:
        """Return how often (in steps) vehicle states should be logged."""
        return self._log_interval

    # ------------------------------------------------------------------
    # SUMO availability check
    # ------------------------------------------------------------------
    def check_availability(self) -> bool:
        """Check whether SUMO is reachable on this machine.

        Detection order:
        1. ``SUMO_HOME`` environment variable.
        2. ``sumo`` or ``sumo-gui`` on the system ``PATH``.

        Returns
        -------
        bool
            *True* if SUMO was detected.
        """
        # Check SUMO_HOME
        sumo_home = os.environ.get("SUMO_HOME", "")
        if sumo_home and os.path.isdir(sumo_home):
            logger.debug("SUMO detected via SUMO_HOME: %s", sumo_home)
            return True

        # Fallback: check PATH for the sumo binary
        if shutil.which("sumo") or shutil.which("sumo-gui"):
            logger.debug("SUMO detected on system PATH.")
            return True

        # Fallback: sumolib.checkBinary (pip-installed eclipse-sumo)
        try:
            import sumolib  # type: ignore[import-untyped]

            sumolib.checkBinary("sumo")
            logger.debug("SUMO detected via sumolib.")
            return True
        except Exception:
            pass

        logger.warning("SUMO not detected. Set SUMO_HOME or add sumo to PATH.")
        return False
