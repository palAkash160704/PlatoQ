"""
TraCI connection manager.

Encapsulates the low-level TraCI connection life-cycle so that the rest
of the simulation layer can remain decoupled from TraCI internals.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from platooning.utils.logger import get_logger

logger = get_logger(__name__)


class TraCIManager:
    """Manage the TraCI connection to a running SUMO instance.

    Parameters
    ----------
    config : dict
        The ``simulation`` section of the project configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._connected = False

    # ------------------------------------------------------------------
    # SUMO binary resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _find_sumo_binary(gui: bool = False) -> str:
        """Locate the SUMO binary.

        Resolution order:
        1. ``SUMO_HOME/bin/sumo`` (or ``sumo-gui``).
        2. ``sumo`` (or ``sumo-gui``) on the system ``PATH``.

        Parameters
        ----------
        gui : bool
            If *True*, look for ``sumo-gui`` instead of ``sumo``.

        Returns
        -------
        str
            Absolute path (or bare command name) of the SUMO binary.

        Raises
        ------
        FileNotFoundError
            If no SUMO binary can be found.
        """
        binary_name = "sumo-gui" if gui else "sumo"

        # 1. SUMO_HOME
        sumo_home = os.environ.get("SUMO_HOME", "")
        if sumo_home:
            candidate = Path(sumo_home) / "bin" / binary_name
            # On Windows the executable may have a .exe suffix
            for suffix in ("", ".exe"):
                full = candidate.with_suffix(suffix)
                if full.is_file():
                    logger.debug("SUMO binary found via SUMO_HOME: %s", full)
                    return str(full)

        # 2. System PATH
        found = shutil.which(binary_name)
        if found:
            logger.debug("SUMO binary found on PATH: %s", found)
            return found

        # 3. sumolib.checkBinary (pip-installed eclipse-sumo)
        try:
            import sumolib  # type: ignore[import-untyped]

            found = sumolib.checkBinary(binary_name)
            if found:
                logger.debug("SUMO binary found via sumolib: %s", found)
                return found
        except Exception:
            pass

        raise FileNotFoundError(
            f"SUMO binary '{binary_name}' not found. "
            "Set SUMO_HOME or add the SUMO bin directory to PATH."
        )

    # ------------------------------------------------------------------
    # Connection life-cycle
    # ------------------------------------------------------------------
    def connect(self, sumo_cmd: list[str]) -> None:
        """Open a TraCI connection to SUMO.

        Parameters
        ----------
        sumo_cmd : list[str]
            Command-line arguments to start the SUMO process
            (e.g. ``["sumo", "-c", "sim.sumocfg"]``).

        Raises
        ------
        RuntimeError
            If the TraCI connection fails.
        """
        try:
            import traci  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'traci' package is not installed. "
                "Install SUMO or run: pip install traci"
            ) from exc

        logger.info("Starting SUMO with command: %s", " ".join(sumo_cmd))
        try:
            traci.start(sumo_cmd)
            self._connected = True
            logger.info("TraCI connection established.")
        except Exception as exc:
            self._connected = False
            # Force-close any leaked TraCI state
            import contextlib

            with contextlib.suppress(Exception):
                traci.close()
            raise RuntimeError(f"Failed to start SUMO / TraCI: {exc}") from exc

    def disconnect(self) -> None:
        """Close the TraCI connection and terminate the SUMO process."""
        try:
            import traci  # type: ignore[import-untyped]

            traci.close()
            logger.info("TraCI connection closed.")
        except Exception as exc:
            # May fail if connection was never established or already closed
            if self._connected:
                logger.warning("Error during TraCI disconnect: %s", exc)
        finally:
            self._connected = False

    def step(self) -> None:
        """Advance the SUMO simulation by one time-step."""
        import traci  # type: ignore[import-untyped]

        traci.simulationStep()

    def get_simulation_time(self) -> float:
        """Return the current SUMO simulation time in seconds."""
        import traci  # type: ignore[import-untyped]

        return traci.simulation.getTime()

    def get_vehicle_ids(self) -> list[str]:
        """Return the IDs of all vehicles currently in the simulation."""
        import traci  # type: ignore[import-untyped]

        return list(traci.vehicle.getIDList())

    def get_vehicle_position(self, vehicle_id: str) -> tuple[float, float]:
        """Return the (x, y) position of a vehicle."""
        import traci  # type: ignore[import-untyped]

        return traci.vehicle.getPosition(vehicle_id)

    def get_vehicle_speed(self, vehicle_id: str) -> float:
        """Return the current speed of a vehicle (m/s)."""
        import traci  # type: ignore[import-untyped]

        return traci.vehicle.getSpeed(vehicle_id)

    def get_vehicle_acceleration(self, vehicle_id: str) -> float:
        """Return the current acceleration of a vehicle (m/s²)."""
        import traci  # type: ignore[import-untyped]

        return traci.vehicle.getAcceleration(vehicle_id)

    def get_vehicle_lane_id(self, vehicle_id: str) -> str:
        """Return the lane ID the vehicle is currently on."""
        import traci  # type: ignore[import-untyped]

        return traci.vehicle.getLaneID(vehicle_id)

    def get_vehicle_route_id(self, vehicle_id: str) -> str:
        """Return the route ID assigned to the vehicle."""
        import traci  # type: ignore[import-untyped]

        return traci.vehicle.getRouteID(vehicle_id)

    def get_vehicle_type(self, vehicle_id: str) -> str:
        """Return the vehicle type ID."""
        import traci  # type: ignore[import-untyped]

        return traci.vehicle.getTypeID(vehicle_id)

    def get_min_expected_vehicles(self) -> int:
        """Return the minimum number of vehicles expected to arrive."""
        import traci  # type: ignore[import-untyped]

        return traci.simulation.getMinExpectedNumber()

    @property
    def is_connected(self) -> bool:
        """Return whether a TraCI connection is currently active."""
        return self._connected

    # ------------------------------------------------------------------
    # SUMO command construction helper
    # ------------------------------------------------------------------
    def build_sumo_cmd(
        self,
        config_file: str,
        *,
        gui: bool = False,
        step_length: float | None = None,
        seed: int | None = None,
        end_time: float | None = None,
    ) -> list[str]:
        """Build a SUMO command-line invocation.

        Parameters
        ----------
        config_file : str
            Path to the ``.sumocfg`` file.
        gui : bool
            Use ``sumo-gui`` instead of ``sumo``.
        step_length : float, optional
            Override the step-length defined in the ``.sumocfg``.
        seed : int, optional
            Override the random seed.
        end_time : float, optional
            Override the simulation end time.

        Returns
        -------
        list[str]
            The complete command as a list of strings.
        """
        binary = self._find_sumo_binary(gui=gui)
        cmd = [binary, "-c", config_file]

        if step_length is not None:
            cmd.extend(["--step-length", str(step_length)])
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        if end_time is not None:
            cmd.extend(["--end", str(end_time)])

        return cmd
