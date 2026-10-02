"""
Application entry point for the Hybrid Quantum-Classical Vehicle Platooning system.

Phase 1 — runs the SUMO simulation, extracts vehicle states via TraCI,
and logs a compact summary.

Usage::

    # Headless (default)
    python -m platooning.main

    # With SUMO-GUI for visual inspection
    python -m platooning.main --gui
"""

from __future__ import annotations

import argparse
import sys

from platooning import __version__
from platooning.config.settings import load_config, validate_config
from platooning.simulation.simulator import SUMOSimulator
from platooning.utils.logger import get_logger, setup_logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BANNER = """
==================================================
Hybrid Quantum-Classical Vehicle Platooning
v{version}
==================================================
"""


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hybrid Quantum-Classical Vehicle Platooning",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Launch SUMO-GUI instead of headless SUMO.",
    )
    return parser.parse_args()


def _format_vehicle_line(state) -> str:  # noqa: ANN001
    """Format a single VehicleState into a compact log line."""
    return (
        f"  {state.vehicle_id:<4s} | "
        f"x={state.position_x:8.2f} | "
        f"y={state.position_y:6.2f} | "
        f"speed={state.speed:5.2f} | "
        f"accel={state.acceleration:+6.2f} | "
        f"lane={state.lane_id}"
    )


def run_simulation(config: dict, gui: bool = False) -> None:
    """Run the SUMO simulation loop and log vehicle states.

    Parameters
    ----------
    config : dict
        Full project configuration.
    gui : bool
        Whether to launch SUMO-GUI.
    """
    logger = get_logger(__name__)

    sim_config = dict(config["simulation"])
    if gui:
        sim_config["gui"] = True

    simulator = SUMOSimulator(sim_config)

    # Check availability first
    if not simulator.check_availability():
        logger.error(
            "SUMO integration unavailable — SUMO installation required.\n"
            "See docs/development_setup.md for installation instructions."
        )
        return

    logger.info("Starting SUMO simulation...")

    try:
        simulator.start()

        log_interval = simulator.log_interval
        while simulator.has_vehicles_pending():
            simulator.step()

            # Periodic compact logging
            if log_interval > 0 and simulator.step_count % log_interval == 0:
                states = simulator.get_vehicle_states()
                sim_t = simulator.simulation_time
                logger.info("[SIM] t=%.1fs | vehicles=%d", sim_t, len(states))
                for s in states:
                    logger.info(_format_vehicle_line(s))

        # Final state snapshot
        states = simulator.get_vehicle_states()
        logger.info(
            "Simulation completed. Total steps: %d | Final time: %.1fs",
            simulator.step_count,
            simulator.simulation_time,
        )

    except FileNotFoundError as exc:
        logger.error("Simulation file error: %s", exc)
    except RuntimeError as exc:
        logger.error("Simulation runtime error: %s", exc)
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")
    finally:
        simulator.stop()


def main() -> None:
    """Application entry point."""
    args = _parse_args()
    print(_BANNER.format(version=__version__))

    # 1. Load configuration ------------------------------------------------
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = get_logger(__name__)
    logger.info("Configuration loaded successfully.")

    # 2. Validate configuration --------------------------------------------
    errors = validate_config(config)
    if errors:
        for err in errors:
            logger.error("Config validation error: %s", err)
        sys.exit(1)
    logger.info("Configuration validated — no errors.")

    # 3. Report simulation backend -----------------------------------------
    sim_backend = config["simulation"]["simulator"]
    logger.info("Simulation backend: %s", sim_backend.upper())

    # 4. Check SUMO and run simulation -------------------------------------
    run_simulation(config, gui=args.gui)


if __name__ == "__main__":
    main()
