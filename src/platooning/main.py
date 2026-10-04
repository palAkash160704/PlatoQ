"""
Application entry point for the Hybrid Quantum-Classical Vehicle Platooning system.

Phase 2 — runs the SUMO simulation, extracts vehicle states via TraCI,
broadcasts V2V messages, and logs compact summaries of both simulation
and communication activity.

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
from platooning.communication.network_model import NetworkModel
from platooning.communication.v2v import V2VNetwork
from platooning.config.settings import load_config, validate_config
from platooning.platooning.dynamic_manager import DynamicPlatoonManager
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
    """Run the SUMO simulation loop with V2V communication.

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

    # --- V2V setup --------------------------------------------------------
    comm_config = config.get("communication", {})
    seed = config.get("simulation", {}).get("seed", 42)
    network_model = NetworkModel.from_config(comm_config)
    v2v = V2VNetwork(comm_config, seed=seed, network_model=network_model)

    logger.info("Starting SUMO simulation...")
    logger.info(
        "V2V: range=%dm  latency=%dms  loss=%.1f%%  freq=%dHz  stale=%dms",
        int(network_model.range_meters),
        int(network_model.base_latency_ms),
        network_model.packet_loss_rate * 100,
        int(network_model.message_frequency_hz),
        int(network_model.stale_threshold_ms),
    )

    # --- Platoon Manager setup --------------------------------------------
    opt_method = config.get("optimization", {}).get("method", "classical")
    pm = DynamicPlatoonManager(config, optimizer_mode=opt_method)

    try:
        simulator.start()

        log_interval = simulator.log_interval
        while simulator.has_vehicles_pending():
            simulator.step()
            sim_time = simulator.simulation_time

            # --- V2V communication ---
            states = simulator.get_vehicle_states()

            # Each vehicle broadcasts its state
            for state in states:
                v2v.broadcast(state, states, sim_time)

            # Deliver messages whose delivery_time has arrived
            v2v.deliver(sim_time)

            # --- Platoon Management ---
            # All receivers combined perspective
            latest_known = {
                s.vehicle_id: v2v.get_all_known_states(s.vehicle_id) for s in states
            }
            pm.update_vehicle_states(latest_known, sim_time)

            # --- Periodic logging ---
            if log_interval > 0 and simulator.step_count % log_interval == 0:
                logger.info("[SIM] t=%.1fs | vehicles=%d", sim_time, len(states))
                for s in states:
                    logger.info(_format_vehicle_line(s))

                # V2V statistics
                stats = v2v.get_statistics()
                stale_count = v2v.count_stale_states(sim_time)
                if stats.messages_sent > 0:
                    logger.info(
                        "[V2V] sent=%d | delivered=%d | dropped=%d",
                        stats.messages_sent,
                        stats.messages_delivered,
                        stats.messages_dropped,
                    )
                    logger.info(
                        "[V2V] delivery_rate=%.1f%% | avg_latency=%.1fms | stale=%d",
                        stats.delivery_rate * 100,
                        stats.average_latency_ms,
                        stale_count,
                    )

        # Final statistics
        stats = v2v.get_statistics()
        logger.info(
            "Simulation completed. Total steps: %d | Final time: %.1fs",
            simulator.step_count,
            simulator.simulation_time,
        )
        logger.info(
            "[V2V FINAL] sent=%d delivered=%d dropped=%d "
            "delivery_rate=%.1f%% avg_latency=%.1fms",
            stats.messages_sent,
            stats.messages_delivered,
            stats.messages_dropped,
            stats.delivery_rate * 100,
            stats.average_latency_ms,
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
