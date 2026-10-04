"""
Phase 4 Integration Test.

Pipeline:
SUMO -> VehicleState -> V2V -> latest_known_states -> QUBO builder -> QUBO exact solver -> decoder -> Platoon objects
"""

import pytest

from platooning.communication.network_model import NetworkModel
from platooning.communication.v2v import V2VNetwork
from platooning.config.settings import load_config
from platooning.optimization.qubo_builder import (
    build_qubo,
    decode_solution,
    solve_qubo_exact,
    validate_solution,
)
from platooning.simulation.simulator import SUMOSimulator


@pytest.mark.skipif(
    not SUMOSimulator({"config_file": ""}).check_availability(),
    reason="SUMO not installed",
)
class TestPhase4Integration:
    """Verify that QUBO formulation works properly within the SUMO simulation loop."""

    def test_sumo_v2v_qubo_pipeline(self) -> None:
        config = load_config()

        # Override for testing
        config["simulation"]["gui"] = False
        config["platooning"]["enabled"] = True
        config["platooning"]["formation_interval_ms"] = 100
        config["platooning"]["max_formation_distance_m"] = 200.0
        config["platooning"]["max_speed_difference_mps"] = 10.0
        config["platooning"]["minimum_platoon_size"] = 2
        config["platooning"]["maximum_platoon_size"] = 5
        config["communication"]["stale_threshold_ms"] = 500
        config["communication"]["latency_ms"] = 0
        config["communication"]["packet_loss_rate"] = 0.0

        simulator = SUMOSimulator(config["simulation"])
        network_model = NetworkModel.from_config(config["communication"])
        v2v = V2VNetwork(config["communication"], network_model=network_model)

        simulator.start()

        platoons_formed = False

        try:
            for _ in range(100):
                if not simulator.has_vehicles_pending():
                    break

                simulator.step()
                sim_time = simulator.simulation_time

                states = simulator.get_vehicle_states()

                for state in states:
                    v2v.broadcast(state, states, sim_time)

                v2v.deliver(sim_time)

                latest_known = {
                    s.vehicle_id: v2v.get_all_known_states(s.vehicle_id) for s in states
                }

                # Extract global perspective
                global_states = {}
                for _receiver_id, senders in latest_known.items():
                    for sender_id, state in senders.items():
                        existing = global_states.get(sender_id)
                        if existing is None or state.timestamp > existing.timestamp:
                            global_states[sender_id] = state

                valid_states = list(global_states.values())

                # N=5, max_exact_vars=25.
                # Variables: N*K + K*Slacks = 5*2 + 2*4 = 18. Exact solver handles this easily.
                if len(valid_states) > 0:
                    q_matrix, num_vars = build_qubo(valid_states, config, sim_time)
                    if num_vars > 0:
                        sol, energy = solve_qubo_exact(q_matrix, num_vars)

                        val = validate_solution(sol, valid_states, config, sim_time)
                        assert (
                            val["valid"] is True
                        ), f"QUBO produced invalid solution: {val['violations']}"

                        platoons = decode_solution(sol, valid_states, config)
                        if len(platoons) > 0:
                            platoons_formed = True
                            for p in platoons:
                                assert p.size >= 2
                                assert p.size <= 5
                                assert p.status == "FORMED"

            assert platoons_formed is True

        finally:
            simulator.stop()
