"""
Phase 3 Integration Test.

Runs the complete pipeline:
SUMO -> TraCI -> SUMOSimulator -> VehicleState -> V2VNetwork -> latest_known_states -> PlatoonManager -> Platoons
"""

import pytest

from platooning.communication.network_model import NetworkModel
from platooning.communication.v2v import V2VNetwork
from platooning.config.settings import load_config
from platooning.platooning.platoon_manager import PlatoonManager
from platooning.simulation.simulator import SUMOSimulator


@pytest.mark.skipif(
    not SUMOSimulator({"config_file": ""}).check_availability(),
    reason="SUMO not installed",
)
class TestPhase3Integration:
    """Verify that PlatoonManager works properly within the SUMO simulation loop."""

    def test_sumo_v2v_platoon_pipeline(self) -> None:
        # 1. Load default config which uses basic_platooning.sumocfg
        config = load_config()

        # Override for testing
        config["simulation"]["gui"] = False
        config["platooning"]["enabled"] = True
        config["platooning"]["formation_interval_ms"] = 100  # check frequently
        config["platooning"]["max_formation_distance_m"] = 200.0
        config["platooning"]["max_speed_difference_mps"] = 10.0  # lenient
        config["platooning"]["minimum_platoon_size"] = 2
        config["platooning"]["maximum_platoon_size"] = 5
        config["communication"]["stale_threshold_ms"] = 500
        config["communication"][
            "latency_ms"
        ] = 0  # instant delivery for test reliability
        config["communication"]["packet_loss_rate"] = 0.0

        simulator = SUMOSimulator(config["simulation"])
        network_model = NetworkModel.from_config(config["communication"])
        v2v = V2VNetwork(config["communication"], network_model=network_model)
        pm = PlatoonManager(config)

        simulator.start()

        platoons_formed = False

        try:
            # Run simulation for a few seconds (e.g. 100 steps = 10 seconds)
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

                pm.update_vehicle_states(latest_known, sim_time)

                if len(pm.get_all_platoons()) > 0:
                    platoons_formed = True

                    # Validate platoon constraints directly from the manager's view
                    for p in pm.get_all_platoons():
                        assert p.size >= 2
                        assert p.size <= 5
                        assert p.leader_id in p.vehicle_ids
                        assert p.route_id != ""
                        assert p.status == "FORMED"

            # We expect at least one platoon to have formed, since the baseline
            # basic_platooning.rou.xml has 5 cooperative vehicles starting close together.
            assert platoons_formed is True

        finally:
            simulator.stop()
