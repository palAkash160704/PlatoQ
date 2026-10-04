from platooning.models.vehicle import VehicleState
from platooning.platooning.dynamic_manager import DynamicPlatoonManager


def test_qaoa_fallback_to_classical(monkeypatch):
    config = {
        "communication": {"stale_threshold_ms": 200.0},
        "platooning": {
            "enabled": True,
            "max_formation_distance_m": 50.0,
            "max_speed_difference_mps": 3.0,
            "minimum_platoon_size": 2,
            "maximum_platoon_size": 5,
        },
        "dynamic": {
            "reconfiguration_interval_ms": 2000,
            "reconfiguration_cooldown_ms": 1000,
            "minimum_improvement": 0.5,
        },
        "qubo": {"enabled": True, "solver": "exact"},
        "quantum": {"backend": "aer_simulator", "shots": 1, "reps": 1},
    }

    # Mock run_qaoa to return an empty or infeasible result
    def mock_run_qaoa(*args, **kwargs):
        from platooning.optimization.quantum.qaoa_solver import QAOAResult

        return None, QAOAResult(
            best_bitstring=[1] * 10,
            best_energy=1000.0,
            counts={"1111111111": 100},  # garbage bitstring
            optimal_parameters=[0.1, 0.2],
            runtime_seconds=0.1,
        )

    monkeypatch.setattr(
        "platooning.optimization.quantum.comparison.run_qaoa", mock_run_qaoa
    )

    dm = DynamicPlatoonManager(config, optimizer_mode="qaoa")
    states = [
        VehicleState("V1", 0.0, 100.0, 0.0, 30.0, "r1", "l1"),
        VehicleState("V2", 0.0, 95.0, 0.0, 30.0, "r1", "l1"),
    ]
    latest_known = {s.vehicle_id: {s.vehicle_id: s} for s in states}

    # Should not crash, should fall back to classical which forms 1 platoon
    dm.update_vehicle_states(latest_known, 0.0)

    assert len(dm.platoons) == 1
    assert dm.platoons[0].size == 2
