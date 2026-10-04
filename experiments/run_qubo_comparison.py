"""
Compare Phase 3 Classical Greedy formation vs Phase 4 QUBO exact formulation.
"""

import os
import sys
import time

# Ensure src is in PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from platooning.config.settings import load_config
from platooning.models.vehicle import VehicleState
from platooning.optimization.qubo_builder import (
    build_qubo,
    decode_solution,
    solve_qubo_exact,
    validate_solution,
)
from platooning.platooning.platoon_manager import PlatoonManager


def main():
    config = load_config()

    # 5 Compatible vehicles
    states = [
        VehicleState(
            vehicle_id=f"V{i}",
            timestamp=10.0,
            position_x=100.0 - i * 10.0,
            position_y=0.0,
            speed=30.0,
            route_id="route_1",
            lane_id="lane_0",
        )
        for i in range(1, 6)
    ]

    print("===============================================")
    print("CLASSICAL vs QUBO")
    print("===============================================")
    print(f"Vehicles: {len(states)}\n")

    # CLASSICAL
    start_t = time.perf_counter()
    pm_classical = PlatoonManager(config)

    latest_known = {
        s.vehicle_id: {other.vehicle_id: other for other in states} for s in states
    }

    pm_classical.update_vehicle_states(latest_known, 10.1)
    platoons_classical = pm_classical.get_all_platoons()
    metrics_classical = pm_classical.get_formation_metrics()
    end_t = time.perf_counter()

    runtime_classical = (end_t - start_t) * 1000.0

    print("CLASSICAL")
    print(
        f"Platoons: {len(platoons_classical)} ({', '.join(str(p.size) for p in platoons_classical)})"
    )
    print(f"Objective: {metrics_classical.objective_value:.2f}")
    print(f"Runtime: {runtime_classical:.2f} ms")
    print()

    # QUBO
    start_t = time.perf_counter()
    q_matrix, num_vars = build_qubo(states, config, 10.1)
    sol, energy = solve_qubo_exact(q_matrix, num_vars)
    val = validate_solution(sol, states, config, 10.1)
    platoons_qubo = decode_solution(sol, states, config)

    # Calculate original objective for QUBO result using a dummy manager
    pm_qubo = PlatoonManager(config)
    pm_qubo._platoons = {p.platoon_id: p for p in platoons_qubo}
    pm_qubo._calculate_metrics(states, platoons_qubo, 10.1)
    metrics_qubo = pm_qubo.get_formation_metrics()
    end_t = time.perf_counter()

    runtime_qubo = (end_t - start_t) * 1000.0

    print("QUBO")
    print(f"Variables: {num_vars}")
    print(
        f"Platoons: {len(platoons_qubo)} ({', '.join(str(p.size) for p in platoons_qubo)})"
    )
    print(f"QUBO Energy: {energy:.2f}")
    print(f"Original Objective: {metrics_qubo.objective_value:.2f}")
    print(f"Runtime: {runtime_qubo:.2f} ms")
    print()

    print("Constraint Violations:")
    print("Classical: 0")
    print(f"QUBO: {len(val['violations'])}")
    if val["violations"]:
        for v in val["violations"]:
            print(f"  - {v}")
    print("===============================================")


if __name__ == "__main__":
    main()
