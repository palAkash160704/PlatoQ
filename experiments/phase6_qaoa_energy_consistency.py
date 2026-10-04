import sys
from platooning.optimization.quantum.decoder import select_best_feasible, build_variable_mapping
from platooning.optimization.qubo_builder import build_qubo, solve_qubo_exact
from platooning.models.vehicle import VehicleState

def main():
    config = {
        'communication': {'stale_threshold_ms': 200.0},
        'platooning': {
            'enabled': True,
            'max_formation_distance_m': 50.0,
            'max_speed_difference_mps': 3.0,
            'minimum_platoon_size': 2,
            'maximum_platoon_size': 5,
            'weight_membership': 10.0,
            'weight_distance_penalty': 0.5,
            'weight_speed_penalty': 1.0,
            'weight_ungrouped_penalty': 5.0,
        },
        'qubo': {
            'enabled': True,
            'solver': 'exact',
            'weight_pairwise_distance': 0.1,
            'weight_pairwise_speed': 0.5,
            'penalty_assignment': 1000.0,
            'penalty_incompatibility': 1000.0,
            'penalty_size': 1000.0,
        },
    }

    print("============================================================")
    print("PHASE 6.2 - QAOA ENERGY CONSISTENCY EXPERIMENT")
    print("============================================================\n")

    scenarios = [
        ("Config A (2 Vehicles)", [
            VehicleState('V1', 0.0, 100.0, 0.0, 30.0, 'r1', 'l1'),
            VehicleState('V2', 0.0, 95.0, 0.0, 30.0, 'r1', 'l1'),
        ]),
        ("Config B (3 Vehicles)", [
            VehicleState('V1', 0.0, 100.0, 0.0, 30.0, 'r1', 'l1'),
            VehicleState('V2', 0.0, 95.0, 0.0, 30.0, 'r1', 'l1'),
            VehicleState('V3', 0.0, 90.0, 0.0, 30.0, 'r1', 'l1'),
        ]),
        ("Config C (4 Vehicles)", [
            VehicleState('V1', 0.0, 100.0, 0.0, 30.0, 'r1', 'l1'),
            VehicleState('V2', 0.0, 95.0, 0.0, 30.0, 'r1', 'l1'),
            VehicleState('V3', 0.0, 90.0, 0.0, 30.0, 'r1', 'l1'),
            VehicleState('V4', 0.0, 85.0, 0.0, 30.0, 'r1', 'l1'),
        ])
    ]

    for name, states in scenarios:
        print(f"Testing {name}:")
        q_matrix, num_vars = build_qubo(states, config, 0.0)
        
        # 1. Exact QUBO Solver
        exact_bits, exact_energy = solve_qubo_exact(q_matrix, num_vars)
        
        # 2. Feed the optimal bits from Exact QUBO into QAOA decoding
        counts = {''.join(str(b) for b in reversed(exact_bits)): 100}
        
        # select_best_feasible should evaluate the true energy of this bitstring
        best_bits, qaoa_repaired_energy, feasible = select_best_feasible(counts, q_matrix, states, config, 0.0)
        
        diff = abs(exact_energy - qaoa_repaired_energy)
        
        print(f"  Exact QUBO Energy   : {exact_energy:.4f}")
        print(f"  QAOA Decoded Energy : {qaoa_repaired_energy:.4f}")
        print(f"  Difference          : {diff:.4f}")
        print(f"  Feasible            : {feasible}")
        print()

if __name__ == "__main__":
    main()
