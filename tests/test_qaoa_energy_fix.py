import pytest
from platooning.optimization.quantum.decoder import select_best_feasible, build_variable_mapping
from platooning.optimization.qubo_builder import build_qubo
from platooning.models.vehicle import VehicleState

def test_qaoa_energy_discrepancy_resolved():
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

    states = [
        VehicleState('V1', 0.0, 100.0, 0.0, 30.0, 'r1', 'l1'),
        VehicleState('V2', 0.0, 95.0, 0.0, 30.0, 'r1', 'l1'),
        VehicleState('V3', 0.0, 90.0, 0.0, 30.0, 'r1', 'l1'),
    ]
    
    q_matrix, num_vars = build_qubo(states, config, 0.0)
    raw_bits = [0] * num_vars
    x_indices, s_indices, _ = build_variable_mapping(3, config)
    
    raw_bits[x_indices[(0, 0)]] = 1
    raw_bits[x_indices[(1, 0)]] = 1
    raw_bits[x_indices[(2, 0)]] = 1
    
    raw_energy = 0.0
    for (i, j), coef in q_matrix.items():
        if raw_bits[i] == 1 and raw_bits[j] == 1:
            raw_energy += coef
            
    assert raw_energy > 500.0
    
    counts = {''.join(str(b) for b in reversed(raw_bits)): 100}
    best_bits, best_energy, feasible = select_best_feasible(counts, q_matrix, states, config, 0.0)
    
    assert feasible is True
    assert best_energy < 500.0
    
    from platooning.optimization.qubo_builder import solve_qubo_exact
    exact_bits, exact_energy = solve_qubo_exact(q_matrix, num_vars)
    
    assert abs(best_energy - exact_energy) < 1e-5
