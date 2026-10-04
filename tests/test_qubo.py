"""
Tests for Phase 4 QUBO formulation.
"""

import math

import pytest

from platooning.models.vehicle import VehicleState
from platooning.optimization.qubo_builder import (
    build_qubo,
    decode_solution,
    evaluate_energy,
    solve_qubo_exact,
    validate_solution,
)
from platooning.platooning.formation import form_platoons


@pytest.fixture
def config():
    return {
        "communication": {"stale_threshold_ms": 500.0},
        "platooning": {
            "enabled": True,
            "formation_interval_ms": 100,
            "max_formation_distance_m": 500.0,
            "max_speed_difference_mps": 10.0,
            "minimum_platoon_size": 2,
            "maximum_platoon_size": 5,
            "weight_membership": 10.0,
            "weight_distance_penalty": 0.5,
            "weight_speed_penalty": 1.0,
            "weight_ungrouped_penalty": 5.0,
        },
        "qubo": {
            "enabled": True,
            "solver": "exact",
            "max_exact_variables": 25,
            "weight_pairwise_distance": 0.1,
            "weight_pairwise_speed": 0.5,
            "penalty_assignment": 1000.0,
            "penalty_incompatibility": 1000.0,
            "penalty_size": 1000.0,
        },
    }


def create_states(
    n: int,
    start_x: float = 100.0,
    speed: float = 30.0,
    route: str = "r1",
    lane: str = "l1",
) -> list[VehicleState]:
    return [
        VehicleState(
            vehicle_id=f"v{i}",
            timestamp=10.0,
            position_x=start_x - i * 5.0,
            position_y=0.0,
            speed=speed,
            route_id=route,
            lane_id=lane,
        )
        for i in range(1, n + 1)
    ]


def test_qubo_variable_mapping(config):
    """TEST 1: Variable mapping is deterministic."""
    states = create_states(2)
    q_matrix, num_vars = build_qubo(states, config, 10.1)

    # N=2. x vars = 3. Slacks = 1. Total = 4.
    assert num_vars == 4

    # Check that it produces consistent output
    q_matrix2, num_vars2 = build_qubo(states, config, 10.1)
    assert q_matrix == q_matrix2


def test_qubo_expected_variables(config):
    """TEST 2: QUBO contains expected variables."""
    states = create_states(3)
    q_matrix, num_vars = build_qubo(states, config, 10.1)
    # N=3. x vars = 6. Slacks = 3. Total = 9.
    assert num_vars == 9


def test_qubo_single_vehicle(config):
    """TEST 3: Single vehicle."""
    states = create_states(1)
    q_matrix, num_vars = build_qubo(states, config, 10.1)
    # N=1 -> 1 var.
    assert num_vars == 1


def test_qubo_two_compatible(config):
    """TEST 4: Two compatible vehicles."""
    states = create_states(2)
    q_matrix, num_vars = build_qubo(states, config, 10.1)

    solution, energy = solve_qubo_exact(q_matrix, num_vars)
    val = validate_solution(solution, states, config, 10.1)
    assert val["valid"] is True

    platoons = decode_solution(solution, states, config)
    assert len(platoons) == 1
    assert platoons[0].size == 2


def test_qubo_two_incompatible(config):
    """TEST 5: Two incompatible vehicles."""
    states = create_states(2)
    # Make them incompatible by route
    states[1].route_id = "r2"

    q_matrix, num_vars = build_qubo(states, config, 10.1)
    solution, energy = solve_qubo_exact(q_matrix, num_vars)

    # Should be valid (meaning they are ungrouped, or size 1 penalty prevents formation)
    val = validate_solution(solution, states, config, 10.1)
    assert val["valid"] is True

    platoons = decode_solution(solution, states, config)
    # No platoon should form
    assert len(platoons) == 0


def test_qubo_three_compatible(config):
    """TEST 6: Three compatible vehicles."""
    states = create_states(3)
    q_matrix, num_vars = build_qubo(states, config, 10.1)
    solution, energy = solve_qubo_exact(q_matrix, num_vars)

    platoons = decode_solution(solution, states, config)
    assert len(platoons) == 1
    assert platoons[0].size == 3


def test_qubo_six_compatible_max_five(config):
    """TEST 7: Six compatible vehicles with maximum size = 5."""
    states = create_states(6)
    q_matrix, num_vars = build_qubo(states, config, 10.1)

    # N=6. x vars = 21. Slacks = 14. Total = 35.
    # Vars = 35.
    # 30 is > max_exact_variables (25)
    with pytest.raises(ValueError):
        solve_qubo_exact(q_matrix, num_vars, max_vars=25)


def test_qubo_different_lanes(config):
    """TEST 8: Different lanes."""
    states = create_states(2)
    states[1].lane_id = "l2"
    q_matrix, num_vars = build_qubo(states, config, 10.1)
    solution, _ = solve_qubo_exact(q_matrix, num_vars)
    assert len(decode_solution(solution, states, config)) == 0


def test_qubo_different_routes(config):
    """TEST 9: Different routes."""
    states = create_states(2)
    states[1].route_id = "r2"
    q_matrix, num_vars = build_qubo(states, config, 10.1)
    solution, _ = solve_qubo_exact(q_matrix, num_vars)
    assert len(decode_solution(solution, states, config)) == 0


def test_qubo_stale_state(config):
    """TEST 10: Stale state."""
    states = create_states(2)
    states[1].timestamp = 0.0  # very stale
    q_matrix, num_vars = build_qubo(states, config, 10.1)
    solution, _ = solve_qubo_exact(q_matrix, num_vars)
    assert len(decode_solution(solution, states, config)) == 0


def test_qubo_ungrouped_vehicle(config):
    """TEST 11: Ungrouped vehicle."""
    states = create_states(3)
    states[2].route_id = "r2"  # v3 is alone
    q_matrix, num_vars = build_qubo(states, config, 10.1)
    solution, _ = solve_qubo_exact(q_matrix, num_vars)

    platoons = decode_solution(solution, states, config)
    assert len(platoons) == 1
    assert platoons[0].size == 2
    assert "v3" not in platoons[0].vehicle_ids


def test_qubo_invalid_binary_solution(config):
    """TEST 12: Invalid binary solution."""
    states = create_states(2)
    q_matrix, num_vars = build_qubo(states, config, 10.1)

    # Create an invalid assignment
    invalid_sol = [1] * num_vars
    val = validate_solution(invalid_sol, states, config, 10.1)
    assert val["valid"] is False
    assert any("size" in v for v in val["violations"])


def test_qubo_energy_calculation(config):
    """TEST 13: QUBO energy calculation."""
    states = create_states(2)
    q_matrix, num_vars = build_qubo(states, config, 10.1)

    # N=2. Variables: x_{0,0}, x_{1,0}, x_{1,1}, s_{0,2}
    # valid solution: x_{0,0}=1, x_{1,0}=1, x_{1,1}=0, s_{0,2}=1
    sol = [1, 1, 0, 1]
    energy = evaluate_energy(q_matrix, sol)

    w_mem = 10.0
    w_ung = 5.0

    dist = 5.0
    w_d = 0.1
    w_s = 0.5

    expected_linear = -2 * (w_mem + w_ung)
    expected_pair = 1 * (w_d * dist + w_s * 0.0)

    assert math.isclose(energy, expected_linear + expected_pair)


def test_qubo_exact_solver(config):
    """TEST 14: Exact solver finds minimum energy."""
    states = create_states(2)
    q_matrix, num_vars = build_qubo(states, config, 10.1)
    solution, best_energy = solve_qubo_exact(q_matrix, num_vars)

    assert best_energy < 0  # Because of membership reward
    assert solution == [1, 1, 0, 1]


def test_qubo_determinism(config):
    """TEST 15: Determinism."""
    states = create_states(3)
    q_matrix1, num_vars1 = build_qubo(states, config, 10.1)
    sol1, energy1 = solve_qubo_exact(q_matrix1, num_vars1)

    q_matrix2, num_vars2 = build_qubo(states, config, 10.1)
    sol2, energy2 = solve_qubo_exact(q_matrix2, num_vars2)

    assert sol1 == sol2
    assert energy1 == energy2


def test_phase3_cross_validation(config):
    """TEST 16: Phase 3 vs QUBO cross-validation."""
    states = create_states(4)
    states[3].route_id = "diff"

    # Phase 3 classical greedy
    platoons_classical = form_platoons(states, config, 10.1)

    # Phase 4 QUBO exact
    q_matrix, num_vars = build_qubo(states, config, 10.1)
    sol, energy = solve_qubo_exact(q_matrix, num_vars)
    platoons_qubo = decode_solution(sol, states, config)

    # Both should produce 1 platoon of size 3
    assert len(platoons_classical) == 1
    assert len(platoons_qubo) == 1

    assert platoons_classical[0].size == 3
    assert platoons_qubo[0].size == 3


def test_qubo_distance_compatibility_50m(config):
    """TEST 17: Compatibility distance regression test."""
    config["platooning"]["max_formation_distance_m"] = 50.0

    # Vehicles at 100m distance -> should be incompatible
    states_100 = create_states(2)
    states_100[0].position_x = 100.0
    states_100[1].position_x = 0.0  # diff is 100

    q_matrix, n = build_qubo(states_100, config, 10.1)
    sol, _ = solve_qubo_exact(q_matrix, n)
    platoons = decode_solution(sol, states_100, config)
    assert len(platoons) == 0

    # Vehicles at 40m distance -> should be compatible
    states_40 = create_states(2)
    states_40[0].position_x = 100.0
    states_40[1].position_x = 60.0  # diff is 40

    q_matrix, n = build_qubo(states_40, config, 10.1)
    sol, _ = solve_qubo_exact(q_matrix, n)
    platoons = decode_solution(sol, states_40, config)
    assert len(platoons) == 1
    assert platoons[0].size == 2
