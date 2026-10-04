"""
Phase 5 QAOA tests — comprehensive test suite.

Tests cover:
1.  QUBO → Ising conversion
2.  Energy equivalence verification
3.  QAOA circuit creation
4.  Deterministic seed behaviour
5.  Bitstring decoding
6.  Leader activation validation
7.  Compatibility validation
8.  Size constraint validation
9.  Feasible solution detection
10. Infeasible solution detection
11. Exact solver comparison
12. 3-vehicle QAOA scenario
13. 4-vehicle QAOA scenario
14. 5-vehicle QAOA scenario
15. Sample analysis
"""

import itertools
import math

import pytest

from platooning.models.vehicle import VehicleState
from platooning.optimization.quantum.comparison import (
    run_classical,
    run_exact_qubo,
    run_qaoa,
)
from platooning.optimization.quantum.decoder import (
    PlatoonConfiguration,
    analyze_samples,
    build_variable_mapping,
    decode_solution,
    select_best_feasible,
    validate_solution,
)
from platooning.optimization.quantum.ising import (
    qubo_to_ising,
    verify_energy_equivalence,
)
from platooning.optimization.quantum.qaoa_solver import QAOASolver
from platooning.optimization.qubo_builder import (
    build_qubo,
    solve_qubo_exact,
)


# ======================================================================
# Fixtures
# ======================================================================
@pytest.fixture
def config():
    """Standard test configuration with relaxed constraints for testing."""
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
        "quantum": {
            "backend": "aer_simulator",
            "shots": 512,
            "reps": 1,
            "optimizer": "COBYLA",
            "seed": 42,
        },
    }


def create_states(
    n: int,
    start_x: float = 100.0,
    speed: float = 30.0,
    spacing: float = 5.0,
    route: str = "r1",
    lane: str = "l1",
) -> list[VehicleState]:
    """Create n test vehicles spaced evenly."""
    return [
        VehicleState(
            vehicle_id=f"v{i}",
            timestamp=10.0,
            position_x=start_x - i * spacing,
            position_y=0.0,
            speed=speed,
            route_id=route,
            lane_id=lane,
        )
        for i in range(1, n + 1)
    ]


# ======================================================================
# Test 1: QUBO → Ising conversion
# ======================================================================
class TestIsingConversion:
    def test_simple_diagonal(self):
        """Single diagonal QUBO entry converts correctly."""
        q = {(0, 0): -2.0}
        ising = qubo_to_ising(q, 1)

        # Q_{00} = -2.0
        # offset = -2.0 / 2 = -1.0
        # h[0] = -(-2.0 / 2) = 1.0
        assert ising.num_qubits == 1
        assert math.isclose(ising.offset, -1.0)
        assert math.isclose(ising.h.get(0, 0.0), 1.0)

    def test_off_diagonal(self):
        """Single off-diagonal entry converts correctly."""
        q = {(0, 1): 4.0}
        ising = qubo_to_ising(q, 2)

        # offset = 4.0/4 = 1.0
        # h[0] = -(4.0/4) = -1.0
        # h[1] = -(4.0/4) = -1.0
        # J[0,1] = 4.0/4 = 1.0
        assert math.isclose(ising.offset, 1.0)
        assert math.isclose(ising.h.get(0, 0.0), -1.0)
        assert math.isclose(ising.h.get(1, 0.0), -1.0)
        assert math.isclose(ising.J.get((0, 1), 0.0), 1.0)

    def test_empty_qubo(self):
        """Empty QUBO produces zero Ising."""
        q: dict[tuple[int, int], float] = {}
        ising = qubo_to_ising(q, 0)
        assert ising.num_qubits == 0
        assert ising.offset == 0.0
        assert len(ising.h) == 0
        assert len(ising.J) == 0

    def test_real_qubo_conversion(self, config):
        """Phase 4 QUBO for 2 vehicles converts correctly."""
        states = create_states(2)
        q_matrix, num_vars = build_qubo(states, config, 10.1)

        ising = qubo_to_ising(q_matrix, num_vars)
        assert ising.num_qubits == num_vars
        assert ising.num_qubits > 0


# ======================================================================
# Test 2: Energy equivalence
# ======================================================================
class TestEnergyEquivalence:
    def test_all_bitstrings_2vars(self):
        """QUBO and Ising energies agree for all 2-variable bitstrings."""
        q = {(0, 0): -3.0, (1, 1): -2.0, (0, 1): 1.5}
        ising = qubo_to_ising(q, 2)

        for bits in itertools.product([0, 1], repeat=2):
            bits_list = list(bits)
            qubo_e, ising_e, match = verify_energy_equivalence(q, ising, bits_list)
            assert match, f"Mismatch at {bits}: QUBO={qubo_e}, Ising={ising_e}"

    def test_real_qubo_all_bitstrings(self, config):
        """Phase 4 QUBO for 2 vehicles: energies match for all bitstrings."""
        states = create_states(2)
        q_matrix, num_vars = build_qubo(states, config, 10.1)
        ising = qubo_to_ising(q_matrix, num_vars)

        for bits in itertools.product([0, 1], repeat=num_vars):
            bits_list = list(bits)
            qubo_e, ising_e, match = verify_energy_equivalence(
                q_matrix, ising, bits_list
            )
            assert match, f"Mismatch at {bits}: QUBO={qubo_e}, Ising={ising_e}"

    def test_3vehicle_qubo_spot_check(self, config):
        """3-vehicle QUBO: spot-check the exact solution bitstring."""
        states = create_states(3)
        q_matrix, num_vars = build_qubo(states, config, 10.1)
        sol, energy = solve_qubo_exact(q_matrix, num_vars)

        ising = qubo_to_ising(q_matrix, num_vars)
        qubo_e, ising_e, match = verify_energy_equivalence(q_matrix, ising, sol)
        assert match


# ======================================================================
# Test 3: QAOA circuit creation
# ======================================================================
class TestQAOACircuit:
    def test_circuit_depth_1(self, config):
        """QAOA p=1 circuit has correct structure."""
        import numpy as np

        from platooning.optimization.quantum.qaoa_solver import QAOASolver

        states = create_states(2)
        q_matrix, num_vars = build_qubo(states, config, 10.1)
        ising = qubo_to_ising(q_matrix, num_vars)

        solver = QAOASolver()
        gammas = np.array([0.5])
        betas = np.array([0.3])

        qc = solver._build_qaoa_circuit(ising, gammas, betas)
        assert qc.num_qubits == num_vars

    def test_circuit_depth_2(self, config):
        """QAOA p=2 circuit creates without error."""
        import numpy as np

        states = create_states(2)
        q_matrix, num_vars = build_qubo(states, config, 10.1)
        ising = qubo_to_ising(q_matrix, num_vars)

        solver = QAOASolver()
        gammas = np.array([0.5, 0.3])
        betas = np.array([0.3, 0.2])

        qc = solver._build_qaoa_circuit(ising, gammas, betas)
        assert qc.num_qubits == num_vars


# ======================================================================
# Test 4: Deterministic seed behaviour
# ======================================================================
class TestDeterminism:
    def test_same_seed_same_result(self, config):
        """Running QAOA with the same seed produces identical results."""
        states = create_states(2)
        q_matrix, num_vars = build_qubo(states, config, 10.1)

        solver = QAOASolver(reps=1, shots=256, seed=42)

        result1 = solver.solve(q_matrix, num_vars, seed=42)
        result2 = solver.solve(q_matrix, num_vars, seed=42)

        assert result1.best_bitstring == result2.best_bitstring
        assert math.isclose(result1.best_energy, result2.best_energy)


# ======================================================================
# Test 5: Bitstring decoding
# ======================================================================
class TestDecoding:
    def test_decode_valid_2vehicle(self, config):
        """Decode a valid 2-vehicle solution."""
        states = create_states(2)
        q_matrix, num_vars = build_qubo(states, config, 10.1)
        sol, _ = solve_qubo_exact(q_matrix, num_vars)

        decoded = decode_solution(sol, states, config)
        assert len(decoded.platoons) == 1
        assert decoded.platoons[0].size == 2

    def test_decode_all_zeros(self, config):
        """All-zeros bitstring produces no platoons."""
        states = create_states(3)
        q_matrix, num_vars = build_qubo(states, config, 10.1)

        all_zeros = [0] * num_vars
        decoded = decode_solution(all_zeros, states, config)
        assert len(decoded.platoons) == 0
        assert len(decoded.ungrouped) == 3

    def test_decode_preserves_vehicle_ids(self, config):
        """Decoded platoons contain actual vehicle IDs, not indices."""
        states = create_states(2)
        q_matrix, num_vars = build_qubo(states, config, 10.1)
        sol, _ = solve_qubo_exact(q_matrix, num_vars)

        decoded = decode_solution(sol, states, config)
        all_ids = set()
        for p in decoded.platoons:
            all_ids.update(p.vehicle_ids)
        for uid in decoded.ungrouped:
            all_ids.add(uid)

        expected_ids = {s.vehicle_id for s in states}
        # Every decoded vehicle ID should be a known vehicle
        assert all_ids.issubset(expected_ids)


# ======================================================================
# Test 6: Leader activation validation
# ======================================================================
class TestLeaderValidation:
    def test_leader_not_activated(self, config):
        """Detect when members are assigned but leader is not self-assigned."""
        states = create_states(2)
        q_matrix, num_vars = build_qubo(states, config, 10.1)

        # Manually create invalid: x_{1,0}=1 but x_{0,0}=0
        # Variable mapping: (0,0)=0, (1,0)=1, (1,1)=2, s_{0,2}=3
        bad_sol = [0, 1, 0, 0]
        val = validate_solution(bad_sol, states, config, 10.1)

        # This should detect a leader issue (size violation at minimum)
        assert val.feasible is False


# ======================================================================
# Test 7: Compatibility validation
# ======================================================================
class TestCompatibilityValidation:
    def test_incompatible_routes(self, config):
        """Detect incompatible routes in decoded solution."""
        states = create_states(2)
        states[1].route_id = "r2"

        # Force both into same platoon
        q_matrix, num_vars = build_qubo(states, config, 10.1)
        # Build a bitstring that forces assignment
        x_indices, _, _ = build_variable_mapping(2, config)
        forced = [0] * num_vars
        forced[x_indices[(0, 0)]] = 1
        forced[x_indices[(1, 0)]] = 1

        val = validate_solution(forced, states, config, 10.1)
        assert val.feasible is False
        has_route_violation = any("route" in v.lower() for v in val.violations)
        assert has_route_violation

    def test_incompatible_lanes(self, config):
        """Detect incompatible lanes in decoded solution."""
        states = create_states(2)
        states[1].lane_id = "l2"

        q_matrix, num_vars = build_qubo(states, config, 10.1)
        x_indices, _, _ = build_variable_mapping(2, config)
        forced = [0] * num_vars
        forced[x_indices[(0, 0)]] = 1
        forced[x_indices[(1, 0)]] = 1

        val = validate_solution(forced, states, config, 10.1)
        assert val.feasible is False
        has_lane_violation = any("lane" in v.lower() for v in val.violations)
        assert has_lane_violation


# ======================================================================
# Test 8: Size constraint validation
# ======================================================================
class TestSizeValidation:
    def test_solo_vehicle_invalid(self, config):
        """A single self-assigned vehicle (size=1) violates min_size=2."""
        states = create_states(2)
        q_matrix, num_vars = build_qubo(states, config, 10.1)

        # x_{0,0}=1, x_{1,1}=1 (two solo platoons of size 1)
        x_indices, _, _ = build_variable_mapping(2, config)
        sol = [0] * num_vars
        sol[x_indices[(0, 0)]] = 1
        sol[x_indices[(1, 1)]] = 1

        val = validate_solution(sol, states, config, 10.1)
        assert val.feasible is False
        has_size_violation = any("size" in v.lower() for v in val.violations)
        assert has_size_violation


# ======================================================================
# Test 9: Feasible solution detection
# ======================================================================
class TestFeasibleDetection:
    def test_exact_solution_is_feasible(self, config):
        """Exact solver solution should be feasible."""
        states = create_states(3)
        q_matrix, num_vars = build_qubo(states, config, 10.1)
        sol, _ = solve_qubo_exact(q_matrix, num_vars)

        val = validate_solution(sol, states, config, 10.1)
        assert val.feasible is True
        assert val.violation_count == 0


# ======================================================================
# Test 10: Infeasible solution detection
# ======================================================================
class TestInfeasibleDetection:
    def test_all_ones_infeasible(self, config):
        """All-ones bitstring should be infeasible."""
        states = create_states(3)
        q_matrix, num_vars = build_qubo(states, config, 10.1)

        all_ones = [1] * num_vars
        val = validate_solution(all_ones, states, config, 10.1)
        assert val.feasible is False
        assert val.violation_count > 0


# ======================================================================
# Test 11: Exact solver comparison
# ======================================================================
class TestExactComparison:
    def test_qaoa_energy_geq_exact(self, config):
        """QAOA best energy should be >= exact solver energy (minimisation)."""
        states = create_states(2)
        q_matrix, num_vars = build_qubo(states, config, 10.1)

        sol, exact_energy = solve_qubo_exact(q_matrix, num_vars)

        solver = QAOASolver(reps=1, shots=512, seed=42)
        result = solver.solve(q_matrix, num_vars)

        # QAOA energy should be >= exact (it's a minimisation problem)
        assert result.best_energy >= exact_energy - 1e-6


# ======================================================================
# Test 12: 3-vehicle QAOA scenario
# ======================================================================
class TestQAOA3Vehicle:
    def test_3vehicle_qaoa_runs(self, config):
        """QAOA completes successfully for 3 vehicles."""
        states = create_states(3)
        q_matrix, num_vars = build_qubo(states, config, 10.1)

        solver = QAOASolver(reps=1, shots=256, seed=42)
        result = solver.solve(q_matrix, num_vars)

        assert result.success is True
        assert len(result.best_bitstring) == num_vars
        assert result.runtime_seconds > 0

    def test_3vehicle_decodable(self, config):
        """QAOA output for 3 vehicles can be decoded."""
        states = create_states(3)
        q_matrix, num_vars = build_qubo(states, config, 10.1)

        solver = QAOASolver(reps=1, shots=256, seed=42)
        result = solver.solve(q_matrix, num_vars)

        decoded = decode_solution(result.best_bitstring, states, config)
        assert isinstance(decoded, PlatoonConfiguration)


# ======================================================================
# Test 13: 4-vehicle QAOA scenario
# ======================================================================
class TestQAOA4Vehicle:
    def test_4vehicle_qaoa_runs(self, config):
        """QAOA completes successfully for 4 vehicles."""
        states = create_states(4)
        q_matrix, num_vars = build_qubo(states, config, 10.1)

        solver = QAOASolver(reps=1, shots=256, seed=42)
        result = solver.solve(q_matrix, num_vars)

        assert result.success is True
        assert len(result.best_bitstring) == num_vars


# ======================================================================
# Test 14: 5-vehicle QAOA scenario
# ======================================================================
class TestQAOA5Vehicle:
    def test_5vehicle_qaoa_runs(self, config):
        """QAOA completes successfully for 5 vehicles."""
        states = create_states(5)
        q_matrix, num_vars = build_qubo(states, config, 10.1)

        solver = QAOASolver(reps=1, shots=256, seed=42)
        result = solver.solve(q_matrix, num_vars)

        assert result.success is True
        assert len(result.best_bitstring) == num_vars
        assert result.runtime_seconds > 0

    def test_5vehicle_best_feasible_selection(self, config):
        """Best feasible selection works for 5 vehicles."""
        states = create_states(5)
        q_matrix, num_vars = build_qubo(states, config, 10.1)

        solver = QAOASolver(reps=1, shots=256, seed=42)
        result = solver.solve(q_matrix, num_vars)

        best_bits, best_energy, is_feasible = select_best_feasible(
            result.counts, q_matrix, states, config, 10.1
        )

        assert best_bits is not None
        assert len(best_bits) == num_vars


# ======================================================================
# Test 15: Sample analysis
# ======================================================================
class TestSampleAnalysis:
    def test_analyze_samples(self, config):
        """Sample analysis returns structured data."""
        states = create_states(2)
        q_matrix, num_vars = build_qubo(states, config, 10.1)

        solver = QAOASolver(reps=1, shots=256, seed=42)
        result = solver.solve(q_matrix, num_vars)

        analyses = analyze_samples(
            result.counts, q_matrix, states, config, 10.1, top_k=5
        )

        assert len(analyses) > 0
        for a in analyses:
            assert "energy" in a
            assert "feasible" in a
            assert "count" in a
            assert "frequency" in a
            assert a["frequency"] > 0.0


# ======================================================================
# Test: Comparison framework
# ======================================================================
class TestComparison:
    def test_run_classical(self, config):
        """Classical greedy runs and returns MethodResult."""
        states = create_states(3)
        result = run_classical(states, config, 10.1)
        assert result.method == "classical_greedy"
        assert result.runtime_seconds >= 0

    def test_run_exact_qubo(self, config):
        """Exact QUBO runs and returns MethodResult."""
        states = create_states(3)
        result = run_exact_qubo(states, config, 10.1)
        assert result.method == "exact_qubo"
        assert result.qubo_energy < float("inf")

    def test_run_qaoa(self, config):
        """QAOA runs and returns MethodResult + QAOAResult."""
        states = create_states(2)
        method_result, qaoa_result = run_qaoa(
            states, config, 10.1, reps=1, shots=256, seed=42
        )
        assert method_result.method == "qaoa_p1"
        assert qaoa_result.success is True


# ======================================================================
# Test: Variable mapping consistency
# ======================================================================
class TestVariableMapping:
    def test_mapping_matches_qubo_builder(self, config):
        """Decoder variable mapping matches the QUBO builder exactly."""
        for n in range(2, 5):
            states = create_states(n)
            _, num_from_builder = build_qubo(states, config, 10.1)
            _, _, num_from_decoder = build_variable_mapping(n, config)
            assert num_from_builder == num_from_decoder, (
                f"Variable count mismatch for {n} vehicles: "
                f"builder={num_from_builder}, decoder={num_from_decoder}"
            )
