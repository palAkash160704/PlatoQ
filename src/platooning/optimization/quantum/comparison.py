"""
Comparison module for Phase 5 — Classical vs Exact QUBO vs QAOA.

Runs the same input scenario through all three approaches and computes
structured comparison metrics. Does NOT make claims of quantum advantage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from platooning.models.vehicle import VehicleState
from platooning.optimization.quantum.decoder import (
    decode_solution,
    select_best_feasible,
    validate_solution,
)
from platooning.optimization.quantum.qaoa_solver import QAOAResult, QAOASolver
from platooning.optimization.qubo_builder import (
    build_qubo,
    solve_qubo_exact,
)
from platooning.optimization.qubo_builder import (
    decode_solution as qubo_decode_solution,
)
from platooning.optimization.qubo_builder import (
    validate_solution as qubo_validate_solution,
)
from platooning.platooning.formation import form_platoons
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MethodResult:
    """Result from a single optimisation method."""

    method: str = ""
    objective: float = 0.0
    qubo_energy: float = float("inf")
    num_platoons: int = 0
    average_platoon_size: float = 0.0
    largest_platoon_size: int = 0
    grouped_vehicles: int = 0
    ungrouped_vehicles: int = 0
    constraint_violations: int = 0
    feasible: bool = True
    runtime_seconds: float = 0.0
    platoon_details: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Structured result comparing all three methods.

    Attributes
    ----------
    scenario_name : str
        Name of the input scenario.
    num_vehicles : int
        Number of input vehicles.
    num_qubo_vars : int
        Number of QUBO binary variables.
    classical : MethodResult
        Classical greedy formation result.
    exact_qubo : MethodResult
        Exact QUBO solver result.
    qaoa_results : dict[int, MethodResult]
        QAOA results keyed by depth p.
    metrics : dict
        Derived comparison metrics.
    """

    scenario_name: str = ""
    num_vehicles: int = 0
    num_qubo_vars: int = 0
    classical: MethodResult = field(default_factory=MethodResult)
    exact_qubo: MethodResult = field(default_factory=MethodResult)
    qaoa_results: dict[int, MethodResult] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


def _compute_objective(
    platoons: list,
    states: list[VehicleState],
    config: dict[str, Any],
) -> float:
    """Compute the Phase 3 objective value for a set of platoons."""
    plat_conf = config.get("platooning", {})
    w_membership = float(plat_conf.get("weight_membership", 10.0))
    w_dist = float(plat_conf.get("weight_distance_penalty", 0.5))
    w_speed = float(plat_conf.get("weight_speed_penalty", 1.0))
    w_ungrouped = float(plat_conf.get("weight_ungrouped_penalty", 5.0))

    state_dict = {s.vehicle_id: s for s in states}
    grouped = sum(p.size for p in platoons)
    ungrouped = len(states) - grouped

    objective = grouped * w_membership - ungrouped * w_ungrouped

    for p in platoons:
        leader_state = state_dict.get(p.leader_id)
        if leader_state is None:
            continue

        for i in range(len(p.vehicle_ids) - 1):
            v1 = state_dict.get(p.vehicle_ids[i])
            v2 = state_dict.get(p.vehicle_ids[i + 1])
            if v1 and v2:
                dist = (
                    (v1.position_x - v2.position_x) ** 2
                    + (v1.position_y - v2.position_y) ** 2
                ) ** 0.5
                objective -= dist * w_dist

        for vid in p.vehicle_ids:
            v = state_dict.get(vid)
            if v:
                speed_diff = abs(leader_state.speed - v.speed)
                objective -= speed_diff * w_speed

    return objective


def run_classical(
    states: list[VehicleState],
    config: dict[str, Any],
    current_time: float,
) -> MethodResult:
    """Run Phase 3 classical greedy formation."""
    result = MethodResult(method="classical_greedy")

    start = time.perf_counter()
    platoons = form_platoons(states, config, current_time)
    elapsed = time.perf_counter() - start

    result.num_platoons = len(platoons)
    result.runtime_seconds = elapsed

    if platoons:
        result.average_platoon_size = sum(p.size for p in platoons) / len(platoons)
        result.largest_platoon_size = max(p.size for p in platoons)
        result.grouped_vehicles = sum(p.size for p in platoons)

    result.ungrouped_vehicles = len(states) - result.grouped_vehicles
    result.objective = _compute_objective(platoons, states, config)
    result.constraint_violations = 0
    result.feasible = True

    result.platoon_details = [
        {
            "platoon_id": p.platoon_id,
            "leader": p.leader_id,
            "members": p.vehicle_ids,
            "size": p.size,
        }
        for p in platoons
    ]

    return result


def run_exact_qubo(
    states: list[VehicleState],
    config: dict[str, Any],
    current_time: float,
) -> MethodResult:
    """Run Phase 4 exact QUBO solver."""
    result = MethodResult(method="exact_qubo")

    start = time.perf_counter()
    q_matrix, num_vars = build_qubo(states, config, current_time)
    sol, energy = solve_qubo_exact(q_matrix, num_vars)
    elapsed = time.perf_counter() - start

    result.qubo_energy = energy
    result.runtime_seconds = elapsed

    # Validate using Phase 4 validator
    val = qubo_validate_solution(sol, states, config, current_time)
    result.feasible = val["valid"]
    result.constraint_violations = len(val["violations"])

    # Decode
    platoons = qubo_decode_solution(sol, states, config)
    result.num_platoons = len(platoons)

    if platoons:
        result.average_platoon_size = sum(p.size for p in platoons) / len(platoons)
        result.largest_platoon_size = max(p.size for p in platoons)
        result.grouped_vehicles = sum(p.size for p in platoons)

    result.ungrouped_vehicles = len(states) - result.grouped_vehicles
    result.objective = _compute_objective(platoons, states, config)

    result.platoon_details = [
        {
            "platoon_id": p.platoon_id,
            "leader": p.leader_id,
            "members": p.vehicle_ids,
            "size": p.size,
        }
        for p in platoons
    ]

    return result


def run_qaoa(
    states: list[VehicleState],
    config: dict[str, Any],
    current_time: float,
    reps: int = 1,
    shots: int = 1024,
    optimizer: str = "COBYLA",
    seed: int = 42,
) -> tuple[MethodResult, QAOAResult]:
    """Run Phase 5 QAOA solver.

    Returns
    -------
    tuple
        (MethodResult for comparison, raw QAOAResult for detailed analysis)
    """
    result = MethodResult(method=f"qaoa_p{reps}")

    # Build QUBO
    q_matrix, num_vars = build_qubo(states, config, current_time)

    # Run QAOA
    solver = QAOASolver(reps=reps, shots=shots, optimizer=optimizer, seed=seed)

    start = time.perf_counter()
    qaoa_result = solver.solve(q_matrix, num_vars)
    elapsed = time.perf_counter() - start

    result.runtime_seconds = elapsed

    # Select best feasible solution from all samples
    best_bits, best_energy, is_feasible = select_best_feasible(
        qaoa_result.counts, q_matrix, states, config, current_time
    )

    if best_bits is None:
        result.qubo_energy = float("inf")
        result.feasible = False
        return result, qaoa_result

    result.qubo_energy = best_energy
    result.feasible = is_feasible

    # Validate
    validation = validate_solution(best_bits, states, config, current_time)
    result.constraint_violations = validation.violation_count

    # Decode
    decoded = decode_solution(best_bits, states, config)
    platoons = decoded.platoons

    result.num_platoons = len(platoons)
    if platoons:
        result.average_platoon_size = sum(p.size for p in platoons) / len(platoons)
        result.largest_platoon_size = max(p.size for p in platoons)
        result.grouped_vehicles = sum(p.size for p in platoons)

    result.ungrouped_vehicles = len(states) - result.grouped_vehicles
    result.objective = _compute_objective(platoons, states, config)

    result.platoon_details = [
        {
            "platoon_id": p.platoon_id,
            "leader": p.leader_id,
            "members": p.vehicle_ids,
            "size": p.size,
        }
        for p in platoons
    ]

    return result, qaoa_result


def compare_all(
    states: list[VehicleState],
    config: dict[str, Any],
    current_time: float,
    scenario_name: str = "default",
    qaoa_depths: list[int] | None = None,
    qaoa_shots: int = 1024,
    qaoa_optimizer: str = "COBYLA",
    qaoa_seed: int = 42,
) -> ComparisonResult:
    """Run all three approaches and compute comparison metrics.

    Parameters
    ----------
    states : list[VehicleState]
        Input vehicle states (identical for all methods).
    config : dict
        Project configuration.
    current_time : float
        Simulation time.
    scenario_name : str
        Label for this scenario.
    qaoa_depths : list[int], optional
        QAOA depths to test. Defaults to [1, 2, 3].
    qaoa_shots : int
        QAOA measurement shots.
    qaoa_optimizer : str
        QAOA classical optimiser.
    qaoa_seed : int
        Random seed.

    Returns
    -------
    ComparisonResult
    """
    if qaoa_depths is None:
        qaoa_depths = [1, 2, 3]

    q_matrix, num_vars = build_qubo(states, config, current_time)

    comparison = ComparisonResult(
        scenario_name=scenario_name,
        num_vehicles=len(states),
        num_qubo_vars=num_vars,
    )

    # 1. Classical
    logger.info("[COMPARE] Running classical greedy...")
    comparison.classical = run_classical(states, config, current_time)

    # 2. Exact QUBO
    logger.info("[COMPARE] Running exact QUBO solver...")
    comparison.exact_qubo = run_exact_qubo(states, config, current_time)

    # 3. QAOA at each depth
    for p in qaoa_depths:
        logger.info("[COMPARE] Running QAOA p=%d...", p)
        method_result, qaoa_raw = run_qaoa(
            states,
            config,
            current_time,
            reps=p,
            shots=qaoa_shots,
            optimizer=qaoa_optimizer,
            seed=qaoa_seed,
        )
        comparison.qaoa_results[p] = method_result

    # 4. Compute derived metrics
    exact_obj = comparison.exact_qubo.objective
    exact_energy = comparison.exact_qubo.qubo_energy
    epsilon = 1e-10

    comparison.metrics = {}
    for p, qaoa_res in comparison.qaoa_results.items():
        key = f"qaoa_p{p}"
        obj_gap = exact_obj - qaoa_res.objective
        rel_gap = obj_gap / max(abs(exact_obj), epsilon)
        energy_gap = qaoa_res.qubo_energy - exact_energy

        comparison.metrics[key] = {
            "objective_gap": obj_gap,
            "relative_objective_gap": rel_gap,
            "energy_gap": energy_gap,
            "feasible": qaoa_res.feasible,
            "runtime_ratio": (
                qaoa_res.runtime_seconds
                / max(comparison.exact_qubo.runtime_seconds, 1e-10)
            ),
        }

    # Log summary
    logger.info("[COMPARE] ========================================")
    logger.info(
        "[COMPARE] Classical: objective=%.2f  platoons=%d  runtime=%.4fs",
        comparison.classical.objective,
        comparison.classical.num_platoons,
        comparison.classical.runtime_seconds,
    )
    logger.info(
        "[COMPARE] Exact QUBO: objective=%.2f  energy=%.2f  platoons=%d  runtime=%.4fs",
        comparison.exact_qubo.objective,
        comparison.exact_qubo.qubo_energy,
        comparison.exact_qubo.num_platoons,
        comparison.exact_qubo.runtime_seconds,
    )
    for p, qaoa_res in comparison.qaoa_results.items():
        logger.info(
            "[COMPARE] QAOA p=%d: objective=%.2f  energy=%.2f  feasible=%s  runtime=%.4fs",
            p,
            qaoa_res.objective,
            qaoa_res.qubo_energy,
            qaoa_res.feasible,
            qaoa_res.runtime_seconds,
        )
    logger.info("[COMPARE] ========================================")

    return comparison


def comparison_to_dict(comp: ComparisonResult) -> dict[str, Any]:
    """Serialise a ComparisonResult to a JSON-compatible dictionary."""
    from dataclasses import asdict

    d: dict[str, Any] = {
        "scenario": comp.scenario_name,
        "num_vehicles": comp.num_vehicles,
        "num_qubo_vars": comp.num_qubo_vars,
        "classical": asdict(comp.classical),
        "exact_qubo": asdict(comp.exact_qubo),
        "qaoa": {},
        "metrics": comp.metrics,
    }

    for p, qaoa_res in comp.qaoa_results.items():
        d["qaoa"][f"p{p}"] = asdict(qaoa_res)

    return d
