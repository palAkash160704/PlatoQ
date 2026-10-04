"""
Hybrid quantum-classical optimiser for platoon assignment.

Coordinates the QUBO formulation with the QAOA solver and exposes
the same ``BaseOptimizer`` interface used by the classical backend
so that the ``PlatoonManager`` can switch between backends transparently.
"""

from __future__ import annotations

from typing import Any

from platooning.models.vehicle import VehicleState
from platooning.optimization.classical.optimizer import BaseOptimizer
from platooning.optimization.quantum.decoder import (
    ConstraintValidationResult,
    PlatoonConfiguration,
    decode_solution,
    select_best_feasible,
    validate_solution,
)
from platooning.optimization.quantum.qaoa_solver import QAOASolver
from platooning.optimization.qubo_builder import build_qubo
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


class QuantumOptimizer(BaseOptimizer):
    """Hybrid quantum-classical platoon optimiser.

    Parameters
    ----------
    config : dict, optional
        The full project configuration (accesses ``quantum`` and ``qubo``
        sections).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._solution: Any = None
        self._last_qaoa_result: Any = None

        quantum_conf = self._config.get("quantum", {})
        self._reps = int(quantum_conf.get("reps", 1))
        self._shots = int(quantum_conf.get("shots", 1024))
        self._optimizer = str(quantum_conf.get("optimizer", "COBYLA"))
        self._seed = int(quantum_conf.get("seed", 42))

    def optimize(
        self, vehicle_states: list[VehicleState], **kwargs: Any
    ) -> Any:
        """Run the hybrid quantum-classical optimisation pipeline.

        Parameters
        ----------
        vehicle_states : list[VehicleState]
            Current vehicle states.
        **kwargs :
            Optional overrides: ``current_time``, ``reps``, ``shots``,
            ``optimizer``, ``seed``.

        Returns
        -------
        dict
            Solution dict with ``platoons``, ``energy``, ``feasible``, etc.
        """
        current_time = kwargs.get("current_time", 0.0)
        reps = kwargs.get("reps", self._reps)
        shots = kwargs.get("shots", self._shots)
        optimizer = kwargs.get("optimizer", self._optimizer)
        seed = kwargs.get("seed", self._seed)

        # 1. Build QUBO
        q_matrix, num_vars = build_qubo(vehicle_states, self._config, current_time)

        if num_vars == 0:
            self._solution = {"platoons": [], "energy": 0.0, "feasible": True}
            return self._solution

        # 2. Run QAOA
        solver = QAOASolver(reps=reps, shots=shots, optimizer=optimizer, seed=seed)
        qaoa_result = solver.solve(q_matrix, num_vars)
        self._last_qaoa_result = qaoa_result

        # 3. Select best feasible solution
        best_bits, best_energy, is_feasible = select_best_feasible(
            qaoa_result.counts, q_matrix, vehicle_states, self._config, current_time
        )

        # 4. Decode
        if best_bits is not None:
            decoded = decode_solution(best_bits, vehicle_states, self._config)
            validation = validate_solution(
                best_bits, vehicle_states, self._config, current_time
            )
        else:
            decoded = PlatoonConfiguration()
            validation = ConstraintValidationResult(feasible=False)

        self._solution = {
            "platoons": decoded.platoons,
            "ungrouped": decoded.ungrouped,
            "energy": best_energy,
            "feasible": is_feasible,
            "constraint_violations": validation.violation_count,
            "qaoa_result": qaoa_result,
        }

        return self._solution

    def evaluate(self, solution: Any) -> dict[str, float]:
        """Evaluate a quantum solution.

        Returns
        -------
        dict[str, float]
            Named quality metrics.
        """
        if solution is None:
            return {"cost": float("inf"), "feasible": 0.0}

        return {
            "cost": solution.get("energy", float("inf")),
            "feasible": 1.0 if solution.get("feasible", False) else 0.0,
            "constraint_violations": float(
                solution.get("constraint_violations", 0)
            ),
            "num_platoons": float(len(solution.get("platoons", []))),
        }

    def get_solution(self) -> Any:
        """Return the last computed quantum/hybrid solution."""
        return self._solution
