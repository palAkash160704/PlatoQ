"""
Classical platoon optimization.

Provides the ``ClassicalOptimizer`` — the baseline optimiser used to solve
the platoon assignment problem using conventional algorithms (e.g. greedy,
ILP, or OR-Tools).  This implementation will be **compared** against the
hybrid quantum-classical optimiser in the evaluation phase.

.. todo:: Phase 4 — implement classical assignment / partitioning solver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from platooning.models.vehicle import VehicleState
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


# ======================================================================
# Abstract base
# ======================================================================
class BaseOptimizer(ABC):
    """Abstract interface for platoon optimisers.

    Every optimiser backend (classical, QUBO, quantum/hybrid) must
    implement this interface so that the ``PlatoonManager`` can treat
    them interchangeably.
    """

    @abstractmethod
    def optimize(self, vehicle_states: list[VehicleState], **kwargs: Any) -> Any:
        """Run the optimisation and return a solution object.

        Parameters
        ----------
        vehicle_states : list[VehicleState]
            Current state of every candidate vehicle.
        **kwargs :
            Backend-specific parameters.

        Returns
        -------
        Any
            A solution representation (format is backend-dependent).
        """

    @abstractmethod
    def evaluate(self, solution: Any) -> dict[str, float]:
        """Evaluate the quality of a given solution.

        Returns
        -------
        dict[str, float]
            Named quality metrics (e.g. ``{"cost": 12.5, "gap_violation": 0}``).
        """

    @abstractmethod
    def get_solution(self) -> Any:
        """Return the most recent solution, or *None* if not yet computed."""


# ======================================================================
# Classical implementation (placeholder)
# ======================================================================
class ClassicalOptimizer(BaseOptimizer):
    """Classical (non-quantum) platoon optimiser.

    Parameters
    ----------
    config : dict
        The ``optimization`` section of the project configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._solution: Any = None
        logger.debug("ClassicalOptimizer initialised.")

    def optimize(self, vehicle_states: list[VehicleState], **kwargs: Any) -> Any:
        """Solve the platoon assignment problem classically.

        .. todo:: Phase 4 — implement greedy / ILP / OR-Tools solver.
        """
        # TODO — Phase 4
        raise NotImplementedError("ClassicalOptimizer.optimize() not yet implemented.")

    def evaluate(self, solution: Any) -> dict[str, float]:
        """Evaluate a classical solution.

        .. todo:: Phase 4 — implement cost / constraint metrics.
        """
        # TODO — Phase 4
        raise NotImplementedError("ClassicalOptimizer.evaluate() not yet implemented.")

    def get_solution(self) -> Any:
        """Return the last computed solution."""
        return self._solution
