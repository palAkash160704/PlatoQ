"""
Evaluation metrics for experiment results.

Defines the metrics used to compare classical vs. hybrid quantum-classical
optimisation for cooperative vehicle platooning.

.. todo:: Phase 8 — implement metric computation functions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlatooningMetrics:
    """Aggregated metrics for a single experiment run.

    Attributes
    ----------
    solution_cost : float
        Objective function value of the optimiser solution.
    computation_time_s : float
        Wall-clock time spent in the optimiser (seconds).
    num_platoons_formed : int
        Number of platoons formed during the run.
    average_platoon_size : float
        Mean number of vehicles per platoon.
    gap_violations : int
        Count of time-steps where inter-vehicle gap was below minimum.
    average_speed_deviation : float
        Mean absolute speed deviation from target (m/s).
    """

    solution_cost: float = 0.0
    computation_time_s: float = 0.0
    num_platoons_formed: int = 0
    average_platoon_size: float = 0.0
    gap_violations: int = 0
    average_speed_deviation: float = 0.0
