"""
Experiment runner.

Orchestrates the full experiment pipeline:
scenario → simulation → data collection → optimisation → evaluation.

.. todo:: Phase 8 — implement the run loop.
"""

from __future__ import annotations

from platooning.experiments.scenarios import Scenario
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentRunner:
    """Execute and manage experiment runs.

    .. todo:: Phase 8 — implement ``run()`` with simulation + optimisation loop.
    """

    def run(self, scenario: Scenario) -> dict:
        """Run a single experiment scenario and return collected metrics.

        .. todo:: Phase 8
        """
        # TODO — Phase 8
        raise NotImplementedError("ExperimentRunner.run() not yet implemented.")
