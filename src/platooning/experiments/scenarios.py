"""
Experiment scenario definitions.

A *scenario* bundles together a SUMO network, traffic demand, platooning
parameters, and optimisation backend selection for a reproducible
experimental run.

.. todo:: Phase 8 — implement scenario loading from YAML files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scenario:
    """Immutable experiment scenario descriptor.

    Attributes
    ----------
    name : str
        Human-readable scenario name.
    description : str
        Brief description of what the scenario tests.
    simulation_config : dict
        Overrides for the ``simulation`` config section.
    platooning_config : dict
        Overrides for the ``platooning`` config section.
    optimization_method : str
        Which optimiser backend to use (``"classical"`` | ``"qubo"`` | ``"quantum"``).
    """

    name: str = "default"
    description: str = ""
    simulation_config: dict[str, Any] = field(default_factory=dict)
    platooning_config: dict[str, Any] = field(default_factory=dict)
    optimization_method: str = "classical"
