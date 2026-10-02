"""
Configuration loading and validation.

Reads ``config/config.yaml`` (relative to the repository root) and
exposes helper functions used by the rest of the application.

All values in the default configuration file are **development defaults**
and have **not** been experimentally validated.  They must be tuned per
experiment scenario.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Locate repository root and default config
# ---------------------------------------------------------------------------
# The repo root is assumed to be three levels above this file:
#   src/platooning/config/settings.py → repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and merge with environment variables.

    Parameters
    ----------
    path:
        Explicit path to a YAML config file.  When *None*, the default
        ``config/config.yaml`` at the repository root is used.

    Returns
    -------
    dict:
        Parsed configuration dictionary.
    """
    # Load .env if present (does not override existing env vars)
    dotenv_path = _REPO_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)

    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    # Overlay SUMO_HOME from environment if set
    sumo_home = os.getenv("SUMO_HOME")
    if sumo_home:
        config.setdefault("simulation", {})["sumo_home"] = sumo_home

    return config


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate the loaded configuration and return a list of error messages.

    Returns an empty list when the configuration is valid.
    """
    errors: list[str] = []

    # --- simulation --------------------------------------------------------
    sim = config.get("simulation")
    if not sim:
        errors.append("Missing 'simulation' section.")
    else:
        if sim.get("simulator") not in ("sumo",):
            errors.append(
                f"Unsupported simulator: {sim.get('simulator')}. "
                "Only 'sumo' is currently supported."
            )
        step = sim.get("step_length")
        if step is not None and (not isinstance(step, (int, float)) or step <= 0):
            errors.append("simulation.step_length must be a positive number.")

    # --- communication -----------------------------------------------------
    comm = config.get("communication")
    if not comm:
        errors.append("Missing 'communication' section.")
    else:
        if (r := comm.get("range_meters")) is not None and r <= 0:
            errors.append("communication.range_meters must be positive.")
        plr = comm.get("packet_loss_rate")
        if plr is not None and not (0.0 <= plr <= 1.0):
            errors.append("communication.packet_loss_rate must be in [0.0, 1.0].")

    # --- platooning --------------------------------------------------------
    plat = config.get("platooning")
    if not plat:
        errors.append("Missing 'platooning' section.")
    else:
        mn = plat.get("minimum_platoon_size", 2)
        mx = plat.get("maximum_platoon_size", 10)
        if mn < 2:
            errors.append("platooning.minimum_platoon_size must be >= 2.")
        if mx < mn:
            errors.append(
                "platooning.maximum_platoon_size must be >= minimum_platoon_size."
            )

    # --- optimization ------------------------------------------------------
    opt = config.get("optimization")
    if not opt:
        errors.append("Missing 'optimization' section.")
    else:
        allowed_methods = ("classical", "qubo", "quantum")
        if opt.get("method") not in allowed_methods:
            errors.append(f"optimization.method must be one of {allowed_methods}.")

    return errors
