"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest

from platooning.config.settings import load_config, validate_config

# Path to the default config shipped with the repo
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_DEFAULT_CONFIG = _CONFIG_DIR / "config.yaml"


class TestConfigLoading:
    """Verify that configuration files load correctly."""

    def test_load_default_config(self) -> None:
        """The default config.yaml should load without errors."""
        config = load_config(_DEFAULT_CONFIG)
        assert isinstance(config, dict)
        assert "simulation" in config
        assert "communication" in config
        assert "platooning" in config
        assert "optimization" in config

    def test_load_missing_file_raises(self) -> None:
        """Loading a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_sumo_home_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SUMO_HOME from the environment should appear in the loaded config."""
        monkeypatch.setenv("SUMO_HOME", "/opt/sumo")
        config = load_config(_DEFAULT_CONFIG)
        assert config["simulation"].get("sumo_home") == "/opt/sumo"


class TestConfigValidation:
    """Verify that validate_config catches invalid configurations."""

    def test_valid_config_has_no_errors(self) -> None:
        """The shipped default config should pass validation."""
        config = load_config(_DEFAULT_CONFIG)
        errors = validate_config(config)
        assert errors == []

    def test_missing_simulation_section(self) -> None:
        """Omitting the simulation section should produce an error."""
        config = {
            "communication": {},
            "platooning": {"minimum_platoon_size": 2},
            "optimization": {"method": "classical"},
        }
        errors = validate_config(config)
        assert any("simulation" in e.lower() for e in errors)

    def test_invalid_simulator(self) -> None:
        """An unsupported simulator name should be rejected."""
        config = load_config(_DEFAULT_CONFIG)
        config["simulation"]["simulator"] = "carla"
        errors = validate_config(config)
        assert any("simulator" in e.lower() for e in errors)

    def test_negative_step_length(self) -> None:
        """A negative step_length should be rejected."""
        config = load_config(_DEFAULT_CONFIG)
        config["simulation"]["step_length"] = -1.0
        errors = validate_config(config)
        assert any("step_length" in e for e in errors)

    def test_invalid_packet_loss_rate(self) -> None:
        """packet_loss_rate outside [0, 1] should be rejected."""
        config = load_config(_DEFAULT_CONFIG)
        config["communication"]["packet_loss_rate"] = 1.5
        errors = validate_config(config)
        assert any("packet_loss_rate" in e for e in errors)

    def test_min_platoon_size_too_small(self) -> None:
        """minimum_platoon_size < 2 should be rejected."""
        config = load_config(_DEFAULT_CONFIG)
        config["platooning"]["minimum_platoon_size"] = 1
        errors = validate_config(config)
        assert any("minimum_platoon_size" in e for e in errors)

    def test_max_less_than_min(self) -> None:
        """maximum < minimum should be rejected."""
        config = load_config(_DEFAULT_CONFIG)
        config["platooning"]["minimum_platoon_size"] = 5
        config["platooning"]["maximum_platoon_size"] = 3
        errors = validate_config(config)
        assert any("maximum_platoon_size" in e for e in errors)

    def test_invalid_optimization_method(self) -> None:
        """An unknown optimization method should be rejected."""
        config = load_config(_DEFAULT_CONFIG)
        config["optimization"]["method"] = "magic"
        errors = validate_config(config)
        assert any("method" in e for e in errors)
