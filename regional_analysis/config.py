"""Load and validate regional candidate-discovery configuration files."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a regional-analysis configuration is missing or malformed."""


@dataclass(frozen=True)
class EarthEngineConfig:
    """Earth Engine asset identifiers used by candidate discovery."""

    hydrolakes_asset: str
    states_asset: str


@dataclass(frozen=True)
class RegionConfig:
    """State abbreviations that define the regional domain."""

    states: tuple[str, ...]


@dataclass(frozen=True)
class SelectionConfig:
    """Experimental screening settings for the candidate inventory."""

    min_area_km2: float


@dataclass(frozen=True)
class DiscoveryConfig:
    """Validated configuration for a regional HydroLAKES discovery run."""

    earth_engine: EarthEngineConfig
    region: RegionConfig
    selection: SelectionConfig
    source_path: Path | None = None

    def with_min_area_km2(self, min_area_km2: float) -> DiscoveryConfig:
        """Return a copy with a validated one-run area override."""

        value = _positive_number(min_area_km2, "selection.min_area_km2")
        return replace(self, selection=SelectionConfig(value))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a YAML mapping.")
    return value


def _required_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name not in config:
        raise ConfigError(f"Missing required configuration section: {name}.")
    return _mapping(config[name], f"Configuration section '{name}'")


def _nonempty_string(section: Mapping[str, Any], key: str, label: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string.")
    return value.strip()


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a positive number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ConfigError(f"{label} must be greater than zero.")
    return number


def validate_config(config: Any, *, source_path: Path | None = None) -> DiscoveryConfig:
    """Validate parsed YAML data and return an immutable configuration."""

    root = _mapping(config, "Configuration root")
    earth_engine = _required_section(root, "earth_engine")
    region = _required_section(root, "region")
    selection = _required_section(root, "selection")

    hydrolakes_asset = _nonempty_string(
        earth_engine,
        "hydrolakes_asset",
        "earth_engine.hydrolakes_asset",
    )
    states_asset = _nonempty_string(
        earth_engine,
        "states_asset",
        "earth_engine.states_asset",
    )

    raw_states = region.get("states")
    if not isinstance(raw_states, list) or not raw_states:
        raise ConfigError("region.states must be a non-empty YAML list.")

    states: list[str] = []
    for index, state in enumerate(raw_states):
        if not isinstance(state, str) or not state.strip():
            raise ConfigError(
                f"region.states[{index}] must be a non-empty state abbreviation."
            )
        abbreviation = state.strip().upper()
        if len(abbreviation) != 2 or not abbreviation.isalpha():
            raise ConfigError(
                f"region.states[{index}] must be a two-letter state abbreviation."
            )
        states.append(abbreviation)

    duplicates = sorted({state for state in states if states.count(state) > 1})
    if duplicates:
        raise ConfigError(
            "region.states contains duplicate abbreviations: "
            + ", ".join(duplicates)
            + "."
        )

    if "min_area_km2" not in selection:
        raise ConfigError("Missing required setting: selection.min_area_km2.")
    min_area_km2 = _positive_number(
        selection["min_area_km2"],
        "selection.min_area_km2",
    )

    return DiscoveryConfig(
        earth_engine=EarthEngineConfig(
            hydrolakes_asset=hydrolakes_asset,
            states_asset=states_asset,
        ),
        region=RegionConfig(states=tuple(states)),
        selection=SelectionConfig(min_area_km2=min_area_km2),
        source_path=source_path,
    )


def load_config(path: str | Path) -> DiscoveryConfig:
    """Load a YAML file without importing or initializing Earth Engine."""

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"Could not read configuration '{config_path}': {exc}"
        ) from exc

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in '{config_path}': {exc}") from exc

    return validate_config(parsed, source_path=config_path)
