"""Exact unit conversions used by the hydrometry workflows.

This module converts physical units only.  In particular, converting a height
from feet to metres does not transform its vertical datum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class UnitConversionError(ValueError):
    """Raised when a value or unit cannot be converted safely."""


# International foot and derived exact conversion constants.
FOOT_TO_METRE = 0.3048
ACRE_TO_SQUARE_METRE = 4_046.856_422_4
SQUARE_KILOMETRE_TO_SQUARE_METRE = 1_000_000.0
ACRE_FOOT_TO_CUBIC_METRE = 1_233.481_837_547_52


_LENGTH_FACTORS = {"m": 1.0, "ft": FOOT_TO_METRE}
_AREA_FACTORS = {
    "m2": 1.0,
    "km2": SQUARE_KILOMETRE_TO_SQUARE_METRE,
    "acre": ACRE_TO_SQUARE_METRE,
}
_VOLUME_FACTORS = {"m3": 1.0, "acre_ft": ACRE_FOOT_TO_CUBIC_METRE}

_UNIT_ALIASES = {
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "ft": "ft",
    "foot": "ft",
    "feet": "ft",
    "m2": "m2",
    "m^2": "m2",
    "m²": "m2",
    "square_meter": "m2",
    "square_meters": "m2",
    "square_metre": "m2",
    "square_metres": "m2",
    "km2": "km2",
    "km^2": "km2",
    "km²": "km2",
    "square_kilometer": "km2",
    "square_kilometers": "km2",
    "square_kilometre": "km2",
    "square_kilometres": "km2",
    "acre": "acre",
    "acres": "acre",
    "m3": "m3",
    "m^3": "m3",
    "m³": "m3",
    "cubic_meter": "m3",
    "cubic_meters": "m3",
    "cubic_metre": "m3",
    "cubic_metres": "m3",
    "af": "acre_ft",
    "acre-ft": "acre_ft",
    "acre_ft": "acre_ft",
    "acre_foot": "acre_ft",
    "acre_feet": "acre_ft",
    "acre-foot": "acre_ft",
    "acre-feet": "acre_ft",
}


def canonical_unit(unit: str) -> str:
    """Return a canonical unit token or raise for an unsupported unit."""

    if not isinstance(unit, str) or not unit.strip():
        raise UnitConversionError("unit must be a non-empty string.")
    token = unit.strip().casefold().replace(" ", "_")
    try:
        return _UNIT_ALIASES[token]
    except KeyError as exc:
        raise UnitConversionError(f"Unsupported hydrometry unit {unit!r}.") from exc


def _finite(value: float, label: str = "value") -> float:
    if isinstance(value, bool):
        raise UnitConversionError(f"{label} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise UnitConversionError(f"{label} must be a finite number.") from exc
    if not math.isfinite(result):
        raise UnitConversionError(f"{label} must be a finite number.")
    return result


def _convert(
    value: float,
    from_unit: str,
    to_unit: str,
    factors: dict[str, float],
    dimension: str,
) -> float:
    source = canonical_unit(from_unit)
    target = canonical_unit(to_unit)
    if source not in factors or target not in factors:
        raise UnitConversionError(
            f"Cannot convert {dimension} from {from_unit!r} to {to_unit!r}."
        )
    return _finite(value) * factors[source] / factors[target]


def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a finite length or elevation value."""

    return _convert(value, from_unit, to_unit, _LENGTH_FACTORS, "length")


def convert_area(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a finite surface-area value."""

    return _convert(value, from_unit, to_unit, _AREA_FACTORS, "area")


def convert_volume(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a finite storage-volume value."""

    return _convert(value, from_unit, to_unit, _VOLUME_FACTORS, "volume")


@dataclass(frozen=True, slots=True)
class ConvertedVerticalValue:
    """A unit-converted height with its datum explicitly unchanged."""

    value: float
    unit: str
    vertical_datum: str
    datum_transformation_applied: bool = False


def convert_vertical_value(
    value: float,
    *,
    from_unit: str,
    to_unit: str,
    vertical_datum: str,
) -> ConvertedVerticalValue:
    """Convert height units while retaining, not transforming, the datum."""

    if not isinstance(vertical_datum, str) or not vertical_datum.strip():
        raise UnitConversionError("vertical_datum must be explicitly documented.")
    return ConvertedVerticalValue(
        value=convert_length(value, from_unit, to_unit),
        unit=canonical_unit(to_unit),
        vertical_datum=vertical_datum.strip(),
    )


__all__ = [
    "ACRE_FOOT_TO_CUBIC_METRE",
    "ACRE_TO_SQUARE_METRE",
    "FOOT_TO_METRE",
    "SQUARE_KILOMETRE_TO_SQUARE_METRE",
    "ConvertedVerticalValue",
    "UnitConversionError",
    "canonical_unit",
    "convert_area",
    "convert_length",
    "convert_vertical_value",
    "convert_volume",
]
