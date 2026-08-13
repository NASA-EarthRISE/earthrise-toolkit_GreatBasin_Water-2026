"""Strictly bounded official-table stage-to-area/storage interpolation."""

from __future__ import annotations

import bisect
import math
from dataclasses import asdict, dataclass
from typing import Any

from .capacity_tables import NormalizedCapacityTable
from .models import (
    CapabilityLevel,
    HeightType,
    HydrometryVariable,
    NormalizedTimeSeriesRecord,
    SourceQualityClass,
)
from .units import canonical_unit, convert_area, convert_length, convert_volume


class VerticalCompatibilityError(ValueError):
    """Raised when a stage cannot safely use a capacity table."""


class StageCapacityError(ValueError):
    """Raised when a stage value violates the interpolation contract."""


def _datum_token(datum: str | None) -> str | None:
    if datum is None or not str(datum).strip():
        return None
    token = "".join(
        character for character in str(datum).upper() if character.isalnum()
    )
    if token.startswith(("NAVD88", "NAVD1988")):
        return "NAVD88"
    if token.startswith(("NGVD29", "NGVD1929")):
        return "NGVD29"
    aliases = {
        "NAVD88": "NAVD88",
        "NAVD1988": "NAVD88",
        "NORTHAMERICANVERTICALDATUM1988": "NAVD88",
        "NGVD29": "NGVD29",
        "NGVD1929": "NGVD29",
        "NATIONALGEODETICVERTICALDATUM1929": "NGVD29",
    }
    return aliases.get(token, token)


@dataclass(frozen=True, slots=True)
class VerticalCompatibilityDecision:
    """Auditable decision made before using an official capacity relation."""

    compatible: bool
    status: str
    stage_vertical_datum: str | None
    table_vertical_datum: str | None
    stage_unit: str | None
    normalized_stage_unit: str
    height_type: HeightType
    stage_variable: HydrometryVariable
    vertical_transformation_applied: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["height_type"] = self.height_type.value
        result["stage_variable"] = self.stage_variable.value
        return result


def evaluate_vertical_compatibility(
    *,
    stage_vertical_datum: str | None,
    stage_unit: str | None,
    height_type: HeightType,
    stage_variable: HydrometryVariable,
    table_vertical_datum: str | None,
) -> VerticalCompatibilityDecision:
    """Evaluate the Bear Lake vertical-reference gate without transforming datum."""

    common = {
        "stage_vertical_datum": stage_vertical_datum,
        "table_vertical_datum": table_vertical_datum,
        "stage_unit": stage_unit,
        "normalized_stage_unit": "m",
        "height_type": height_type,
        "stage_variable": stage_variable,
        "vertical_transformation_applied": False,
    }
    if stage_variable is not HydrometryVariable.WATER_LEVEL:
        return VerticalCompatibilityDecision(
            compatible=False,
            status="blocked_not_absolute_water_elevation",
            reason="The source series is not an absolute water-elevation series.",
            **common,
        )
    if height_type not in {
        HeightType.ORTHOMETRIC_HEIGHT,
        HeightType.GEOID_REFERENCED_HEIGHT,
    }:
        return VerticalCompatibilityDecision(
            compatible=False,
            status="blocked_height_type",
            reason=f"Height type {height_type.value!r} is not an absolute height.",
            **common,
        )
    stage_datum = _datum_token(stage_vertical_datum)
    table_datum = _datum_token(table_vertical_datum)
    if stage_datum is None or table_datum is None:
        return VerticalCompatibilityDecision(
            compatible=False,
            status="blocked_unknown_datum",
            reason="Both the stage and capacity-table vertical datums are required.",
            **common,
        )
    if stage_datum != table_datum:
        return VerticalCompatibilityDecision(
            compatible=False,
            status="blocked_datum_mismatch",
            reason=(
                f"Stage datum {stage_vertical_datum!r} does not match table datum "
                f"{table_vertical_datum!r}."
            ),
            **common,
        )
    if stage_unit is None:
        return VerticalCompatibilityDecision(
            compatible=False,
            status="blocked_unknown_unit",
            reason="The source stage unit is required.",
            **common,
        )
    try:
        canonical = canonical_unit(stage_unit)
        convert_length(0.0, canonical, "m")
    except ValueError as exc:
        return VerticalCompatibilityDecision(
            compatible=False,
            status="blocked_incompatible_unit",
            reason=str(exc),
            **common,
        )
    return VerticalCompatibilityDecision(
        compatible=True,
        status="compatible_no_vertical_transformation",
        reason=(
            "Stage and capacity table share a documented vertical datum; only "
            "a physical unit conversion is required."
        ),
        **common,
    )


def require_vertical_compatibility(
    **kwargs: Any,
) -> VerticalCompatibilityDecision:
    """Return a passing vertical decision or raise with its exact blocking reason."""

    decision = evaluate_vertical_compatibility(**kwargs)
    if not decision.compatible:
        raise VerticalCompatibilityError(f"{decision.status}: {decision.reason}")
    return decision


@dataclass(frozen=True, slots=True)
class StageCapacityResult:
    """One bounded interpolation result with complete method provenance."""

    input_elevation: float | None
    input_unit: str
    input_vertical_datum: str
    normalized_elevation_m: float | None
    derived_area_m2: float | None
    derived_area_km2: float | None
    derived_area_acres: float | None
    derived_storage_m3: float | None
    derived_storage_acre_ft: float | None
    lower_bracket_elevation: float | None
    upper_bracket_elevation: float | None
    interpolation_fraction: float | None
    within_supported_range: bool
    range_status: str
    method: str
    source_quality_class: SourceQualityClass
    capability_level: CapabilityLevel
    capacity_table_source: str
    capacity_table_hash: str
    capacity_table_version: str
    d_area_dh_m2_per_m: float | None
    d_storage_dh_m3_per_m: float | None
    storage_change_per_0_01_m_m3: float | None
    warning_flags: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_quality_class"] = self.source_quality_class.value
        result["capability_level"] = int(self.capability_level)
        return result


class StageCapacityInterpolator:
    """Auditable piecewise-linear interpolation over a validated table."""

    method = "official_eav_piecewise_linear"

    def __init__(self, table: NormalizedCapacityTable) -> None:
        if not isinstance(table, NormalizedCapacityTable):
            raise StageCapacityError("table must be a NormalizedCapacityTable.")
        if table.validation.validation_status != "passed":
            raise StageCapacityError("Capacity-table validation has not passed.")
        self.table = table
        self._elevations = tuple(row.elevation_m_navd88 for row in table.rows)

    def _empty_result(
        self,
        *,
        input_elevation: float | None,
        input_unit: str,
        input_vertical_datum: str,
        normalized_elevation_m: float | None,
        range_status: str,
        warning: str,
    ) -> StageCapacityResult:
        metadata = self.table.metadata
        return StageCapacityResult(
            input_elevation=input_elevation,
            input_unit=canonical_unit(input_unit),
            input_vertical_datum=input_vertical_datum,
            normalized_elevation_m=normalized_elevation_m,
            derived_area_m2=None,
            derived_area_km2=None,
            derived_area_acres=None,
            derived_storage_m3=None,
            derived_storage_acre_ft=None,
            lower_bracket_elevation=None,
            upper_bracket_elevation=None,
            interpolation_fraction=None,
            within_supported_range=False,
            range_status=range_status,
            method=self.method,
            source_quality_class=SourceQualityClass.STAGE_OFFICIAL_CAPACITY,
            capability_level=CapabilityLevel.ABSOLUTE_STORAGE,
            capacity_table_source=metadata.source_doi,
            capacity_table_hash=metadata.source_hash,
            capacity_table_version=metadata.source_version,
            d_area_dh_m2_per_m=None,
            d_storage_dh_m3_per_m=None,
            storage_change_per_0_01_m_m3=None,
            warning_flags=(warning,),
        )

    def derive(
        self,
        input_elevation: float | None,
        *,
        input_unit: str,
        input_vertical_datum: str,
        height_type: HeightType = HeightType.ORTHOMETRIC_HEIGHT,
        stage_variable: HydrometryVariable = HydrometryVariable.WATER_LEVEL,
    ) -> StageCapacityResult:
        """Derive area and storage without extrapolating beyond official rows."""

        require_vertical_compatibility(
            stage_vertical_datum=input_vertical_datum,
            stage_unit=input_unit,
            height_type=height_type,
            stage_variable=stage_variable,
            table_vertical_datum=self.table.metadata.vertical_datum,
        )
        if input_elevation is None:
            return self._empty_result(
                input_elevation=None,
                input_unit=input_unit,
                input_vertical_datum=input_vertical_datum,
                normalized_elevation_m=None,
                range_status="null_input",
                warning="source_stage_is_null",
            )
        if isinstance(input_elevation, bool):
            raise StageCapacityError("input_elevation must be finite or null.")
        try:
            original = float(input_elevation)
        except (TypeError, ValueError) as exc:
            raise StageCapacityError("input_elevation must be finite or null.") from exc
        if not math.isfinite(original):
            raise StageCapacityError("input_elevation must be finite or null.")
        elevation_m = convert_length(original, input_unit, "m")
        if elevation_m < self._elevations[0]:
            return self._empty_result(
                input_elevation=original,
                input_unit=input_unit,
                input_vertical_datum=input_vertical_datum,
                normalized_elevation_m=elevation_m,
                range_status="below_supported_range",
                warning="no_extrapolation_below_table",
            )
        if elevation_m > self._elevations[-1]:
            return self._empty_result(
                input_elevation=original,
                input_unit=input_unit,
                input_vertical_datum=input_vertical_datum,
                normalized_elevation_m=elevation_m,
                range_status="above_supported_range",
                warning="no_extrapolation_above_table",
            )

        index = bisect.bisect_left(self._elevations, elevation_m)
        exact_node = index < len(self._elevations) and (
            self._elevations[index] == elevation_m
        )
        if exact_node:
            node = self.table.rows[index]
            area_m2 = node.area_m2
            storage_m3 = node.storage_m3
            # A piecewise-linear derivative is not unique at an interior knot.
            # Use the upper one-sided segment except at the maximum table node,
            # where the lower one-sided segment is the only supported choice.
            if index == len(self.table.rows) - 1:
                lower = self.table.rows[index - 1]
                upper = node
                fraction = 1.0
            else:
                lower = node
                upper = self.table.rows[index + 1]
                fraction = 0.0
            interval = upper.elevation_m_navd88 - lower.elevation_m_navd88
            area_slope = (upper.area_m2 - lower.area_m2) / interval
            storage_slope = (upper.storage_m3 - lower.storage_m3) / interval
            warnings = ("exact_table_node_uses_one_sided_sensitivity",)
        else:
            lower = self.table.rows[index - 1]
            upper = self.table.rows[index]
            interval = upper.elevation_m_navd88 - lower.elevation_m_navd88
            fraction = (elevation_m - lower.elevation_m_navd88) / interval
            area_slope = (upper.area_m2 - lower.area_m2) / interval
            storage_slope = (upper.storage_m3 - lower.storage_m3) / interval
            area_m2 = lower.area_m2 + fraction * (upper.area_m2 - lower.area_m2)
            storage_m3 = lower.storage_m3 + fraction * (
                upper.storage_m3 - lower.storage_m3
            )
            warnings = ()

        metadata = self.table.metadata
        return StageCapacityResult(
            input_elevation=original,
            input_unit=canonical_unit(input_unit),
            input_vertical_datum=input_vertical_datum,
            normalized_elevation_m=elevation_m,
            derived_area_m2=area_m2,
            derived_area_km2=convert_area(area_m2, "m2", "km2"),
            derived_area_acres=convert_area(area_m2, "m2", "acre"),
            derived_storage_m3=storage_m3,
            derived_storage_acre_ft=convert_volume(storage_m3, "m3", "acre_ft"),
            lower_bracket_elevation=lower.elevation_m_navd88,
            upper_bracket_elevation=upper.elevation_m_navd88,
            interpolation_fraction=fraction,
            within_supported_range=True,
            range_status="within_supported_range",
            method=self.method,
            source_quality_class=SourceQualityClass.STAGE_OFFICIAL_CAPACITY,
            capability_level=CapabilityLevel.ABSOLUTE_STORAGE,
            capacity_table_source=metadata.source_doi,
            capacity_table_hash=metadata.source_hash,
            capacity_table_version=metadata.source_version,
            d_area_dh_m2_per_m=area_slope,
            d_storage_dh_m3_per_m=storage_slope,
            storage_change_per_0_01_m_m3=(
                None if storage_slope is None else storage_slope * 0.01
            ),
            warning_flags=warnings,
        )

    def derive_record(self, record: NormalizedTimeSeriesRecord) -> StageCapacityResult:
        """Derive from a normalized source record after enforcing its metadata."""

        return self.derive(
            record.value,
            input_unit=record.unit,
            input_vertical_datum=(record.vertical_datum.source_vertical_datum or ""),
            height_type=record.height_type,
            stage_variable=record.variable,
        )


def interpolate_stage_capacity(
    table: NormalizedCapacityTable,
    input_elevation: float | None,
    **kwargs: Any,
) -> StageCapacityResult:
    """Convenience wrapper around :class:`StageCapacityInterpolator`."""

    return StageCapacityInterpolator(table).derive(input_elevation, **kwargs)


__all__ = [
    "StageCapacityError",
    "StageCapacityInterpolator",
    "StageCapacityResult",
    "VerticalCompatibilityDecision",
    "VerticalCompatibilityError",
    "evaluate_vertical_compatibility",
    "interpolate_stage_capacity",
    "require_vertical_compatibility",
]
