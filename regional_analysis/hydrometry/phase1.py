"""Operational Phase 1 hydrometry products for the five-pilot registry.

This module deliberately has two narrow scientific pathways:

* authoritative agency-reported elevation, area, and total storage from RISE;
* Bear Lake area/storage derived from its official NAVD88 elevation-area-volume
  table with bounded piecewise-linear interpolation.

Great Salt Lake and Lake Abert remain level-only.  No Earth Engine or QGIS
code is imported here, and no reservoir storage is reconstructed from stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .adapters.usbr_rise import (
    RiseAdapter,
    RiseCatalogItem,
    RiseLocation,
    RiseObservation,
)
from .adapters.usgs_waterdata import (
    SeriesSelectionStatus,
    UsgsWaterDataAdapter,
    select_water_level_series,
)
from .cache import CacheMode, CacheStore
from .capacity_tables import NormalizedCapacityTable
from .http import HttpClient
from .manager_metrics import (
    ChangeMetricResult,
    MetricObservation,
    SeasonalContextResult,
    calculate_change_metrics,
    calculate_freshness,
    calculate_seasonal_context,
)
from .models import HeightType, HydrometryVariable
from .registry import HydrometryProfile, HydrometryRegistry, SourceSpec, load_registry
from .stage_capacity import StageCapacityInterpolator, evaluate_vertical_compatibility
from .units import canonical_unit

PHASE1_SCHEMA_VERSION = "hydrometry-phase1-daily-v1"
MANAGER_SCHEMA_VERSION = "hydrometry-manager-summary-v1"
PROVENANCE_SCHEMA_VERSION = "hydrometry-phase1-provenance-v1"
DEFAULT_CONFIG = Path("configs/hydrometry_pilots.yaml")
DEFAULT_OUTPUT_DIR = Path("data/hydrometry/phase1")
DEFAULT_CACHE_DIR = Path("data/hydrometry/cache")
DEFAULT_REFERENCE_DIR = Path("data/hydrometry/reference")
DEFAULT_FIGURE_DIR = Path("outputs/hydrometry_phase1")
SUPPORTED_PROFILE_IDS = (
    "great_salt",
    "lake_abert",
    "bear_lake",
    "lake_powell",
    "flaming_gorge",
)
SUPPORTED_METRICS = ("water_elevation", "surface_area", "total_storage")

DAILY_COLUMNS = (
    "schema_version",
    "profile_id",
    "subprofile_id",
    "Hylak_id",
    "display_name",
    "state",
    "timestamp_utc",
    "calendar_date",
    "metric",
    "value",
    "unit",
    "value_original",
    "unit_original",
    "vertical_datum",
    "height_type",
    "method",
    "observation_type",
    "source_quality_class",
    "capability_level",
    "provider",
    "source_record_id",
    "source_location_id",
    "source_item_id",
    "source_parameter_code",
    "quality_status",
    "approval_status",
    "provisional",
    "within_supported_range",
    "availability_status",
    "freshness_status",
    "uncertainty_status",
    "uncertainty_lower",
    "uncertainty_upper",
    "uncertainty_unit",
    "capacity_table_source",
    "capacity_table_hash",
    "capacity_table_version",
    "retrieved_at_utc",
    "warning_flags",
    "source_observation_timestamp_utc",
    "lower_bracket_elevation_m",
    "upper_bracket_elevation_m",
    "interpolation_fraction",
    "d_area_dh_m2_per_m",
    "d_storage_dh_m3_per_m",
    "storage_change_per_0_01_m_m3",
)

CURRENT_COLUMNS = (
    "schema_version",
    "profile_id",
    "subprofile_id",
    "Hylak_id",
    "display_name",
    "metric",
    "latest_date",
    "latest_value",
    "unit",
    "method",
    "provider",
    "source_record_id",
    "source_location_id",
    "source_item_id",
    "quality_status",
    "approval_status",
    "provisional",
    "freshness_status",
    "days_since_latest_observation",
    "source_quality_class",
    "capability_level",
    "availability_status",
    "warning",
)

CHANGE_COLUMNS = (
    "schema_version",
    "profile_id",
    "subprofile_id",
    "display_name",
    "metric",
    "unit",
    "method",
    "latest_date",
    "latest_value",
    "baseline_target_date",
    "baseline_actual_date",
    "baseline_value",
    "requested_window_days",
    "actual_elapsed_days",
    "absolute_change",
    "percent_change",
    "comparison_status",
)

SEASONAL_COLUMNS = (
    "schema_version",
    "profile_id",
    "subprofile_id",
    "display_name",
    "metric",
    "unit",
    "method",
    "latest_date",
    "latest_value",
    "seasonal_percentile",
    "seasonal_median",
    "seasonal_q25",
    "seasonal_q75",
    "seasonal_minimum",
    "seasonal_maximum",
    "observation_count",
    "distinct_year_count",
    "window_days",
    "seasonal_status",
)


class Phase1Error(RuntimeError):
    """Raised when Phase 1 cannot safely produce its scientific outputs."""


@dataclass(frozen=True, slots=True)
class Phase1Settings:
    """Validated pinned settings read from the registry YAML."""

    history_start: date
    analysis_end: date
    change_windows_days: tuple[int, ...]
    baseline_match_tolerance_days: int
    seasonal_window_days: int
    seasonal_min_distinct_years: int
    seasonal_min_observations: int
    freshness_warning_days: int
    interpolation_method: str
    allow_extrapolation: bool

    def __post_init__(self) -> None:
        if self.history_start > self.analysis_end:
            raise Phase1Error("phase1 history_start must be on or before analysis_end")
        if not self.change_windows_days or any(
            isinstance(value, bool) or value <= 0 for value in self.change_windows_days
        ):
            raise Phase1Error("phase1 change windows must be positive integers")
        if len(set(self.change_windows_days)) != len(self.change_windows_days):
            raise Phase1Error("phase1 change windows must be unique")
        for name in (
            "baseline_match_tolerance_days",
            "seasonal_window_days",
            "seasonal_min_distinct_years",
            "seasonal_min_observations",
            "freshness_warning_days",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise Phase1Error(f"phase1 {name} must be a nonnegative integer")
        if self.interpolation_method != "piecewise_linear":
            raise Phase1Error("Phase 1 supports only piecewise_linear interpolation")
        if self.allow_extrapolation:
            raise Phase1Error("Phase 1 forbids capacity-table extrapolation")


@dataclass(slots=True)
class Phase1Adapters:
    """Injectable provider group for production and credential-free tests."""

    usgs: Any
    rise: Any
    client: Any


@dataclass(slots=True)
class Phase1Bundle:
    """All deterministic local outputs from one Phase 1 run."""

    daily: pd.DataFrame
    current_conditions: pd.DataFrame
    change_metrics: pd.DataFrame
    seasonal_context: pd.DataFrame
    manager_summary: dict[str, Any]
    provenance: dict[str, Any]
    capacity_table: NormalizedCapacityTable
    capacity_validation: pd.DataFrame
    area_volume_diagnostic: pd.DataFrame
    reference_manifest: Mapping[str, Any]
    request_count: int
    elapsed_seconds: float


def _public_manifest(value: Any) -> dict[str, Any]:
    if hasattr(value, "as_dict"):
        result = value.as_dict()
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise Phase1Error("Bear reference manifest is not serializable")
    return result


def _date(value: object, label: str) -> date:
    try:
        parsed = date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise Phase1Error(f"{label} must be YYYY-MM-DD") from exc
    return parsed


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise Phase1Error(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Phase1Error(f"{label} must be an integer") from exc
    if parsed < (0 if allow_zero else 1):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise Phase1Error(f"{label} must be {qualifier}")
    return parsed


def load_phase1_settings(path: str | Path) -> Phase1Settings:
    """Load the single Phase 1 settings block from the authoritative registry."""

    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Phase1Error(f"Unable to load Phase 1 config {source}: {exc}") from exc
    if not isinstance(raw, Mapping) or not isinstance(raw.get("phase1"), Mapping):
        raise Phase1Error(f"{source} has no phase1 mapping")
    phase1 = raw["phase1"]
    interpolation = phase1.get("capacity_interpolation")
    if not isinstance(interpolation, Mapping):
        raise Phase1Error("phase1.capacity_interpolation must be a mapping")
    windows = phase1.get("change_windows_days")
    if not isinstance(windows, list):
        raise Phase1Error("phase1.change_windows_days must be a list")
    return Phase1Settings(
        history_start=_date(phase1.get("history_start"), "phase1.history_start"),
        analysis_end=_date(phase1.get("analysis_end"), "phase1.analysis_end"),
        change_windows_days=tuple(
            _positive_int(value, "phase1.change_windows_days") for value in windows
        ),
        baseline_match_tolerance_days=_positive_int(
            phase1.get("baseline_match_tolerance_days"),
            "phase1.baseline_match_tolerance_days",
            allow_zero=True,
        ),
        seasonal_window_days=_positive_int(
            phase1.get("seasonal_window_days"),
            "phase1.seasonal_window_days",
            allow_zero=True,
        ),
        seasonal_min_distinct_years=_positive_int(
            phase1.get("seasonal_min_distinct_years"),
            "phase1.seasonal_min_distinct_years",
        ),
        seasonal_min_observations=_positive_int(
            phase1.get("seasonal_min_observations"),
            "phase1.seasonal_min_observations",
        ),
        freshness_warning_days=_positive_int(
            phase1.get("freshness_warning_days"),
            "phase1.freshness_warning_days",
            allow_zero=True,
        ),
        interpolation_method=str(interpolation.get("method") or ""),
        allow_extrapolation=interpolation.get("allow_extrapolation") is True,
    )


def build_phase1_adapters(
    *, mode: CacheMode, cache_dir: Path, refresh: bool
) -> Phase1Adapters:
    """Build providers over the shared bounded Phase 0 HTTP/cache layer."""

    client = HttpClient(timeout=30.0, max_retries=3)
    cache = CacheStore(cache_dir)
    common = {"client": client, "mode": mode, "cache": cache, "refresh": refresh}
    return Phase1Adapters(
        usgs=UsgsWaterDataAdapter(**common),
        rise=RiseAdapter(**common),
        client=client,
    )


def _iso(value: datetime | date | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return value.isoformat()


def _json_list(values: Sequence[str] | None) -> str:
    return json.dumps(sorted(set(values or ())), separators=(",", ":"))


def _profile_map(registry: HydrometryRegistry) -> dict[str, HydrometryProfile]:
    mapping = {profile.profile_id: profile for profile in registry.profiles}
    if set(mapping) != set(SUPPORTED_PROFILE_IDS):
        raise Phase1Error(
            "Phase 1 requires exactly the five Phase 0 pilot profiles: "
            + ", ".join(SUPPORTED_PROFILE_IDS)
        )
    return mapping


def _selected_profiles(
    registry: HydrometryRegistry, profile_ids: Sequence[str] | None
) -> list[HydrometryProfile]:
    mapping = _profile_map(registry)
    if not profile_ids:
        return [mapping[profile_id] for profile_id in SUPPORTED_PROFILE_IDS]
    if len(set(profile_ids)) != len(profile_ids):
        raise Phase1Error("--profiles contains duplicate profile IDs")
    unknown = sorted(set(profile_ids) - set(mapping))
    if unknown:
        raise Phase1Error("Unknown profile ID(s): " + ", ".join(unknown))
    requested = set(profile_ids)
    return [
        mapping[profile_id]
        for profile_id in SUPPORTED_PROFILE_IDS
        if profile_id in requested
    ]


def _source_subprofile(
    profile: HydrometryProfile, source: SourceSpec
) -> tuple[str, str]:
    if profile.profile_id != "great_salt":
        return profile.profile_id, profile.display_name
    arm = str(source.metadata.get("arm") or "").casefold()
    if arm not in {"north", "south"}:
        raise Phase1Error(
            "Great Salt source is missing explicit north/south arm metadata"
        )
    return f"great_salt_{arm}_arm", f"Great Salt {arm} arm"


def _quality_summary(values: Sequence[str | None]) -> str:
    unique = sorted(
        {
            str(value)
            for value in values
            if value is not None and not pd.isna(value) and str(value).strip()
        }
    )
    return "mixed" if len(unique) > 1 else unique[0] if unique else "unknown"


def _provisional_summary(values: Sequence[bool | None]) -> bool | None:
    known = {
        bool(value) for value in values if value is not None and not pd.isna(value)
    }
    return next(iter(known)) if len(known) == 1 else None


def _latest_date(values: Sequence[datetime]) -> date | None:
    return max((value.date() for value in values), default=None)


def _freshness(latest: date | None, settings: Phase1Settings) -> str:
    return calculate_freshness(
        latest,
        analysis_end=settings.analysis_end,
        freshness_warning_days=settings.freshness_warning_days,
    ).freshness_status


def _daily_base(
    profile: HydrometryProfile,
    *,
    subprofile_id: str,
    display_name: str,
    timestamp: datetime,
    metric: str,
    value: float | None,
    unit: str,
    value_original: float | None,
    unit_original: str,
    vertical_datum: str | None,
    height_type: str,
    method: str,
    observation_type: str,
    source_quality_class: str | None,
    capability_level: int,
    provider: str,
    source_record_id: str | None,
    source_location_id: str | None,
    source_item_id: str | None,
    source_parameter_code: str | None,
    quality_status: str,
    approval_status: str | None,
    provisional: bool | None,
    within_supported_range: bool | None,
    availability_status: str,
    freshness: str,
    retrieved_at: datetime,
    warning_flags: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "schema_version": PHASE1_SCHEMA_VERSION,
        "profile_id": profile.profile_id,
        "subprofile_id": subprofile_id,
        "Hylak_id": profile.Hylak_id,
        "display_name": display_name,
        "state": profile.state,
        "timestamp_utc": _iso(timestamp),
        "calendar_date": timestamp.date().isoformat(),
        "metric": metric,
        "value": value,
        "unit": unit,
        "value_original": value_original,
        "unit_original": unit_original,
        "vertical_datum": vertical_datum,
        "height_type": height_type,
        "method": method,
        "observation_type": observation_type,
        "source_quality_class": source_quality_class,
        "capability_level": capability_level,
        "provider": provider,
        "source_record_id": source_record_id,
        "source_location_id": source_location_id,
        "source_item_id": source_item_id,
        "source_parameter_code": source_parameter_code,
        "quality_status": quality_status,
        "approval_status": approval_status,
        "provisional": provisional,
        "within_supported_range": within_supported_range,
        "availability_status": availability_status,
        "freshness_status": freshness,
        "uncertainty_status": "not_quantified",
        "uncertainty_lower": None,
        "uncertainty_upper": None,
        "uncertainty_unit": unit,
        "capacity_table_source": None,
        "capacity_table_hash": None,
        "capacity_table_version": None,
        "retrieved_at_utc": _iso(retrieved_at),
        "warning_flags": _json_list(warning_flags),
        "source_observation_timestamp_utc": _iso(timestamp),
        "lower_bracket_elevation_m": None,
        "upper_bracket_elevation_m": None,
        "interpolation_fraction": None,
        "d_area_dh_m2_per_m": None,
        "d_storage_dh_m3_per_m": None,
        "storage_change_per_0_01_m_m3": None,
    }


def _fetch_usgs_level_rows(
    adapter: Any,
    profile: HydrometryProfile,
    source: SourceSpec,
    *,
    start: date,
    end: date,
    settings: Phase1Settings,
    retrieved_at: datetime,
) -> tuple[list[dict[str, Any]], dict[str, Any], Any, Any]:
    location = adapter.fetch_monitoring_location(source.source_record_id)
    series_all = adapter.discover_series(source.source_record_id)
    selection = select_water_level_series(
        series_all,
        expected_unit=source.expected_unit,
        expected_vertical_datum=source.expected_vertical_datum,
        prefer_daily=True,
    )
    if selection.status is not SeriesSelectionStatus.SELECTED or not selection.selected:
        raise Phase1Error(f"{source.source_id}: {selection.reason}")
    series = selection.selected
    parameter = adapter.fetch_parameter_metadata(series.parameter_code)
    if parameter.unit.casefold() != series.unit.casefold():
        raise Phase1Error(
            f"{source.source_id}: parameter unit {parameter.unit!r} does not match "
            f"series unit {series.unit!r}"
        )
    if series.height_type is not HeightType.ORTHOMETRIC_HEIGHT:
        raise Phase1Error(
            f"{source.source_id}: selected series is not absolute elevation"
        )
    observations = adapter.fetch_observations(
        series,
        start=start,
        end=end,
        max_pages=20,
        max_observations=20_000,
    )
    subprofile_id, display_name = _source_subprofile(profile, source)
    latest = _latest_date([observation.timestamp_utc for observation in observations])
    rows = [
        _daily_base(
            profile,
            subprofile_id=subprofile_id,
            display_name=display_name,
            timestamp=observation.timestamp_utc,
            metric="water_elevation",
            value=observation.value,
            unit=observation.unit,
            value_original=observation.value,
            unit_original=observation.unit,
            vertical_datum=series.vertical_datum or location.vertical_datum,
            height_type=series.height_type.value,
            method="agency_observed",
            observation_type="observed",
            source_quality_class=None,
            capability_level=2,
            provider="usgs_waterdata",
            source_record_id=source.source_record_id,
            source_location_id=location.location_id,
            source_item_id=series.time_series_id,
            source_parameter_code=observation.parameter_code,
            quality_status=observation.qualifier or "provider_quality_not_reported",
            approval_status=observation.approval_status,
            provisional=observation.provisional,
            within_supported_range=None,
            availability_status="available",
            freshness=_freshness(latest, settings),
            retrieved_at=retrieved_at,
        )
        for observation in observations
    ]
    metadata = {
        "profile_id": profile.profile_id,
        "subprofile_id": subprofile_id,
        "source_id": source.source_id,
        "provider": "usgs_waterdata",
        "source_record_id": source.source_record_id,
        "source_location_id": location.location_id,
        "source_item_id": series.time_series_id,
        "parameter_code": series.parameter_code,
        "statistic_code": series.statistic_code,
        "unit": series.unit,
        "vertical_datum": series.vertical_datum or location.vertical_datum,
        "height_type": series.height_type.value,
        "row_count": len(rows),
        "temporal_start": _iso(
            min((item.timestamp_utc for item in observations), default=None)
        ),
        "temporal_end": _iso(
            max((item.timestamp_utc for item in observations), default=None)
        ),
        "approval_status": _quality_summary(
            [item.approval_status for item in observations]
        ),
        "provisional": _provisional_summary(
            [item.provisional for item in observations]
        ),
        "source_url_public": series.source_url_public,
    }
    return rows, metadata, series, location


def _rise_source_id(item: RiseCatalogItem) -> str:
    variable = item.variable.value if item.variable else "unclassified"
    return f"rise_record_{item.record_id}:{variable}:{item.item_id}"


def _single_rise_item(
    inventory: Any, variable: HydrometryVariable, *, profile_id: str
) -> RiseCatalogItem:
    ids = tuple(inventory.candidates_by_variable.get(variable, ()))
    if len(ids) != 1:
        raise Phase1Error(
            f"{profile_id}: expected exactly one validated RISE {variable.value} item; "
            f"found {list(ids)}"
        )
    return next(item for item in inventory.items if item.item_id == ids[0])


def _validate_direct_item(item: RiseCatalogItem, variable: HydrometryVariable) -> None:
    expected_units = {
        HydrometryVariable.RESERVOIR_ELEVATION: {"ft"},
        HydrometryVariable.SURFACE_AREA: {"acres"},
        HydrometryVariable.STORAGE: {"af"},
    }[variable]
    if item.variable is not variable:
        raise Phase1Error(f"RISE item {item.item_id} is not {variable.value}")
    if item.unit not in expected_units:
        raise Phase1Error(
            f"RISE item {item.item_id} has unexpected {variable.value} unit {item.unit!r}"
        )
    if item.validation_issues:
        raise Phase1Error(
            f"RISE item {item.item_id} requires review: {', '.join(item.validation_issues)}"
        )


def _validate_item_against_source(item: RiseCatalogItem, source: SourceSpec) -> None:
    """Revalidate a configured RISE identity without using it as a fallback."""

    if item.variable is not source.variable:
        raise Phase1Error(
            f"RISE item {item.item_id} variable {item.variable} does not match "
            f"configured {source.variable.value}"
        )
    if source.expected_unit is not None and (
        item.unit is None
        or canonical_unit(item.unit) != canonical_unit(source.expected_unit)
    ):
        raise Phase1Error(
            f"RISE item {item.item_id} unit {item.unit!r} does not match "
            f"configured {source.expected_unit!r}"
        )
    expected_title = str(source.metadata.get("expected_title_contains") or "").strip()
    if expected_title and expected_title.casefold() not in item.title.casefold():
        raise Phase1Error(
            f"RISE item {item.item_id} title does not contain {expected_title!r}"
        )
    expected_frequency = str(
        source.metadata.get("expected_update_frequency") or ""
    ).strip()
    if expected_frequency and (
        not item.update_frequency
        or expected_frequency.casefold() not in item.update_frequency.casefold()
    ):
        raise Phase1Error(
            f"RISE item {item.item_id} update frequency {item.update_frequency!r} "
            f"does not match {expected_frequency!r}"
        )


def _rise_daily_rows(
    profile: HydrometryProfile,
    item: RiseCatalogItem,
    location: RiseLocation,
    observations: Sequence[RiseObservation],
    *,
    settings: Phase1Settings,
    retrieved_at: datetime,
) -> list[dict[str, Any]]:
    metric = {
        HydrometryVariable.RESERVOIR_ELEVATION: "water_elevation",
        HydrometryVariable.SURFACE_AREA: "surface_area",
        HydrometryVariable.STORAGE: "total_storage",
    }.get(item.variable)
    if metric is None:
        raise Phase1Error(f"RISE item {item.item_id} is not a manager metric")
    latest = _latest_date(
        [
            observation.timestamp_utc
            for observation in observations
            if observation.value is not None
        ]
    )
    output_unit = "acre-ft" if item.unit == "af" else item.unit or ""
    return [
        _daily_base(
            profile,
            subprofile_id=profile.profile_id,
            display_name=profile.display_name,
            timestamp=observation.timestamp_utc,
            metric=metric,
            value=observation.value,
            unit=output_unit,
            value_original=observation.value,
            unit_original=item.unit or "",
            vertical_datum=item.vertical_datum,
            height_type=(
                HeightType.PROJECT_DATUM_ELEVATION.value
                if item.variable is HydrometryVariable.RESERVOIR_ELEVATION
                else HeightType.UNKNOWN.value
            ),
            method="agency_reported",
            observation_type="reported",
            source_quality_class="A",
            capability_level=4,
            provider="usbr_rise",
            source_record_id=str(item.record_id),
            source_location_id=str(location.location_id),
            source_item_id=str(item.item_id),
            source_parameter_code=str(item.parameter_id) if item.parameter_id else None,
            quality_status=observation.status or "provider_status_not_reported",
            approval_status=None,
            provisional=item.provisional,
            within_supported_range=None,
            availability_status=(
                "available" if observation.value is not None else "source_value_missing"
            ),
            freshness=_freshness(latest, settings),
            retrieved_at=retrieved_at,
            warning_flags=(
                (
                    "operational_elevation_datum_not_documented_for_capacity_recomputation",
                )
                if profile.profile_id == "lake_powell" and metric == "water_elevation"
                else ()
            ),
        )
        for observation in observations
    ]


def _fetch_rise_rows(
    adapter: Any,
    profile: HydrometryProfile,
    *,
    start: date,
    end: date,
    settings: Phase1Settings,
    retrieved_at: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    record_ids = {int(source.source_record_id) for source in profile.sources}
    if len(record_ids) != 1:
        raise Phase1Error(f"{profile.profile_id}: expected one RISE record")
    record_id = next(iter(record_ids))
    record = adapter.fetch_record(
        record_id, expected_title_contains=profile.display_name
    )
    location = adapter.fetch_location(
        record.location_id,
        expected_name_contains=profile.display_name,
        expected_record_id=record.record_id,
    )
    inventory = adapter.inventory_record_items(
        record,
        expected_location_name=profile.display_name,
        target_variables=(
            HydrometryVariable.RESERVOIR_ELEVATION,
            HydrometryVariable.SURFACE_AREA,
            HydrometryVariable.STORAGE,
        ),
    )
    items = [
        _single_rise_item(inventory, variable, profile_id=profile.profile_id)
        for variable in (
            HydrometryVariable.RESERVOIR_ELEVATION,
            HydrometryVariable.SURFACE_AREA,
            HydrometryVariable.STORAGE,
        )
    ]
    configured = {
        source.variable: source
        for source in profile.sources
        if source.source_item_id is not None
        and source.variable
        in {
            HydrometryVariable.RESERVOIR_ELEVATION,
            HydrometryVariable.SURFACE_AREA,
            HydrometryVariable.STORAGE,
        }
    }
    if set(configured) != {
        HydrometryVariable.RESERVOIR_ELEVATION,
        HydrometryVariable.SURFACE_AREA,
        HydrometryVariable.STORAGE,
    }:
        raise Phase1Error(
            f"{profile.profile_id}: Phase 1 requires configured elevation, area, "
            "and total-storage item identities"
        )
    # Discovery remains semantic; configured item IDs are assertions against
    # the Phase 0 validated source registry and are never fallback choices.
    for item in items:
        if item.variable is None:
            raise Phase1Error(f"RISE item {item.item_id} has no validated variable")
        _validate_direct_item(item, item.variable)
        source = configured[item.variable]
        _validate_item_against_source(item, source)
        if item.item_id != int(source.source_item_id or ""):
            raise Phase1Error(
                f"{profile.profile_id}: discovered {item.variable.value} item "
                f"{item.item_id}, expected configured item {source.source_item_id}"
            )
    if profile.profile_id == "flaming_gorge":
        bank_source = next(
            source
            for source in profile.sources
            if source.variable is HydrometryVariable.BANK_STORAGE
        )
        bank = next(
            (
                item
                for item in inventory.items
                if item.item_id == int(bank_source.source_item_id or "")
            ),
            None,
        )
        if bank is None or bank.variable is not HydrometryVariable.BANK_STORAGE:
            raise Phase1Error(
                "Flaming Gorge configured bank-storage item was not retained "
                "as bank storage"
            )
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for item in items:
        observations = adapter.fetch_timeseries(
            item, start=start, end=end, page_size=1_000, max_pages=100
        )
        timestamps = [value.timestamp_utc for value in observations]
        if len(timestamps) != len(set(timestamps)):
            raise Phase1Error(f"RISE item {item.item_id} returned duplicate timestamps")
        observations = tuple(
            sorted(observations, key=lambda value: value.timestamp_utc)
        )
        timestamps = [value.timestamp_utc for value in observations]
        rows.extend(
            _rise_daily_rows(
                profile,
                item,
                location,
                observations,
                settings=settings,
                retrieved_at=retrieved_at,
            )
        )
        sources.append(
            {
                "profile_id": profile.profile_id,
                "source_id": _rise_source_id(item),
                "provider": "usbr_rise",
                "source_record_id": str(record.record_id),
                "source_location_id": str(location.location_id),
                "source_item_id": str(item.item_id),
                "title": item.title,
                "variable": item.variable.value,
                "unit": item.unit,
                "row_count": len(observations),
                "temporal_start": _iso(min(timestamps, default=None)),
                "temporal_end": _iso(max(timestamps, default=None)),
                "provisional": item.provisional,
                "source_url_public": item.source_url_public,
            }
        )
    return rows, sources


def _bear_derived_rows(
    level_rows: Sequence[dict[str, Any]],
    profile: HydrometryProfile,
    *,
    table: NormalizedCapacityTable,
    settings: Phase1Settings,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    gate = evaluate_vertical_compatibility(
        height_type=HeightType.ORTHOMETRIC_HEIGHT,
        stage_vertical_datum="NAVD88",
        stage_unit="ft",
        table_vertical_datum=table.metadata.vertical_datum,
        stage_variable=HydrometryVariable.WATER_LEVEL,
    )
    if not gate.compatible:
        raise Phase1Error(f"Bear Lake vertical-reference gate failed: {gate.reason}")
    interpolator = StageCapacityInterpolator(table)
    derived: list[dict[str, Any]] = []
    counts = defaultdict(int)
    for level in level_rows:
        if level["metric"] != "water_elevation":
            continue
        result = interpolator.derive(
            level["value_original"],
            input_unit=level["unit_original"],
            input_vertical_datum=level["vertical_datum"],
        )
        availability = (
            "available" if result.within_supported_range else "out_of_supported_range"
        )
        counts["within_range" if result.within_supported_range else "out_of_range"] += 1
        common = {
            **level,
            "method": "official_eav_piecewise_linear",
            "observation_type": "derived",
            "source_quality_class": "B",
            "capability_level": 4,
            "provider": "usgs_sciencebase",
            "source_record_id": table.metadata.source_doi,
            "source_location_id": level["source_location_id"],
            "source_item_id": table.metadata.source_file,
            "source_parameter_code": level["source_parameter_code"],
            "within_supported_range": result.within_supported_range,
            "availability_status": availability,
            "capacity_table_source": result.capacity_table_source,
            "capacity_table_hash": result.capacity_table_hash,
            "capacity_table_version": result.capacity_table_version,
            "warning_flags": _json_list(result.warning_flags),
            "lower_bracket_elevation_m": result.lower_bracket_elevation,
            "upper_bracket_elevation_m": result.upper_bracket_elevation,
            "interpolation_fraction": result.interpolation_fraction,
            "d_area_dh_m2_per_m": result.d_area_dh_m2_per_m,
            "d_storage_dh_m3_per_m": result.d_storage_dh_m3_per_m,
            "storage_change_per_0_01_m_m3": result.storage_change_per_0_01_m_m3,
        }
        area = {
            **common,
            "metric": "surface_area",
            "value": result.derived_area_acres,
            "unit": "acres",
            "value_original": None,
            "unit_original": None,
            "vertical_datum": None,
            "height_type": HeightType.UNKNOWN.value,
            "uncertainty_unit": "acres",
        }
        storage = {
            **common,
            "metric": "total_storage",
            "value": result.derived_storage_acre_ft,
            "unit": "acre-ft",
            "value_original": None,
            "unit_original": None,
            "vertical_datum": None,
            "height_type": HeightType.UNKNOWN.value,
            "uncertainty_unit": "acre-ft",
        }
        derived.extend((area, storage))
    return derived, dict(counts)


def normalize_daily_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return the stable long-form schema with deterministic ordering."""

    frame = pd.DataFrame(rows)
    for column in DAILY_COLUMNS:
        if column not in frame:
            frame[column] = None
    frame = frame.loc[:, DAILY_COLUMNS].copy()
    if frame.empty:
        return frame
    if (
        frame["value"].notna().any()
        and not pd.to_numeric(
            frame.loc[frame["value"].notna(), "value"], errors="coerce"
        )
        .notna()
        .all()
    ):
        raise Phase1Error("daily hydrometry contains nonnumeric values")
    if set(frame["metric"].dropna()) - set(SUPPORTED_METRICS):
        raise Phase1Error("daily hydrometry contains an unsupported manager metric")
    duplicate_key = ["subprofile_id", "calendar_date", "metric"]
    duplicates = frame.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, duplicate_key].head().to_dict("records")
        raise Phase1Error(f"daily hydrometry contains duplicate keys: {sample}")
    return frame.sort_values(duplicate_key, kind="stable").reset_index(drop=True)


def _series_groups(frame: pd.DataFrame):
    valid = frame.loc[
        frame["value"].notna() & frame["availability_status"].eq("available")
    ].copy()
    for key, group in valid.groupby(
        ["profile_id", "subprofile_id", "display_name", "metric", "unit", "method"],
        sort=True,
        dropna=False,
    ):
        yield key, group.sort_values("calendar_date", kind="stable")


def build_manager_tables(
    daily: pd.DataFrame, settings: Phase1Settings, profiles: Sequence[HydrometryProfile]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate current, change, and seasonal tables from local daily rows."""

    current_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    seasonal_rows: list[dict[str, Any]] = []
    supported: set[tuple[str, str]] = set()
    for key, group in _series_groups(daily):
        profile_id, subprofile_id, display_name, metric, unit, method = key
        supported.add((subprofile_id, metric))
        last = group.iloc[-1]
        latest = date.fromisoformat(str(last["calendar_date"]))
        days_since = (settings.analysis_end - latest).days
        current_rows.append(
            {
                "schema_version": MANAGER_SCHEMA_VERSION,
                "profile_id": profile_id,
                "subprofile_id": subprofile_id,
                "Hylak_id": int(last["Hylak_id"]),
                "display_name": display_name,
                "metric": metric,
                "latest_date": latest.isoformat(),
                "latest_value": float(last["value"]),
                "unit": unit,
                "method": method,
                "provider": last["provider"],
                "source_record_id": last["source_record_id"],
                "source_location_id": last["source_location_id"],
                "source_item_id": last["source_item_id"],
                # Current conditions describe the latest source observation,
                # not a decade-wide mixture of historical approval states.
                "quality_status": (
                    "unknown"
                    if pd.isna(last["quality_status"])
                    else str(last["quality_status"])
                ),
                "approval_status": (
                    "unknown"
                    if pd.isna(last["approval_status"])
                    else str(last["approval_status"])
                ),
                "provisional": (
                    None if pd.isna(last["provisional"]) else bool(last["provisional"])
                ),
                "freshness_status": _freshness(latest, settings),
                "days_since_latest_observation": days_since,
                "source_quality_class": last["source_quality_class"],
                "capability_level": int(last["capability_level"]),
                "availability_status": "available",
                "warning": last["warning_flags"],
            }
        )
        observations = [
            MetricObservation(
                timestamp=date.fromisoformat(str(row.calendar_date)),
                value=float(row.value),
                metric=metric,
                unit=unit,
                method=method,
                provisional=(
                    None if pd.isna(row.provisional) else bool(row.provisional)
                ),
                approval_status=(
                    None if pd.isna(row.approval_status) else str(row.approval_status)
                ),
            )
            for row in group.itertuples(index=False)
        ]
        for result in calculate_change_metrics(
            observations,
            windows_days=settings.change_windows_days,
            analysis_end=settings.analysis_end,
            baseline_match_tolerance_days=settings.baseline_match_tolerance_days,
        ):
            change_rows.append(
                {
                    "schema_version": MANAGER_SCHEMA_VERSION,
                    "profile_id": profile_id,
                    "subprofile_id": subprofile_id,
                    "display_name": display_name,
                    "metric": metric,
                    "unit": unit,
                    "method": method,
                    **_change_dict(result),
                }
            )
        seasonal = calculate_seasonal_context(
            observations,
            analysis_end=settings.analysis_end,
            seasonal_window_days=settings.seasonal_window_days,
            minimum_distinct_years=settings.seasonal_min_distinct_years,
            minimum_observations=settings.seasonal_min_observations,
        )
        seasonal_rows.append(
            {
                "schema_version": MANAGER_SCHEMA_VERSION,
                "profile_id": profile_id,
                "subprofile_id": subprofile_id,
                "display_name": display_name,
                "metric": metric,
                "unit": unit,
                "method": method,
                **_seasonal_dict(seasonal),
            }
        )
    # Unsupported area/storage are explicit null products in the manager
    # contract, never fabricated daily observations and never zeros.
    for profile in profiles:
        subprofiles = (
            (
                ("great_salt_south_arm", "Great Salt south arm"),
                ("great_salt_north_arm", "Great Salt north arm"),
            )
            if profile.profile_id == "great_salt"
            else ((profile.profile_id, profile.display_name),)
        )
        for subprofile_id, display_name in subprofiles:
            for metric in SUPPORTED_METRICS:
                if (subprofile_id, metric) in supported:
                    continue
                current_rows.append(
                    {
                        "schema_version": MANAGER_SCHEMA_VERSION,
                        "profile_id": profile.profile_id,
                        "subprofile_id": subprofile_id,
                        "Hylak_id": profile.Hylak_id,
                        "display_name": display_name,
                        "metric": metric,
                        "latest_date": None,
                        "latest_value": None,
                        "unit": None,
                        "method": None,
                        "provider": None,
                        "source_record_id": None,
                        "source_location_id": None,
                        "source_item_id": None,
                        "quality_status": "unknown",
                        "approval_status": "unknown",
                        "provisional": None,
                        "freshness_status": "unavailable",
                        "days_since_latest_observation": None,
                        "source_quality_class": None,
                        "capability_level": 2 if metric == "water_elevation" else None,
                        "availability_status": "unavailable_for_profile",
                        "warning": "metric is not supported by an authoritative Phase 1 source",
                    }
                )
    current = pd.DataFrame(current_rows, columns=CURRENT_COLUMNS).sort_values(
        ["profile_id", "subprofile_id", "metric"], kind="stable"
    )
    changes = pd.DataFrame(change_rows, columns=CHANGE_COLUMNS).sort_values(
        ["profile_id", "subprofile_id", "metric", "requested_window_days"],
        kind="stable",
    )
    seasonal = pd.DataFrame(seasonal_rows, columns=SEASONAL_COLUMNS).sort_values(
        ["profile_id", "subprofile_id", "metric"], kind="stable"
    )
    return (
        current.reset_index(drop=True),
        changes.reset_index(drop=True),
        seasonal.reset_index(drop=True),
    )


def _change_dict(result: ChangeMetricResult) -> dict[str, Any]:
    return result.as_dict()


def _seasonal_dict(result: SeasonalContextResult) -> dict[str, Any]:
    raw = result.as_dict()
    return {
        "latest_date": raw["latest_date"],
        "latest_value": raw["latest_value"],
        "seasonal_percentile": raw["empirical_percentile"],
        "seasonal_median": raw["median"],
        "seasonal_q25": raw["q25"],
        "seasonal_q75": raw["q75"],
        "seasonal_minimum": raw["minimum"],
        "seasonal_maximum": raw["maximum"],
        "observation_count": raw["observation_count"],
        "distinct_year_count": raw["distinct_year_count"],
        "window_days": raw["seasonal_window_days"],
        "seasonal_status": raw["status"],
    }


def build_manager_summary(
    profiles: Sequence[HydrometryProfile],
    current: pd.DataFrame,
    changes: pd.DataFrame,
    seasonal: pd.DataFrame,
    *,
    settings: Phase1Settings,
    sources: Sequence[Mapping[str, Any]],
    generated_at: datetime,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the versioned service-facing JSON contract."""

    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        clean = frame.astype(object).where(pd.notna(frame), None)
        return clean.to_dict("records")

    result_profiles: list[dict[str, Any]] = []
    for profile in profiles:
        conditions = current.loc[current["profile_id"].eq(profile.profile_id)]
        result_profiles.append(
            {
                "identity": {
                    "profile_id": profile.profile_id,
                    "Hylak_id": profile.Hylak_id,
                    "display_name": profile.display_name,
                    "state": profile.state,
                },
                "capabilities": {
                    "current_level": 4
                    if profile.profile_id
                    in {"bear_lake", "lake_powell", "flaming_gorge"}
                    else 2,
                    "direct_total_storage": profile.profile_id
                    in {"lake_powell", "flaming_gorge"},
                    "official_table_derived_storage": profile.profile_id == "bear_lake",
                },
                "current_conditions": records(conditions),
                "change_metrics": records(
                    changes.loc[changes["profile_id"].eq(profile.profile_id)]
                ),
                "seasonal_context": records(
                    seasonal.loc[seasonal["profile_id"].eq(profile.profile_id)]
                ),
                "sources": [
                    dict(source)
                    for source in sources
                    if source.get("profile_id") == profile.profile_id
                ],
                "warnings": list(profile.key_constraints),
                "provenance": {
                    "analysis_start": settings.history_start.isoformat(),
                    "analysis_end": settings.analysis_end.isoformat(),
                },
            }
        )
    return {
        "schema_version": MANAGER_SCHEMA_VERSION,
        "analysis_start": settings.history_start.isoformat(),
        "analysis_end": settings.analysis_end.isoformat(),
        "generated_at": _iso(generated_at),
        "profiles": result_profiles,
        "provenance": dict(provenance),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _area_volume_diagnostic_frame(table: NormalizedCapacityTable) -> pd.DataFrame:
    """Return the required adjacent-row discrepancy summary."""

    _segments, summary = table.area_volume_diagnostics()
    return pd.DataFrame(
        [
            {
                **summary.as_dict(),
                "source_doi": table.metadata.source_doi,
                "source_file": table.metadata.source_file,
                "source_hash": table.metadata.source_hash,
                "diagnostic_status": "descriptive_not_hard_acceptance_gate",
            }
        ]
    )


def write_phase1_outputs(bundle: Phase1Bundle, output_dir: Path) -> list[Path]:
    """Atomically serialize all Phase 1 data products."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pilot_daily_hydrometry.csv": bundle.daily,
        "pilot_current_conditions.csv": bundle.current_conditions,
        "pilot_change_metrics.csv": bundle.change_metrics,
        "pilot_seasonal_context.csv": bundle.seasonal_context,
        "bear_lake_eav_normalized.csv": pd.DataFrame(
            bundle.capacity_table.normalized_records()
        ),
        "bear_lake_eav_validation.csv": bundle.capacity_validation,
        "bear_lake_eav_area_volume_diagnostic.csv": bundle.area_volume_diagnostic,
    }
    written: list[Path] = []
    for name, frame in outputs.items():
        path = output_dir / name
        _atomic_csv(frame, path)
        written.append(path)
    manager_path = output_dir / "pilot_manager_summary.json"
    _atomic_text(
        manager_path,
        json.dumps(bundle.manager_summary, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )
    written.append(manager_path)
    provenance = dict(bundle.provenance)
    provenance["output_sha256"] = {path.name: _sha256(path) for path in written}
    provenance_path = output_dir / "pilot_phase1_provenance.json"
    _atomic_text(
        provenance_path,
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    written.append(provenance_path)
    return written


def run_phase1(
    registry: HydrometryRegistry,
    settings: Phase1Settings,
    *,
    mode: CacheMode,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    reference_dir: Path = DEFAULT_REFERENCE_DIR,
    refresh: bool = False,
    refresh_reference_data: bool = False,
    profile_ids: Sequence[str] | None = None,
    adapters: Phase1Adapters | None = None,
    reference_source: Any | None = None,
    now: datetime | None = None,
) -> Phase1Bundle:
    """Acquire once per source, then derive and summarize entirely locally."""

    if mode is CacheMode.OFFLINE and (refresh or refresh_reference_data):
        raise Phase1Error("offline mode cannot refresh provider or reference data")
    profiles = _selected_profiles(registry, profile_ids)
    adapters = adapters or build_phase1_adapters(
        mode=mode, cache_dir=cache_dir, refresh=refresh
    )
    generated_at = (now or datetime.now(UTC)).astimezone(UTC)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    bear_levels: list[dict[str, Any]] = []
    for profile in profiles:
        if profile.profile_id in {"great_salt", "lake_abert", "bear_lake"}:
            for source in profile.sources:
                source_rows, metadata, _series, _location = _fetch_usgs_level_rows(
                    adapters.usgs,
                    profile,
                    source,
                    start=settings.history_start,
                    end=settings.analysis_end,
                    settings=settings,
                    retrieved_at=generated_at,
                )
                rows.extend(source_rows)
                sources.append(metadata)
                if profile.profile_id == "bear_lake":
                    bear_levels.extend(source_rows)
        elif profile.profile_id in {"lake_powell", "flaming_gorge"}:
            direct_rows, direct_sources = _fetch_rise_rows(
                adapters.rise,
                profile,
                start=settings.history_start,
                end=settings.analysis_end,
                settings=settings,
                retrieved_at=generated_at,
            )
            rows.extend(direct_rows)
            sources.extend(direct_sources)
    # Bear Lake's official source is the sole derived-storage gate. The shared
    # table is also serialized by full runs for an auditable local reference.
    if reference_source is None:
        from .sciencebase import BEAR_LAKE_TITLE, acquire_bear_lake_eav

        configured_bear = _profile_map(registry)["bear_lake"]
        bear_catalog = next(
            source
            for source in configured_bear.bathymetry_sources
            if source.catalog_id == "usgs_bear_lake_eav_tbdem"
        )
        if not bear_catalog.doi:
            raise Phase1Error("Bear Lake E-A-V source has no configured DOI")
        source = acquire_bear_lake_eav(
            client=adapters.client,
            mode=mode,
            reference_dir=reference_dir,
            doi=bear_catalog.doi,
            expected_title=BEAR_LAKE_TITLE,
            refresh=refresh_reference_data,
        )
    else:
        source = reference_source
    capacity_table = source.table
    if any(profile.profile_id == "bear_lake" for profile in profiles):
        bear_profile = next(
            profile for profile in profiles if profile.profile_id == "bear_lake"
        )
        derived, derived_counts = _bear_derived_rows(
            bear_levels, bear_profile, table=capacity_table, settings=settings
        )
        rows.extend(derived)
    else:
        derived_counts = {"within_range": 0, "out_of_range": 0}
    daily = normalize_daily_frame(rows)
    current, changes, seasonal = build_manager_tables(daily, settings, profiles)
    elapsed = time.perf_counter() - started
    request_count = int(getattr(adapters.client, "request_count", 0))
    reference_manifest = _public_manifest(source.manifest)
    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generated_at_utc": _iso(generated_at),
        "mode": mode.value,
        "refresh": refresh,
        "refresh_reference_data": refresh_reference_data,
        "analysis_start": settings.history_start.isoformat(),
        "analysis_end": settings.analysis_end.isoformat(),
        "profile_ids": [profile.profile_id for profile in profiles],
        "request_count": request_count,
        "elapsed_seconds": round(elapsed, 3),
        "daily_row_count": len(daily),
        "bear_derived_counts": {
            "source_stage_rows": len(bear_levels),
            "within_range": derived_counts.get("within_range", 0),
            "out_of_range": derived_counts.get("out_of_range", 0),
            "null_stage_rows": sum(
                row.get("value_original") is None for row in bear_levels
            ),
        },
        "bear_vertical_compatibility": {
            "stage_datum": "NAVD88",
            "table_datum": capacity_table.metadata.vertical_datum,
            "transformation_applied": False,
            "status": "compatible_no_vertical_transformation",
        },
        "bear_reference_manifest": reference_manifest,
        "bear_source_metadata_warnings": [
            (
                "The official FGDC entity description calls the E-A-V CSV a "
                "relation for the 'north arm of Bear Lake'; the release title and "
                "abstract identify Bear Lake as a whole, and the table maximum area "
                "(287.08 km2) is consistent with the whole-lake envelope. Phase 1 "
                "retains this as a source-metadata inconsistency requiring review."
            )
        ],
        "powell_storage_method": "agency_reported",
        "flaming_gorge_storage_method": "agency_reported",
        "flaming_gorge_bank_storage_excluded": True,
        "earth_engine_used": False,
        "qgis_modified": False,
        "great_salt_storage_calculated": False,
        "lake_abert_storage_calculated": False,
        "lake_powell_storage_reconstructed": False,
        "flaming_gorge_storage_reconstructed": False,
        "bear_lake_storage_derived": True,
    }
    manager = build_manager_summary(
        profiles,
        current,
        changes,
        seasonal,
        settings=settings,
        sources=sources,
        generated_at=generated_at,
        provenance={
            "mode": mode.value,
            "reference_sha256": reference_manifest.get("sha256"),
            "daily_row_count": len(daily),
        },
    )
    return Phase1Bundle(
        daily=daily,
        current_conditions=current,
        change_metrics=changes,
        seasonal_context=seasonal,
        manager_summary=manager,
        provenance=provenance,
        capacity_table=capacity_table,
        capacity_validation=pd.DataFrame([capacity_table.validation.as_dict()]),
        area_volume_diagnostic=_area_volume_diagnostic_frame(capacity_table),
        reference_manifest=reference_manifest,
        request_count=request_count,
        elapsed_seconds=elapsed,
    )


def render_manager_report(bundle: Phase1Bundle) -> str:
    """Render concise, method-explicit manager Markdown."""

    lines = [
        "# Hydrometry Phase 1 manager summary",
        "",
        (
            f"Pinned analysis window: {bundle.manager_summary['analysis_start']} "
            f"through {bundle.manager_summary['analysis_end']}."
        ),
        "",
        (
            "Observed water elevation, agency-reported total storage, and "
            "official-table-derived storage are intentionally distinct products."
        ),
        "",
    ]
    for profile in bundle.manager_summary["profiles"]:
        identity = profile["identity"]
        lines.extend((f"## {identity['display_name']}", ""))
        capability = profile["capabilities"]["current_level"]
        lines.append(f"Current capability: Level {capability}.")
        lines.append("")
        for row in profile["current_conditions"]:
            metric = str(row["metric"]).replace("_", " ")
            series_prefix = (
                f"{row['display_name']} — "
                if identity["profile_id"] == "great_salt"
                else ""
            )
            if row["availability_status"] != "available":
                lines.append(f"- {series_prefix}{metric.title()}: Unavailable.")
                continue
            method_label = {
                "agency_observed": "Observed",
                "agency_reported": "Agency-reported",
                "official_eav_piecewise_linear": "Official-table-derived",
            }.get(row["method"], str(row["method"]))
            provisional = (
                "yes"
                if row["provisional"] is True
                else "no"
                if row["provisional"] is False
                else "unknown"
            )
            lines.append(
                f"- {series_prefix}{method_label} {metric}: "
                f"{row['latest_value']:,.3f} "
                f"{row['unit']} on {row['latest_date']} "
                f"({row['freshness_status']}; quality {row['quality_status']}; "
                f"approval {row['approval_status']}; provisional {provisional}; "
                f"source {row['provider']} record {row['source_record_id']} item "
                f"{row['source_item_id'] or 'not applicable'})."
            )
        lines.append("")
        profile_changes = profile["change_metrics"]
        if profile_changes:
            lines.append("Changes:")
            for row in profile_changes:
                series_prefix = (
                    f"{row['display_name']} — "
                    if identity["profile_id"] == "great_salt"
                    else ""
                )
                value = row["absolute_change"]
                rendered = (
                    "unavailable" if value is None else f"{value:,.3f} {row['unit']}"
                )
                lines.append(
                    f"- {series_prefix}{row['metric'].replace('_', ' ').title()}, "
                    f"{row['requested_window_days']}-day: {rendered} "
                    f"({row['comparison_status']})."
                )
            lines.append("")
        seasonal = profile["seasonal_context"]
        if seasonal:
            lines.append("Seasonal context:")
            for row in seasonal:
                series_prefix = (
                    f"{row['display_name']} — "
                    if identity["profile_id"] == "great_salt"
                    else ""
                )
                percentile = row["seasonal_percentile"]
                text = (
                    "unavailable"
                    if percentile is None
                    else f"{percentile:.1f}th percentile"
                )
                lines.append(
                    f"- {series_prefix}{row['metric'].replace('_', ' ').title()}: "
                    f"{text} "
                    f"({row['seasonal_status']})."
                )
            lines.append("")
        lines.append("Key warnings: " + "; ".join(profile["warnings"]))
        next_step = {
            "great_salt": (
                "Document NGVD29-to-NAVD88 relationships and preserve separate "
                "north/south-arm basin treatment before any volume pathway."
            ),
            "lake_abert": (
                "Confirm an authoritative bathymetry or capacity relation before "
                "attempting absolute storage."
            ),
            "bear_lake": (
                "Quantify stage/table uncertainty and stabilize the service contract "
                "before any UI integration."
            ),
            "lake_powell": (
                "Document the operational elevation datum relationship to the NAVD88 "
                "capacity product before independent recomputation."
            ),
            "flaming_gorge": (
                "Acquire and datum-validate the ACAP table, using reported item 337 "
                "only as a future independent validation source."
            ),
        }[identity["profile_id"]]
        lines.append(f"Next capability step: {next_step}")
        lines.append("")
    lines.extend(
        (
            "## Scope boundary",
            "",
            (
                "Lake Powell and Flaming Gorge storage is reported directly by USBR "
                "RISE; it was not recalculated. Bear Lake storage is a Class-B value "
                "derived from observed NAVD88 elevation and the official USGS E-A-V "
                "table. Great Salt Lake and Lake Abert remain level-only."
            ),
            "",
        )
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--refresh-reference-data", action="store_true")
    parser.add_argument("--start", type=lambda value: _date(value, "--start"))
    parser.add_argument("--end", type=lambda value: _date(value, "--end"))
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.offline and (args.refresh or args.refresh_reference_data):
            raise Phase1Error("--offline cannot be combined with a refresh option")
        if (args.start is None) != (args.end is None):
            raise Phase1Error("--start and --end must be supplied together")
        settings = load_phase1_settings(args.config)
        if args.start is not None:
            settings = Phase1Settings(
                history_start=args.start,
                analysis_end=args.end,
                change_windows_days=settings.change_windows_days,
                baseline_match_tolerance_days=settings.baseline_match_tolerance_days,
                seasonal_window_days=settings.seasonal_window_days,
                seasonal_min_distinct_years=settings.seasonal_min_distinct_years,
                seasonal_min_observations=settings.seasonal_min_observations,
                freshness_warning_days=settings.freshness_warning_days,
                interpolation_method=settings.interpolation_method,
                allow_extrapolation=settings.allow_extrapolation,
            )
        registry = load_registry(args.config)
        mode = CacheMode.LIVE if args.live else CacheMode.OFFLINE
        bundle = run_phase1(
            registry,
            settings,
            mode=mode,
            cache_dir=args.cache_dir,
            reference_dir=args.reference_dir,
            refresh=args.refresh,
            refresh_reference_data=args.refresh_reference_data,
            profile_ids=args.profiles,
        )
        written = write_phase1_outputs(bundle, args.output_dir)
        args.figure_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.figure_dir / "manager_summary.md"
        _atomic_text(report_path, render_manager_report(bundle))
        written.append(report_path)
        if not args.skip_plots:
            from .plot_phase1 import create_phase1_figures

            written.extend(create_phase1_figures(bundle, args.figure_dir))
        print(
            f"Hydrometry Phase 1: {len(bundle.manager_summary['profiles'])} profile(s), "
            f"{bundle.request_count} live request(s), {len(bundle.daily):,} daily row(s)."
        )
        print(
            f"Bear E-A-V: {len(bundle.capacity_table.rows):,} rows; "
            f"SHA-256 {bundle.reference_manifest.get('sha256')}."
        )
        print(f"Wrote {len(written)} files; only Bear Lake storage was derived.")
        return 0
    except (Phase1Error, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
