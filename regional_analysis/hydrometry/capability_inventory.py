"""Phase 0 pilot hydrometry capability inventory CLI.

The inventory probes small provider metadata and bounded recent time series,
normalizes a preview, and writes a conservative capability report.  It does not
calculate volume, interpolate a capacity relation, download bathymetry, or use
Earth Engine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from .adapters.hydrocron import HydrocronAdapter
from .adapters.usbr_rise import RiseAdapter, RiseCatalogItem
from .adapters.usgs_waterdata import (
    SeriesSelectionStatus,
    UsgsWaterDataAdapter,
    select_water_level_series,
)
from .bathymetry_catalog import (
    BathymetryCatalogProbe,
    BathymetryProbeResult,
    entries_for_profile,
    get_bathymetry_entry,
)
from .cache import CacheMode, CacheStore
from .capability import CapabilityEvidence, classify_capability
from .http import HttpClient
from .models import (
    CapabilityLevel,
    CapabilityRecord,
    HeightType,
    HydrometryVariable,
    NormalizationStatus,
    NormalizedTimeSeriesRecord,
    Provider,
    VerticalReference,
    ensure_no_sensitive_keys,
    vertical_reference_supports_absolute_storage,
)
from .registry import HydrometryProfile, HydrometryRegistry, SourceSpec, load_registry

DEFAULT_CONFIG = Path("configs/hydrometry_pilots.yaml")
DEFAULT_OUTPUT_DIR = Path("data/hydrometry")
DEFAULT_CACHE_DIR = Path("data/hydrometry/cache")
DEFAULT_REPORT_DIR = Path("outputs/hydrometry_phase0")
PREVIEW_ROWS_PER_SERIES = 10
PREVIEW_COLUMNS = (
    "profile_id",
    "Hylak_id",
    "display_name",
    "provider",
    "source_id",
    "source_record_id",
    "source_item_id",
    "variable",
    "value",
    "unit",
    "timestamp_utc",
    "quality_status",
    "provisional",
    "approval_status",
    "height_type",
    "retrieved_at_utc",
    "source_version",
    "source_url_public",
    "raw_parameter_code",
    "raw_statistic_code",
    "vertical_datum",
    "vertical_reference",
    "raw_metadata",
)
OUTPUT_FILENAMES = {
    "capability": "pilot_capability_inventory.csv",
    "sources": "pilot_source_inventory.csv",
    "vertical": "pilot_vertical_references.csv",
    "probes": "pilot_probe_status.csv",
    "preview": "pilot_timeseries_preview.csv",
    "bathymetry": "pilot_bathymetry_catalog.csv",
    "provenance": "pilot_hydrometry_provenance.json",
}


class InventoryError(RuntimeError):
    """Raised for invalid inventory execution rather than provider failures."""


@dataclass(slots=True)
class InventoryAdapters:
    """Injectable adapter group used by the CLI and credential-free tests."""

    usgs: Any
    rise: Any
    hydrocron: Any
    bathymetry: Any
    client: Any


@dataclass(slots=True)
class _ProfileState:
    confirmed_sources: set[str] = field(default_factory=set)
    likely_sources: set[str] = field(default_factory=set)
    unverified_sources: set[str] = field(default_factory=set)
    water_level_available: bool = False
    direct_storage_available: bool = False
    potential_direct_storage: bool = False
    official_capacity_available: bool = False
    surveyed_bathymetry_available: bool = False
    potential_quantitative_3d: bool = False
    vertical_references: list[VerticalReference] = field(default_factory=list)


@dataclass(slots=True)
class InventoryBundle:
    """Provider-neutral rows ready for deterministic local serialization."""

    capability_records: list[CapabilityRecord]
    source_rows: list[dict[str, Any]]
    vertical_rows: list[dict[str, Any]]
    probe_rows: list[dict[str, Any]]
    preview_records: list[NormalizedTimeSeriesRecord]
    bathymetry_rows: list[dict[str, Any]]
    provenance: dict[str, Any]


def _iso(value: datetime | date | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return value.isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _error_text(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text}"[:2_000]


def _profile_matches_location(profile: HydrometryProfile, returned_name: str) -> bool:
    excluded = {"lake", "reservoir", "the"}
    expected = {
        token.casefold().strip("(),.-")
        for token in profile.display_name.split()
        if token.casefold().strip("(),.-") not in excluded
    }
    actual = {token.casefold().strip("(),.-") for token in returned_name.split()}
    return bool(expected) and expected.issubset(actual)


def _target_vertical_datum(profile: HydrometryProfile) -> str | None:
    datums = {
        entry.vertical_datum.split()[0]
        for entry in entries_for_profile(profile.profile_id)
        if entry.vertical_datum
    }
    return next(iter(datums)) if len(datums) == 1 else None


def _probe_row(
    profile: HydrometryProfile,
    *,
    provider: str,
    probe_type: str,
    source_id: str,
    source_record_id: str | None,
    source_item_id: str | None,
    status: str,
    message: str,
    source_url_public: str | None = None,
    observation_count: int | None = None,
    retrieved_at_utc: datetime | None = None,
    swot_pld_id: str | None = None,
    collection_name: str | None = None,
    product_versions: Sequence[str] = (),
    observation_start: datetime | None = None,
    observation_end: datetime | None = None,
) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "Hylak_id": profile.Hylak_id,
        "display_name": profile.display_name,
        "provider": provider,
        "probe_type": probe_type,
        "source_id": source_id,
        "source_record_id": source_record_id,
        "source_item_id": source_item_id,
        "status": status,
        "message": message,
        "observation_count": observation_count,
        "source_url_public": source_url_public,
        "retrieved_at_utc": _iso(retrieved_at_utc),
        "swot_pld_id": swot_pld_id,
        "collection_name": collection_name,
        "product_versions": " | ".join(product_versions) or None,
        "observation_start": _iso(observation_start),
        "observation_end": _iso(observation_end),
    }


def _source_row(
    profile: HydrometryProfile,
    source: SourceSpec,
    **values: Any,
) -> dict[str, Any]:
    row = {
        "profile_id": profile.profile_id,
        "Hylak_id": profile.Hylak_id,
        "display_name": profile.display_name,
        "state": profile.state,
        "provider": source.provider.value,
        "source_id": source.source_id,
        "source_record_id": source.source_record_id,
        "source_item_id": source.source_item_id,
        "configured": True,
        "priority": source.priority,
        "role": source.role,
        "variable": source.variable.value,
        "unit": source.expected_unit,
        "parameter_name": None,
        "parameter_code": None,
        "statistic_code": None,
        "title": None,
        "series_start": None,
        "series_end": None,
        "update_frequency": None,
        "provisional": None,
        "approval_status": None,
        "observed_start": None,
        "observed_end": None,
        "freshness_days": None,
        "location_id": None,
        "location_name": None,
        "vertical_datum": source.expected_vertical_datum,
        "height_type": source.expected_height_type.value,
        "selection_status": "configured",
        "status": "unverified",
        "source_url_public": None,
        "warning": None,
    }
    row.update(values)
    return row


def _vertical_row(
    profile: HydrometryProfile,
    source: SourceSpec,
    reference: VerticalReference,
    *,
    source_item_id: str | None,
    status: str,
) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "Hylak_id": profile.Hylak_id,
        "display_name": profile.display_name,
        "provider": source.provider.value,
        "source_id": source.source_id,
        "source_record_id": source.source_record_id,
        "source_item_id": source_item_id,
        "status": status,
        **reference.as_public_dict(),
    }


def _expected_vertical_reference(
    profile: HydrometryProfile,
    source: SourceSpec,
) -> VerticalReference:
    return VerticalReference(
        height_type=source.expected_height_type,
        source_vertical_datum=source.expected_vertical_datum,
        source_vertical_units=source.expected_unit,
        target_vertical_datum=_target_vertical_datum(profile),
        normalization_status=NormalizationStatus.NOT_ATTEMPTED,
        notes="Configured expectation only; the provider source was not validated.",
    )


def _preview(
    records: Sequence[NormalizedTimeSeriesRecord],
) -> list[NormalizedTimeSeriesRecord]:
    grouped: dict[tuple[str, str, str | None], list[NormalizedTimeSeriesRecord]] = (
        defaultdict(list)
    )
    for record in records:
        grouped[(record.profile_id, record.source_id, record.source_item_id)].append(
            record
        )
    selected: list[NormalizedTimeSeriesRecord] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda item: item.timestamp_utc)
        selected.extend(rows[-PREVIEW_ROWS_PER_SERIES:])
    return selected


def build_adapters(
    *,
    mode: CacheMode,
    cache_dir: Path,
    refresh: bool,
) -> InventoryAdapters:
    """Construct production adapters with one bounded shared HTTP client."""

    client = HttpClient(timeout=30.0, max_retries=3)
    cache = CacheStore(cache_dir)
    common = {"client": client, "mode": mode, "cache": cache, "refresh": refresh}
    return InventoryAdapters(
        usgs=UsgsWaterDataAdapter(**common),
        rise=RiseAdapter(**common),
        hydrocron=HydrocronAdapter(**common),
        bathymetry=BathymetryCatalogProbe(**common),
        client=client,
    )


def _probe_usgs_source(
    profile: HydrometryProfile,
    source: SourceSpec,
    *,
    adapter: Any,
    start: date,
    end: date,
    retrieved_at: datetime,
    state: _ProfileState,
    source_rows: list[dict[str, Any]],
    vertical_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    records: list[NormalizedTimeSeriesRecord],
) -> None:
    try:
        location = adapter.fetch_monitoring_location(source.source_record_id)
        if not _profile_matches_location(profile, location.name):
            raise InventoryError(
                f"Returned monitoring location {location.name!r} does not match "
                f"profile {profile.display_name!r}."
            )
        series = adapter.discover_series(source.source_record_id)
        selection = select_water_level_series(
            series,
            expected_unit=source.expected_unit,
            expected_vertical_datum=source.expected_vertical_datum,
            prefer_daily=True,
        )
        for item in series:
            selected = selection.selected is item
            source_rows.append(
                _source_row(
                    profile,
                    source,
                    source_item_id=item.time_series_id,
                    variable=item.variable.value if item.variable else None,
                    unit=item.unit,
                    parameter_name=item.parameter_name,
                    parameter_code=item.parameter_code,
                    statistic_code=item.statistic_code,
                    title=item.parameter_description,
                    series_start=_iso(item.begin_utc),
                    series_end=_iso(item.end_utc),
                    vertical_datum=item.vertical_datum or location.vertical_datum,
                    height_type=item.height_type.value,
                    selection_status="selected" if selected else "not_selected",
                    status=("selected_metadata" if selected else "discovered"),
                    source_url_public=item.source_url_public,
                    location_id=location.location_id,
                    location_name=location.name,
                    warning=(None if item.variable else "unclassified series variable"),
                )
            )
        if (
            selection.status is not SeriesSelectionStatus.SELECTED
            or not selection.selected
        ):
            state.unverified_sources.add(source.source_id)
            reference = _expected_vertical_reference(profile, source)
            vertical_rows.append(
                _vertical_row(
                    profile,
                    source,
                    reference,
                    source_item_id=None,
                    status="requires_review",
                )
            )
            probe_rows.append(
                _probe_row(
                    profile,
                    provider=source.provider.value,
                    probe_type="series_selection",
                    source_id=source.source_id,
                    source_record_id=source.source_record_id,
                    source_item_id=None,
                    status="requires_review",
                    message=selection.reason,
                    source_url_public=location.source_url_public,
                )
            )
            return

        chosen = selection.selected
        parameter = adapter.fetch_parameter_metadata(chosen.parameter_code)
        if parameter.unit.casefold() != chosen.unit.casefold():
            raise InventoryError(
                f"Observed-property unit {parameter.unit!r} does not match "
                f"selected series unit {chosen.unit!r}."
            )
        observations = adapter.fetch_observations(chosen, start=start, end=end)
        normalized = adapter.normalize_observations(
            observations,
            series=chosen,
            location=location,
            profile_id=profile.profile_id,
            hylak_id=profile.Hylak_id,
            display_name=profile.display_name,
            source_id=source.source_id,
            retrieved_at_utc=retrieved_at,
            target_vertical_datum=_target_vertical_datum(profile),
        )
        reference = adapter.vertical_reference(
            location,
            chosen,
            target_vertical_datum=_target_vertical_datum(profile),
        )
        vertical_rows.append(
            _vertical_row(
                profile,
                source,
                reference,
                source_item_id=chosen.time_series_id,
                status="confirmed_source_value_retained",
            )
        )
        state.vertical_references.append(reference)
        records.extend(normalized)
        selected_row = next(
            row
            for row in source_rows
            if row["source_id"] == source.source_id
            and row["source_item_id"] == chosen.time_series_id
        )
        selected_row["provisional"] = (
            any(item.provisional is True for item in observations)
            if observations
            else None
        )
        selected_row["approval_status"] = (
            " | ".join(
                sorted(
                    {
                        item.approval_status
                        for item in observations
                        if item.approval_status
                    }
                )
            )
            or None
        )
        selected_row["observed_start"] = (
            _iso(min(item.timestamp_utc for item in observations))
            if observations
            else None
        )
        selected_row["observed_end"] = (
            _iso(max(item.timestamp_utc for item in observations))
            if observations
            else None
        )
        selected_row["freshness_days"] = (
            (
                retrieved_at.date()
                - max(item.timestamp_utc for item in observations).date()
            ).days
            if observations
            else (
                (retrieved_at.date() - chosen.end_utc.date()).days
                if chosen.end_utc
                else None
            )
        )
        # A uniquely discovered, semantically validated series establishes
        # Level-2 source capability even when a recent bounded window is empty
        # (Lake Abert has a short historical record). Recent observations and
        # freshness remain separate evidence in the probe/source rows.
        state.water_level_available = True
        state.confirmed_sources.add(source.source_id)
        if observations:
            status = "confirmed"
        else:
            status = "confirmed_series_no_recent_observations"
        selected_row["status"] = status
        probe_rows.append(
            _probe_row(
                profile,
                provider=source.provider.value,
                probe_type="bounded_timeseries",
                source_id=source.source_id,
                source_record_id=source.source_record_id,
                source_item_id=chosen.time_series_id,
                status=status,
                message=(
                    f"{selection.reason} Parameter metadata: "
                    f"{parameter.parameter_name}; {parameter.unit}."
                ),
                observation_count=len(observations),
                source_url_public=chosen.source_url_public,
                retrieved_at_utc=retrieved_at,
            )
        )
    except Exception as exc:  # noqa: BLE001 - isolate one provider source failure
        state.unverified_sources.add(source.source_id)
        if not any(row["source_id"] == source.source_id for row in source_rows):
            source_rows.append(
                _source_row(profile, source, status="failed", warning=_error_text(exc))
            )
        if not any(row["source_id"] == source.source_id for row in vertical_rows):
            vertical_rows.append(
                _vertical_row(
                    profile,
                    source,
                    _expected_vertical_reference(profile, source),
                    source_item_id=source.source_item_id,
                    status="unverified",
                )
            )
        probe_rows.append(
            _probe_row(
                profile,
                provider=source.provider.value,
                probe_type="source",
                source_id=source.source_id,
                source_record_id=source.source_record_id,
                source_item_id=source.source_item_id,
                status="failed",
                message=_error_text(exc),
            )
        )


def _rise_reference(
    profile: HydrometryProfile,
    source: SourceSpec | None,
    item: RiseCatalogItem,
) -> VerticalReference:
    height_type = HeightType.UNKNOWN
    if item.variable in {
        HydrometryVariable.RESERVOIR_ELEVATION,
        HydrometryVariable.WATER_LEVEL,
        HydrometryVariable.GAGE_HEIGHT,
    }:
        height_type = (
            source.expected_height_type
            if source is not None
            else HeightType.PROJECT_DATUM_ELEVATION
        )
    return VerticalReference(
        height_type=height_type,
        source_vertical_datum=item.vertical_datum,
        source_vertical_units=item.unit,
        target_vertical_datum=_target_vertical_datum(profile),
        normalization_status=NormalizationStatus.SOURCE_VALUE_RETAINED,
        notes=(
            "RISE source value retained; item metadata vertical datum is "
            f"{item.vertical_datum or 'not reported'}. No Phase 0 transformation."
        ),
    )


def _rise_inventory_row(
    profile: HydrometryProfile,
    configured_source: SourceSpec,
    item: RiseCatalogItem,
    *,
    location: Any,
    configured: bool,
    selection_status: str,
) -> dict[str, Any]:
    return _source_row(
        profile,
        configured_source,
        source_id=(
            configured_source.source_id
            if configured
            else _rise_discovered_source_id(item)
        ),
        source_item_id=str(item.item_id),
        configured=configured,
        priority=configured_source.priority if configured else None,
        role=configured_source.role if configured else "record_item_discovery",
        variable=item.variable.value if item.variable else None,
        unit=item.unit,
        parameter_name=item.parameter_name,
        parameter_code=str(item.parameter_id) if item.parameter_id else None,
        statistic_code=item.transformation,
        title=item.title,
        series_start=_iso(item.temporal_start),
        series_end=_iso(item.temporal_end),
        update_frequency=item.update_frequency,
        provisional=item.provisional,
        vertical_datum=item.vertical_datum,
        height_type=(
            configured_source.expected_height_type.value
            if configured
            else (
                HeightType.PROJECT_DATUM_ELEVATION.value
                if item.variable is HydrometryVariable.RESERVOIR_ELEVATION
                else HeightType.UNKNOWN.value
            )
        ),
        selection_status=selection_status,
        status="requires_review" if item.requires_review else "discovered",
        source_url_public=item.source_url_public,
        location_id=location.location_id,
        location_name=location.name,
        warning="; ".join(item.validation_issues) or None,
    )


def _rise_discovered_source_id(item: RiseCatalogItem) -> str:
    """Return an item-identity source ID independent of a configured anchor."""

    variable = item.variable.value if item.variable else "unclassified"
    return f"rise_record_{item.record_id}:{variable}:{item.item_id}"


def _validate_configured_rise_item(source: SourceSpec, item: RiseCatalogItem) -> None:
    if item.variable is not source.variable:
        raise InventoryError(
            f"RISE item {item.item_id} variable {item.variable} does not match "
            f"configured {source.variable.value}."
        )
    title_part = source.metadata.get("expected_title_contains")
    if title_part and str(title_part).casefold() not in item.title.casefold():
        raise InventoryError(
            f"RISE item {item.item_id} title {item.title!r} does not contain "
            f"configured text {title_part!r}."
        )
    aliases = {
        "acre-feet": "af",
        "acre-foot": "af",
        "acre-ft": "af",
        "acre feet": "af",
        "acre": "acres",
    }

    def canonical_unit(value: str) -> str:
        normalized = value.casefold().strip()
        return aliases.get(normalized, normalized)

    if source.expected_unit and (
        not item.unit
        or canonical_unit(item.unit) != canonical_unit(source.expected_unit)
    ):
        raise InventoryError(
            f"RISE item {item.item_id} unit {item.unit!r} does not match "
            f"configured {source.expected_unit!r}."
        )
    expected_frequency = source.metadata.get("expected_update_frequency")
    if expected_frequency and (
        not item.update_frequency
        or str(expected_frequency).casefold() not in item.update_frequency.casefold()
    ):
        raise InventoryError(
            f"RISE item {item.item_id} update frequency "
            f"{item.update_frequency!r} does not match configured "
            f"{expected_frequency!r}."
        )


def _probe_rise_profile(
    profile: HydrometryProfile,
    sources: Sequence[SourceSpec],
    *,
    adapter: Any,
    start: date,
    end: date,
    retrieved_at: datetime,
    state: _ProfileState,
    source_rows: list[dict[str, Any]],
    vertical_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    records: list[NormalizedTimeSeriesRecord],
) -> None:
    by_record: dict[str, list[SourceSpec]] = defaultdict(list)
    for source in sources:
        by_record[source.source_record_id].append(source)

    for record_id, record_sources in sorted(by_record.items()):
        anchor = min(record_sources, key=lambda item: item.priority)
        try:
            record = adapter.fetch_record(
                int(record_id), expected_title_contains=profile.display_name
            )
            location = adapter.fetch_location(
                record.location_id,
                expected_name_contains=profile.display_name,
                expected_record_id=record.record_id,
            )
            inventory = adapter.inventory_record_items(
                record, expected_location_name=profile.display_name
            )
        except Exception as exc:  # noqa: BLE001 - isolate one provider record failure
            for source in record_sources:
                state.unverified_sources.add(source.source_id)
                source_rows.append(
                    _source_row(
                        profile, source, status="failed", warning=_error_text(exc)
                    )
                )
                vertical_rows.append(
                    _vertical_row(
                        profile,
                        source,
                        _expected_vertical_reference(profile, source),
                        source_item_id=source.source_item_id,
                        status="unverified",
                    )
                )
            probe_rows.append(
                _probe_row(
                    profile,
                    provider=Provider.USBR_RISE.value,
                    probe_type="record_inventory",
                    source_id=anchor.source_id,
                    source_record_id=record_id,
                    source_item_id=None,
                    status="failed",
                    message=_error_text(exc),
                )
            )
            continue

        configured_by_item = {
            int(source.source_item_id): source
            for source in record_sources
            if source.source_item_id is not None
        }
        items = {item.item_id: item for item in inventory.items}
        for item in inventory.items:
            configured = item.item_id in configured_by_item
            configured_source = configured_by_item.get(item.item_id, anchor)
            source_rows.append(
                _rise_inventory_row(
                    profile,
                    configured_source,
                    item,
                    location=location,
                    configured=configured,
                    selection_status=(
                        "configured_item" if configured else "record_discovery"
                    ),
                )
            )
        probe_rows.append(
            _probe_row(
                profile,
                provider=Provider.USBR_RISE.value,
                probe_type="record_inventory",
                source_id=anchor.source_id,
                source_record_id=record_id,
                source_item_id=None,
                status="confirmed",
                message=(
                    f"Validated {record.title!r}; discovered {len(items)} exact "
                    f"relationship item(s) at {location.name!r} without guessing IDs."
                ),
                source_url_public=record.source_url_public,
            )
        )
        if any(source.source_item_id is None for source in record_sources):
            state.confirmed_sources.add(anchor.source_id)

        chosen: dict[int, SourceSpec | None] = {}
        for item_id, source in configured_by_item.items():
            if item_id not in items:
                state.unverified_sources.add(source.source_id)
                probe_rows.append(
                    _probe_row(
                        profile,
                        provider=Provider.USBR_RISE.value,
                        probe_type="configured_item",
                        source_id=source.source_id,
                        source_record_id=record_id,
                        source_item_id=str(item_id),
                        status="failed",
                        message=f"Configured item {item_id} is absent from record.",
                    )
                )
            else:
                chosen[item_id] = source

        # An inventory-only source discovers, rather than guesses, unique
        # elevation/area/total-storage items. Flaming Gorge additionally checks
        # the unique direct total-storage path exposed by its record.
        inventory_only = any(source.source_item_id is None for source in record_sources)
        discover_variables = (
            (
                HydrometryVariable.RESERVOIR_ELEVATION,
                HydrometryVariable.SURFACE_AREA,
                HydrometryVariable.STORAGE,
            )
            if inventory_only
            else (HydrometryVariable.STORAGE,)
        )
        for variable in discover_variables:
            candidates = inventory.candidates_by_variable.get(variable, ())
            if len(candidates) == 1:
                chosen.setdefault(candidates[0], None)
            elif len(candidates) > 1:
                state.unverified_sources.add(
                    f"{anchor.source_id}:{variable.value}:requires_review"
                )

        for item_id, source in sorted(chosen.items()):
            item = items[item_id]
            source_id = (
                source.source_id
                if source is not None
                else _rise_discovered_source_id(item)
            )
            try:
                if source is not None:
                    _validate_configured_rise_item(source, item)
                observations = adapter.fetch_timeseries(item, start=start, end=end)
                reference = _rise_reference(profile, source, item)
                normalized = adapter.normalize_observations(
                    observations,
                    item=item,
                    profile_id=profile.profile_id,
                    hylak_id=profile.Hylak_id,
                    display_name=profile.display_name,
                    source_id=source_id,
                    vertical_reference=reference,
                    retrieved_at_utc=retrieved_at,
                    source_version="rise-jsonapi",
                )
                records.extend(normalized)
                item_row = next(
                    row
                    for row in source_rows
                    if row["profile_id"] == profile.profile_id
                    and row["source_record_id"] == record_id
                    and row["source_item_id"] == str(item.item_id)
                )
                # A discovered record item receives one stable derived source
                # ID everywhere: source inventory, probe, normalized records,
                # and capability provenance.
                item_row["source_id"] = source_id
                item_row["observed_start"] = (
                    _iso(min(value.timestamp_utc for value in observations))
                    if observations
                    else None
                )
                item_row["observed_end"] = (
                    _iso(max(value.timestamp_utc for value in observations))
                    if observations
                    else None
                )
                item_row["freshness_days"] = (
                    (
                        retrieved_at.date()
                        - max(value.timestamp_utc for value in observations).date()
                    ).days
                    if observations
                    else None
                )
                vertical_source = source or anchor
                vertical_row = _vertical_row(
                    profile,
                    vertical_source,
                    reference,
                    source_item_id=str(item.item_id),
                    status="confirmed_source_value_retained",
                )
                vertical_row["source_id"] = source_id
                vertical_rows.append(vertical_row)
                state.vertical_references.append(reference)
                if observations:
                    state.confirmed_sources.add(source_id)
                    status = "confirmed"
                    if item.variable is HydrometryVariable.RESERVOIR_ELEVATION:
                        state.water_level_available = True
                    if item.variable is HydrometryVariable.STORAGE:
                        state.potential_direct_storage = True
                        # Powell remains explicitly blocked pending a documented
                        # operational/capacity vertical relationship. Flaming
                        # Gorge direct agency total storage can establish L4.
                        if profile.profile_id == "flaming_gorge":
                            state.direct_storage_available = True
                else:
                    state.unverified_sources.add(source_id)
                    status = "no_observations_in_window"
                item_row["status"] = status
                probe_rows.append(
                    _probe_row(
                        profile,
                        provider=Provider.USBR_RISE.value,
                        probe_type="bounded_timeseries",
                        source_id=source_id,
                        source_record_id=record_id,
                        source_item_id=str(item.item_id),
                        status=status,
                        message=(
                            f"Validated {item.parameter_name!r}, unit {item.unit!r}, "
                            f"update frequency {item.update_frequency or 'not reported'}, "
                            f"provisional={item.provisional}."
                        ),
                        observation_count=len(observations),
                        source_url_public=item.source_url_public,
                        retrieved_at_utc=retrieved_at,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - isolate one provider item failure
                state.unverified_sources.add(source_id)
                for row in source_rows:
                    if (
                        row["profile_id"] == profile.profile_id
                        and row["source_record_id"] == record_id
                        and row["source_item_id"] == str(item.item_id)
                    ):
                        row["status"] = "failed"
                        row["warning"] = _error_text(exc)
                probe_rows.append(
                    _probe_row(
                        profile,
                        provider=Provider.USBR_RISE.value,
                        probe_type="bounded_timeseries",
                        source_id=source_id,
                        source_record_id=record_id,
                        source_item_id=str(item.item_id),
                        status="failed",
                        message=_error_text(exc),
                        source_url_public=item.source_url_public,
                    )
                )


def _probe_bathymetry(
    profile: HydrometryProfile,
    *,
    adapter: Any,
    state: _ProfileState,
    rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
) -> None:
    configured = profile.bathymetry_sources
    if not configured:
        rows.append(
            {
                "profile_id": profile.profile_id,
                "Hylak_id": profile.Hylak_id,
                "display_name": profile.display_name,
                "catalog_id": None,
                "status": "not_configured",
                "warnings": (
                    "No authoritative bathymetry or capacity source is confirmed."
                ),
            }
        )
        return
    for specification in configured:
        entry = get_bathymetry_entry(specification.catalog_id)
        mismatch: str | None = None
        if entry is None:
            mismatch = (
                "configured catalog ID is absent from the static metadata catalog"
            )
        elif entry.profile_id != profile.profile_id:
            mismatch = f"static catalog profile is {entry.profile_id!r}"
        elif entry.provider != specification.provider.value:
            mismatch = f"static provider is {entry.provider!r}"
        elif entry.doi != specification.doi:
            mismatch = f"static DOI is {entry.doi!r}"
        elif entry.source_record_id != specification.source_record_id:
            mismatch = f"static record ID is {entry.source_record_id!r}"
        elif entry.product_type != specification.product_type:
            mismatch = f"static product type is {entry.product_type!r}"
        elif (
            entry.source_quality_candidate is not specification.source_quality_candidate
        ):
            mismatch = (
                "static source-quality candidate is "
                f"{entry.source_quality_candidate.value!r}"
            )
        if mismatch:
            state.unverified_sources.add(specification.catalog_id)
            rows.append(
                {
                    "profile_id": profile.profile_id,
                    "Hylak_id": profile.Hylak_id,
                    "display_name": profile.display_name,
                    "catalog_id": specification.catalog_id,
                    "status": "failed_config_catalog_mismatch",
                    "warnings": mismatch,
                }
            )
            probe_rows.append(
                _probe_row(
                    profile,
                    provider=specification.provider.value,
                    probe_type="bathymetry_metadata",
                    source_id=specification.catalog_id,
                    source_record_id=(
                        specification.source_record_id or specification.doi
                    ),
                    source_item_id=None,
                    status="failed_config_catalog_mismatch",
                    message=mismatch,
                )
            )
            continue
        assert entry is not None
        result: BathymetryProbeResult | None = None
        error: str | None = None
        try:
            result = adapter.verify(entry)
            confirmed = result.status.startswith("confirmed")
            if confirmed:
                state.confirmed_sources.add(entry.catalog_id)
                state.official_capacity_available |= bool(
                    entry.relation_table_available
                )
                state.surveyed_bathymetry_available |= bool(entry.raster_available)
                state.potential_quantitative_3d |= bool(entry.raster_available)
        except Exception as exc:  # noqa: BLE001 - isolate one metadata source failure
            error = _error_text(exc)
            state.unverified_sources.add(entry.catalog_id)
        row = asdict(entry)
        row["Hylak_id"] = row.pop("hylak_id")
        row["source_quality_candidate"] = entry.source_quality_candidate.value
        row["warnings"] = " | ".join(entry.warnings)
        row.update(
            {
                "status": result.status if result else "failed",
                "returned_title": result.returned_title if result else None,
                "returned_identifier": (result.returned_identifier if result else None),
                "retrieved_at_utc": result.retrieved_at_utc if result else None,
                "source_url_public": (
                    result.source_url_public if result else entry.landing_url
                ),
                "probe_warning": result.warning if result else error,
                "verification_scope": (
                    "landing identifier and title verified; scientific metadata "
                    "are curated from the cited official record; no product file "
                    "was downloaded"
                    if result
                    else "metadata verification failed"
                ),
            }
        )
        rows.append(row)
        probe_rows.append(
            _probe_row(
                profile,
                provider=entry.provider,
                probe_type="bathymetry_metadata",
                source_id=entry.catalog_id,
                source_record_id=entry.source_record_id or entry.doi,
                source_item_id=None,
                status=result.status if result else "failed",
                message=(result.warning or "Official landing metadata validated.")
                if result
                else error or "Unknown bathymetry probe error.",
                source_url_public=(
                    result.source_url_public if result else entry.landing_url
                ),
            )
        )


def _probe_hydrocron(
    profile: HydrometryProfile,
    *,
    adapter: Any,
    start: date,
    end: date,
    state: _ProfileState,
    probe_rows: list[dict[str, Any]],
    records: list[NormalizedTimeSeriesRecord],
    retrieved_at: datetime,
) -> None:
    try:
        result = adapter.probe(
            swot_pld_id=profile.swot.swot_pld_id,
            start=datetime.combine(start, datetime.min.time(), tzinfo=UTC),
            end=datetime.combine(end, datetime.max.time(), tzinfo=UTC),
        )
        source_id = f"hydrocron:{profile.profile_id}"
        if result.status == "confirmed":
            state.confirmed_sources.add(source_id)
            normalized = adapter.normalize_observations(
                result.observations,
                profile_id=profile.profile_id,
                hylak_id=profile.Hylak_id,
                display_name=profile.display_name,
                source_id=source_id,
                retrieved_at_utc=retrieved_at,
                source_url_public=result.source_url_public,
            )
            records.extend(normalized)
            if any(
                record.variable is HydrometryVariable.WATER_LEVEL
                for record in normalized
            ):
                state.water_level_available = True
        else:
            state.unverified_sources.add(source_id)
        probe_rows.append(
            _probe_row(
                profile,
                provider=Provider.HYDROCRON.value,
                probe_type="capability",
                source_id=source_id,
                source_record_id=profile.swot.swot_pld_id,
                source_item_id=None,
                status=result.status,
                message=result.message,
                observation_count=len(result.observations),
                source_url_public=result.source_url_public,
                retrieved_at_utc=retrieved_at,
                swot_pld_id=result.swot_pld_id,
                collection_name=result.collection_name,
                product_versions=result.product_versions,
                observation_start=(
                    min(item.timestamp_utc for item in result.observations)
                    if result.observations
                    else None
                ),
                observation_end=(
                    max(item.timestamp_utc for item in result.observations)
                    if result.observations
                    else None
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001 - retain a structured probe failure
        state.unverified_sources.add(f"hydrocron:{profile.profile_id}")
        probe_rows.append(
            _probe_row(
                profile,
                provider=Provider.HYDROCRON.value,
                probe_type="capability",
                source_id=f"hydrocron:{profile.profile_id}",
                source_record_id=profile.swot.swot_pld_id,
                source_item_id=None,
                status="failed",
                message=_error_text(exc),
            )
        )


def _capability_record(
    profile: HydrometryProfile,
    state: _ProfileState,
    *,
    mode: CacheMode,
    start: date,
    end: date,
) -> CapabilityRecord:
    relation_documented = any(
        vertical_reference_supports_absolute_storage(reference)
        for reference in state.vertical_references
    )
    source_datum_known = any(
        reference.source_vertical_datum is not None
        for reference in state.vertical_references
    )
    multiple_arm_resolved = profile.profile_id != "great_salt"
    potential_relation = state.official_capacity_available
    potential_bathy = state.surveyed_bathymetry_available
    potential_3d = state.potential_quantitative_3d

    # Bear Lake's configured potential is conditional on both live/cached source
    # confirmation and official product confirmation. Lake Abert has only a
    # relative-storage research path. Great Salt and Powell retain explicit
    # datum/arm blockers even when all metadata probes pass.
    if profile.profile_id == "bear_lake" and not state.water_level_available:
        potential_relation = False
        potential_bathy = False
        potential_3d = False

    missing: list[str] = []
    if not state.water_level_available and profile.profile_id != "flaming_gorge":
        missing.append("validated recent water-level observations")
    if profile.profile_id == "flaming_gorge":
        if not state.water_level_available:
            missing.append("validated recent reservoir-elevation observations")
        if not state.direct_storage_available:
            missing.append("validated direct total-storage observations")
    if profile.profile_id == "lake_abert":
        missing.append("authoritative bathymetry or capacity relation")
    if profile.profile_id == "lake_powell":
        missing.append(
            "documented RISE operational-stage/storage relationship to NAVD88 capacity"
        )
    datum_status = "source datum retained; no normalization attempted"
    if profile.profile_id == "great_salt":
        datum_status = "blocked: NGVD29 gauges versus NAVD88 bathymetry"
    elif profile.profile_id == "lake_powell":
        datum_status = "blocked: RISE item datum not linked to NAVD88 capacity"
    elif profile.profile_id == "flaming_gorge" and state.direct_storage_available:
        datum_status = (
            "direct total storage requires no height normalization; RISE elevation "
            "and ACAP vertical relationship remains unresolved"
        )
    elif not source_datum_known and state.water_level_available:
        datum_status = "requires_review: source vertical datum not reported"

    evidence = CapabilityEvidence(
        extent_available=True,
        water_level_available=state.water_level_available,
        direct_authoritative_storage_available=state.direct_storage_available,
        direct_storage_variable=(
            HydrometryVariable.STORAGE if state.direct_storage_available else None
        ),
        official_capacity_relation_available=state.official_capacity_available,
        surveyed_bathymetry_available=state.surveyed_bathymetry_available,
        storage_computation_implemented=False,
        source_vertical_datum_known=source_datum_known,
        vertical_relation_documented=relation_documented,
        multiple_arm_constraint_resolved=multiple_arm_resolved,
        potential_water_level=bool(profile.sources),
        potential_relative_storage=(
            profile.potential_capability_ceiling
            >= CapabilityLevel.RELATIVE_STORAGE_CHANGE
        ),
        potential_direct_storage=state.potential_direct_storage,
        potential_official_capacity_relation=potential_relation,
        potential_surveyed_bathymetry=potential_bathy,
        potential_quantitative_3d=potential_3d,
        datum_status=datum_status,
        missing_requirements=tuple(missing),
        key_warning="; ".join(profile.key_constraints),
    )
    return classify_capability(
        profile,
        evidence,
        confirmed_sources=sorted(state.confirmed_sources),
        likely_sources=sorted(state.likely_sources),
        unverified_sources=sorted(state.unverified_sources),
        provenance={
            "phase": 0,
            "mode": mode.value,
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "volume_calculated": False,
        },
    )


def run_inventory(
    registry: HydrometryRegistry,
    *,
    mode: CacheMode,
    start: date,
    end: date,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    profile_ids: Sequence[str] | None = None,
    adapters: InventoryAdapters | None = None,
    now: datetime | None = None,
) -> InventoryBundle:
    """Probe configured sources once and return serializable Phase 0 results."""

    if start > end:
        raise InventoryError("start must be on or before end")
    if mode is CacheMode.OFFLINE and refresh:
        raise InventoryError("--refresh cannot be combined with --offline")
    selected_profiles: list[HydrometryProfile]
    if profile_ids:
        unknown = sorted(
            set(profile_ids) - {profile.profile_id for profile in registry.profiles}
        )
        if unknown:
            raise InventoryError("Unknown profile ID(s): " + ", ".join(unknown))
        requested = set(profile_ids)
        selected_profiles = [
            profile for profile in registry.profiles if profile.profile_id in requested
        ]
    else:
        selected_profiles = list(registry.profiles)
    if not selected_profiles:
        raise InventoryError("At least one profile must be selected.")

    adapters = adapters or build_adapters(
        mode=mode, cache_dir=cache_dir, refresh=refresh
    )
    retrieved_at = (now or datetime.now(UTC)).astimezone(UTC)
    started = time.perf_counter()
    source_rows: list[dict[str, Any]] = []
    vertical_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    records: list[NormalizedTimeSeriesRecord] = []
    bathymetry_rows: list[dict[str, Any]] = []
    states: dict[str, _ProfileState] = {}

    for profile in selected_profiles:
        state = _ProfileState()
        states[profile.profile_id] = state
        usgs_sources = [
            source
            for source in profile.sources
            if source.provider is Provider.USGS_WATERDATA
        ]
        rise_sources = [
            source
            for source in profile.sources
            if source.provider is Provider.USBR_RISE
        ]
        for source in usgs_sources:
            _probe_usgs_source(
                profile,
                source,
                adapter=adapters.usgs,
                start=start,
                end=end,
                retrieved_at=retrieved_at,
                state=state,
                source_rows=source_rows,
                vertical_rows=vertical_rows,
                probe_rows=probe_rows,
                records=records,
            )
        if rise_sources:
            _probe_rise_profile(
                profile,
                rise_sources,
                adapter=adapters.rise,
                start=start,
                end=end,
                retrieved_at=retrieved_at,
                state=state,
                source_rows=source_rows,
                vertical_rows=vertical_rows,
                probe_rows=probe_rows,
                records=records,
            )
        _probe_hydrocron(
            profile,
            adapter=adapters.hydrocron,
            start=start,
            end=end,
            state=state,
            probe_rows=probe_rows,
            records=records,
            retrieved_at=retrieved_at,
        )
        _probe_bathymetry(
            profile,
            adapter=adapters.bathymetry,
            state=state,
            rows=bathymetry_rows,
            probe_rows=probe_rows,
        )

    capabilities = [
        _capability_record(
            profile, states[profile.profile_id], mode=mode, start=start, end=end
        )
        for profile in selected_profiles
    ]
    elapsed = time.perf_counter() - started
    request_count = int(getattr(adapters.client, "request_count", 0))
    cache_files = (
        sorted(
            path.relative_to(cache_dir).as_posix()
            for path in cache_dir.rglob("*")
            if path.is_file()
        )
        if cache_dir.exists()
        else []
    )
    runtime_cache = getattr(adapters.usgs, "cache", None)
    accessed_raw_paths = list(getattr(runtime_cache, "accessed_raw_response_paths", ()))
    provenance = {
        "schema_version": "hydrometry-phase0-provenance-v1",
        "generated_at_utc": _iso(retrieved_at),
        "registry_schema_version": registry.schema_version,
        "registry_path": str(registry.source_path) if registry.source_path else None,
        "registry_sha256": (
            hashlib.sha256(registry.source_path.read_bytes()).hexdigest()
            if registry.source_path and registry.source_path.is_file()
            else None
        ),
        "mode": mode.value,
        "refresh": refresh,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "profiles": [profile.profile_id for profile in selected_profiles],
        "request_count": request_count,
        "elapsed_seconds": round(elapsed, 3),
        "cache_dir": str(cache_dir),
        "cache_files": cache_files,
        "cache_raw_responses_used": sorted(accessed_raw_paths),
        "normalized_record_count": len(records),
        "preview_record_count": len(_preview(records)),
        "volume_calculated": False,
        "earth_engine_used": False,
        "public_output_validated": True,
    }
    ensure_no_sensitive_keys(provenance, context="inventory provenance")
    return InventoryBundle(
        capability_records=capabilities,
        source_rows=source_rows,
        vertical_rows=vertical_rows,
        probe_rows=probe_rows,
        preview_records=_preview(records),
        bathymetry_rows=bathymetry_rows,
        provenance=provenance,
    )


def _flatten_capability(record: CapabilityRecord) -> dict[str, Any]:
    row = record.as_public_dict()
    row["highest_current_capability"] = int(record.highest_current_capability)
    row["highest_potential_capability"] = int(record.highest_potential_capability)
    row["source_quality_class"] = (
        record.source_quality_class.value if record.source_quality_class else None
    )
    row["source_quality_status"] = (
        "confirmed_current"
        if record.absolute_storage_available and record.source_quality_class is not None
        else "potential_candidate"
        if record.source_quality_class is not None
        else "unassigned"
    )
    for key in (
        "confirmed_sources",
        "likely_sources",
        "unverified_sources",
        "missing_requirements",
    ):
        row[key] = " | ".join(row[key])
    row["provenance"] = _json(row["provenance"])
    return row


def _flatten_preview(record: NormalizedTimeSeriesRecord) -> dict[str, Any]:
    row = record.as_public_dict()
    vertical = row.pop("vertical_datum")
    row["provider"] = record.provider.value
    row["variable"] = record.variable.value
    row["height_type"] = record.height_type.value
    row["vertical_datum"] = vertical.get("source_vertical_datum")
    row["vertical_reference"] = _json(vertical)
    row["raw_metadata"] = _json(row["raw_metadata"])
    return row


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    if rows:
        resolved_fields = list(fieldnames or ())
        for row in rows:
            for key in row:
                if key not in resolved_fields:
                    resolved_fields.append(key)
    else:
        resolved_fields = list(fieldnames or ("status",))
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=resolved_fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    _json(value)
                    if isinstance(value, (dict, list, tuple, set))
                    else value
                )
                for key, value in row.items()
            }
        )
    _atomic_text(path, stream.getvalue())


def _macro_report_rows(bundle: InventoryBundle) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle.source_rows:
        grouped[str(row["profile_id"])].append(row)
    return grouped


def render_capability_report(bundle: InventoryBundle) -> str:
    """Render a compact scientific capability report without volume claims."""

    source_by_profile = _macro_report_rows(bundle)
    bathy_by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    probe_by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle.bathymetry_rows:
        bathy_by_profile[str(row["profile_id"])].append(row)
    for row in bundle.probe_rows:
        probe_by_profile[str(row["profile_id"])].append(row)
    lines = [
        "# Hydrometry Phase 0 pilot capability report",
        "",
        (
            f"Generated `{bundle.provenance['generated_at_utc']}` in "
            f"**{bundle.provenance['mode']}** mode for "
            f"`{bundle.provenance['window_start']}` through "
            f"`{bundle.provenance['window_end']}`. The run made "
            f"{bundle.provenance['request_count']} live HTTP request(s)."
        ),
        "",
        (
            "Phase 0 inventories capability and metadata only. It performs no "
            "stage--capacity interpolation, bathymetric integration, or volume "
            "calculation. HydroLAKES IDs and SWOT Prior Lake Database IDs are "
            "separate; configured pilot profiles still require a PLD crosswalk."
        ),
        "",
    ]
    for capability in bundle.capability_records:
        pid = capability.profile_id
        sources = source_by_profile[pid]
        probes = probe_by_profile[pid]
        date_starts = sorted(
            str(row["series_start"]) for row in sources if row.get("series_start")
        )
        date_ends = sorted(
            str(row["series_end"]) for row in sources if row.get("series_end")
        )
        provisional = sorted(
            {
                str(row["provisional"])
                for row in sources
                if row.get("provisional") is not None
            }
        )
        bathymetry = [
            (
                f"{row.get('catalog_id') or 'none'} "
                f"({row.get('status')}; "
                f"{row.get('returned_title') or row.get('title') or 'title not reported'}; "
                f"type={row.get('product_type') or 'not reported'}; "
                f"datum={row.get('vertical_datum') or 'not established'} "
                f"{row.get('vertical_units') or ''}; "
                f"raster={row.get('raster_available', 'not reported')}; "
                f"relation table={row.get('relation_table_available', 'not reported')})"
            )
            for row in bathy_by_profile[pid]
        ]
        next_phase = {
            "great_salt": (
                "reconcile NGVD29 gauge elevations with NAVD88 topobathymetry and "
                "define independent north/south-arm handling"
            ),
            "lake_abert": (
                "identify an authoritative bathymetry or capacity source before "
                "attempting anything beyond relative change"
            ),
            "bear_lake": (
                "document the USGS-to-E-A-V vertical relationship, then test the "
                "official capacity relation with uncertainty"
            ),
            "lake_powell": (
                "document how RISE elevation/storage relate to the NAVD88 capacity "
                "product before using the relation"
            ),
            "flaming_gorge": (
                "retain RISE item 337 as the direct Level-4 source and validate the "
                "ACAP datum/table as an independent future comparison"
            ),
        }.get(
            pid,
            "resolve the remaining source-identity and vertical-reference checks",
        )
        lines.extend(
            [
                f"## {capability.display_name}",
                "",
                f"- Profile / HydroLAKES ID: `{pid}` / `{capability.Hylak_id}`",
                (
                    "- Capability: current level "
                    f"**{int(capability.highest_current_capability)}**; potential "
                    f"level **{int(capability.highest_potential_capability)}**; "
                    "source-quality class "
                    f"**{capability.source_quality_class.value if capability.source_quality_class else 'unassigned'}** "
                    f"({'confirmed current' if capability.absolute_storage_available else 'potential candidate'})"
                ),
                f"- Confirmed sources: {', '.join(capability.confirmed_sources) or 'none'}",
                f"- Unverified/review sources: {', '.join(capability.unverified_sources) or 'none'}",
                (
                    "- Source coverage: "
                    f"{date_starts[0] if date_starts else 'not reported'} to "
                    f"{date_ends[-1] if date_ends else 'not reported'}; "
                    f"provisional flags: {', '.join(provisional) or 'not reported'}"
                ),
                f"- Vertical datum: {capability.datum_status}",
                (
                    "- Bathymetry/capacity metadata: "
                    + ("; ".join(bathymetry) or "none configured")
                ),
                f"- Blocking gap: {'; '.join(capability.missing_requirements) or 'none recorded'}",
                f"- Key warning: {capability.key_warning or 'none'}",
                "- Source details:",
                *(
                    [
                        (
                            f"  - `{row['source_id']}` / item "
                            f"`{row.get('source_item_id') or 'not selected'}`: "
                            f"{row.get('title') or 'title not reported'}; "
                            f"{row.get('variable') or 'unclassified'}, "
                            f"{row.get('unit') or 'unit not reported'}; "
                            f"datum={row.get('vertical_datum') or 'not established'}; "
                            f"observed {row.get('observed_start') or 'not retrieved'} "
                            f"to {row.get('observed_end') or 'not retrieved'}; "
                            f"freshness {row.get('freshness_days') if row.get('freshness_days') is not None else 'not known'} days; "
                            f"provisional={row.get('provisional')}; "
                            f"approval={row.get('approval_status') or 'not reported'}"
                        )
                        for row in sources
                        if row.get("selection_status")
                        in {"selected", "configured_item", "record_discovery"}
                        and row.get("status")
                        in {
                            "confirmed",
                            "confirmed_series_no_recent_observations",
                            "no_observations_in_window",
                            "failed",
                        }
                    ]
                    or ["  - No source produced a validated bounded preview."]
                ),
                (
                    "- Probe failures/review: "
                    + (
                        "; ".join(
                            f"{row['source_id']}: {row['status']}"
                            for row in probes
                            if row["status"]
                            not in {
                                "confirmed",
                                "confirmed_metadata_only",
                                "requires_crosswalk",
                            }
                        )
                        or "none"
                    )
                ),
                f"- Next phase: {next_phase}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Cross-cutting limitations",
            "",
            (
                "The inventory is a five-lake capability probe, not a validated "
                "storage product. Provider provisional flags are retained. No "
                "SWOT availability is claimed without a PLD crosswalk, and no "
                "HydroLAKES modeled depth or volume promotes a capability level."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_inventory_outputs(
    bundle: InventoryBundle,
    *,
    output_dir: Path,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> dict[str, Path]:
    """Atomically write the seven required outputs and Markdown report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: output_dir / name for key, name in OUTPUT_FILENAMES.items()}
    _write_csv(
        paths["capability"],
        [_flatten_capability(record) for record in bundle.capability_records],
    )
    _write_csv(paths["sources"], bundle.source_rows)
    _write_csv(paths["vertical"], bundle.vertical_rows)
    _write_csv(paths["probes"], bundle.probe_rows)
    _write_csv(
        paths["preview"],
        [_flatten_preview(record) for record in bundle.preview_records],
        fieldnames=PREVIEW_COLUMNS,
    )
    _write_csv(paths["bathymetry"], bundle.bathymetry_rows)

    cache_dir = Path(str(bundle.provenance["cache_dir"]))
    linked_manifests = CacheStore(cache_dir).link_normalized_output(
        paths["provenance"],
        raw_response_paths=bundle.provenance.get("cache_raw_responses_used", ()),
    )

    output_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in paths.items()
        if key != "provenance"
    }
    provenance = dict(bundle.provenance)
    provenance["output_sha256"] = output_hashes
    provenance["cache_manifests_linked_to_normalized_output"] = linked_manifests
    ensure_no_sensitive_keys(provenance, context="written provenance")
    _atomic_text(
        paths["provenance"], json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    report_path = report_dir / "capability_report.md"
    _atomic_text(report_path, render_capability_report(bundle))
    paths["report"] = report_path
    return paths


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected YYYY-MM-DD date, got {value!r}."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe five pilot hydrometry sources and write a conservative Phase 0 "
            "capability inventory. No volume is calculated."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--live", action="store_true", help="Allow bounded HTTP requests."
    )
    mode.add_argument(
        "--offline",
        action="store_true",
        help="Use valid cache entries only; never access the network.",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Refresh valid caches in live mode."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Recent inclusive window length (default: 90).",
    )
    parser.add_argument(
        "--start", type=_parse_date, help="Explicit start date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end", type=_parse_date, help="Explicit end date (YYYY-MM-DD)."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--profiles",
        nargs="+",
        help="Exact profile IDs to probe; all five are used when omitted.",
    )
    return parser


def _window(args: argparse.Namespace) -> tuple[date, date]:
    if args.days <= 0:
        raise InventoryError("--days must be positive")
    if (args.start is None) != (args.end is None):
        raise InventoryError("--start and --end must be supplied together")
    if args.start is not None:
        if args.start > args.end:
            raise InventoryError("--start must be on or before --end")
        return args.start, args.end
    end = datetime.now(UTC).date()
    return end - timedelta(days=args.days - 1), end


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.offline and args.refresh:
            raise InventoryError("--refresh cannot be combined with --offline")
        start, end = _window(args)
        registry = load_registry(args.config)
        mode = CacheMode.LIVE if args.live else CacheMode.OFFLINE
        bundle = run_inventory(
            registry,
            mode=mode,
            start=start,
            end=end,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
            profile_ids=args.profiles,
        )
        paths = write_inventory_outputs(bundle, output_dir=args.output_dir)
    except Exception as exc:  # noqa: BLE001 - CLI translates all failures cleanly
        parser.exit(2, f"hydrometry inventory failed: {_error_text(exc)}\n")
    print(
        f"Hydrometry Phase 0: {len(bundle.capability_records)} profile(s), "
        f"{bundle.provenance['request_count']} live request(s), "
        f"{len(bundle.preview_records)} preview row(s)."
    )
    for capability in bundle.capability_records:
        print(
            f"  {capability.display_name}: current L{int(capability.highest_current_capability)}, "
            f"potential L{int(capability.highest_potential_capability)}"
        )
    print(f"Wrote {len(paths)} files; no volume was calculated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
