"""Bounded SWOT Hydrocron capability probe for explicitly supplied PLD IDs.

Phase 0 does not discover or infer Prior Lake Database identifiers and does not
perform Earthdata authentication.  A missing PLD ID therefore produces a
structured ``requires_crosswalk`` result without making a network request.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from regional_analysis.hydrometry.adapters.base import BaseAdapter
from regional_analysis.hydrometry.models import (
    HeightType,
    HydrometryVariable,
    NormalizationStatus,
    NormalizedTimeSeriesRecord,
    Provider,
    VerticalReference,
)

HYDROCRON_TIMESERIES_URL = (
    "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
)
DEFAULT_LAKE_COLLECTION = "SWOT_L2_HR_LakeSP_D"
DEFAULT_FIELDS = (
    "lake_id",
    "time_str",
    "wse",
    "area_total",
    "quality_f",
    "collection_shortname",
    "collection_version",
    "crid",
    "PLD_version",
    "range_start_time",
)
PLD_ID_PATTERN = re.compile(r"^\d{10}$")


class HydrocronError(RuntimeError):
    """Base exception for a failed Hydrocron capability probe."""


class HydrocronSchemaError(HydrocronError):
    """Raised when Hydrocron returns a malformed payload."""


class HydrocronIdentityError(HydrocronError):
    """Raised when response lake IDs differ from the configured PLD ID."""


@dataclass(frozen=True, slots=True)
class HydrocronObservation:
    """One normalized-enough SWOT prior-lake observation."""

    swot_pld_id: str
    timestamp_utc: datetime
    wse: float | None
    wse_unit: str | None
    area_total: float | None
    area_unit: str | None
    quality_flag: int | None
    collection_shortname: str | None
    collection_version: str | None
    crid: str | None
    pld_version: str | None
    range_start_time: datetime | None


@dataclass(frozen=True, slots=True)
class HydrocronProbeResult:
    """Capability status plus observations, with no HydroLAKES/PLD conflation."""

    status: str
    swot_pld_id: str | None
    collection_name: str | None
    product_versions: tuple[str, ...]
    observations: tuple[HydrocronObservation, ...]
    message: str
    source_url_public: str | None


def validate_pld_id(swot_pld_id: str) -> str:
    """Validate the documented ten-digit PriorLake identifier format."""

    if not isinstance(swot_pld_id, str) or not PLD_ID_PATTERN.fullmatch(
        swot_pld_id.strip()
    ):
        raise ValueError(
            "SWOT PriorLake IDs must be explicit ten-digit PLD strings; a "
            "HydroLAKES Hylak_id is not interchangeable."
        )
    return swot_pld_id.strip()


def build_hydrocron_query(
    swot_pld_id: str,
    *,
    start: datetime,
    end: datetime,
    collection_name: str = DEFAULT_LAKE_COLLECTION,
    fields: Sequence[str] = DEFAULT_FIELDS,
) -> dict[str, str]:
    """Construct one bounded, noncompact GeoJSON query."""

    pld_id = validate_pld_id(swot_pld_id)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Hydrocron start/end must be timezone-aware.")
    start_utc = start.astimezone(UTC)
    end_utc = end.astimezone(UTC)
    if start_utc > end_utc:
        raise ValueError("Hydrocron start must be on or before end.")
    if not collection_name.strip():
        raise ValueError("Hydrocron collection_name must be non-empty.")
    clean_fields = tuple(str(field).strip() for field in fields)
    if not clean_fields or any(not field for field in clean_fields):
        raise ValueError("Hydrocron fields must be non-empty.")
    required = {"lake_id", "time_str", "collection_shortname", "PLD_version"}
    if not required.issubset(clean_fields):
        raise ValueError(
            "Hydrocron fields must retain identity, time, collection, and "
            "PLD-version provenance."
        )
    return {
        "feature": "PriorLake",
        "feature_id": pld_id,
        "start_time": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output": "geojson",
        "compact": "false",
        "collection_name": collection_name.strip(),
        "fields": ",".join(clean_fields),
    }


def _optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip() or None


def _optional_float(value: Any, label: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HydrocronSchemaError(f"{label} must be numeric or null.") from exc
    if not math.isfinite(result):
        raise HydrocronSchemaError(f"{label} must be finite or null.")
    # SWOT fill values are not observations; retain a missing value rather than
    # pretending the documented large negative sentinel is a physical height.
    if result <= -9.0e11:
        return None
    return result


def _optional_integer(value: Any, label: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HydrocronSchemaError(f"{label} must be an integer or null.") from exc


def _optional_datetime(value: Any, label: str) -> datetime | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise HydrocronSchemaError(f"{label} must be an ISO timestamp or null.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HydrocronSchemaError(f"Invalid {label}: {value!r}.") from exc
    if parsed.tzinfo is None:
        raise HydrocronSchemaError(f"{label} must include a UTC offset.")
    return parsed.astimezone(UTC)


def parse_hydrocron_response(
    payload: Mapping[str, Any],
    *,
    expected_pld_id: str,
) -> tuple[HydrocronObservation, ...]:
    """Parse the documented noncompact GeoJSON response envelope."""

    pld_id = validate_pld_id(expected_pld_id)
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise HydrocronSchemaError("Hydrocron payload is missing results.")
    geojson = results.get("geojson")
    if not isinstance(geojson, Mapping) or geojson.get("type") != "FeatureCollection":
        raise HydrocronSchemaError(
            "Hydrocron results.geojson must be a FeatureCollection."
        )
    features = geojson.get("features")
    if not isinstance(features, list):
        raise HydrocronSchemaError("Hydrocron FeatureCollection needs features.")

    observations: list[HydrocronObservation] = []
    for index, feature in enumerate(features):
        if not isinstance(feature, Mapping) or feature.get("type") != "Feature":
            raise HydrocronSchemaError(f"Hydrocron feature {index} is malformed.")
        properties = feature.get("properties")
        if not isinstance(properties, Mapping):
            raise HydrocronSchemaError(
                f"Hydrocron feature {index} is missing properties."
            )
        returned_id = str(properties.get("lake_id", "")).strip()
        if returned_id != pld_id:
            raise HydrocronIdentityError(
                f"Hydrocron returned lake_id {returned_id!r}; expected PLD ID "
                f"{pld_id!r}."
            )
        timestamp = _optional_datetime(properties.get("time_str"), "time_str")
        if timestamp is None:
            raise HydrocronSchemaError(
                f"Hydrocron feature {index} is missing time_str."
            )
        observations.append(
            HydrocronObservation(
                swot_pld_id=pld_id,
                timestamp_utc=timestamp,
                wse=_optional_float(properties.get("wse"), "wse"),
                wse_unit=_optional_string(properties.get("wse_units")),
                area_total=_optional_float(properties.get("area_total"), "area_total"),
                area_unit=_optional_string(properties.get("area_total_units")),
                quality_flag=_optional_integer(
                    properties.get("quality_f"), "quality_f"
                ),
                collection_shortname=_optional_string(
                    properties.get("collection_shortname")
                ),
                collection_version=_optional_string(
                    properties.get("collection_version")
                ),
                crid=_optional_string(properties.get("crid")),
                pld_version=_optional_string(properties.get("PLD_version")),
                range_start_time=_optional_datetime(
                    properties.get("range_start_time"), "range_start_time"
                ),
            )
        )
    return tuple(observations)


class HydrocronAdapter(BaseAdapter):
    """Capability-only Hydrocron adapter with explicit no-crosswalk behavior."""

    provider = Provider.HYDROCRON.value

    @staticmethod
    def normalize_observations(
        observations: Sequence[HydrocronObservation],
        *,
        profile_id: str,
        hylak_id: int,
        display_name: str,
        source_id: str,
        retrieved_at_utc: datetime,
        source_url_public: str,
    ) -> tuple[NormalizedTimeSeriesRecord, ...]:
        """Normalize available WSE and area fields without inventing values."""

        records: list[NormalizedTimeSeriesRecord] = []
        for observation in observations:
            common_metadata = {
                "quality_flag": observation.quality_flag,
                "collection_shortname": observation.collection_shortname,
                "collection_version": observation.collection_version,
                "crid": observation.crid,
                "pld_version": observation.pld_version,
                "range_start_time": (
                    observation.range_start_time.isoformat()
                    if observation.range_start_time
                    else None
                ),
            }
            source_version = observation.collection_version or observation.crid
            if observation.wse is not None and observation.wse_unit:
                reference = VerticalReference(
                    height_type=HeightType.GEOID_REFERENCED_HEIGHT,
                    source_vertical_datum=(
                        "SWOT product geoid reference; retain collection version"
                    ),
                    source_vertical_units=observation.wse_unit,
                    normalization_status=NormalizationStatus.SOURCE_VALUE_RETAINED,
                    notes=(
                        "Hydrocron WSE retained as published; no Phase 0 datum "
                        "transformation or storage calculation was attempted."
                    ),
                )
                records.append(
                    NormalizedTimeSeriesRecord(
                        profile_id=profile_id,
                        Hylak_id=hylak_id,
                        display_name=display_name,
                        provider=Provider.HYDROCRON,
                        source_id=source_id,
                        source_record_id=observation.swot_pld_id,
                        source_item_id=observation.collection_shortname,
                        variable=HydrometryVariable.WATER_LEVEL,
                        value=observation.wse,
                        unit=observation.wse_unit,
                        timestamp_utc=observation.timestamp_utc,
                        quality_status=(
                            str(observation.quality_flag)
                            if observation.quality_flag is not None
                            else "not_reported"
                        ),
                        provisional=None,
                        approval_status=None,
                        height_type=reference.height_type,
                        vertical_datum=reference,
                        retrieved_at_utc=retrieved_at_utc,
                        source_version=source_version,
                        source_url_public=source_url_public,
                        raw_parameter_code="wse",
                        raw_statistic_code=None,
                        raw_metadata=common_metadata,
                    )
                )
            if observation.area_total is not None and observation.area_unit:
                reference = VerticalReference(
                    height_type=HeightType.UNKNOWN,
                    source_vertical_datum=None,
                    source_vertical_units=observation.area_unit,
                    normalization_status=NormalizationStatus.SOURCE_VALUE_RETAINED,
                    notes="Hydrocron total area retained as published.",
                )
                records.append(
                    NormalizedTimeSeriesRecord(
                        profile_id=profile_id,
                        Hylak_id=hylak_id,
                        display_name=display_name,
                        provider=Provider.HYDROCRON,
                        source_id=source_id,
                        source_record_id=observation.swot_pld_id,
                        source_item_id=observation.collection_shortname,
                        variable=HydrometryVariable.SURFACE_AREA,
                        value=observation.area_total,
                        unit=observation.area_unit,
                        timestamp_utc=observation.timestamp_utc,
                        quality_status=(
                            str(observation.quality_flag)
                            if observation.quality_flag is not None
                            else "not_reported"
                        ),
                        provisional=None,
                        approval_status=None,
                        height_type=reference.height_type,
                        vertical_datum=reference,
                        retrieved_at_utc=retrieved_at_utc,
                        source_version=source_version,
                        source_url_public=source_url_public,
                        raw_parameter_code="area_total",
                        raw_statistic_code=None,
                        raw_metadata=common_metadata,
                    )
                )
        return tuple(records)

    def probe(
        self,
        *,
        swot_pld_id: str | None,
        start: datetime,
        end: datetime,
        collection_name: str = DEFAULT_LAKE_COLLECTION,
    ) -> HydrocronProbeResult:
        """Return crosswalk status or query one explicitly configured PLD ID."""

        if swot_pld_id is None:
            return HydrocronProbeResult(
                status="requires_crosswalk",
                swot_pld_id=None,
                collection_name=None,
                product_versions=(),
                observations=(),
                message=(
                    "No SWOT PLD ID is configured. HydroLAKES Hylak_id and SWOT "
                    "Prior Lake Database IDs are separate; no satellite "
                    "availability is confirmed."
                ),
                source_url_public=None,
            )
        query = build_hydrocron_query(
            swot_pld_id,
            start=start,
            end=end,
            collection_name=collection_name,
        )
        response = self._get(
            HYDROCRON_TIMESERIES_URL,
            params=query,
            headers={"Accept": "application/json"},
            response_schema="hydrocron-timeseries-envelope-v1",
        )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise HydrocronSchemaError("Hydrocron response must be a JSON object.")
        observations = parse_hydrocron_response(
            payload, expected_pld_id=validate_pld_id(swot_pld_id)
        )
        versions = tuple(
            sorted(
                {
                    value
                    for observation in observations
                    for value in (
                        observation.collection_version,
                        observation.crid,
                        observation.pld_version,
                    )
                    if value
                }
            )
        )
        return HydrocronProbeResult(
            status="confirmed" if observations else "no_observations",
            swot_pld_id=validate_pld_id(swot_pld_id),
            collection_name=collection_name,
            product_versions=versions,
            observations=observations,
            message=(
                f"Hydrocron returned {len(observations)} PriorLake observation(s); "
                "direct LakeSP and Hydrocron collection versions may differ."
            ),
            source_url_public=response.url_public,
        )
