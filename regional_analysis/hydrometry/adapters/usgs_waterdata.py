"""Modern Water Data for the Nation OGC API adapter.

Series are discovered from ``time-series-metadata`` and selected from returned
parameter descriptions, units, computation periods, and vertical references.
Configured parameter codes are assertions only; this module never assumes that
gage height is an absolute water-surface elevation.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any

from ..cache import CachedResponseData, CacheMode, CacheStore
from ..http import HttpClient, next_page_url
from ..models import (
    HeightType,
    HydrometryVariable,
    NormalizationStatus,
    NormalizedTimeSeriesRecord,
    Provider,
    VerticalReference,
)
from .base import (
    BaseAdapter,
    ProviderIdentityError,
    ProviderSchemaError,
    ProviderValidationError,
)

USGS_WATERDATA_BASE_URL = "https://api.waterdata.usgs.gov/ogcapi/v0"
USGS_PROVIDER = "usgs_waterdata"
USGS_SOURCE_VERSION = "ogcapi-v0"
MAX_SERIES_PER_LOCATION = 500
MAX_OBSERVATIONS = 20_000


class UsgsWaterDataError(RuntimeError):
    """Base exception for modern USGS Water Data operations."""


class UsgsSchemaError(UsgsWaterDataError, ProviderSchemaError):
    """Raised when a WDFN payload does not match the expected OGC schema."""


class UsgsIdentityError(UsgsWaterDataError, ProviderIdentityError):
    """Raised when a returned location or series has the wrong identity."""


class UsgsValidationError(UsgsWaterDataError, ProviderValidationError):
    """Raised when a selected series or observation violates metadata."""


class SeriesSelectionStatus(str, Enum):
    SELECTED = "selected"
    REQUIRES_REVIEW = "requires_review"
    NOT_AVAILABLE = "not_available"


class UsgsDataKind(str, Enum):
    DAILY = "daily"
    CONTINUOUS = "continuous"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class MonitoringLocation:
    location_id: str
    monitoring_location_number: str
    name: str
    site_type: str | None
    state_name: str | None
    longitude: float
    latitude: float
    altitude: float | None
    altitude_unit: str | None
    vertical_datum: str | None
    vertical_datum_name: str | None
    source_url_public: str


@dataclass(frozen=True, slots=True)
class UsgsParameterMetadata:
    parameter_code: str
    parameter_name: str
    parameter_description: str
    unit: str
    medium: str | None
    source_url_public: str


@dataclass(frozen=True, slots=True)
class UsgsSeriesMetadata:
    time_series_id: str
    monitoring_location_id: str
    parameter_code: str
    parameter_name: str
    parameter_description: str
    unit: str
    statistic_code: str | None
    computation_identifier: str | None
    computation_period_identifier: str | None
    begin_utc: datetime | None
    end_utc: datetime | None
    primary: bool | None
    web_description: str | None
    variable: HydrometryVariable | None
    height_type: HeightType
    vertical_datum: str | None
    data_kind: UsgsDataKind
    source_url_public: str

    @property
    def is_absolute_water_level(self) -> bool:
        return self.variable is HydrometryVariable.WATER_LEVEL and self.height_type in {
            HeightType.ORTHOMETRIC_HEIGHT,
            HeightType.GEOID_REFERENCED_HEIGHT,
        }


@dataclass(frozen=True, slots=True)
class SeriesSelection:
    status: SeriesSelectionStatus
    selected: UsgsSeriesMetadata | None
    candidates: tuple[UsgsSeriesMetadata, ...]
    reason: str

    @property
    def requires_review(self) -> bool:
        return self.status is SeriesSelectionStatus.REQUIRES_REVIEW


@dataclass(frozen=True, slots=True)
class UsgsObservation:
    time_series_id: str
    monitoring_location_id: str
    timestamp_utc: datetime
    value: float
    unit: str
    parameter_code: str
    statistic_code: str | None
    approval_status: str | None
    provisional: bool | None
    qualifier: str | None
    last_modified_utc: datetime | None


def _properties(
    feature: Any, label: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(feature, Mapping) or feature.get("type") != "Feature":
        raise UsgsSchemaError(f"{label} must be a GeoJSON Feature.")
    properties = feature.get("properties")
    if not isinstance(properties, Mapping):
        raise UsgsSchemaError(f"{label}.properties must be an object.")
    geometry = feature.get("geometry")
    if geometry is None:
        geometry = {}
    if not isinstance(geometry, Mapping):
        raise UsgsSchemaError(f"{label}.geometry must be an object or null.")
    return properties, geometry


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UsgsSchemaError(f"{label} must be a non-empty string.")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any, label: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise UsgsSchemaError(f"{label} must be numeric or null.") from exc
    if not math.isfinite(number):
        raise UsgsSchemaError(f"{label} must be finite.")
    return number


def _datetime(value: Any, label: str) -> datetime | None:
    if value is None or value == "":
        return None
    text = _required_string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise UsgsSchemaError(f"{label} is not an ISO-8601 timestamp: {text}") from exc
    if parsed.tzinfo is None:
        # Metadata begin/end values can omit an offset.  WDFN exposes parallel
        # *_utc fields where possible; a bare value is treated as UTC only when
        # that is the sole provider representation.
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _normalize_datum(value: str | None) -> str | None:
    if value is None:
        return None
    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    if "NAVD88" in compact or "NAVD1988" in compact:
        return "NAVD88"
    if "NGVD29" in compact or "NGVD1929" in compact:
        return "NGVD29"
    if "PROJECTDATUM" in compact or "RESERVOIRDATUM" in compact:
        return "project_datum"
    return value.strip()


def _datum_from_description(value: str) -> str | None:
    """Extract only an explicit vertical reference from descriptive prose."""

    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    if "NAVD88" in compact or "NAVD1988" in compact:
        return "NAVD88"
    if "NGVD29" in compact or "NGVD1929" in compact:
        return "NGVD29"
    if "PROJECTDATUM" in compact or "RESERVOIRDATUM" in compact:
        return "project_datum"
    return None


def _classify_series(
    *, parameter_name: str, parameter_description: str, web_description: str | None
) -> tuple[HydrometryVariable | None, HeightType, str | None]:
    text = " ".join(
        part
        for part in (parameter_name, parameter_description, web_description)
        if part
    ).casefold()
    datum = _datum_from_description(text)
    if "gage height" in text or "gauge height" in text:
        return HydrometryVariable.GAGE_HEIGHT, HeightType.LOCAL_GAGE_HEIGHT, datum
    water_words = "water surface" in text or "lake level" in text
    if water_words and "elevation" in text:
        if "above datum" in text and datum not in {"NAVD88", "NGVD29"}:
            return (
                HydrometryVariable.RESERVOIR_ELEVATION,
                HeightType.PROJECT_DATUM_ELEVATION,
                datum or "project_datum",
            )
        if datum in {"NAVD88", "NGVD29"}:
            return HydrometryVariable.WATER_LEVEL, HeightType.ORTHOMETRIC_HEIGHT, datum
        return HydrometryVariable.WATER_LEVEL, HeightType.UNKNOWN, None
    return None, HeightType.UNKNOWN, datum


def _data_kind(period: str | None, computation: str | None) -> UsgsDataKind:
    period_text = (period or "").casefold()
    computation_text = (computation or "").casefold()
    if "daily" in period_text:
        return UsgsDataKind.DAILY
    if "point" in period_text or "instant" in computation_text:
        return UsgsDataKind.CONTINUOUS
    return UsgsDataKind.UNKNOWN


def parse_monitoring_location(
    feature: Any,
    *,
    expected_location_id: str,
    source_url_public: str,
) -> MonitoringLocation:
    """Parse and identity-check one monitoring-location feature."""

    properties, geometry = _properties(feature, "monitoring location")
    feature_id = _optional_string(feature.get("id"))
    location_id = feature_id or _required_string(
        properties.get("id"), "monitoring location id"
    )
    if location_id != expected_location_id:
        raise UsgsIdentityError(
            f"Requested {expected_location_id}, received monitoring location {location_id}."
        )
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, Sequence) or isinstance(coordinates, (str, bytes)):
        raise UsgsSchemaError("monitoring location must have point coordinates.")
    if len(coordinates) < 2:
        raise UsgsSchemaError("monitoring location coordinates are incomplete.")
    longitude = _optional_float(coordinates[0], "longitude")
    latitude = _optional_float(coordinates[1], "latitude")
    if longitude is None or latitude is None:
        raise UsgsSchemaError("monitoring location coordinates cannot be null.")
    if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
        raise UsgsSchemaError("monitoring location coordinates are out of range.")
    number = _required_string(
        properties.get("monitoring_location_number"),
        "monitoring_location_number",
    )
    expected_number = expected_location_id.removeprefix("USGS-")
    if number != expected_number:
        raise UsgsIdentityError(
            f"Location number {number} does not match {expected_location_id}."
        )
    return MonitoringLocation(
        location_id=location_id,
        monitoring_location_number=number,
        name=_required_string(properties.get("monitoring_location_name"), "name"),
        site_type=_optional_string(properties.get("site_type")),
        state_name=_optional_string(properties.get("state_name")),
        longitude=longitude,
        latitude=latitude,
        altitude=_optional_float(properties.get("altitude"), "altitude"),
        altitude_unit=_optional_string(
            properties.get("altitude_unit")
            or properties.get("altitude_unit_of_measure")
        ),
        vertical_datum=_normalize_datum(
            _optional_string(properties.get("vertical_datum"))
        ),
        vertical_datum_name=_optional_string(properties.get("vertical_datum_name")),
        source_url_public=source_url_public,
    )


def parse_parameter_metadata(
    feature: Any, *, expected_parameter_code: str, source_url_public: str
) -> UsgsParameterMetadata:
    properties, _geometry = _properties(feature, "parameter metadata")
    code = _optional_string(feature.get("id")) or _required_string(
        properties.get("id"), "parameter code"
    )
    if code != expected_parameter_code:
        raise UsgsIdentityError(
            f"Requested parameter {expected_parameter_code}, received {code}."
        )
    return UsgsParameterMetadata(
        parameter_code=code,
        parameter_name=_required_string(
            properties.get("parameter_name"), "parameter_name"
        ),
        parameter_description=_required_string(
            properties.get("parameter_description"), "parameter_description"
        ),
        unit=_required_string(properties.get("unit_of_measure"), "unit_of_measure"),
        medium=_optional_string(properties.get("medium")),
        source_url_public=source_url_public,
    )


def parse_series_metadata(
    feature: Any,
    *,
    expected_location_id: str,
    source_url_public: str,
) -> UsgsSeriesMetadata:
    """Parse one discovered daily/continuous time-series description."""

    properties, _geometry = _properties(feature, "time-series metadata")
    series_id = _optional_string(feature.get("id")) or _required_string(
        properties.get("id"), "time series id"
    )
    location_id = _required_string(
        properties.get("monitoring_location_id"), "monitoring_location_id"
    )
    if location_id != expected_location_id:
        raise UsgsIdentityError(
            f"Series {series_id} belongs to {location_id}, not {expected_location_id}."
        )
    description = _required_string(
        properties.get("parameter_description"), "parameter_description"
    )
    # Early v0 responses have differed on whether the short name appears as
    # ``parameter_name`` or ``parameter``; descriptions remain the semantic
    # authority and are required.  Falling back to the description is explicit,
    # not a guess at the variable.
    name = (
        _optional_string(properties.get("parameter_name"))
        or _optional_string(properties.get("parameter"))
        or description
    )
    web_description = _optional_string(properties.get("web_description"))
    variable, height_type, datum = _classify_series(
        parameter_name=name,
        parameter_description=description,
        web_description=web_description,
    )
    period = _optional_string(properties.get("computation_period_identifier"))
    computation = _optional_string(properties.get("computation_identifier"))
    primary_raw = properties.get("primary")
    primary: bool | None
    if primary_raw is None:
        primary = None
    elif isinstance(primary_raw, bool):
        primary = primary_raw
    else:
        primary = str(primary_raw).casefold() == "primary"
    return UsgsSeriesMetadata(
        time_series_id=series_id,
        monitoring_location_id=location_id,
        parameter_code=_required_string(
            properties.get("parameter_code"), "parameter_code"
        ),
        parameter_name=name,
        parameter_description=description,
        unit=_required_string(properties.get("unit_of_measure"), "unit_of_measure"),
        statistic_code=_optional_string(properties.get("statistic_id")),
        computation_identifier=computation,
        computation_period_identifier=period,
        begin_utc=_datetime(
            properties.get("begin_utc") or properties.get("begin"), "series begin"
        ),
        end_utc=_datetime(
            properties.get("end_utc") or properties.get("end"), "series end"
        ),
        primary=primary,
        web_description=web_description,
        variable=variable,
        height_type=height_type,
        vertical_datum=datum,
        data_kind=_data_kind(period, computation),
        source_url_public=source_url_public,
    )


def select_water_level_series(
    series: Sequence[UsgsSeriesMetadata],
    *,
    expected_unit: str | None = None,
    expected_vertical_datum: str | None = None,
    prefer_daily: bool = True,
) -> SeriesSelection:
    """Select a uniquely validated absolute level series or require review.

    Parameter codes contribute to provenance but never establish semantic
    suitability by themselves.  Gage-height and project-datum series remain
    distinct from an orthometric lake elevation.
    """

    candidates = [item for item in series if item.is_absolute_water_level]
    if expected_unit is not None:
        candidates = [
            item
            for item in candidates
            if item.unit.casefold() == expected_unit.casefold()
        ]
    normalized_expected = _normalize_datum(expected_vertical_datum)
    if normalized_expected is not None:
        candidates = [
            item for item in candidates if item.vertical_datum == normalized_expected
        ]
    if not candidates:
        return SeriesSelection(
            status=SeriesSelectionStatus.REQUIRES_REVIEW,
            selected=None,
            candidates=(),
            reason=(
                "No discovered series has validated absolute water-elevation "
                "semantics, expected units, and expected vertical datum."
            ),
        )

    if prefer_daily:
        daily = [item for item in candidates if item.data_kind is UsgsDataKind.DAILY]
        if daily:
            candidates = daily
    primary = [item for item in candidates if item.primary is True]
    if primary:
        candidates = primary
    # Prefer a mean daily statistic when the provider exposes several daily
    # summaries, but never use 00003 as a substitute for semantic validation.
    mean = [item for item in candidates if item.statistic_code == "00003"]
    if mean:
        candidates = mean
    ordered = tuple(sorted(candidates, key=lambda item: item.time_series_id))
    if len(ordered) != 1:
        return SeriesSelection(
            status=SeriesSelectionStatus.REQUIRES_REVIEW,
            selected=None,
            candidates=ordered,
            reason=(
                f"{len(ordered)} equally preferred absolute level series remain; "
                "no series was guessed."
            ),
        )
    return SeriesSelection(
        status=SeriesSelectionStatus.SELECTED,
        selected=ordered[0],
        candidates=ordered,
        reason="One daily/continuous series uniquely passed semantic, unit, and datum validation.",
    )


def _validate_observation_identity(
    properties: Mapping[str, Any],
    expected_series: UsgsSeriesMetadata,
) -> tuple[str, str, str, str, str | None]:
    """Validate provider identity fields before accepting or skipping a row."""

    series_id = _required_string(
        properties.get("time_series_id"), "observation.time_series_id"
    )
    if series_id != expected_series.time_series_id:
        raise UsgsIdentityError(
            f"Observation belongs to series {series_id}, not "
            f"{expected_series.time_series_id}."
        )
    location_id = _required_string(
        properties.get("monitoring_location_id"),
        "observation.monitoring_location_id",
    )
    if location_id != expected_series.monitoring_location_id:
        raise UsgsIdentityError(
            f"Observation belongs to {location_id}, not "
            f"{expected_series.monitoring_location_id}."
        )
    parameter = _required_string(properties.get("parameter_code"), "parameter_code")
    unit = _required_string(properties.get("unit_of_measure"), "unit_of_measure")
    statistic = _optional_string(properties.get("statistic_id"))
    if parameter != expected_series.parameter_code:
        raise UsgsValidationError(
            f"Observation parameter {parameter} != {expected_series.parameter_code}."
        )
    if unit.casefold() != expected_series.unit.casefold():
        raise UsgsValidationError(
            f"Observation unit {unit!r} != series unit {expected_series.unit!r}."
        )
    if statistic != expected_series.statistic_code:
        raise UsgsValidationError(
            f"Observation statistic {statistic!r} != series statistic "
            f"{expected_series.statistic_code!r}."
        )
    return series_id, location_id, parameter, unit, statistic


def parse_observation(
    feature: Any,
    *,
    expected_series: UsgsSeriesMetadata,
) -> UsgsObservation:
    properties, _geometry = _properties(feature, "USGS observation")
    series_id, location_id, parameter, unit, statistic = _validate_observation_identity(
        properties, expected_series
    )
    value = _optional_float(properties.get("value"), "observation.value")
    if value is None:
        raise UsgsValidationError("Observation value is null.")
    approval = _optional_string(properties.get("approval_status"))
    provisional = None if approval is None else approval.casefold() == "provisional"
    timestamp = _datetime(properties.get("time"), "observation.time")
    if timestamp is None:
        raise UsgsSchemaError("observation.time cannot be null")
    return UsgsObservation(
        time_series_id=series_id,
        monitoring_location_id=location_id,
        timestamp_utc=timestamp,
        value=value,
        unit=unit,
        parameter_code=parameter,
        statistic_code=statistic,
        approval_status=approval,
        provisional=provisional,
        qualifier=_optional_string(properties.get("qualifier")),
        last_modified_utc=_datetime(properties.get("last_modified"), "last_modified"),
    )


class UsgsWaterDataAdapter(BaseAdapter):
    """Bounded modern WDFN adapter with discovery-first series selection."""

    provider = USGS_PROVIDER

    def __init__(
        self,
        *,
        client: HttpClient,
        mode: CacheMode,
        cache: CacheStore | None = None,
        refresh: bool = False,
        api_key: str | None = None,
        base_url: str = USGS_WATERDATA_BASE_URL,
    ) -> None:
        super().__init__(client=client, mode=mode, cache=cache, refresh=refresh)
        self.base_url = base_url.rstrip("/")
        self._headers = {"Accept": "application/geo+json"}
        key = api_key if api_key is not None else os.environ.get("WDFN_API_KEY")
        if key:
            self._headers["X-Api-Key"] = key

    def _items_url(self, collection: str) -> str:
        return f"{self.base_url}/collections/{collection}/items"

    def _json(self, response: CachedResponseData) -> Mapping[str, Any]:
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise UsgsSchemaError(
                f"Payload from {response.url_public} is not an object."
            )
        return payload

    def _feature_pages(
        self,
        collection: str,
        *,
        params: dict[str, object],
        max_pages: int,
        max_features: int,
    ) -> Iterator[tuple[Any, str]]:
        if max_pages <= 0 or max_features <= 0:
            raise ValueError("max_pages and max_features must be positive")
        url: str | None = self._items_url(collection)
        page_params: dict[str, object] | None = {"f": "json", **params}
        seen: set[str] = set()
        count = 0
        for _page in range(max_pages):
            if url is None:
                return
            response = self._get(
                url,
                params=page_params,
                headers=self._headers,
                response_schema=f"wdfn-{collection}-{USGS_SOURCE_VERSION}",
            )
            if response.url_public in seen:
                raise UsgsSchemaError(f"Cyclic pagination link: {response.url_public}")
            seen.add(response.url_public)
            payload = self._json(response)
            features = payload.get("features")
            if not isinstance(features, list):
                raise UsgsSchemaError(f"{collection} features must be an array.")
            count += len(features)
            if count > max_features:
                raise UsgsValidationError(
                    f"{collection} query exceeds bounded maximum of {max_features} rows."
                )
            for feature in features:
                yield feature, response.url_public
            url = next_page_url(payload)
            page_params = None
        if url is not None:
            raise UsgsValidationError(
                f"{collection} query exceeded configured maximum of {max_pages} pages."
            )

    def fetch_monitoring_location(self, location_id: str) -> MonitoringLocation:
        url = f"{self._items_url('monitoring-locations')}/{location_id}"
        response = self._get(
            url,
            params={"f": "json"},
            headers=self._headers,
            response_schema="wdfn-monitoring-location-ogcapi-v0",
        )
        payload = self._json(response)
        return parse_monitoring_location(
            payload,
            expected_location_id=location_id,
            source_url_public=response.url_public,
        )

    def fetch_parameter_metadata(self, parameter_code: str) -> UsgsParameterMetadata:
        url = f"{self._items_url('parameter-codes')}/{parameter_code}"
        response = self._get(
            url,
            params={"f": "json"},
            headers=self._headers,
            response_schema="wdfn-parameter-code-ogcapi-v0",
        )
        return parse_parameter_metadata(
            self._json(response),
            expected_parameter_code=parameter_code,
            source_url_public=response.url_public,
        )

    def discover_series(
        self,
        location_id: str,
        *,
        max_series: int = MAX_SERIES_PER_LOCATION,
        max_pages: int = 10,
    ) -> tuple[UsgsSeriesMetadata, ...]:
        rows = [
            parse_series_metadata(
                feature,
                expected_location_id=location_id,
                source_url_public=url,
            )
            for feature, url in self._feature_pages(
                "time-series-metadata",
                params={"monitoring_location_id": location_id, "limit": max_series},
                max_pages=max_pages,
                max_features=max_series,
            )
        ]
        ids = [item.time_series_id for item in rows]
        if len(ids) != len(set(ids)):
            raise UsgsSchemaError(
                f"Duplicate time-series IDs returned for {location_id}."
            )
        return tuple(sorted(rows, key=lambda item: item.time_series_id))

    def fetch_observations(
        self,
        series: UsgsSeriesMetadata,
        *,
        start: date,
        end: date,
        page_size: int = 1_000,
        max_pages: int = 20,
        max_observations: int = MAX_OBSERVATIONS,
    ) -> tuple[UsgsObservation, ...]:
        if start > end:
            raise ValueError("start must be on or before end")
        if page_size <= 0 or page_size > 10_000:
            raise ValueError("page_size must be between 1 and 10000")
        if series.data_kind is UsgsDataKind.UNKNOWN:
            raise UsgsValidationError(
                f"Series {series.time_series_id} is neither daily nor continuous."
            )
        collection = series.data_kind.value
        observations_list: list[UsgsObservation] = []
        for feature, _url in self._feature_pages(
            collection,
            params={
                "time_series_id": series.time_series_id,
                "datetime": f"{start.isoformat()}/{end.isoformat()}",
                "limit": page_size,
            },
            max_pages=max_pages,
            max_features=max_observations,
        ):
            properties, _geometry = _properties(feature, "USGS observation")
            _validate_observation_identity(properties, series)
            if _optional_float(properties.get("value"), "observation.value") is None:
                # WDFN may return an otherwise valid observation timestamp with
                # no measured value. It is not a zero and must not be normalized.
                continue
            observations_list.append(parse_observation(feature, expected_series=series))
        observations = tuple(observations_list)
        timestamps = [item.timestamp_utc for item in observations]
        if timestamps != sorted(timestamps):
            observations = tuple(
                sorted(observations, key=lambda item: item.timestamp_utc)
            )
        return observations

    @staticmethod
    def vertical_reference(
        location: MonitoringLocation,
        series: UsgsSeriesMetadata,
        *,
        target_vertical_datum: str | None = None,
    ) -> VerticalReference:
        source_datum = series.vertical_datum or location.vertical_datum
        notes = (
            f"Series-derived datum={series.vertical_datum or 'unknown'}; "
            f"monitoring-location datum={location.vertical_datum or 'unknown'}. "
            "Phase 0 retains the source value and performs no transformation."
        )
        return VerticalReference(
            height_type=series.height_type,
            source_vertical_datum=source_datum,
            source_vertical_units=series.unit,
            target_vertical_datum=target_vertical_datum,
            normalization_status=NormalizationStatus.SOURCE_VALUE_RETAINED,
            notes=notes,
        )

    @staticmethod
    def normalize_observations(
        observations: Sequence[UsgsObservation],
        *,
        series: UsgsSeriesMetadata,
        location: MonitoringLocation,
        profile_id: str,
        hylak_id: int,
        display_name: str,
        source_id: str,
        retrieved_at_utc: datetime,
        target_vertical_datum: str | None = None,
    ) -> tuple[NormalizedTimeSeriesRecord, ...]:
        if series.variable not in {
            HydrometryVariable.WATER_LEVEL,
            HydrometryVariable.GAGE_HEIGHT,
            HydrometryVariable.RESERVOIR_ELEVATION,
        }:
            raise UsgsValidationError(
                f"Series {series.time_series_id} has no validated level variable."
            )
        reference = UsgsWaterDataAdapter.vertical_reference(
            location, series, target_vertical_datum=target_vertical_datum
        )
        return tuple(
            NormalizedTimeSeriesRecord(
                profile_id=profile_id,
                Hylak_id=hylak_id,
                display_name=display_name,
                provider=Provider.USGS_WATERDATA,
                source_id=source_id,
                source_record_id=location.location_id,
                source_item_id=series.time_series_id,
                variable=series.variable,
                value=observation.value,
                unit=observation.unit,
                timestamp_utc=observation.timestamp_utc,
                quality_status=observation.qualifier or "provider_quality_not_reported",
                provisional=observation.provisional,
                approval_status=observation.approval_status,
                height_type=series.height_type,
                vertical_datum=reference,
                retrieved_at_utc=retrieved_at_utc,
                source_version=USGS_SOURCE_VERSION,
                source_url_public=series.source_url_public,
                raw_parameter_code=observation.parameter_code,
                raw_statistic_code=observation.statistic_code,
                raw_metadata=MappingProxyType(
                    {
                        "qualifier": observation.qualifier,
                        "last_modified_utc": (
                            observation.last_modified_utc.isoformat()
                            if observation.last_modified_utc
                            else None
                        ),
                        "series_vertical_datum": series.vertical_datum,
                        "site_vertical_datum": location.vertical_datum,
                        "data_kind": series.data_kind.value,
                    }
                ),
            )
            for observation in observations
        )
