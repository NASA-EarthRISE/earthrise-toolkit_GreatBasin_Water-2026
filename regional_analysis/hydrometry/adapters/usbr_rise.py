"""Bounded adapter for the USBR Reclamation Information Sharing Environment.

RISE exposes JSON:API resources.  This adapter keeps provider dictionaries
inside the parsing boundary and returns immutable records.  It deliberately
validates both the requested identifier and every returned result row because
unknown RISE query parameters can be ignored by the service without an error.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from types import MappingProxyType
from typing import Any

from regional_analysis.hydrometry.adapters.base import BaseAdapter
from regional_analysis.hydrometry.models import (
    HeightType,
    HydrometryVariable,
    NormalizedTimeSeriesRecord,
    Provider,
    VerticalReference,
)

RISE_BASE_URL = "https://data.usbr.gov/rise/api"
RISE_ACCEPT = "application/vnd.api+json"
RISE_HEADERS = {"Accept": RISE_ACCEPT}
PROVISIONAL_PHRASE = "provisional and subject to revision"


class RiseError(RuntimeError):
    """Base exception for a failed RISE operation."""


class RiseSchemaError(RiseError):
    """Raised when a response does not match the required JSON:API schema."""


class RiseIdentityError(RiseError):
    """Raised when a record, item, location, or result identity is wrong."""


class RiseUnitError(RiseError):
    """Raised when a configured item returns an unexpected unit."""


@dataclass(frozen=True, slots=True)
class RiseCatalogRecord:
    """Validated RISE catalog-record metadata."""

    record_id: int
    title: str
    description: str
    generation_effort: str | None
    location_id: int
    item_ids: tuple[int, ...]
    updated_at: datetime | None
    source_url_public: str


@dataclass(frozen=True, slots=True)
class RiseCatalogItem:
    """One RISE catalog item with an explicit normalized variable."""

    item_id: int
    record_id: int
    title: str
    description: str
    variable: HydrometryVariable | None
    provider_variable: str | None
    parameter_name: str | None
    parameter_group: str | None
    parameter_id: int | None
    unit: str | None
    timestep: str | None
    transformation: str | None
    temporal_start: datetime | None
    temporal_end: datetime | None
    update_frequency: str | None
    updated_at: datetime | None
    vertical_datum: str | None
    provisional: bool
    disclaimer: str
    data_structure: str | None
    binary_file_url: str | None
    validation_issues: tuple[str, ...]
    source_url_public: str

    @property
    def requires_review(self) -> bool:
        """Return whether the item cannot be selected without review."""

        return self.variable is None or bool(self.validation_issues)


@dataclass(frozen=True, slots=True)
class RiseObservation:
    """One validated observed result from a specific RISE item."""

    result_id: int
    item_id: int
    location_id: int | None
    parameter_id: int | None
    timestamp_utc: datetime
    value: float | None
    status: str | None
    source_code: str | None
    created_at: datetime | None
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class RiseLocation:
    """Validated RISE location metadata used for source identity checks."""

    location_id: int
    name: str
    location_type: str | None
    longitude: float | None
    latitude: float | None
    horizontal_datum: str | None
    vertical_datum: str | None
    states: tuple[str, ...]
    record_ids: tuple[int, ...]
    source_url_public: str


@dataclass(frozen=True, slots=True)
class RiseItemInventory:
    """Record-level discovery result without an implicit item choice."""

    record: RiseCatalogRecord
    items: tuple[RiseCatalogItem, ...]
    candidates_by_variable: Mapping[HydrometryVariable, tuple[int, ...]]
    requires_review: bool


@dataclass(frozen=True, slots=True)
class RiseValidatedSource:
    """Record, location, and item metadata validated as one source chain."""

    record: RiseCatalogRecord
    location: RiseLocation
    item: RiseCatalogItem


_PARAMETER_RULES: tuple[tuple[str, HydrometryVariable, frozenset[str]], ...] = (
    (
        "lake/reservoir elevation",
        HydrometryVariable.RESERVOIR_ELEVATION,
        frozenset({"ft", "m"}),
    ),
    (
        "lake/reservoir area",
        HydrometryVariable.SURFACE_AREA,
        frozenset({"acres", "acre", "m2", "km2"}),
    ),
    (
        "lake/reservoir storage",
        HydrometryVariable.STORAGE,
        frozenset({"af", "acre-ft", "acre-feet", "m3"}),
    ),
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RiseSchemaError(f"{label} must be a JSON object.")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise RiseSchemaError(f"{label} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RiseSchemaError(f"{label} must be an integer.") from exc
    return result


def _resource_id(resource: Mapping[str, Any], kind: str) -> int:
    expected = f"/rise/api/{kind}/"
    identifier = resource.get("id")
    if not isinstance(identifier, str) or not identifier.startswith(expected):
        raise RiseSchemaError(
            f"Expected a {kind!r} JSON:API identifier, got {identifier!r}."
        )
    return _integer(identifier.removeprefix(expected), f"{kind} id")


def _optional_datetime(value: Any, label: str) -> datetime | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise RiseSchemaError(f"{label} must be an ISO timestamp or null.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RiseSchemaError(f"Invalid {label}: {value!r}.") from exc
    if parsed.tzinfo is None:
        raise RiseSchemaError(f"{label} must include a UTC offset.")
    return parsed.astimezone(UTC)


def _relationship_id(
    relationships: Mapping[str, Any],
    name: str,
    kind: str,
) -> int:
    relationship = _mapping(relationships.get(name), f"relationships.{name}")
    data = _mapping(relationship.get("data"), f"relationships.{name}.data")
    return _resource_id(data, kind)


def _single_resource(payload: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise RiseSchemaError(f"RISE {kind} response must contain one data object.")
    actual_type = data.get("type")
    expected_type = "".join(part.title() for part in kind.split("-"))
    if actual_type != expected_type:
        raise RiseSchemaError(
            f"Expected JSON:API type {expected_type!r}, got {actual_type!r}."
        )
    return data


def _parameter_variable(
    parameter_name: Any,
    unit: Any,
) -> tuple[HydrometryVariable | None, tuple[str, ...]]:
    if not isinstance(parameter_name, str) or not parameter_name.strip():
        return None, ("missing_parameter_name",)
    if not isinstance(unit, str) or not unit.strip():
        return None, ("missing_parameter_unit",)

    normalized_name = " ".join(parameter_name.lower().split())
    normalized_unit = unit.lower().replace("²", "2").replace("³", "3").strip()
    for expected_name, variable, allowed_units in _PARAMETER_RULES:
        if normalized_name == expected_name:
            if normalized_unit not in allowed_units:
                return None, (f"unexpected_unit:{unit}",)
            return variable, ()

    # Bank storage is intentionally not silently equated to reservoir storage.
    if normalized_name == "lake/reservoir bank storage":
        if normalized_unit not in {"af", "acre-ft", "acre-feet", "m3"}:
            return None, (f"unexpected_unit:{unit}",)
        return HydrometryVariable.BANK_STORAGE, ()
    if normalized_name == "lake/reservoir change in storage":
        if normalized_unit not in {"af", "acre-ft", "acre-feet", "m3"}:
            return None, (f"unexpected_unit:{unit}",)
        return HydrometryVariable.RELATIVE_STORAGE_CHANGE, ()
    return None, (f"unrecognized_parameter:{parameter_name}",)


def _provider_variable_name(parameter_name: Any) -> str | None:
    """Retain distinctions that have no safe shared-model equivalent."""

    if not isinstance(parameter_name, str):
        return None
    normalized = " ".join(parameter_name.lower().split())
    return {
        "lake/reservoir elevation": "reservoir_elevation",
        "lake/reservoir area": "surface_area",
        "lake/reservoir storage": "total_reservoir_storage",
        "lake/reservoir bank storage": "bank_storage",
        "lake/reservoir change in storage": "storage_change",
    }.get(normalized, normalized.replace("/", "_").replace(" ", "_"))


def _canonical_unit(value: str) -> str:
    normalized = value.casefold().strip().replace("²", "2").replace("³", "3")
    aliases = {
        "acre-feet": "af",
        "acre-foot": "af",
        "acre-ft": "af",
        "acre foot": "af",
        "acre feet": "af",
        "acre": "acres",
    }
    return aliases.get(normalized, normalized)


def parse_catalog_record(
    payload: Mapping[str, Any],
    *,
    expected_record_id: int,
    expected_title_contains: str | None = None,
) -> RiseCatalogRecord:
    """Parse and validate one RISE catalog record."""

    resource = _single_resource(payload, "catalog-record")
    record_id = _resource_id(resource, "catalog-record")
    attributes = _mapping(resource.get("attributes"), "catalog record attributes")
    attribute_id = _integer(attributes.get("_id"), "catalog record attributes._id")
    if record_id != expected_record_id or attribute_id != expected_record_id:
        raise RiseIdentityError(
            "RISE catalog-record identity mismatch: "
            f"requested {expected_record_id}, response {record_id}/{attribute_id}."
        )
    title = attributes.get("recordTitle")
    if not isinstance(title, str) or not title.strip():
        raise RiseSchemaError("RISE catalog record is missing recordTitle.")
    if (
        expected_title_contains
        and expected_title_contains.casefold() not in title.casefold()
    ):
        raise RiseIdentityError(
            f"RISE record {record_id} title {title!r} does not identify "
            f"{expected_title_contains!r}."
        )

    relationships = _mapping(resource.get("relationships"), "catalog relationships")
    location_id = _relationship_id(relationships, "location", "location")
    items_relationship = _mapping(
        relationships.get("catalogItems"), "relationships.catalogItems"
    )
    item_resources = items_relationship.get("data")
    if not isinstance(item_resources, list):
        raise RiseSchemaError("relationships.catalogItems.data must be an array.")
    item_ids = tuple(
        _resource_id(_mapping(item, "catalog item relationship"), "catalog-item")
        for item in item_resources
    )
    if len(item_ids) != len(set(item_ids)):
        raise RiseSchemaError(f"RISE record {record_id} contains duplicate item IDs.")

    return RiseCatalogRecord(
        record_id=record_id,
        title=title.strip(),
        description=str(attributes.get("recordDescription") or "").strip(),
        generation_effort=(
            str(attributes["generationEffort"]).strip()
            if attributes.get("generationEffort")
            else None
        ),
        location_id=location_id,
        item_ids=item_ids,
        updated_at=_optional_datetime(
            attributes.get("updateDate"), "record updateDate"
        ),
        source_url_public=f"{RISE_BASE_URL}/catalog-record/{record_id}",
    )


def parse_catalog_item(
    payload: Mapping[str, Any],
    *,
    expected_item_id: int,
    expected_record_id: int,
    expected_location_name: str | None = None,
    expected_parameter_name: str | None = None,
    expected_unit: str | None = None,
) -> RiseCatalogItem:
    """Parse one catalog item and enforce configured identity/unit assertions."""

    resource = _single_resource(payload, "catalog-item")
    item_id = _resource_id(resource, "catalog-item")
    attributes = _mapping(resource.get("attributes"), "catalog item attributes")
    attribute_id = _integer(attributes.get("_id"), "catalog item attributes._id")
    if item_id != expected_item_id or attribute_id != expected_item_id:
        raise RiseIdentityError(
            "RISE catalog-item identity mismatch: "
            f"requested {expected_item_id}, response {item_id}/{attribute_id}."
        )
    relationships = _mapping(resource.get("relationships"), "item relationships")
    record_id = _relationship_id(relationships, "catalogRecord", "catalog-record")
    if record_id != expected_record_id:
        raise RiseIdentityError(
            f"RISE item {item_id} belongs to record {record_id}, not "
            f"configured record {expected_record_id}."
        )

    title = attributes.get("itemTitle")
    if not isinstance(title, str) or not title.strip():
        raise RiseSchemaError(f"RISE item {item_id} is missing itemTitle.")
    if (
        expected_location_name
        and expected_location_name.casefold() not in title.casefold()
    ):
        raise RiseIdentityError(
            f"RISE item {item_id} title {title!r} does not identify "
            f"{expected_location_name!r}."
        )

    parameter_name = attributes.get("parameterName")
    unit = attributes.get("parameterUnit")
    variable, validation_issues = _parameter_variable(parameter_name, unit)
    if expected_parameter_name and (
        not isinstance(parameter_name, str)
        or parameter_name.casefold() != expected_parameter_name.casefold()
    ):
        raise RiseIdentityError(
            f"RISE item {item_id} parameter is {parameter_name!r}, expected "
            f"{expected_parameter_name!r}."
        )
    if expected_unit and (
        not isinstance(unit, str)
        or _canonical_unit(unit) != _canonical_unit(expected_unit)
    ):
        raise RiseUnitError(
            f"RISE item {item_id} unit is {unit!r}, expected {expected_unit!r}."
        )

    disclaimer = str(attributes.get("disclaimer") or "").strip()
    update_frequency = attributes.get("updateFrequency")
    if isinstance(update_frequency, Mapping):
        update_frequency = update_frequency.get("updatefrequency")
    if update_frequency is not None:
        update_frequency = str(update_frequency).strip() or None

    return RiseCatalogItem(
        item_id=item_id,
        record_id=record_id,
        title=title.strip(),
        description=str(attributes.get("itemDescription") or "").strip(),
        variable=variable,
        provider_variable=_provider_variable_name(parameter_name),
        parameter_name=str(parameter_name).strip() if parameter_name else None,
        parameter_group=(
            str(attributes["parameterGroup"]).strip()
            if attributes.get("parameterGroup")
            else None
        ),
        parameter_id=(
            _integer(attributes["parameterId"], "parameterId")
            if attributes.get("parameterId") is not None
            else None
        ),
        unit=str(unit).strip() if unit else None,
        timestep=(
            str(attributes["parameterTimestep"]).strip()
            if attributes.get("parameterTimestep")
            else None
        ),
        transformation=(
            str(attributes["parameterTransformation"]).strip()
            if attributes.get("parameterTransformation")
            else None
        ),
        temporal_start=_optional_datetime(
            attributes.get("temporalStartDate"), "temporalStartDate"
        ),
        temporal_end=_optional_datetime(
            attributes.get("temporalEndDate"), "temporalEndDate"
        ),
        update_frequency=update_frequency,
        updated_at=_optional_datetime(attributes.get("updateDate"), "item updateDate"),
        vertical_datum=(
            str(attributes["verticalDatum"]).strip()
            if attributes.get("verticalDatum")
            else None
        ),
        provisional=PROVISIONAL_PHRASE in disclaimer.casefold(),
        disclaimer=disclaimer,
        data_structure=(
            str(attributes["dataStructure"]).strip()
            if attributes.get("dataStructure")
            else None
        ),
        binary_file_url=(
            str(attributes["binaryFilePath"]).strip()
            if attributes.get("binaryFilePath")
            else None
        ),
        validation_issues=validation_issues,
        source_url_public=f"{RISE_BASE_URL}/catalog-item/{item_id}",
    )


def parse_location(
    payload: Mapping[str, Any],
    *,
    expected_location_id: int,
    expected_name_contains: str | None = None,
    expected_record_id: int | None = None,
) -> RiseLocation:
    """Parse a RISE location and validate lake/record identity."""

    resource = _single_resource(payload, "location")
    location_id = _resource_id(resource, "location")
    attributes = _mapping(resource.get("attributes"), "location attributes")
    attribute_id = _integer(attributes.get("_id"), "location attributes._id")
    if location_id != expected_location_id or attribute_id != expected_location_id:
        raise RiseIdentityError(
            "RISE location identity mismatch: "
            f"requested {expected_location_id}, response {location_id}/{attribute_id}."
        )
    name = attributes.get("locationName")
    if not isinstance(name, str) or not name.strip():
        raise RiseSchemaError(f"RISE location {location_id} is missing locationName.")
    if (
        expected_name_contains
        and expected_name_contains.casefold() not in name.casefold()
    ):
        raise RiseIdentityError(
            f"RISE location {location_id} name {name!r} does not identify "
            f"{expected_name_contains!r}."
        )
    coordinates = attributes.get("locationCoordinates")
    longitude: float | None = None
    latitude: float | None = None
    if isinstance(coordinates, Mapping):
        values = coordinates.get("coordinates")
        if (
            coordinates.get("type") == "Point"
            and isinstance(values, list)
            and len(values) >= 2
        ):
            longitude, latitude = float(values[0]), float(values[1])
            if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
                raise RiseSchemaError("RISE location coordinates are out of range.")
    relationships = _mapping(resource.get("relationships"), "location relationships")
    raw_records = _mapping(
        relationships.get("catalogRecords"), "relationships.catalogRecords"
    ).get("data")
    raw_states = _mapping(relationships.get("states"), "relationships.states").get(
        "data"
    )
    if not isinstance(raw_records, list) or not isinstance(raw_states, list):
        raise RiseSchemaError("RISE location states/catalogRecords must be arrays.")
    record_ids = tuple(
        _resource_id(_mapping(item, "catalog record relationship"), "catalog-record")
        for item in raw_records
    )
    if expected_record_id is not None and expected_record_id not in record_ids:
        raise RiseIdentityError(
            f"RISE location {location_id} is not related to configured record "
            f"{expected_record_id}."
        )
    states = tuple(
        str(_mapping(item, "state relationship").get("id", "")).rsplit("/", 1)[-1]
        for item in raw_states
    )
    horizontal = attributes.get("horizontalDatum")
    vertical = attributes.get("verticalDatum")
    return RiseLocation(
        location_id=location_id,
        name=name.strip(),
        location_type=_optional_string_value(attributes.get("locationTypeName")),
        longitude=longitude,
        latitude=latitude,
        horizontal_datum=(
            _optional_string_value(horizontal.get("_id"))
            if isinstance(horizontal, Mapping)
            else None
        ),
        vertical_datum=(
            _optional_string_value(vertical.get("_id"))
            if isinstance(vertical, Mapping)
            else None
        ),
        states=states,
        record_ids=record_ids,
        source_url_public=f"{RISE_BASE_URL}/location/{location_id}",
    )


def _optional_string_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip() or None


def parse_results_page(
    payload: Mapping[str, Any],
    *,
    expected_item: RiseCatalogItem,
) -> tuple[tuple[RiseObservation, ...], int, int, int]:
    """Parse one result page and reject rows from any other item."""

    data = payload.get("data")
    if not isinstance(data, list):
        raise RiseSchemaError("RISE result response data must be an array.")
    meta = _mapping(payload.get("meta"), "RISE result meta")
    total = _integer(meta.get("totalItems"), "meta.totalItems")
    page_size = _integer(meta.get("itemsPerPage"), "meta.itemsPerPage")
    current_page = _integer(meta.get("currentPage"), "meta.currentPage")
    if total < 0 or page_size <= 0 or current_page <= 0:
        raise RiseSchemaError("RISE pagination metadata contains invalid values.")

    observations: list[RiseObservation] = []
    for index, raw in enumerate(data):
        resource = _mapping(raw, f"result data[{index}]")
        if resource.get("type") != "Result":
            raise RiseSchemaError(f"result data[{index}] has a non-Result type.")
        result_id = _resource_id(resource, "result")
        attributes = _mapping(resource.get("attributes"), "result attributes")
        item_id_candidates: list[int] = []
        for field_name in ("itemId", "catalogItemId"):
            if attributes.get(field_name) is not None:
                item_id_candidates.append(
                    _integer(attributes[field_name], f"result {field_name}")
                )
        relationships = resource.get("relationships")
        if isinstance(relationships, Mapping) and relationships.get("catalogItem"):
            item_id_candidates.append(
                _relationship_id(relationships, "catalogItem", "catalog-item")
            )
        if not item_id_candidates:
            raise RiseSchemaError(
                f"RISE result {result_id} has no catalog-item identity."
            )
        if len(set(item_id_candidates)) != 1:
            raise RiseIdentityError(
                f"RISE result {result_id} has conflicting item identities: "
                f"{item_id_candidates}."
            )
        item_id = item_id_candidates[0]
        if item_id != expected_item.item_id:
            raise RiseIdentityError(
                f"RISE result {result_id} belongs to item {item_id}, not requested "
                f"item {expected_item.item_id}."
            )
        parameter_id = (
            _integer(attributes["parameterId"], "result parameterId")
            if attributes.get("parameterId") is not None
            else None
        )
        if (
            expected_item.parameter_id is not None
            and parameter_id != expected_item.parameter_id
        ):
            raise RiseIdentityError(
                f"RISE result {result_id} parameter {parameter_id} does not match "
                f"item parameter {expected_item.parameter_id}."
            )
        value = attributes.get("result")
        if value is None:
            numeric_value = None
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RiseSchemaError(f"RISE result {result_id} value is not numeric.")
        else:
            numeric_value = float(value)
        if numeric_value is not None and not math.isfinite(numeric_value):
            raise RiseSchemaError(f"RISE result {result_id} value is not finite.")
        timestamp = _optional_datetime(attributes.get("dateTime"), "result dateTime")
        if timestamp is None:
            raise RiseSchemaError(f"RISE result {result_id} is missing dateTime.")
        observations.append(
            RiseObservation(
                result_id=result_id,
                item_id=item_id,
                location_id=(
                    _integer(attributes["locationId"], "result locationId")
                    if attributes.get("locationId") is not None
                    else None
                ),
                parameter_id=parameter_id,
                timestamp_utc=timestamp,
                value=numeric_value,
                status=(
                    str(attributes["status"]).strip()
                    if attributes.get("status") is not None
                    else None
                ),
                source_code=(
                    str(attributes["sourceCode"]).strip()
                    if attributes.get("sourceCode")
                    else None
                ),
                created_at=_optional_datetime(
                    attributes.get("createDate"), "result createDate"
                ),
                updated_at=_optional_datetime(
                    attributes.get("updateDate"), "result updateDate"
                ),
            )
        )
    return tuple(observations), total, page_size, current_page


def _unwrap_json(response: Any) -> Mapping[str, Any]:
    """Support the shared client's response wrapper as well as simple mocks."""

    if isinstance(response, Mapping):
        return response
    json_method = getattr(response, "json", None)
    if callable(json_method):
        value = json_method()
        if isinstance(value, Mapping):
            return value
    for attribute in ("json_data", "payload", "data"):
        value = getattr(response, attribute, None)
        if isinstance(value, Mapping):
            return value
    raise RiseSchemaError("HTTP client did not return a JSON object.")


class RiseAdapter(BaseAdapter):
    """Bounded RISE catalog and time-series client."""

    provider = Provider.USBR_RISE.value

    def _get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        schema: str,
    ) -> Mapping[str, Any]:
        response = self._get(
            url,
            params=dict(params or {}),
            headers=RISE_HEADERS,
            response_schema=schema,
        )
        return _unwrap_json(response)

    def fetch_record(
        self,
        record_id: int,
        *,
        expected_title_contains: str | None = None,
    ) -> RiseCatalogRecord:
        payload = self._get_json(
            f"{RISE_BASE_URL}/catalog-record/{int(record_id)}",
            schema="rise-jsonapi-catalog-record-v1",
        )
        return parse_catalog_record(
            payload,
            expected_record_id=int(record_id),
            expected_title_contains=expected_title_contains,
        )

    def fetch_item(
        self,
        item_id: int,
        *,
        expected_record_id: int,
        expected_location_name: str | None = None,
        expected_parameter_name: str | None = None,
        expected_unit: str | None = None,
    ) -> RiseCatalogItem:
        payload = self._get_json(
            f"{RISE_BASE_URL}/catalog-item/{int(item_id)}",
            schema="rise-jsonapi-catalog-item-v1",
        )
        return parse_catalog_item(
            payload,
            expected_item_id=int(item_id),
            expected_record_id=int(expected_record_id),
            expected_location_name=expected_location_name,
            expected_parameter_name=expected_parameter_name,
            expected_unit=expected_unit,
        )

    def fetch_location(
        self,
        location_id: int,
        *,
        expected_name_contains: str | None = None,
        expected_record_id: int | None = None,
    ) -> RiseLocation:
        """Fetch and validate one RISE location resource."""

        payload = self._get_json(
            f"{RISE_BASE_URL}/location/{int(location_id)}",
            schema="rise-jsonapi-location-v1",
        )
        return parse_location(
            payload,
            expected_location_id=int(location_id),
            expected_name_contains=expected_name_contains,
            expected_record_id=expected_record_id,
        )

    def validate_item_source(
        self,
        *,
        record_id: int,
        item_id: int,
        expected_record_title_contains: str,
        expected_location_name_contains: str,
        expected_parameter_name: str,
        expected_unit: str,
    ) -> RiseValidatedSource:
        """Validate the complete record-location-item identity chain."""

        record = self.fetch_record(
            record_id,
            expected_title_contains=expected_record_title_contains,
        )
        if item_id not in record.item_ids:
            raise RiseIdentityError(
                f"RISE item {item_id} is not listed by record {record_id}."
            )
        location = self.fetch_location(
            record.location_id,
            expected_name_contains=expected_location_name_contains,
            expected_record_id=record_id,
        )
        item = self.fetch_item(
            item_id,
            expected_record_id=record_id,
            expected_location_name=expected_location_name_contains,
            expected_parameter_name=expected_parameter_name,
            expected_unit=expected_unit,
        )
        return RiseValidatedSource(record=record, location=location, item=item)

    def inventory_record_items(
        self,
        record: RiseCatalogRecord,
        *,
        expected_location_name: str | None = None,
        target_variables: Sequence[HydrometryVariable] = (
            HydrometryVariable.RESERVOIR_ELEVATION,
            HydrometryVariable.SURFACE_AREA,
            HydrometryVariable.STORAGE,
        ),
        max_items: int = 100,
    ) -> RiseItemInventory:
        """Fetch each relationship item exactly once without selecting one."""

        if len(record.item_ids) > max_items:
            raise RiseError(
                f"RISE record {record.record_id} exposes {len(record.item_ids)} "
                f"items, exceeding bounded maximum {max_items}."
            )
        items = tuple(
            self.fetch_item(
                item_id,
                expected_record_id=record.record_id,
                expected_location_name=expected_location_name,
            )
            for item_id in record.item_ids
        )
        by_variable: dict[HydrometryVariable, tuple[int, ...]] = {}
        for variable in HydrometryVariable:
            candidates = tuple(
                item.item_id
                for item in items
                if item.variable is variable and not item.validation_issues
            )
            if candidates:
                by_variable[variable] = candidates
        return RiseItemInventory(
            record=record,
            items=items,
            candidates_by_variable=MappingProxyType(by_variable),
            requires_review=any(
                len(by_variable.get(variable, ())) != 1 for variable in target_variables
            ),
        )

    def fetch_timeseries(
        self,
        item: RiseCatalogItem,
        *,
        start: date,
        end: date,
        page_size: int = 1000,
        max_pages: int = 100,
    ) -> tuple[RiseObservation, ...]:
        """Fetch a bounded interval from one validated time-series item."""

        if start > end:
            raise ValueError("start must be on or before end.")
        if item.data_structure and item.data_structure.casefold() != "time series":
            raise RiseSchemaError(f"RISE item {item.item_id} is not a time series.")
        if item.variable is None:
            raise RiseSchemaError(
                f"RISE item {item.item_id} has no validated hydrometry variable."
            )
        if page_size <= 0 or page_size > 10_000 or max_pages <= 0:
            raise ValueError("page_size/max_pages are outside bounded limits.")

        rows: list[RiseObservation] = []
        total_expected: int | None = None
        for page in range(1, max_pages + 1):
            payload = self._get_json(
                f"{RISE_BASE_URL}/result",
                params={
                    "itemId": item.item_id,
                    "dateTime[after]": start.isoformat(),
                    "dateTime[before]": end.isoformat(),
                    "itemsPerPage": page_size,
                    "page": page,
                },
                schema="rise-jsonapi-result-v1",
            )
            observations, total, actual_page_size, current = parse_results_page(
                payload, expected_item=item
            )
            outside = [
                observation.timestamp_utc
                for observation in observations
                if not start <= observation.timestamp_utc.date() <= end
            ]
            if outside:
                raise RiseSchemaError(
                    f"RISE item {item.item_id} returned {len(outside)} timestamp(s) "
                    "outside the requested bounded interval."
                )
            if current != page:
                raise RiseSchemaError(
                    f"RISE returned page {current} while page {page} was requested."
                )
            if total_expected is None:
                total_expected = total
                if total > page_size * max_pages:
                    raise RiseError(
                        f"RISE query would return {total} rows, exceeding bounded "
                        f"maximum {page_size * max_pages}."
                    )
            elif total != total_expected:
                raise RiseSchemaError("RISE totalItems changed during pagination.")
            rows.extend(observations)
            if current * actual_page_size >= total:
                if len(rows) != total:
                    raise RiseSchemaError(
                        f"RISE pagination returned {len(rows)} of {total} rows."
                    )
                timestamps = [row.timestamp_utc for row in rows]
                if len(timestamps) != len(set(timestamps)):
                    raise RiseSchemaError(
                        f"RISE item {item.item_id} returned duplicate timestamps."
                    )
                return tuple(sorted(rows, key=lambda row: row.timestamp_utc))
        raise RiseError(f"RISE results exceeded configured maximum {max_pages} pages.")

    @staticmethod
    def normalize_observations(
        observations: Sequence[RiseObservation],
        *,
        item: RiseCatalogItem,
        profile_id: str,
        hylak_id: int,
        display_name: str,
        source_id: str,
        vertical_reference: VerticalReference,
        retrieved_at_utc: datetime,
        source_version: str | None = None,
    ) -> tuple[NormalizedTimeSeriesRecord, ...]:
        """Convert validated provider observations to shared records."""

        if item.variable is None or item.unit is None:
            raise RiseSchemaError(
                f"RISE item {item.item_id} is not ready for normalization."
            )
        if item.validation_issues:
            raise RiseSchemaError(
                f"RISE item {item.item_id} requires review: "
                + ", ".join(item.validation_issues)
            )
        if vertical_reference.height_type is not HeightType.UNKNOWN and (
            item.variable
            not in {
                HydrometryVariable.RESERVOIR_ELEVATION,
                HydrometryVariable.WATER_LEVEL,
                HydrometryVariable.GAGE_HEIGHT,
            }
        ):
            # Non-height variables carry UNKNOWN height semantics in normalized data.
            raise RiseSchemaError(
                "Area/storage observations must use an unknown-height vertical "
                "reference; datum metadata belongs to the source item instead."
            )
        return tuple(
            NormalizedTimeSeriesRecord(
                profile_id=profile_id,
                Hylak_id=hylak_id,
                display_name=display_name,
                provider=Provider.USBR_RISE,
                source_id=source_id,
                source_record_id=str(item.record_id),
                source_item_id=str(item.item_id),
                variable=item.variable,
                value=observation.value,
                unit=item.unit,
                timestamp_utc=observation.timestamp_utc,
                quality_status=observation.status or "provider_status_not_reported",
                provisional=item.provisional,
                approval_status=None,
                height_type=vertical_reference.height_type,
                vertical_datum=vertical_reference,
                retrieved_at_utc=retrieved_at_utc,
                source_version=source_version,
                source_url_public=item.source_url_public,
                raw_parameter_code=(
                    str(item.parameter_id) if item.parameter_id is not None else None
                ),
                raw_statistic_code=item.transformation,
                raw_metadata={
                    "result_id": observation.result_id,
                    "location_id": observation.location_id,
                    "source_code": observation.source_code,
                    "parameter_name": item.parameter_name,
                    "parameter_group": item.parameter_group,
                    "timestep": item.timestep,
                    "update_frequency": item.update_frequency,
                    "item_vertical_datum": item.vertical_datum,
                    "disclaimer": item.disclaimer,
                },
            )
            for observation in observations
            if observation.value is not None
        )
