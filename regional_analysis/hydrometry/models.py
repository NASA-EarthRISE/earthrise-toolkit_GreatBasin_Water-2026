"""Normalized, provider-independent models for Phase 0 hydrometry work.

The models in this module deliberately contain no networking or provider
selection logic.  Provider adapters may parse their native payloads, but data
leaving an adapter should use these explicit types.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any


class HydrometryModelError(ValueError):
    """Raised when normalized hydrometry data violate a model contract."""


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class CapabilityLevel(IntEnum):
    """Ordered lake-product capability levels."""

    SURFACE_EXTENT = 1
    WATER_LEVEL = 2
    RELATIVE_STORAGE_CHANGE = 3
    ABSOLUTE_STORAGE = 4
    QUANTITATIVE_3D = 5


class SourceQualityClass(_StringEnum):
    """Evidence class for a storage or 3D capability pathway."""

    DIRECT_AUTHORITATIVE_STORAGE = "A"
    STAGE_OFFICIAL_CAPACITY = "B"
    STAGE_SURVEYED_BATHYMETRY = "C"
    EMPIRICAL_AREA_STAGE = "D"
    SATELLITE_RELATIVE_STORAGE = "E"
    MODELED_COARSE_FALLBACK = "F"


class HeightType(_StringEnum):
    """Meaning of a source height value, independent of its units."""

    LOCAL_GAGE_HEIGHT = "local_gage_height"
    PROJECT_DATUM_ELEVATION = "project_datum_elevation"
    ORTHOMETRIC_HEIGHT = "orthometric_height"
    ELLIPSOIDAL_HEIGHT = "ellipsoidal_height"
    GEOID_REFERENCED_HEIGHT = "geoid_referenced_height"
    UNKNOWN = "unknown"


class NormalizationStatus(_StringEnum):
    """Status of vertical-reference normalization.

    ``NORMALIZED`` is included so a later phase can represent a completed,
    documented transformation.  Phase 0 does not perform such transformations.
    """

    SOURCE_VALUE_RETAINED = "source_value_retained"
    NOT_ATTEMPTED = "normalization_not_attempted"
    BLOCKED = "normalization_blocked"
    POSSIBLE_NOT_IMPLEMENTED = "normalization_possible_but_not_implemented"
    NORMALIZED = "normalized"


class Provider(_StringEnum):
    """Providers represented in the pilot registry."""

    USGS_WATERDATA = "usgs_waterdata"
    USBR_RISE = "usbr_rise"
    HYDROCRON = "hydrocron"
    USGS_SCIENCEBASE = "usgs_sciencebase"


class HydrometryVariable(_StringEnum):
    """Normalized variables used by source specifications and time series."""

    WATER_LEVEL = "water_level"
    RESERVOIR_ELEVATION = "reservoir_elevation"
    GAGE_HEIGHT = "gage_height"
    SURFACE_AREA = "surface_area"
    STORAGE = "storage"
    BANK_STORAGE = "bank_storage"
    RELATIVE_STORAGE_CHANGE = "relative_storage_change"
    SOURCE_INVENTORY = "source_inventory"
    BATHYMETRY = "bathymetry"
    CAPACITY_RELATION = "capacity_relation"


_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def ensure_no_sensitive_keys(value: Any, *, context: str = "metadata") -> None:
    """Reject credential-like keys before data can be serialized or cached."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if _is_sensitive_key(key):
                raise HydrometryModelError(
                    f"{context} contains prohibited credential field {key!r}."
                )
            ensure_no_sensitive_keys(item, context=f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ensure_no_sensitive_keys(item, context=f"{context}[{index}]")


def _freeze(value: Any) -> Any:
    """Return a recursively immutable defensive copy."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _serialize(value: Any) -> Any:
    """Convert normalized models to detached JSON-compatible values."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if is_dataclass(value):
        return {
            item.name: _serialize(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset, set)):
        return [_serialize(item) for item in value]
    return value


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HydrometryModelError(f"{label} must be a non-empty string.")
    return value.strip()


def _optional_string(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, label)


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise HydrometryModelError(f"{label} must be a timezone-aware datetime.")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class VerticalReference:
    """Source and target vertical-reference metadata without transformation."""

    height_type: HeightType
    source_vertical_datum: str | None
    source_vertical_units: str | None
    datum_epoch: str | None = None
    gage_datum_offset: float | None = None
    gage_datum_offset_source: str | None = None
    target_vertical_datum: str | None = None
    transformation_method: str | None = None
    transformation_uncertainty: float | None = None
    normalization_status: NormalizationStatus = NormalizationStatus.NOT_ATTEMPTED
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_vertical_datum",
            _optional_string(self.source_vertical_datum, "source_vertical_datum"),
        )
        object.__setattr__(
            self,
            "source_vertical_units",
            _optional_string(self.source_vertical_units, "source_vertical_units"),
        )
        for name in (
            "datum_epoch",
            "gage_datum_offset_source",
            "target_vertical_datum",
            "transformation_method",
        ):
            object.__setattr__(self, name, _optional_string(getattr(self, name), name))

        if self.gage_datum_offset is not None and not math.isfinite(
            self.gage_datum_offset
        ):
            raise HydrometryModelError("gage_datum_offset must be finite.")
        if self.gage_datum_offset is not None and not self.gage_datum_offset_source:
            raise HydrometryModelError(
                "A gage_datum_offset requires gage_datum_offset_source."
            )
        if self.transformation_uncertainty is not None and (
            not math.isfinite(self.transformation_uncertainty)
            or self.transformation_uncertainty < 0
        ):
            raise HydrometryModelError(
                "transformation_uncertainty must be finite and non-negative."
            )
        if self.normalization_status is NormalizationStatus.NORMALIZED:
            if not self.source_vertical_datum or not self.target_vertical_datum:
                raise HydrometryModelError(
                    "Normalized heights require source and target vertical datums."
                )
            if self.source_vertical_datum != self.target_vertical_datum and not (
                self.transformation_method
            ):
                raise HydrometryModelError(
                    "A cross-datum normalized height requires a transformation_method."
                )
            if self.height_type is HeightType.LOCAL_GAGE_HEIGHT and (
                self.gage_datum_offset is None or not self.gage_datum_offset_source
            ):
                raise HydrometryModelError(
                    "A normalized local gage height requires a published gage "
                    "datum offset and source."
                )

    def as_public_dict(self) -> dict[str, Any]:
        """Return a detached, serializable representation."""

        return _serialize(self)


def vertical_reference_supports_absolute_storage(
    reference: VerticalReference,
) -> bool:
    """Return whether a stage value is ready for an absolute-storage relation.

    Matching units alone are intentionally insufficient.  A local gage height
    also needs a published offset even when source and target datum labels match.
    """

    if reference.height_type is HeightType.UNKNOWN:
        return False
    if not reference.source_vertical_datum or not reference.target_vertical_datum:
        return False
    if reference.height_type is HeightType.LOCAL_GAGE_HEIGHT and (
        reference.gage_datum_offset is None or not reference.gage_datum_offset_source
    ):
        return False
    if reference.source_vertical_datum == reference.target_vertical_datum:
        return reference.normalization_status in {
            NormalizationStatus.SOURCE_VALUE_RETAINED,
            NormalizationStatus.NORMALIZED,
        }
    return reference.normalization_status is NormalizationStatus.NORMALIZED


@dataclass(frozen=True, slots=True)
class NormalizedTimeSeriesRecord:
    """One provider-neutral hydrometry observation."""

    profile_id: str
    Hylak_id: int
    display_name: str
    provider: Provider
    source_id: str
    source_record_id: str | None
    source_item_id: str | None
    variable: HydrometryVariable
    value: float
    unit: str
    timestamp_utc: datetime
    quality_status: str
    provisional: bool | None
    approval_status: str | None
    height_type: HeightType
    vertical_datum: VerticalReference
    retrieved_at_utc: datetime
    source_version: str | None
    source_url_public: str
    raw_parameter_code: str | None
    raw_statistic_code: str | None
    raw_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "display_name",
            "source_id",
            "unit",
            "quality_status",
            "source_url_public",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "source_record_id",
            "source_item_id",
            "approval_status",
            "source_version",
            "raw_parameter_code",
            "raw_statistic_code",
        ):
            object.__setattr__(self, name, _optional_string(getattr(self, name), name))
        if isinstance(self.Hylak_id, bool) or not isinstance(self.Hylak_id, int):
            raise HydrometryModelError("Hylak_id must be an integer.")
        if not math.isfinite(self.value):
            raise HydrometryModelError("value must be finite.")
        if not isinstance(self.provisional, (bool, type(None))):
            raise HydrometryModelError("provisional must be true, false, or null.")
        if self.height_type is not self.vertical_datum.height_type:
            raise HydrometryModelError(
                "height_type must match vertical_datum.height_type."
            )
        object.__setattr__(
            self, "timestamp_utc", _utc(self.timestamp_utc, "timestamp_utc")
        )
        object.__setattr__(
            self,
            "retrieved_at_utc",
            _utc(self.retrieved_at_utc, "retrieved_at_utc"),
        )
        ensure_no_sensitive_keys(self.raw_metadata, context="raw_metadata")
        object.__setattr__(self, "raw_metadata", _freeze(self.raw_metadata))

    @property
    def hylak_id(self) -> int:
        """Return the persistent HydroLAKES ID with Python-style casing."""

        return self.Hylak_id

    def as_public_dict(self) -> dict[str, Any]:
        """Return a detached record that contains no authentication material."""

        result = _serialize(self)
        ensure_no_sensitive_keys(result, context="normalized_record")
        return result


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    """Conservative Phase 0 capability assessment for one pilot lake."""

    profile_id: str
    Hylak_id: int
    display_name: str
    state: str
    extent_available: bool
    water_level_available: bool
    relative_storage_available: bool
    absolute_storage_available: bool
    quantitative_3d_available: bool
    highest_current_capability: CapabilityLevel
    highest_potential_capability: CapabilityLevel
    source_quality_class: SourceQualityClass | None
    confirmed_sources: tuple[str, ...]
    likely_sources: tuple[str, ...]
    unverified_sources: tuple[str, ...]
    missing_requirements: tuple[str, ...]
    datum_status: str
    key_warning: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("profile_id", "display_name", "state", "datum_status"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.highest_potential_capability < self.highest_current_capability:
            raise HydrometryModelError(
                "highest_potential_capability cannot be below current capability."
            )
        if self.absolute_storage_available and (
            self.highest_current_capability < CapabilityLevel.ABSOLUTE_STORAGE
        ):
            raise HydrometryModelError(
                "absolute_storage_available requires capability level 4 or higher."
            )
        if self.quantitative_3d_available and (
            self.highest_current_capability < CapabilityLevel.QUANTITATIVE_3D
        ):
            raise HydrometryModelError(
                "quantitative_3d_available requires capability level 5."
            )
        for name in (
            "confirmed_sources",
            "likely_sources",
            "unverified_sources",
            "missing_requirements",
        ):
            object.__setattr__(
                self, name, tuple(str(item) for item in getattr(self, name))
            )
        ensure_no_sensitive_keys(self.provenance, context="capability.provenance")
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    @property
    def hylak_id(self) -> int:
        return self.Hylak_id

    def as_public_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible capability record."""

        result = _serialize(self)
        ensure_no_sensitive_keys(result, context="capability_record")
        return result
