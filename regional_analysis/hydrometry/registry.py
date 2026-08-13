"""Validated research registry for the five Phase 0 hydrometry pilots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from .models import (
    CapabilityLevel,
    HeightType,
    HydrometryModelError,
    HydrometryVariable,
    Provider,
    SourceQualityClass,
    ensure_no_sensitive_keys,
)


class HydrometryRegistryError(ValueError):
    """Raised when the Phase 0 pilot registry is missing or malformed."""


EXPECTED_PILOT_HYLAK_IDS = frozenset({67, 776, 778, 784, 802})
HYDROCRON_REQUIRES_CROSSWALK = "requires_crosswalk"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HydrometryRegistryError(f"{label} must be a YAML mapping.")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise HydrometryRegistryError(f"{label} must be a YAML list.")
    return value


def _string(value: Any, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        suffix = " or null" if optional else ""
        raise HydrometryRegistryError(f"{label} must be a non-empty string{suffix}.")
    return value.strip()


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HydrometryRegistryError(f"{label} must be an integer.")
    if positive and value <= 0:
        raise HydrometryRegistryError(f"{label} must be greater than zero.")
    return value


def _enum(enum_type: type, value: Any, label: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(str(member.value) for member in enum_type)
        raise HydrometryRegistryError(f"{label} must be one of: {allowed}.") from exc


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType(
                {str(key): freeze(nested) for key, nested in item.items()}
            )
        if isinstance(item, list | tuple):
            return tuple(freeze(nested) for nested in item)
        return item

    return freeze(value)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One exact source to probe; priority is ordering, never fallback."""

    source_id: str
    provider: Provider
    priority: int
    source_record_id: str
    source_item_id: str | None
    variable: HydrometryVariable
    expected_unit: str | None
    expected_vertical_datum: str | None
    expected_height_type: HeightType
    role: str
    required: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.priority <= 0:
            raise HydrometryRegistryError("Source priority must be greater than zero.")
        ensure_no_sensitive_keys(self.metadata, context=f"source {self.source_id}")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "provider": self.provider.value,
            "priority": self.priority,
            "source_record_id": self.source_record_id,
            "source_item_id": self.source_item_id,
            "variable": self.variable.value,
            "expected_unit": self.expected_unit,
            "expected_vertical_datum": self.expected_vertical_datum,
            "expected_height_type": self.expected_height_type.value,
            "role": self.role,
            "required": self.required,
            "metadata": _detached(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class BathymetrySourceSpec:
    """Metadata-only official bathymetry or capacity source."""

    catalog_id: str
    provider: Provider
    priority: int
    doi: str | None
    source_record_id: str | None
    product_type: str
    source_quality_candidate: SourceQualityClass
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.priority <= 0:
            raise HydrometryRegistryError(
                "Bathymetry source priority must be greater than zero."
            )
        if not self.doi and not self.source_record_id:
            raise HydrometryRegistryError(
                "A bathymetry source requires a DOI or source_record_id."
            )
        ensure_no_sensitive_keys(
            self.metadata, context=f"bathymetry source {self.catalog_id}"
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "provider": self.provider.value,
            "priority": self.priority,
            "doi": self.doi,
            "source_record_id": self.source_record_id,
            "product_type": self.product_type,
            "source_quality_candidate": self.source_quality_candidate.value,
            "metadata": _detached(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SwotCapabilitySpec:
    """Explicit SWOT/Hydrocron crosswalk state for one HydroLAKES feature."""

    swot_pld_id: str | None
    hydrocron_status: str

    def __post_init__(self) -> None:
        if self.swot_pld_id is None and (
            self.hydrocron_status != HYDROCRON_REQUIRES_CROSSWALK
        ):
            raise HydrometryRegistryError(
                "A null SWOT PLD ID must have hydrocron_status=requires_crosswalk."
            )


@dataclass(frozen=True, slots=True)
class HydrometryProfile:
    """Immutable research profile for one Phase 0 pilot lake."""

    profile_id: str
    Hylak_id: int
    display_name: str
    state: str
    role: str
    legacy_preset_name: str | None
    potential_capability_ceiling: CapabilityLevel
    potential_source_quality_class: SourceQualityClass | None
    key_constraints: tuple[str, ...]
    swot: SwotCapabilitySpec
    sources: tuple[SourceSpec, ...]
    bathymetry_sources: tuple[BathymetrySourceSpec, ...]

    def __post_init__(self) -> None:
        source_ids = [source.source_id for source in self.sources]
        duplicates = sorted(
            source_id
            for source_id in set(source_ids)
            if source_ids.count(source_id) > 1
        )
        if duplicates:
            raise HydrometryRegistryError(
                f"Profile {self.profile_id!r} has duplicate source IDs: "
                + ", ".join(duplicates)
                + "."
            )
        catalog_ids = [source.catalog_id for source in self.bathymetry_sources]
        duplicates = sorted(
            catalog_id
            for catalog_id in set(catalog_ids)
            if catalog_ids.count(catalog_id) > 1
        )
        if duplicates:
            raise HydrometryRegistryError(
                f"Profile {self.profile_id!r} has duplicate bathymetry catalog IDs: "
                + ", ".join(duplicates)
                + "."
            )

    @property
    def hylak_id(self) -> int:
        return self.Hylak_id

    def as_public_dict(self) -> dict[str, Any]:
        """Return a detached profile without authentication material."""

        result = {
            "profile_id": self.profile_id,
            "Hylak_id": self.Hylak_id,
            "display_name": self.display_name,
            "state": self.state,
            "role": self.role,
            "legacy_preset_name": self.legacy_preset_name,
            "potential_capability_ceiling": int(self.potential_capability_ceiling),
            "potential_source_quality_class": (
                self.potential_source_quality_class.value
                if self.potential_source_quality_class
                else None
            ),
            "key_constraints": list(self.key_constraints),
            "swot": {
                "swot_pld_id": self.swot.swot_pld_id,
                "hydrocron_status": self.swot.hydrocron_status,
            },
            "sources": [source.as_public_dict() for source in self.sources],
            "bathymetry_sources": [
                source.as_public_dict() for source in self.bathymetry_sources
            ],
        }
        ensure_no_sensitive_keys(result, context=f"profile {self.profile_id}")
        return result


def _detached(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _detached(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_detached(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class HydrometryRegistry:
    """Five-pilot registry with exact, non-fallback lookup methods."""

    schema_version: str
    profiles: tuple[HydrometryProfile, ...]
    source_path: Path | None = None
    _by_profile_id: Mapping[str, HydrometryProfile] = field(
        init=False, repr=False, compare=False
    )
    _by_hylak_id: Mapping[int, HydrometryProfile] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if len(self.profiles) != 5:
            raise HydrometryRegistryError(
                f"Phase 0 requires exactly five pilot profiles; found {len(self.profiles)}."
            )
        profile_ids = [profile.profile_id for profile in self.profiles]
        hylak_ids = [profile.Hylak_id for profile in self.profiles]
        if len(set(profile_ids)) != len(profile_ids):
            raise HydrometryRegistryError("Hydrometry profile IDs must be unique.")
        if len(set(hylak_ids)) != len(hylak_ids):
            raise HydrometryRegistryError("Hydrometry Hylak IDs must be unique.")
        if set(hylak_ids) != EXPECTED_PILOT_HYLAK_IDS:
            raise HydrometryRegistryError(
                "Phase 0 pilot Hylak IDs must be exactly: "
                + ", ".join(str(value) for value in sorted(EXPECTED_PILOT_HYLAK_IDS))
                + "."
            )
        object.__setattr__(
            self,
            "_by_profile_id",
            MappingProxyType(
                {profile.profile_id: profile for profile in self.profiles}
            ),
        )
        object.__setattr__(
            self,
            "_by_hylak_id",
            MappingProxyType({profile.Hylak_id: profile for profile in self.profiles}),
        )

    def get_profile(
        self, profile_id: str, *, strict: bool = False
    ) -> HydrometryProfile | None:
        """Return only an exact profile match; no fallback is attempted."""

        profile = self._by_profile_id.get(profile_id)
        if profile is None and strict:
            raise KeyError(profile_id)
        return profile

    def get_profile_for_hylak_id(
        self, hylak_id: int, *, strict: bool = False
    ) -> HydrometryProfile | None:
        """Return an exact HydroLAKES identity match."""

        profile = self._by_hylak_id.get(hylak_id)
        if profile is None and strict:
            raise KeyError(hylak_id)
        return profile

    def get_source(
        self, profile_id: str, source_id: str, *, strict: bool = False
    ) -> SourceSpec | None:
        """Look up an exact source ID without substituting another priority."""

        profile = self.get_profile(profile_id, strict=strict)
        if profile is None:
            return None
        source = next(
            (item for item in profile.sources if item.source_id == source_id), None
        )
        if source is None and strict:
            raise KeyError((profile_id, source_id))
        return source

    def ordered_sources(self, profile_id: str) -> tuple[SourceSpec, ...]:
        """Return all configured sources in deterministic probe order.

        Ordering does not authorize an adapter to replace a failed source with
        the next one; each returned source must be probed and reported by ID.
        """

        profile = self.get_profile(profile_id, strict=True)
        assert profile is not None
        return tuple(
            sorted(profile.sources, key=lambda item: (item.priority, item.source_id))
        )

    def as_public_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "profiles": [profile.as_public_dict() for profile in self.profiles],
        }
        ensure_no_sensitive_keys(result, context="hydrometry registry")
        return result


def _parse_source(raw: Any, label: str) -> SourceSpec:
    item = _mapping(raw, label)
    return SourceSpec(
        source_id=_string(item.get("source_id"), f"{label}.source_id"),
        provider=_enum(Provider, item.get("provider"), f"{label}.provider"),
        priority=_integer(item.get("priority"), f"{label}.priority", positive=True),
        source_record_id=_string(
            item.get("source_record_id"), f"{label}.source_record_id"
        ),
        source_item_id=_string(
            item.get("source_item_id"), f"{label}.source_item_id", optional=True
        ),
        variable=_enum(HydrometryVariable, item.get("variable"), f"{label}.variable"),
        expected_unit=_string(
            item.get("expected_unit"), f"{label}.expected_unit", optional=True
        ),
        expected_vertical_datum=_string(
            item.get("expected_vertical_datum"),
            f"{label}.expected_vertical_datum",
            optional=True,
        ),
        expected_height_type=_enum(
            HeightType,
            item.get("expected_height_type"),
            f"{label}.expected_height_type",
        ),
        role=_string(item.get("role"), f"{label}.role"),
        required=item.get("required") is True,
        metadata=_mapping(item.get("metadata", {}), f"{label}.metadata"),
    )


def _parse_bathymetry_source(raw: Any, label: str) -> BathymetrySourceSpec:
    item = _mapping(raw, label)
    return BathymetrySourceSpec(
        catalog_id=_string(item.get("catalog_id"), f"{label}.catalog_id"),
        provider=_enum(Provider, item.get("provider"), f"{label}.provider"),
        priority=_integer(item.get("priority"), f"{label}.priority", positive=True),
        doi=_string(item.get("doi"), f"{label}.doi", optional=True),
        source_record_id=_string(
            item.get("source_record_id"),
            f"{label}.source_record_id",
            optional=True,
        ),
        product_type=_string(item.get("product_type"), f"{label}.product_type"),
        source_quality_candidate=_enum(
            SourceQualityClass,
            item.get("source_quality_candidate"),
            f"{label}.source_quality_candidate",
        ),
        metadata=_mapping(item.get("metadata", {}), f"{label}.metadata"),
    )


def _parse_profile(raw: Any, index: int) -> HydrometryProfile:
    label = f"profiles[{index}]"
    item = _mapping(raw, label)
    swot_raw = _mapping(item.get("swot"), f"{label}.swot")
    raw_quality = item.get("potential_source_quality_class")
    quality = (
        None
        if raw_quality is None
        else _enum(
            SourceQualityClass,
            raw_quality,
            f"{label}.potential_source_quality_class",
        )
    )
    constraints = tuple(
        _string(value, f"{label}.key_constraints[{position}]")
        for position, value in enumerate(
            _list(item.get("key_constraints", []), f"{label}.key_constraints")
        )
    )
    sources = tuple(
        _parse_source(value, f"{label}.sources[{position}]")
        for position, value in enumerate(_list(item.get("sources"), f"{label}.sources"))
    )
    bathymetry_sources = tuple(
        _parse_bathymetry_source(value, f"{label}.bathymetry_sources[{position}]")
        for position, value in enumerate(
            _list(
                item.get("bathymetry_sources", []),
                f"{label}.bathymetry_sources",
            )
        )
    )
    return HydrometryProfile(
        profile_id=_string(item.get("profile_id"), f"{label}.profile_id"),
        Hylak_id=_integer(item.get("Hylak_id"), f"{label}.Hylak_id", positive=True),
        display_name=_string(item.get("display_name"), f"{label}.display_name"),
        state=_string(item.get("state"), f"{label}.state"),
        role=_string(item.get("role"), f"{label}.role"),
        legacy_preset_name=_string(
            item.get("legacy_preset_name"),
            f"{label}.legacy_preset_name",
            optional=True,
        ),
        potential_capability_ceiling=_enum(
            CapabilityLevel,
            item.get("potential_capability_ceiling"),
            f"{label}.potential_capability_ceiling",
        ),
        potential_source_quality_class=quality,
        key_constraints=constraints,
        swot=SwotCapabilitySpec(
            swot_pld_id=_string(
                swot_raw.get("swot_pld_id"),
                f"{label}.swot.swot_pld_id",
                optional=True,
            ),
            hydrocron_status=_string(
                swot_raw.get("hydrocron_status"),
                f"{label}.swot.hydrocron_status",
            ),
        ),
        sources=sources,
        bathymetry_sources=bathymetry_sources,
    )


def validate_registry_config(
    config: Any, *, source_path: Path | None = None
) -> HydrometryRegistry:
    """Validate parsed YAML and return an immutable five-pilot registry."""

    try:
        ensure_no_sensitive_keys(config, context="hydrometry configuration")
    except HydrometryModelError as exc:
        raise HydrometryRegistryError(str(exc)) from exc
    root = _mapping(config, "Hydrometry configuration root")
    schema_version = _string(root.get("schema_version"), "schema_version")
    raw_profiles = _list(root.get("profiles"), "profiles")
    profiles = tuple(
        _parse_profile(raw_profile, index)
        for index, raw_profile in enumerate(raw_profiles)
    )
    return HydrometryRegistry(
        schema_version=schema_version,
        profiles=profiles,
        source_path=source_path,
    )


def load_registry(path: str | Path) -> HydrometryRegistry:
    """Load a credential-free YAML registry without any network dependency."""

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HydrometryRegistryError(
            f"Could not read hydrometry configuration {config_path!s}: {exc}"
        ) from exc
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise HydrometryRegistryError(
            f"Invalid YAML in hydrometry configuration {config_path!s}: {exc}"
        ) from exc
    return validate_registry_config(parsed, source_path=config_path)
