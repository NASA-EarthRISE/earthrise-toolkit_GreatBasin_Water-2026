"""Conservative Phase 0 capability classification rules.

These rules classify verified evidence; they do not calculate volume and never
promote HydroLAKES modeled attributes into an absolute-storage capability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .models import (
    CapabilityLevel,
    CapabilityRecord,
    HydrometryModelError,
    HydrometryVariable,
    SourceQualityClass,
)
from .registry import HydrometryProfile


@dataclass(frozen=True, slots=True)
class CapabilityEvidence:
    """Provider-validation evidence supplied to the capability classifier."""

    extent_available: bool = True
    water_level_available: bool = False
    relative_storage_available: bool = False
    direct_authoritative_storage_available: bool = False
    direct_storage_variable: HydrometryVariable | None = None
    official_capacity_relation_available: bool = False
    surveyed_bathymetry_available: bool = False
    quantitative_3d_available: bool = False
    empirical_hypsometry_available: bool = False
    modeled_fallback_available: bool = False
    hydrolakes_modeled_volume_present: bool = False
    storage_computation_implemented: bool = False
    source_vertical_datum_known: bool = False
    vertical_relation_documented: bool = False
    multiple_arm_constraint_resolved: bool = True
    potential_water_level: bool = False
    potential_relative_storage: bool = False
    potential_direct_storage: bool = False
    potential_official_capacity_relation: bool = False
    potential_surveyed_bathymetry: bool = False
    potential_quantitative_3d: bool = False
    datum_status: str = "unknown"
    missing_requirements: tuple[str, ...] = ()
    key_warning: str = ""

    def __post_init__(self) -> None:
        if self.direct_authoritative_storage_available and (
            self.direct_storage_variable is not HydrometryVariable.STORAGE
        ):
            raise HydrometryModelError(
                "Direct authoritative storage requires a validated total-storage "
                "variable; bank storage and other variables cannot confer Level 4."
            )
        if not self.direct_authoritative_storage_available and (
            self.direct_storage_variable is not None
        ):
            raise HydrometryModelError(
                "direct_storage_variable requires direct storage availability."
            )


def _current_absolute_storage(evidence: CapabilityEvidence) -> bool:
    if evidence.direct_authoritative_storage_available:
        return True

    relation_available = (
        evidence.official_capacity_relation_available
        or evidence.surveyed_bathymetry_available
        or evidence.empirical_hypsometry_available
    )
    return bool(
        evidence.water_level_available
        and relation_available
        and evidence.storage_computation_implemented
        and evidence.source_vertical_datum_known
        and evidence.vertical_relation_documented
        and evidence.multiple_arm_constraint_resolved
    )


def _current_quality(
    evidence: CapabilityEvidence,
) -> SourceQualityClass | None:
    if evidence.direct_authoritative_storage_available:
        return SourceQualityClass.DIRECT_AUTHORITATIVE_STORAGE
    if evidence.storage_computation_implemented:
        if evidence.official_capacity_relation_available:
            return SourceQualityClass.STAGE_OFFICIAL_CAPACITY
        if evidence.surveyed_bathymetry_available:
            return SourceQualityClass.STAGE_SURVEYED_BATHYMETRY
        if evidence.empirical_hypsometry_available:
            return SourceQualityClass.EMPIRICAL_AREA_STAGE
    if evidence.relative_storage_available:
        return SourceQualityClass.SATELLITE_RELATIVE_STORAGE
    if evidence.modeled_fallback_available:
        return SourceQualityClass.MODELED_COARSE_FALLBACK
    return None


def _potential_level(
    profile: HydrometryProfile,
    evidence: CapabilityEvidence,
    current_level: CapabilityLevel,
) -> CapabilityLevel:
    potential = current_level
    water_level_path = evidence.water_level_available or evidence.potential_water_level
    if evidence.potential_water_level:
        potential = max(potential, CapabilityLevel.WATER_LEVEL)
    if evidence.potential_relative_storage:
        potential = max(potential, CapabilityLevel.RELATIVE_STORAGE_CHANGE)
    if evidence.potential_direct_storage or (
        water_level_path
        and (
            evidence.potential_official_capacity_relation
            or evidence.potential_surveyed_bathymetry
        )
    ):
        potential = max(potential, CapabilityLevel.ABSOLUTE_STORAGE)
    if evidence.potential_quantitative_3d:
        potential = max(potential, CapabilityLevel.QUANTITATIVE_3D)
    return min(potential, profile.potential_capability_ceiling)


def _auto_missing_requirements(evidence: CapabilityEvidence) -> list[str]:
    missing = list(evidence.missing_requirements)
    if evidence.direct_authoritative_storage_available:
        # A validated agency total-storage series is already an independent
        # Level-4 pathway. Stage/capacity datum work may remain useful for a
        # future cross-check, but it is not a blocker for the direct product.
        return list(dict.fromkeys(missing))
    stage_relation_present = (
        evidence.official_capacity_relation_available
        or evidence.surveyed_bathymetry_available
        or evidence.potential_official_capacity_relation
        or evidence.potential_surveyed_bathymetry
    )
    if stage_relation_present and not evidence.source_vertical_datum_known:
        missing.append("source vertical datum confirmation")
    if stage_relation_present and not evidence.vertical_relation_documented:
        missing.append("documented stage-to-capacity vertical relationship")
    if (
        evidence.water_level_available
        and (
            evidence.official_capacity_relation_available
            or evidence.surveyed_bathymetry_available
        )
        and not evidence.storage_computation_implemented
    ):
        missing.append("stage-to-storage calculation (not implemented in Phase 0)")
    if not evidence.multiple_arm_constraint_resolved:
        missing.append("independent water-body arm handling")
    return list(dict.fromkeys(missing))


def classify_capability(
    profile: HydrometryProfile,
    evidence: CapabilityEvidence,
    *,
    confirmed_sources: Sequence[str] = (),
    likely_sources: Sequence[str] = (),
    unverified_sources: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
) -> CapabilityRecord:
    """Build a conservative capability record from validated Phase 0 evidence.

    A direct, validated *total-storage* series can establish Level 4. Stage plus
    capacity or bathymetry cannot establish Level 4 until a compatible vertical
    relationship and an implemented calculation both exist. The
    ``hydrolakes_modeled_volume_present`` flag is retained only to test that it
    has no effect on capability.
    """

    absolute_storage = _current_absolute_storage(evidence)
    relative_storage = evidence.relative_storage_available or absolute_storage

    if evidence.quantitative_3d_available:
        current = CapabilityLevel.QUANTITATIVE_3D
    elif absolute_storage:
        current = CapabilityLevel.ABSOLUTE_STORAGE
    elif relative_storage:
        current = CapabilityLevel.RELATIVE_STORAGE_CHANGE
    elif evidence.water_level_available:
        current = CapabilityLevel.WATER_LEVEL
    else:
        # All configured pilots have HydroLAKES surface-extent support. Keeping
        # Level 1 explicit avoids inventing a Level 0 outside the product model.
        current = CapabilityLevel.SURFACE_EXTENT

    potential = _potential_level(profile, evidence, current)
    quality = _current_quality(evidence) or profile.potential_source_quality_class
    warning = evidence.key_warning or "; ".join(profile.key_constraints)

    return CapabilityRecord(
        profile_id=profile.profile_id,
        Hylak_id=profile.Hylak_id,
        display_name=profile.display_name,
        state=profile.state,
        extent_available=evidence.extent_available,
        water_level_available=evidence.water_level_available,
        relative_storage_available=relative_storage,
        absolute_storage_available=absolute_storage,
        quantitative_3d_available=evidence.quantitative_3d_available,
        highest_current_capability=current,
        highest_potential_capability=potential,
        source_quality_class=quality,
        confirmed_sources=tuple(confirmed_sources),
        likely_sources=tuple(likely_sources),
        unverified_sources=tuple(unverified_sources),
        missing_requirements=tuple(_auto_missing_requirements(evidence)),
        datum_status=evidence.datum_status,
        key_warning=warning,
        provenance=dict(provenance or {}),
    )
