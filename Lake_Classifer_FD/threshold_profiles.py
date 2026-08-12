"""Immutable provenance registry for the QGIS plugin's shipped thresholds.

This module deliberately uses only the Python standard library so it can be
imported and tested without QGIS, Earth Engine, NumPy, or pandas. It records the
two lake-specific profiles already shipped by the plugin; regional
quick-transfer results are not plugin defaults.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

INDEX_NAMES: Final = ("ndwi", "ndvi", "ndmi")
SCOPE_LAKE_SPECIFIC: Final = "lake_specific"
STATUS_SHIPPED: Final = "shipped"

NDWI_NDVI_SOURCE: Final = "data/threshold_cv_summary.csv"
NDMI_SOURCE: Final = "data/ndmi_threshold_cv_all.csv"
NDWI_NDVI_PRODUCER_NOTE: Final = (
    "Produced by the superseded threshold_optimization_cv.py workflow; "
    "that producer script is not present in this repository."
)
NDMI_PRODUCER_NOTE: Final = (
    "Produced by threshold_optimization_ndmi.py, which is present in this repository."
)


@dataclass(frozen=True, slots=True)
class ThresholdProfile:
    """One immutable, fully attributed plugin threshold profile."""

    profile_id: str
    preset_key: str
    display_name: str
    Hylak_id: int
    scope: str
    status: str
    ndwi: float
    ndvi: float
    ndmi: float
    ndwi_reference: str
    ndvi_reference: str
    ndmi_reference: str
    ndwi_source: str
    ndvi_source: str
    ndmi_source: str
    aoi_definition: str
    aoi_bounds_wgs84: tuple[float, float, float, float]
    provenance_note: str

    @property
    def hylak_id(self) -> int:
        """Return the persistent HydroLAKES identifier with Python-style casing."""

        return self.Hylak_id

    @property
    def thresholds(self) -> dict[str, float]:
        """Return a detached compatibility dictionary of index thresholds."""

        return {"ndwi": self.ndwi, "ndvi": self.ndvi, "ndmi": self.ndmi}

    @property
    def references(self) -> dict[str, str]:
        """Return a detached index-to-reference mapping."""

        return {
            "ndwi": self.ndwi_reference,
            "ndvi": self.ndvi_reference,
            "ndmi": self.ndmi_reference,
        }

    @property
    def sources(self) -> dict[str, str]:
        """Return a detached index-to-source-table mapping."""

        return {
            "ndwi": self.ndwi_source,
            "ndvi": self.ndvi_source,
            "ndmi": self.ndmi_source,
        }


_COMMON_PROVENANCE = (
    f"NDWI and NDVI: {NDWI_NDVI_PRODUCER_NOTE} "
    f"NDMI: {NDMI_PRODUCER_NOTE} Shipped values preserve the plugin's existing "
    "three-decimal defaults rather than silently replacing them with the fuller "
    "precision stored in the calibration tables. Hylak_id is identity metadata "
    "only; the plugin continues to use its legacy rectangular preset AOI rather "
    "than a HydroLAKES polygon."
)

GREAT_SALT_LAKE_PROFILE: Final = ThresholdProfile(
    profile_id="great_salt_lake_ut",
    preset_key="Great Salt Lake, UT",
    display_name="Great Salt Lake",
    Hylak_id=67,
    scope=SCOPE_LAKE_SPECIFIC,
    status=STATUS_SHIPPED,
    ndwi=-0.055,
    ndvi=0.165,
    ndmi=0.053,
    ndwi_reference="OPERA",
    ndvi_reference="Dynamic World",
    ndmi_reference="NWI",
    ndwi_source=NDWI_NDVI_SOURCE,
    ndvi_source=NDWI_NDVI_SOURCE,
    ndmi_source=NDMI_SOURCE,
    aoi_definition=(
        "Legacy QGIS rectangular preset in EPSG:4326; Hylak_id is identity "
        "metadata only and does not select a HydroLAKES polygon"
    ),
    aoi_bounds_wgs84=(-113.31, 40.51, -111.85, 41.87),
    provenance_note=_COMMON_PROVENANCE,
)

LAKE_ABERT_PROFILE: Final = ThresholdProfile(
    profile_id="lake_abert_or",
    preset_key="Lake Abert, OR",
    display_name="Lake Abert",
    Hylak_id=776,
    scope=SCOPE_LAKE_SPECIFIC,
    status=STATUS_SHIPPED,
    ndwi=-0.063,
    ndvi=0.130,
    ndmi=0.123,
    ndwi_reference="OPERA",
    ndvi_reference="Dynamic World",
    ndmi_reference="NWI",
    ndwi_source=NDWI_NDVI_SOURCE,
    ndvi_source=NDWI_NDVI_SOURCE,
    ndmi_source=NDMI_SOURCE,
    aoi_definition=(
        "Legacy QGIS rectangular preset in EPSG:4326; Hylak_id is identity "
        "metadata only and does not select a HydroLAKES polygon"
    ),
    aoi_bounds_wgs84=(-120.61, 42.40, -120.00, 42.85),
    provenance_note=_COMMON_PROVENANCE,
)

SHIPPED_THRESHOLD_PROFILES: Final = (
    GREAT_SALT_LAKE_PROFILE,
    LAKE_ABERT_PROFILE,
)

_BY_PROFILE_ID: Final[Mapping[str, ThresholdProfile]] = MappingProxyType(
    {profile.profile_id: profile for profile in SHIPPED_THRESHOLD_PROFILES}
)
_BY_PRESET_KEY: Final[Mapping[str, ThresholdProfile]] = MappingProxyType(
    {profile.preset_key: profile for profile in SHIPPED_THRESHOLD_PROFILES}
)
_BY_HYLAK_ID: Final[Mapping[int, ThresholdProfile]] = MappingProxyType(
    {profile.hylak_id: profile for profile in SHIPPED_THRESHOLD_PROFILES}
)

# Compatibility contract consumed by dock.py. This detached dictionary has
# exactly the historical preset -> index -> float shape. The immutable profile
# records remain authoritative even if an external caller mutates this copy.
DEFAULT_THRESHOLDS: Final = {
    profile.preset_key: profile.thresholds for profile in SHIPPED_THRESHOLD_PROFILES
}


def _lookup(
    registry: Mapping[object, ThresholdProfile],
    key: object,
    *,
    strict: bool,
) -> ThresholdProfile | None:
    profile = registry.get(key)
    if profile is None and strict:
        raise KeyError(key)
    return profile


def get_threshold_profile(
    profile_id: str,
    *,
    strict: bool = False,
) -> ThresholdProfile | None:
    """Look up a profile ID, returning ``None`` or raising in strict mode."""

    return _lookup(_BY_PROFILE_ID, profile_id, strict=strict)


def get_profile_for_preset(
    preset_key: str,
    *,
    strict: bool = False,
) -> ThresholdProfile | None:
    """Look up the exact QGIS preset label."""

    return _lookup(_BY_PRESET_KEY, preset_key, strict=strict)


def get_profile_for_hylak_id(
    hylak_id: int,
    *,
    strict: bool = False,
) -> ThresholdProfile | None:
    """Look up the persistent HydroLAKES identifier."""

    return _lookup(_BY_HYLAK_ID, hylak_id, strict=strict)


def default_thresholds_copy() -> dict[str, dict[str, float]]:
    """Return a fresh compatibility mapping for callers that need mutation."""

    return {
        profile.preset_key: profile.thresholds for profile in SHIPPED_THRESHOLD_PROFILES
    }
