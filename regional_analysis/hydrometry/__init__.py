"""Research-stage hydrometry source discovery and capability inventory.

The package is deliberately independent from the QGIS threshold registry.  It
normalizes provider metadata and bounded observations, but Phase 0 performs no
stage--storage interpolation or volume calculation.
"""

from .models import (
    CapabilityLevel,
    CapabilityRecord,
    HeightType,
    HydrometryVariable,
    NormalizationStatus,
    NormalizedTimeSeriesRecord,
    Provider,
    SourceQualityClass,
    VerticalReference,
)
from .registry import HydrometryRegistry, load_registry

__all__ = [
    "CapabilityLevel",
    "CapabilityRecord",
    "HeightType",
    "HydrometryRegistry",
    "HydrometryVariable",
    "NormalizationStatus",
    "NormalizedTimeSeriesRecord",
    "Provider",
    "SourceQualityClass",
    "VerticalReference",
    "load_registry",
]
