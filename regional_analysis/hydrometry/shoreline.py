"""Pure shoreline and inundation-change helpers for quantitative grids."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage


class ShorelineError(ValueError):
    """Raised when shoreline masks or grid metadata are invalid."""


INUNDATION_CHANGE_CODES = {
    0: "unchanged_dry",
    1: "persistent_water",
    2: "newly_inundated",
    3: "exposed_or_lost_water",
    255: "nodata_or_unsupported",
}


def _mask(value: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=bool)
    if result.ndim != 2:
        raise ShorelineError(f"{label} must be a two-dimensional mask.")
    return result


def inundation_change_codes(
    latest_inundation: np.ndarray,
    prior_inundation: np.ndarray,
    *,
    supported_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Encode deterministic latest-versus-prior inundation change classes."""

    latest = _mask(latest_inundation, "latest_inundation")
    prior = _mask(prior_inundation, "prior_inundation")
    if latest.shape != prior.shape:
        raise ShorelineError("Latest and prior inundation shapes must match.")
    supported = (
        np.ones(latest.shape, dtype=bool)
        if supported_mask is None
        else _mask(supported_mask, "supported_mask")
    )
    if supported.shape != latest.shape:
        raise ShorelineError("supported_mask must match inundation shape.")
    if np.any((latest | prior) & ~supported):
        raise ShorelineError("Inundation cannot occur outside the supported domain.")
    result = np.full(latest.shape, 255, dtype=np.uint8)
    result[supported & ~latest & ~prior] = 0
    result[supported & latest & prior] = 1
    result[supported & latest & ~prior] = 2
    result[supported & ~latest & prior] = 3
    return result


def shoreline_mask(inundation: np.ndarray, *, connectivity: int = 8) -> np.ndarray:
    """Return wet-side boundary pixels, including boundaries around dry islands."""

    wet = _mask(inundation, "inundation")
    if connectivity not in {4, 8}:
        raise ShorelineError("connectivity must be 4 or 8.")
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    interior = ndimage.binary_erosion(wet, structure=structure, border_value=0)
    return wet & ~interior


@dataclass(frozen=True, slots=True)
class ShorelineDisplacement:
    """Symmetric raster shoreline-distance proxy in horizontal metres."""

    mean_m: float | None
    median_m: float | None
    p95_m: float | None
    maximum_m: float | None
    first_shoreline_cell_count: int
    second_shoreline_cell_count: int
    status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def shoreline_displacement_proxy(
    first_inundation: np.ndarray,
    second_inundation: np.ndarray,
    *,
    pixel_width_m: float,
    pixel_height_m: float,
    connectivity: int = 8,
) -> ShorelineDisplacement:
    """Calculate bidirectional nearest-shoreline raster distances.

    This is a resolution diagnostic, not a surveyed horizontal-error estimate.
    Both directions are pooled so expansion and contraction receive equal
    geometric treatment.
    """

    first = _mask(first_inundation, "first_inundation")
    second = _mask(second_inundation, "second_inundation")
    if first.shape != second.shape:
        raise ShorelineError("Inundation masks must have identical shapes.")
    width = float(pixel_width_m)
    height = float(pixel_height_m)
    if any(not math.isfinite(item) or item <= 0 for item in (width, height)):
        raise ShorelineError("Pixel dimensions must be finite and positive.")
    first_shore = shoreline_mask(first, connectivity=connectivity)
    second_shore = shoreline_mask(second, connectivity=connectivity)
    first_count = int(np.count_nonzero(first_shore))
    second_count = int(np.count_nonzero(second_shore))
    if first_count == 0 or second_count == 0:
        return ShorelineDisplacement(
            mean_m=None,
            median_m=None,
            p95_m=None,
            maximum_m=None,
            first_shoreline_cell_count=first_count,
            second_shoreline_cell_count=second_count,
            status="unavailable_empty_shoreline",
        )
    first_to_second = ndimage.distance_transform_edt(
        ~second_shore, sampling=(height, width)
    )[first_shore]
    second_to_first = ndimage.distance_transform_edt(
        ~first_shore, sampling=(height, width)
    )[second_shore]
    distances = np.concatenate((first_to_second, second_to_first)).astype(
        np.float64, copy=False
    )
    return ShorelineDisplacement(
        mean_m=float(np.mean(distances, dtype=np.float64)),
        median_m=float(np.median(distances)),
        p95_m=float(np.percentile(distances, 95)),
        maximum_m=float(np.max(distances)),
        first_shoreline_cell_count=first_count,
        second_shoreline_cell_count=second_count,
        status="calculated_raster_proxy",
    )


__all__ = [
    "INUNDATION_CHANGE_CODES",
    "ShorelineDisplacement",
    "ShorelineError",
    "inundation_change_codes",
    "shoreline_displacement_proxy",
    "shoreline_mask",
]
