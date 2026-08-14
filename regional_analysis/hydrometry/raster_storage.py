"""Pure quantitative raster science for the Bear Lake Phase 2 prototype.

Array functions are independent of rasterio and are therefore straightforward
to test with exact synthetic basins.  File-backed helpers import rasterio only
inside the function and process block windows; they never materialize the full
source raster in memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from datetime import date
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage


class RasterStorageError(ValueError):
    """Raised when raster storage inputs violate the quantitative contract."""


class ConnectivityError(RasterStorageError):
    """Raised when the trusted lake seed cannot define one inundated component."""


class ValidationPolicyError(RasterStorageError):
    """Raised when a numerical validation policy is incomplete or inconsistent."""


SUPPORTED_BASIN_METHODS = {
    "all_valid_below_stage",
    "fixed_basin_below_stage",
    "connected_seed",
}
SUPPORTED_GATE_STATUSES = {"validated", "numerically_consistent", "failed"}
SUPPORTED_SOURCE_METHOD_STATUSES = {
    "not_assessed",
    "source_method_consistent",
    "source_method_consistent_with_numeric_area_warning",
    "source_method_inconsistent",
}


def _rasterio() -> Any:
    try:
        import pyexpat  # noqa: F401
        import sqlite3  # noqa: F401

        import rasterio
    except ImportError as exc:  # pragma: no cover - depends on local GDAL build
        raise RasterStorageError(
            "File-backed raster processing requires rasterio/GDAL."
        ) from exc
    return rasterio


def _datum_token(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    token = "".join(
        character for character in str(value).upper() if character.isalnum()
    )
    if "NAVD88" in token or "NORTHAMERICANVERTICALDATUM1988" in token:
        return "NAVD88"
    return token


def _positive_pixel_dimensions(width: float, height: float) -> tuple[float, float]:
    values = (float(width), float(height))
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise RasterStorageError("Pixel dimensions must be finite and positive.")
    return values


def _array_inputs(
    elevation_m: np.ndarray,
    valid_mask: np.ndarray | None,
    basin_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    elevation = np.asarray(elevation_m, dtype=np.float64)
    if elevation.ndim != 2:
        raise RasterStorageError("Elevation must be a two-dimensional array.")
    finite = np.isfinite(elevation)
    if valid_mask is None:
        valid = finite
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != elevation.shape:
            raise RasterStorageError("valid_mask must match elevation shape.")
        valid = valid & finite
    basin = None
    if basin_mask is not None:
        basin = np.asarray(basin_mask, dtype=bool)
        if basin.shape != elevation.shape:
            raise RasterStorageError("basin_mask must match elevation shape.")
    return elevation, valid, basin


def _floating_stage_plane_dtype(
    configured: Any | None, sample_elevation: np.ndarray | None = None
) -> np.dtype[Any]:
    """Resolve the floating dtype used by the raster stage-plane operation."""

    candidate = configured
    if candidate is None and sample_elevation is not None:
        candidate = np.asanyarray(sample_elevation).dtype
    dtype = np.dtype(np.float64 if candidate is None else candidate)
    # Integer rasters still accept continuous water levels. They must not
    # truncate the requested stage to the source integer dtype.
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(np.float64)
    if dtype.itemsize > np.dtype(np.float64).itemsize:
        raise RasterStorageError(
            f"Stage-plane dtype {dtype} exceeds supported float64 precision."
        )
    return dtype


def _effective_stage_plane(stage_m: float, dtype: np.dtype[Any]) -> float:
    requested = float(stage_m)
    if not math.isfinite(requested):
        raise RasterStorageError("Stage must be finite.")
    with np.errstate(over="ignore", invalid="ignore"):
        effective = float(dtype.type(requested))
    if not math.isfinite(effective):
        raise RasterStorageError(
            f"Stage {requested!r} is not representable by stage-plane dtype {dtype}."
        )
    return effective


def _stage_plane_warnings(
    requested_stage_m: float, effective_stage_m: float, dtype: np.dtype[Any]
) -> tuple[str, ...]:
    if requested_stage_m == effective_stage_m:
        return ()
    return (f"requested_stage_aligned_to_{dtype.name}_plane",)


def _prepare_window_stream(
    windows: Iterable[ElevationWindow], stage_plane_dtype: Any | None
) -> tuple[Iterable[ElevationWindow], np.dtype[Any]]:
    """Peek one window so array APIs can retain the source floating dtype."""

    iterator = iter(windows)
    try:
        first = next(iterator)
    except StopIteration:
        return (), _floating_stage_plane_dtype(stage_plane_dtype)
    dtype = _floating_stage_plane_dtype(stage_plane_dtype, first.elevation_m)
    return chain((first,), iterator), dtype


def _seed_coordinates(
    trusted_seed: tuple[int, int] | Sequence[tuple[int, int]] | np.ndarray | None,
    shape: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    if trusted_seed is None:
        raise ConnectivityError("connected_seed requires a documented trusted seed.")
    if isinstance(trusted_seed, np.ndarray):
        if trusted_seed.dtype == bool:
            if trusted_seed.shape != shape:
                raise ConnectivityError(
                    "Trusted seed mask must match the raster shape."
                )
            coordinates = [
                tuple(int(item) for item in row) for row in np.argwhere(trusted_seed)
            ]
        else:
            coordinates = [tuple(int(item) for item in row) for row in trusted_seed]
    elif (
        len(trusted_seed) == 2
        and isinstance(trusted_seed[0], (int, np.integer))
        and isinstance(trusted_seed[1], (int, np.integer))
    ):
        coordinates = [(int(trusted_seed[0]), int(trusted_seed[1]))]
    else:
        coordinates = [tuple(int(item) for item in row) for row in trusted_seed]
    if not coordinates:
        raise ConnectivityError("Trusted seed must contain at least one pixel.")
    for row, column in coordinates:
        if row < 0 or column < 0 or row >= shape[0] or column >= shape[1]:
            raise ConnectivityError(
                f"Trusted seed {(row, column)} is outside the raster."
            )
    return tuple(coordinates)


def compute_connected_inundation(
    below_stage: np.ndarray,
    *,
    trusted_seed: tuple[int, int] | Sequence[tuple[int, int]] | np.ndarray,
    connectivity: int = 8,
) -> np.ndarray:
    """Return the one flooded component containing a verified trusted seed.

    Every configured seed pixel must be inundated, and all seed pixels must
    occupy the same component.  This makes a stale or ambiguous seed fail
    loudly rather than silently selecting a different depression.
    """

    flooded = np.asarray(below_stage, dtype=bool)
    if flooded.ndim != 2:
        raise ConnectivityError("below_stage must be a two-dimensional mask.")
    if connectivity not in {4, 8}:
        raise ConnectivityError("connectivity must be 4 or 8.")
    seeds = _seed_coordinates(trusted_seed, flooded.shape)
    for seed in seeds:
        if not flooded[seed]:
            raise ConnectivityError(
                f"Trusted seed pixel {seed} is not inundated at this stage."
            )
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, _ = ndimage.label(flooded, structure=structure)
    seed_labels = {int(labels[seed]) for seed in seeds}
    seed_labels.discard(0)
    if len(seed_labels) != 1:
        raise ConnectivityError(
            "Trusted seed pixels do not identify one connected lake component."
        )
    return labels == seed_labels.pop()


@dataclass(slots=True)
class StageRasterResult:
    """Depth, inundation, and exact grid sums for one stage."""

    stage_m_navd88: float | None
    inundation: np.ndarray
    depth_m: np.ndarray
    supported_mask: np.ndarray
    area_m2: float | None
    storage_m3: float | None
    flooded_cell_count: int
    cell_area_m2: float
    basin_method: str
    connectivity: int | None
    status: str
    calculation_stage_m_navd88: float | None = None
    stage_plane_dtype: str | None = None
    warning_flags: tuple[str, ...] = ()

    def scalar_dict(self) -> dict[str, Any]:
        """Return scalars only; large arrays are intentionally not serialized."""

        return {
            "stage_m_navd88": self.stage_m_navd88,
            "requested_stage_m_navd88": self.stage_m_navd88,
            "calculation_stage_m_navd88": self.calculation_stage_m_navd88,
            "stage_plane_dtype": self.stage_plane_dtype,
            "area_m2": self.area_m2,
            "storage_m3": self.storage_m3,
            "flooded_cell_count": self.flooded_cell_count,
            "cell_area_m2": self.cell_area_m2,
            "basin_method": self.basin_method,
            "connectivity": self.connectivity,
            "status": self.status,
            "warning_flags": list(self.warning_flags),
        }


def compute_stage_raster(
    elevation_m: np.ndarray,
    stage_m: float | None,
    *,
    pixel_width_m: float,
    pixel_height_m: float,
    valid_mask: np.ndarray | None = None,
    basin_mask: np.ndarray | None = None,
    basin_method: str = "connected_seed",
    trusted_seed: tuple[int, int]
    | Sequence[tuple[int, int]]
    | np.ndarray
    | None = None,
    connectivity: int = 8,
    supported_stage_range_m: tuple[float, float] | None = None,
    stage_vertical_datum: str = "NAVD88",
    raster_vertical_datum: str = "NAVD88",
    stage_plane_dtype: Any | None = None,
) -> StageRasterResult:
    """Calculate connected inundation, water depth, area, and storage.

    A cell is flooded when its bed elevation is less than or equal to the
    effective stage plane. The requested stage is aligned to the source
    floating dtype before comparison and depth arithmetic, matching raster
    calculator behavior while retaining both values in the result. Cells
    exactly at the effective stage are wet with zero depth. Dry supported
    cells have depth zero; unsupported/nodata cells have ``NaN`` depth.
    """

    source_array = np.asanyarray(elevation_m)
    plane_dtype = _floating_stage_plane_dtype(stage_plane_dtype, source_array)
    elevation, valid, basin = _array_inputs(elevation_m, valid_mask, basin_mask)
    width, height = _positive_pixel_dimensions(pixel_width_m, pixel_height_m)
    cell_area = width * height
    if basin_method not in SUPPORTED_BASIN_METHODS:
        raise RasterStorageError(f"Unsupported basin method {basin_method!r}.")
    if basin_method in {"fixed_basin_below_stage", "connected_seed"} and basin is None:
        if basin_method == "fixed_basin_below_stage":
            raise RasterStorageError("fixed_basin_below_stage requires basin_mask.")
        basin = np.ones_like(valid)
    supported = valid if basin is None else valid & basin
    blank_depth = np.full(elevation.shape, np.nan, dtype=np.float64)
    blank_depth[supported] = 0.0
    blank_inundation = np.zeros(elevation.shape, dtype=bool)

    if _datum_token(stage_vertical_datum) != _datum_token(raster_vertical_datum):
        raise RasterStorageError(
            "Stage and raster vertical datums must match; no transformation is inferred."
        )
    if stage_m is None:
        return StageRasterResult(
            stage_m_navd88=None,
            inundation=blank_inundation,
            depth_m=blank_depth,
            supported_mask=supported,
            area_m2=None,
            storage_m3=None,
            flooded_cell_count=0,
            cell_area_m2=cell_area,
            basin_method=basin_method,
            connectivity=connectivity if basin_method == "connected_seed" else None,
            status="unavailable_null_stage",
            calculation_stage_m_navd88=None,
            stage_plane_dtype=plane_dtype.name,
            warning_flags=("null_stage_not_calculated",),
        )
    stage = float(stage_m)
    if not math.isfinite(stage):
        raise RasterStorageError("stage_m must be finite or null.")
    calculation_stage = _effective_stage_plane(stage, plane_dtype)
    plane_warnings = _stage_plane_warnings(stage, calculation_stage, plane_dtype)
    if supported_stage_range_m is not None:
        lower, upper = (float(item) for item in supported_stage_range_m)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise RasterStorageError("supported_stage_range_m is invalid.")
        if stage < lower or stage > upper:
            direction = "below" if stage < lower else "above"
            return StageRasterResult(
                stage_m_navd88=stage,
                inundation=blank_inundation,
                depth_m=blank_depth,
                supported_mask=supported,
                area_m2=None,
                storage_m3=None,
                flooded_cell_count=0,
                cell_area_m2=cell_area,
                basin_method=basin_method,
                connectivity=connectivity if basin_method == "connected_seed" else None,
                status=f"{direction}_supported_range",
                calculation_stage_m_navd88=calculation_stage,
                stage_plane_dtype=plane_dtype.name,
                warning_flags=("no_extrapolation", *plane_warnings),
            )

    candidate = supported & (elevation <= calculation_stage)
    if basin_method == "connected_seed":
        inundation = compute_connected_inundation(
            candidate,
            trusted_seed=trusted_seed,
            connectivity=connectivity,
        )
    else:
        inundation = candidate
    depth = blank_depth
    depth[inundation] = calculation_stage - elevation[inundation]
    flooded_count = int(np.count_nonzero(inundation))
    area = float(np.float64(flooded_count) * np.float64(cell_area))
    storage = float(np.sum(depth[inundation], dtype=np.float64) * np.float64(cell_area))
    return StageRasterResult(
        stage_m_navd88=stage,
        inundation=inundation,
        depth_m=depth,
        supported_mask=supported,
        area_m2=area,
        storage_m3=storage,
        flooded_cell_count=flooded_count,
        cell_area_m2=cell_area,
        basin_method=basin_method,
        connectivity=connectivity if basin_method == "connected_seed" else None,
        status="calculated",
        calculation_stage_m_navd88=calculation_stage,
        stage_plane_dtype=plane_dtype.name,
        warning_flags=plane_warnings,
    )


@dataclass(frozen=True, slots=True)
class ElevationWindow:
    """One source-raster block and optional precomputed inundation masks."""

    elevation_m: np.ndarray
    valid_mask: np.ndarray
    basin_mask: np.ndarray | None = None
    inundation_by_stage: Mapping[float, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RasterScalarResult:
    """Memory-safe scalar integration result for one raster/stage pair."""

    stage_m_navd88: float
    area_m2: float
    storage_m3: float
    flooded_cell_count: int
    valid_cell_count: int
    cell_area_m2: float
    calculation_method: str
    basin_method: str
    source_resolution_m: float
    working_resolution_m: float
    processing_seconds: float
    peak_memory_estimate_bytes: int
    calculation_stage_m_navd88: float | None = None
    stage_plane_dtype: str | None = None
    warning_flags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["requested_stage_m_navd88"] = self.stage_m_navd88
        return result


def integrate_elevation_windows(
    windows: Iterable[ElevationWindow],
    stages_m: Sequence[float],
    *,
    pixel_width_m: float,
    pixel_height_m: float,
    basin_method: str = "fixed_basin_below_stage",
    source_resolution_m: float | None = None,
    working_resolution_m: float | None = None,
    calculation_method: str = "direct_windowed_inclusive_float64",
    stage_plane_dtype: Any | None = None,
) -> tuple[RasterScalarResult, ...]:
    """Integrate many stages in one streaming pass over elevation windows.

    Connected masks must be computed globally and supplied in each window via
    ``inundation_by_stage``.  Performing component labeling independently per
    window would be scientifically incorrect and is therefore rejected.
    """

    width, height = _positive_pixel_dimensions(pixel_width_m, pixel_height_m)
    cell_area = width * height
    if basin_method not in SUPPORTED_BASIN_METHODS:
        raise RasterStorageError(f"Unsupported basin method {basin_method!r}.")
    stages = tuple(float(stage) for stage in stages_m)
    if not stages or any(not math.isfinite(stage) for stage in stages):
        raise RasterStorageError("At least one finite stage is required.")
    if len(set(stages)) != len(stages):
        raise RasterStorageError("Stages must be unique.")
    prepared_windows, plane_dtype = _prepare_window_stream(windows, stage_plane_dtype)
    calculation_stages = tuple(
        _effective_stage_plane(stage, plane_dtype) for stage in stages
    )
    if len(set(calculation_stages)) != len(calculation_stages):
        raise RasterStorageError(
            "Requested stages collapse to duplicate effective stage planes."
        )
    areas = np.zeros(len(stages), dtype=np.float64)
    volumes = np.zeros(len(stages), dtype=np.float64)
    counts = np.zeros(len(stages), dtype=np.int64)
    valid_count = 0
    peak_memory = 0
    started = time.perf_counter()
    for window in prepared_windows:
        elevation, valid, basin = _array_inputs(
            window.elevation_m, window.valid_mask, window.basin_mask
        )
        domain = valid if basin is None else valid & basin
        valid_count += int(np.count_nonzero(domain))
        peak_memory = max(
            peak_memory,
            int(elevation.nbytes + valid.nbytes + domain.nbytes),
        )
        for index, (stage, calculation_stage) in enumerate(
            zip(stages, calculation_stages, strict=True)
        ):
            if basin_method == "connected_seed":
                supplied = window.inundation_by_stage.get(stage)
                if supplied is None:
                    raise RasterStorageError(
                        "Windowed connected integration requires a globally "
                        "computed inundation mask for every stage."
                    )
                flooded = np.asarray(supplied, dtype=bool)
                if flooded.shape != elevation.shape:
                    raise RasterStorageError(
                        "A supplied inundation window has an incompatible shape."
                    )
                if np.any(flooded & ~domain):
                    raise RasterStorageError(
                        "A supplied inundation mask includes unsupported cells."
                    )
                if np.any(flooded & (elevation > calculation_stage)):
                    raise RasterStorageError(
                        "A supplied inundation mask includes cells above the "
                        "effective stage plane."
                    )
            else:
                flooded = domain & (elevation <= calculation_stage)
            count = int(np.count_nonzero(flooded))
            counts[index] += count
            areas[index] += np.float64(count) * np.float64(cell_area)
            if count:
                volumes[index] += np.sum(
                    calculation_stage - elevation[flooded], dtype=np.float64
                ) * np.float64(cell_area)
            peak_memory = max(peak_memory, int(elevation.nbytes + 3 * flooded.nbytes))
    elapsed = time.perf_counter() - started
    source_resolution = source_resolution_m or math.sqrt(cell_area)
    working_resolution = working_resolution_m or math.sqrt(cell_area)
    return tuple(
        RasterScalarResult(
            stage_m_navd88=stage,
            area_m2=float(areas[index]),
            storage_m3=float(volumes[index]),
            flooded_cell_count=int(counts[index]),
            valid_cell_count=valid_count,
            cell_area_m2=cell_area,
            calculation_method=calculation_method,
            basin_method=basin_method,
            source_resolution_m=float(source_resolution),
            working_resolution_m=float(working_resolution),
            processing_seconds=elapsed,
            peak_memory_estimate_bytes=peak_memory,
            calculation_stage_m_navd88=calculation_stages[index],
            stage_plane_dtype=plane_dtype.name,
            warning_flags=_stage_plane_warnings(
                stage, calculation_stages[index], plane_dtype
            ),
        )
        for index, stage in enumerate(stages)
    )


def integrate_threshold_event_windows(
    windows: Iterable[ElevationWindow],
    stages_m: Sequence[float],
    *,
    pixel_width_m: float,
    pixel_height_m: float,
    basin_method: str = "fixed_basin_below_stage",
    source_resolution_m: float | None = None,
    working_resolution_m: float | None = None,
    stage_plane_dtype: Any | None = None,
) -> tuple[RasterScalarResult, ...]:
    """Integrate many thresholds in one pass using exact event attribution.

    Each valid bed elevation is assigned to the first sorted effective stage
    greater than or equal to it. Per-stage counts and actual float64
    bed-elevation sums are then accumulated cumulatively. This has exact
    ``bed <= effective stage`` semantics without histogram bin rounding and
    avoids comparing every source cell independently with every stage.

    Connected masks cannot be reconstructed from scalar events; callers must
    use the working-grid connectivity engine for that method.
    """

    width, height = _positive_pixel_dimensions(pixel_width_m, pixel_height_m)
    if basin_method not in {"all_valid_below_stage", "fixed_basin_below_stage"}:
        raise RasterStorageError(
            "Threshold-event integration supports only all-valid or fixed-basin domains."
        )
    original = tuple(float(stage) for stage in stages_m)
    if not original or any(not math.isfinite(stage) for stage in original):
        raise RasterStorageError("At least one finite stage is required.")
    if len(set(original)) != len(original):
        raise RasterStorageError("Stages must be unique.")
    prepared_windows, plane_dtype = _prepare_window_stream(windows, stage_plane_dtype)
    effective = np.asarray(
        [_effective_stage_plane(stage, plane_dtype) for stage in original],
        dtype=np.float64,
    )
    if len(set(effective.tolist())) != len(original):
        raise RasterStorageError(
            "Requested stages collapse to duplicate effective stage planes."
        )
    order = np.argsort(effective, kind="stable")
    stages = effective[order]
    event_counts = np.zeros(stages.size + 1, dtype=np.int64)
    event_sums = np.zeros(stages.size + 1, dtype=np.float64)
    valid_count = 0
    peak_memory = int(stages.nbytes + event_counts.nbytes + event_sums.nbytes)
    started = time.perf_counter()
    for window in prepared_windows:
        elevation, valid, basin = _array_inputs(
            window.elevation_m, window.valid_mask, window.basin_mask
        )
        if basin_method == "fixed_basin_below_stage" and basin is None:
            raise RasterStorageError(
                "fixed_basin_below_stage event integration requires basin_mask."
            )
        domain = valid if basin is None else valid & basin
        values = elevation[domain]
        valid_count += int(values.size)
        if values.size:
            first_stage = np.searchsorted(stages, values, side="left")
            event_counts += np.bincount(first_stage, minlength=stages.size + 1).astype(
                np.int64
            )
            event_sums += np.bincount(
                first_stage, weights=values, minlength=stages.size + 1
            ).astype(np.float64)
            peak_memory = max(
                peak_memory,
                int(
                    elevation.nbytes
                    + valid.nbytes
                    + domain.nbytes
                    + values.nbytes
                    + first_stage.nbytes
                ),
            )
    cumulative_counts = np.cumsum(event_counts[:-1], dtype=np.int64)
    cumulative_sums = np.cumsum(event_sums[:-1], dtype=np.float64)
    cell_area = np.float64(width * height)
    elapsed = time.perf_counter() - started
    source_resolution = float(source_resolution_m or math.sqrt(float(cell_area)))
    working_resolution = float(working_resolution_m or math.sqrt(float(cell_area)))
    results: list[RasterScalarResult | None] = [None] * len(original)
    for index, stage in enumerate(stages):
        original_index = int(order[index])
        requested_stage = original[original_index]
        count = int(cumulative_counts[index])
        area = float(np.float64(count) * cell_area)
        storage = float(
            (np.float64(stage) * np.float64(count) - cumulative_sums[index]) * cell_area
        )
        results[original_index] = RasterScalarResult(
            stage_m_navd88=requested_stage,
            area_m2=area,
            storage_m3=storage,
            flooded_cell_count=count,
            valid_cell_count=valid_count,
            cell_area_m2=float(cell_area),
            calculation_method="threshold_event_inclusive_windowed_float64",
            basin_method=basin_method,
            source_resolution_m=source_resolution,
            working_resolution_m=working_resolution,
            processing_seconds=elapsed,
            peak_memory_estimate_bytes=peak_memory,
            calculation_stage_m_navd88=float(stage),
            stage_plane_dtype=plane_dtype.name,
            warning_flags=_stage_plane_warnings(
                requested_stage, float(stage), plane_dtype
            ),
        )
    if any(result is None for result in results):  # pragma: no cover - defensive
        raise RasterStorageError("Threshold-event result attribution is incomplete.")
    return tuple(result for result in results if result is not None)


def integrate_raster_stages(
    raster_path: str | Path,
    stages_m: Sequence[float],
    *,
    basin_mask_path: str | Path | None = None,
    inundation_mask_paths: Mapping[float, str | Path] | None = None,
    basin_method: str = "fixed_basin_below_stage",
    source_resolution_m: float | None = None,
) -> tuple[RasterScalarResult, ...]:
    """Integrate stages block-by-block from a file-backed elevation raster."""

    rasterio = _rasterio()
    stages = tuple(float(stage) for stage in stages_m)
    with ExitStack() as stack:
        source = stack.enter_context(rasterio.open(raster_path, "r"))
        transform = source.transform
        if not math.isclose(transform.b, 0.0, abs_tol=1e-12) or not math.isclose(
            transform.d, 0.0, abs_tol=1e-12
        ):
            raise RasterStorageError("Rotated/sheared grids are not supported.")
        basin_source = (
            stack.enter_context(rasterio.open(basin_mask_path, "r"))
            if basin_mask_path is not None
            else None
        )
        stage_sources = {
            float(stage): stack.enter_context(rasterio.open(path, "r"))
            for stage, path in (inundation_mask_paths or {}).items()
        }
        comparison_sources = ([basin_source] if basin_source else []) + list(
            stage_sources.values()
        )
        for comparison in comparison_sources:
            if (
                comparison.width != source.width
                or comparison.height != source.height
                or comparison.transform != source.transform
                or comparison.crs != source.crs
            ):
                raise RasterStorageError(
                    "All masks must use the elevation source grid."
                )

        def block_iterator() -> Iterable[ElevationWindow]:
            for _, window in source.block_windows(1):
                values = source.read(1, window=window, masked=True)
                elevation = np.asarray(values.data, dtype=np.float64)
                valid = ~np.ma.getmaskarray(values) & np.isfinite(elevation)
                basin = (
                    basin_source.read(1, window=window).astype(bool)
                    if basin_source is not None
                    else None
                )
                supplied = {
                    stage: dataset.read(1, window=window).astype(bool)
                    for stage, dataset in stage_sources.items()
                }
                yield ElevationWindow(elevation, valid, basin, supplied)

        return integrate_elevation_windows(
            block_iterator(),
            stages,
            pixel_width_m=abs(float(transform.a)),
            pixel_height_m=abs(float(transform.e)),
            basin_method=basin_method,
            source_resolution_m=source_resolution_m,
            working_resolution_m=math.sqrt(abs(float(transform.a * transform.e))),
            stage_plane_dtype=source.dtypes[0],
        )


def integrate_raster_threshold_events(
    raster_path: str | Path,
    stages_m: Sequence[float],
    *,
    basin_mask_path: str | Path | None = None,
    basin_method: str = "all_valid_below_stage",
    source_resolution_m: float | None = None,
) -> tuple[RasterScalarResult, ...]:
    """Apply exact threshold-event integration to a block-window raster."""

    rasterio = _rasterio()
    with ExitStack() as stack:
        source = stack.enter_context(rasterio.open(raster_path, "r"))
        transform = source.transform
        if not math.isclose(transform.b, 0.0, abs_tol=1e-12) or not math.isclose(
            transform.d, 0.0, abs_tol=1e-12
        ):
            raise RasterStorageError("Rotated/sheared grids are not supported.")
        basin_source = (
            stack.enter_context(rasterio.open(basin_mask_path, "r"))
            if basin_mask_path is not None
            else None
        )
        if basin_source is not None and (
            basin_source.width != source.width
            or basin_source.height != source.height
            or basin_source.transform != source.transform
            or basin_source.crs != source.crs
        ):
            raise RasterStorageError("The basin mask must use the source grid.")

        def block_iterator() -> Iterable[ElevationWindow]:
            for _, window in source.block_windows(1):
                values = source.read(1, window=window, masked=True)
                elevation = np.asarray(values.data, dtype=np.float64)
                valid = ~np.ma.getmaskarray(values) & np.isfinite(elevation)
                basin = (
                    basin_source.read(1, window=window).astype(bool)
                    if basin_source is not None
                    else None
                )
                yield ElevationWindow(elevation, valid, basin)

        return integrate_threshold_event_windows(
            block_iterator(),
            stages_m,
            pixel_width_m=abs(float(transform.a)),
            pixel_height_m=abs(float(transform.e)),
            basin_method=basin_method,
            source_resolution_m=source_resolution_m,
            working_resolution_m=math.sqrt(abs(float(transform.a * transform.e))),
            stage_plane_dtype=source.dtypes[0],
        )


@dataclass(frozen=True, slots=True)
class ElevationHistogram:
    """Streaming inclusive-edge histogram with exact bed-elevation sums."""

    bin_origin_m: float
    bin_width_m: float
    counts: np.ndarray
    elevation_sums_m: np.ndarray
    cell_area_m2: float
    valid_cell_count: int
    processing_seconds: float = 0.0
    peak_memory_estimate_bytes: int = 0
    origin_count: int = 0
    origin_elevation_sum_m: float = 0.0
    stage_plane_dtype: str = "float64"
    bin_semantics: str = (
        "origin retained separately; bins are (left,right]; "
        "flooded when bed <= effective aligned stage"
    )

    def __post_init__(self) -> None:
        counts = np.array(self.counts, dtype=np.int64, copy=True)
        sums = np.array(self.elevation_sums_m, dtype=np.float64, copy=True)
        if counts.ndim != 1 or sums.shape != counts.shape or counts.size == 0:
            raise RasterStorageError("Histogram count and sum arrays must align.")
        if np.any(counts < 0) or not np.all(np.isfinite(sums)):
            raise RasterStorageError("Histogram contains invalid count or sum values.")
        if not math.isfinite(self.bin_origin_m):
            raise RasterStorageError("Histogram origin must be finite.")
        if not math.isfinite(self.bin_width_m) or self.bin_width_m <= 0:
            raise RasterStorageError("Histogram bin width must be positive.")
        if not math.isfinite(self.cell_area_m2) or self.cell_area_m2 <= 0:
            raise RasterStorageError("Histogram cell area must be positive.")
        plane_dtype = _floating_stage_plane_dtype(self.stage_plane_dtype)
        if self.origin_count < 0 or not math.isfinite(self.origin_elevation_sum_m):
            raise RasterStorageError("Histogram origin totals are invalid.")
        if int(counts.sum(dtype=np.int64)) + self.origin_count != self.valid_cell_count:
            raise RasterStorageError("Histogram valid-cell count is inconsistent.")
        counts.setflags(write=False)
        sums.setflags(write=False)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "elevation_sums_m", sums)
        object.__setattr__(self, "stage_plane_dtype", plane_dtype.name)

    @property
    def upper_edge_m(self) -> float:
        return self.bin_origin_m + self.bin_width_m * self.counts.size

    @property
    def effective_bin_origin_m(self) -> float:
        dtype = _floating_stage_plane_dtype(self.stage_plane_dtype)
        return _effective_stage_plane(self.bin_origin_m, dtype)

    @property
    def effective_upper_edge_m(self) -> float:
        dtype = _floating_stage_plane_dtype(self.stage_plane_dtype)
        return _effective_stage_plane(self.upper_edge_m, dtype)


def build_elevation_histogram(
    windows: Iterable[ElevationWindow],
    *,
    bin_origin_m: float,
    bin_width_m: float,
    bin_count: int,
    cell_area_m2: float,
    stage_plane_dtype: Any | None = None,
) -> ElevationHistogram:
    """Build a float64/count64 histogram without retaining source windows."""

    if bin_count <= 0:
        raise RasterStorageError("bin_count must be positive.")
    if not math.isfinite(bin_width_m) or bin_width_m <= 0:
        raise RasterStorageError("bin_width_m must be positive.")
    counts = np.zeros(bin_count, dtype=np.int64)
    sums = np.zeros(bin_count, dtype=np.float64)
    prepared_windows, plane_dtype = _prepare_window_stream(windows, stage_plane_dtype)
    nominal_edges = float(bin_origin_m) + np.arange(
        bin_count + 1, dtype=np.float64
    ) * float(bin_width_m)
    edges = np.asarray(nominal_edges, dtype=plane_dtype).astype(np.float64)
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise RasterStorageError(
            "Histogram edges are not distinct in the effective stage-plane dtype."
        )
    origin_count = 0
    origin_sum = np.float64(0.0)
    peak_memory = int(counts.nbytes + sums.nbytes + edges.nbytes)
    started = time.perf_counter()
    for window in prepared_windows:
        elevation, valid, basin = _array_inputs(
            window.elevation_m, window.valid_mask, window.basin_mask
        )
        domain = valid if basin is None else valid & basin
        values = elevation[domain]
        peak_memory = max(
            peak_memory,
            int(
                counts.nbytes
                + sums.nbytes
                + elevation.nbytes
                + valid.nbytes
                + domain.nbytes
                + values.nbytes
            ),
        )
        if not values.size:
            continue
        if np.any(values < edges[0]) or np.any(values > edges[-1]):
            raise RasterStorageError(
                "A valid elevation lies outside the configured histogram range."
            )
        at_origin = values == edges[0]
        if np.any(at_origin):
            origin_values = values[at_origin]
            origin_count += int(origin_values.size)
            origin_sum += np.sum(origin_values, dtype=np.float64)
        remaining = values[~at_origin]
        if remaining.size:
            # Equality belongs to the interval ending at that edge, yielding
            # exact inclusive evaluation at every configured stage plane.
            indices = np.searchsorted(edges, remaining, side="left") - 1
            if np.any(indices < 0) or np.any(indices >= bin_count):
                raise RasterStorageError(
                    "A valid elevation lies outside the configured histogram range."
                )
            counts += np.bincount(indices, minlength=bin_count).astype(np.int64)
            sums += np.bincount(indices, weights=remaining, minlength=bin_count).astype(
                np.float64
            )
    return ElevationHistogram(
        bin_origin_m=float(bin_origin_m),
        bin_width_m=float(bin_width_m),
        counts=counts,
        elevation_sums_m=sums,
        cell_area_m2=float(cell_area_m2),
        valid_cell_count=int(counts.sum(dtype=np.int64)) + origin_count,
        processing_seconds=time.perf_counter() - started,
        peak_memory_estimate_bytes=peak_memory,
        origin_count=origin_count,
        origin_elevation_sum_m=float(origin_sum),
        stage_plane_dtype=plane_dtype.name,
    )


def build_raster_elevation_histogram(
    raster_path: str | Path,
    *,
    bin_origin_m: float,
    bin_width_m: float,
    bin_count: int,
    basin_mask_path: str | Path | None = None,
) -> ElevationHistogram:
    """Build the aligned histogram in one block-window pass over a raster."""

    rasterio = _rasterio()
    with ExitStack() as stack:
        source = stack.enter_context(rasterio.open(raster_path, "r"))
        transform = source.transform
        if not math.isclose(transform.b, 0.0, abs_tol=1e-12) or not math.isclose(
            transform.d, 0.0, abs_tol=1e-12
        ):
            raise RasterStorageError("Rotated/sheared grids are not supported.")
        basin_source = (
            stack.enter_context(rasterio.open(basin_mask_path, "r"))
            if basin_mask_path is not None
            else None
        )
        if basin_source is not None and (
            basin_source.width != source.width
            or basin_source.height != source.height
            or basin_source.transform != source.transform
            or basin_source.crs != source.crs
        ):
            raise RasterStorageError("The basin mask must use the source grid.")

        def block_iterator() -> Iterable[ElevationWindow]:
            for _, window in source.block_windows(1):
                values = source.read(1, window=window, masked=True)
                elevation = np.asarray(values.data, dtype=np.float64)
                valid = ~np.ma.getmaskarray(values) & np.isfinite(elevation)
                basin = (
                    basin_source.read(1, window=window).astype(bool)
                    if basin_source is not None
                    else None
                )
                yield ElevationWindow(elevation, valid, basin)

        return build_elevation_histogram(
            block_iterator(),
            bin_origin_m=bin_origin_m,
            bin_width_m=bin_width_m,
            bin_count=bin_count,
            cell_area_m2=abs(float(transform.a * transform.e)),
            stage_plane_dtype=source.dtypes[0],
        )


def evaluate_elevation_histogram(
    histogram: ElevationHistogram,
    stage_m: float,
    *,
    source_resolution_m: float,
    basin_method: str,
) -> RasterScalarResult:
    """Evaluate exact cumulative sums at a configured histogram bin edge."""

    stage = float(stage_m)
    if not math.isfinite(stage):
        raise RasterStorageError("Histogram stage must be finite.")
    floating_index = (stage - histogram.bin_origin_m) / histogram.bin_width_m
    index = round(floating_index)
    if not math.isclose(floating_index, index, rel_tol=0.0, abs_tol=1e-8):
        raise RasterStorageError(
            "Histogram evaluation is exact only at an aligned bin edge; use "
            "direct windowed summation for an arbitrary stage."
        )
    if index < 0 or index > histogram.counts.size:
        raise RasterStorageError("Stage is outside the histogram support.")
    plane_dtype = _floating_stage_plane_dtype(histogram.stage_plane_dtype)
    calculation_stage = _effective_stage_plane(stage, plane_dtype)
    expected_effective_edge = _effective_stage_plane(
        histogram.bin_origin_m + index * histogram.bin_width_m, plane_dtype
    )
    if calculation_stage != expected_effective_edge:
        raise RasterStorageError(
            "Histogram stage does not resolve to its configured effective bin edge."
        )
    flooded_count = histogram.origin_count + int(
        histogram.counts[:index].sum(dtype=np.int64)
    )
    bed_sum = histogram.origin_elevation_sum_m + float(
        histogram.elevation_sums_m[:index].sum(dtype=np.float64)
    )
    area = float(np.float64(flooded_count) * histogram.cell_area_m2)
    storage = float(
        (
            np.float64(calculation_stage) * np.float64(flooded_count)
            - np.float64(bed_sum)
        )
        * histogram.cell_area_m2
    )
    return RasterScalarResult(
        stage_m_navd88=stage,
        area_m2=area,
        storage_m3=storage,
        flooded_cell_count=flooded_count,
        valid_cell_count=histogram.valid_cell_count,
        cell_area_m2=histogram.cell_area_m2,
        calculation_method="cumulative_aligned_inclusive_histogram_float64",
        basin_method=basin_method,
        source_resolution_m=float(source_resolution_m),
        working_resolution_m=float(source_resolution_m),
        processing_seconds=histogram.processing_seconds,
        peak_memory_estimate_bytes=histogram.peak_memory_estimate_bytes,
        calculation_stage_m_navd88=calculation_stage,
        stage_plane_dtype=plane_dtype.name,
        warning_flags=_stage_plane_warnings(stage, calculation_stage, plane_dtype),
    )


def average_aggregate(
    elevation_m: np.ndarray,
    valid_mask: np.ndarray,
    *,
    row_factor: int,
    column_factor: int,
    minimum_valid_fraction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate continuous elevation by valid-cell arithmetic mean.

    Edge cells are cropped rather than padded with synthetic elevations.  The
    returned coverage fraction makes that behavior explicit to callers.
    """

    elevation, valid, _ = _array_inputs(elevation_m, valid_mask, None)
    if row_factor <= 0 or column_factor <= 0:
        raise RasterStorageError("Aggregation factors must be positive integers.")
    if not 0 < minimum_valid_fraction <= 1:
        raise RasterStorageError("minimum_valid_fraction must be in (0, 1].")
    rows = elevation.shape[0] // row_factor
    columns = elevation.shape[1] // column_factor
    if rows == 0 or columns == 0:
        raise RasterStorageError("Aggregation factors exceed the source grid.")
    cropped_elevation = elevation[: rows * row_factor, : columns * column_factor]
    cropped_valid = valid[: rows * row_factor, : columns * column_factor]
    shape = (rows, row_factor, columns, column_factor)
    values = cropped_elevation.reshape(shape)
    weights = cropped_valid.reshape(shape)
    counts = weights.sum(axis=(1, 3), dtype=np.int64)
    total = np.where(weights, values, 0.0).sum(axis=(1, 3), dtype=np.float64)
    coverage = counts.astype(np.float64) / float(row_factor * column_factor)
    output_valid = coverage >= minimum_valid_fraction
    aggregated = np.full((rows, columns), np.nan, dtype=np.float64)
    nonzero = counts > 0
    aggregated[nonzero] = total[nonzero] / counts[nonzero]
    return aggregated, output_valid, coverage


@dataclass(frozen=True, slots=True)
class OfficialEavPoint:
    validation_stage_id: str
    stage_m_navd88: float
    official_area_m2: float
    official_storage_m3: float
    stage_date: date | None = None


@dataclass(frozen=True, slots=True)
class EavValidationRecord:
    validation_stage_id: str
    stage_m_navd88: float
    stage_ft_navd88: float
    stage_date: str | None
    official_area_m2: float
    raster_area_m2: float
    absolute_area_error_m2: float
    relative_area_error: float | None
    official_storage_m3: float
    raster_storage_m3: float
    absolute_storage_error_m3: float
    relative_storage_error: float | None
    basin_method: str
    source_resolution: float
    working_resolution: float
    calculation_method: str
    validation_status: str
    calculation_stage_m_navd88: float | None = None
    stage_plane_dtype: str | None = None
    warning_flags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["warning_flags"] = ";".join(self.warning_flags)
        return result


def _signed_relative_error(calculated: float, official: float) -> float | None:
    if official == 0:
        return 0.0 if calculated == 0 else None
    return (calculated - official) / abs(official)


def build_eav_validation_records(
    official_points: Iterable[OfficialEavPoint],
    raster_results: Mapping[str, RasterScalarResult],
    *,
    validation_status: str = "calculated",
) -> tuple[EavValidationRecord, ...]:
    """Join official stages to measured raster sums without tuning either."""

    records: list[EavValidationRecord] = []
    seen_stage_ids: set[str] = set()
    for point in official_points:
        if point.validation_stage_id in seen_stage_ids:
            raise RasterStorageError(
                f"Duplicate official stage ID {point.validation_stage_id!r}."
            )
        seen_stage_ids.add(point.validation_stage_id)
        raster = raster_results.get(point.validation_stage_id)
        if raster is None:
            raise RasterStorageError(
                f"Missing raster result for {point.validation_stage_id!r}."
            )
        if not math.isclose(
            point.stage_m_navd88,
            raster.stage_m_navd88,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise RasterStorageError(
                f"Stage mismatch for {point.validation_stage_id!r}."
            )
        area_error = raster.area_m2 - point.official_area_m2
        storage_error = raster.storage_m3 - point.official_storage_m3
        records.append(
            EavValidationRecord(
                validation_stage_id=point.validation_stage_id,
                stage_m_navd88=point.stage_m_navd88,
                stage_ft_navd88=point.stage_m_navd88 / 0.3048,
                stage_date=(
                    point.stage_date.isoformat()
                    if point.stage_date is not None
                    else None
                ),
                official_area_m2=point.official_area_m2,
                raster_area_m2=raster.area_m2,
                absolute_area_error_m2=abs(area_error),
                relative_area_error=_signed_relative_error(
                    raster.area_m2, point.official_area_m2
                ),
                official_storage_m3=point.official_storage_m3,
                raster_storage_m3=raster.storage_m3,
                absolute_storage_error_m3=abs(storage_error),
                relative_storage_error=_signed_relative_error(
                    raster.storage_m3, point.official_storage_m3
                ),
                basin_method=raster.basin_method,
                source_resolution=raster.source_resolution_m,
                working_resolution=raster.working_resolution_m,
                calculation_method=raster.calculation_method,
                validation_status=validation_status,
                calculation_stage_m_navd88=raster.calculation_stage_m_navd88,
                stage_plane_dtype=raster.stage_plane_dtype,
                warning_flags=raster.warning_flags,
            )
        )
    return tuple(records)


@dataclass(frozen=True, slots=True)
class EavValidationSummary:
    median_absolute_relative_area_error: float | None
    maximum_absolute_relative_area_error: float | None
    mean_signed_relative_area_error: float | None
    median_absolute_relative_storage_error: float | None
    maximum_absolute_relative_storage_error: float | None
    mean_signed_relative_storage_error: float | None
    resolution_sensitivity: str
    stage_dependence: str
    systematic_bias_diagnostics: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _error_summary(values: Iterable[float | None]) -> tuple[float | None, ...]:
    finite = np.asarray(
        [value for value in values if value is not None and math.isfinite(value)],
        dtype=np.float64,
    )
    if not finite.size:
        return None, None, None
    return (
        float(np.median(np.abs(finite))),
        float(np.max(np.abs(finite))),
        float(np.mean(finite, dtype=np.float64)),
    )


def summarize_eav_validation(
    records: Sequence[EavValidationRecord],
    *,
    resolution_sensitivity: str = "not_evaluated",
) -> EavValidationSummary:
    """Summarize magnitude, sign, and stage dependence of E-A-V residuals."""

    if not records:
        raise RasterStorageError("At least one E-A-V validation record is required.")
    area = _error_summary(record.relative_area_error for record in records)
    storage = _error_summary(record.relative_storage_error for record in records)
    stages = np.asarray([record.stage_m_navd88 for record in records], dtype=np.float64)
    available_area = np.asarray(
        [
            np.nan if record.relative_area_error is None else record.relative_area_error
            for record in records
        ],
        dtype=np.float64,
    )
    available_storage = np.asarray(
        [
            np.nan
            if record.relative_storage_error is None
            else record.relative_storage_error
            for record in records
        ],
        dtype=np.float64,
    )

    def slope_label(values: np.ndarray) -> str:
        selected = np.isfinite(values)
        if np.count_nonzero(selected) < 2 or np.ptp(stages[selected]) == 0:
            return "insufficient_stages"
        slope = float(np.polyfit(stages[selected], values[selected], 1)[0])
        return f"signed_relative_error_slope_per_m={slope:.12g}"

    signs = [
        value
        for value in (*available_area.tolist(), *available_storage.tolist())
        if math.isfinite(value)
    ]
    if not signs:
        bias = "not_available"
    elif all(value >= 0 for value in signs):
        bias = "nonnegative_bias"
    elif all(value <= 0 for value in signs):
        bias = "nonpositive_bias"
    else:
        bias = "mixed_signed_residuals"
    return EavValidationSummary(
        median_absolute_relative_area_error=area[0],
        maximum_absolute_relative_area_error=area[1],
        mean_signed_relative_area_error=area[2],
        median_absolute_relative_storage_error=storage[0],
        maximum_absolute_relative_storage_error=storage[1],
        mean_signed_relative_storage_error=storage[2],
        resolution_sensitivity=resolution_sensitivity,
        stage_dependence=f"area:{slope_label(available_area)};storage:{slope_label(available_storage)}",
        systematic_bias_diagnostics=bias,
    )


@dataclass(frozen=True, slots=True)
class ResolutionStageMetric:
    stage_id: str
    stage_m_navd88: float
    area_relative_error_vs_source: float | None
    storage_relative_error_vs_source: float | None
    area_relative_error_vs_next_finer: float | None
    storage_relative_error_vs_next_finer: float | None
    area_relative_error_vs_official: float | None
    storage_relative_error_vs_official: float | None
    shoreline_displacement_proxy_m: float | None = None


@dataclass(frozen=True, slots=True)
class ResolutionBenchmark:
    resolution_m: float
    width: int
    height: int
    cell_count: int
    file_size_bytes: int | None
    processing_seconds: float
    peak_memory_estimate_bytes: int
    resampling_method: str
    grid_role: str
    stage_metrics: tuple[ResolutionStageMetric, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0:
            raise RasterStorageError("Benchmark resolution must be positive.")
        if self.width <= 0 or self.height <= 0:
            raise RasterStorageError("Benchmark dimensions must be positive.")
        if self.grid_role != "quantitative_candidate":
            raise RasterStorageError(
                "Authoritative resolution benchmarks cannot use the rendering grid."
            )
        if self.cell_count != self.width * self.height:
            raise RasterStorageError("Benchmark cell count is inconsistent.")
        if not self.stage_metrics:
            raise RasterStorageError(
                "A resolution benchmark requires at least one validation stage."
            )
        stage_ids = [metric.stage_id for metric in self.stage_metrics]
        if len(set(stage_ids)) != len(stage_ids):
            raise RasterStorageError("Benchmark validation stage IDs must be unique.")

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["stage_metrics"] = [asdict(item) for item in self.stage_metrics]
        return result


def build_resolution_benchmark(
    *,
    resolution_m: float,
    width: int,
    height: int,
    candidate_results: Mapping[str, RasterScalarResult],
    source_results: Mapping[str, RasterScalarResult],
    official_points: Mapping[str, OfficialEavPoint],
    next_finer_results: Mapping[str, RasterScalarResult] | None = None,
    processing_seconds: float,
    peak_memory_estimate_bytes: int,
    resampling_method: str = "average",
    file_size_bytes: int | None = None,
    shoreline_displacement_by_stage_m: Mapping[str, float | None] | None = None,
) -> ResolutionBenchmark:
    """Construct deterministic source/official errors for one candidate grid."""

    if set(candidate_results) != set(source_results):
        raise RasterStorageError("Candidate and source stages must match exactly.")
    finer_results = source_results if next_finer_results is None else next_finer_results
    if set(candidate_results) != set(finer_results):
        raise RasterStorageError("Candidate and next-finer stages must match exactly.")
    metrics: list[ResolutionStageMetric] = []
    for stage_id in sorted(candidate_results):
        candidate = candidate_results[stage_id]
        source = source_results[stage_id]
        finer = finer_results[stage_id]
        official = official_points.get(stage_id)
        if official is None:
            raise RasterStorageError(f"Missing official E-A-V point for {stage_id!r}.")
        if official.validation_stage_id != stage_id:
            raise RasterStorageError(
                f"Official E-A-V identity mismatch for {stage_id!r}."
            )
        stage_values = (
            candidate.stage_m_navd88,
            source.stage_m_navd88,
            finer.stage_m_navd88,
            official.stage_m_navd88,
        )
        if any(
            not math.isclose(stage_values[0], value, rel_tol=0.0, abs_tol=1e-8)
            for value in stage_values[1:]
        ):
            raise RasterStorageError(f"Stage mismatch in benchmark {stage_id!r}.")
        metrics.append(
            ResolutionStageMetric(
                stage_id=stage_id,
                stage_m_navd88=candidate.stage_m_navd88,
                area_relative_error_vs_source=_signed_relative_error(
                    candidate.area_m2, source.area_m2
                ),
                storage_relative_error_vs_source=_signed_relative_error(
                    candidate.storage_m3, source.storage_m3
                ),
                area_relative_error_vs_next_finer=_signed_relative_error(
                    candidate.area_m2, finer.area_m2
                ),
                storage_relative_error_vs_next_finer=_signed_relative_error(
                    candidate.storage_m3, finer.storage_m3
                ),
                area_relative_error_vs_official=_signed_relative_error(
                    candidate.area_m2, official.official_area_m2
                ),
                storage_relative_error_vs_official=_signed_relative_error(
                    candidate.storage_m3, official.official_storage_m3
                ),
                shoreline_displacement_proxy_m=(
                    (shoreline_displacement_by_stage_m or {}).get(stage_id)
                ),
            )
        )
    return ResolutionBenchmark(
        resolution_m=float(resolution_m),
        width=int(width),
        height=int(height),
        cell_count=int(width) * int(height),
        file_size_bytes=file_size_bytes,
        processing_seconds=float(processing_seconds),
        peak_memory_estimate_bytes=int(peak_memory_estimate_bytes),
        resampling_method=resampling_method,
        grid_role="quantitative_candidate",
        stage_metrics=tuple(metrics),
    )


def _maximum_absolute(values: Iterable[float | None]) -> float:
    available = [
        abs(value) for value in values if value is not None and math.isfinite(value)
    ]
    return max(available, default=0.0)


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    """Evidence-backed numerical decision controlling all polished rendering."""

    evidence_used: tuple[str, ...]
    selected_spatial_resolution_m: float | None
    source_max_area_relative_error: float
    source_max_storage_relative_error: float
    working_max_area_relative_error_vs_source: float | None
    working_max_storage_relative_error_vs_source: float | None
    accepted_source_area_relative_error: float
    accepted_source_storage_relative_error: float
    accepted_working_area_relative_error: float
    accepted_working_storage_relative_error: float
    source_resolution_residuals: str
    resolution_convergence_behavior: str
    accepted_error_envelope: str
    scientific_rationale: str
    official_uncertainty_supports_formal_validation: bool
    status: str
    stagewise_source_gate_passed: bool = True
    resolution_convergence_gate_passed: bool = True
    selected_basin_method_gate_passed: bool = True
    source_method_consistency_gate_passed: bool | None = None
    source_exact_numeric_gate_passed: bool | None = None
    source_storage_numeric_precision_gate_passed: bool | None = None
    source_numeric_area_precision_exception_count: int | None = None
    source_numeric_area_precision_exception_stage_ids: tuple[str, ...] = ()
    source_method_max_area_equivalent_radial_displacement_m: float | None = None
    native_source_cell_diagonal_m: float | None = None
    source_method_consistency_rule: str = ""
    source_method_consistency_warning: str = ""
    source_method_consistency_status: str = "not_assessed"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_numeric_area_precision_exception_stage_ids",
            tuple(self.source_numeric_area_precision_exception_stage_ids),
        )
        if self.status not in SUPPORTED_GATE_STATUSES:
            raise ValidationPolicyError(f"Invalid validation status {self.status!r}.")
        if not self.evidence_used or not self.scientific_rationale.strip():
            raise ValidationPolicyError(
                "Validation policy needs evidence and rationale."
            )
        if self.status != "failed" and self.selected_spatial_resolution_m is None:
            raise ValidationPolicyError(
                "A passing policy requires a selected quantitative resolution."
            )
        if self.status != "failed" and not all(
            (
                self.stagewise_source_gate_passed,
                self.resolution_convergence_gate_passed,
                self.selected_basin_method_gate_passed,
            )
        ):
            raise ValidationPolicyError(
                "A passing policy cannot retain a failed scientific gate."
            )
        if self.source_method_consistency_status not in (
            SUPPORTED_SOURCE_METHOD_STATUSES
        ):
            raise ValidationPolicyError(
                "Invalid source-method consistency status "
                f"{self.source_method_consistency_status!r}."
            )
        source_method_values = (
            self.source_method_consistency_gate_passed,
            self.source_exact_numeric_gate_passed,
            self.source_storage_numeric_precision_gate_passed,
            self.source_numeric_area_precision_exception_count,
            self.source_method_max_area_equivalent_radial_displacement_m,
            self.native_source_cell_diagonal_m,
        )
        source_method_supplied = any(
            value is not None for value in source_method_values
        )
        source_method_text_supplied = bool(
            self.source_method_consistency_rule.strip()
            or self.source_method_consistency_warning.strip()
            or self.source_method_consistency_status != "not_assessed"
            or self.source_numeric_area_precision_exception_stage_ids
        )
        if not source_method_supplied and not source_method_text_supplied:
            return
        if any(value is None for value in source_method_values):
            raise ValidationPolicyError(
                "Source-method evidence must supply every numeric and Boolean field."
            )
        for name in (
            "source_method_consistency_gate_passed",
            "source_exact_numeric_gate_passed",
            "source_storage_numeric_precision_gate_passed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValidationPolicyError(f"{name} must be Boolean.")
        count = self.source_numeric_area_precision_exception_count
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValidationPolicyError(
                "source_numeric_area_precision_exception_count must be a "
                "non-negative integer."
            )
        stage_ids = self.source_numeric_area_precision_exception_stage_ids
        if len(stage_ids) != count or len(set(stage_ids)) != len(stage_ids):
            raise ValidationPolicyError(
                "Source numeric area exception count and unique stage IDs disagree."
            )
        if any(not isinstance(value, str) or not value.strip() for value in stage_ids):
            raise ValidationPolicyError(
                "Source numeric area exception stage IDs must be non-empty strings."
            )
        maximum_displacement = (
            self.source_method_max_area_equivalent_radial_displacement_m
        )
        native_diagonal = self.native_source_cell_diagonal_m
        if (
            not isinstance(maximum_displacement, (int, float))
            or isinstance(maximum_displacement, bool)
            or not math.isfinite(maximum_displacement)
            or maximum_displacement < 0
        ):
            raise ValidationPolicyError(
                "Source-method maximum radial displacement must be finite and "
                "non-negative."
            )
        if (
            not isinstance(native_diagonal, (int, float))
            or isinstance(native_diagonal, bool)
            or not math.isfinite(native_diagonal)
            or native_diagonal <= 0
        ):
            raise ValidationPolicyError(
                "Native source-cell diagonal must be finite and positive."
            )
        if not self.source_method_consistency_rule.strip():
            raise ValidationPolicyError("Source-method evidence requires its rule.")
        if not self.source_method_consistency_warning.strip():
            raise ValidationPolicyError("Source-method evidence requires its warning.")
        if self.source_method_consistency_status == "not_assessed":
            raise ValidationPolicyError(
                "Complete source-method evidence cannot have status 'not_assessed'."
            )
        expected_exact = bool(
            self.source_storage_numeric_precision_gate_passed and count == 0
        )
        if self.source_exact_numeric_gate_passed != expected_exact:
            raise ValidationPolicyError(
                "Exact numeric status disagrees with storage precision and area "
                "exception evidence."
            )
        expected_method = bool(
            self.source_storage_numeric_precision_gate_passed
            and maximum_displacement <= native_diagonal
        )
        if self.source_method_consistency_gate_passed != expected_method:
            raise ValidationPolicyError(
                "Source-method gate disagrees with storage precision and native-grid "
                "radial-displacement evidence."
            )
        if (
            self.source_method_consistency_gate_passed
            != self.stagewise_source_gate_passed
        ):
            raise ValidationPolicyError(
                "Source-method and stagewise source gate decisions disagree."
            )
        expected_source_status = (
            "source_method_inconsistent"
            if not self.source_method_consistency_gate_passed
            else (
                "source_method_consistent"
                if self.source_exact_numeric_gate_passed
                else "source_method_consistent_with_numeric_area_warning"
            )
        )
        if self.source_method_consistency_status != expected_source_status:
            raise ValidationPolicyError(
                "Source-method status disagrees with its retained gate evidence."
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def policy_sha256(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_validation_policy(
    source_records: Sequence[EavValidationRecord],
    selected_benchmark: ResolutionBenchmark | None,
    *,
    accepted_source_area_relative_error: float,
    accepted_source_storage_relative_error: float,
    accepted_working_area_relative_error: float,
    accepted_working_storage_relative_error: float,
    evidence_used: Sequence[str],
    scientific_rationale: str,
    official_uncertainty_supports_formal_validation: bool,
    resolution_convergence_behavior: str,
    stagewise_source_gate_passed: bool = True,
    resolution_convergence_gate_passed: bool = True,
    selected_basin_method_gate_passed: bool = True,
    source_method_consistency_gate_passed: bool | None = None,
    source_exact_numeric_gate_passed: bool | None = None,
    source_storage_numeric_precision_gate_passed: bool | None = None,
    source_numeric_area_precision_exception_count: int | None = None,
    source_numeric_area_precision_exception_stage_ids: Sequence[str] = (),
    source_method_max_area_equivalent_radial_displacement_m: float | None = None,
    native_source_cell_diagonal_m: float | None = None,
    source_method_consistency_rule: str = "",
    source_method_consistency_warning: str = "",
    source_method_consistency_status: str = "not_assessed",
) -> ValidationPolicy:
    """Apply an explicit, caller-justified two-level numerical gate.

    Error envelopes are arguments because the raster engine must not conceal an
    unexplained universal tolerance.  Callers must persist both the evidence
    and scientific rationale alongside the numeric envelope.
    """

    if not source_records:
        raise ValidationPolicyError(
            "Source-resolution validation records are required."
        )
    source_stage_ids = [record.validation_stage_id for record in source_records]
    if len(set(source_stage_ids)) != len(source_stage_ids):
        raise ValidationPolicyError("Source validation stage IDs must be unique.")
    if any(
        record.relative_area_error is None
        or not math.isfinite(record.relative_area_error)
        or record.relative_storage_error is None
        or not math.isfinite(record.relative_storage_error)
        for record in source_records
    ):
        raise ValidationPolicyError(
            "Every source stage needs finite relative area and storage errors."
        )
    thresholds = (
        accepted_source_area_relative_error,
        accepted_source_storage_relative_error,
        accepted_working_area_relative_error,
        accepted_working_storage_relative_error,
    )
    if any(not math.isfinite(value) or value < 0 for value in thresholds):
        raise ValidationPolicyError("Accepted relative errors must be non-negative.")
    source_area = _maximum_absolute(
        record.relative_area_error for record in source_records
    )
    source_storage = _maximum_absolute(
        record.relative_storage_error for record in source_records
    )
    working_area: float | None = None
    working_storage: float | None = None
    working_passes = selected_benchmark is not None
    if selected_benchmark is not None:
        benchmark_stage_ids = [
            metric.stage_id for metric in selected_benchmark.stage_metrics
        ]
        if set(benchmark_stage_ids) != set(source_stage_ids):
            raise ValidationPolicyError(
                "Working-grid and source validation stages must match exactly."
            )
        if any(
            metric.area_relative_error_vs_source is None
            or not math.isfinite(metric.area_relative_error_vs_source)
            or metric.storage_relative_error_vs_source is None
            or not math.isfinite(metric.storage_relative_error_vs_source)
            for metric in selected_benchmark.stage_metrics
        ):
            raise ValidationPolicyError(
                "Every working-grid stage needs finite source-relative errors."
            )
        working_area = _maximum_absolute(
            metric.area_relative_error_vs_source
            for metric in selected_benchmark.stage_metrics
        )
        working_storage = _maximum_absolute(
            metric.storage_relative_error_vs_source
            for metric in selected_benchmark.stage_metrics
        )
        working_passes = (
            working_area <= accepted_working_area_relative_error
            and working_storage <= accepted_working_storage_relative_error
        )
    source_passes = (
        source_area <= accepted_source_area_relative_error
        and source_storage <= accepted_source_storage_relative_error
    )
    if (
        source_passes
        and working_passes
        and stagewise_source_gate_passed
        and resolution_convergence_gate_passed
        and selected_basin_method_gate_passed
    ):
        status = (
            "validated"
            if official_uncertainty_supports_formal_validation
            else "numerically_consistent"
        )
    else:
        status = "failed"
    envelope = (
        f"source area<={accepted_source_area_relative_error:g}, "
        f"source storage<={accepted_source_storage_relative_error:g}, "
        f"working area<={accepted_working_area_relative_error:g}, "
        f"working storage<={accepted_working_storage_relative_error:g}"
    )
    return ValidationPolicy(
        evidence_used=tuple(str(item) for item in evidence_used),
        selected_spatial_resolution_m=(
            selected_benchmark.resolution_m if selected_benchmark is not None else None
        ),
        source_max_area_relative_error=source_area,
        source_max_storage_relative_error=source_storage,
        working_max_area_relative_error_vs_source=working_area,
        working_max_storage_relative_error_vs_source=working_storage,
        accepted_source_area_relative_error=accepted_source_area_relative_error,
        accepted_source_storage_relative_error=accepted_source_storage_relative_error,
        accepted_working_area_relative_error=accepted_working_area_relative_error,
        accepted_working_storage_relative_error=accepted_working_storage_relative_error,
        source_resolution_residuals=(
            f"max_abs_area={source_area:.12g};max_abs_storage={source_storage:.12g}"
        ),
        resolution_convergence_behavior=resolution_convergence_behavior,
        accepted_error_envelope=envelope,
        scientific_rationale=scientific_rationale,
        official_uncertainty_supports_formal_validation=(
            official_uncertainty_supports_formal_validation
        ),
        status=status,
        stagewise_source_gate_passed=stagewise_source_gate_passed,
        resolution_convergence_gate_passed=resolution_convergence_gate_passed,
        selected_basin_method_gate_passed=selected_basin_method_gate_passed,
        source_method_consistency_gate_passed=(source_method_consistency_gate_passed),
        source_exact_numeric_gate_passed=source_exact_numeric_gate_passed,
        source_storage_numeric_precision_gate_passed=(
            source_storage_numeric_precision_gate_passed
        ),
        source_numeric_area_precision_exception_count=(
            source_numeric_area_precision_exception_count
        ),
        source_numeric_area_precision_exception_stage_ids=tuple(
            str(value) for value in source_numeric_area_precision_exception_stage_ids
        ),
        source_method_max_area_equivalent_radial_displacement_m=(
            source_method_max_area_equivalent_radial_displacement_m
        ),
        native_source_cell_diagonal_m=native_source_cell_diagonal_m,
        source_method_consistency_rule=source_method_consistency_rule,
        source_method_consistency_warning=source_method_consistency_warning,
        source_method_consistency_status=source_method_consistency_status,
    )


@dataclass(frozen=True, slots=True)
class RenderGateResult:
    allowed: bool
    validation_status: str
    reason: str
    validation_policy_sha256: str
    source_method_consistency_status: str = "not_assessed"
    warning: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_render_gate(policy: ValidationPolicy) -> RenderGateResult:
    """Return the hard render decision; failed numerical models never render."""

    allowed = policy.status in {"validated", "numerically_consistent"}
    reason = (
        "Numerical source and working-grid gates passed."
        if allowed
        else "Numerical validation failed; polished 3D, animation, and HTML are blocked."
    )
    warning = policy.source_method_consistency_warning.strip() or None
    if policy.source_method_consistency_status != "not_assessed":
        reason += f" Source-method status: {policy.source_method_consistency_status}."
    if warning is not None:
        reason += f" Warning: {warning}"
    return RenderGateResult(
        allowed=allowed,
        validation_status=policy.status,
        reason=reason,
        validation_policy_sha256=policy.policy_sha256,
        source_method_consistency_status=policy.source_method_consistency_status,
        warning=warning,
    )


__all__ = [
    "ConnectivityError",
    "EavValidationRecord",
    "EavValidationSummary",
    "ElevationHistogram",
    "ElevationWindow",
    "OfficialEavPoint",
    "RasterScalarResult",
    "RasterStorageError",
    "RenderGateResult",
    "ResolutionBenchmark",
    "ResolutionStageMetric",
    "StageRasterResult",
    "ValidationPolicy",
    "ValidationPolicyError",
    "average_aggregate",
    "build_eav_validation_records",
    "build_elevation_histogram",
    "build_raster_elevation_histogram",
    "build_resolution_benchmark",
    "compute_connected_inundation",
    "compute_stage_raster",
    "evaluate_elevation_histogram",
    "evaluate_render_gate",
    "evaluate_validation_policy",
    "integrate_elevation_windows",
    "integrate_raster_stages",
    "integrate_raster_threshold_events",
    "integrate_threshold_event_windows",
    "summarize_eav_validation",
]
