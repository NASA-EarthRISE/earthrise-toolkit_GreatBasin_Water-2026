"""Validation-gated inspection of an authoritative elevation raster.

The functions in this module deliberately do not know about Bear Lake or a
particular archive layout.  Callers must supply the expected source metadata;
the raster is never assigned a guessed CRS, datum, unit, or role.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


class RasterInspectionError(ValueError):
    """Raised when a raster cannot satisfy the quantitative source gate."""


class AmbiguousRasterError(RasterInspectionError):
    """Raised when archive metadata do not identify one unique raster."""


def _rasterio() -> Any:
    """Import rasterio lazily after loading ``pyexpat`` on Windows.

    The project environment currently needs this import order because GDAL's
    DLL search can otherwise shadow the Python expat DLL.  Keeping the import
    lazy also leaves the array-only scientific functions usable without GDAL.
    """

    try:
        import pyexpat  # noqa: F401
        import sqlite3  # noqa: F401

        import rasterio
    except ImportError as exc:  # pragma: no cover - depends on local GDAL build
        raise RasterInspectionError(
            "Raster inspection requires a working rasterio/GDAL installation."
        ) from exc
    return rasterio


def _normalized_datum(value: str) -> str:
    token = "".join(character for character in value.upper() if character.isalnum())
    if (
        "NAVD88" in token
        or "NORTHAMERICANVERTICALDATUM1988" in token
        or "NORTHAMERICANVERTICALDATUMOF1988" in token
    ):
        return "NAVD88"
    return token


def _is_metre_unit(value: str) -> bool:
    token = value.strip().casefold().replace("_", " ")
    return token in {"m", "meter", "meters", "metre", "metres"}


def _tag_value(tags: Mapping[str, str], names: Iterable[str]) -> str | None:
    normalized = {
        "".join(character for character in key.casefold() if character.isalnum()): value
        for key, value in tags.items()
    }
    for name in names:
        key = "".join(character for character in name.casefold() if character.isalnum())
        value = normalized.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


@dataclass(frozen=True, slots=True)
class ArchiveRasterCandidate:
    """One raster-like archive member and the metadata known before extraction."""

    archive_path: str
    uncompressed_size: int
    expected_by_metadata: bool = False
    format_hint: str | None = None
    width: int | None = None
    height: int | None = None
    resolution_m: float | None = None

    def __post_init__(self) -> None:
        path = str(self.archive_path).strip().replace("\\", "/")
        if not path or path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise RasterInspectionError(f"Unsafe archive member path {path!r}.")
        if self.uncompressed_size < 0:
            raise RasterInspectionError("uncompressed_size cannot be negative.")
        object.__setattr__(self, "archive_path", path)


def select_archive_raster_candidate(
    candidates: Iterable[ArchiveRasterCandidate],
    *,
    expected_filename: str | None = None,
    expected_resolution_m: float | None = None,
    resolution_tolerance_m: float = 0.05,
) -> ArchiveRasterCandidate:
    """Select one documented elevation raster or fail rather than guessing.

    Selection applies explicit evidence in order: supported raster format,
    configured filename, official metadata role, and known resolution.  If
    more than one candidate survives, the archive remains ambiguous.
    """

    supported = {".tif", ".tiff", ".img"}
    remaining = [
        item
        for item in candidates
        if PurePosixPath(item.archive_path).suffix.casefold() in supported
    ]
    if not remaining:
        raise RasterInspectionError(
            "The archive contains no supported raster candidate."
        )

    if expected_filename is not None:
        expected = PurePosixPath(expected_filename.replace("\\", "/")).name.casefold()
        exact = [
            item
            for item in remaining
            if PurePosixPath(item.archive_path).name.casefold() == expected
        ]
        if len(exact) != 1:
            raise AmbiguousRasterError(
                f"Expected raster {expected_filename!r} matched {len(exact)} entries."
            )
        remaining = exact
    else:
        documented = [item for item in remaining if item.expected_by_metadata]
        if documented:
            remaining = documented

    if expected_resolution_m is not None and len(remaining) > 1:
        resolution_matches = [
            item
            for item in remaining
            if item.resolution_m is not None
            and math.isclose(
                item.resolution_m,
                expected_resolution_m,
                rel_tol=0.0,
                abs_tol=resolution_tolerance_m,
            )
        ]
        if resolution_matches:
            remaining = resolution_matches

    if len(remaining) != 1:
        names = sorted(item.archive_path for item in remaining)
        raise AmbiguousRasterError(
            "Raster selection remains ambiguous after applying official metadata: "
            + ", ".join(names)
        )
    return remaining[0]


@dataclass(frozen=True, slots=True)
class RasterInspection:
    """Auditable metadata and value-range result for one elevation raster."""

    path: str
    driver: str
    width: int
    height: int
    band_count: int
    data_type: str
    nodata: float | None
    mask_flags: tuple[str, ...]
    bounds: tuple[float, float, float, float]
    transform: tuple[float, float, float, float, float, float]
    pixel_width_m: float
    pixel_height_m: float
    cell_area_m2: float
    block_shape: tuple[int, int]
    compression: str | None
    horizontal_crs_wkt: str
    horizontal_epsg: int | None
    horizontal_units: str
    vertical_datum: str
    vertical_units: str
    scale: float
    offset: float
    minimum_valid_elevation_m: float
    maximum_valid_elevation_m: float
    valid_cell_count: int
    peak_window_memory_bytes: int
    inspection_seconds: float
    tags: tuple[tuple[str, str], ...]
    validation_status: str = "passed"

    @property
    def resolution_m(self) -> float:
        """Return geometric-mean resolution for compact reporting."""

        return math.sqrt(self.cell_area_m2)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extent_overlap_fraction(
    actual: tuple[float, float, float, float],
    expected: tuple[float, float, float, float],
) -> float:
    left = max(actual[0], expected[0])
    bottom = max(actual[1], expected[1])
    right = min(actual[2], expected[2])
    top = min(actual[3], expected[3])
    intersection = max(0.0, right - left) * max(0.0, top - bottom)
    expected_area = (expected[2] - expected[0]) * (expected[3] - expected[1])
    if expected_area <= 0:
        raise RasterInspectionError("Expected bounds must have positive area.")
    return intersection / expected_area


def inspect_elevation_raster(
    path: str | Path,
    *,
    expected_resolution_m: float,
    vertical_datum: str | None = None,
    vertical_units: str | None = None,
    required_vertical_datum: str = "NAVD88",
    resolution_tolerance_m: float = 0.05,
    expected_bounds: tuple[float, float, float, float] | None = None,
    minimum_extent_overlap_fraction: float = 0.9,
) -> RasterInspection:
    """Inspect a source raster and enforce the quantitative topobathy gate.

    Vertical metadata may come from the raster or explicit, already-audited
    official metadata.  Conflicting embedded and supplied values always fail.
    The source file is opened read-only and never modified.
    """

    raster_path = Path(path)
    if not raster_path.is_file():
        raise RasterInspectionError(f"Elevation raster does not exist: {raster_path}")
    if not math.isfinite(expected_resolution_m) or expected_resolution_m <= 0:
        raise RasterInspectionError("expected_resolution_m must be positive.")
    rasterio = _rasterio()
    started = time.perf_counter()
    with rasterio.open(raster_path, "r") as dataset:
        if dataset.count != 1:
            raise RasterInspectionError(
                f"Expected one authoritative elevation band; found {dataset.count}."
            )
        if dataset.crs is None or not dataset.crs.is_projected:
            raise RasterInspectionError("A projected horizontal CRS is required.")
        horizontal_units = str(dataset.crs.linear_units or "")
        if not _is_metre_unit(horizontal_units):
            raise RasterInspectionError(
                f"Horizontal CRS units must be metres, not {horizontal_units!r}."
            )

        transform = dataset.transform
        if not math.isclose(transform.b, 0.0, abs_tol=1e-12) or not math.isclose(
            transform.d, 0.0, abs_tol=1e-12
        ):
            raise RasterInspectionError(
                "Rotated or sheared rasters are unsupported by the Phase 2 grid model."
            )
        pixel_width = abs(float(transform.a))
        pixel_height = abs(float(transform.e))
        if not math.isclose(
            pixel_width,
            expected_resolution_m,
            rel_tol=0.0,
            abs_tol=resolution_tolerance_m,
        ) or not math.isclose(
            pixel_height,
            expected_resolution_m,
            rel_tol=0.0,
            abs_tol=resolution_tolerance_m,
        ):
            raise RasterInspectionError(
                "Raster pixel dimensions do not match the documented source "
                f"resolution: {pixel_width:g} x {pixel_height:g} m."
            )

        tags = {**dataset.tags(), **dataset.tags(1)}
        embedded_datum = _tag_value(
            tags,
            ("vertical_datum", "vert_datum", "vdatum", "verticalreference"),
        )
        embedded_units = _tag_value(
            tags,
            ("vertical_units", "elevation_units", "z_units", "unittype"),
        )
        if (
            embedded_datum
            and vertical_datum
            and _normalized_datum(embedded_datum) != _normalized_datum(vertical_datum)
        ):
            raise RasterInspectionError(
                "Embedded and official-metadata vertical datums conflict."
            )
        if (
            embedded_units
            and vertical_units
            and _is_metre_unit(embedded_units) != _is_metre_unit(vertical_units)
        ):
            raise RasterInspectionError(
                "Embedded and official-metadata vertical units conflict."
            )
        resolved_datum = embedded_datum or vertical_datum
        resolved_units = embedded_units or vertical_units
        if not resolved_datum or (
            _normalized_datum(resolved_datum)
            != _normalized_datum(required_vertical_datum)
        ):
            raise RasterInspectionError(
                f"Vertical datum must be established as {required_vertical_datum}."
            )
        if not resolved_units or not _is_metre_unit(resolved_units):
            raise RasterInspectionError("Vertical elevation units must be metres.")

        scales = tuple(float(item) for item in dataset.scales)
        offsets = tuple(float(item) for item in dataset.offsets)
        scale = scales[0] if scales else 1.0
        offset = offsets[0] if offsets else 0.0
        if not math.isclose(scale, 1.0) or not math.isclose(offset, 0.0):
            raise RasterInspectionError(
                f"Unexplained raster scale/offset ({scale}, {offset}) is not allowed."
            )

        flags = tuple(sorted(flag.name for flag in dataset.mask_flag_enums[0]))
        if dataset.nodata is None and flags == ("all_valid",):
            raise RasterInspectionError(
                "An explicit nodata value or valid-data mask is required."
            )
        bounds = tuple(float(value) for value in dataset.bounds)
        if (
            expected_bounds is not None
            and _extent_overlap_fraction(bounds, expected_bounds)
            < minimum_extent_overlap_fraction
        ):
            raise RasterInspectionError(
                "Raster extent does not cover the expected lake domain."
            )

        minimum = math.inf
        maximum = -math.inf
        valid_count = 0
        peak_memory = 0
        for _, window in dataset.block_windows(1):
            values = dataset.read(1, window=window, masked=True)
            data = values.compressed()
            if data.size:
                finite = data[np.isfinite(data)]
                if finite.size:
                    minimum = min(minimum, float(finite.min()))
                    maximum = max(maximum, float(finite.max()))
                    valid_count += int(finite.size)
            peak_memory = max(
                peak_memory,
                int(values.data.nbytes + np.ma.getmaskarray(values).nbytes),
            )
        if valid_count == 0 or not math.isfinite(minimum) or not math.isfinite(maximum):
            raise RasterInspectionError("Raster contains no finite valid elevations.")

        profile = dataset.profile
        compression = profile.get("compress")
        block_shape = tuple(int(item) for item in dataset.block_shapes[0])
        horizontal_epsg = dataset.crs.to_epsg()
        return RasterInspection(
            path=str(raster_path.resolve()),
            driver=str(dataset.driver),
            width=int(dataset.width),
            height=int(dataset.height),
            band_count=int(dataset.count),
            data_type=str(dataset.dtypes[0]),
            nodata=None if dataset.nodata is None else float(dataset.nodata),
            mask_flags=flags,
            bounds=bounds,
            transform=(
                float(transform.a),
                float(transform.b),
                float(transform.c),
                float(transform.d),
                float(transform.e),
                float(transform.f),
            ),
            pixel_width_m=pixel_width,
            pixel_height_m=pixel_height,
            cell_area_m2=pixel_width * pixel_height,
            block_shape=block_shape,
            compression=None if compression is None else str(compression),
            horizontal_crs_wkt=dataset.crs.to_wkt(),
            horizontal_epsg=horizontal_epsg,
            horizontal_units=horizontal_units,
            vertical_datum=resolved_datum,
            vertical_units=resolved_units,
            scale=scale,
            offset=offset,
            minimum_valid_elevation_m=minimum,
            maximum_valid_elevation_m=maximum,
            valid_cell_count=valid_count,
            peak_window_memory_bytes=peak_memory,
            inspection_seconds=time.perf_counter() - started,
            tags=tuple(sorted((str(key), str(value)) for key, value in tags.items())),
        )


__all__ = [
    "AmbiguousRasterError",
    "ArchiveRasterCandidate",
    "RasterInspection",
    "RasterInspectionError",
    "inspect_elevation_raster",
    "select_archive_raster_candidate",
]
