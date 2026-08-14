"""QGIS-ready Bear Lake Phase 2 raster and vector products.

This module is intentionally downstream of the quantitative raster engine.
Callers must provide the connected inundation mask selected by that engine;
the functions here never substitute a simple ``elevation <= stage`` flood.
All GeoTIFF writes are atomic, tiled, losslessly compressed, and tagged with a
link to the Phase 2 provenance record.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, mapping, shape
from shapely.ops import unary_union

try:
    from shapely import make_valid
except ImportError:  # Shapely 1.8 compatibility
    from shapely.validation import make_valid

from .shoreline import INUNDATION_CHANGE_CODES, inundation_change_codes

CHANGE_CODES = dict(INUNDATION_CHANGE_CODES)
INUNDATION_CODES = {0: "dry", 1: "water", 255: "nodata_or_unsupported"}
QUANTITATIVE_GATE_STATUSES = frozenset({"validated", "numerically_consistent"})
FLOAT_NODATA = -9999.0
BYTE_NODATA = 255
POLYGONIZATION_METHOD = (
    "rasterio_shapes_connectivity8_then_shapely_make_valid_if_needed"
)


class SpatialProductError(RuntimeError):
    """Raised when a quantitative spatial product is unsafe or inconsistent."""


class OptionalVectorDependencyError(SpatialProductError):
    """Raised when GeoPackage output is requested without a vector writer."""


@dataclass(frozen=True, slots=True)
class SpatialState:
    """One observed stage and its already validated connected inundation mask."""

    state_id: str
    stage_m_navd88: float
    stage_ft_navd88: float
    inundation: np.ndarray
    observation_date: str | date | datetime | None = None
    frame_order: int | None = None
    derived_area_m2: float | None = None
    derived_storage_m3: float | None = None
    method: str = "connected_inundation"
    validation_status: str = "numerically_consistent"
    source: str = "USGS observed stage"
    properties: Mapping[str, Any] = field(default_factory=dict)
    calculation_stage_m_navd88: float | None = None

    def __post_init__(self) -> None:
        if not self.state_id.strip():
            raise SpatialProductError("state_id must be non-empty")
        if not np.isfinite(self.stage_m_navd88):
            raise SpatialProductError("stage_m_navd88 must be finite")
        if not np.isfinite(self.stage_ft_navd88):
            raise SpatialProductError("stage_ft_navd88 must be finite")
        if self.calculation_stage_m_navd88 is not None and not np.isfinite(
            self.calculation_stage_m_navd88
        ):
            raise SpatialProductError("calculation_stage_m_navd88 must be finite")
        array = np.asarray(self.inundation)
        if array.ndim != 2 or array.dtype.kind not in "biu":
            raise SpatialProductError("inundation must be a two-dimensional mask")

    @property
    def water_surface_stage_m_navd88(self) -> float:
        """Return the explicit numerical plane used to derive the water mask."""

        if self.calculation_stage_m_navd88 is None:
            return float(self.stage_m_navd88)
        return float(self.calculation_stage_m_navd88)


@dataclass(frozen=True, slots=True)
class VectorFeature:
    """Provider-independent vector feature used before optional GPKG writing."""

    geometry: Any
    properties: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PolygonizationResult:
    """Polygonized water geometry and its quantitative consistency check."""

    geometry: Any
    raster_area_m2: float
    polygon_area_m2: float
    absolute_area_difference_m2: float
    repair_method: str = "none"


@dataclass(frozen=True, slots=True)
class SpatialProductResult:
    """Paths and hashes written by :func:`write_quantitative_spatial_products`."""

    rasters: Mapping[str, Path]
    sha256: Mapping[str, str]
    vector_layers: Mapping[str, tuple[VectorFeature, ...]]
    geopackage: Path | None


def _rasterio_modules() -> tuple[Any, Any, Any]:
    """Import rasterio after pyexpat, required by the pinned Windows runtime."""

    try:
        import pyexpat

        _ = pyexpat
        import rasterio
        from rasterio import features
        from rasterio.enums import Resampling
        from rasterio.env import set_gdal_config
    except (ImportError, OSError) as error:  # pragma: no cover - environment guard
        raise SpatialProductError(
            "rasterio with its GDAL runtime is required for Phase 2 spatial products"
        ) from error
    # The host may carry PROJ_LIB from unrelated GIS software. Prefer the
    # version-matched database bundled with rasterio's GDAL runtime.
    bundled_proj = Path(rasterio.__path__[0]) / "proj_data"
    if (bundled_proj / "proj.db").is_file():
        set_gdal_config("PROJ_LIB", str(bundled_proj))
        set_gdal_config("PROJ_DATA", str(bundled_proj))
    return rasterio, features, Resampling


def _as_iso(value: str | date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _json_tag(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a local output."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_quantitative_gate(status: str) -> None:
    """Reject spatial/render production unless numerical validation passed."""

    if str(status).strip().lower() not in QUANTITATIVE_GATE_STATUSES:
        raise SpatialProductError(
            "Quantitative render gate is closed: validation status must be "
            "'validated' or 'numerically_consistent'."
        )


def _validate_grid(array: np.ndarray, transform: Any, crs: Any) -> None:
    if np.asarray(array).ndim != 2:
        raise SpatialProductError("Raster arrays must be two-dimensional")
    if abs(float(transform.b)) > 1e-12 or abs(float(transform.d)) > 1e-12:
        raise SpatialProductError("Rotated transforms are not supported for products")
    if float(transform.a) <= 0 or float(transform.e) >= 0:
        raise SpatialProductError("Expected a north-up projected raster transform")
    rasterio, _, _ = _rasterio_modules()
    parsed_crs = rasterio.crs.CRS.from_user_input(crs)
    if not parsed_crs.is_projected:
        raise SpatialProductError("Phase 2 outputs require a projected CRS")
    units = str(parsed_crs.linear_units or "").lower()
    if units not in {"metre", "meter", "metres", "meters", "m"}:
        raise SpatialProductError("Phase 2 outputs require metre horizontal units")


def _overview_factors(height: int, width: int) -> list[int]:
    factors: list[int] = []
    factor = 2
    while min(height, width) // factor >= 128:
        factors.append(factor)
        factor *= 2
    return factors


def write_tiled_geotiff(
    path: Path,
    array: np.ndarray,
    *,
    transform: Any,
    crs: Any,
    nodata: float,
    band_description: str,
    tags: Mapping[str, Any],
    dtype: str | np.dtype[Any] | None = None,
    categorical_codes: Mapping[int, str] | None = None,
    build_overviews: bool = True,
) -> Path:
    """Atomically write one lossless, tiled, QGIS-ready GeoTIFF band."""

    data = np.asarray(array)
    _validate_grid(data, transform, crs)
    rasterio, _, Resampling = _rasterio_modules()
    output_dtype = np.dtype(dtype or data.dtype)
    if output_dtype.kind not in "fiu":
        raise SpatialProductError(f"Unsupported GeoTIFF dtype {output_dtype}")
    if not band_description.strip():
        raise SpatialProductError("band_description must be non-empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tif.part", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    predictor = 3 if output_dtype.kind == "f" else 2
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": output_dtype.name,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "DEFLATE",
        "predictor": predictor,
        "zlevel": 6,
        "bigtiff": "IF_SAFER",
    }
    serialized_tags = {
        str(key): _json_tag(value)
        if isinstance(value, (dict, list, tuple))
        else str(value)
        for key, value in tags.items()
        if value is not None
    }
    if categorical_codes is not None:
        serialized_tags["CATEGORY_CODES"] = _json_tag(categorical_codes)
    try:
        with rasterio.open(temporary, "w", **profile) as dataset:
            dataset.write(data.astype(output_dtype, copy=False), 1)
            dataset.set_band_description(1, band_description)
            dataset.update_tags(**serialized_tags)
            factors = _overview_factors(*data.shape) if build_overviews else []
            if factors:
                method = (
                    Resampling.nearest
                    if categorical_codes is not None
                    else Resampling.average
                )
                dataset.build_overviews(factors, method)
                dataset.update_tags(ns="rio_overview", resampling=method.name)
        with rasterio.open(temporary) as check:
            if check.count != 1 or check.width != data.shape[1]:
                raise SpatialProductError("GeoTIFF validation failed after write")
            if check.crs is None or not check.crs.is_projected:
                raise SpatialProductError("GeoTIFF lost its projected CRS")
            if check.nodata != nodata:
                raise SpatialProductError(
                    "GeoTIFF nodata metadata changed during write"
                )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def depth_and_inundation_arrays(
    elevation_m: np.ndarray,
    *,
    connected_inundation: np.ndarray,
    supported_mask: np.ndarray,
    stage_m_navd88: float,
    calculation_stage_m_navd88: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create depth and inundation arrays using the mask's calculation plane.

    ``stage_m_navd88`` retains the nominal observed value.  A separately
    supplied calculation stage allows exact reproduction of a source method
    that evaluates a float32 water-surface plane without mislabelling the
    observation itself.
    """

    elevation = np.asarray(elevation_m, dtype=np.float64)
    inundated = np.asarray(connected_inundation, dtype=bool)
    supported = np.asarray(supported_mask, dtype=bool)
    if elevation.shape != inundated.shape or elevation.shape != supported.shape:
        raise SpatialProductError("Elevation, inundation, and support grids must align")
    nominal_stage = float(stage_m_navd88)
    calculation_stage = (
        nominal_stage
        if calculation_stage_m_navd88 is None
        else float(calculation_stage_m_navd88)
    )
    if not np.isfinite(nominal_stage) or not np.isfinite(calculation_stage):
        raise SpatialProductError("Nominal and calculation stages must be finite")
    if np.any(inundated & ~supported):
        raise SpatialProductError(
            "Connected inundation extends outside supported pixels"
        )
    if np.any(inundated & (~np.isfinite(elevation) | (elevation > calculation_stage))):
        raise SpatialProductError("Connected inundation contains dry or invalid pixels")
    depth = np.full(elevation.shape, FLOAT_NODATA, dtype=np.float32)
    depth[supported] = 0.0
    depth[inundated] = (calculation_stage - elevation[inundated]).astype(np.float32)
    categories = np.full(elevation.shape, BYTE_NODATA, dtype=np.uint8)
    categories[supported] = 0
    categories[inundated] = 1
    return depth, categories


def polygonize_inundation(
    inundation: np.ndarray,
    *,
    transform: Any,
    cell_area_m2: float | None = None,
    relative_tolerance: float = 1e-9,
) -> PolygonizationResult:
    """Polygonize water pixels, preserving holes and verifying exact area."""

    water = np.asarray(inundation, dtype=bool)
    if water.ndim != 2 or not water.any():
        raise SpatialProductError("Cannot polygonize an empty inundation mask")
    _, features, _ = _rasterio_modules()
    geometries = [
        shape(item)
        for item, value in features.shapes(
            water.astype(np.uint8), mask=water, transform=transform, connectivity=8
        )
        if int(value) == 1
    ]
    geometry = unary_union(geometries)
    repair_method = "none"
    if not geometry.is_valid:
        # Eight-neighbour raster connectivity can create point-touching rings
        # that OGC polygon topology represents as separate valid parts.  The
        # repair preserves the exact cell support and is accepted only after
        # the same raster-versus-vector area check below.
        geometry = make_valid(geometry)
        repair_method = "shapely_make_valid_for_8_neighbor_point_touches"
    if geometry.is_empty or not geometry.is_valid:
        raise SpatialProductError("Polygonized inundation geometry is empty or invalid")
    pixel_area = cell_area_m2
    if pixel_area is None:
        pixel_area = abs(float(transform.a * transform.e - transform.b * transform.d))
    raster_area = float(np.count_nonzero(water)) * float(pixel_area)
    polygon_area = float(geometry.area)
    difference = abs(polygon_area - raster_area)
    allowed = max(1e-8, raster_area * relative_tolerance)
    if difference > allowed:
        raise SpatialProductError(
            f"Polygon area differs from raster area by {difference:.6g} m2"
        )
    return PolygonizationResult(
        geometry,
        raster_area,
        polygon_area,
        difference,
        repair_method,
    )


def depth_contour_features(
    depth_m: np.ndarray,
    *,
    transform: Any,
    levels_m: Sequence[float],
    stage_m_navd88: float,
    stage_date: str | date | datetime | None,
    source: str,
    calculation_stage_m_navd88: float | None = None,
) -> tuple[VectorFeature, ...]:
    """Extract projected depth contours for valid levels from a north-up grid."""

    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise SpatialProductError("Depth grid must be two-dimensional")
    if abs(float(transform.b)) > 1e-12 or abs(float(transform.d)) > 1e-12:
        raise SpatialProductError("Depth contouring requires a north-up grid")
    valid = np.isfinite(depth) & (depth != FLOAT_NODATA)
    positive = depth[valid & (depth > 0)]
    if positive.size == 0:
        return ()
    levels = sorted(
        {
            float(level)
            for level in levels_m
            if np.isfinite(level) and 0 < float(level) <= float(positive.max())
        }
    )
    if not levels:
        return ()
    values = np.where(valid, depth, np.nan)
    columns = np.arange(depth.shape[1], dtype=float) + 0.5
    rows = np.arange(depth.shape[0], dtype=float) + 0.5
    x = float(transform.c) + float(transform.a) * columns
    y = float(transform.f) + float(transform.e) * rows
    figure, axis = plt.subplots()
    try:
        contour_set = axis.contour(x, y, values, levels=levels)
        output: list[VectorFeature] = []
        for level, segments in zip(
            contour_set.levels, contour_set.allsegs, strict=True
        ):
            for segment in segments:
                if len(segment) < 2:
                    continue
                line = LineString(segment)
                if line.is_empty or line.length <= 0:
                    continue
                output.append(
                    VectorFeature(
                        geometry=line,
                        properties={
                            "depth_m": float(level),
                            "stage_m_navd88": float(stage_m_navd88),
                            "calculation_stage_m_navd88": float(
                                stage_m_navd88
                                if calculation_stage_m_navd88 is None
                                else calculation_stage_m_navd88
                            ),
                            "stage_date": _as_iso(stage_date),
                            "source": source,
                        },
                    )
                )
        return tuple(output)
    finally:
        plt.close(figure)


def _state_properties(state: SpatialState) -> dict[str, Any]:
    timestamp = _as_iso(state.observation_date)
    properties = {
        "state_id": state.state_id,
        "stage_m_navd88": float(state.stage_m_navd88),
        "calculation_stage_m_navd88": state.water_surface_stage_m_navd88,
        "stage_ft_navd88": float(state.stage_ft_navd88),
        "date": timestamp,
        "start_datetime": timestamp,
        "end_datetime": timestamp,
        "frame_order": state.frame_order,
        "derived_area_m2": state.derived_area_m2,
        "derived_storage_m3": state.derived_storage_m3,
        "method": state.method,
        "validation_status": state.validation_status,
        "source": state.source,
        "polygonization_method": POLYGONIZATION_METHOD,
    }
    properties.update(dict(state.properties))
    return properties


def build_phase2_vector_layers(
    states: Sequence[SpatialState],
    *,
    transform: Any,
    current_state_id: str = "latest",
    year_ago_state_id: str = "year_ago",
    current_depth_m: np.ndarray | None = None,
    contour_levels_m: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 20.0),
) -> dict[str, tuple[VectorFeature, ...]]:
    """Build all required vector layers before optional GeoPackage writing."""

    by_id = {state.state_id: state for state in states}
    if len(by_id) != len(states):
        raise SpatialProductError("Spatial state IDs must be unique")
    if current_state_id not in by_id or year_ago_state_id not in by_id:
        raise SpatialProductError("Latest and year-ago states are required")
    polygons = {
        state.state_id: polygonize_inundation(
            state.inundation, transform=transform
        ).geometry
        for state in states
    }
    current = by_id[current_state_id]
    shoreline_current = (
        VectorFeature(polygons[current_state_id].boundary, _state_properties(current)),
    )
    shorelines_selected = tuple(
        VectorFeature(polygons[state.state_id].boundary, _state_properties(state))
        for state in states
    )
    shoreline_change = tuple(
        VectorFeature(
            polygons[state_id].boundary,
            {**_state_properties(by_id[state_id]), "comparison_role": role},
        )
        for state_id, role in (
            (year_ago_state_id, "year_ago"),
            (current_state_id, "latest"),
        )
    )
    water_surfaces = tuple(
        VectorFeature(polygons[state.state_id], _state_properties(state))
        for state in states
    )
    contours: tuple[VectorFeature, ...] = ()
    if current_depth_m is not None:
        contours = depth_contour_features(
            current_depth_m,
            transform=transform,
            levels_m=contour_levels_m,
            stage_m_navd88=current.stage_m_navd88,
            calculation_stage_m_navd88=current.water_surface_stage_m_navd88,
            stage_date=current.observation_date,
            source=current.source,
        )
    return {
        "shoreline_current": shoreline_current,
        "shorelines_selected": shorelines_selected,
        "shoreline_change_latest_vs_year_ago": shoreline_change,
        "depth_contours_current": contours,
        "water_surface_current": (
            VectorFeature(polygons[current_state_id], _state_properties(current)),
        ),
        "water_surfaces_selected": water_surfaces,
    }


def write_geopackage(
    path: Path,
    layers: Mapping[str, Sequence[VectorFeature]],
    *,
    crs: Any,
    ogr2ogr_executable: Path | None = None,
    runner: Any | None = None,
) -> Path:
    """Atomically write named layers using GeoPandas or safe ``ogr2ogr`` argv.

    ``ogr2ogr`` is invoked without a shell and receives one temporary GeoJSON
    file per layer.  This fallback keeps the main environment independent of
    Fiona/GeoPandas while still producing a standards-based GeoPackage when a
    trusted GDAL executable is available.
    """

    try:
        import geopandas
    except (ImportError, OSError):
        geopandas = None
    path.parent.mkdir(parents=True, exist_ok=True)
    if geopandas is not None:
        temporary = path.with_name(f".{path.stem}.part.gpkg")
        temporary.unlink(missing_ok=True)
        try:
            first = True
            for layer_name, features in layers.items():
                if not features:
                    continue
                frame = geopandas.GeoDataFrame(
                    [dict(feature.properties) for feature in features],
                    geometry=[feature.geometry for feature in features],
                    crs=crs,
                )
                frame.to_file(
                    temporary,
                    layer=layer_name,
                    driver="GPKG",
                    mode="w" if first else "a",
                    index=False,
                )
                first = False
            if first:
                raise SpatialProductError("No non-empty vector layers were available")
            _normalize_geopackage_timestamps(temporary)
            os.replace(temporary, path)
            return path
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    executable = ogr2ogr_executable
    if executable is None:
        discovered = shutil.which("ogr2ogr")
        candidates = [
            Path(discovered) if discovered else None,
            Path(r"C:\Program Files\GDAL\ogr2ogr.exe"),
        ]
        executable = next(
            (
                candidate
                for candidate in candidates
                if candidate and candidate.is_file()
            ),
            None,
        )
    if executable is None or not Path(executable).is_file():
        raise OptionalVectorDependencyError(
            "GeoPackage output requires GeoPandas/Fiona, pyogrio, or a trusted "
            "ogr2ogr executable. All GeoTIFF products remain available."
        )
    run = runner or subprocess.run
    rasterio, _, _ = _rasterio_modules()
    crs_text = rasterio.crs.CRS.from_user_input(crs).to_string()
    with tempfile.TemporaryDirectory(prefix="bear_lake_gpkg_", dir=path.parent) as raw:
        workspace = Path(raw)
        temporary = workspace / path.name
        first = True
        try:
            for layer_name, features in layers.items():
                if not features:
                    continue
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,62}", layer_name) is None:
                    raise SpatialProductError(
                        f"Unsafe or invalid GeoPackage layer name: {layer_name!r}"
                    )
                geojson_path = workspace / f"{layer_name}.geojson"
                feature_collection = {
                    "type": "FeatureCollection",
                    "name": layer_name,
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": mapping(feature.geometry),
                            "properties": {
                                str(key): (
                                    value.item()
                                    if isinstance(value, np.generic)
                                    else value
                                )
                                for key, value in feature.properties.items()
                            },
                        }
                        for feature in features
                    ],
                }
                geojson_path.write_text(
                    json.dumps(feature_collection, default=str), encoding="utf-8"
                )
                command = [
                    str(executable),
                    "-f",
                    "GPKG",
                    str(temporary),
                    str(geojson_path),
                    "-nln",
                    layer_name,
                    "-a_srs",
                    crs_text,
                ]
                if not first:
                    command.extend(["-update", "-append"])
                run(command, check=True, capture_output=True, text=True)
                first = False
            if first:
                raise SpatialProductError("No non-empty vector layers were available")
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise SpatialProductError("ogr2ogr did not create a GeoPackage")
            _normalize_geopackage_timestamps(temporary)
            os.replace(temporary, path)
        except subprocess.CalledProcessError as error:
            raise SpatialProductError(
                f"ogr2ogr failed while creating {path.name}: {error.stderr or error}"
            ) from error
    return path


def _normalize_geopackage_timestamps(path: Path) -> None:
    """Remove writer clock time so offline product hashes are reproducible."""

    fixed = "1980-01-01T00:00:00.000Z"
    try:
        with closing(sqlite3.connect(path)) as database:
            tables = {
                str(row[0])
                for row in database.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "gpkg_contents" not in tables:
                raise SpatialProductError("Vector writer output is not a GeoPackage")
            database.execute("UPDATE gpkg_contents SET last_change = ?", (fixed,))
            if "gpkg_metadata_reference" in tables:
                database.execute(
                    "UPDATE gpkg_metadata_reference SET timestamp = ?", (fixed,)
                )
            database.commit()
    except sqlite3.Error as error:
        raise SpatialProductError(
            f"Could not normalize GeoPackage metadata timestamps: {error}"
        ) from error


def _state_suffix(state_id: str) -> str:
    aliases = {
        "latest": "current",
        "current": "current",
        "recent_low": "recent_low",
        "recent_high": "recent_high",
        "year_ago": "year_ago",
        "historical_median": "historical_median",
    }
    return aliases.get(state_id, state_id.strip().lower().replace(" ", "_"))


def write_quantitative_spatial_products(
    output_dir: Path,
    *,
    elevation_m: np.ndarray,
    basin_mask: np.ndarray,
    coverage_mask: np.ndarray,
    transform: Any,
    crs: Any,
    states: Sequence[SpatialState],
    validation_status: str,
    provenance_path: Path,
    year_ago_state_id: str = "year_ago",
    current_state_id: str = "latest",
    common_tags: Mapping[str, Any] | None = None,
    write_vectors: bool = True,
) -> SpatialProductResult:
    """Write the required quantitative rasters and optional vector package."""

    require_quantitative_gate(validation_status)
    elevation = np.asarray(elevation_m, dtype=np.float32)
    basin = np.asarray(basin_mask, dtype=bool)
    coverage = np.asarray(coverage_mask, dtype=bool)
    if elevation.shape != basin.shape or elevation.shape != coverage.shape:
        raise SpatialProductError("Reference raster grids must align")
    supported = basin & coverage & np.isfinite(elevation)
    by_id = {state.state_id: state for state in states}
    if len(by_id) != len(states):
        raise SpatialProductError("Spatial state IDs must be unique")
    if current_state_id not in by_id or year_ago_state_id not in by_id:
        raise SpatialProductError("Latest and year-ago states are required")
    for state in states:
        if np.asarray(state.inundation).shape != elevation.shape:
            raise SpatialProductError(f"State {state.state_id} grid is misaligned")

    relative_provenance = os.path.relpath(provenance_path, output_dir)
    tags = {
        "PHASE2_SCHEMA_VERSION": "bear-lake-spatial-v1",
        "PROVENANCE_PATH": relative_provenance,
        "VERTICAL_DATUM": "NAVD88",
        "VERTICAL_UNITS": "metre",
        "VALIDATION_STATUS": validation_status,
        **dict(common_tags or {}),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    rasters: dict[str, Path] = {}

    topobathy = np.where(supported, elevation, FLOAT_NODATA).astype(np.float32)
    rasters["topobathy_working"] = write_tiled_geotiff(
        output_dir / "bear_lake_topobathy_working.tif",
        topobathy,
        transform=transform,
        crs=crs,
        nodata=FLOAT_NODATA,
        dtype="float32",
        band_description="Bear Lake working-grid topobathymetric elevation (m NAVD88)",
        tags={**tags, "QUANTITATIVE_ROLE": "working_grid_elevation"},
    )
    basin_data = np.full(elevation.shape, BYTE_NODATA, dtype=np.uint8)
    basin_data[coverage] = basin[coverage].astype(np.uint8)
    rasters["basin_mask"] = write_tiled_geotiff(
        output_dir / "bear_lake_basin_mask.tif",
        basin_data,
        transform=transform,
        crs=crs,
        nodata=BYTE_NODATA,
        dtype="uint8",
        band_description="Validated quantitative processing-domain mask",
        tags={
            **tags,
            "QUANTITATIVE_ROLE": "processing_domain",
            "DOMAIN_LIMITATION": "not_an_independent_official_basin_boundary",
        },
        categorical_codes={
            0: "outside_processing_domain",
            1: "inside_processing_domain",
            255: "unsupported",
        },
    )
    coverage_data = coverage.astype(np.uint8)
    rasters["coverage"] = write_tiled_geotiff(
        output_dir / "bear_lake_bathymetry_coverage.tif",
        coverage_data,
        transform=transform,
        crs=crs,
        nodata=BYTE_NODATA,
        dtype="uint8",
        band_description="Bear Lake source topobathymetry coverage",
        tags={**tags, "QUANTITATIVE_ROLE": "source_coverage"},
        categorical_codes={0: "unsupported", 1: "supported"},
    )

    state_depths: dict[str, np.ndarray] = {}
    for state in states:
        depth, inundation = depth_and_inundation_arrays(
            elevation,
            connected_inundation=state.inundation,
            supported_mask=supported,
            stage_m_navd88=state.stage_m_navd88,
            calculation_stage_m_navd88=state.water_surface_stage_m_navd88,
        )
        suffix = _state_suffix(state.state_id)
        state_tags = {
            **tags,
            "STATE_ID": state.state_id,
            "STAGE_M_NAVD88": state.stage_m_navd88,
            "CALCULATION_STAGE_M_NAVD88": state.water_surface_stage_m_navd88,
            "STAGE_COMPARISON": (
                "bed_elevation_less_than_or_equal_to_calculation_stage"
            ),
            "STAGE_FT_NAVD88": state.stage_ft_navd88,
            "STAGE_DATE": _as_iso(state.observation_date),
            "METHOD": state.method,
        }
        state_depths[state.state_id] = depth
        rasters[f"depth_{suffix}"] = write_tiled_geotiff(
            output_dir / f"bear_lake_depth_{suffix}.tif",
            depth,
            transform=transform,
            crs=crs,
            nodata=FLOAT_NODATA,
            dtype="float32",
            band_description=f"Bear Lake water depth at {state.state_id} stage (m)",
            tags={**state_tags, "QUANTITATIVE_ROLE": "water_depth"},
        )
        rasters[f"inundation_{suffix}"] = write_tiled_geotiff(
            output_dir / f"bear_lake_inundation_{suffix}.tif",
            inundation,
            transform=transform,
            crs=crs,
            nodata=BYTE_NODATA,
            dtype="uint8",
            band_description=f"Bear Lake connected inundation at {state.state_id} stage",
            tags={**state_tags, "QUANTITATIVE_ROLE": "connected_inundation"},
            categorical_codes=INUNDATION_CODES,
        )

    current = by_id[current_state_id]
    prior = by_id[year_ago_state_id]
    depth_change = np.full(elevation.shape, FLOAT_NODATA, dtype=np.float32)
    depth_change[supported] = (
        state_depths[current_state_id][supported]
        - state_depths[year_ago_state_id][supported]
    )
    change_tags = {
        **tags,
        "LATEST_STATE_ID": current.state_id,
        "LATEST_STAGE_DATE": _as_iso(current.observation_date),
        "PRIOR_STATE_ID": prior.state_id,
        "PRIOR_STAGE_DATE": _as_iso(prior.observation_date),
    }
    rasters["depth_change_latest_vs_year_ago"] = write_tiled_geotiff(
        output_dir / "bear_lake_depth_change_latest_vs_year_ago.tif",
        depth_change,
        transform=transform,
        crs=crs,
        nodata=FLOAT_NODATA,
        dtype="float32",
        band_description="Bear Lake depth change: latest minus year-ago (m)",
        tags={**change_tags, "QUANTITATIVE_ROLE": "depth_change"},
    )
    change = inundation_change_codes(
        current.inundation, prior.inundation, supported_mask=supported
    )
    rasters["inundation_change_latest_vs_year_ago"] = write_tiled_geotiff(
        output_dir / "bear_lake_inundation_change_latest_vs_year_ago.tif",
        change,
        transform=transform,
        crs=crs,
        nodata=BYTE_NODATA,
        dtype="uint8",
        band_description="Bear Lake inundation change: latest versus year-ago",
        tags={**change_tags, "QUANTITATIVE_ROLE": "inundation_change"},
        categorical_codes=CHANGE_CODES,
    )

    vector_layers = build_phase2_vector_layers(
        states,
        transform=transform,
        current_state_id=current_state_id,
        year_ago_state_id=year_ago_state_id,
        current_depth_m=state_depths[current_state_id],
    )
    vector_layers = {
        name: tuple(
            VectorFeature(
                feature.geometry,
                {**dict(feature.properties), "provenance_path": relative_provenance},
            )
            for feature in features
        )
        for name, features in vector_layers.items()
    }
    geopackage: Path | None = None
    if write_vectors:
        geopackage = write_geopackage(
            output_dir / "bear_lake_phase2.gpkg", vector_layers, crs=crs
        )
    hashes = {name: sha256_file(path) for name, path in rasters.items()}
    if geopackage is not None:
        hashes["geopackage"] = sha256_file(geopackage)
    return SpatialProductResult(
        rasters=rasters,
        sha256=hashes,
        vector_layers={
            name: tuple(features) for name, features in vector_layers.items()
        },
        geopackage=geopackage,
    )


__all__ = [
    "BYTE_NODATA",
    "CHANGE_CODES",
    "FLOAT_NODATA",
    "INUNDATION_CODES",
    "QUANTITATIVE_GATE_STATUSES",
    "OptionalVectorDependencyError",
    "PolygonizationResult",
    "SpatialProductError",
    "SpatialProductResult",
    "SpatialState",
    "VectorFeature",
    "build_phase2_vector_layers",
    "depth_and_inundation_arrays",
    "depth_contour_features",
    "inundation_change_codes",
    "polygonize_inundation",
    "require_quantitative_gate",
    "sha256_file",
    "write_geopackage",
    "write_quantitative_spatial_products",
    "write_tiled_geotiff",
]
