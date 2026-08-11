"""Bounded Earth Engine sampling and local optimization for quick examples.

This module is an exploratory fast path. It deliberately duplicates the small
Landsat preprocessing subset from ``final_threshold_optimization.py`` because
importing that regression script would authenticate and initialize Earth Engine
at module import. Samples stay small, are transferred with
``ee.data.computeFeatures``, and are never sent to an Earth Engine export task.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

LANDSAT_COLLECTIONS = {
    "L5": "LANDSAT/LT05/C02/T1_L2",
    "L7": "LANDSAT/LE07/C02/T1_L2",
    "L8": "LANDSAT/LC08/C02/T1_L2",
    "L9": "LANDSAT/LC09/C02/T1_L2",
}

LANDSAT_BAND_MAP = {
    "L5": {
        "blue": "SR_B1",
        "green": "SR_B2",
        "red": "SR_B3",
        "nir": "SR_B4",
        "swir1": "SR_B5",
    },
    "L7": {
        "blue": "SR_B1",
        "green": "SR_B2",
        "red": "SR_B3",
        "nir": "SR_B4",
        "swir1": "SR_B5",
    },
    "L8": {
        "blue": "SR_B2",
        "green": "SR_B3",
        "red": "SR_B4",
        "nir": "SR_B5",
        "swir1": "SR_B6",
    },
    "L9": {
        "blue": "SR_B2",
        "green": "SR_B3",
        "red": "SR_B4",
        "nir": "SR_B5",
        "swir1": "SR_B6",
    },
}

LANDSAT_SENSOR_WINDOWS = {
    "L5": (dt.date(1984, 1, 1), dt.date(2012, 5, 5)),
    "L7": (dt.date(1999, 1, 1), dt.date(2022, 4, 6)),
    "L8": (dt.date(2013, 4, 11), dt.date.max),
    "L9": (dt.date(2022, 2, 10), dt.date.max),
}

LANDSAT_SENSORS = ("L5", "L7", "L8", "L9")
JRC_COLLECTION = "JRC/GSW1_4/MonthlyHistory"
DYNAMIC_WORLD_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
OPERA_COLLECTION = "OPERA/DSWX/L3_V1/HLS"
OPERA_WATER_BAND = "WTR_Water_classification"
MS_PER_DAY = 86_400_000
JOIN_MEASURE_PROPERTY = "_match_delta_ms"

REFERENCE_INDEX = {"JRC": "NDWI", "DW": "NDVI", "OPERA": "NDWI"}
SUPPORTED_REFERENCES = tuple(REFERENCE_INDEX)
DYNAMIC_WORLD_VEGETATION = (1, 2, 3, 4, 5)
DYNAMIC_WORLD_NONVEGETATION = (0, 6, 7)

SAMPLE_COLUMNS = (
    "Hylak_id",
    "Lake_name",
    "state",
    "Lake_area",
    "reference",
    "index",
    "index_value",
    "label",
    "acquisition_time",
    "reference_time",
    "match_delta_days",
    "reference_tile_count",
    "sensor",
    "lon",
    "lat",
)

SUMMARY_COLUMNS = (
    "Hylak_id",
    "Lake_name",
    "state",
    "Lake_area",
    "reference",
    "index",
    "date_start",
    "date_end",
    "n_samples",
    "n_train",
    "n_test",
    "n_blocks",
    "positive_fraction",
    "threshold_mean",
    "threshold_std",
    "cv_iou",
    "test_iou",
    "test_f1",
    "test_precision",
    "test_recall",
    "status",
    "skip_reason",
    "sensors",
    "sample_date_start",
    "sample_date_end",
    "buffer_km",
    "samples_per_image",
    "join_window_days",
    "elapsed_seconds",
)


class QuickSamplingError(RuntimeError):
    """Raised when a bounded quick-example sample cannot be produced safely."""


class SampleLimitError(QuickSamplingError):
    """Raised before an unexpectedly large sample can be fully transferred."""


class InsufficientSampleError(ValueError):
    """Raised when a local sample cannot support the requested validation."""


@dataclass(frozen=True)
class LakeSpec:
    """Local metadata needed to construct and report one HydroLAKES example."""

    hylak_id: int
    lake_name: str
    state: str
    lake_area: float
    centroid_lon: float
    centroid_lat: float


@dataclass(frozen=True)
class QuickSamplingSettings:
    """Validated controls for the explicitly exploratory fast mode."""

    buffer_km: float = 5.0
    samples_per_image: int = 20
    page_size: int = 1000
    max_rows_per_combination: int = 10_000
    analysis_end: dt.date = dt.date(2026, 8, 9)
    jrc_start: dt.date = dt.date(2017, 1, 1)
    jrc_end: dt.date = dt.date(2021, 12, 31)
    dw_start: dt.date = dt.date(2017, 1, 1)
    opera_start: dt.date = dt.date(2023, 4, 4)
    jrc_window_days: int = 31
    dw_window_days: int = 3
    opera_window_days: int = 5
    cloud_cover_max: float = 20.0
    block_size_m: int = 1000
    train_fraction: float = 0.80
    random_seed: int = 42
    cv_splits: int = 5
    grid_steps: int = 150
    min_samples: int = 40
    min_blocks: int = 5
    min_split_samples: int = 10

    def __post_init__(self) -> None:
        positive = {
            "buffer_km": self.buffer_km,
            "samples_per_image": self.samples_per_image,
            "page_size": self.page_size,
            "max_rows_per_combination": self.max_rows_per_combination,
            "jrc_window_days": self.jrc_window_days,
            "dw_window_days": self.dw_window_days,
            "opera_window_days": self.opera_window_days,
            "cloud_cover_max": self.cloud_cover_max,
            "block_size_m": self.block_size_m,
            "cv_splits": self.cv_splits,
            "grid_steps": self.grid_steps,
            "min_samples": self.min_samples,
            "min_blocks": self.min_blocks,
            "min_split_samples": self.min_split_samples,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(
                "Quick-example settings must be positive: " + ", ".join(invalid)
            )
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between zero and one.")
        if self.cv_splits < 2:
            raise ValueError("cv_splits must be at least two.")
        if self.jrc_start > self.jrc_end:
            raise ValueError("The configured JRC window is empty.")

    def with_overrides(self, **overrides: Any) -> QuickSamplingSettings:
        """Return a validated copy containing CLI or notebook overrides."""

        values = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **values)

    def reference_window(self, reference: str) -> tuple[dt.date, dt.date]:
        """Return the inclusive fast-mode window for a reference."""

        normalized = normalize_reference(reference)
        if normalized == "JRC":
            window = self.jrc_start, min(self.jrc_end, self.analysis_end)
            label = "JRC"
        elif normalized == "DW":
            window = self.dw_start, self.analysis_end
            label = "Dynamic World"
        else:
            window = self.opera_start, self.analysis_end
            label = "OPERA"
        if window[0] > window[1]:
            raise ValueError(f"The configured {label} window is empty.")
        return window

    def join_window_days(self, reference: str) -> int:
        normalized = normalize_reference(reference)
        return {
            "JRC": self.jrc_window_days,
            "DW": self.dw_window_days,
            "OPERA": self.opera_window_days,
        }[normalized]


@dataclass
class SampleTransfer:
    """Normalized samples plus the actual schema returned by computeFeatures."""

    frame: pd.DataFrame
    raw_columns: tuple[str, ...]


def parse_iso_date(value: str) -> dt.date:
    """Parse a strict ISO date for CLI and notebook overrides."""

    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date '{value}'; expected YYYY-MM-DD.") from exc


def normalize_reference(reference: str) -> str:
    """Normalize and validate a quick-example reference name."""

    normalized = str(reference).strip().upper()
    if normalized == "DYNAMIC_WORLD":
        normalized = "DW"
    if normalized not in SUPPORTED_REFERENCES:
        raise ValueError(
            f"Unsupported reference '{reference}'. Choose from "
            + ", ".join(SUPPORTED_REFERENCES)
            + "."
        )
    return normalized


def normalize_references(references: Sequence[str]) -> tuple[str, ...]:
    """Validate references while preserving user order and rejecting duplicates."""

    normalized = tuple(normalize_reference(reference) for reference in references)
    duplicates = sorted({ref for ref in normalized if normalized.count(ref) > 1})
    if duplicates:
        raise ValueError("Duplicate references: " + ", ".join(duplicates) + ".")
    if not normalized:
        raise ValueError("At least one reference is required.")
    return normalized


def load_candidate_inventory(path: str | Path) -> pd.DataFrame:
    """Load the local western candidate inventory used to resolve IDs."""

    candidate_path = Path(path)
    try:
        table = pd.read_csv(candidate_path)
    except OSError as exc:
        raise QuickSamplingError(
            f"Could not read candidate inventory '{candidate_path}': {exc}"
        ) from exc
    required = {
        "Hylak_id",
        "Lake_name",
        "state",
        "Lake_area",
        "centroid_lon",
        "centroid_lat",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise QuickSamplingError(
            "Candidate inventory is missing required columns: "
            + ", ".join(missing)
            + "."
        )
    ids = pd.to_numeric(table["Hylak_id"], errors="coerce")
    if ids.isna().any() or ids.duplicated().any():
        raise QuickSamplingError(
            "Candidate inventory must contain unique, non-missing Hylak_id values."
        )
    table = table.copy()
    table["Hylak_id"] = ids.astype(int)
    return table


def select_lake_specs(
    inventory: pd.DataFrame,
    lake_ids: Sequence[int],
) -> tuple[LakeSpec, ...]:
    """Resolve ordered candidate IDs to local lake metadata."""

    if not lake_ids:
        raise ValueError("At least one HydroLAKES ID is required.")
    normalized: list[int] = []
    for raw_id in lake_ids:
        if isinstance(raw_id, bool):
            raise TypeError(f"Invalid HydroLAKES ID: {raw_id!r}.")
        try:
            hylak_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid HydroLAKES ID: {raw_id!r}.") from exc
        if hylak_id <= 0 or str(raw_id).strip() != str(hylak_id):
            raise ValueError(f"Invalid HydroLAKES ID: {raw_id!r}.")
        normalized.append(hylak_id)
    duplicates = sorted({value for value in normalized if normalized.count(value) > 1})
    if duplicates:
        raise ValueError(
            "Duplicate HydroLAKES IDs: "
            + ", ".join(str(value) for value in duplicates)
            + "."
        )

    indexed = inventory.set_index("Hylak_id", drop=False)
    missing = [value for value in normalized if value not in indexed.index]
    if missing:
        raise ValueError(
            "HydroLAKES IDs are not in the western candidate inventory: "
            + ", ".join(str(value) for value in missing)
            + "."
        )

    specs: list[LakeSpec] = []
    for hylak_id in normalized:
        row = indexed.loc[hylak_id]
        name = "" if pd.isna(row["Lake_name"]) else str(row["Lake_name"])
        specs.append(
            LakeSpec(
                hylak_id=hylak_id,
                lake_name=name,
                state=str(row["state"]),
                lake_area=float(row["Lake_area"]),
                centroid_lon=float(row["centroid_lon"]),
                centroid_lat=float(row["centroid_lat"]),
            )
        )
    return tuple(specs)


def build_buffered_lake_aoi(
    ee_module: Any,
    hydrolakes_asset: str,
    lake: LakeSpec,
    buffer_km: float,
) -> tuple[Any, Any]:
    """Return the HydroLAKES feature and its consistently buffered AOI."""

    feature = ee_module.Feature(
        ee_module.FeatureCollection(hydrolakes_asset)
        .filter(ee_module.Filter.eq("Hylak_id", lake.hylak_id))
        .first()
    )
    aoi = feature.geometry().buffer(float(buffer_km) * 1000.0)
    return feature, aoi


def _cloud_mask(image: Any) -> Any:
    """Preserve the baseline high-confidence-cloud-only mask semantics."""

    qa = image.select("QA_PIXEL")
    cloud_high = qa.rightShift(8).bitwiseAnd(3).eq(3)
    return image.updateMask(cloud_high.Not())


def _scale_and_mask(image: Any) -> Any:
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2).clamp(0, 1)
    return _cloud_mask(image.addBands(optical, overwrite=True))


def _harmonize(image: Any, sensor: str) -> Any:
    mapping = LANDSAT_BAND_MAP[sensor]
    return image.select(list(mapping.values())).rename(list(mapping.keys()))


def _sensor_collection(
    ee_module: Any,
    sensor: str,
    aoi: Any,
    start: dt.date,
    end: dt.date,
    cloud_cover_max: float,
) -> Any | None:
    sensor_start, sensor_end = LANDSAT_SENSOR_WINDOWS[sensor]
    actual_start = max(start, sensor_start)
    actual_end = end if sensor_end == dt.date.max else min(end, sensor_end)
    if actual_start > actual_end:
        return None
    end_exclusive = actual_end + dt.timedelta(days=1)
    return (
        ee_module.ImageCollection(LANDSAT_COLLECTIONS[sensor])
        .filterBounds(aoi)
        .filterDate(actual_start.isoformat(), end_exclusive.isoformat())
        .filter(ee_module.Filter.lt("CLOUD_COVER", float(cloud_cover_max)))
        .map(_scale_and_mask)
        .map(lambda image: _harmonize(image, sensor).set("sensor", sensor))
    )


def build_landsat_collection(
    ee_module: Any,
    aoi: Any,
    start: dt.date,
    end: dt.date,
    *,
    cloud_cover_max: float = 20.0,
    sensors: Sequence[str] = LANDSAT_SENSORS,
) -> Any:
    """Build the baseline-aligned harmonized Landsat collection."""

    collections = [
        collection
        for collection in (
            _sensor_collection(
                ee_module,
                sensor,
                aoi,
                start,
                end,
                cloud_cover_max,
            )
            for sensor in sensors
        )
        if collection is not None
    ]
    if not collections:
        return ee_module.ImageCollection([])
    merged = collections[0]
    for collection in collections[1:]:
        merged = merged.merge(collection)
    return merged


def _ndwi(image: Any) -> Any:
    return image.normalizedDifference(["green", "nir"]).rename("index_value")


def _ndvi(image: Any) -> Any:
    return image.normalizedDifference(["nir", "red"]).rename("index_value")


def _nearest_time_join(
    ee_module: Any,
    primary: Any,
    secondary: Any,
    window_days: int,
    match_key: str,
) -> Any:
    time_filter = ee_module.Filter.maxDifference(
        difference=int(window_days * MS_PER_DAY),
        leftField="system:time_start",
        rightField="system:time_start",
    )
    joined = ee_module.Join.saveBest(
        matchKey=match_key,
        measureKey=JOIN_MEASURE_PROPERTY,
        outer=False,
    ).apply(primary, secondary, time_filter)
    return joined


def _matched_image_and_measure(
    ee_module: Any,
    primary: Any,
    match_key: str,
) -> tuple[Any, Any]:
    """Return the saved match and the join measure attached to that match."""

    matched = ee_module.Image(primary.get(match_key))
    return matched, matched.get(JOIN_MEASURE_PROPERTY)


def _jrc_labelled_collection(
    ee_module: Any,
    landsat: Any,
    aoi: Any,
    start: dt.date,
    end: dt.date,
    window_days: int,
) -> Any:
    end_exclusive = (end + dt.timedelta(days=1)).isoformat()
    jrc = (
        ee_module.ImageCollection(JRC_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start.isoformat(), end_exclusive)
    )

    def prepare(image: Any) -> Any:
        image = ee_module.Image(image)
        matched, match_delta_ms = _matched_image_and_measure(ee_module, image, "jrc")
        reference = matched.select("water")
        valid = reference.neq(0)
        label = reference.eq(2).rename("label").updateMask(valid)
        return (
            _ndwi(image)
            .addBands(label)
            .set(
                {
                    "system:time_start": image.get("system:time_start"),
                    "sensor": image.get("sensor"),
                    "reference_time": reference.get("system:time_start"),
                    "match_delta_ms": match_delta_ms,
                    "reference_tile_count": 1,
                }
            )
        )

    return _nearest_time_join(
        ee_module,
        landsat,
        jrc,
        window_days,
        "jrc",
    ).map(prepare)


def _dw_labelled_collection(
    ee_module: Any,
    landsat: Any,
    aoi: Any,
    start: dt.date,
    end: dt.date,
    window_days: int,
) -> Any:
    """Label NDVI while excluding sampled snow pixels, not whole scenes."""

    end_exclusive = (end + dt.timedelta(days=1)).isoformat()
    dynamic_world = (
        ee_module.ImageCollection(DYNAMIC_WORLD_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start.isoformat(), end_exclusive)
    )

    def valid_label(tile: Any) -> Any:
        label = (
            ee_module.Image(tile)
            .select("label")
            .remap(
                [*DYNAMIC_WORLD_VEGETATION, *DYNAMIC_WORLD_NONVEGETATION],
                [1] * len(DYNAMIC_WORLD_VEGETATION)
                + [0] * len(DYNAMIC_WORLD_NONVEGETATION),
                -1,
            )
            .rename("label")
        )
        return label.updateMask(label.gte(0))

    def prepare(image: Any) -> Any:
        image = ee_module.Image(image)
        matched, match_delta_ms = _matched_image_and_measure(ee_module, image, "dw")
        matched_time = matched.get("system:time_start")
        same_time = dynamic_world.filter(
            ee_module.Filter.eq("system:time_start", matched_time)
        )
        # Use every adjacent tile from the selected timestamp so boundary-
        # crossing lakes are not sampled from one arbitrarily selected tile.
        # This adds no AOI-wide reduction or client-side request.
        label = same_time.map(valid_label).mosaic()
        return (
            _ndvi(image)
            .addBands(label)
            .set(
                {
                    "system:time_start": image.get("system:time_start"),
                    "sensor": image.get("sensor"),
                    "reference_time": matched_time,
                    "match_delta_ms": match_delta_ms,
                    "reference_tile_count": same_time.size(),
                }
            )
        )

    return _nearest_time_join(
        ee_module,
        landsat,
        dynamic_world,
        window_days,
        "dw",
    ).map(prepare)


def _opera_valid_label(ee_module: Any, image: Any) -> Any:
    water = ee_module.Image(image).select(OPERA_WATER_BAND)
    label = water.remap([0, 1], [0, 1], -1).rename("label")
    return label.updateMask(label.gte(0))


def _opera_labelled_collection(
    ee_module: Any,
    landsat: Any,
    aoi: Any,
    start: dt.date,
    end: dt.date,
    window_days: int,
) -> Any:
    """Label NDWI from nearest-time GEE-native OPERA same-time mosaics."""

    end_exclusive = (end + dt.timedelta(days=1)).isoformat()
    opera = (
        ee_module.ImageCollection(OPERA_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start.isoformat(), end_exclusive)
    )
    joined = _nearest_time_join(
        ee_module,
        landsat,
        opera,
        window_days,
        "opera",
    )

    def prepare(image: Any) -> Any:
        image = ee_module.Image(image)
        matched, match_delta_ms = _matched_image_and_measure(ee_module, image, "opera")
        matched_time = matched.get("system:time_start")
        same_time = opera.filter(ee_module.Filter.eq("system:time_start", matched_time))
        label = same_time.map(lambda tile: _opera_valid_label(ee_module, tile)).mosaic()
        return (
            _ndwi(image)
            .addBands(label)
            .set(
                {
                    "system:time_start": image.get("system:time_start"),
                    "sensor": image.get("sensor"),
                    "reference_time": matched_time,
                    "match_delta_ms": match_delta_ms,
                    "reference_tile_count": same_time.size(),
                }
            )
        )

    return joined.map(prepare)


def build_labelled_collection(
    ee_module: Any,
    aoi: Any,
    reference: str,
    settings: QuickSamplingSettings,
) -> Any:
    """Build one Landsat/reference-matched image collection."""

    normalized = normalize_reference(reference)
    start, end = settings.reference_window(normalized)
    landsat = build_landsat_collection(
        ee_module,
        aoi,
        start,
        end,
        cloud_cover_max=settings.cloud_cover_max,
    )
    window_days = settings.join_window_days(normalized)
    if normalized == "JRC":
        return _jrc_labelled_collection(
            ee_module, landsat, aoi, start, end, window_days
        )
    if normalized == "DW":
        return _dw_labelled_collection(ee_module, landsat, aoi, start, end, window_days)
    return _opera_labelled_collection(ee_module, landsat, aoi, start, end, window_days)


def build_sample_collection(
    ee_module: Any,
    labelled_images: Any,
    aoi: Any,
    lake: LakeSpec,
    reference: str,
    settings: QuickSamplingSettings,
) -> Any:
    """Construct a small metadata-complete sampled FeatureCollection."""

    normalized = normalize_reference(reference)
    index_name = REFERENCE_INDEX[normalized]

    def sample_image(image: Any) -> Any:
        image = ee_module.Image(image)
        seed = ee_module.Number(image.get("system:time_start")).mod(1_000_000)
        sample_image = image.select(["index_value", "label"]).addBands(
            ee_module.Image.pixelLonLat()
        )
        points = sample_image.sample(
            region=aoi,
            scale=30,
            numPixels=settings.samples_per_image,
            seed=seed,
            geometries=False,
            dropNulls=True,
        )

        def add_metadata(feature: Any) -> Any:
            enriched = feature.set(
                {
                    "Hylak_id": lake.hylak_id,
                    "Lake_name": lake.lake_name,
                    "state": lake.state,
                    "Lake_area": lake.lake_area,
                    "reference": normalized,
                    "index": index_name,
                    "acquisition_time": image.get("system:time_start"),
                    "reference_time": image.get("reference_time"),
                    "match_delta_days": ee_module.Number(
                        image.get("match_delta_ms")
                    ).divide(MS_PER_DAY),
                    "reference_tile_count": image.get("reference_tile_count"),
                    "sensor": image.get("sensor"),
                    "lon": feature.get("longitude"),
                    "lat": feature.get("latitude"),
                }
            )
            return ee_module.Feature(
                None,
                enriched.toDictionary(list(SAMPLE_COLUMNS)),
            )

        return points.map(add_metadata)

    return labelled_images.map(sample_image).flatten()


def normalize_sample_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the actual computeFeatures pandas schema deterministically."""

    if not isinstance(raw, pd.DataFrame):
        raise QuickSamplingError(
            "ee.data.computeFeatures did not return a pandas DataFrame; got "
            f"{type(raw).__name__}."
        )
    if raw.empty:
        empty = pd.DataFrame(columns=[*SAMPLE_COLUMNS, "acquisition_date"])
        empty["acquisition_date"] = pd.to_datetime(empty["acquisition_date"], utc=True)
        return empty
    table = raw.copy()
    table = table.drop(
        columns=[
            column
            for column in ("geo", "geometry", ".geo", "system:index")
            if column in table.columns
        ],
        errors="ignore",
    )
    missing = [column for column in SAMPLE_COLUMNS if column not in table.columns]
    if missing:
        raise QuickSamplingError(
            "Unexpected computeFeatures schema. Missing columns: "
            + ", ".join(missing)
            + "; returned columns: "
            + ", ".join(str(column) for column in raw.columns)
            + "."
        )
    table = table.reindex(columns=SAMPLE_COLUMNS)
    numeric_columns = (
        "Hylak_id",
        "Lake_area",
        "index_value",
        "label",
        "acquisition_time",
        "reference_time",
        "match_delta_days",
        "reference_tile_count",
        "lon",
        "lat",
    )
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    invalid_label = ~table["label"].isin([0, 1])
    if table["label"].isna().any() or invalid_label.any():
        raise QuickSamplingError("Transferred labels must contain only 0 and 1.")
    if table[["index_value", "lon", "lat", "acquisition_time"]].isna().any().any():
        raise QuickSamplingError(
            "Transferred samples contain missing index, coordinate, or time values."
        )
    finite_coordinates = np.isfinite(table[["lon", "lat"]].to_numpy()).all()
    if not finite_coordinates:
        raise QuickSamplingError("Transferred samples contain non-finite coordinates.")
    invalid_longitude = ~table["lon"].between(-180, 180, inclusive="both")
    invalid_latitude = ~table["lat"].between(-90, 90, inclusive="both")
    if invalid_longitude.any() or invalid_latitude.any():
        raise QuickSamplingError(
            "Transferred sample coordinates fall outside longitude/latitude "
            "ranges; expected WGS84 longitude in [-180, 180] and latitude in "
            "[-90, 90]."
        )
    table["Hylak_id"] = table["Hylak_id"].astype(int)
    table["label"] = table["label"].astype(int)
    table["acquisition_date"] = pd.to_datetime(
        table["acquisition_time"], unit="ms", utc=True
    )
    return table.sort_values(
        ["acquisition_time", "lon", "lat"], kind="mergesort"
    ).reset_index(drop=True)


def sample_collection_to_dataframe(
    ee_module: Any,
    feature_collection: Any,
    settings: QuickSamplingSettings,
) -> SampleTransfer:
    """Retrieve at most the configured row limit through computeFeatures."""

    bounded = feature_collection.limit(settings.max_rows_per_combination + 1)
    try:
        raw = ee_module.data.computeFeatures(
            {
                "expression": bounded,
                "fileFormat": "PANDAS_DATAFRAME",
                "pageSize": settings.page_size,
            }
        )
    except Exception as exc:
        raise QuickSamplingError(
            f"Direct Earth Engine sample transfer failed: {type(exc).__name__}: {exc}"
        ) from exc
    if len(raw) > settings.max_rows_per_combination:
        raise SampleLimitError(
            "Sample exceeds max_rows_per_combination="
            f"{settings.max_rows_per_combination}. Lower samples_per_image or "
            "shorten the temporal window before retrying."
        )
    return SampleTransfer(
        frame=normalize_sample_dataframe(raw),
        raw_columns=tuple(str(column) for column in raw.columns),
    )


def sample_diagnostics(samples: pd.DataFrame) -> dict[str, Any]:
    """Summarize one transferred lake/reference sample."""

    if samples.empty:
        return {
            "n_rows": 0,
            "positive_fraction": np.nan,
            "date_start": None,
            "date_end": None,
            "sensors": (),
            "n_acquisitions": 0,
        }
    dates = samples["acquisition_date"]
    return {
        "n_rows": len(samples),
        "positive_fraction": float(samples["label"].mean()),
        "date_start": dates.min().date().isoformat(),
        "date_end": dates.max().date().isoformat(),
        "sensors": tuple(sorted(samples["sensor"].dropna().astype(str).unique())),
        "n_acquisitions": int(samples["acquisition_time"].nunique()),
    }


def utm_epsg_from_lon_lat(lon: float, lat: float) -> int:
    """Return the local UTM EPSG code used by the baseline workflow."""

    if not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise ValueError("Longitude/latitude are outside valid ranges.")
    zone = min(60, max(1, int((lon + 180) // 6) + 1))
    return (32600 if lat >= 0 else 32700) + zone


def assign_blocks_and_split(
    samples: pd.DataFrame,
    epsg: int,
    *,
    block_size_m: int = 1000,
    train_fraction: float = 0.80,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign 1 km UTM blocks wholly to deterministic train/test splits."""

    if samples.empty:
        result = samples.copy()
        result["block_id"] = pd.Series(dtype="string")
        result["split"] = pd.Series(dtype="string")
        return result
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(samples["lon"].to_numpy(), samples["lat"].to_numpy())
    result = samples.copy()
    block_x = np.floor(np.asarray(x) / block_size_m).astype(int)
    block_y = np.floor(np.asarray(y) / block_size_m).astype(int)
    result["block_id"] = [
        f"{x_value}_{y_value}"
        for x_value, y_value in zip(block_x, block_y, strict=True)
    ]
    blocks = np.array(sorted(result["block_id"].unique()))
    np.random.default_rng(seed).shuffle(blocks)
    train_count = int(len(blocks) * train_fraction)
    train_count = min(max(train_count, 1), max(len(blocks) - 1, 1))
    split_by_block = {block: "train" for block in blocks[:train_count]}
    split_by_block.update({block: "test" for block in blocks[train_count:]})
    result["split"] = result["block_id"].map(split_by_block)
    return result


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    _tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def grid_search_threshold(
    values: Sequence[float],
    labels: Sequence[int],
    *,
    n_steps: int = 150,
) -> pd.DataFrame:
    """Sweep the central 99% index range and score ``value >= threshold``."""

    value_array = np.asarray(values, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    if not len(value_array) or len(value_array) != len(label_array):
        raise ValueError("Threshold values and labels must be non-empty and aligned.")
    low, high = np.percentile(value_array, [0.5, 99.5])
    rows = []
    for threshold in np.linspace(low, high, n_steps):
        predictions = value_array >= threshold
        rows.append(
            {"threshold": float(threshold), **_metrics(label_array, predictions)}
        )
    return pd.DataFrame(rows)


def evaluate_threshold(
    values: Sequence[float], labels: Sequence[int], threshold: float
) -> dict[str, float]:
    value_array = np.asarray(values, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    return _metrics(label_array, value_array >= threshold)


def cv_optimal_threshold(
    training: pd.DataFrame,
    *,
    n_splits: int = 5,
    grid_steps: int = 150,
) -> tuple[float, float, float]:
    """Fit thresholds inside grouped training folds and score held-out folds."""

    groups = training["block_id"].to_numpy()
    unique_groups = np.unique(groups)
    fold_count = min(n_splits, len(unique_groups))
    if fold_count < 2:
        raise ValueError("Grouped CV requires at least two training blocks.")
    values = training["index_value"].to_numpy(dtype=float)
    labels = training["label"].to_numpy(dtype=int)
    splitter = GroupKFold(n_splits=fold_count)
    thresholds: list[float] = []
    scores: list[float] = []
    for train_index, held_index in splitter.split(values, labels, groups):
        if np.unique(labels[train_index]).size < 2:
            raise InsufficientSampleError(
                "a grouped-CV training fold contains only one label class"
            )
        grid = grid_search_threshold(
            values[train_index], labels[train_index], n_steps=grid_steps
        )
        threshold = float(grid.loc[grid["iou"].idxmax(), "threshold"])
        thresholds.append(threshold)
        scores.append(
            evaluate_threshold(values[held_index], labels[held_index], threshold)["iou"]
        )
    return (
        float(np.mean(thresholds)),
        float(np.std(thresholds)),
        float(np.mean(scores)),
    )


def base_summary(
    lake: LakeSpec,
    reference: str,
    settings: QuickSamplingSettings,
) -> dict[str, Any]:
    start, end = settings.reference_window(reference)
    summary: dict[str, Any] = {column: np.nan for column in SUMMARY_COLUMNS}
    summary.update(
        {
            "Hylak_id": lake.hylak_id,
            "Lake_name": lake.lake_name,
            "state": lake.state,
            "Lake_area": lake.lake_area,
            "reference": reference,
            "index": REFERENCE_INDEX[reference],
            "date_start": start.isoformat(),
            "date_end": end.isoformat(),
            "n_samples": 0,
            "n_train": 0,
            "n_test": 0,
            "n_blocks": 0,
            "status": "skipped",
            "skip_reason": "",
            "sensors": "",
            "buffer_km": settings.buffer_km,
            "samples_per_image": settings.samples_per_image,
            "join_window_days": settings.join_window_days(reference),
        }
    )
    return summary


def optimize_samples(
    samples: pd.DataFrame,
    lake: LakeSpec,
    reference: str,
    settings: QuickSamplingSettings,
) -> dict[str, Any]:
    """Run the baseline-style spatial split/CV/test optimization locally."""

    normalized = normalize_reference(reference)
    summary = base_summary(lake, normalized, settings)
    clean = samples.dropna(subset=["index_value", "label", "lon", "lat"]).copy()
    clean = clean[clean["label"].isin([0, 1])]
    summary["n_samples"] = len(clean)
    if len(clean):
        summary["positive_fraction"] = float(clean["label"].mean())
        dates = pd.to_datetime(clean["acquisition_time"], unit="ms", utc=True)
        summary["sample_date_start"] = dates.min().date().isoformat()
        summary["sample_date_end"] = dates.max().date().isoformat()
        summary["sensors"] = ",".join(
            sorted(clean["sensor"].dropna().astype(str).unique())
        )

    def skip(reason: str) -> dict[str, Any]:
        summary["skip_reason"] = reason
        return summary

    if len(clean) < settings.min_samples:
        return skip(f"too few samples ({len(clean)} < {settings.min_samples})")
    if clean["label"].nunique() < 2:
        return skip("sample contains only one label class")
    if clean["index_value"].nunique() < 2:
        return skip("sample contains no index-value variation")

    epsg = utm_epsg_from_lon_lat(lake.centroid_lon, lake.centroid_lat)
    blocked = assign_blocks_and_split(
        clean,
        epsg,
        block_size_m=settings.block_size_m,
        train_fraction=settings.train_fraction,
        seed=settings.random_seed,
    )
    summary["n_blocks"] = int(blocked["block_id"].nunique())
    if summary["n_blocks"] < settings.min_blocks:
        return skip(
            f"too few spatial blocks ({summary['n_blocks']} < {settings.min_blocks})"
        )

    training = blocked[blocked["split"] == "train"]
    testing = blocked[blocked["split"] == "test"]
    summary["n_train"] = len(training)
    summary["n_test"] = len(testing)
    if len(training) < settings.min_split_samples:
        return skip("too few training samples")
    if len(testing) < settings.min_split_samples:
        return skip("too few held-out test samples")
    if training["block_id"].nunique() < 2:
        return skip("too few training blocks for grouped CV")
    if training["label"].nunique() < 2:
        return skip("training split contains only one label class")
    if testing["label"].nunique() < 2:
        return skip("test split contains only one label class")

    try:
        threshold_mean, threshold_std, cv_iou = cv_optimal_threshold(
            training,
            n_splits=settings.cv_splits,
            grid_steps=settings.grid_steps,
        )
    except InsufficientSampleError as exc:
        return skip(str(exc))
    test_metrics = evaluate_threshold(
        testing["index_value"], testing["label"], threshold_mean
    )
    summary.update(
        {
            # Kept only in the in-memory result. summary_dataframe() preserves
            # the established compact output contract and drops private keys.
            "_threshold_exact": threshold_mean,
            "threshold_mean": round(threshold_mean, 4),
            "threshold_std": round(threshold_std, 4),
            "cv_iou": round(cv_iou, 3),
            "test_iou": round(test_metrics["iou"], 3),
            "test_f1": round(test_metrics["f1"], 3),
            "test_precision": round(test_metrics["precision"], 3),
            "test_recall": round(test_metrics["recall"], 3),
            "status": "ok",
            "skip_reason": "",
        }
    )
    return summary


def summary_dataframe(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return deterministic quick-example summary columns."""

    return pd.DataFrame.from_records(rows).reindex(columns=SUMMARY_COLUMNS)
