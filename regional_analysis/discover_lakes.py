"""Discover and QA western-U.S. HydroLAKES candidates in Earth Engine.

This module deliberately stops at a metadata-only candidate inventory. It does
not acquire Landsat imagery, sample pixels, calibrate thresholds, or choose a
final experimental lake sample.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ConfigError, DiscoveryConfig, load_config

REQUIRED_HYDROLAKES_PROPERTIES = (
    "Hylak_id",
    "Lake_area",
    "Lake_type",
    "Country",
)

OPTIONAL_HYDROLAKES_PROPERTIES = (
    "Lake_name",
    "Elevation",
    "Vol_total",
    "Depth_avg",
    "Dis_avg",
    "Res_time",
    "Slope_100",
    "Wshd_area",
    "Pour_long",
    "Pour_lat",
)

# The first eight columns are the stable minimum contract requested for the
# local candidate inventory. Country and available optional metadata follow in
# a deterministic HydroLAKES order.
CORE_OUTPUT_COLUMNS = (
    "Hylak_id",
    "Lake_name",
    "Lake_area",
    "Lake_type",
    "Elevation",
    "state",
    "centroid_lon",
    "centroid_lat",
)

EXTRA_OUTPUT_COLUMNS = (
    "Country",
    "Vol_total",
    "Depth_avg",
    "Dis_avg",
    "Res_time",
    "Slope_100",
    "Wshd_area",
    "Pour_long",
    "Pour_lat",
)

# HydroLAKES v1.0 uses numeric sentinel values for several fields. Raw values
# remain unchanged in the reproducibility CSV; summaries report their counts so
# they are not mistaken for complete measured/modelled metadata.
HYDROLAKES_SENTINEL_VALUES: dict[str, tuple[float, ...]] = {
    "Dis_avg": (-9999.0,),
    "Res_time": (-9999.0, -1.0),
    "Slope_100": (-1.0,),
    "Wshd_area": (-9999.0,),
}

ANCHOR_REPORT_COLUMNS = CORE_OUTPUT_COLUMNS

# Exact rectangles from the existing calibration scripts. They are broad study
# AOIs, not lake boundaries, and are used only when name-based anchor QA is not
# reliable. Do not import the calibration modules: some authenticate and run
# live scientific work at import time.
ANCHOR_AOI_BOUNDS = {
    "Great Salt Lake": (
        -113.3061047,
        40.51087812,
        -111.8534658,
        41.87473984,
    ),
    "Lake Abert": (
        -120.6098988,
        42.40054901,
        -119.9981325,
        42.85172324,
    ),
}

DEFAULT_CONFIG_PATH = Path("configs/western_us.yaml")
DEFAULT_OUTPUT_PATH = Path("data/regional/western_hydrolakes_candidates.csv")
DEFAULT_PAGE_SIZE = 5000
_STATE_MATCH_COUNT = "_state_match_count"
_STATE_MATCHES = "_state_matches"
_REGIONAL_STATE_MATCH_COUNT = "_regional_state_match_count"
_ANCHOR_CONTAINS_CENTER = "_anchor_contains_center"
_ANCHOR_INTERSECTION_KM2 = "_anchor_intersection_km2"
_ANCHOR_PASSES_AREA = "_anchor_passes_area_screen"
_ANCHOR_PASSES_STATE = "_anchor_passes_state_screen"
_ANCHOR_IS_CANDIDATE = "_anchor_is_candidate"
_ANCHOR_LIKELY = "_anchor_likely"


class DiscoveryError(RuntimeError):
    """Raised when live candidate discovery cannot be completed safely."""


class CandidateTableError(ValueError):
    """Raised when downloaded candidate metadata violates table invariants."""


@dataclass(frozen=True)
class HydroLakesSchema:
    """Validated HydroLAKES property availability for the configured asset."""

    available_properties: tuple[str, ...]
    retained_optional_properties: tuple[str, ...]
    missing_optional_properties: tuple[str, ...]


@dataclass
class AnchorCheck:
    """Result of name-based or geographic QA for one existing study lake."""

    anchor_name: str
    method: str
    records: pd.DataFrame
    note: str


@dataclass
class DiscoveryResult:
    """Candidate metadata plus server-side collections retained for QA maps."""

    schema: HydroLakesSchema
    state_collection: Any
    candidate_collection: Any
    table: pd.DataFrame
    summary: dict[str, Any]
    boundary_ambiguities: pd.DataFrame
    anchor_checks: dict[str, AnchorCheck]


def initialize_earth_engine() -> tuple[Any, str]:
    """Initialize Earth Engine from ambient ADC and the required EE_PROJECT.

    No authentication flow is initiated here. If Application Default
    Credentials or the configured project are unavailable, ``ee.Initialize``
    raises and its original error text is included in the exception.
    """

    try:
        project = os.environ["EE_PROJECT"].strip()
    except KeyError as exc:
        raise DiscoveryError(
            "EE_PROJECT is not set. Set it to the registered Earth Engine "
            "Cloud project before running; no fallback project is used."
        ) from exc

    if not project:
        raise DiscoveryError(
            "EE_PROJECT is empty. Set it to the registered Earth Engine "
            "Cloud project before running."
        )

    try:
        import ee
    except ImportError as exc:
        raise DiscoveryError(
            "earthengine-api is not installed in the active Python environment."
        ) from exc

    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise DiscoveryError(
            f"Earth Engine initialization failed for project '{project}' "
            f"using ambient credentials: {type(exc).__name__}: {exc}"
        ) from exc

    return ee, project


def validate_hydrolakes_schema(
    ee_module: Any,
    hydrolakes_asset: str,
) -> HydroLakesSchema:
    """Inspect one HydroLAKES feature and validate required properties."""

    try:
        collection = ee_module.FeatureCollection(hydrolakes_asset)
        inspection = ee_module.Dictionary(
            {
                "feature_count": collection.size(),
                "properties": ee_module.Feature(collection.first()).propertyNames(),
            }
        ).getInfo()
    except Exception as exc:
        raise DiscoveryError(
            f"Could not inspect HydroLAKES asset '{hydrolakes_asset}': "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not inspection or int(inspection.get("feature_count", 0)) == 0:
        raise DiscoveryError(
            f"HydroLAKES asset '{hydrolakes_asset}' contains no features."
        )

    available = tuple(str(name) for name in inspection.get("properties", []))
    available_set = set(available)
    missing_required = [
        name for name in REQUIRED_HYDROLAKES_PROPERTIES if name not in available_set
    ]
    if missing_required:
        raise DiscoveryError(
            "HydroLAKES schema is missing required properties: "
            + ", ".join(missing_required)
            + ". Available properties: "
            + ", ".join(sorted(available_set))
            + "."
        )

    retained_optional = tuple(
        name for name in OPTIONAL_HYDROLAKES_PROPERTIES if name in available_set
    )
    missing_optional = tuple(
        name for name in OPTIONAL_HYDROLAKES_PROPERTIES if name not in available_set
    )
    return HydroLakesSchema(
        available_properties=available,
        retained_optional_properties=retained_optional,
        missing_optional_properties=missing_optional,
    )


def build_western_state_collection(
    ee_module: Any,
    states_asset: str,
    state_codes: Sequence[str],
) -> Any:
    """Build and validate the configured TIGER state collection server-side."""

    requested = tuple(state_codes)
    collection = (
        ee_module.FeatureCollection(states_asset)
        .filter(ee_module.Filter.inList("STUSPS", list(requested)))
        .select(["STUSPS"])
    )
    try:
        found = tuple(
            sorted(
                str(code)
                for code in collection.aggregate_array("STUSPS").distinct().getInfo()
            )
        )
    except Exception as exc:
        raise DiscoveryError(
            f"Could not inspect configured states in '{states_asset}': "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    missing = sorted(set(requested) - set(found))
    unexpected = sorted(set(found) - set(requested))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise DiscoveryError(
            "Configured TIGER state lookup did not match the requested domain ("
            + "; ".join(details)
            + ")."
        )
    return collection


def add_centroid_metadata(feature: Any) -> Any:
    """Attach geodesic centroid longitude and latitude to one EE feature."""

    centroid = feature.geometry().centroid(maxError=1)
    coordinates = centroid.coordinates()
    return feature.set(
        {
            "centroid_lon": coordinates.get(0),
            "centroid_lat": coordinates.get(1),
        }
    )


def assign_state(
    feature: Any,
    state_collection: Any,
    ambiguity_state_collection: Any | None = None,
) -> Any:
    """Assign a configured state, with alphabetical tie-breaking on borders.

    ``state_collection`` defines eligible states. ``ambiguity_state_collection``
    may contain all TIGER states, allowing ties at configured/nonconfigured
    borders to be reported too. Eligible STUSPS codes take precedence, then the
    lexicographically first eligible code is selected.
    """

    centroid = feature.geometry().centroid(maxError=1)
    regional_matches = state_collection.filterBounds(centroid)
    regional_codes = regional_matches.aggregate_array("STUSPS").distinct().sort()
    all_states = (
        state_collection
        if ambiguity_state_collection is None
        else ambiguity_state_collection
    )
    all_matches = all_states.filterBounds(centroid)
    all_codes = all_matches.aggregate_array("STUSPS").distinct().sort()
    # Appending an empty value makes get(0) safe when no state matches.
    assigned_state = regional_codes.cat([""]).get(0)
    return feature.set(
        {
            "state": assigned_state,
            _REGIONAL_STATE_MATCH_COUNT: regional_codes.size(),
            _STATE_MATCH_COUNT: all_codes.size(),
            _STATE_MATCHES: all_codes.join("|"),
        }
    )


def produce_candidate_hydrolakes_collection(
    ee_module: Any,
    hydrolakes_asset: str,
    state_collection: Any,
    min_area_km2: float,
    *,
    ambiguity_state_collection: Any | None = None,
) -> Any:
    """Return lakes whose polygon centroid belongs to a configured state.

    Polygon/state intersection is used only as a server-side prefilter. Final
    membership is determined exclusively from the HydroLAKES polygon centroid.
    Lake type is retained as metadata and is not used as a filter.
    """

    regional_geometry = state_collection.geometry()
    candidates = (
        ee_module.FeatureCollection(hydrolakes_asset)
        .filter(ee_module.Filter.gte("Lake_area", float(min_area_km2)))
        .filterBounds(regional_geometry)
        .map(add_centroid_metadata)
        .map(
            lambda feature: assign_state(
                feature,
                state_collection,
                ambiguity_state_collection,
            )
        )
        .filter(ee_module.Filter.gt(_REGIONAL_STATE_MATCH_COUNT, 0))
        .sort("Lake_area", False)
    )
    return candidates


def candidate_output_columns(
    available_properties: Iterable[str] = (),
) -> list[str]:
    """Return the deterministic local CSV column order for a schema."""

    available = set(available_properties)
    columns = list(CORE_OUTPUT_COLUMNS)
    # Country is a required HydroLAKES property and is always useful metadata.
    columns.append("Country")
    columns.extend(
        column
        for column in EXTRA_OUTPUT_COLUMNS
        if column != "Country" and column in available
    )
    return columns


def _transfer_properties(schema: HydroLakesSchema) -> list[str]:
    available = set(schema.available_properties)
    source_properties = [
        name
        for name in (
            *REQUIRED_HYDROLAKES_PROPERTIES,
            *OPTIONAL_HYDROLAKES_PROPERTIES,
        )
        if name in available
    ]
    return [*source_properties, "state", "centroid_lon", "centroid_lat"]


def feature_collection_to_dataframe(
    ee_module: Any,
    collection: Any,
    properties: Sequence[str],
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> pd.DataFrame:
    """Download selected feature metadata in pages, with geometry removed.

    There is one ``getInfo`` call per metadata page, never one per feature.
    Typical 25 km² western inventories fit in a single page; pagination protects
    larger experiments from a single oversized interactive response.
    """

    if page_size <= 0:
        raise ValueError("page_size must be greater than zero.")
    ordered_properties = list(dict.fromkeys(properties))

    def properties_only(feature: Any) -> Any:
        return ee_module.Feature(
            None,
            feature.toDictionary(ordered_properties),
        )

    metadata = collection.map(properties_only)
    try:
        total = int(metadata.size().getInfo())
    except Exception as exc:
        raise DiscoveryError(
            "Could not determine the candidate metadata transfer size: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    records: list[Mapping[str, Any]] = []
    for offset in range(0, total, page_size):
        count = min(page_size, total - offset)
        try:
            page = metadata.toList(count, offset).getInfo()
        except Exception as exc:
            raise DiscoveryError(
                "Candidate metadata transfer failed for records "
                f"{offset} through {offset + count - 1}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        records.extend(feature.get("properties", {}) for feature in page)

    return pd.DataFrame.from_records(records, columns=ordered_properties)


def prepare_candidate_table(
    records: pd.DataFrame | Iterable[Mapping[str, Any]],
    available_properties: Iterable[str] = (),
) -> pd.DataFrame:
    """Normalize metadata to deterministic columns and descending lake area."""

    if isinstance(records, pd.DataFrame):
        raw = records.copy()
    else:
        raw = pd.DataFrame.from_records(records)

    columns = candidate_output_columns(available_properties)
    table = raw.reindex(columns=columns).copy()

    numeric_columns = (
        "Lake_area",
        "Elevation",
        "centroid_lon",
        "centroid_lat",
        "Vol_total",
        "Depth_avg",
        "Dis_avg",
        "Res_time",
        "Slope_100",
        "Wshd_area",
        "Pour_long",
        "Pour_lat",
    )
    for column in numeric_columns:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")

    if "Lake_area" in table.columns:
        table["_hylak_sort_numeric"] = pd.to_numeric(
            table["Hylak_id"],
            errors="coerce",
        )
        table["_hylak_sort_text"] = table["Hylak_id"].astype("string")
        table = table.sort_values(
            ["Lake_area", "_hylak_sort_numeric", "_hylak_sort_text"],
            ascending=[False, True, True],
            kind="mergesort",
            na_position="last",
        ).drop(columns=["_hylak_sort_numeric", "_hylak_sort_text"])
    return table.reset_index(drop=True)


def transfer_candidate_metadata(
    ee_module: Any,
    candidate_collection: Any,
    schema: HydroLakesSchema,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> pd.DataFrame:
    """Transfer metadata only and return a normalized local candidate table."""

    raw = feature_collection_to_dataframe(
        ee_module,
        candidate_collection,
        _transfer_properties(schema),
        page_size=page_size,
    )
    return prepare_candidate_table(raw, schema.available_properties)


def validate_candidate_table(
    table: pd.DataFrame,
    configured_states: Iterable[str],
) -> None:
    """Validate persistent IDs, areas, centroids, and configured membership."""

    missing_columns = [
        column for column in CORE_OUTPUT_COLUMNS if column not in table.columns
    ]
    if missing_columns:
        raise CandidateTableError(
            "Candidate table is missing required columns: "
            + ", ".join(missing_columns)
            + "."
        )

    identifiers = table["Hylak_id"]
    missing_ids = identifiers.isna() | identifiers.astype(str).str.strip().eq("")
    if missing_ids.any():
        rows = table.index[missing_ids].tolist()
        raise CandidateTableError(
            f"Candidate table has missing Hylak_id values at rows {rows}."
        )

    duplicate_ids = identifiers[identifiers.duplicated(keep=False)]
    if not duplicate_ids.empty:
        duplicates = sorted({str(value) for value in duplicate_ids.tolist()})
        raise CandidateTableError(
            "Candidate table has duplicate Hylak_id values: "
            + ", ".join(duplicates)
            + "."
        )

    area = pd.to_numeric(table["Lake_area"], errors="coerce")
    invalid_area = area.isna() | area.le(0)
    if invalid_area.any():
        rows = table.index[invalid_area].tolist()
        raise CandidateTableError(
            f"Lake_area must be positive and numeric; invalid rows: {rows}."
        )

    longitude = pd.to_numeric(table["centroid_lon"], errors="coerce")
    invalid_longitude = longitude.isna() | ~longitude.between(-180, 180)
    if invalid_longitude.any():
        rows = table.index[invalid_longitude].tolist()
        raise CandidateTableError(
            f"centroid_lon must be within [-180, 180]; invalid rows: {rows}."
        )

    latitude = pd.to_numeric(table["centroid_lat"], errors="coerce")
    invalid_latitude = latitude.isna() | ~latitude.between(-90, 90)
    if invalid_latitude.any():
        rows = table.index[invalid_latitude].tolist()
        raise CandidateTableError(
            f"centroid_lat must be within [-90, 90]; invalid rows: {rows}."
        )

    allowed_states = set(configured_states)
    invalid_state = ~table["state"].isin(allowed_states)
    if invalid_state.any():
        invalid_values = sorted(
            {str(value) for value in table.loc[invalid_state, "state"].tolist()}
        )
        raise CandidateTableError(
            "Candidate table contains states outside the configured domain: "
            + ", ".join(invalid_values)
            + "."
        )


def write_candidate_csv(
    table: pd.DataFrame,
    output_path: str | Path,
    configured_states: Iterable[str],
) -> Path:
    """Validate and write the metadata-only candidate inventory."""

    validate_candidate_table(table, configured_states)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False, lineterminator="\n")
    return destination


def _missing_text(values: pd.Series) -> pd.Series:
    return values.isna() | values.astype(str).str.strip().eq("")


def summarize_candidate_table(table: pd.DataFrame) -> dict[str, Any]:
    """Return compact population, metadata-completeness, and area diagnostics."""

    area = pd.to_numeric(table["Lake_area"], errors="coerce")
    name_missing = _missing_text(table["Lake_name"])
    lake_type = table["Lake_type"].astype("string").fillna("<missing>")
    quantile_levels = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

    optional_missing: dict[str, int] = {}
    for column in OPTIONAL_HYDROLAKES_PROPERTIES:
        if column not in table.columns:
            optional_missing[column] = len(table)
        elif column == "Lake_name":
            optional_missing[column] = int(_missing_text(table[column]).sum())
        else:
            optional_missing[column] = int(table[column].isna().sum())

    sentinel_counts: dict[str, dict[str, int]] = {}
    for column, sentinels in HYDROLAKES_SENTINEL_VALUES.items():
        if column not in table.columns:
            continue
        numeric = pd.to_numeric(table[column], errors="coerce")
        counts = {
            f"{sentinel:g}": int(numeric.eq(sentinel).sum())
            for sentinel in sentinels
            if numeric.eq(sentinel).any()
        }
        if counts:
            sentinel_counts[column] = counts

    return {
        "total_candidates": len(table),
        "counts_by_state": {
            str(key): int(value)
            for key, value in table["state"].value_counts().sort_index().items()
        },
        "counts_by_lake_type": {
            str(key): int(value)
            for key, value in lake_type.value_counts().sort_index().items()
        },
        "named_count": int((~name_missing).sum()),
        "unnamed_count": int(name_missing.sum()),
        "area_km2": {
            "min": float(area.min()) if len(area) else None,
            "max": float(area.max()) if len(area) else None,
            "mean": float(area.mean()) if len(area) else None,
            "median": float(area.median()) if len(area) else None,
            "quantiles": {
                f"q{int(level * 100):02d}": float(value)
                for level, value in area.quantile(quantile_levels).items()
            }
            if len(area)
            else {},
        },
        "missing_optional_values": optional_missing,
        "hydrolakes_sentinel_values": sentinel_counts,
    }


def boundary_ambiguity_table(
    ee_module: Any,
    candidate_collection: Any,
    schema: HydroLakesSchema,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> pd.DataFrame:
    """Download the usually empty set of centroid/state boundary ties."""

    properties = [
        *_transfer_properties(schema),
        _STATE_MATCH_COUNT,
        _STATE_MATCHES,
    ]
    raw = feature_collection_to_dataframe(
        ee_module,
        candidate_collection.filter(ee_module.Filter.gt(_STATE_MATCH_COUNT, 1)),
        properties,
        page_size=page_size,
    )
    columns = [
        *ANCHOR_REPORT_COLUMNS,
        _STATE_MATCH_COUNT,
        _STATE_MATCHES,
    ]
    for column in columns:
        if column not in raw.columns:
            raw[column] = pd.NA
    return (
        raw.reindex(columns=columns)
        .sort_values(
            ["Hylak_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _exact_name_matches(table: pd.DataFrame, anchor_name: str) -> pd.DataFrame:
    if "Lake_name" not in table.columns:
        return pd.DataFrame(columns=ANCHOR_REPORT_COLUMNS)
    names = table["Lake_name"].astype("string").str.strip().str.casefold()
    mask = names.eq(anchor_name.casefold()).fillna(False)
    return table.loc[mask, list(ANCHOR_REPORT_COLUMNS)].reset_index(drop=True)


def _geographic_anchor_diagnostic(
    ee_module: Any,
    hydrolakes_asset: str,
    state_collection: Any,
    ambiguity_state_collection: Any,
    schema: HydroLakesSchema,
    bounds: Sequence[float],
    min_area_km2: float,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> pd.DataFrame:
    """Rank source lakes intersecting an unchanged reference AOI rectangle.

    The source query intentionally precedes area and state screening so an
    anchor excluded by a one-run threshold override can still be diagnosed.
    Candidate-status flags explain whether each intersecting source feature
    would enter the configured inventory.
    """

    aoi = ee_module.Geometry.Rectangle(list(bounds))
    aoi_center = aoi.centroid(maxError=1)

    def add_diagnostic(feature: Any) -> Any:
        geometry = feature.geometry()
        passes_area = ee_module.Number(feature.get("Lake_area")).gte(
            float(min_area_km2)
        )
        passes_state = ee_module.Number(feature.get(_REGIONAL_STATE_MATCH_COUNT)).gt(0)
        return feature.set(
            {
                _ANCHOR_CONTAINS_CENTER: geometry.contains(aoi_center, 1),
                _ANCHOR_INTERSECTION_KM2: geometry.intersection(aoi, 1)
                .area(1)
                .divide(1_000_000),
                _ANCHOR_PASSES_AREA: passes_area,
                _ANCHOR_PASSES_STATE: passes_state,
                _ANCHOR_IS_CANDIDATE: passes_area.And(passes_state),
            }
        )

    collection = (
        ee_module.FeatureCollection(hydrolakes_asset)
        .filterBounds(aoi)
        .map(add_centroid_metadata)
        .map(
            lambda feature: assign_state(
                feature,
                state_collection,
                ambiguity_state_collection,
            )
        )
        .map(add_diagnostic)
    )
    properties = [
        *_transfer_properties(schema),
        _ANCHOR_CONTAINS_CENTER,
        _ANCHOR_INTERSECTION_KM2,
        _ANCHOR_PASSES_AREA,
        _ANCHOR_PASSES_STATE,
        _ANCHOR_IS_CANDIDATE,
    ]
    raw = feature_collection_to_dataframe(
        ee_module,
        collection,
        properties,
        page_size=page_size,
    )
    report_columns = [
        *ANCHOR_REPORT_COLUMNS,
        _ANCHOR_LIKELY,
        _ANCHOR_IS_CANDIDATE,
        _ANCHOR_PASSES_AREA,
        _ANCHOR_PASSES_STATE,
        _ANCHOR_CONTAINS_CENTER,
        _ANCHOR_INTERSECTION_KM2,
    ]
    source_columns = [column for column in report_columns if column != _ANCHOR_LIKELY]
    for column in source_columns:
        if column not in raw.columns:
            raw[column] = pd.NA
    if raw.empty:
        return raw.reindex(columns=report_columns)
    raw["Lake_area"] = pd.to_numeric(raw["Lake_area"], errors="coerce")
    raw[_ANCHOR_INTERSECTION_KM2] = pd.to_numeric(
        raw[_ANCHOR_INTERSECTION_KM2],
        errors="coerce",
    )
    ranked = (
        raw.reindex(columns=source_columns)
        .sort_values(
            [_ANCHOR_CONTAINS_CENTER, _ANCHOR_INTERSECTION_KM2, "Lake_area"],
            ascending=[False, False, False],
            kind="mergesort",
            na_position="last",
        )
        .reset_index(drop=True)
    )
    ranked[_ANCHOR_LIKELY] = False
    ranked.loc[0, _ANCHOR_LIKELY] = True
    return ranked.reindex(columns=report_columns).head(10)


def check_anchor_sites(
    ee_module: Any,
    table: pd.DataFrame,
    schema: HydroLakesSchema,
    hydrolakes_asset: str,
    state_collection: Any,
    ambiguity_state_collection: Any,
    min_area_km2: float,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> dict[str, AnchorCheck]:
    """Check the existing study lakes by name, then unchanged AOIs if needed."""

    checks: dict[str, AnchorCheck] = {}
    for anchor_name, bounds in ANCHOR_AOI_BOUNDS.items():
        name_matches = _exact_name_matches(table, anchor_name)
        if len(name_matches) == 1:
            checks[anchor_name] = AnchorCheck(
                anchor_name=anchor_name,
                method="case-insensitive exact Lake_name match",
                records=name_matches,
                note="One unambiguous candidate name match was found.",
            )
            continue

        geographic = _geographic_anchor_diagnostic(
            ee_module,
            hydrolakes_asset,
            state_collection,
            ambiguity_state_collection,
            schema,
            bounds,
            min_area_km2,
            page_size=page_size,
        )
        if name_matches.empty:
            reason = "No case-insensitive exact Lake_name match was found."
        else:
            reason = (
                f"{len(name_matches)} exact Lake_name matches were found, so the "
                "name was not unique."
            )
        naming_note = ""
        if not geographic.empty:
            likely_name = geographic.iloc[0]["Lake_name"]
            if pd.isna(likely_name) or not str(likely_name).strip():
                naming_note = " The top-ranked source feature has no Lake_name."
            elif str(likely_name).strip().casefold() != anchor_name.casefold():
                naming_note = (
                    " The top-ranked source feature is named "
                    f"'{str(likely_name).strip()}' in HydroLAKES."
                )
        checks[anchor_name] = AnchorCheck(
            anchor_name=anchor_name,
            method="existing-AOI geographic diagnostic",
            records=geographic,
            note=(
                reason + " Candidates intersecting the unchanged repository AOI are "
                "ranked by center containment, intersecting area, and Lake_area; "
                "the top-ranked record is explicitly flagged. Up to ten source "
                "features are shown with area/state screening status. This is a "
                "naming/geometry diagnostic, not a new AOI definition." + naming_note
            ),
        )
    return checks


def discover_candidate_inventory(
    ee_module: Any,
    config: DiscoveryConfig,
    *,
    schema: HydroLakesSchema | None = None,
    state_collection: Any | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> DiscoveryResult:
    """Run schema, state, candidate, transfer, and QA stages."""

    if schema is None:
        schema = validate_hydrolakes_schema(
            ee_module,
            config.earth_engine.hydrolakes_asset,
        )
    if state_collection is None:
        state_collection = build_western_state_collection(
            ee_module,
            config.earth_engine.states_asset,
            config.region.states,
        )
    ambiguity_state_collection = ee_module.FeatureCollection(
        config.earth_engine.states_asset
    ).select(["STUSPS"])
    candidates = produce_candidate_hydrolakes_collection(
        ee_module,
        config.earth_engine.hydrolakes_asset,
        state_collection,
        config.selection.min_area_km2,
        ambiguity_state_collection=ambiguity_state_collection,
    )
    table = transfer_candidate_metadata(
        ee_module,
        candidates,
        schema,
        page_size=page_size,
    )
    validate_candidate_table(table, config.region.states)
    ambiguities = boundary_ambiguity_table(
        ee_module,
        candidates,
        schema,
        page_size=page_size,
    )
    anchors = check_anchor_sites(
        ee_module,
        table,
        schema,
        config.earth_engine.hydrolakes_asset,
        state_collection,
        ambiguity_state_collection,
        config.selection.min_area_km2,
        page_size=page_size,
    )
    return DiscoveryResult(
        schema=schema,
        state_collection=state_collection,
        candidate_collection=candidates,
        table=table,
        summary=summarize_candidate_table(table),
        boundary_ambiguities=ambiguities,
        anchor_checks=anchors,
    )


def print_summary(result: DiscoveryResult, output_path: str | Path) -> None:
    """Print compact inventory, ambiguity, and anchor diagnostics."""

    summary = result.summary
    area = summary["area_km2"]
    print(f"Wrote candidate inventory: {output_path}")
    print(f"Candidates: {summary['total_candidates']}")
    print(
        "States: "
        + ", ".join(
            f"{state}={count}" for state, count in summary["counts_by_state"].items()
        )
    )
    print(
        "Lake_type: "
        + ", ".join(
            f"{lake_type}={count}"
            for lake_type, count in summary["counts_by_lake_type"].items()
        )
    )
    print(
        f"Names: {summary['named_count']} present, {summary['unnamed_count']} missing"
    )
    if summary["total_candidates"]:
        quantiles = ", ".join(
            f"{name}={value:.3f}" for name, value in area["quantiles"].items()
        )
        print(
            f"Lake_area km²: min={area['min']:.3f}, "
            f"median={area['median']:.3f}, max={area['max']:.3f}; {quantiles}"
        )
    print(
        "Optional schema fields absent: "
        + (", ".join(result.schema.missing_optional_properties) or "none")
    )
    missing_optional_values = {
        field: count
        for field, count in summary["missing_optional_values"].items()
        if count
    }
    print(
        "Missing optional values: "
        + (
            ", ".join(
                f"{field}={count}" for field, count in missing_optional_values.items()
            )
            or "none"
        )
    )
    sentinel_values = summary["hydrolakes_sentinel_values"]
    print(
        "HydroLAKES sentinel values: "
        + (
            ", ".join(
                f"{field}[{value}]={count}"
                for field, counts in sentinel_values.items()
                for value, count in counts.items()
            )
            or "none"
        )
    )
    print(f"State-boundary centroid ambiguities: {len(result.boundary_ambiguities)}")
    if not result.boundary_ambiguities.empty:
        print(result.boundary_ambiguities.to_string(index=False))

    for anchor_name, check in result.anchor_checks.items():
        print(f"Anchor — {anchor_name}: {check.method}")
        print(f"  {check.note}")
        if check.records.empty:
            print("  No candidate HydroLAKES records found.")
        else:
            print(check.records.to_string(index=False))


def _positive_cli_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without touching Earth Engine."""

    parser = argparse.ArgumentParser(
        description=(
            "Discover metadata-only HydroLAKES candidates whose centroids fall "
            "within the configured western states. No Landsat sampling or "
            "threshold calibration is performed."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML configuration path (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Candidate CSV path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--min-area-km2",
        type=_positive_cli_float,
        help=(
            "Override selection.min_area_km2 for this execution only; this is "
            "an experimental screening threshold."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.min_area_km2 is not None:
            config = config.with_min_area_km2(args.min_area_km2)

        ee_module, project = initialize_earth_engine()
        print(f"Earth Engine project: {project}")
        result = discover_candidate_inventory(ee_module, config)
        destination = write_candidate_csv(
            result.table,
            args.output,
            config.region.states,
        )
        print_summary(result, destination)
    except (ConfigError, DiscoveryError, CandidateTableError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
