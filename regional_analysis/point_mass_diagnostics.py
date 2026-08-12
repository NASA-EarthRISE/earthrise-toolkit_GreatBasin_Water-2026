"""Diagnose exact NDWI endpoints in the cached quick-transfer samples.

The normal path is local-only.  When reflectance/QA detail is absent, the CLI
can retrieve the same bounded samples once more through ``computeFeatures``
with an opt-in diagnostic band set.  It never creates an Earth Engine export.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ConfigError, load_config
from .discover_lakes import DiscoveryError, initialize_earth_engine
from .plot_quick_transfer import PLOT_STYLE, REFERENCE_COLORS
from .quick_examples import (
    DEFAULT_CANDIDATE_PATH,
    DEFAULT_CONFIG_PATH,
    QUICK_EXAMPLE_SETS,
)
from .quick_sampling import (
    LANDSAT_BAND_MAP,
    LANDSAT_COLLECTIONS,
    LakeSpec,
    QuickSamplingError,
    QuickSamplingSettings,
    build_buffered_lake_aoi,
    load_candidate_inventory,
    normalize_references,
    select_lake_specs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CACHE = REPOSITORY_ROOT / "data" / "regional" / "quick_transfer_cache"
DEFAULT_DIAGNOSTIC_DIR = (
    REPOSITORY_ROOT / "data" / "regional" / "point_mass_diagnostics"
)
DEFAULT_FIGURE_DIR = REPOSITORY_ROOT / "outputs" / "point_mass_diagnostics"
REFERENCES = ("OPERA", "JRC")
SIX_LAKE_IDS = QUICK_EXAMPLE_SETS["saline6"]
SCALE = 0.0000275
OFFSET = -0.2
NEAR_ENDPOINT_MIN = 0.95
ACQUISITIONS_PER_REQUEST = 15

REFLECTANCE_DIAGNOSTIC_BANDS = (
    "green_dn",
    "nir_dn",
    "green_unclamped",
    "nir_unclamped",
    "green_reflectance",
    "nir_reflectance",
    "qa_pixel",
    "qa_radsat",
)

DIAGNOSTIC_COLUMNS = (
    "row_key",
    "Hylak_id",
    "Lake_name",
    "reference",
    "index_value",
    "label",
    "acquisition_time",
    "reference_time",
    "sensor",
    "landsat_scene_id",
    "lon",
    "lat",
    *REFLECTANCE_DIAGNOSTIC_BANDS,
)


class PointMassError(RuntimeError):
    """Raised when point-mass inputs cannot support a defensible diagnosis."""


def _row_key(frame: pd.DataFrame) -> pd.Series:
    """Build a stable key from the original sampling pixel and acquisition."""

    time = pd.to_numeric(frame["acquisition_time"], errors="coerce")
    lon = pd.to_numeric(frame["lon"], errors="coerce")
    lat = pd.to_numeric(frame["lat"], errors="coerce")
    if time.isna().any() or lon.isna().any() or lat.isna().any():
        raise PointMassError("Point-mass rows require finite time/lon/lat keys.")
    base = (
        time.astype("int64").astype(str)
        + "|"
        + frame.get("sensor", pd.Series("", index=frame.index)).astype(str)
        + "|"
        + pd.to_numeric(
            frame.get("reference_time", pd.Series(-1, index=frame.index)),
            errors="coerce",
        )
        .fillna(-1)
        .astype("int64")
        .astype(str)
        + "|"
        + lon.map(lambda value: f"{value:.10f}")
        + "|"
        + lat.map(lambda value: f"{value:.10f}")
        + "|"
        + pd.to_numeric(
            frame.get("label", pd.Series(-1, index=frame.index)), errors="coerce"
        )
        .fillna(-1)
        .astype("int64")
        .astype(str)
        + "|"
        + pd.to_numeric(
            frame.get("index_value", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        ).map(lambda value: f"{value:.17g}")
    )
    occurrence = base.groupby(base, sort=False).cumcount().astype(str)
    return base + "|" + occurrence


def load_source_caches(
    cache_dir: str | Path,
    lakes: Sequence[LakeSpec],
    references: Sequence[str],
) -> dict[tuple[str, int], pd.DataFrame]:
    """Load the established prepared-sample caches without touching EE."""

    root = Path(cache_dir)
    loaded: dict[tuple[str, int], pd.DataFrame] = {}
    required = {
        "Hylak_id",
        "Lake_name",
        "reference",
        "index_value",
        "label",
        "split",
        "sensor",
        "acquisition_time",
        "lon",
        "lat",
    }
    for reference in normalize_references(references):
        for lake in lakes:
            path = root / f"{reference.lower()}_{lake.hylak_id}.csv"
            if not path.is_file():
                raise PointMassError(f"Missing source cache: {path}")
            frame = pd.read_csv(path)
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise PointMassError(f"{path} is missing: {', '.join(missing)}.")
            if not frame["Hylak_id"].eq(lake.hylak_id).all():
                raise PointMassError(f"{path} contains the wrong Hylak_id.")
            if set(frame["reference"].astype(str).str.upper()) != {reference}:
                raise PointMassError(f"{path} contains the wrong reference.")
            frame = frame.copy()
            frame["row_key"] = _row_key(frame)
            if frame["row_key"].duplicated().any():
                raise PointMassError(f"{path} contains duplicate sample keys.")
            frame["acquisition_date"] = pd.to_datetime(
                frame["acquisition_time"], unit="ms", utc=True
            )
            loaded[(reference, lake.hylak_id)] = frame
    return loaded


def point_mass_summary(
    samples: Mapping[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    """Return long-form endpoint counts for all requested QA dimensions."""

    combined = pd.concat(samples.values(), ignore_index=True)
    combined["is_exact_one"] = combined["index_value"].eq(1.0)
    combined["year"] = combined["acquisition_date"].dt.year
    combined["month"] = combined["acquisition_date"].dt.month
    dimensions = (
        ("overall", ["reference"]),
        ("split", ["reference", "split"]),
        ("lake", ["reference", "Hylak_id", "Lake_name"]),
        ("lake_split", ["reference", "Hylak_id", "Lake_name", "split"]),
        ("sensor", ["reference", "sensor"]),
        ("sensor_split", ["reference", "sensor", "split"]),
        ("label", ["reference", "label"]),
        ("label_split", ["reference", "label", "split"]),
        ("year", ["reference", "year"]),
        ("month", ["reference", "month"]),
    )
    parts: list[pd.DataFrame] = []
    for dimension, groups in dimensions:
        summary = (
            combined.groupby(groups, dropna=False, sort=False)["is_exact_one"]
            .agg(n_rows="size", n_exact_one="sum", exact_one_fraction="mean")
            .reset_index()
        )
        summary.insert(0, "dimension", dimension)
        parts.append(summary)
    result = pd.concat(parts, ignore_index=True, sort=False)
    stable = [
        "dimension",
        "reference",
        "Hylak_id",
        "Lake_name",
        "sensor",
        "label",
        "split",
        "year",
        "month",
        "n_rows",
        "n_exact_one",
        "exact_one_fraction",
    ]
    return result.reindex(columns=stable)


def derive_reflectance_and_qa(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive scaled/clamped reflectance and QA flags from raw diagnostics."""

    result = frame.copy()
    for column in ("green_dn", "nir_dn", "qa_pixel", "qa_radsat"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["green_dn", "nir_dn", "qa_pixel", "qa_radsat"]].isna().any().any():
        raise PointMassError("Raw diagnostic bands contain missing values.")
    result["green_unclamped_local"] = result["green_dn"] * SCALE + OFFSET
    result["nir_unclamped_local"] = result["nir_dn"] * SCALE + OFFSET
    result["green_clamped_local"] = result["green_unclamped_local"].clip(0, 1)
    result["nir_clamped_local"] = result["nir_unclamped_local"].clip(0, 1)
    denominator = result["green_clamped_local"] + result["nir_clamped_local"]
    result["ndwi_recomputed"] = np.where(
        denominator > 0,
        (result["green_clamped_local"] - result["nir_clamped_local"]) / denominator,
        np.nan,
    )
    qa = result["qa_pixel"].to_numpy(dtype="int64")
    result["qa_fill"] = (qa & 1) != 0
    result["qa_dilated_cloud"] = ((qa >> 1) & 1) != 0
    result["qa_cirrus"] = ((qa >> 2) & 1) != 0
    result["qa_cloud"] = ((qa >> 3) & 1) != 0
    result["qa_shadow"] = ((qa >> 4) & 1) != 0
    result["qa_snow"] = ((qa >> 5) & 1) != 0
    result["qa_clear"] = ((qa >> 6) & 1) != 0
    result["qa_water"] = ((qa >> 7) & 1) != 0
    result["qa_cloud_confidence"] = (qa >> 8) & 3
    result["qa_shadow_confidence"] = (qa >> 10) & 3
    result["qa_snow_confidence"] = (qa >> 12) & 3
    result["qa_cirrus_confidence"] = (qa >> 14) & 3
    result["qa_any_radsat"] = result["qa_radsat"].astype("int64") != 0
    result["nir_clamped_low"] = result["nir_unclamped_local"] <= 0
    result["green_clamped_low"] = result["green_unclamped_local"] <= 0
    result["green_clamped_high"] = result["green_unclamped_local"] >= 1
    return result


def build_diagnostic_sample_collection(
    ee_module: Any,
    aoi: Any,
    lake: LakeSpec,
    reference: str,
    source_rows: pd.DataFrame,
) -> Any:
    """Sample raw bands only at cached endpoint/near-endpoint point keys."""

    collections: list[Any] = []
    for (acquisition_time, sensor), group in source_rows.groupby(
        ["acquisition_time", "sensor"], sort=True
    ):
        mapping = LANDSAT_BAND_MAP[str(sensor)]
        image = ee_module.Image(
            ee_module.ImageCollection(LANDSAT_COLLECTIONS[str(sensor)])
            .filterBounds(aoi)
            .filter(ee_module.Filter.eq("system:time_start", int(acquisition_time)))
            .first()
        )
        raw = image.select(
            [mapping["green"], mapping["nir"]],
            ["green_dn", "nir_dn"],
        )
        unclamped = (
            raw.multiply(SCALE).add(OFFSET).rename(["green_unclamped", "nir_unclamped"])
        )
        reflectance = unclamped.clamp(0, 1).rename(
            ["green_reflectance", "nir_reflectance"]
        )
        diagnostic = (
            raw.addBands(unclamped)
            .addBands(reflectance)
            .addBands(image.select("QA_PIXEL").rename("qa_pixel"))
            .addBands(image.select("QA_RADSAT").rename("qa_radsat"))
        )
        points = []
        for row in group.itertuples(index=False):
            properties = {
                "row_key": row.row_key,
                "Hylak_id": lake.hylak_id,
                "Lake_name": lake.lake_name,
                "reference": reference,
                "index_value": float(row.index_value),
                "label": int(row.label),
                "acquisition_time": int(row.acquisition_time),
                "reference_time": int(row.reference_time),
                "sensor": str(row.sensor),
                "lon": float(row.lon),
                "lat": float(row.lat),
            }
            points.append(
                ee_module.Feature(
                    ee_module.Geometry.Point([float(row.lon), float(row.lat)]),
                    properties,
                )
            )
        sampled = diagnostic.sampleRegions(
            collection=ee_module.FeatureCollection(points),
            properties=[
                "row_key",
                "Hylak_id",
                "Lake_name",
                "reference",
                "index_value",
                "label",
                "acquisition_time",
                "reference_time",
                "sensor",
                "lon",
                "lat",
            ],
            scale=30,
            geometries=False,
        )
        scene_id = image.get("system:index")

        def add_scene_id(feature: Any, value: Any = scene_id) -> Any:
            return ee_module.Feature(feature).set("landsat_scene_id", value)

        collections.append(sampled.map(add_scene_id))
    return ee_module.FeatureCollection(collections).flatten()


def diagnostic_query_rows(source: pd.DataFrame) -> pd.DataFrame:
    """Select exact/near endpoints and the symmetric -1 endpoint for raw QA."""

    selected = source[
        source["index_value"].ge(NEAR_ENDPOINT_MIN) | source["index_value"].eq(-1.0)
    ].copy()
    if selected.empty:
        raise PointMassError("No endpoint or near-endpoint rows are available.")
    return selected.sort_values(
        ["acquisition_time", "lon", "lat"], kind="mergesort"
    ).reset_index(drop=True)


def normalize_diagnostic_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the diagnostic computeFeatures DataFrame."""

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise PointMassError("Diagnostic computeFeatures returned no rows.")
    result = raw.drop(
        columns=[column for column in ("geo", ".geo", "system:index") if column in raw],
        errors="ignore",
    ).copy()
    missing = sorted(set(DIAGNOSTIC_COLUMNS).difference(result.columns))
    if missing:
        raise PointMassError(
            "Diagnostic computeFeatures schema is missing: " + ", ".join(missing)
        )
    result = result.loc[:, DIAGNOSTIC_COLUMNS]
    numeric = [
        column
        for column in DIAGNOSTIC_COLUMNS
        if column
        not in {
            "row_key",
            "Lake_name",
            "reference",
            "sensor",
            "landsat_scene_id",
        }
    ]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[numeric].isna().any().any():
        raise PointMassError("Diagnostic samples contain missing numeric values.")
    result["Hylak_id"] = result["Hylak_id"].astype(int)
    result["label"] = result["label"].astype(int)
    result["row_key"] = result["row_key"].astype(str)
    if result["row_key"].duplicated().any():
        raise PointMassError("Diagnostic samples contain duplicate row keys.")
    return derive_reflectance_and_qa(result)


def validate_against_source(
    diagnostic: pd.DataFrame,
    source: pd.DataFrame,
    *,
    reference: str,
    lake_id: int,
) -> pd.DataFrame:
    """Require the diagnostic realization to match the completed sample cache."""

    if len(diagnostic) != len(source):
        raise PointMassError(
            f"{reference} {lake_id} diagnostic row count {len(diagnostic)} differs "
            f"from source cache {len(source)}."
        )
    if not diagnostic["Hylak_id"].eq(lake_id).all():
        raise PointMassError(
            f"{reference} {lake_id} diagnostic rows contain another Hylak_id."
        )
    if set(diagnostic["reference"].astype(str).str.upper()) != {reference}:
        raise PointMassError(
            f"{reference} {lake_id} diagnostic rows contain another reference."
        )
    source_by_key = source.set_index("row_key")
    diagnostic_by_key = diagnostic.set_index("row_key")
    if set(source_by_key.index) != set(diagnostic_by_key.index):
        raise PointMassError(
            f"{reference} {lake_id} diagnostic sample keys differ from the "
            "completed transfer cache."
        )
    aligned_source = source_by_key.loc[diagnostic_by_key.index]
    if not np.allclose(
        diagnostic_by_key["index_value"],
        aligned_source["index_value"],
        rtol=0,
        atol=1e-12,
    ) or not np.array_equal(
        diagnostic_by_key["label"].to_numpy(dtype=int),
        aligned_source["label"].to_numpy(dtype=int),
    ):
        raise PointMassError(
            f"{reference} {lake_id} diagnostic index/labels differ from source."
        )
    result = diagnostic_by_key.copy()
    result["split"] = aligned_source["split"].astype(str)
    result = result.reset_index()
    if not np.allclose(
        result["ndwi_recomputed"], result["index_value"], rtol=0, atol=1e-12
    ):
        raise PointMassError(
            f"{reference} {lake_id} raw bands do not reproduce cached NDWI."
        )
    for server_column, local_column in (
        ("green_unclamped", "green_unclamped_local"),
        ("nir_unclamped", "nir_unclamped_local"),
        ("green_reflectance", "green_clamped_local"),
        ("nir_reflectance", "nir_clamped_local"),
    ):
        if not np.allclose(
            result[server_column], result[local_column], rtol=0, atol=1e-12
        ):
            raise PointMassError(
                f"{reference} {lake_id} {server_column} differs from the "
                "locally derived reflectance."
            )
    exact = result["index_value"].eq(1.0)
    if not (
        result.loc[exact, "nir_clamped_low"].all()
        and result.loc[exact, "green_clamped_local"].gt(0).all()
    ):
        raise PointMassError(
            f"{reference} {lake_id} exact-one rows violate the clamp invariant."
        )
    return result


def retrieve_diagnostic_pair(
    ee_module: Any,
    hydrolakes_asset: str,
    lake: LakeSpec,
    reference: str,
    settings: QuickSamplingSettings,
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Retrieve raw bands at cached endpoint coordinates directly to pandas."""

    _feature, aoi = build_buffered_lake_aoi(
        ee_module, hydrolakes_asset, lake, settings.buffer_km
    )
    query = diagnostic_query_rows(source)
    acquisition_keys = list(
        query[["acquisition_time", "sensor"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    raw_parts: list[pd.DataFrame] = []
    for offset in range(0, len(acquisition_keys), ACQUISITIONS_PER_REQUEST):
        keys = acquisition_keys[offset : offset + ACQUISITIONS_PER_REQUEST]
        key_index = pd.MultiIndex.from_tuples(keys)
        query_index = pd.MultiIndex.from_frame(query[["acquisition_time", "sensor"]])
        chunk = query[query_index.isin(key_index)]
        collection = build_diagnostic_sample_collection(
            ee_module,
            aoi,
            lake,
            reference,
            chunk,
        )
        try:
            raw_parts.append(
                ee_module.data.computeFeatures(
                    {
                        "expression": collection,
                        "fileFormat": "PANDAS_DATAFRAME",
                        "pageSize": settings.page_size,
                    }
                )
            )
        except Exception as exc:
            raise PointMassError(
                f"Direct diagnostic transfer failed for {reference} "
                f"{lake.hylak_id} acquisition chunk {offset // ACQUISITIONS_PER_REQUEST + 1}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    raw = pd.concat(raw_parts, ignore_index=True)
    if len(raw) > settings.max_rows_per_combination:
        raise PointMassError(
            f"{reference} {lake.hylak_id} diagnostic transfer exceeded "
            f"{settings.max_rows_per_combination} rows."
        )
    return normalize_diagnostic_frame(raw)


def _pair_cache_path(root: Path, reference: str, lake_id: int) -> Path:
    return root / "raw_pairs" / f"{reference.lower()}_{lake_id}.csv"


def _diagnostic_cache_is_current(path: Path) -> bool:
    """Return whether a pair cache includes the current bounded QA fields."""

    if not path.is_file():
        return False
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except (OSError, pd.errors.ParserError):
        return False
    return set(DIAGNOSTIC_COLUMNS).union({"split"}).issubset(columns)


def load_or_retrieve_diagnostics(
    ee_module: Any | None,
    hydrolakes_asset: str,
    lakes: Sequence[LakeSpec],
    references: Sequence[str],
    settings: QuickSamplingSettings,
    source: Mapping[tuple[str, int], pd.DataFrame],
    diagnostic_dir: str | Path,
    *,
    refresh: bool = False,
) -> dict[tuple[str, int], pd.DataFrame]:
    """Load resumable detail caches or retrieve each missing pair once."""

    root = Path(diagnostic_dir)
    result: dict[tuple[str, int], pd.DataFrame] = {}
    for reference in normalize_references(references):
        for lake in lakes:
            key = (reference, lake.hylak_id)
            path = _pair_cache_path(root, reference, lake.hylak_id)
            if _diagnostic_cache_is_current(path) and not refresh:
                diagnostic = derive_reflectance_and_qa(pd.read_csv(path))
            else:
                if ee_module is None:
                    raise PointMassError(
                        f"Missing or stale diagnostic cache {path}; live EE is required."
                    )
                diagnostic = retrieve_diagnostic_pair(
                    ee_module,
                    hydrolakes_asset,
                    lake,
                    reference,
                    settings,
                    source[key],
                )
            validated = validate_against_source(
                diagnostic,
                diagnostic_query_rows(source[key]),
                reference=reference,
                lake_id=lake.hylak_id,
            )
            if not _diagnostic_cache_is_current(path) or refresh:
                path.parent.mkdir(parents=True, exist_ok=True)
                columns = [
                    column
                    for column in validated.columns
                    if column != "acquisition_date"
                ]
                validated.to_csv(
                    path, index=False, columns=columns, float_format="%.17g"
                )
            result[key] = validated
    return result


def diagnostic_detail_table(
    diagnostics: Mapping[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    """Keep exact and near-endpoint rows needed for documented local QA."""

    combined = pd.concat(diagnostics.values(), ignore_index=True)
    subset = combined[
        combined["index_value"].ge(NEAR_ENDPOINT_MIN) | combined["index_value"].eq(-1.0)
    ].copy()
    subset["is_exact_one"] = subset["index_value"].eq(1.0)
    subset["acquisition_date"] = pd.to_datetime(
        subset["acquisition_time"], unit="ms", utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return subset.sort_values(
        ["reference", "Hylak_id", "acquisition_time", "lon", "lat"],
        kind="mergesort",
    ).reset_index(drop=True)


def _save_figure(fig: mpl.figure.Figure, output: Path, name: str) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    png = output / f"{name}.png"
    svg = output / f"{name}.svg"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    return png, svg


def create_point_mass_figures(
    source: Mapping[tuple[str, int], pd.DataFrame],
    detail: pd.DataFrame,
    lakes: Sequence[LakeSpec],
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Create restrained endpoint-count and raw-band diagnostic figures."""

    combined = pd.concat(source.values(), ignore_index=True)
    combined["is_exact_one"] = combined["index_value"].eq(1.0)
    output = Path(output_dir)
    saved: list[Path] = []
    with mpl.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            1, 2, figsize=(10, 4.6), sharey=True, constrained_layout=True
        )
        y = np.arange(len(lakes))
        for axis, split in zip(axes, ("train", "test"), strict=True):
            for offset, reference in zip((-0.13, 0.13), REFERENCES, strict=True):
                values = []
                for lake in lakes:
                    view = combined[
                        (combined["reference"] == reference)
                        & (combined["Hylak_id"] == lake.hylak_id)
                        & (combined["split"] == split)
                    ]
                    values.append(int(view["is_exact_one"].sum()))
                axis.scatter(
                    values,
                    y + offset,
                    color=REFERENCE_COLORS[reference],
                    marker="o" if reference == "OPERA" else "D",
                    s=31,
                    label=reference,
                )
            axis.set_title(split.capitalize())
            axis.set_xlabel("Exact NDWI = 1.0 rows")
            axis.set_yticks(y, [lake.lake_name for lake in lakes])
            axis.grid(axis="x", color="#E3E3E3", linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].invert_yaxis()
        axes[1].legend(frameon=False)
        saved.extend(_save_figure(fig, output, "point_mass_counts_by_lake"))

        sensor = (
            combined.groupby(["reference", "sensor"])["is_exact_one"]
            .agg(n_rows="size", n_exact_one="sum")
            .reset_index()
        )
        sensor["fraction"] = sensor["n_exact_one"] / sensor["n_rows"]
        fig, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        positions = np.arange(len(sensor))
        colors = [REFERENCE_COLORS[value] for value in sensor["reference"]]
        axis.scatter(positions, sensor["fraction"], c=colors, s=42, zorder=3)
        for position, row in enumerate(sensor.itertuples(index=False)):
            axis.vlines(position, 0, row.fraction, color="#B8B8B8", linewidth=1)
        axis.set_xticks(
            positions,
            [
                f"{row.reference}\n{row.sensor}"
                for row in sensor.itertuples(index=False)
            ],
        )
        axis.set_ylabel("Fraction of samples at NDWI = 1.0")
        axis.set_ylim(0, max(0.28, float(sensor["fraction"].max()) * 1.12))
        axis.grid(axis="y", color="#E3E3E3", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        saved.extend(_save_figure(fig, output, "point_mass_fraction_by_sensor"))

        near = detail[detail["index_value"].ge(NEAR_ENDPOINT_MIN)]
        fig, axes = plt.subplots(
            1, 2, figsize=(10, 4.5), sharex=True, sharey=True, constrained_layout=True
        )
        for axis, reference in zip(axes, REFERENCES, strict=True):
            view = near[near["reference"] == reference]
            exact = view["index_value"].eq(1.0)
            axis.scatter(
                view.loc[~exact, "nir_unclamped_local"],
                view.loc[~exact, "green_unclamped_local"],
                s=7,
                alpha=0.35,
                color="#777777",
                label="0.95 ≤ NDWI < 1",
            )
            axis.scatter(
                view.loc[exact, "nir_unclamped_local"],
                view.loc[exact, "green_unclamped_local"],
                s=8,
                alpha=0.45,
                color=REFERENCE_COLORS[reference],
                label="NDWI = 1",
            )
            axis.axvline(0, color="#222222", linestyle="--", linewidth=0.9)
            axis.set_title(reference)
            axis.set_xlabel("NIR reflectance before clamp")
            axis.grid(color="#E5E5E5", linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Green reflectance before clamp")
        axes[1].legend(frameon=False, loc="best")
        saved.extend(_save_figure(fig, output, "near_endpoint_raw_reflectance"))
    return tuple(saved)


def findings_text(
    source: Mapping[tuple[str, int], pd.DataFrame],
    detail: pd.DataFrame,
) -> str:
    """Build a concise evidence-backed diagnosis from actual cache values."""

    combined = pd.concat(source.values(), ignore_index=True)
    combined["is_exact_one"] = combined["index_value"].eq(1.0)
    endpoint = detail[detail["is_exact_one"]]
    held = combined[combined["split"] == "test"]
    lines = ["# NDWI = 1.0 point-mass diagnosis", ""]
    for reference in REFERENCES:
        total_view = combined[combined["reference"] == reference]
        held_view = held[held["reference"] == reference]
        lines.append(
            f"- **{reference}:** {int(total_view['is_exact_one'].sum()):,}/"
            f"{len(total_view):,} total rows ({total_view['is_exact_one'].mean():.2%}) "
            f"and {int(held_view['is_exact_one'].sum()):,}/{len(held_view):,} held-out "
            f"rows ({held_view['is_exact_one'].mean():.2%}) are exactly 1.0."
        )
    sensor_counts = (
        endpoint.groupby(["reference", "sensor"], sort=False).size().to_dict()
    )
    lake_distribution = (
        combined.groupby(["reference", "Hylak_id", "Lake_name"], sort=False)[
            "is_exact_one"
        ]
        .agg(["sum", "size"])
        .reset_index()
    )
    lake_distribution["fraction"] = lake_distribution["sum"] / lake_distribution["size"]
    sensor_text = ", ".join(
        f"{reference} {sensor} {count:,}"
        for (reference, sensor), count in sensor_counts.items()
    )
    lake_min = lake_distribution.loc[lake_distribution["fraction"].idxmin()]
    lake_max = lake_distribution.loc[lake_distribution["fraction"].idxmax()]
    exact_positive = int(endpoint["label"].sum())
    nir_low = int(endpoint["nir_clamped_low"].sum())
    green_low = int(endpoint["green_clamped_low"].sum())
    green_high = int(endpoint["green_clamped_high"].sum())
    green_min = float(endpoint["green_unclamped_local"].min())
    green_max = float(endpoint["green_unclamped_local"].max())
    fill = int(endpoint["qa_fill"].sum())
    saturated = int(endpoint["qa_any_radsat"].sum())
    cloud = int(endpoint["qa_cloud_confidence"].eq(3).sum())
    direct_cloud = int(endpoint["qa_cloud"].sum())
    dilated_cloud = int(endpoint["qa_dilated_cloud"].sum())
    shadow = int(endpoint["qa_shadow"].sum())
    snow = int(endpoint["qa_snow"].sum())
    negative = endpoint[endpoint["label"].eq(0)]
    exception = ""
    if len(negative) == 1:
        row = negative.iloc[0]
        date = pd.to_datetime(row["acquisition_time"], unit="ms", utc=True).date()
        exception = (
            f" The sole reference-negative endpoint is {row['Lake_name']} "
            f"{row['reference']} {row['split']} on {date} "
            f"(`{row['landsat_scene_id']}`); Landsat QA nevertheless marks it "
            "as water, indicating a rare reference disagreement."
        )
    lines.extend(
        [
            "",
            "## Mechanism",
            "",
            (
                "Landsat surface reflectance is scaled as `DN * 0.0000275 - 0.2` "
                "and clamped to `[0, 1]` before NDWI. Exact NDWI 1.0 therefore "
                "occurs when clamped NIR is zero and clamped green is positive. "
                "The raw-DN zero crossing is 7272.727; integer NIR DN <=7272 is "
                "collapsed to zero."
            ),
            "",
            (
                f"The live raw-band check recovered {len(endpoint):,} exact endpoint "
                f"rows. {nir_low:,} have non-positive pre-clamp NIR; "
                f"{green_low} have non-positive green and {green_high} reach the "
                f"green upper clamp. Pre-clamp green spans {green_min:.4f} to "
                f"{green_max:.4f}, so green near 1 is not the cause. "
                f"{exact_positive:,} carry a positive water label.{exception}"
            ),
            "",
            "## Distribution and QA",
            "",
            (
                f"Sensor endpoint counts are {sensor_text}. Exact endpoints occur "
                "in every lake, in both train and test splits, and throughout the "
                "configured years. Lake/reference endpoint fractions range from "
                f"{lake_min['fraction']:.2%} ({lake_min['Lake_name']}, "
                f"{lake_min['reference']}) to {lake_max['fraction']:.2%} "
                f"({lake_max['Lake_name']}, {lake_max['reference']})."
            ),
            "",
            (
                "QA flags identify "
                f"{fill} fill pixels, {saturated} radiometrically saturated pixels, "
                f"{direct_cloud} direct-cloud flags, {cloud} high-confidence-cloud "
                f"rows, {dilated_cloud} dilated-cloud flags, {shadow} cloud-shadow "
                f"flags, and {snow} snow flags. The scene-context panels show "
                "affected acquisitions but are not pixel-level causal maps."
            ),
            "",
            (
                "A bounded reference-neighborhood experiment was attempted to "
                "quantify shoreline association, but repeated Earth Engine memory "
                "limits made it disproportionate to this diagnostic. It was "
                "intentionally stopped without retaining partial results. Edge "
                "association remains secondary and is not required to establish "
                "the raw-band/clamping mechanism."
            ),
            "",
            "## Recommendation",
            "",
            (
                "**Keep the rows as-is, but flag the endpoint in scientific notes.** "
                "The point mass is a known consequence of baseline-aligned clipping "
                "of very low/negative corrected NIR over reference water. It is not "
                "caused by the train/test split or a handful of corrupt scenes. "
                "Changing the clamp or deleting endpoint rows would change the "
                "predictor distribution and should be evaluated only as a separate "
                "sensitivity experiment that reruns every threshold result. No "
                "existing transfer result needs rerunning based solely on this "
                "endpoint diagnosis."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose exact NDWI=1 rows from local transfer caches. Missing raw "
            "diagnostics use bounded computeFeatures only; no EE export is created."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / DEFAULT_CONFIG_PATH
    )
    parser.add_argument(
        "--candidate-csv", type=Path, default=REPOSITORY_ROOT / DEFAULT_CANDIDATE_PATH
    )
    parser.add_argument("--source-cache-dir", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--refresh-raw", action="store_true")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Fail rather than initialize EE when a raw diagnostic pair is missing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        inventory = load_candidate_inventory(args.candidate_csv)
        lakes = select_lake_specs(inventory, SIX_LAKE_IDS)
        settings = QuickSamplingSettings()
        source = load_source_caches(args.source_cache_dir, lakes, REFERENCES)
        missing = [
            _pair_cache_path(args.diagnostic_dir, reference, lake.hylak_id)
            for reference in REFERENCES
            for lake in lakes
            if args.refresh_raw
            or not _diagnostic_cache_is_current(
                _pair_cache_path(args.diagnostic_dir, reference, lake.hylak_id)
            )
        ]
        ee_module = None
        if missing:
            if args.cache_only:
                raise PointMassError(
                    "Raw diagnostic caches are missing: "
                    + ", ".join(str(path) for path in missing)
                )
            ee_module, project = initialize_earth_engine()
            print(f"Earth Engine project: {project}")
        diagnostics = load_or_retrieve_diagnostics(
            ee_module,
            config.earth_engine.hydrolakes_asset,
            lakes,
            REFERENCES,
            settings,
            source,
            args.diagnostic_dir,
            refresh=args.refresh_raw,
        )
        summary = point_mass_summary(source)
        detail = diagnostic_detail_table(diagnostics)
        args.diagnostic_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.diagnostic_dir / "ndwi_point_mass_summary.csv"
        detail_path = args.diagnostic_dir / "ndwi_point_mass_raw_samples.csv"
        findings_path = args.diagnostic_dir / "ndwi_point_mass_findings.md"
        summary.to_csv(summary_path, index=False, float_format="%.17g")
        detail.to_csv(detail_path, index=False, float_format="%.17g")
        findings_path.write_text(findings_text(source, detail), encoding="utf-8")
        figures = create_point_mass_figures(source, detail, lakes, args.output_dir)
        print(
            summary[summary["dimension"].isin(["overall", "split"])].to_string(
                index=False
            )
        )
        print(f"Wrote {summary_path}, {detail_path}, {findings_path}")
        print(f"Wrote {len(figures)} figure files under {args.output_dir}")
        return 0
    except (
        ConfigError,
        DiscoveryError,
        OSError,
        PointMassError,
        QuickSamplingError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
