"""Validation-gated Bear Lake topobathymetry and spatial prototype.

Phase 2 deliberately has a narrow scope: it combines the official Bear Lake
USGS topobathymetric DEM with the already-normalized Phase 1 NAVD88 stage and
elevation--area--volume relation.  Network acquisition, raster calculations,
and rendering are separate CLI actions so the 1.34 GB source archive is never
downloaded as an incidental side effect of metadata inspection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import yaml

from .binary_download import (
    BinaryDownloadError,
    BinaryDownloadRequest,
    StreamingBinaryDownloader,
    calculate_disk_requirements,
    inspect_zip_archive,
    inventory_records,
    metadata_file_records,
    require_sufficient_disk_space,
    safe_extract_selected,
    select_archive_raster,
    select_bear_lake_phase2_files,
    write_extraction_manifest,
)
from .cache import CacheMode
from .manager_metrics import MetricObservation, calculate_change_metric
from .models import HeightType, HydrometryVariable
from .phase2_metadata import (
    Phase2MetadataError,
    inspect_spatial_coverage_bundle,
    read_previous_live_run,
    write_metadata_audit_markdown,
)
from .registry import load_registry
from .sciencebase import (
    BEAR_LAKE_TITLE,
    BearLakeEavSource,
    ScienceBaseFile,
    acquire_bear_lake_eav,
)
from .stage_capacity import StageCapacityInterpolator
from .units import convert_length

PHASE2_SCHEMA_VERSION = "hydrometry-bear-lake-phase2-v2"
DEFAULT_CONFIG = Path("configs/hydrometry_pilots.yaml")
DEFAULT_PHASE1_DIR = Path("data/hydrometry/phase1")
DEFAULT_REFERENCE_DIR = Path("data/hydrometry/reference")
DEFAULT_OUTPUT_DIR = Path("data/hydrometry/phase2/bear_lake")
DEFAULT_FIGURE_DIR = Path("outputs/hydrometry_phase2/bear_lake")
PHASE2_GDAL_CACHEMAX_BYTES = 128 * 1024**2
PHASE2_MEMORY_ESTIMATE_SCOPE = (
    "Algorithmic NumPy working-array estimate only; excludes the bounded GDAL "
    "block cache, Python/runtime memory, and operating-system file cache."
)

PHASE2_SCIENTIFIC_HASH_CONTRACT = "phase2-scientific-content-sha256-v2"
# These fields are execution telemetry, not scientific content.  Their raw bytes
# remain protected separately by ``quantitative_artifact_sha256``.  Keep this
# allowlist artifact-specific so a newly introduced field is never silently
# removed from the scientific replay contract.
_SCIENTIFIC_JSON_RUNTIME_EXCLUSIONS: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "source_raster_inspection.json": (("inspection_seconds",),),
    "source/extraction_manifest.json": (("created_at_utc",),),
    "eav_raster_validation_summary.json": (
        ("source_processing_seconds",),
        ("total_validation_seconds",),
    ),
}
_SCIENTIFIC_CSV_RUNTIME_EXCLUSIONS: Mapping[str, tuple[str, ...]] = {
    "source_stage_scalars.csv": ("processing_seconds",),
    "working_stage_scalars.csv": ("processing_seconds",),
    "resolution_benchmark.csv": ("processing_seconds",),
}

SOURCE_METHOD_CONSISTENCY_RULE = (
    "at every source validation stage, the equal-area-radius displacement "
    "between official and raster area must not exceed the native source-cell "
    "diagonal and storage must remain inside its stage-specific numeric "
    "precision envelope"
)
SOURCE_METHOD_CONSISTENCY_WARNING = (
    "Equal-area-radius displacement is a scalar area-equivalent consistency "
    "proxy; it does not bound maximum local shoreline-position error."
)


class Phase2Error(RuntimeError):
    """Raised when a Phase 2 scientific, source, or execution gate fails."""


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Phase2Error(f"{label} must be a non-empty string.")
    return value.strip()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Phase2Error(f"{label} must be a positive integer.")
    return value


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise Phase2Error(f"{label} must be positive.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Phase2Error(f"{label} must be positive.") from exc
    if not math.isfinite(result) or result <= 0:
        raise Phase2Error(f"{label} must be positive.")
    return result


def _validated_vertical_reference(record: Mapping[str, Any]) -> tuple[str, str]:
    """Require the structured official metadata to establish NAVD88 metres."""

    datum = _required_text(record.get("vertical_datum"), "FGDC vertical datum")
    units = _required_text(record.get("vertical_units"), "FGDC vertical units")
    datum_token = "".join(
        character for character in datum.upper() if character.isalnum()
    )
    unit_token = units.strip().casefold().replace("_", " ")
    if not (
        "NAVD88" in datum_token
        or "NORTHAMERICANVERTICALDATUMOF1988" in datum_token
        or "NORTHAMERICANVERTICALDATUM1988" in datum_token
    ):
        raise Phase2Error("Official FGDC vertical metadata does not establish NAVD88.")
    if unit_token not in {"m", "meter", "meters", "metre", "metres"}:
        raise Phase2Error(
            "Official FGDC vertical metadata does not use metre elevation units."
        )
    return datum, units


@dataclass(frozen=True, slots=True)
class TrustedSeed:
    """Curated, explicit interior seed used by connected inundation."""

    crs: str
    longitude: float
    latitude: float
    description: str

    def __post_init__(self) -> None:
        _required_text(self.crs, "phase2 trusted_seed.crs")
        if not (-180.0 <= self.longitude <= 180.0):
            raise Phase2Error("trusted_seed.longitude is outside longitude bounds.")
        if not (-90.0 <= self.latitude <= 90.0):
            raise Phase2Error("trusted_seed.latitude is outside latitude bounds.")
        _required_text(self.description, "phase2 trusted_seed.description")


@dataclass(frozen=True, slots=True)
class Phase2Settings:
    """Strict Bear Lake Phase 2 settings loaded from the shared registry YAML."""

    config_path: Path
    profile_id: str
    sciencebase_item_id: str
    raster_archive_name: str
    spatial_metadata_archive_name: str
    fgdc_metadata_name: str
    thumbnail_name: str
    candidate_spatial_resolutions_m: tuple[float, ...]
    validation_grid_steps: int
    connectivity: int
    selected_state_count: int
    animation_frame_count: int
    trusted_seed: TrustedSeed
    quantitative_resampling: str
    comparison_resampling: str
    minimum_valid_fraction: float
    rendering_resolution_m: float
    interactive_html_enabled: bool
    interactive_html_maximum_file_size_mb: float
    data_dir: Path
    figure_dir: Path
    phase1_history_start: date
    phase1_analysis_end: date
    baseline_match_tolerance_days: int
    doi: str


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Phase2Error(f"{label} must be a mapping.")
    return value


def load_phase2_settings(path: str | Path = DEFAULT_CONFIG) -> Phase2Settings:
    """Load and cross-check the narrow Phase 2 config and Bear registry profile."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Phase2Error(
            f"Could not load Phase 2 config {config_path}: {exc}"
        ) from exc
    root = _mapping(raw, "hydrometry config")
    phase2 = _mapping(root.get("phase2"), "phase2")
    bear = _mapping(phase2.get("bear_lake"), "phase2.bear_lake")
    phase1 = _mapping(root.get("phase1"), "phase1")
    outputs = _mapping(bear.get("outputs"), "phase2.bear_lake.outputs")
    seed = _mapping(bear.get("trusted_seed"), "phase2.bear_lake.trusted_seed")
    resampling = _mapping(bear.get("resampling"), "phase2.bear_lake.resampling")
    interactive = _mapping(
        bear.get("interactive_html"), "phase2.bear_lake.interactive_html"
    )
    resolutions = bear.get("candidate_spatial_resolutions_m")
    if not isinstance(resolutions, list) or not resolutions:
        raise Phase2Error("candidate_spatial_resolutions_m must be a non-empty list.")
    parsed_resolutions = tuple(
        _positive_float(value, "candidate_spatial_resolutions_m")
        for value in resolutions
    )
    if tuple(sorted(set(parsed_resolutions))) != parsed_resolutions:
        raise Phase2Error(
            "candidate_spatial_resolutions_m must be unique and increasing."
        )
    connectivity = _positive_int(bear.get("connectivity"), "phase2 connectivity")
    if connectivity not in {4, 8}:
        raise Phase2Error("phase2 connectivity must be 4 or 8.")
    registry = load_registry(config_path)
    profile = registry.get_profile(
        _required_text(bear.get("profile_id"), "profile_id"), strict=True
    )
    if profile is None:  # pragma: no cover - strict lookup raises
        raise Phase2Error("Configured Bear Lake hydrometry profile is missing.")
    bathymetry = [
        source
        for source in profile.bathymetry_sources
        if source.catalog_id == "usgs_bear_lake_eav_tbdem"
    ]
    if len(bathymetry) != 1 or not bathymetry[0].doi:
        raise Phase2Error("Bear Lake Phase 2 requires one configured official DOI.")
    item_id = _required_text(bear.get("sciencebase_item_id"), "sciencebase_item_id")
    if item_id != "698226f4b66b018d00102e42":
        raise Phase2Error(
            "Configured Bear Lake ScienceBase item identity is unexpected."
        )
    minimum_valid_fraction = _positive_float(
        resampling.get("minimum_valid_fraction"),
        "resampling.minimum_valid_fraction",
    )
    if minimum_valid_fraction > 1:
        raise Phase2Error("resampling.minimum_valid_fraction cannot exceed 1.")
    return Phase2Settings(
        config_path=config_path,
        profile_id=profile.profile_id,
        sciencebase_item_id=item_id,
        raster_archive_name=_required_text(
            bear.get("raster_archive_name"), "raster_archive_name"
        ),
        spatial_metadata_archive_name=_required_text(
            bear.get("spatial_metadata_archive_name"),
            "spatial_metadata_archive_name",
        ),
        fgdc_metadata_name=_required_text(
            bear.get("fgdc_metadata_name"), "fgdc_metadata_name"
        ),
        thumbnail_name=_required_text(bear.get("thumbnail_name"), "thumbnail_name"),
        candidate_spatial_resolutions_m=parsed_resolutions,
        validation_grid_steps=_positive_int(
            bear.get("validation_grid_steps"), "validation_grid_steps"
        ),
        connectivity=connectivity,
        selected_state_count=_positive_int(
            bear.get("selected_state_count"), "selected_state_count"
        ),
        animation_frame_count=_positive_int(
            bear.get("animation_frame_count"), "animation_frame_count"
        ),
        trusted_seed=TrustedSeed(
            crs=_required_text(seed.get("crs"), "trusted_seed.crs"),
            longitude=float(seed.get("longitude")),
            latitude=float(seed.get("latitude")),
            description=_required_text(
                seed.get("description"), "trusted_seed.description"
            ),
        ),
        quantitative_resampling=_required_text(
            resampling.get("quantitative"), "resampling.quantitative"
        ),
        comparison_resampling=_required_text(
            resampling.get("comparison"), "resampling.comparison"
        ),
        minimum_valid_fraction=minimum_valid_fraction,
        rendering_resolution_m=_positive_float(
            resampling.get("rendering_resolution_m"),
            "resampling.rendering_resolution_m",
        ),
        interactive_html_enabled=interactive.get("enabled") is True,
        interactive_html_maximum_file_size_mb=_positive_float(
            interactive.get("maximum_file_size_mb"),
            "interactive_html.maximum_file_size_mb",
        ),
        data_dir=Path(_required_text(outputs.get("data_dir"), "outputs.data_dir")),
        figure_dir=Path(
            _required_text(outputs.get("figure_dir"), "outputs.figure_dir")
        ),
        phase1_history_start=date.fromisoformat(
            _required_text(phase1.get("history_start"), "phase1.history_start")
        ),
        phase1_analysis_end=date.fromisoformat(
            _required_text(phase1.get("analysis_end"), "phase1.analysis_end")
        ),
        baseline_match_tolerance_days=_positive_int(
            phase1.get("baseline_match_tolerance_days"),
            "phase1.baseline_match_tolerance_days",
        ),
        doi=bathymetry[0].doi,
    )


@dataclass(frozen=True, slots=True)
class SelectedStage:
    """One real Bear Lake stage observation selected for a Phase 2 product."""

    state_id: str
    observation_date: date
    stage_ft_navd88: float
    stage_m_navd88: float
    source_record_id: str
    source_parameter_code: str
    approval_status: str | None
    provisional: bool | None
    quality_status: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["observation_date"] = self.observation_date.isoformat()
        return result


def _clean_optional(value: Any) -> Any:
    return None if pd.isna(value) else value


def load_bear_stage_observations(
    phase1_dir: str | Path = DEFAULT_PHASE1_DIR,
) -> pd.DataFrame:
    """Load the authoritative Phase 1 Bear Lake observed-stage rows only."""

    source = Path(phase1_dir) / "pilot_daily_hydrometry.csv"
    if not source.is_file():
        raise Phase2Error(f"Required Phase 1 daily table is missing: {source}")
    frame = pd.read_csv(source, low_memory=False)
    required = {
        "profile_id",
        "metric",
        "method",
        "observation_type",
        "calendar_date",
        "value",
        "unit",
        "vertical_datum",
        "height_type",
        "source_record_id",
        "source_parameter_code",
        "approval_status",
        "provisional",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise Phase2Error(f"Phase 1 daily table lacks columns: {', '.join(missing)}")
    stage = frame.loc[
        (frame["profile_id"] == "bear_lake")
        & (frame["metric"] == "water_elevation")
        & (frame["method"] == "agency_observed")
        & (frame["observation_type"] == "observed")
    ].copy()
    if stage.empty:
        raise Phase2Error("Phase 1 has no Bear Lake observed stage records.")
    if set(stage["unit"].dropna().astype(str)) != {"ft"}:
        raise Phase2Error("Bear Lake Phase 1 stage must be in feet.")
    if set(stage["vertical_datum"].dropna().astype(str)) != {"NAVD88"}:
        raise Phase2Error("Bear Lake Phase 1 stage must explicitly use NAVD88.")
    if stage["calendar_date"].duplicated().any():
        raise Phase2Error("Bear Lake Phase 1 stage has duplicate calendar dates.")
    stage["calendar_date"] = pd.to_datetime(
        stage["calendar_date"], errors="raise"
    ).dt.date
    stage["value"] = pd.to_numeric(stage["value"], errors="raise")
    if not np.isfinite(stage["value"].to_numpy(dtype=float)).all():
        raise Phase2Error("Bear Lake Phase 1 stage contains non-finite values.")
    return stage.sort_values("calendar_date", kind="mergesort").reset_index(drop=True)


def _stage_from_row(state_id: str, row: pd.Series) -> SelectedStage:
    provisional = _clean_optional(row["provisional"])
    if provisional is not None:
        provisional = bool(provisional)
    return SelectedStage(
        state_id=state_id,
        observation_date=row["calendar_date"],
        stage_ft_navd88=float(row["value"]),
        stage_m_navd88=convert_length(float(row["value"]), "ft", "m"),
        source_record_id=str(row["source_record_id"]),
        source_parameter_code=str(row["source_parameter_code"]),
        approval_status=(
            None
            if _clean_optional(row["approval_status"]) is None
            else str(row["approval_status"])
        ),
        provisional=provisional,
        quality_status=(
            None
            if "quality_status" not in row
            or _clean_optional(row["quality_status"]) is None
            else str(row["quality_status"])
        ),
    )


def select_stage_states(
    stage: pd.DataFrame,
    *,
    baseline_match_tolerance_days: int = 3,
) -> tuple[SelectedStage, ...]:
    """Select five deterministic states without synthesizing observations."""

    ordered = stage.sort_values("calendar_date", kind="mergesort").reset_index(
        drop=True
    )
    latest = ordered.iloc[-1]
    observations = tuple(
        MetricObservation(
            timestamp=row.calendar_date,
            value=float(row.value),
            metric="water_elevation",
            unit="ft",
            method="agency_observed",
            provisional=(None if pd.isna(row.provisional) else bool(row.provisional)),
            approval_status=(
                None if pd.isna(row.approval_status) else str(row.approval_status)
            ),
        )
        for row in ordered.itertuples(index=False)
    )
    year_change = calculate_change_metric(
        observations,
        requested_window_days=365,
        analysis_end=latest["calendar_date"],
        baseline_match_tolerance_days=baseline_match_tolerance_days,
    )
    if year_change.baseline_actual_date is None:
        raise Phase2Error(
            "No real Bear Lake observation matches the one-year baseline."
        )
    year_ago = ordered.loc[
        ordered["calendar_date"] == year_change.baseline_actual_date
    ].iloc[0]
    recent_low = ordered.loc[ordered["value"].idxmin()]
    recent_high = ordered.loc[ordered["value"].idxmax()]
    median_value = float(ordered["value"].median())
    median_row = ordered.iloc[
        min(
            range(len(ordered)),
            key=lambda index: (
                abs(float(ordered.iloc[index]["value"]) - median_value),
                ordered.iloc[index]["calendar_date"],
            ),
        )
    ]
    rows = (
        _stage_from_row("latest", latest),
        _stage_from_row("year_ago", year_ago),
        _stage_from_row("recent_low", recent_low),
        _stage_from_row("recent_high", recent_high),
        _stage_from_row("historical_median", median_row),
    )
    if len({row.observation_date for row in rows}) < 3:
        raise Phase2Error("Selected stage states do not span enough real observations.")
    return rows


def select_animation_stages(
    stage: pd.DataFrame, *, frame_count: int
) -> tuple[SelectedStage, ...]:
    """Select deterministic chronological real observations including key states."""

    if frame_count < 3:
        raise Phase2Error("Animation requires at least three frames.")
    ordered = stage.sort_values("calendar_date", kind="mergesort").reset_index(
        drop=True
    )
    target_indices = np.linspace(0, len(ordered) - 1, frame_count).round().astype(int)
    required_indices = {
        int(ordered["value"].idxmin()),
        int(ordered["value"].idxmax()),
        len(ordered) - 1,
    }
    chosen = {int(value) for value in target_indices} | required_indices
    while len(chosen) > frame_count:
        removable = sorted(chosen.difference(required_indices))
        if not removable:
            break
        # Remove the point with the smallest chronological separation.
        remove = min(
            removable,
            key=lambda value: min(
                abs(value - other) for other in chosen if other != value
            ),
        )
        chosen.remove(remove)
    while len(chosen) < frame_count:
        candidates = [index for index in range(len(ordered)) if index not in chosen]
        add = max(
            candidates,
            key=lambda value: min(abs(value - other) for other in chosen),
        )
        chosen.add(add)
    return tuple(
        _stage_from_row(f"frame_{number:02d}", ordered.iloc[index])
        for number, index in enumerate(sorted(chosen), start=1)
    )


def load_bear_eav_source(
    settings: Phase2Settings,
    *,
    reference_dir: str | Path = DEFAULT_REFERENCE_DIR,
) -> BearLakeEavSource:
    """Replay and hash-validate the authoritative Phase 1 E-A-V reference."""

    # Offline replay deliberately prevents Phase 2 from redownloading or
    # reparsing a source already made authoritative by Phase 1.
    from .http import HttpClient

    source = acquire_bear_lake_eav(
        client=HttpClient(),
        mode=CacheMode.OFFLINE,
        reference_dir=reference_dir,
        doi=settings.doi,
        expected_title=BEAR_LAKE_TITLE,
        refresh=False,
    )
    if source.item.item_id != settings.sciencebase_item_id:
        raise Phase2Error(
            "The replayed ScienceBase item does not match the configured "
            f"item ID {settings.sciencebase_item_id!r}."
        )
    return source


def official_stage_values(
    eav_source: BearLakeEavSource, states: Sequence[SelectedStage]
) -> pd.DataFrame:
    """Evaluate Phase 1's unchanged official interpolation for selected states."""

    interpolator = StageCapacityInterpolator(eav_source.table)
    records: list[dict[str, Any]] = []
    for state in states:
        result = interpolator.derive(
            state.stage_ft_navd88,
            input_unit="ft",
            input_vertical_datum="NAVD88",
            height_type=HeightType.ORTHOMETRIC_HEIGHT,
            stage_variable=HydrometryVariable.WATER_LEVEL,
        )
        if not result.within_supported_range:
            raise Phase2Error(
                f"Selected state {state.state_id} is outside the official E-A-V range."
            )
        records.append(
            {
                **state.as_dict(),
                "date": state.observation_date.isoformat(),
                "stage_original_ft": state.stage_ft_navd88,
                "source": "USGS-10055000 observed NAVD88 water elevation",
                "official_table_area_m2": result.derived_area_m2,
                "official_table_storage_m3": result.derived_storage_m3,
                "official_table_source": result.capacity_table_source,
                "official_table_hash": result.capacity_table_hash,
            }
        )
    return pd.DataFrame.from_records(records)


def validation_stages(
    eav_source: BearLakeEavSource, selected_states: Sequence[SelectedStage]
) -> pd.DataFrame:
    """Build the required table-quantile and real-observation validation stages."""

    table = eav_source.table.to_dataframe()
    elevation = table["elevation_m_navd88"].to_numpy(dtype=float)
    indexes = {
        "safe_near_minimum": max(1, round(0.01 * (len(table) - 1))),
        "table_q10": round(0.10 * (len(table) - 1)),
        "table_q25": round(0.25 * (len(table) - 1)),
        "table_median": round(0.50 * (len(table) - 1)),
        "table_q75": round(0.75 * (len(table) - 1)),
        "table_q90": round(0.90 * (len(table) - 1)),
    }
    rows: list[dict[str, Any]] = []
    interpolator = StageCapacityInterpolator(eav_source.table)
    for name, index in indexes.items():
        stage_m = float(elevation[index])
        row = table.iloc[index]
        rows.append(
            {
                "validation_stage_id": name,
                "stage_m_navd88": stage_m,
                "stage_ft_navd88": convert_length(stage_m, "m", "ft"),
                "stage_date": None,
                "official_area_m2": float(row["area_m2"]),
                "official_storage_m3": float(row["storage_m3"]),
            }
        )
    required = {"recent_low", "recent_high", "latest"}
    for state in selected_states:
        if state.state_id not in required:
            continue
        result = interpolator.derive(
            state.stage_m_navd88,
            input_unit="m",
            input_vertical_datum="NAVD88",
            height_type=HeightType.ORTHOMETRIC_HEIGHT,
            stage_variable=HydrometryVariable.WATER_LEVEL,
        )
        rows.append(
            {
                "validation_stage_id": f"observed_{state.state_id}",
                "stage_m_navd88": state.stage_m_navd88,
                "stage_ft_navd88": state.stage_ft_navd88,
                "stage_date": state.observation_date.isoformat(),
                "official_area_m2": result.derived_area_m2,
                "official_storage_m3": result.derived_storage_m3,
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if frame["stage_m_navd88"].duplicated().any():
        frame = frame.drop_duplicates("stage_m_navd88", keep="last")
    return frame.sort_values("stage_m_navd88", kind="mergesort").reset_index(drop=True)


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _scientific_artifact_key(path: Path, root: Path) -> str | None:
    """Return one platform-independent relative key for hash canonicalization."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _canonical_json_without_paths(
    path: Path, excluded_paths: Sequence[tuple[str, ...]]
) -> bytes:
    """Canonicalize JSON after removing only explicitly named runtime fields."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase2Error(
            f"Could not canonicalize scientific JSON {path}: {exc}"
        ) from exc
    for excluded_path in excluded_paths:
        if not excluded_path:
            raise Phase2Error("Scientific JSON exclusions cannot be empty.")
        parent: Any = value
        for component in excluded_path[:-1]:
            if not isinstance(parent, Mapping) or component not in parent:
                raise Phase2Error(
                    "Scientific JSON runtime field is missing: "
                    + "/".join(excluded_path)
                )
            parent = parent[component]
        leaf = excluded_path[-1]
        if not isinstance(parent, dict) or leaf not in parent:
            raise Phase2Error(
                "Scientific JSON runtime field is missing: " + "/".join(excluded_path)
            )
        del parent[leaf]
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_csv_without_columns(
    path: Path, excluded_columns: Sequence[str]
) -> bytes:
    """Canonicalize CSV rows after explicit runtime-column removal."""

    try:
        rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
    except (OSError, csv.Error) as exc:
        raise Phase2Error(
            f"Could not canonicalize scientific CSV {path}: {exc}"
        ) from exc
    if not rows:
        raise Phase2Error(f"Scientific CSV is empty: {path}")
    header = rows[0]
    if len(header) != len(set(header)):
        raise Phase2Error(f"Scientific CSV has duplicate columns: {path}")
    missing = sorted(set(excluded_columns).difference(header))
    if missing:
        raise Phase2Error(
            f"Scientific CSV lacks declared runtime columns: {', '.join(missing)}"
        )
    retained_indexes = [
        index for index, name in enumerate(header) if name not in excluded_columns
    ]
    canonical_rows: list[list[str]] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise Phase2Error(
                f"Scientific CSV row {row_number} has {len(row)} fields; "
                f"expected {len(header)}: {path}"
            )
        canonical_rows.append([row[index] for index in retained_indexes])
    payload = {
        "columns": [header[index] for index in retained_indexes],
        "rows": canonical_rows,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _scientific_content_sha256(path: Path, *, root: Path) -> str:
    """Hash scientific content while retaining raw runtime bytes elsewhere."""

    key = _scientific_artifact_key(path, root)
    if key in _SCIENTIFIC_JSON_RUNTIME_EXCLUSIONS:
        payload = _canonical_json_without_paths(
            path, _SCIENTIFIC_JSON_RUNTIME_EXCLUSIONS[key]
        )
        return hashlib.sha256(payload).hexdigest()
    if key in _SCIENTIFIC_CSV_RUNTIME_EXCLUSIONS:
        payload = _canonical_csv_without_columns(
            path, _SCIENTIFIC_CSV_RUNTIME_EXCLUSIONS[key]
        )
        return hashlib.sha256(payload).hexdigest()
    return _sha256(path)


def _scientific_hash_contract() -> dict[str, Any]:
    """Describe the fail-closed replay-hash boundary in retained provenance."""

    return {
        "contract_version": PHASE2_SCIENTIFIC_HASH_CONTRACT,
        "algorithm": "SHA-256",
        "canonical_json": "UTF-8 JSON with sorted keys and compact separators",
        "canonical_csv": (
            "UTF-8 header and ordered string rows encoded as canonical JSON"
        ),
        "json_runtime_fields_excluded": {
            key: ["/".join(path) for path in paths]
            for key, paths in sorted(_SCIENTIFIC_JSON_RUNTIME_EXCLUSIONS.items())
        },
        "csv_runtime_columns_excluded": {
            key: list(columns)
            for key, columns in sorted(_SCIENTIFIC_CSV_RUNTIME_EXCLUSIONS.items())
        },
        "all_other_artifacts": "raw file bytes",
        "raw_hash_field": "quantitative_artifact_sha256",
        "render_tamper_guard_hash_field": "quantitative_artifact_sha256",
        "scientific_replay_hash_field": ("quantitative_scientific_content_sha256"),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bear Lake validation-gated topobathymetry and 3D prototype."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    network = parser.add_mutually_exclusive_group()
    network.add_argument("--live", action="store_true")
    network.add_argument("--offline", action="store_true")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--metadata-only", action="store_true")
    action.add_argument("--download-only", action="store_true")
    action.add_argument("--extract-reference", action="store_true")
    action.add_argument("--validate-raster-model", action="store_true")
    action.add_argument("--build-spatial-products", action="store_true")
    action.add_argument("--render-only", action="store_true")
    action.add_argument("--all", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    return parser


def _preflight(settings: Phase2Settings, reference_dir: Path) -> None:
    required = (
        Path("regional_analysis/hydrometry/phase1.py"),
        Path("regional_analysis/hydrometry/sciencebase.py"),
        DEFAULT_PHASE1_DIR / "pilot_daily_hydrometry.csv",
        DEFAULT_PHASE1_DIR / "bear_lake_eav_normalized.csv",
        reference_dir / "bear_lake" / "bear_lake_eav_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise Phase2Error("Phase 2 hard preflight is missing: " + ", ".join(missing))
    if settings.profile_id != "bear_lake":
        raise Phase2Error("Phase 2 is intentionally restricted to Bear Lake.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_phase2_settings(args.config)
        if args.data_dir is not None:
            object.__setattr__(settings, "data_dir", args.data_dir)
        if args.figure_dir is not None:
            object.__setattr__(settings, "figure_dir", args.figure_dir)
        _preflight(settings, args.reference_dir)
        if args.refresh and args.offline:
            raise Phase2Error("--refresh cannot be combined with --offline.")
        if args.download_only and not args.live:
            raise Phase2Error("--download-only requires explicit --live.")
        if args.live and any(
            (
                args.extract_reference,
                args.validate_raster_model,
                args.build_spatial_products,
                args.render_only,
                args.all,
            )
        ):
            raise Phase2Error(
                "Extraction, validation, spatial builds, rendering, and --all "
                "are local actions and require --offline."
            )
        if not args.metadata_only and not (args.live or args.offline):
            raise Phase2Error("Choose --live or --offline for this action.")
        # The full action implementations are composed below the metadata,
        # raster, and render subsystems.  This local preflight is intentionally
        # executed for every action before any network or file mutation.
        return _run_action(args, settings)
    except (BinaryDownloadError, Phase2Error, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2


def _run_action(args: argparse.Namespace, settings: Phase2Settings) -> int:
    """Run one action, bounding GDAL's process-global cache for raster work."""

    raster_action = any(
        bool(getattr(args, name, False))
        for name in (
            "extract_reference",
            "validate_raster_model",
            "build_spatial_products",
            "render_only",
            "all",
        )
    )
    if raster_action:
        with _bounded_rasterio_runtime():
            return _run_action_body(args, settings)
    return _run_action_body(args, settings)


def _run_action_body(args: argparse.Namespace, settings: Phase2Settings) -> int:
    """Dispatch a Phase 2 action after all local scientific inputs validate."""

    stage = load_bear_stage_observations(DEFAULT_PHASE1_DIR)
    states = select_stage_states(
        stage,
        baseline_match_tolerance_days=settings.baseline_match_tolerance_days,
    )
    eav = load_bear_eav_source(settings, reference_dir=args.reference_dir)
    official = official_stage_values(eav, states)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    selected_state_path = settings.data_dir / "selected_stage_states.csv"
    # Before raster validation, rewrite the small official-only state table so
    # schema/provenance changes cannot leave a stale pending artifact.  A
    # completed validation/build owns the populated table and is never erased
    # by a later metadata replay.
    if not (settings.data_dir / "validation_policy.json").is_file():
        pending = official.copy()
        for column in (
            "source_resolution_raster_area_m2",
            "source_resolution_raster_storage_m3",
            "working_grid_area_m2",
            "working_grid_storage_m3",
        ):
            pending[column] = np.nan
        pending["validation_status"] = "pending_raster_validation"
        _atomic_csv(pending, selected_state_path)
    if args.metadata_only:
        # Filled by the metadata/binary integration below.  Keeping this
        # branch explicit guarantees metadata-only never reaches the archive
        # download path.
        result = _run_metadata_audit(
            settings,
            eav,
            live=not args.offline,
            refresh=args.refresh,
        )
        print(
            f"Bear Lake Phase 2 metadata audit: {result['status']}; "
            f"archive {result['archive_size_bytes']:,} bytes."
        )
        return 0
    if args.download_only:
        result = _download_archive(settings, eav, refresh=args.refresh)
        print(
            f"Bear Lake source archive: {result['archive_size_bytes']:,} bytes, "
            f"SHA-256 {result['archive_sha256']}; "
            f"{result['raster_candidate_count']} raster candidate(s)."
        )
        return 0
    if args.extract_reference:
        result = _extract_and_inspect(settings, eav)
        print(
            f"Bear Lake source raster: {result['width']:,} x {result['height']:,}, "
            f"{result['resolution_m']:.3f} m, {result['vertical_datum']}."
        )
        return 0
    if args.validate_raster_model:
        result = _validate_raster_model(settings, eav, stage, states)
        print(
            f"Bear Lake raster validation: {result['status']}; "
            f"selected {result.get('selected_spatial_resolution_m')} m."
        )
        return 0
    if args.build_spatial_products:
        result = _build_spatial_products(settings, eav, states)
        print(
            f"Bear Lake spatial products: {result['raster_count']} rasters and "
            f"{result['vector_layer_count']} vector layers."
        )
        return 0
    if args.render_only:
        result = _render_products(settings, eav, stage)
        print(
            f"Bear Lake render products: {result['static_3d_count']} static 3D, "
            f"GIF={result['gif_created']}, HTML={result['html_created']}."
        )
        return 0
    if args.all:
        _extract_and_inspect(settings, eav)
        validation = _validate_raster_model(settings, eav, stage, states)
        if validation["status"] == "failed":
            raise Phase2Error(
                "Source/working raster gate failed; spatial and polished rendering stopped."
            )
        _build_spatial_products(settings, eav, states)
        _render_products(settings, eav, stage)
        print("Bear Lake Phase 2 full offline workflow completed.")
        return 0
    raise Phase2Error(
        "The requested action is unavailable until the metadata audit has completed."
    )


def _source_raster_path(settings: Phase2Settings) -> Path:
    inspection_path = settings.data_dir / "source_raster_inspection.json"
    if not inspection_path.is_file():
        raise Phase2Error("Run --extract-reference before raster calculations.")
    record = json.loads(inspection_path.read_text(encoding="utf-8"))
    if record.get("status") != "source_raster_inspection_passed":
        raise Phase2Error("The source raster inspection gate has not passed.")
    path = Path(_required_text(record.get("path"), "source raster path"))
    if not path.is_file() or _sha256(path) != record.get("raster_sha256"):
        raise Phase2Error("The extracted source raster is missing or hash-mismatched.")
    return path


def _require_matching_source_crs(
    inspection: Any, metadata_record: Mapping[str, Any]
) -> None:
    """Require the embedded raster CRS to equal the parsed official PRJ CRS."""

    expected_epsg = metadata_record.get("horizontal_epsg")
    if isinstance(expected_epsg, bool):
        raise Phase2Error("Official spatial metadata has an invalid EPSG code.")
    try:
        parsed_epsg = int(expected_epsg)
    except (TypeError, ValueError) as exc:
        raise Phase2Error(
            "Official spatial metadata lacks a resolved EPSG code."
        ) from exc
    if inspection.horizontal_epsg != parsed_epsg:
        raise Phase2Error(
            "Source raster CRS does not match the parsed official spatial metadata."
        )


@dataclass(slots=True)
class WorkingGrid:
    resolution_m: float
    pixel_width_m: float
    pixel_height_m: float
    resampling_method: str
    elevation_m: np.ndarray
    valid_mask: np.ndarray
    transform: Any
    crs: Any
    seed: tuple[int, int]
    processing_seconds: float
    peak_memory_estimate_bytes: int


def _working_grid(
    raster_path: Path,
    settings: Phase2Settings,
    resolution_m: float,
    *,
    resampling_method: str = "average",
) -> WorkingGrid:
    """Average-resample the source into one explicit quantitative/render grid."""

    _prepare_rasterio_runtime()
    import rasterio
    from pyproj import Transformer
    from rasterio.enums import Resampling

    resampling_by_name = {
        "average": Resampling.average,
        "bilinear": Resampling.bilinear,
    }
    if resampling_method not in resampling_by_name:
        raise Phase2Error(
            f"Unsupported continuous-elevation resampling {resampling_method!r}."
        )
    quantitative_resampling = resampling_by_name[resampling_method]

    started = time.perf_counter()
    with rasterio.open(raster_path) as source:
        if not math.isclose(source.transform.b, 0.0, abs_tol=1e-12) or not math.isclose(
            source.transform.d, 0.0, abs_tol=1e-12
        ):
            raise Phase2Error("Rotated/sheared quantitative grids are unsupported.")
        source_pixel_width = abs(float(source.transform.a))
        source_pixel_height = abs(float(source.transform.e))
        width = max(1, round(source.width * source_pixel_width / resolution_m))
        height = max(1, round(source.height * source_pixel_height / resolution_m))
        transform = source.transform * source.transform.scale(
            source.width / width, source.height / height
        )
        values = source.read(
            1,
            out_shape=(height, width),
            masked=True,
            resampling=quantitative_resampling,
        )
        filled = values.filled(np.nan)
        elevation = np.asarray(filled, dtype=np.float64)
        raw_coverage = source.read_masks(
            1, out_shape=(height, width), resampling=Resampling.average
        )
        if not math.isclose(settings.minimum_valid_fraction, 1.0):
            raise Phase2Error(
                "Quantitative working grids require complete source coverage "
                "(minimum_valid_fraction=1.0); fractional weighting is not implemented."
            )
        valid = (raw_coverage == 255) & np.isfinite(elevation)
        transformer = Transformer.from_crs(
            settings.trusted_seed.crs, source.crs, always_xy=True
        )
        x, y = transformer.transform(
            settings.trusted_seed.longitude, settings.trusted_seed.latitude
        )
        row, column = rasterio.transform.rowcol(transform, x, y)
        seed = (int(row), int(column))
        if not (0 <= seed[0] < height and 0 <= seed[1] < width) or not valid[seed]:
            raise Phase2Error(
                f"Trusted seed is unsupported on the {resolution_m:g} m grid."
            )
        # Include conservative transient allocations from masked read, fill,
        # float64 conversion, coverage conversion, and final arrays.
        peak = int(
            values.data.nbytes
            + np.ma.getmaskarray(values).nbytes
            + filled.nbytes
            + elevation.nbytes
            + raw_coverage.nbytes
            + valid.nbytes
        )
        return WorkingGrid(
            resolution_m=float(resolution_m),
            pixel_width_m=abs(float(transform.a)),
            pixel_height_m=abs(float(transform.e)),
            resampling_method=resampling_method,
            elevation_m=elevation,
            valid_mask=valid,
            transform=transform,
            crs=source.crs,
            seed=seed,
            processing_seconds=time.perf_counter() - started,
            peak_memory_estimate_bytes=peak,
        )


def _official_points(frame: pd.DataFrame) -> dict[str, Any]:
    from .raster_storage import OfficialEavPoint

    return {
        str(row.validation_stage_id): OfficialEavPoint(
            validation_stage_id=str(row.validation_stage_id),
            stage_m_navd88=float(row.stage_m_navd88),
            official_area_m2=float(row.official_area_m2),
            official_storage_m3=float(row.official_storage_m3),
            stage_date=(
                None
                if pd.isna(row.stage_date)
                else date.fromisoformat(str(row.stage_date))
            ),
        )
        for row in frame.itertuples(index=False)
    }


def _unique_stage_rows(
    validation: pd.DataFrame,
    states: Sequence[SelectedStage],
    animation: Sequence[SelectedStage],
) -> pd.DataFrame:
    rows = [
        {
            "stage_key": f"validation:{row.validation_stage_id}",
            "stage_m_navd88": float(row.stage_m_navd88),
        }
        for row in validation.itertuples(index=False)
    ]
    rows.extend(
        {
            "stage_key": f"selected:{state.state_id}",
            "stage_m_navd88": state.stage_m_navd88,
        }
        for state in states
    )
    rows.extend(
        {
            "stage_key": f"animation:{state.observation_date.isoformat()}",
            "stage_m_navd88": state.stage_m_navd88,
        }
        for state in animation
    )
    return pd.DataFrame.from_records(rows).drop_duplicates(
        "stage_m_navd88", keep="first"
    )


def _grid_scalars(
    grid: WorkingGrid,
    stages: pd.DataFrame,
    settings: Phase2Settings,
    *,
    supported_stage_range_m: tuple[float, float],
    source_resolution_m: float,
    stage_plane_dtype: str,
    basin_method: str = "connected_seed",
    retained_mask_stages: Sequence[float] = (),
) -> tuple[dict[float, Any], dict[float, np.ndarray]]:
    from .raster_storage import RasterScalarResult, compute_stage_raster

    results: dict[float, Any] = {}
    masks: dict[float, np.ndarray] = {}
    retained = {float(value) for value in retained_mask_stages}
    valid_cell_count = int(np.count_nonzero(grid.valid_mask))
    grid_cell_count = int(grid.elevation_m.size)
    for stage in stages["stage_m_navd88"].to_numpy(dtype=float):
        started = time.perf_counter()
        result = compute_stage_raster(
            grid.elevation_m,
            stage,
            pixel_width_m=abs(float(grid.transform.a)),
            pixel_height_m=abs(float(grid.transform.e)),
            valid_mask=grid.valid_mask,
            basin_mask=grid.valid_mask,
            basin_method=basin_method,
            trusted_seed=(grid.seed if basin_method == "connected_seed" else None),
            connectivity=settings.connectivity,
            supported_stage_range_m=supported_stage_range_m,
            stage_plane_dtype=stage_plane_dtype,
        )
        if (
            result.status != "calculated"
            or result.area_m2 is None
            or result.storage_m3 is None
            or result.calculation_stage_m_navd88 is None
            or result.stage_plane_dtype is None
        ):
            raise Phase2Error(
                f"Stage {stage:g} m could not be calculated on the {grid.resolution_m:g} m grid: "
                f"{result.status}."
            )
        results[float(stage)] = RasterScalarResult(
            stage_m_navd88=float(stage),
            area_m2=float(result.area_m2),
            storage_m3=float(result.storage_m3),
            flooded_cell_count=result.flooded_cell_count,
            valid_cell_count=valid_cell_count,
            cell_area_m2=result.cell_area_m2,
            calculation_method=f"{basin_method}_full_grid_inclusive_float64",
            basin_method=basin_method,
            source_resolution_m=source_resolution_m,
            working_resolution_m=math.sqrt(grid.pixel_width_m * grid.pixel_height_m),
            processing_seconds=time.perf_counter() - started,
            peak_memory_estimate_bytes=int(
                grid.peak_memory_estimate_bytes
                + result.inundation.nbytes
                + result.depth_m.nbytes
                + result.supported_mask.nbytes
                # Conservative connected-component candidate/label workspace.
                + (5 * grid_cell_count if basin_method == "connected_seed" else 0)
            ),
            calculation_stage_m_navd88=result.calculation_stage_m_navd88,
            stage_plane_dtype=result.stage_plane_dtype,
            warning_flags=result.warning_flags,
        )
        if float(stage) in retained:
            masks[float(stage)] = result.inundation.copy()
    return results, masks


def _expanded_validation_stages(
    eav: BearLakeEavSource,
    required: pd.DataFrame,
    *,
    grid_steps: int,
) -> pd.DataFrame:
    """Add deterministic table rows while preserving named required stages."""

    table = eav.table.to_dataframe()
    indexes = np.unique(
        np.rint(np.linspace(1, len(table) - 1, grid_steps)).astype(np.int64)
    )
    dense = pd.DataFrame.from_records(
        [
            {
                "validation_stage_id": f"table_grid_{position:03d}",
                "stage_m_navd88": float(table.iloc[index].elevation_m_navd88),
                "stage_ft_navd88": convert_length(
                    float(table.iloc[index].elevation_m_navd88), "m", "ft"
                ),
                "stage_date": None,
                "official_area_m2": float(table.iloc[index].area_m2),
                "official_storage_m3": float(table.iloc[index].storage_m3),
            }
            for position, index in enumerate(indexes)
        ]
    )
    # Required named rows win when a dense-grid row has the same elevation.
    return (
        pd.concat((dense, required), ignore_index=True)
        .drop_duplicates("stage_m_navd88", keep="last")
        .sort_values("stage_m_navd88", kind="mergesort")
        .reset_index(drop=True)
    )


def _grid_threshold_scalars(
    grid: WorkingGrid,
    stages: pd.DataFrame,
    *,
    source_resolution_m: float,
    stage_plane_dtype: str,
    chunk_rows: int = 256,
) -> dict[float, Any]:
    """Integrate all-valid working-grid stages in bounded row windows."""

    from .raster_storage import ElevationWindow, integrate_threshold_event_windows

    values = tuple(stages["stage_m_navd88"].to_numpy(dtype=float))

    def windows() -> Any:
        for start in range(0, grid.elevation_m.shape[0], chunk_rows):
            stop = min(start + chunk_rows, grid.elevation_m.shape[0])
            yield ElevationWindow(
                grid.elevation_m[start:stop], grid.valid_mask[start:stop]
            )

    measured = integrate_threshold_event_windows(
        windows(),
        values,
        pixel_width_m=grid.pixel_width_m,
        pixel_height_m=grid.pixel_height_m,
        basin_method="all_valid_below_stage",
        source_resolution_m=source_resolution_m,
        working_resolution_m=math.sqrt(grid.pixel_width_m * grid.pixel_height_m),
        stage_plane_dtype=stage_plane_dtype,
    )
    return dict(zip(values, measured, strict=True))


def _precision_envelope(
    eav: BearLakeEavSource,
    validation: pd.DataFrame,
    *,
    source_dtype: str,
    source_cell_area_m2: float,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Derive stage-specific bounds from published and raster precision."""

    dtype = np.dtype(source_dtype)
    if not np.issubdtype(dtype, np.floating):
        raise Phase2Error(
            f"Source precision envelope requires a floating raster, not {dtype}."
        )
    table_stages = np.asarray(
        [row.elevation_m_navd88 for row in eav.table.rows], dtype=np.float64
    )
    interpolator = StageCapacityInterpolator(eav.table)
    rows: list[dict[str, Any]] = []
    for item in validation.itertuples(index=False):
        stage = float(item.stage_m_navd88)
        result = interpolator.derive(
            stage,
            input_unit="m",
            input_vertical_datum="NAVD88",
            height_type=HeightType.ORTHOMETRIC_HEIGHT,
            stage_variable=HydrometryVariable.WATER_LEVEL,
        )
        aligned = bool(np.any(np.isclose(table_stages, stage, rtol=0.0, atol=1e-9)))
        interpolation_half_step = (
            0.0 if aligned else eav.table.validation.elevation_step_median / 2.0
        )
        raster_spacing = abs(float(np.spacing(dtype.type(stage))))
        stage_precision = interpolation_half_step + raster_spacing
        area_absolute = (
            abs(float(result.d_area_dh_m2_per_m or 0.0)) * stage_precision
            + source_cell_area_m2
        )
        storage_absolute = float(item.official_area_m2) * stage_precision + 0.01
        area_relative = area_absolute / max(float(item.official_area_m2), 1.0)
        storage_relative = storage_absolute / max(float(item.official_storage_m3), 1.0)
        rows.append(
            {
                "validation_stage_id": str(item.validation_stage_id),
                "stage_m_navd88": stage,
                "table_stage_aligned": aligned,
                "interpolation_half_step_m": interpolation_half_step,
                "raster_dtype_spacing_m": raster_spacing,
                "stage_precision_m": stage_precision,
                "accepted_area_absolute_error_m2": area_absolute,
                "accepted_storage_absolute_error_m3": storage_absolute,
                "accepted_area_relative_error": area_relative,
                "accepted_storage_relative_error": storage_relative,
            }
        )
    return (
        max(row["accepted_area_relative_error"] for row in rows),
        max(row["accepted_storage_relative_error"] for row in rows),
        rows,
    )


def _stagewise_precision_gate(
    records: Sequence[Any], envelope_rows: Sequence[Mapping[str, Any]]
) -> tuple[bool, list[dict[str, Any]]]:
    """Require each source residual to satisfy its own absolute envelope."""

    envelopes = {str(row["validation_stage_id"]): dict(row) for row in envelope_rows}
    if set(envelopes) != {record.validation_stage_id for record in records}:
        raise Phase2Error("Source records and precision envelopes do not align.")
    audited: list[dict[str, Any]] = []
    for record in records:
        row = dict(envelopes[record.validation_stage_id])
        row["actual_area_absolute_error_m2"] = record.absolute_area_error_m2
        row["actual_storage_absolute_error_m3"] = record.absolute_storage_error_m3
        row["area_gate_passed"] = bool(
            record.absolute_area_error_m2
            <= float(row["accepted_area_absolute_error_m2"])
        )
        row["storage_gate_passed"] = bool(
            record.absolute_storage_error_m3
            <= float(row["accepted_storage_absolute_error_m3"])
        )
        row["stage_gate_passed"] = bool(
            row["area_gate_passed"] and row["storage_gate_passed"]
        )
        # Retain the original field names for compatibility while making it
        # explicit that this first gate tests exact numeric precision.  Phase 2
        # separately assesses source-method consistency at the native spatial
        # support; a method-consistent result need not reproduce every ArcGIS
        # area count inside a one-ULP derivative approximation.
        row["numeric_area_precision_gate_passed"] = row["area_gate_passed"]
        row["numeric_storage_precision_gate_passed"] = row["storage_gate_passed"]
        row["exact_numeric_stage_gate_passed"] = row["stage_gate_passed"]
        audited.append(row)
    return all(row["stage_gate_passed"] for row in audited), audited


def _stagewise_source_method_consistency_gate(
    records: Sequence[Any],
    audited_precision_rows: Sequence[Mapping[str, Any]],
    *,
    source_pixel_width_m: float,
    source_pixel_height_m: float,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    """Assess source-method consistency at the native grid's spatial support.

    The exact numeric area and storage diagnostics remain unchanged.  Area is
    additionally expressed as the difference between equal-area-circle radii,
    a resolution-scaled proxy suitable for comparing independent processing
    methods.  It is not interpreted as a local shoreline-position error.
    Storage has no analogous horizontal-support exception and must still pass
    the stage-specific numeric precision gate.
    """

    pixel_width = float(source_pixel_width_m)
    pixel_height = float(source_pixel_height_m)
    if (
        not math.isfinite(pixel_width)
        or not math.isfinite(pixel_height)
        or pixel_width <= 0.0
        or pixel_height <= 0.0
    ):
        raise Phase2Error("Native source pixel dimensions must be finite and positive.")
    cell_diagonal = math.hypot(pixel_width, pixel_height)

    precision_by_id = {
        str(row["validation_stage_id"]): dict(row) for row in audited_precision_rows
    }
    record_ids = {str(record.validation_stage_id) for record in records}
    if len(precision_by_id) != len(audited_precision_rows):
        raise Phase2Error("Source precision rows contain duplicate validation IDs.")
    if len(record_ids) != len(records):
        raise Phase2Error("Source validation records contain duplicate validation IDs.")
    if set(precision_by_id) != record_ids:
        raise Phase2Error("Source records and audited precision rows do not align.")

    audited: list[dict[str, Any]] = []
    exception_ids: list[str] = []
    for record in records:
        stage_id = str(record.validation_stage_id)
        row = dict(precision_by_id[stage_id])
        official_area = float(record.official_area_m2)
        raster_area = float(record.raster_area_m2)
        if (
            not math.isfinite(official_area)
            or not math.isfinite(raster_area)
            or official_area < 0.0
            or raster_area < 0.0
        ):
            raise Phase2Error(
                f"Source areas must be finite and nonnegative for {stage_id!r}."
            )
        official_radius = math.sqrt(official_area / math.pi)
        raster_radius = math.sqrt(raster_area / math.pi)
        radial_displacement = abs(official_radius - raster_radius)
        numeric_area_passed = bool(row["numeric_area_precision_gate_passed"])
        storage_precision_passed = bool(row["numeric_storage_precision_gate_passed"])
        if not numeric_area_passed:
            exception_ids.append(stage_id)
        area_scale_passed = radial_displacement <= cell_diagonal
        method_passed = bool(area_scale_passed and storage_precision_passed)
        row.update(
            {
                "official_area_equivalent_radius_m": official_radius,
                "raster_area_equivalent_radius_m": raster_radius,
                "source_area_equivalent_radial_displacement_m": (radial_displacement),
                "native_source_cell_diagonal_m": cell_diagonal,
                "source_method_area_scale_gate_passed": area_scale_passed,
                "source_method_storage_precision_gate_passed": (
                    storage_precision_passed
                ),
                "source_method_consistency_gate_passed": method_passed,
                "numeric_area_precision_exception": not numeric_area_passed,
            }
        )
        audited.append(row)

    maximum_displacement = max(
        row["source_area_equivalent_radial_displacement_m"] for row in audited
    )
    method_passed = all(row["source_method_consistency_gate_passed"] for row in audited)
    storage_passed = all(
        row["source_method_storage_precision_gate_passed"] for row in audited
    )
    exact_numeric_passed = all(
        row["exact_numeric_stage_gate_passed"] for row in audited
    )
    summary = {
        "source_method_consistency_gate_passed": method_passed,
        "source_exact_numeric_gate_passed": exact_numeric_passed,
        "source_storage_numeric_precision_gate_passed": storage_passed,
        "source_numeric_area_precision_exception_count": len(exception_ids),
        "source_numeric_area_precision_exception_stage_ids": exception_ids,
        "source_method_max_area_equivalent_radial_displacement_m": (
            maximum_displacement
        ),
        "native_source_cell_diagonal_m": cell_diagonal,
        "source_method_consistency_rule": SOURCE_METHOD_CONSISTENCY_RULE,
        "source_method_consistency_warning": SOURCE_METHOD_CONSISTENCY_WARNING,
    }
    # Repeat the compact audit summary in the CSV so the file remains
    # independently interpretable when separated from validation_policy.json.
    for row in audited:
        row.update(
            {
                "source_numeric_area_precision_exception_count": len(exception_ids),
                "source_method_max_area_equivalent_radial_displacement_m": (
                    maximum_displacement
                ),
                "source_method_consistency_rule": SOURCE_METHOD_CONSISTENCY_RULE,
                "source_method_consistency_warning": (
                    SOURCE_METHOD_CONSISTENCY_WARNING
                ),
            }
        )
    return method_passed, audited, summary


def _select_converged_resolution(
    benchmarks: Mapping[float, Any],
    *,
    observed_stage_ids: Sequence[str],
) -> tuple[float | None, bool, list[dict[str, Any]]]:
    """Select the finest candidate with robust, use-range convergence.

    Thresholded raster area is discrete, so a finer grid can be very slightly
    worse at an individual plane even when its error distribution is clearly
    converging.  Selection therefore uses a rule fixed independently of the
    measured Bear Lake residuals: the finer grid must improve the full-stage
    median and 95th percentile and the maximum over the preselected observed
    stages, for both area and storage.  Individual stage regressions remain in
    the returned audit rows but do not by themselves veto convergence.
    """

    resolutions = sorted(float(value) for value in benchmarks)
    observed_ids = {str(value) for value in observed_stage_ids}
    if not observed_ids:
        raise Phase2Error("Resolution convergence requires fixed observed stages.")
    diagnostics: list[dict[str, Any]] = []
    selected: float | None = None
    for finer, coarser in pairwise(resolutions):
        fine_metrics = {item.stage_id: item for item in benchmarks[finer].stage_metrics}
        coarse_metrics = {
            item.stage_id: item for item in benchmarks[coarser].stage_metrics
        }
        if set(fine_metrics) != set(coarse_metrics):
            raise Phase2Error(
                f"Resolution metrics do not align for {finer:g} and {coarser:g} m."
            )
        if not observed_ids.issubset(fine_metrics):
            missing = ", ".join(sorted(observed_ids.difference(fine_metrics)))
            raise Phase2Error(
                f"Resolution metrics lack fixed observed stages: {missing}."
            )

        def errors(metrics: Mapping[str, Any], field: str) -> np.ndarray:
            values = np.asarray(
                [abs(float(getattr(metrics[key], field))) for key in sorted(metrics)],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(values)):
                raise Phase2Error("Resolution convergence metrics must be finite.")
            return values

        def observed_max(metrics: Mapping[str, Any], field: str) -> float:
            values = [abs(float(getattr(metrics[key], field))) for key in observed_ids]
            if any(not math.isfinite(value) for value in values):
                raise Phase2Error("Observed resolution metrics must be finite.")
            return max(values)

        fine_area = errors(fine_metrics, "area_relative_error_vs_source")
        coarse_area = errors(coarse_metrics, "area_relative_error_vs_source")
        fine_storage = errors(fine_metrics, "storage_relative_error_vs_source")
        coarse_storage = errors(coarse_metrics, "storage_relative_error_vs_source")
        summary = {
            "candidate_full_median_abs_area_error": float(np.median(fine_area)),
            "next_coarser_full_median_abs_area_error": float(np.median(coarse_area)),
            "candidate_full_p95_abs_area_error": float(np.quantile(fine_area, 0.95)),
            "next_coarser_full_p95_abs_area_error": float(
                np.quantile(coarse_area, 0.95)
            ),
            "candidate_full_median_abs_storage_error": float(np.median(fine_storage)),
            "next_coarser_full_median_abs_storage_error": float(
                np.median(coarse_storage)
            ),
            "candidate_full_p95_abs_storage_error": float(
                np.quantile(fine_storage, 0.95)
            ),
            "next_coarser_full_p95_abs_storage_error": float(
                np.quantile(coarse_storage, 0.95)
            ),
            "candidate_observed_max_abs_area_error": observed_max(
                fine_metrics, "area_relative_error_vs_source"
            ),
            "next_coarser_observed_max_abs_area_error": observed_max(
                coarse_metrics, "area_relative_error_vs_source"
            ),
            "candidate_observed_max_abs_storage_error": observed_max(
                fine_metrics, "storage_relative_error_vs_source"
            ),
            "next_coarser_observed_max_abs_storage_error": observed_max(
                coarse_metrics, "storage_relative_error_vs_source"
            ),
        }
        summary["area_median_convergence_passed"] = bool(
            summary["candidate_full_median_abs_area_error"]
            <= summary["next_coarser_full_median_abs_area_error"]
        )
        summary["area_p95_convergence_passed"] = bool(
            summary["candidate_full_p95_abs_area_error"]
            <= summary["next_coarser_full_p95_abs_area_error"]
        )
        summary["storage_median_convergence_passed"] = bool(
            summary["candidate_full_median_abs_storage_error"]
            <= summary["next_coarser_full_median_abs_storage_error"]
        )
        summary["storage_p95_convergence_passed"] = bool(
            summary["candidate_full_p95_abs_storage_error"]
            <= summary["next_coarser_full_p95_abs_storage_error"]
        )
        summary["area_observed_max_convergence_passed"] = bool(
            summary["candidate_observed_max_abs_area_error"]
            <= summary["next_coarser_observed_max_abs_area_error"]
        )
        summary["storage_observed_max_convergence_passed"] = bool(
            summary["candidate_observed_max_abs_storage_error"]
            <= summary["next_coarser_observed_max_abs_storage_error"]
        )
        pair_passes = all(
            bool(summary[name])
            for name in (
                "area_median_convergence_passed",
                "area_p95_convergence_passed",
                "storage_median_convergence_passed",
                "storage_p95_convergence_passed",
                "area_observed_max_convergence_passed",
                "storage_observed_max_convergence_passed",
            )
        )
        summary["pair_distributional_convergence_passed"] = pair_passes
        if pair_passes and selected is None:
            selected = finer

        for stage_id in sorted(fine_metrics):
            fine = fine_metrics[stage_id]
            coarse = coarse_metrics[stage_id]
            area_fine = abs(float(fine.area_relative_error_vs_source))
            area_coarse = abs(float(coarse.area_relative_error_vs_source))
            storage_fine = abs(float(fine.storage_relative_error_vs_source))
            storage_coarse = abs(float(coarse.storage_relative_error_vs_source))
            area_pass = area_fine <= area_coarse
            storage_pass = storage_fine <= storage_coarse
            diagnostics.append(
                {
                    "candidate_resolution_m": finer,
                    "next_coarser_resolution_m": coarser,
                    "validation_stage_id": stage_id,
                    "candidate_abs_area_error_vs_source": area_fine,
                    "next_coarser_abs_area_error_vs_source": area_coarse,
                    "candidate_abs_storage_error_vs_source": storage_fine,
                    "next_coarser_abs_storage_error_vs_source": storage_coarse,
                    "area_convergence_passed": area_pass,
                    "storage_convergence_passed": storage_pass,
                    "fixed_observed_stage": stage_id in observed_ids,
                    **summary,
                }
            )
    return selected, selected is not None, diagnostics


def _assess_connected_basin_stage(
    *,
    connected_area_m2: float,
    connected_storage_m3: float,
    all_valid_area_m2: float,
    all_valid_storage_m3: float,
    selected_grid_cell_diagonal_m: float,
) -> dict[str, Any]:
    """Assess connectivity effects independently on one working grid.

    The reference is the all-valid result on the same grid, so this gate does
    not conflate basin selection with resampling error.  The area criterion is
    expressed as the exact difference between radii of equal-area circles and
    is limited to one selected-grid cell diagonal.  The storage criterion
    requires the excluded storage fraction to be no greater than the excluded
    area fraction, meaning the excluded cells are no deeper on average than
    the full inundated basin.
    """

    values = {
        "connected_area_m2": float(connected_area_m2),
        "connected_storage_m3": float(connected_storage_m3),
        "all_valid_area_m2": float(all_valid_area_m2),
        "all_valid_storage_m3": float(all_valid_storage_m3),
        "selected_grid_cell_diagonal_m": float(selected_grid_cell_diagonal_m),
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise Phase2Error("Connected-basin gate inputs must be finite.")
    if values["all_valid_area_m2"] <= 0 or values["all_valid_storage_m3"] <= 0:
        raise Phase2Error(
            "Connected-basin gate requires positive all-valid area and storage."
        )
    if values["selected_grid_cell_diagonal_m"] <= 0:
        raise Phase2Error("Selected-grid cell diagonal must be positive.")
    if values["connected_area_m2"] < 0 or values["connected_storage_m3"] < 0:
        raise Phase2Error("Connected-basin area and storage cannot be negative.")
    if (
        values["connected_area_m2"] > values["all_valid_area_m2"]
        or values["connected_storage_m3"] > values["all_valid_storage_m3"]
    ):
        raise Phase2Error(
            "Connected-seed results cannot exceed same-grid all-valid results."
        )

    area_difference = values["all_valid_area_m2"] - values["connected_area_m2"]
    storage_difference = values["all_valid_storage_m3"] - values["connected_storage_m3"]
    area_relative_difference = area_difference / values["all_valid_area_m2"]
    storage_relative_difference = storage_difference / values["all_valid_storage_m3"]
    radial_displacement = abs(
        math.sqrt(values["all_valid_area_m2"] / math.pi)
        - math.sqrt(values["connected_area_m2"] / math.pi)
    )
    area_scale_passed = radial_displacement <= values["selected_grid_cell_diagonal_m"]
    storage_shallowness_passed = storage_relative_difference <= area_relative_difference
    return {
        "absolute_area_relative_difference_vs_same_grid_all_valid": (
            area_relative_difference
        ),
        "absolute_storage_relative_difference_vs_same_grid_all_valid": (
            storage_relative_difference
        ),
        "area_equivalent_radial_displacement_m_vs_same_grid_all_valid": (
            radial_displacement
        ),
        "selected_grid_cell_diagonal_m": values["selected_grid_cell_diagonal_m"],
        "area_scale_gate_passed": area_scale_passed,
        "storage_shallowness_gate_passed": storage_shallowness_passed,
        "basin_stage_gate_passed": bool(
            area_scale_passed and storage_shallowness_passed
        ),
    }


def _validate_raster_model(
    settings: Phase2Settings,
    eav: BearLakeEavSource,
    stage_frame: pd.DataFrame,
    selected_states: Sequence[SelectedStage],
) -> dict[str, Any]:
    """Validate source and working grids with like-for-like scalar methods."""

    from .raster_storage import (
        build_eav_validation_records,
        build_resolution_benchmark,
        evaluate_render_gate,
        evaluate_validation_policy,
        integrate_raster_stages,
        integrate_raster_threshold_events,
        summarize_eav_validation,
    )

    raster_path = _source_raster_path(settings)
    inspection = json.loads(
        (settings.data_dir / "source_raster_inspection.json").read_text(
            encoding="utf-8"
        )
    )
    pixel_width_m = float(inspection["pixel_width_m"])
    pixel_height_m = float(inspection["pixel_height_m"])
    source_cell_area_m2 = pixel_width_m * pixel_height_m
    source_resolution_m = math.sqrt(source_cell_area_m2)
    source_dtype = _required_text(inspection.get("data_type"), "source data type")
    supported_stage_range_m = (
        float(eav.table.elevation_min_m),
        float(eav.table.elevation_max_m),
    )
    required_validation = validation_stages(eav, selected_states)
    validation = _expanded_validation_stages(
        eav,
        required_validation,
        grid_steps=settings.validation_grid_steps,
    )
    animation = select_animation_stages(
        stage_frame, frame_count=settings.animation_frame_count
    )
    source_stages = _unique_stage_rows(validation, selected_states, animation)
    observed_stages = _unique_stage_rows(
        validation.iloc[0:0], selected_states, animation
    )
    stage_values = tuple(source_stages["stage_m_navd88"].to_numpy(dtype=float))
    if (
        min(stage_values) < supported_stage_range_m[0]
        or max(stage_values) > supported_stage_range_m[1]
    ):
        raise Phase2Error("A requested stage lies outside the official E-A-V range.")

    started = time.perf_counter()
    source_results_raw = integrate_raster_threshold_events(
        raster_path,
        stage_values,
        basin_method="all_valid_below_stage",
        source_resolution_m=source_resolution_m,
    )
    expected_stage_plane_dtype = np.dtype(source_dtype).name
    if any(
        result.calculation_stage_m_navd88 is None
        or result.stage_plane_dtype != expected_stage_plane_dtype
        for result in source_results_raw
    ):
        raise Phase2Error(
            "Source scalar integration did not retain the inspected raster dtype plane."
        )
    source_by_stage = dict(zip(stage_values, source_results_raw, strict=True))
    _atomic_csv(
        pd.DataFrame.from_records(
            [
                {
                    "stage_key": row.stage_key,
                    **source_by_stage[float(row.stage_m_navd88)].as_dict(),
                }
                for row in source_stages.itertuples(index=False)
            ]
        ),
        settings.data_dir / "source_stage_scalars.csv",
    )

    official = _official_points(validation)
    source_validation_results = {
        stage_id: source_by_stage[point.stage_m_navd88]
        for stage_id, point in official.items()
    }
    source_records = build_eav_validation_records(
        official.values(), source_validation_results, validation_status="calculated"
    )
    source_area_envelope, source_storage_envelope, envelope_rows = _precision_envelope(
        eav,
        validation,
        source_dtype=source_dtype,
        source_cell_area_m2=source_cell_area_m2,
    )
    exact_stagewise_source_passed, audited_envelopes = _stagewise_precision_gate(
        source_records, envelope_rows
    )
    (
        source_method_consistency_passed,
        audited_envelopes,
        source_method_summary,
    ) = _stagewise_source_method_consistency_gate(
        source_records,
        audited_envelopes,
        source_pixel_width_m=pixel_width_m,
        source_pixel_height_m=pixel_height_m,
    )

    # A separate direct pass checks the event accumulator without accepting a
    # one-cell count discrepancy. Storage is allowed only floating summation
    # noise (or the published 0.01 m3 table precision, whichever is larger).
    direct_ids = ("safe_near_minimum", "table_median", "table_q90")
    direct = integrate_raster_stages(
        raster_path,
        tuple(official[name].stage_m_navd88 for name in direct_ids),
        basin_method="all_valid_below_stage",
        source_resolution_m=source_resolution_m,
    )
    for name, measured in zip(direct_ids, direct, strict=True):
        event = source_validation_results[name]
        if (
            measured.calculation_stage_m_navd88 != event.calculation_stage_m_navd88
            or measured.stage_plane_dtype != event.stage_plane_dtype
        ):
            raise Phase2Error(
                "Source threshold-event and direct calculation planes disagree."
            )
        if measured.flooded_cell_count != event.flooded_cell_count:
            raise Phase2Error("Source threshold-event and direct cell counts disagree.")
        if not math.isclose(measured.area_m2, event.area_m2, rel_tol=0.0, abs_tol=0.0):
            raise Phase2Error("Source threshold-event and direct area sums disagree.")
        storage_tolerance = max(
            0.01,
            64.0 * np.finfo(np.float64).eps * max(abs(event.storage_m3), 1.0),
        )
        if not math.isclose(
            measured.storage_m3,
            event.storage_m3,
            rel_tol=0.0,
            abs_tol=storage_tolerance,
        ):
            raise Phase2Error(
                "Source threshold-event and direct storage sums disagree."
            )

    validation_stage_columns = validation.loc[
        :, ["validation_stage_id", "stage_m_navd88"]
    ]
    benchmarks: dict[float, Any] = {}
    previous_results: Mapping[str, Any] = source_validation_results
    grid_metadata: dict[float, dict[str, Any]] = {}
    for resolution in settings.candidate_spatial_resolutions_m:
        grid = _working_grid(
            raster_path,
            settings,
            resolution,
            resampling_method=settings.quantitative_resampling,
        )
        all_valid = _grid_threshold_scalars(
            grid,
            validation_stage_columns,
            source_resolution_m=source_resolution_m,
            stage_plane_dtype=source_dtype,
        )
        candidate = {
            stage_id: all_valid[point.stage_m_navd88]
            for stage_id, point in official.items()
        }
        displacement = {
            stage_id: abs(
                candidate[stage_id].area_m2
                - source_validation_results[stage_id].area_m2
            )
            / max(2.0 * math.sqrt(math.pi * point.official_area_m2), 1.0)
            for stage_id, point in official.items()
        }
        scalar_peak = max(
            result.peak_memory_estimate_bytes for result in candidate.values()
        )
        scalar_seconds = max(result.processing_seconds for result in candidate.values())
        benchmarks[float(resolution)] = build_resolution_benchmark(
            resolution_m=resolution,
            width=grid.elevation_m.shape[1],
            height=grid.elevation_m.shape[0],
            candidate_results=candidate,
            source_results=source_validation_results,
            official_points=official,
            next_finer_results=previous_results,
            processing_seconds=grid.processing_seconds + scalar_seconds,
            peak_memory_estimate_bytes=grid.peak_memory_estimate_bytes + scalar_peak,
            resampling_method=settings.quantitative_resampling,
            file_size_bytes=int(grid.elevation_m.size * np.dtype("float32").itemsize),
            shoreline_displacement_by_stage_m=displacement,
        )
        grid_metadata[float(resolution)] = {
            "pixel_width_m": grid.pixel_width_m,
            "pixel_height_m": grid.pixel_height_m,
            "complete_coverage_cell_count": int(np.count_nonzero(grid.valid_mask)),
            "grid_cell_count": int(grid.valid_mask.size),
        }
        previous_results = candidate
        del all_valid, candidate, grid

    observed_validation_ids = tuple(
        str(value)
        for value in validation.loc[
            validation["stage_date"].notna(), "validation_stage_id"
        ]
    )
    selected_resolution, convergence_passed, convergence_rows = (
        _select_converged_resolution(
            benchmarks, observed_stage_ids=observed_validation_ids
        )
    )
    diagnostic_resolution = (
        selected_resolution
        if selected_resolution is not None
        else min(settings.candidate_spatial_resolutions_m)
    )
    selected_benchmark = (
        benchmarks[selected_resolution] if selected_resolution is not None else None
    )
    if selected_resolution is not None:
        resolutions = sorted(benchmarks)
        coarser_resolution = resolutions[resolutions.index(selected_resolution) + 1]
        coarser = benchmarks[coarser_resolution]
        working_area_envelope = max(
            abs(float(metric.area_relative_error_vs_source))
            for metric in coarser.stage_metrics
        )
        working_storage_envelope = max(
            abs(float(metric.storage_relative_error_vs_source))
            for metric in coarser.stage_metrics
        )
    else:
        coarser_resolution = None
        working_area_envelope = source_area_envelope
        working_storage_envelope = source_storage_envelope

    # Connectivity is evaluated only over observed stages. The theoretical
    # near-bottom validation rows cannot invalidate a seed that is documented
    # to remain wet over the observed Phase 1 range.
    selected_grid = _working_grid(
        raster_path,
        settings,
        diagnostic_resolution,
        resampling_method=settings.quantitative_resampling,
    )
    connected_results, _ = _grid_scalars(
        selected_grid,
        observed_stages,
        settings,
        supported_stage_range_m=supported_stage_range_m,
        source_resolution_m=source_resolution_m,
        stage_plane_dtype=source_dtype,
        basin_method="connected_seed",
    )
    observed_all_valid, _ = _grid_scalars(
        selected_grid,
        observed_stages,
        settings,
        supported_stage_range_m=supported_stage_range_m,
        source_resolution_m=source_resolution_m,
        stage_plane_dtype=source_dtype,
        basin_method="all_valid_below_stage",
    )
    selected_cell_diagonal_m = math.hypot(
        selected_grid.pixel_width_m, selected_grid.pixel_height_m
    )
    basin_stage_evidence: dict[float, dict[str, Any]] = {}
    for row in observed_stages.itertuples(index=False):
        stage = float(row.stage_m_navd88)
        connected = connected_results[stage]
        same_grid_all_valid = observed_all_valid[stage]
        source = source_by_stage[stage]
        area_error_vs_source = abs(connected.area_m2 - source.area_m2) / max(
            source.area_m2, 1.0
        )
        storage_error_vs_source = abs(connected.storage_m3 - source.storage_m3) / max(
            source.storage_m3, 1.0
        )
        method_evidence = _assess_connected_basin_stage(
            connected_area_m2=connected.area_m2,
            connected_storage_m3=connected.storage_m3,
            all_valid_area_m2=same_grid_all_valid.area_m2,
            all_valid_storage_m3=same_grid_all_valid.storage_m3,
            selected_grid_cell_diagonal_m=selected_cell_diagonal_m,
        )
        basin_stage_evidence[stage] = {
            "absolute_area_relative_error_vs_source": area_error_vs_source,
            "absolute_storage_relative_error_vs_source": storage_error_vs_source,
            **method_evidence,
        }
    ordered_connected = [
        connected_results[stage] for stage in sorted(connected_results)
    ]
    connected_area_monotonic = all(
        current.area_m2 <= following.area_m2
        for current, following in pairwise(ordered_connected)
    )
    connected_storage_monotonic = all(
        current.storage_m3 <= following.storage_m3
        for current, following in pairwise(ordered_connected)
    )
    connected_area_error = max(
        row["absolute_area_relative_error_vs_source"]
        for row in basin_stage_evidence.values()
    )
    connected_storage_error = max(
        row["absolute_storage_relative_error_vs_source"]
        for row in basin_stage_evidence.values()
    )
    connected_max_displacement_m = max(
        row["area_equivalent_radial_displacement_m_vs_same_grid_all_valid"]
        for row in basin_stage_evidence.values()
    )
    selected_basin_passed = bool(
        all(row["basin_stage_gate_passed"] for row in basin_stage_evidence.values())
        and connected_area_monotonic
        and connected_storage_monotonic
    )
    basin_comparison_rows: list[dict[str, Any]] = []
    for row in observed_stages.itertuples(index=False):
        stage = float(row.stage_m_navd88)
        for method, measured in (
            ("all_valid_below_stage", observed_all_valid[stage]),
            ("connected_seed", connected_results[stage]),
        ):
            basin_comparison_rows.append(
                {
                    "stage_key": row.stage_key,
                    "stage_m_navd88": stage,
                    "calculation_stage_m_navd88": (measured.calculation_stage_m_navd88),
                    "stage_plane_dtype": measured.stage_plane_dtype,
                    "resolution_m": diagnostic_resolution,
                    "basin_method": method,
                    "raster_area_m2": measured.area_m2,
                    "raster_storage_m3": measured.storage_m3,
                    "source_all_valid_area_m2": source_by_stage[stage].area_m2,
                    "source_all_valid_storage_m3": source_by_stage[stage].storage_m3,
                    "selected_for_products": method == "connected_seed",
                    "status": "evaluated_observed_range",
                    **(
                        basin_stage_evidence[stage]
                        if method == "connected_seed"
                        else {}
                    ),
                }
            )
    basin_comparison_rows.append(
        {
            "stage_key": "all",
            "stage_m_navd88": np.nan,
            "resolution_m": diagnostic_resolution,
            "basin_method": "independent_fixed_basin",
            "raster_area_m2": np.nan,
            "raster_storage_m3": np.nan,
            "source_all_valid_area_m2": np.nan,
            "source_all_valid_storage_m3": np.nan,
            "selected_for_products": False,
            "status": "not_available_no_independent_official_basin_mask",
        }
    )
    _atomic_csv(
        pd.DataFrame.from_records(basin_comparison_rows),
        settings.data_dir / "basin_method_comparison.csv",
    )
    _atomic_csv(
        pd.DataFrame.from_records(
            [
                {
                    "stage_key": row.stage_key,
                    **connected_results[float(row.stage_m_navd88)].as_dict(),
                }
                for row in observed_stages.itertuples(index=False)
            ]
        ),
        settings.data_dir / "working_stage_scalars.csv",
    )

    # Complete the primary scalar pass and release its large arrays before
    # allocating the secondary-resampler grid.
    primary_validation = _grid_threshold_scalars(
        selected_grid,
        validation_stage_columns,
        source_resolution_m=source_resolution_m,
        stage_plane_dtype=source_dtype,
    )
    del selected_grid
    # Compare the configured secondary resampler on the same selected grid.
    comparison_grid = _working_grid(
        raster_path,
        settings,
        diagnostic_resolution,
        resampling_method=settings.comparison_resampling,
    )
    comparison_all_valid = _grid_threshold_scalars(
        comparison_grid,
        validation_stage_columns,
        source_resolution_m=source_resolution_m,
        stage_plane_dtype=source_dtype,
    )
    resampling_rows = []
    for stage_id, point in official.items():
        primary = primary_validation[point.stage_m_navd88]
        comparison = comparison_all_valid[point.stage_m_navd88]
        resampling_rows.append(
            {
                "validation_stage_id": stage_id,
                "stage_m_navd88": point.stage_m_navd88,
                "calculation_stage_m_navd88": (primary.calculation_stage_m_navd88),
                "stage_plane_dtype": primary.stage_plane_dtype,
                "resolution_m": diagnostic_resolution,
                "primary_resampling": settings.quantitative_resampling,
                "comparison_resampling": settings.comparison_resampling,
                "primary_area_m2": primary.area_m2,
                "comparison_area_m2": comparison.area_m2,
                "absolute_area_difference_m2": abs(
                    primary.area_m2 - comparison.area_m2
                ),
                "primary_storage_m3": primary.storage_m3,
                "comparison_storage_m3": comparison.storage_m3,
                "absolute_storage_difference_m3": abs(
                    primary.storage_m3 - comparison.storage_m3
                ),
            }
        )
    _atomic_csv(
        pd.DataFrame.from_records(resampling_rows),
        settings.data_dir / "resampling_comparison.csv",
    )
    del comparison_all_valid, comparison_grid, primary_validation

    source_method_status = (
        "source_method_consistent_with_numeric_area_warning"
        if source_method_consistency_passed and not exact_stagewise_source_passed
        else (
            "source_method_consistent"
            if source_method_consistency_passed
            else "source_method_inconsistent"
        )
    )
    policy = evaluate_validation_policy(
        source_records,
        selected_benchmark,
        accepted_source_area_relative_error=source_area_envelope,
        accepted_source_storage_relative_error=source_storage_envelope,
        accepted_working_area_relative_error=working_area_envelope,
        accepted_working_storage_relative_error=working_storage_envelope,
        evidence_used=(
            "official 0.01 m E-A-V elevations and full-precision table values",
            (
                f"actual source {source_dtype} spacing and "
                f"{source_cell_area_m2:g} m2 cells"
            ),
            (
                "official-method inclusive bed <= source-float calculation "
                "plane with nominal stage retained separately"
            ),
            (
                "native-grid equal-area-radius source-method consistency with "
                "strict stagewise storage precision"
            ),
            "direct source-window checks at three named stages",
            (
                "predeclared full-stage median/p95 and fixed-observed-stage "
                "maximum convergence"
            ),
            (
                "connected-seed observed-stage displacement, shallowness, and "
                "monotonicity checks"
            ),
            (
                f"{settings.quantitative_resampling}/"
                f"{settings.comparison_resampling} resampling comparison"
            ),
        ),
        scientific_rationale=(
            "The source calculation follows the official ArcGIS Storage Capacity "
            "evidence by aligning each requested stage to the source floating dtype "
            "and including cells on that plane; the nominal stage remains unchanged "
            "for labels and traceability. Exact numeric area precision remains "
            "an explicit diagnostic. Source-method acceptance requires every "
            "stage's equal-area-radius displacement to remain within one native "
            "source-cell diagonal while storage still satisfies its stage-specific "
            "numeric precision envelope. Equal-area radius is an area-equivalent "
            "scale proxy, not a bound on local shoreline-position error. "
            "A working resolution is selected only when its full-stage median and "
            "95th-percentile errors and fixed-observed-stage maximum errors improve "
            "for both area and storage against the next coarser candidate. The "
            "connected basin is assessed independently: its area-equivalent "
            "displacement cannot exceed one selected-grid cell diagonal, its "
            "relative storage difference cannot exceed its relative area difference, "
            "and its observed-stage area and storage must be monotonic. "
            "No formal source-wide uncertainty is published, so a pass is termed "
            "source-method consistent rather than exact reproduction or formal "
            "validation."
        ),
        official_uncertainty_supports_formal_validation=False,
        resolution_convergence_behavior=(
            "Each finer/coarser pair was compared using full-stage median and p95 "
            "absolute relative errors plus maximum absolute relative errors over "
            "the fixed observed validation stages; per-stage regressions remain "
            "diagnostic in resolution_convergence.csv."
        ),
        stagewise_source_gate_passed=source_method_consistency_passed,
        resolution_convergence_gate_passed=convergence_passed,
        selected_basin_method_gate_passed=selected_basin_passed,
        **source_method_summary,
        source_method_consistency_status=source_method_status,
    )
    gate = evaluate_render_gate(policy)
    source_summary = summarize_eav_validation(
        source_records,
        resolution_sensitivity=(
            "/".join(f"{value:g}" for value in sorted(benchmarks))
            + " m all-valid candidates evaluated"
        ),
    )
    validation_frame = pd.DataFrame.from_records(
        [record.as_dict() for record in source_records]
    ).merge(
        pd.DataFrame.from_records(audited_envelopes),
        on=["validation_stage_id", "stage_m_navd88"],
        how="left",
        validate="one_to_one",
    )
    validation_frame["validation_status"] = policy.status
    _atomic_csv(validation_frame, settings.data_dir / "eav_raster_validation.csv")
    _atomic_json(
        settings.data_dir / "eav_raster_validation_summary.json",
        {
            **source_summary.as_dict(),
            "schema_version": PHASE2_SCHEMA_VERSION,
            "source_processing_seconds": source_results_raw[0].processing_seconds,
            "peak_memory_estimate_bytes": source_results_raw[
                0
            ].peak_memory_estimate_bytes,
            "peak_memory_estimate_scope": PHASE2_MEMORY_ESTIMATE_SCOPE,
            "gdal_cachemax_bytes": PHASE2_GDAL_CACHEMAX_BYTES,
            "source_dtype": source_dtype,
            "stage_plane_dtype": expected_stage_plane_dtype,
            "stage_comparison": (
                "bed_elevation_less_than_or_equal_to_calculation_stage"
            ),
            "calculation_stage_field": "calculation_stage_m_navd88",
            "requested_stage_field": "stage_m_navd88",
            "source_pixel_width_m": pixel_width_m,
            "source_pixel_height_m": pixel_height_m,
            "source_cell_area_m2": source_cell_area_m2,
            "validation_stage_count": len(validation),
            "direct_window_validation_stage_ids": list(direct_ids),
            **source_method_summary,
            "total_validation_seconds": time.perf_counter() - started,
        },
    )

    benchmark_rows = []
    for resolution, benchmark in sorted(benchmarks.items()):
        for metric in benchmark.stage_metrics:
            source = source_validation_results[metric.stage_id]
            benchmark_rows.append(
                {
                    "resolution_m": resolution,
                    **grid_metadata[resolution],
                    "width": benchmark.width,
                    "height": benchmark.height,
                    "cell_count": benchmark.cell_count,
                    "processing_seconds": benchmark.processing_seconds,
                    "peak_memory_estimate_bytes": benchmark.peak_memory_estimate_bytes,
                    "file_size_bytes": benchmark.file_size_bytes,
                    "resampling_method": benchmark.resampling_method,
                    "calculation_stage_m_navd88": (source.calculation_stage_m_navd88),
                    "stage_plane_dtype": source.stage_plane_dtype,
                    "official_area_m2": official[metric.stage_id].official_area_m2,
                    "raster_area_m2": source.area_m2
                    * (1.0 + float(metric.area_relative_error_vs_source)),
                    "relative_area_error": metric.area_relative_error_vs_official,
                    "official_storage_m3": official[
                        metric.stage_id
                    ].official_storage_m3,
                    "raster_storage_m3": source.storage_m3
                    * (1.0 + float(metric.storage_relative_error_vs_source)),
                    "relative_storage_error": metric.storage_relative_error_vs_official,
                    "selected_quantitative_grid": resolution == selected_resolution,
                    **asdict(metric),
                }
            )
    _atomic_csv(
        pd.DataFrame.from_records(benchmark_rows),
        settings.data_dir / "resolution_benchmark.csv",
    )
    _atomic_csv(
        pd.DataFrame.from_records(convergence_rows),
        settings.data_dir / "resolution_convergence.csv",
    )
    validation_inputs = _validation_input_records(settings, eav)
    _atomic_json(
        settings.data_dir / "validation_policy.json",
        {
            **policy.as_dict(),
            "schema_version": PHASE2_SCHEMA_VERSION,
            "validation_policy_sha256": policy.policy_sha256,
            "render_allowed": gate.allowed,
            "render_gate_reason": gate.reason,
            "diagnostic_working_resolution_m": diagnostic_resolution,
            "next_coarser_resolution_m": coarser_resolution,
            "stage_plane_policy": {
                "requested_stage_field": "stage_m_navd88",
                "calculation_stage_field": "calculation_stage_m_navd88",
                "stage_plane_dtype": expected_stage_plane_dtype,
                "comparison": ("bed_elevation_less_than_or_equal_to_calculation_stage"),
                "depth_and_storage_plane": "calculation_stage_m_navd88",
                "evidence": (
                    "official ArcGIS Storage Capacity output from the same float32 "
                    "source raster; equal-plane cells contribute area and zero depth"
                ),
            },
            "selected_connected_max_area_relative_error_vs_source": (
                connected_area_error
            ),
            "selected_connected_max_storage_relative_error_vs_source": (
                connected_storage_error
            ),
            "selected_connected_max_area_equivalent_displacement_m": (
                connected_max_displacement_m
            ),
            "selected_grid_cell_diagonal_m": selected_cell_diagonal_m,
            "connected_area_monotonic_over_observed_stages": (connected_area_monotonic),
            "connected_storage_monotonic_over_observed_stages": (
                connected_storage_monotonic
            ),
            "connected_basin_gate_rule": (
                "against all-valid inundation on the same selected grid at every "
                "observed stage: exact equal-area radial difference <= one grid-cell "
                "diagonal and excluded storage fraction <= excluded area fraction; "
                "connected area and storage must also be monotonic with stage"
            ),
            "connected_basin_stage_evidence": [
                {
                    "stage_key": str(row.stage_key),
                    "stage_m_navd88": float(row.stage_m_navd88),
                    **basin_stage_evidence[float(row.stage_m_navd88)],
                }
                for row in observed_stages.itertuples(index=False)
            ],
            "fixed_observed_convergence_stage_ids": list(observed_validation_ids),
            "resolution_convergence_rule": (
                "finer candidate improves full-stage median and p95 absolute "
                "relative area and storage errors plus maximum absolute errors "
                "over fixed observed validation stages"
            ),
            "stage_specific_precision_envelopes": audited_envelopes,
            "quantitative_coverage_requirement": "complete_source_coverage_only",
            "gdal_cachemax_bytes": PHASE2_GDAL_CACHEMAX_BYTES,
            "peak_memory_estimate_scope": PHASE2_MEMORY_ESTIMATE_SCOPE,
            "validation_inputs": validation_inputs,
            "validation_inputs_sha256": _records_sha256(validation_inputs),
            "shoreline_displacement_proxy_definition": (
                "absolute difference between radii of circles having the connected "
                "and same-grid all-valid inundation areas; diagnostic proxy, not "
                "surveyed shoreline displacement"
            ),
        },
    )
    selected_frame = _selected_stage_frame(
        official_stage_values(eav, selected_states),
        selected_states,
        pd.read_csv(settings.data_dir / "source_stage_scalars.csv"),
        pd.read_csv(settings.data_dir / "working_stage_scalars.csv"),
        validation_status=policy.status,
    )
    _atomic_csv(
        selected_frame,
        settings.data_dir / "selected_stage_states.csv",
    )
    return {
        "status": policy.status,
        "selected_spatial_resolution_m": policy.selected_spatial_resolution_m,
        "render_allowed": gate.allowed,
        "source_max_area_relative_error": policy.source_max_area_relative_error,
        "source_max_storage_relative_error": policy.source_max_storage_relative_error,
    }


def _load_render_gate(
    settings: Phase2Settings, eav: BearLakeEavSource | None = None
) -> tuple[Any, Any, dict[str, Any]]:
    """Reconstruct and hash-check the persisted numerical render policy."""

    from dataclasses import fields

    from .raster_storage import ValidationPolicy, evaluate_render_gate

    path = settings.data_dir / "validation_policy.json"
    if not path.is_file():
        raise Phase2Error(
            "Run --validate-raster-model before spatial or render actions."
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase2Error(f"Could not read the validation policy: {exc}") from exc
    if record.get("schema_version") != PHASE2_SCHEMA_VERSION:
        raise Phase2Error(
            "Validation policy schema is stale; rerun --validate-raster-model."
        )
    names = {item.name for item in fields(ValidationPolicy)}
    missing = sorted(names.difference(record))
    if missing:
        raise Phase2Error(
            "Validation policy lacks required fields: " + ", ".join(missing)
        )
    values = {name: record[name] for name in names}
    values["evidence_used"] = tuple(values["evidence_used"])
    policy = ValidationPolicy(**values)
    retained_hash = str(record.get("validation_policy_sha256", "")).lower()
    if retained_hash != policy.policy_sha256:
        raise Phase2Error("The retained validation policy hash is inconsistent.")
    gate = evaluate_render_gate(policy)
    if bool(record.get("render_allowed")) != gate.allowed:
        raise Phase2Error("The retained render decision disagrees with its policy.")
    if not gate.allowed:
        raise Phase2Error(gate.reason)
    if eav is not None:
        _require_matching_validation_inputs(
            record.get("validation_inputs"),
            _validation_input_records(settings, eav),
            retained_sha256=record.get("validation_inputs_sha256"),
        )
    return policy, gate, record


def _required_frame(path: Path, required: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise Phase2Error(f"Required Phase 2 table is missing: {path}")
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise Phase2Error(f"{path.name} lacks columns: {', '.join(missing)}")
    return frame


def _set_unique_index(
    frame: pd.DataFrame, column: str, *, context: str
) -> pd.DataFrame:
    """Index a frame and retain the former ``verify_integrity`` hard stop."""

    indexed = frame.set_index(column)
    if not indexed.index.is_unique:
        raise Phase2Error(f"{context} contains duplicate {column!r} values.")
    return indexed


def _retained_row_at_stage(frame: pd.DataFrame, stage_m: float) -> pd.Series:
    """Return one exact retained stage row without nearest-stage guessing."""

    if "stage_m_navd88" not in frame:
        raise Phase2Error("Retained scalar table lacks stage_m_navd88.")
    stages = pd.to_numeric(frame["stage_m_navd88"], errors="coerce").to_numpy(
        dtype=float
    )
    indexes = np.flatnonzero(np.isclose(stages, stage_m, rtol=0.0, atol=1e-8))
    if len(indexes) != 1:
        raise Phase2Error(
            f"Expected one retained scalar row at {stage_m:.9f} m; found {len(indexes)}."
        )
    return frame.iloc[int(indexes[0])]


def _scalar_at_stage(
    frame: pd.DataFrame,
    stage_m: float,
    *,
    area_columns: Sequence[str] = ("area_m2", "raster_area_m2"),
    storage_columns: Sequence[str] = ("storage_m3", "raster_storage_m3"),
) -> tuple[float, float]:
    """Return one exact retained scalar pair without nearest-stage guessing."""

    area_column = next((name for name in area_columns if name in frame), None)
    storage_column = next((name for name in storage_columns if name in frame), None)
    if area_column is None or storage_column is None:
        raise Phase2Error("Retained scalar table lacks area or storage values.")
    row = _retained_row_at_stage(frame, stage_m)
    area = float(row[area_column])
    storage = float(row[storage_column])
    if not math.isfinite(area) or not math.isfinite(storage):
        raise Phase2Error("Retained scalar area/storage values must be finite.")
    return area, storage


def _calculation_plane_at_stage(
    frame: pd.DataFrame, stage_m: float
) -> tuple[float, str]:
    """Return the retained effective water plane and its floating dtype."""

    required = {"calculation_stage_m_navd88", "stage_plane_dtype"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise Phase2Error(
            "Retained scalar table lacks calculation-plane fields: "
            + ", ".join(missing)
        )
    row = _retained_row_at_stage(frame, stage_m)
    calculation_stage = float(row["calculation_stage_m_navd88"])
    dtype_name = str(row["stage_plane_dtype"]).strip()
    if not math.isfinite(calculation_stage) or not dtype_name:
        raise Phase2Error("Retained calculation-plane metadata is invalid.")
    try:
        dtype = np.dtype(dtype_name)
    except TypeError as exc:
        raise Phase2Error("Retained stage-plane dtype is invalid.") from exc
    if not np.issubdtype(dtype, np.floating):
        raise Phase2Error("Retained stage-plane dtype must be floating.")
    # Canonicalize parser-level float64 noise back to the explicitly retained
    # source dtype before this plane is used for masks, depths, or rendering.
    return float(dtype.type(calculation_stage)), dtype.name


def _same_calculation_plane(
    left_stage_m: float,
    left_dtype_name: str,
    right_stage_m: float,
    right_dtype_name: str,
) -> bool:
    """Compare effective planes in the source dtype, not parser float64 bits.

    Pandas' default CSV parser may return a float64 value one binary64 ULP from
    the decimal text even when that text exactly represents the retained
    float32 plane.  The scientific identity is the value in ``stage_plane_dtype``;
    recasting both finite values to that declared dtype preserves the strict
    plane check without treating parser-level float64 noise as a new plane.
    """

    if left_dtype_name != right_dtype_name:
        return False
    try:
        dtype = np.dtype(left_dtype_name)
    except TypeError:
        return False
    if not np.issubdtype(dtype, np.floating):
        return False
    left = float(left_stage_m)
    right = float(right_stage_m)
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    return bool(dtype.type(left) == dtype.type(right))


def _artifact_key(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _validation_input_records(
    settings: Phase2Settings, eav: BearLakeEavSource
) -> dict[str, dict[str, str]]:
    """Hash every scientific/source input that the retained policy depends on."""

    request = _archive_request(settings, eav)
    paths = {
        "phase2_config": settings.config_path,
        "phase1_daily_stage": DEFAULT_PHASE1_DIR / "pilot_daily_hydrometry.csv",
        "phase1_normalized_eav": DEFAULT_PHASE1_DIR / "bear_lake_eav_normalized.csv",
        "official_eav_source": eav.source_path,
        "official_eav_metadata": eav.metadata_path,
        "official_eav_manifest": eav.manifest_path,
        "metadata_audit": settings.data_dir / "metadata_audit.json",
        "source_archive_manifest": request.manifest_path,
        "source_archive_inventory": settings.data_dir / "archive_inventory.json",
        "source_extraction_manifest": settings.data_dir
        / "source"
        / "extraction_manifest.json",
        "source_raster_inspection": settings.data_dir / "source_raster_inspection.json",
        "source_raster": _source_raster_path(settings),
    }
    records: dict[str, dict[str, str]] = {}
    for label, path in paths.items():
        path = Path(path)
        if not path.is_file():
            raise Phase2Error(f"Validation input {label!r} is missing: {path}")
        artifact_key = _scientific_artifact_key(path, settings.data_dir)
        canonicalized = (
            artifact_key in _SCIENTIFIC_JSON_RUNTIME_EXCLUSIONS
            or artifact_key in _SCIENTIFIC_CSV_RUNTIME_EXCLUSIONS
        )
        records[label] = {
            "path": _artifact_key(path, settings.data_dir),
            "sha256": _scientific_content_sha256(path, root=settings.data_dir),
            "sha256_scope": (
                PHASE2_SCIENTIFIC_HASH_CONTRACT if canonicalized else "raw_file_bytes"
            ),
        }
    return dict(sorted(records.items()))


def _records_sha256(records: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_matching_validation_inputs(
    retained: Any,
    current: Mapping[str, Mapping[str, str]],
    *,
    retained_sha256: Any,
) -> None:
    """Reject a passing policy when any bound source or scientific input changed."""

    if not isinstance(retained, Mapping) or not retained:
        raise Phase2Error("Validation policy lacks bound scientific input hashes.")
    retained_records = {
        str(label): dict(value) if isinstance(value, Mapping) else value
        for label, value in retained.items()
    }
    expected_digest = _records_sha256(retained_records)
    if str(retained_sha256).casefold() != expected_digest:
        raise Phase2Error("Validation policy input-binding hash is inconsistent.")
    if retained_records != dict(current):
        labels = sorted(
            label
            for label in set(retained_records).union(current)
            if retained_records.get(label) != current.get(label)
        )
        raise Phase2Error(
            "Retained validation is stale because bound inputs changed: "
            + ", ".join(labels)
            + ". Rerun --validate-raster-model."
        )


def _quantitative_paths(
    settings: Phase2Settings,
    eav: BearLakeEavSource,
    products: Any,
    selected_state_path: Path,
) -> tuple[Path, ...]:
    source_raster = _source_raster_path(settings)
    archive_manifest = _archive_request(settings, eav).manifest_path
    paths = [
        source_raster,
        eav.source_path,
        eav.metadata_path,
        eav.manifest_path,
        selected_state_path,
        settings.data_dir / "metadata_audit.json",
        archive_manifest,
        settings.data_dir / "archive_inventory.json",
        settings.data_dir / "source" / "extraction_manifest.json",
        settings.data_dir / "source_raster_inspection.json",
        settings.data_dir / "validation_policy.json",
        settings.data_dir / "eav_raster_validation.csv",
        settings.data_dir / "eav_raster_validation_summary.json",
        settings.data_dir / "resolution_benchmark.csv",
        settings.data_dir / "resolution_convergence.csv",
        settings.data_dir / "resampling_comparison.csv",
        settings.data_dir / "source_stage_scalars.csv",
        settings.data_dir / "working_stage_scalars.csv",
        settings.data_dir / "basin_method_comparison.csv",
        *products.rasters.values(),
    ]
    if products.geopackage is not None:
        paths.append(products.geopackage)
    return tuple(paths)


def _quantitative_hashes(
    settings: Phase2Settings,
    eav: BearLakeEavSource,
    products: Any,
    selected_state_path: Path,
) -> dict[str, str]:
    """Return raw-byte hashes used by the before/after render tamper guard."""

    hashes: dict[str, str] = {}
    for path in _quantitative_paths(settings, eav, products, selected_state_path):
        if not path.is_file():
            raise Phase2Error(f"Quantitative provenance input is missing: {path}")
        hashes[_artifact_key(path, settings.data_dir)] = _sha256(path)
    return dict(sorted(hashes.items()))


def _quantitative_scientific_hashes(
    settings: Phase2Settings,
    eav: BearLakeEavSource,
    products: Any,
    selected_state_path: Path,
) -> dict[str, str]:
    """Return replay-stable hashes with only declared runtime fields removed."""

    hashes: dict[str, str] = {}
    for path in _quantitative_paths(settings, eav, products, selected_state_path):
        if not path.is_file():
            raise Phase2Error(f"Quantitative provenance input is missing: {path}")
        hashes[_artifact_key(path, settings.data_dir)] = _scientific_content_sha256(
            path, root=settings.data_dir
        )
    return dict(sorted(hashes.items()))


def _selected_stage_frame(
    official: pd.DataFrame,
    states: Sequence[SelectedStage],
    source_scalars: pd.DataFrame,
    working_scalars: pd.DataFrame,
    *,
    validation_status: str,
) -> pd.DataFrame:
    """Attach both retained raster scales to the five real stage states."""

    rows: list[dict[str, Any]] = []
    official_by_id = _set_unique_index(
        official, "state_id", context="Official selected-stage table"
    )
    for state in states:
        source_area, source_storage = _scalar_at_stage(
            source_scalars, state.stage_m_navd88
        )
        source_calculation_stage, source_plane_dtype = _calculation_plane_at_stage(
            source_scalars, state.stage_m_navd88
        )
        working_area, working_storage = _scalar_at_stage(
            working_scalars, state.stage_m_navd88
        )
        working_calculation_stage, working_plane_dtype = _calculation_plane_at_stage(
            working_scalars, state.stage_m_navd88
        )
        if not _same_calculation_plane(
            source_calculation_stage,
            source_plane_dtype,
            working_calculation_stage,
            working_plane_dtype,
        ):
            raise Phase2Error(
                "Source and working scalar tables use different calculation planes."
            )
        record = official_by_id.loc[state.state_id].to_dict()
        rows.append(
            {
                "state_id": state.state_id,
                **record,
                "calculation_stage_m_navd88": working_calculation_stage,
                "stage_plane_dtype": working_plane_dtype,
                "stage_comparison": (
                    "bed_elevation_less_than_or_equal_to_calculation_stage"
                ),
                "source_resolution_calculation_stage_m_navd88": (
                    source_calculation_stage
                ),
                "working_grid_calculation_stage_m_navd88": (working_calculation_stage),
                "source_resolution_raster_area_m2": source_area,
                "source_resolution_raster_storage_m3": source_storage,
                "working_grid_area_m2": working_area,
                "working_grid_storage_m3": working_storage,
                "validation_status": validation_status,
            }
        )
    return pd.DataFrame.from_records(rows)


def _stage_metadata(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row.state_id): {
            key: (None if pd.isna(value) else value)
            for key, value in row._asdict().items()
        }
        for row in frame.itertuples(index=False)
    }


def _build_spatial_products(
    settings: Phase2Settings,
    eav: BearLakeEavSource,
    selected_states: Sequence[SelectedStage],
) -> dict[str, Any]:
    """Materialize validated working-grid rasters and QGIS vector layers."""

    from .spatial_products import (
        POLYGONIZATION_METHOD,
        SpatialProductError,
        SpatialState,
        write_quantitative_spatial_products,
    )

    try:
        policy, gate, policy_record = _load_render_gate(settings, eav)
        resolution = policy.selected_spatial_resolution_m
        if resolution is None:
            raise Phase2Error("Passing validation did not select a working resolution.")
        raster_path = _source_raster_path(settings)
        inspection_path = settings.data_dir / "source_raster_inspection.json"
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
        metadata_audit_path = settings.data_dir / "metadata_audit.json"
        metadata_audit = json.loads(metadata_audit_path.read_text(encoding="utf-8"))
        raw_survey_years = metadata_audit.get("source_years")
        if not isinstance(raw_survey_years, list) or not raw_survey_years:
            raise Phase2Error("Metadata audit lacks source survey years.")
        try:
            survey_years = tuple(int(value) for value in raw_survey_years)
            derived_processing_years = tuple(
                int(value)
                for value in metadata_audit.get("derived_processing_years", [])
            )
        except (TypeError, ValueError) as exc:
            raise Phase2Error("Metadata audit source years are invalid.") from exc
        source_resolution_m = math.sqrt(
            float(inspection["pixel_width_m"]) * float(inspection["pixel_height_m"])
        )
        source_dtype = _required_text(inspection.get("data_type"), "source data type")
        source_raster_sha256 = str(inspection["raster_sha256"])
        grid = _working_grid(
            raster_path,
            settings,
            float(resolution),
            resampling_method=settings.quantitative_resampling,
        )
        table = eav.table.to_dataframe()
        supported_range = (
            float(table["elevation_m_navd88"].min()),
            float(table["elevation_m_navd88"].max()),
        )
        stage_rows = pd.DataFrame.from_records(
            [
                {
                    "stage_key": f"selected:{state.state_id}",
                    "stage_m_navd88": state.stage_m_navd88,
                }
                for state in selected_states
            ]
        )
        stage_values = tuple(stage_rows["stage_m_navd88"].to_numpy(dtype=float))
        calculated, masks = _grid_scalars(
            grid,
            stage_rows,
            settings,
            supported_stage_range_m=supported_range,
            source_resolution_m=source_resolution_m,
            stage_plane_dtype=source_dtype,
            retained_mask_stages=stage_values,
        )
        source_scalars = _required_frame(
            settings.data_dir / "source_stage_scalars.csv", ("stage_m_navd88",)
        )
        retained_working = _required_frame(
            settings.data_dir / "working_stage_scalars.csv", ("stage_m_navd88",)
        )
        for stage_value, result in calculated.items():
            retained_area, retained_storage = _scalar_at_stage(
                retained_working, stage_value
            )
            retained_plane, retained_dtype = _calculation_plane_at_stage(
                retained_working, stage_value
            )
            if not math.isclose(
                result.area_m2, retained_area, rel_tol=1e-12, abs_tol=1e-6
            ) or not math.isclose(
                result.storage_m3,
                retained_storage,
                rel_tol=1e-12,
                abs_tol=1e-6,
            ):
                raise Phase2Error(
                    "Rebuilt connected working-grid scalars disagree with the "
                    f"validated table at {stage_value:.9f} m."
                )
            if not _same_calculation_plane(
                float(result.calculation_stage_m_navd88),
                str(result.stage_plane_dtype),
                retained_plane,
                retained_dtype,
            ):
                raise Phase2Error(
                    "Rebuilt working-grid calculation plane disagrees with the "
                    f"validated table at {stage_value:.9f} m."
                )

        official = official_stage_values(eav, selected_states)
        selected_frame = _selected_stage_frame(
            official,
            selected_states,
            source_scalars,
            retained_working,
            validation_status=policy.status,
        )
        selected_by_id = _set_unique_index(
            selected_frame, "state_id", context="Selected-stage table"
        )
        spatial_states: list[SpatialState] = []
        for frame_order, state in enumerate(selected_states, start=1):
            row = selected_by_id.loc[state.state_id]
            spatial_states.append(
                SpatialState(
                    state_id=state.state_id,
                    observation_date=state.observation_date,
                    stage_m_navd88=state.stage_m_navd88,
                    calculation_stage_m_navd88=float(row["calculation_stage_m_navd88"]),
                    stage_ft_navd88=state.stage_ft_navd88,
                    inundation=masks[state.stage_m_navd88],
                    frame_order=frame_order,
                    derived_area_m2=float(row["working_grid_area_m2"]),
                    derived_storage_m3=float(row["working_grid_storage_m3"]),
                    method="connected_seed",
                    validation_status=policy.status,
                    source=(
                        "USGS-10055000 observed NAVD88 stage + USGS surveyed "
                        "Bear Lake topobathymetry"
                    ),
                    properties={
                        "source_record_id": state.source_record_id,
                        "source_parameter_code": state.source_parameter_code,
                        "approval_status": state.approval_status,
                        "provisional": state.provisional,
                        "stage_plane_dtype": str(row["stage_plane_dtype"]),
                        "stage_comparison": str(row["stage_comparison"]),
                        "official_table_area_m2": float(row["official_table_area_m2"]),
                        "official_table_storage_m3": float(
                            row["official_table_storage_m3"]
                        ),
                        "source_method_consistency_status": (
                            policy.source_method_consistency_status
                        ),
                        "source_exact_numeric_gate_passed": (
                            policy.source_exact_numeric_gate_passed
                        ),
                        "source_numeric_area_precision_exception_count": (
                            policy.source_numeric_area_precision_exception_count
                        ),
                        "source_resolution_raster_area_m2": float(
                            row["source_resolution_raster_area_m2"]
                        ),
                        "source_resolution_raster_storage_m3": float(
                            row["source_resolution_raster_storage_m3"]
                        ),
                    },
                )
            )

        provenance_path = settings.data_dir / "bear_lake_phase2_provenance.json"
        base_provenance: dict[str, Any] = {
            "schema_version": PHASE2_SCHEMA_VERSION,
            "status": "building_spatial_products",
            "run_mode": "offline_local_processing",
            "generated_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "git_commit": _git_commit(),
            "profile_id": settings.profile_id,
            "sciencebase_item_id": eav.item.item_id,
            "source_title": eav.item.title,
            "source_version": eav.item.source_version,
            "source_license": eav.item.rights,
            "source_archive_filename": settings.raster_archive_name,
            "source_archive_sha256": inspection.get("archive_sha256"),
            "source_archive_public_url": _archive_request(settings, eav).public_url,
            "source_raster_path": _artifact_key(raster_path, settings.data_dir),
            "source_raster_sha256": source_raster_sha256,
            "source_raster_internal_filename": inspection.get("archive_member"),
            "source_raster_uncompressed_size_bytes": inspection.get(
                "uncompressed_size_bytes"
            ),
            "source_acquisition_records": {
                "metadata_audit": _artifact_key(
                    settings.data_dir / "metadata_audit.json", settings.data_dir
                ),
                "binary_manifest": _artifact_key(
                    _archive_request(settings, eav).manifest_path,
                    settings.data_dir,
                ),
                "archive_inventory": _artifact_key(
                    settings.data_dir / "archive_inventory.json", settings.data_dir
                ),
                "extraction_manifest": _artifact_key(
                    settings.data_dir / "source" / "extraction_manifest.json",
                    settings.data_dir,
                ),
                "raster_inspection": _artifact_key(inspection_path, settings.data_dir),
            },
            "official_eav_doi": settings.doi,
            "official_eav_source": str(eav.source_path.resolve()),
            "official_eav_sha256": eav.table.metadata.source_hash,
            "vertical_datum": inspection["vertical_datum"],
            "vertical_units": inspection["vertical_units"],
            "horizontal_crs": str(grid.crs),
            "source_resolution_m": source_resolution_m,
            "stage_plane_policy": {
                "requested_stage_field": "stage_m_navd88",
                "calculation_stage_field": "calculation_stage_m_navd88",
                "stage_plane_dtype": np.dtype(source_dtype).name,
                "comparison": ("bed_elevation_less_than_or_equal_to_calculation_stage"),
                "depth_and_storage_plane": "calculation_stage_m_navd88",
                "display_stage": "requested observed stage_m_navd88",
                "rationale": (
                    "Uses an inclusive source-float plane consistent with the "
                    "documented ArcGIS Storage Capacity methodology while "
                    "preserving the nominal observed level; exact numeric area "
                    "exceptions remain in the validation policy."
                ),
            },
            "quantitative_resolution_requested_m": float(resolution),
            "quantitative_pixel_width_m": grid.pixel_width_m,
            "quantitative_pixel_height_m": grid.pixel_height_m,
            "rendering_resolution_requested_m": settings.rendering_resolution_m,
            "quantitative_resampling": settings.quantitative_resampling,
            "rendering_resampling": settings.comparison_resampling,
            "minimum_valid_fraction": settings.minimum_valid_fraction,
            "gdal_cachemax_bytes": PHASE2_GDAL_CACHEMAX_BYTES,
            "peak_memory_estimate_scope": PHASE2_MEMORY_ESTIMATE_SCOPE,
            "basin_method": "connected_seed",
            "vector_polygonization_method": POLYGONIZATION_METHOD,
            "processing_domain": (
                "official topobathymetry valid-data extent; no independent fixed "
                "official basin mask is available"
            ),
            "connectivity": settings.connectivity,
            "trusted_seed": {
                "crs": settings.trusted_seed.crs,
                "longitude": settings.trusted_seed.longitude,
                "latitude": settings.trusted_seed.latitude,
                "working_grid_row": grid.seed[0],
                "working_grid_column": grid.seed[1],
                "description": settings.trusted_seed.description,
            },
            "survey_years": list(survey_years),
            "derived_processing_years": list(derived_processing_years),
            "validation_status": policy.status,
            "validation_policy_sha256": policy.policy_sha256,
            "validation_policy": policy.as_dict(),
            "validation_detail": gate.reason,
            "source_method_consistency_status": (
                policy.source_method_consistency_status
            ),
            "source_exact_numeric_gate_passed": (
                policy.source_exact_numeric_gate_passed
            ),
            "source_numeric_area_precision_exception_count": (
                policy.source_numeric_area_precision_exception_count
            ),
            "source_numeric_area_precision_exception_stage_ids": list(
                policy.source_numeric_area_precision_exception_stage_ids
            ),
            "validation_inputs": policy_record["validation_inputs"],
            "validation_inputs_sha256": policy_record["validation_inputs_sha256"],
            "validation_results": {
                "eav_raster_validation": "eav_raster_validation.csv",
                "eav_raster_validation_summary": ("eav_raster_validation_summary.json"),
                "resolution_benchmark": "resolution_benchmark.csv",
                "resolution_convergence": "resolution_convergence.csv",
                "basin_method_comparison": "basin_method_comparison.csv",
                "resampling_comparison": "resampling_comparison.csv",
            },
            "render_allowed": gate.allowed,
            "selected_states": [state.as_dict() for state in selected_states],
            "warnings": [
                "The topobathymetry combines source vintages listed in survey_years.",
                "No formal source-wide uncertainty supports a formal validation claim.",
                "Rendering-grid area and storage are never authoritative.",
                policy.source_method_consistency_warning,
                (
                    "Exact numeric area-precision exceptions are retained at: "
                    + ", ".join(
                        policy.source_numeric_area_precision_exception_stage_ids
                    )
                    if policy.source_numeric_area_precision_exception_stage_ids
                    else "No exact numeric area-precision exceptions were retained."
                ),
            ],
        }
        _atomic_json(provenance_path, base_provenance)
        products = write_quantitative_spatial_products(
            settings.data_dir,
            elevation_m=grid.elevation_m,
            basin_mask=grid.valid_mask,
            coverage_mask=grid.valid_mask,
            transform=grid.transform,
            crs=grid.crs,
            states=spatial_states,
            validation_status=policy.status,
            provenance_path=provenance_path,
            common_tags={
                "GRID_ROLE": "quantitative_spatial",
                "SOURCE_RESOLUTION_M": source_resolution_m,
                "WORKING_PIXEL_WIDTH_M": grid.pixel_width_m,
                "WORKING_PIXEL_HEIGHT_M": grid.pixel_height_m,
                "SOURCE": eav.item.title,
                "SOURCE_SHA256": source_raster_sha256,
                "OFFICIAL_EAV_SHA256": eav.table.metadata.source_hash,
                "BASIN_METHOD": "connected_seed",
                "CONNECTIVITY": settings.connectivity,
                "SURVEY_YEARS": ",".join(str(value) for value in survey_years),
                "SOURCE_METHOD_STATUS": policy.source_method_consistency_status,
                "SOURCE_EXACT_NUMERIC_GATE": policy.source_exact_numeric_gate_passed,
                "SOURCE_NUMERIC_AREA_EXCEPTION_COUNT": (
                    policy.source_numeric_area_precision_exception_count
                ),
            },
        )
        selected_path = settings.data_dir / "selected_stage_states.csv"
        _atomic_csv(selected_frame, selected_path)
        hashes = _quantitative_hashes(
            settings,
            eav,
            products,
            selected_path,
        )
        scientific_hashes = _quantitative_scientific_hashes(
            settings,
            eav,
            products,
            selected_path,
        )
        output_paths = {
            name: _artifact_key(path, settings.data_dir)
            for name, path in products.rasters.items()
        }
        if products.geopackage is not None:
            output_paths["geopackage"] = _artifact_key(
                products.geopackage, settings.data_dir
            )
        final_provenance = {
            **base_provenance,
            "status": "spatial_products_complete",
            "completed_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "spatial_products": dict(sorted(output_paths.items())),
            "quantitative_artifact_sha256": hashes,
            "quantitative_scientific_content_sha256": scientific_hashes,
            "quantitative_scientific_hash_contract": _scientific_hash_contract(),
            "vector_layers": {
                name: len(features) for name, features in products.vector_layers.items()
            },
        }
        _atomic_json(provenance_path, final_provenance)
        return {
            "raster_count": len(products.rasters),
            "vector_layer_count": len(products.vector_layers),
            "geopackage_created": products.geopackage is not None,
            "validation_status": policy.status,
            "provenance_path": str(provenance_path),
        }
    except SpatialProductError as exc:
        raise Phase2Error(str(exc)) from exc


def _read_spatial_products(
    settings: Phase2Settings, provenance: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], Any, Any]:
    """Read and alignment-check the retained working-grid GeoTIFFs."""

    _prepare_rasterio_runtime()
    import rasterio

    required = {
        "topobathy_working",
        "coverage",
        "depth_current",
        "depth_change_latest_vs_year_ago",
        "inundation_change_latest_vs_year_ago",
        "inundation_current",
        "inundation_recent_low",
        "inundation_recent_high",
        "inundation_year_ago",
    }
    records = provenance.get("spatial_products")
    if not isinstance(records, Mapping):
        raise Phase2Error("Phase 2 provenance lacks spatial product paths.")
    expected_hashes = provenance.get("quantitative_artifact_sha256")
    if not isinstance(expected_hashes, Mapping) or not expected_hashes:
        raise Phase2Error("Phase 2 provenance lacks quantitative artifact hashes.")
    missing = sorted(required.difference(records))
    if missing:
        raise Phase2Error(
            "Phase 2 spatial products are incomplete: " + ", ".join(missing)
        )
    arrays: dict[str, np.ndarray] = {}
    reference_transform: Any | None = None
    reference_crs: Any | None = None
    reference_shape: tuple[int, int] | None = None
    for name in sorted(required):
        path = Path(str(records[name]))
        if not path.is_absolute():
            path = settings.data_dir / path
        artifact_key = _artifact_key(path, settings.data_dir)
        if artifact_key not in expected_hashes:
            raise Phase2Error(
                f"Spatial product is not protected by the quantitative hash guard: {name}"
            )
        with rasterio.open(path) as dataset:
            if reference_transform is None:
                reference_transform = dataset.transform
                reference_crs = dataset.crs
                reference_shape = (dataset.height, dataset.width)
            elif (
                dataset.transform != reference_transform
                or dataset.crs != reference_crs
                or (dataset.height, dataset.width) != reference_shape
            ):
                raise Phase2Error(f"Spatial product grid is misaligned: {path.name}")
            arrays[name] = dataset.read(1)
    return arrays, reference_transform, reference_crs


def _source_year_label(provenance: Mapping[str, Any]) -> str:
    source_years = provenance.get("survey_years")
    if not isinstance(source_years, list) or not source_years:
        raise Phase2Error("Spatial provenance lacks source survey years.")
    try:
        label = ", ".join(str(int(value)) for value in source_years)
        processing = provenance.get("derived_processing_years", [])
        if processing:
            label += "; derived processing " + ", ".join(
                str(int(value)) for value in processing
            )
    except (TypeError, ValueError) as exc:
        raise Phase2Error("Spatial provenance contains invalid source years.") from exc
    return label


def _render_state(
    state: SelectedStage,
    inundation: np.ndarray,
    working_scalars: pd.DataFrame,
    validation_status: str,
    *,
    quality_status: str | None = None,
    survey_years: str = "official source years; see provenance",
    validation_detail: str = "",
) -> Any:
    from .render_3d import RenderState

    area, storage = _scalar_at_stage(working_scalars, state.stage_m_navd88)
    calculation_stage, _ = _calculation_plane_at_stage(
        working_scalars, state.stage_m_navd88
    )
    return RenderState(
        state_id=state.state_id,
        observation_date=state.observation_date.isoformat(),
        stage_m_navd88=state.stage_m_navd88,
        calculation_stage_m_navd88=calculation_stage,
        stage_ft_navd88=state.stage_ft_navd88,
        inundation=inundation,
        raster_area_m2=area,
        raster_storage_m3=storage,
        validation_status=validation_status,
        method="connected_seed",
        quality_status=quality_status or state.quality_status or "unknown",
        approval_status=state.approval_status or "unknown",
        provenance_label="bear_lake_phase2_provenance.json",
        survey_years=survey_years,
        validation_detail=validation_detail,
    )


def _render_products(
    settings: Phase2Settings,
    eav: BearLakeEavSource | None,
    stage_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Create 2D, fixed-camera 3D, GIF, and optional HTML render products."""

    from .animate_lake import create_stage_change_gif, write_animation_frame_table
    from .plot_phase2 import Phase2FigureInputs, Phase2PlotError, create_phase2_figures
    from .render_3d import (
        Phase2RenderError,
        RenderConfig,
        create_interactive_html,
        render_static_states,
        run_with_quantitative_hash_guard,
    )

    provenance_path = settings.data_dir / "bear_lake_phase2_provenance.json"
    if not provenance_path.is_file():
        raise Phase2Error("Run --build-spatial-products before --render-only.")
    policy, gate, _ = _load_render_gate(settings, eav)
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase2Error(f"Could not read Phase 2 provenance: {exc}") from exc
    if provenance.get("validation_policy_sha256") != policy.policy_sha256:
        raise Phase2Error("Spatial provenance refers to a different validation policy.")
    expected_hashes = provenance.get("quantitative_artifact_sha256")
    if not isinstance(expected_hashes, Mapping) or not expected_hashes:
        raise Phase2Error("Spatial provenance lacks quantitative artifact hashes.")

    def render_action() -> dict[str, Any]:
        arrays, transform, _ = _read_spatial_products(settings, provenance)
        source_year_text = _source_year_label(provenance)
        validation_detail = (
            f"{policy.source_method_consistency_status}; exact numeric area "
            "exceptions="
            f"{policy.source_numeric_area_precision_exception_count}"
        )
        elevation = arrays["topobathy_working"].astype(np.float64)
        elevation[elevation == -9999.0] = np.nan
        coverage = arrays["coverage"] == 1
        selected = _required_frame(
            settings.data_dir / "selected_stage_states.csv",
            ("state_id", "stage_m_navd88", "validation_status"),
        )
        if set(selected["validation_status"].astype(str)) != {policy.status}:
            raise Phase2Error(
                "Selected states do not share the retained validation status."
            )
        validation = _required_frame(
            settings.data_dir / "eav_raster_validation.csv",
            ("stage_m_navd88", "official_area_m2", "official_storage_m3"),
        )
        benchmark = _required_frame(
            settings.data_dir / "resolution_benchmark.csv",
            ("stage_m_navd88", "resolution_m"),
        )
        figure_inputs = Phase2FigureInputs(
            elevation_m=elevation,
            coverage_mask=coverage,
            depth_current_m=arrays["depth_current"],
            inundation_masks={
                "latest": arrays["inundation_current"] == 1,
                "recent_low": arrays["inundation_recent_low"] == 1,
                "recent_high": arrays["inundation_recent_high"] == 1,
                "year_ago": arrays["inundation_year_ago"] == 1,
            },
            depth_change_m=arrays["depth_change_latest_vs_year_ago"],
            inundation_change=arrays["inundation_change_latest_vs_year_ago"],
            transform=transform,
            validation=validation,
            resolution_benchmark=benchmark,
            stage_metadata=_stage_metadata(selected),
            validation_status=policy.status,
            survey_years=source_year_text,
            validation_detail=validation_detail,
        )
        # Numerical validation is deliberately rendered before any polished 3D.
        figure_paths = create_phase2_figures(figure_inputs, settings.figure_dir)
        validation_figure = settings.figure_dir / "bear_lake_eav_validation.png"
        if not validation_figure.is_file() or validation_figure.stat().st_size == 0:
            raise Phase2Error("Mandatory 2D E-A-V validation figure was not created.")

        spatial_records = provenance["spatial_products"]
        source_path = Path(str(spatial_records["topobathy_working"]))
        if not source_path.is_absolute():
            source_path = settings.data_dir / source_path
        inspection = json.loads(
            (settings.data_dir / "source_raster_inspection.json").read_text(
                encoding="utf-8"
            )
        )
        source_resolution_m = math.sqrt(
            float(inspection["pixel_width_m"]) * float(inspection["pixel_height_m"])
        )
        source_dtype = _required_text(inspection.get("data_type"), "source data type")
        rendering_grid = _working_grid(
            source_path,
            settings,
            settings.rendering_resolution_m,
            resampling_method=settings.comparison_resampling,
        )
        table = eav.table.to_dataframe()
        supported_range = (
            float(table["elevation_m_navd88"].min()),
            float(table["elevation_m_navd88"].max()),
        )
        selected_states = select_stage_states(
            stage_frame,
            baseline_match_tolerance_days=settings.baseline_match_tolerance_days,
        )
        animation_states = select_animation_stages(
            stage_frame, frame_count=settings.animation_frame_count
        )
        all_render_states = (*selected_states, *animation_states)
        stage_rows = pd.DataFrame.from_records(
            [
                {
                    "stage_key": f"render:{item.state_id}:{item.observation_date}",
                    "stage_m_navd88": item.stage_m_navd88,
                }
                for item in all_render_states
            ]
        ).drop_duplicates("stage_m_navd88", keep="first")
        render_stage_values = tuple(stage_rows["stage_m_navd88"].to_numpy(dtype=float))
        _, render_masks = _grid_scalars(
            rendering_grid,
            stage_rows,
            settings,
            supported_stage_range_m=supported_range,
            source_resolution_m=source_resolution_m,
            stage_plane_dtype=source_dtype,
            retained_mask_stages=render_stage_values,
        )
        working_scalars = _required_frame(
            settings.data_dir / "working_stage_scalars.csv", ("stage_m_navd88",)
        )
        selected_by_id = {item.state_id: item for item in selected_states}
        static_states = [
            _render_state(
                selected_by_id[state_id],
                render_masks[selected_by_id[state_id].stage_m_navd88],
                working_scalars,
                policy.status,
                survey_years=source_year_text,
                validation_detail=validation_detail,
            )
            for state_id in ("recent_low", "latest", "recent_high")
        ]
        static_config = RenderConfig(vertical_exaggeration=8.0, dpi=180)
        static_paths = render_static_states(
            rendering_grid.elevation_m,
            transform=rendering_grid.transform,
            states=static_states,
            output_dir=settings.figure_dir,
            gate=gate,
            validation_figure=validation_figure,
            config=static_config,
        )

        source_scalars = _required_frame(
            settings.data_dir / "source_stage_scalars.csv", ("stage_m_navd88",)
        )
        animation_official = _set_unique_index(
            official_stage_values(eav, animation_states),
            "state_id",
            context="Official animation-stage table",
        )
        animation_rows: list[dict[str, Any]] = []
        for frame_number, item in enumerate(animation_states, start=1):
            official = animation_official.loc[item.state_id]
            source_area, source_storage = _scalar_at_stage(
                source_scalars, item.stage_m_navd88
            )
            source_plane, source_plane_dtype = _calculation_plane_at_stage(
                source_scalars, item.stage_m_navd88
            )
            working_area, working_storage = _scalar_at_stage(
                working_scalars, item.stage_m_navd88
            )
            working_plane, working_plane_dtype = _calculation_plane_at_stage(
                working_scalars, item.stage_m_navd88
            )
            if not _same_calculation_plane(
                source_plane,
                source_plane_dtype,
                working_plane,
                working_plane_dtype,
            ):
                raise Phase2Error(
                    "Animation source and working calculation planes disagree."
                )
            observed = stage_frame.loc[
                stage_frame["calendar_date"].eq(item.observation_date)
            ]
            if len(observed) != 1:
                raise Phase2Error(
                    "Animation stage could not be traced to one observation."
                )
            quality = (
                "unknown"
                if "quality_status" not in observed
                or pd.isna(observed.iloc[0]["quality_status"])
                else str(observed.iloc[0]["quality_status"])
            )
            animation_rows.append(
                {
                    "frame_number": frame_number,
                    "target_date": item.observation_date.isoformat(),
                    "observation_date": item.observation_date.isoformat(),
                    "stage_m_navd88": item.stage_m_navd88,
                    "calculation_stage_m_navd88": working_plane,
                    "stage_ft_navd88": item.stage_ft_navd88,
                    "official_area_m2": float(official["official_table_area_m2"]),
                    "official_storage_m3": float(official["official_table_storage_m3"]),
                    "source_resolution_raster_area_m2": source_area,
                    "source_resolution_raster_storage_m3": source_storage,
                    "working_grid_area_m2": working_area,
                    "working_grid_storage_m3": working_storage,
                    "quality_status": quality,
                    "approval_status": item.approval_status or "unknown",
                    "provisional": item.provisional,
                    "validation_status": policy.status,
                }
            )
        animation_frame_path = write_animation_frame_table(
            pd.DataFrame.from_records(animation_rows),
            settings.data_dir / "bear_lake_animation_frames.csv",
        )
        gif_states = [
            _render_state(
                item,
                render_masks[item.stage_m_navd88],
                working_scalars,
                policy.status,
                survey_years=source_year_text,
                validation_detail=validation_detail,
            )
            for item in animation_states
        ]
        gif_path = create_stage_change_gif(
            rendering_grid.elevation_m,
            transform=rendering_grid.transform,
            states=gif_states,
            output_path=settings.figure_dir / "bear_lake_stage_change.gif",
            gate=gate,
            config=RenderConfig(vertical_exaggeration=8.0, stride=2, dpi=120),
            duration_ms=650,
        )
        html_path = settings.figure_dir / "bear_lake_interactive_3d.html"
        if settings.interactive_html_enabled:
            html_result = create_interactive_html(
                rendering_grid.elevation_m,
                transform=rendering_grid.transform,
                states=static_states,
                output_path=html_path,
                gate=gate,
                maximum_file_size_mb=(settings.interactive_html_maximum_file_size_mb),
                config=RenderConfig(vertical_exaggeration=8.0, stride=4),
            )
        else:
            html_path.unlink(missing_ok=True)
            html_result = None
        output_paths = [
            *figure_paths,
            *static_paths,
            animation_frame_path,
            gif_path,
        ]
        if html_result is not None and html_result.created and html_result.path:
            output_paths.append(html_result.path)
        return {
            "figure_paths": output_paths,
            "static_paths": static_paths,
            "gif_path": gif_path,
            "animation_frame_path": animation_frame_path,
            "html_result": html_result,
            "rendering_grid": rendering_grid,
            "static_config": static_config,
        }

    try:
        rendered = run_with_quantitative_hash_guard(
            expected_hashes, render_action, base_dir=settings.data_dir
        )
    except (Phase2PlotError, Phase2RenderError) as exc:
        raise Phase2Error(str(exc)) from exc

    figure_paths = rendered["figure_paths"]
    render_hashes = {
        _artifact_key(path, settings.figure_dir): _sha256(path)
        for path in figure_paths
        if path.is_file()
    }
    render_grid = rendered["rendering_grid"]
    static_config = rendered["static_config"]
    html_result = rendered["html_result"]
    updated = {
        **provenance,
        "status": "render_products_complete",
        "render_completed_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "rendering": {
            "figure_output_dir": str(settings.figure_dir.resolve()),
            "grid_role": "rendering_only_not_authoritative_for_area_or_storage",
            "derived_from": "bear_lake_topobathy_working.tif",
            "requested_resolution_m": settings.rendering_resolution_m,
            "pixel_width_m": render_grid.pixel_width_m,
            "pixel_height_m": render_grid.pixel_height_m,
            "resampling_method": render_grid.resampling_method,
            "vertical_exaggeration": static_config.vertical_exaggeration,
            "camera": asdict(static_config.camera),
            "gif_frame_count": settings.animation_frame_count,
            "gif_duration_ms": 650,
            "interactive_html_enabled": settings.interactive_html_enabled,
            "interactive_html_created": bool(
                html_result is not None and html_result.created
            ),
            "interactive_html_reason": (
                None if html_result is None else html_result.reason
            ),
            "interactive_html_file_size_bytes": (
                0 if html_result is None else html_result.file_size_bytes
            ),
        },
        "render_products": {
            _artifact_key(path, settings.figure_dir): str(path.resolve())
            for path in figure_paths
            if path.is_file()
        },
        "render_artifact_sha256": dict(sorted(render_hashes.items())),
    }
    _atomic_json(provenance_path, updated)
    return {
        "static_3d_count": len(rendered["static_paths"]),
        "gif_created": rendered["gif_path"].is_file(),
        "html_created": bool(html_result is not None and html_result.created),
        "figure_file_count": len(figure_paths),
        "validation_status": policy.status,
    }


def _sciencebase_catalog_transport_url(
    item_id: str,
    source: ScienceBaseFile,
) -> str:
    """Construct the official anonymous Catalog transport for an exact file."""

    normalized_item_id = _required_text(item_id, "ScienceBase item ID")
    expected_key = f"{normalized_item_id}/{source.name}"
    if source.storage_key != expected_key:
        raise Phase2Error(
            "ScienceBase storage key does not match the validated item and filename."
        )
    return (
        "https://www.sciencebase.gov/catalog/file/get/"
        f"{quote(normalized_item_id, safe='')}?{urlencode({'name': source.name})}"
    )


def _sciencebase_archive_transport_url(
    item_id: str,
    source: ScienceBaseFile,
) -> str:
    """Return the anonymous Catalog proxy for an identity-validated file."""

    return _sciencebase_catalog_transport_url(item_id, source)


def _archive_request(
    settings: Phase2Settings,
    eav: BearLakeEavSource,
) -> BinaryDownloadRequest:
    selected = select_bear_lake_phase2_files(
        eav.item,
        raster_archive_name=settings.raster_archive_name,
        spatial_metadata_archive_name=settings.spatial_metadata_archive_name,
    )
    archive = selected.raster_archive
    transport_url = _sciencebase_archive_transport_url(
        eav.item.item_id,
        archive,
    )
    return BinaryDownloadRequest(
        provider="usgs_sciencebase",
        url=transport_url,
        identity_url=transport_url,
        destination=settings.data_dir / "source" / archive.name,
        expected_size_bytes=archive.size_bytes,
        expected_content_types=(archive.content_type,),
    )


def _progress_reporter() -> Any:
    last_reported = -1

    def report(progress: Any) -> None:
        nonlocal last_reported
        expected = progress.expected_bytes
        if expected:
            percent = int(100 * progress.downloaded_bytes / expected)
            bucket = percent // 5
            if bucket != last_reported or progress.complete:
                last_reported = bucket
                print(
                    f"  archive download {min(percent, 100):3d}% "
                    f"({progress.downloaded_bytes:,}/{expected:,} bytes)",
                    flush=True,
                )
        elif progress.complete:
            print(f"  archive download {progress.downloaded_bytes:,} bytes", flush=True)

    return report


def _download_archive(
    settings: Phase2Settings, eav: BearLakeEavSource, *, refresh: bool
) -> dict[str, Any]:
    """Stream, validate, inventory, and disk-budget the official archive."""

    metadata_path = settings.data_dir / "metadata_audit.json"
    if not metadata_path.is_file():
        raise Phase2Error("Run --metadata-only successfully before archive download.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "passed":
        raise Phase2Error("The metadata gate has not passed.")
    request = _archive_request(settings, eav)
    downloader = StreamingBinaryDownloader(read_timeout=300.0)
    started = time.perf_counter()
    result = downloader.download(
        request,
        refresh=refresh,
        allow_resume=True,
        progress_callback=_progress_reporter(),
    )
    inventory = inspect_zip_archive(result.path)
    selected_raster = select_archive_raster(inventory)
    # Conservative derivative budget: twelve float32/byte working products
    # at 5 m plus 1 GiB for all visual products. The extraction budget itself
    # uses the exact central-directory size, never a guessed raster size.
    source_rows = int((metadata.get("fgdc_metadata") or {}).get("raster_rows") or 0)
    source_columns = int(
        (metadata.get("fgdc_metadata") or {}).get("raster_columns") or 0
    )
    source_resolution = float(metadata.get("source_resolution_m") or 0.0)
    if source_rows <= 0 or source_columns <= 0 or source_resolution <= 0:
        raise Phase2Error("Metadata audit lacks valid raster dimensions/resolution.")
    finest = min(settings.candidate_spatial_resolutions_m)
    working_cells = math.ceil(source_rows * source_resolution / finest) * math.ceil(
        source_columns * source_resolution / finest
    )
    derivative_bytes = int(working_cells * (8 * 4 + 8))
    rendering_bytes = 1024**3
    disk = calculate_disk_requirements(
        inventory,
        selected_members=(selected_raster.name,),
        working_directory=settings.data_dir,
        quantitative_derivative_bytes=derivative_bytes,
        rendering_output_bytes=rendering_bytes,
        archive_already_present=True,
    )
    require_sufficient_disk_space(disk)
    record = {
        "schema_version": PHASE2_SCHEMA_VERSION,
        "status": "downloaded_and_inventoried",
        "elapsed_seconds": time.perf_counter() - started,
        "request_count": downloader.request_count,
        "bytes_transferred": result.bytes_transferred,
        "cache_hit": result.cache_hit,
        "archive_path": str(result.path),
        "archive_manifest_path": str(result.manifest_path),
        "archive_size_bytes": inventory.archive_size_bytes,
        "archive_sha256": inventory.archive_sha256,
        "total_uncompressed_bytes": inventory.total_uncompressed_bytes,
        "raster_candidates": list(inventory.raster_candidates),
        "raster_candidate_count": len(inventory.raster_candidates),
        "selected_raster_member": selected_raster.name,
        "selected_raster_uncompressed_bytes": selected_raster.uncompressed_size_bytes,
        "disk_requirements": asdict(disk),
    }
    _atomic_json(settings.data_dir / "archive_inventory.json", record)
    _atomic_csv(
        pd.DataFrame.from_records(inventory_records(inventory)),
        settings.data_dir / "archive_inventory.csv",
    )
    return record


def _prepare_rasterio_runtime() -> None:
    """Prefer the active environment's bundled PROJ/GDAL data on Windows."""

    # Import the standard-library DLL consumers before rasterio/GDAL.  In the
    # pinned Windows environment, loading GDAL first can put incompatible
    # expat/sqlite DLLs earlier in the process search order and make a later
    # ``import sqlite3`` or ``import pyexpat`` fail.  These imports are a
    # runtime-order guard, not optional dependencies.
    import pyexpat  # noqa: F401
    import sqlite3  # noqa: F401

    site = Path(sys.prefix) / "Lib" / "site-packages" / "rasterio"
    proj = site / "proj_data"
    gdal = site / "gdal_data"
    if (proj / "proj.db").is_file():
        os.environ["PROJ_LIB"] = str(proj)
        os.environ["PROJ_DATA"] = str(proj)
    if gdal.is_dir():
        os.environ["GDAL_DATA"] = str(gdal)


@contextmanager
def _bounded_rasterio_runtime() -> Iterator[None]:
    """Cap GDAL's global block cache for one complete local raster action.

    GDAL otherwise defaults its cache to a percentage of physical memory. That
    hidden cache can be much larger than the explicitly windowed NumPy arrays,
    so the outer action scope is intentional: every nested dataset open shares
    one deterministic bound, which rasterio restores when the action ends.
    """

    _prepare_rasterio_runtime()
    import rasterio
    from rasterio.env import get_gdal_config

    with rasterio.Env(GDAL_CACHEMAX=PHASE2_GDAL_CACHEMAX_BYTES):
        configured = int(get_gdal_config("GDAL_CACHEMAX"))
        if configured != PHASE2_GDAL_CACHEMAX_BYTES:
            raise Phase2Error(
                "GDAL cache bound was not applied before Phase 2 raster processing."
            )
        yield


def _extract_and_inspect(
    settings: Phase2Settings, eav: BearLakeEavSource
) -> dict[str, Any]:
    """Replay the validated archive, extract one raster, and inspect it read-only."""

    from .topobathymetry import inspect_elevation_raster

    request = _archive_request(settings, eav)
    downloader = StreamingBinaryDownloader()
    cached = downloader.load_cached(request)
    inventory = inspect_zip_archive(cached.path)
    selected = select_archive_raster(inventory)
    prior = settings.data_dir / "archive_inventory.json"
    if not prior.is_file():
        raise Phase2Error("Archive inventory is missing; run --download-only first.")
    prior_record = json.loads(prior.read_text(encoding="utf-8"))
    if prior_record.get("archive_sha256") != inventory.archive_sha256:
        raise Phase2Error("Archive hash differs from the retained inventory.")
    metadata_record = json.loads(
        (settings.data_dir / "metadata_audit.json").read_text(encoding="utf-8")
    )
    official_vertical_datum, official_vertical_units = _validated_vertical_reference(
        metadata_record
    )
    source_rows = int((metadata_record.get("fgdc_metadata") or {}).get("raster_rows"))
    source_columns = int(
        (metadata_record.get("fgdc_metadata") or {}).get("raster_columns")
    )
    source_resolution = float(metadata_record.get("source_resolution_m"))
    projected_bounds = (metadata_record.get("spatial_metadata_archive") or {}).get(
        "projected_bounds_m"
    ) or {}
    try:
        expected_bounds = tuple(
            float(projected_bounds[name]) for name in ("xmin", "ymin", "xmax", "ymax")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Phase2Error(
            "Metadata audit lacks valid projected Bear Lake coverage bounds."
        ) from exc
    finest = min(settings.candidate_spatial_resolutions_m)
    disk = calculate_disk_requirements(
        inventory,
        selected_members=(selected.name,),
        working_directory=settings.data_dir,
        quantitative_derivative_bytes=int(
            (source_rows * source_resolution / finest)
            * (source_columns * source_resolution / finest)
            * 40
        ),
        rendering_output_bytes=1024**3,
        archive_already_present=True,
    )
    extracted = safe_extract_selected(
        cached.path,
        output_dir=settings.data_dir / "source" / "extracted",
        selected_members=(selected.name,),
        inventory=inventory,
        disk_requirements=disk,
    )
    extraction_manifest_path = settings.data_dir / "source" / "extraction_manifest.json"
    write_extraction_manifest(
        extraction_manifest_path,
        inventory=inventory,
        extracted_members=extracted,
        relative_to=settings.data_dir,
    )
    raster = extracted[0]
    _prepare_rasterio_runtime()
    inspection = inspect_elevation_raster(
        raster.output_path,
        expected_resolution_m=source_resolution,
        vertical_datum=official_vertical_datum,
        vertical_units=official_vertical_units,
        required_vertical_datum="NAVD88",
        expected_bounds=expected_bounds,
    )
    _require_matching_source_crs(inspection, metadata_record)
    small_window = _small_window_probe(
        raster.output_path,
        seed_crs=settings.trusted_seed.crs,
        longitude=settings.trusted_seed.longitude,
        latitude=settings.trusted_seed.latitude,
        observed_min_stage_m=min(
            state.stage_m_navd88
            for state in select_stage_states(load_bear_stage_observations())
        ),
    )
    record = {
        **inspection.as_dict(),
        "resolution_m": inspection.resolution_m,
        "schema_version": PHASE2_SCHEMA_VERSION,
        "status": "source_raster_inspection_passed",
        "archive_member": raster.archive_member,
        "raster_sha256": raster.sha256,
        "uncompressed_size_bytes": raster.size_bytes,
        "archive_sha256": inventory.archive_sha256,
        "extraction_manifest": _artifact_key(
            extraction_manifest_path, settings.data_dir
        ),
        "small_window_probe": small_window,
    }
    _atomic_json(settings.data_dir / "source_raster_inspection.json", record)
    return record


def _small_window_probe(
    raster_path: Path,
    *,
    seed_crs: str,
    longitude: float,
    latitude: float,
    observed_min_stage_m: float,
) -> dict[str, Any]:
    """Read a bounded real window and verify seed/depth arithmetic."""

    _prepare_rasterio_runtime()
    import pyexpat  # noqa: F401

    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window

    with rasterio.open(raster_path) as dataset:
        transformer = Transformer.from_crs(seed_crs, dataset.crs, always_xy=True)
        x, y = transformer.transform(longitude, latitude)
        row, column = dataset.index(x, y)
        if row < 0 or column < 0 or row >= dataset.height or column >= dataset.width:
            raise Phase2Error("Trusted seed lies outside the source raster.")
        window = Window(
            max(0, column - 64),
            max(0, row - 64),
            min(128, dataset.width - max(0, column - 64)),
            min(128, dataset.height - max(0, row - 64)),
        )
        values = dataset.read(1, window=window, masked=True).astype("float64")
        center = dataset.read(1, window=Window(column, row, 1, 1), masked=True)
        if center.mask.item():
            raise Phase2Error("Trusted seed falls on source nodata.")
        seed_bed = float(center.data.item())
        if seed_bed >= observed_min_stage_m:
            raise Phase2Error(
                "Trusted seed is not inundated at the observed minimum stage."
            )
        valid = values.compressed()
        if not len(valid) or not np.isfinite(valid).all():
            raise Phase2Error(
                "Small source window contains no finite valid elevations."
            )
        return {
            "seed_crs_x": x,
            "seed_crs_y": y,
            "seed_row": row,
            "seed_column": column,
            "seed_bed_elevation_m_navd88": seed_bed,
            "observed_min_stage_m_navd88": observed_min_stage_m,
            "seed_depth_at_observed_min_m": observed_min_stage_m - seed_bed,
            "window_width": int(window.width),
            "window_height": int(window.height),
            "window_valid_count": int(valid.size),
            "window_minimum_m": float(valid.min()),
            "window_maximum_m": float(valid.max()),
        }


def _run_metadata_audit(
    settings: Phase2Settings,
    eav: BearLakeEavSource,
    *,
    live: bool,
    refresh: bool = False,
) -> dict[str, Any]:
    """Acquire/replay bounded metadata and enforce the pre-archive gate."""

    started = time.perf_counter()
    audit_path = settings.data_dir / "metadata_audit.json"
    previous_live_run = read_previous_live_run(audit_path)

    selected = select_bear_lake_phase2_files(
        eav.item,
        raster_archive_name=settings.raster_archive_name,
        spatial_metadata_archive_name=settings.spatial_metadata_archive_name,
    )
    if selected.fgdc_xml.name != settings.fgdc_metadata_name:
        raise Phase2Error("Selected FGDC XML does not match the configured identity.")
    if selected.thumbnail.name != settings.thumbnail_name:
        raise Phase2Error("Selected thumbnail does not match the configured identity.")
    archive = selected.raster_archive
    summary = eav.item.summary.casefold()
    summary_compact = "".join(character for character in summary if character.isalnum())
    evidence = {
        "navd88": "navd 88" in summary or "navd88" in summary,
        "nad83_utm12": "utm zone 12" in summary
        and "north american datum of 1983" in summary,
        "geoid12b": "geoid12b" in summary_compact,
        "half_metre": "0.5-meter" in summary or "half-meter" in summary,
        "survey_2002": "2002" in summary,
        "survey_2016": "2016" in summary,
        "survey_2018": "2018" in summary,
    }
    if not all(evidence.values()):
        raise Phase2Error(
            "Official ScienceBase metadata does not confirm every required source fact."
        )
    metadata_root = settings.data_dir / "source_metadata"
    downloader = StreamingBinaryDownloader()
    downloaded: dict[str, Any] = {}
    for source in (
        selected.fgdc_xml,
        selected.spatial_metadata_archive,
        selected.thumbnail,
    ):
        request = BinaryDownloadRequest(
            provider="usgs_sciencebase",
            url=source.download_url,
            destination=metadata_root / source.name,
            expected_size_bytes=source.size_bytes,
            expected_content_types=(source.content_type,),
        )
        result = (
            downloader.download(request, refresh=refresh)
            if live
            else downloader.load_cached(request)
        )
        downloaded[source.name] = {
            "path": str(result.path),
            "manifest_path": str(result.manifest_path),
            "sha256": result.manifest.sha256,
            "bytes_transferred": result.bytes_transferred,
            "cache_hit": result.cache_hit,
        }

    spatial_zip = metadata_root / selected.spatial_metadata_archive.name
    spatial_inventory = inspect_zip_archive(spatial_zip, raster_suffixes=())
    spatial_facts = _inspect_spatial_metadata_bundle(spatial_zip)
    fgdc_facts = _parse_fgdc_metadata(metadata_root / selected.fgdc_xml.name)
    _validated_vertical_reference(fgdc_facts)
    combined_text = " ".join(
        str(value).casefold()
        for value in (
            fgdc_facts.get("raw_text_excerpt"),
            spatial_facts.get("text_excerpt"),
        )
    )
    required_metadata = {
        "navd88": "navd" in combined_text and "1988" in combined_text,
        "nad83": "nad83" in combined_text
        or "north american datum of 1983" in combined_text,
        "metres": "meter" in combined_text or "metre" in combined_text,
        "fgdc_vertical_datum_navd88": True,
        "fgdc_vertical_units_metre": True,
        "official_prj_projected": spatial_facts.get("horizontal_crs_projected") is True,
        "official_prj_metre_units": str(
            spatial_facts.get("horizontal_crs_linear_units") or ""
        ).casefold()
        in {"metre", "meter", "metres", "meters"},
        "official_prj_nad83_2011": spatial_facts.get("prj_confirms_nad83_2011") is True,
        "official_prj_utm_zone_12n": spatial_facts.get("prj_confirms_utm_zone_12n")
        is True,
        "official_prj_epsg_resolved": spatial_facts.get("horizontal_epsg") is not None,
    }
    if not all(required_metadata.values()):
        raise Phase2Error(
            "Downloaded official metadata does not corroborate CRS, datum, and units."
        )
    completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    elapsed_seconds = round(time.perf_counter() - started, 3)
    current_live_run = (
        {
            "completed_at_utc": completed_at,
            "request_count": downloader.request_count,
            "bytes_transferred": sum(
                int(value["bytes_transferred"]) for value in downloaded.values()
            ),
            "elapsed_seconds": elapsed_seconds,
            "refresh_requested": refresh,
        }
        if live
        else None
    )
    geographic_bounds = fgdc_facts.get("geographic_bounds")
    archive_request = _archive_request(settings, eav)
    archive_present = archive_request.destination.is_file()
    raster_inspection_path = settings.data_dir / "source_raster_inspection.json"
    raster_inspection_present = raster_inspection_path.is_file()
    source_resolution = _positive_float(
        fgdc_facts.get("horizontal_resolution_m"),
        "FGDC horizontal resolution",
    )
    record = {
        "schema_version": PHASE2_SCHEMA_VERSION,
        "status": "passed",
        "run_mode": "live" if live else "offline",
        "generated_at_utc": completed_at,
        "elapsed_seconds": elapsed_seconds,
        "refresh_requested": refresh,
        "latest_live_run": current_live_run or previous_live_run,
        "git_commit": _git_commit(),
        "doi": settings.doi,
        "sciencebase_item_id": eav.item.item_id,
        "source_title": eav.item.title,
        "source_version": eav.item.source_version,
        "archive_name": archive.name,
        "archive_size_bytes": archive.size_bytes,
        "pre_download_disk_gate": {
            "compressed_archive_bytes": archive.size_bytes,
            "maximum_partial_transfer_bytes": archive.size_bytes,
            "free_bytes_at_audit": shutil.disk_usage(settings.data_dir).free,
            "uncompressed_size_status": "pending ZIP central-directory inspection",
            "extraction_and_derivative_budget_status": (
                "pending ZIP central-directory inspection; no guessed uncompressed size"
            ),
        },
        "archive_downloaded_by_metadata_action": False,
        "archive_present_at_audit": archive_present,
        "source_raster_inspected_at_audit": raster_inspection_present,
        "large_archive_request_count": 0,
        "horizontal_crs_description": spatial_facts.get("horizontal_crs_name"),
        "horizontal_epsg": spatial_facts.get("horizontal_epsg"),
        "horizontal_crs_evidence": {
            "name": spatial_facts.get("horizontal_crs_name"),
            "identification_method": spatial_facts.get("epsg_identification_method"),
            "official_prj_parsed": True,
        },
        "geographic_bounds": geographic_bounds,
        "vertical_datum": fgdc_facts.get("vertical_datum"),
        "vertical_units": fgdc_facts.get("vertical_units"),
        "source_resolution_m": source_resolution,
        "source_years": spatial_facts.get("survey_source_years", []),
        "derived_processing_years": spatial_facts.get("derived_processing_years", []),
        "license": eav.item.rights,
        "north_arm_wording_warning": (
            "The official FGDC entity description calls the E-A-V table the "
            "'north arm of Bear Lake'; Phase 2 preserves this apparent metadata "
            "inconsistency pending comparison with the spatial extent."
        ),
        "spatial_scope_assessment": {
            "status": "whole_lake_coverage_supported_by_metadata",
            "evidence": (
                "The official coverage shapefile spans approximately "
                f"{spatial_facts.get('projected_width_m', 0) / 1000:.1f} by "
                f"{spatial_facts.get('projected_height_m', 0) / 1000:.1f} km and "
                "contains bathymetry, 2016/2018 lidar, and shoreline-gap records."
            ),
            "limitation": (
                "The raster extent inspection confirms the structured metadata bounds; "
                "the FGDC 'north arm' wording remains preserved as a warning."
                if raster_inspection_present
                else "The large raster archive has not been acquired or inspected; the "
                "FGDC 'north arm' wording remains preserved as a warning."
            ),
        },
        "configured_files": list(
            metadata_file_records(
                (
                    selected.raster_archive,
                    selected.spatial_metadata_archive,
                    selected.fgdc_xml,
                    selected.thumbnail,
                )
            )
        ),
        "bounded_metadata_downloads": downloaded,
        "bounded_metadata_request_count": downloader.request_count,
        "spatial_metadata_archive": {
            "sha256": spatial_inventory.archive_sha256,
            "member_count": len(spatial_inventory.members),
            "compressed_size_bytes": spatial_inventory.archive_size_bytes,
            "uncompressed_size_bytes": spatial_inventory.total_uncompressed_bytes,
            "members": [member.name for member in spatial_inventory.members],
            **spatial_facts,
        },
        "metadata_product_inventory": {
            "expected_internal_raster_filename": (
                json.loads(raster_inspection_path.read_text(encoding="utf-8")).get(
                    "archive_member"
                )
                if raster_inspection_present
                else None
            ),
            "expected_internal_raster_filename_status": (
                "confirmed_by_archive_inventory"
                if raster_inspection_present
                else "requires_archive_central_directory_inspection"
            ),
            "uncertainty_or_error_products": [],
            "uncertainty_status": (
                "FGDC publishes interpolation diagnostics but no separate "
                "uncertainty raster or source-wide formal uncertainty model"
            ),
            "basin_boundary_products": [],
            "basin_boundary_status": (
                "no independent official basin-boundary product identified"
            ),
            "coverage_products": [settings.spatial_metadata_archive_name],
        },
        "fgdc_metadata": fgdc_facts,
        "downloaded_metadata_corroboration": required_metadata,
        "item_summary_evidence": evidence,
    }
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(audit_path, record)
    write_metadata_audit_markdown(record, settings.data_dir / "metadata_audit.md")
    return {**record, "archive_size_bytes": archive.size_bytes}


def _metadata_text(root: ElementTree.Element, paths: Sequence[str]) -> str | None:
    for path in paths:
        value = root.findtext(path)
        if value and value.strip():
            return " ".join(value.split())
    return None


def _metadata_float(root: ElementTree.Element, paths: Sequence[str]) -> float | None:
    value = _metadata_text(root, paths)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_fgdc_metadata(path: Path) -> dict[str, Any]:
    """Parse bounded official FGDC XML without silently resolving its warning."""

    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        raise Phase2Error(f"Could not parse official FGDC XML {path}: {exc}") from exc
    text = " ".join(part.strip() for part in root.itertext() if part.strip())
    north_arm = "north arm of bear lake" in text.casefold()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "title": _metadata_text(
            root, (".//idinfo/citation/citeinfo/title", ".//title")
        ),
        "abstract": _metadata_text(
            root, (".//idinfo/descript/abstract", ".//abstract")
        ),
        "horizontal_datum": _metadata_text(root, (".//horizdn", ".//geodeticDatum")),
        "horizontal_units": _metadata_text(root, (".//plandu", ".//distanceUnits")),
        "horizontal_utm_zone": _metadata_text(root, (".//utmzone",)),
        "horizontal_resolution_m": _metadata_float(root, (".//absres",)),
        "vertical_datum": _metadata_text(root, (".//altdatum", ".//verticalDatum")),
        "vertical_units": _metadata_text(root, (".//altunits", ".//verticalUnits")),
        "geographic_bounds": {
            "west_longitude": _metadata_float(root, (".//westbc",)),
            "east_longitude": _metadata_float(root, (".//eastbc",)),
            "south_latitude": _metadata_float(root, (".//southbc",)),
            "north_latitude": _metadata_float(root, (".//northbc",)),
        },
        "raster_rows": _metadata_text(root, (".//rastinfo/rowcount", ".//rowcount")),
        "raster_columns": _metadata_text(root, (".//rastinfo/colcount", ".//colcount")),
        "nodata": _metadata_text(root, (".//nodataval", ".//NoDataValue")),
        "north_arm_wording_present": north_arm,
        "raw_text_excerpt": text[:12000],
    }


def _inspect_spatial_metadata_bundle(path: Path) -> dict[str, Any]:
    """Inspect structured coverage evidence without extracting the metadata ZIP."""

    try:
        return inspect_spatial_coverage_bundle(path)
    except Phase2MetadataError as exc:
        raise Phase2Error(
            f"Could not inspect spatial metadata ZIP {path}: {exc}"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
