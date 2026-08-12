"""Explain Tier 1 threshold transfer using completed local artifacts only.

This module never initializes Earth Engine and never reconstructs spatial
blocks.  It treats the cached ``block_id`` and ``split`` columns as the
authoritative experimental assignment, verifies the saved local/LOLO/regional
metrics on the cached test rows, and derives descriptive lake-by-reference
diagnostics from those same observations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

from .discover_lakes import HYDROLAKES_SENTINEL_VALUES
from .expanded_lake_selection import (
    CURRENT_IDS,
    HYDROLAKES_ASSET,
    TIER1_IDS,
    TYPE_LABELS,
    validate_selection_manifest,
)
from .expanded_transfer import expanded_output_paths
from .quick_curve_data import (
    discrimination_curves,
    read_curve_cache,
    threshold_response,
    training_threshold_grid,
)
from .quick_sampling import LakeSpec, QuickSamplingSettings, evaluate_threshold

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data" / "regional"
DEFAULT_PRIMARY_CACHE_DIR = DEFAULT_DATA_DIR / "quick_transfer_cache"
DEFAULT_EXPANDED_CACHE_DIR = DEFAULT_DATA_DIR / "expanded_transfer_cache"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR
DEFAULT_CANDIDATE_PATH = DEFAULT_DATA_DIR / "western_hydrolakes_candidates.csv"
DEFAULT_MANIFEST_PATH = DEFAULT_DATA_DIR / "expanded_lake_selection.csv"

REFERENCES = ("OPERA", "JRC")
DEFAULT_GRID_STEPS = 150
DEFAULT_PLATEAU_TOLERANCE = 0.02
RESULT_TOLERANCE = 1e-12
MIN_CORRELATION_LAKES = 8
MIN_CLASS_ACQUISITIONS_FOR_SD = 3

RAW_HYDROLAKES_FIELDS = (
    "Lake_area",
    "Lake_type",
    "Elevation",
    "Depth_avg",
    "Res_time",
    "Wshd_area",
    "Dis_avg",
    "Slope_100",
)

ASSOCIATION_VARIABLES = (
    "test_roc_auc",
    "test_average_precision",
    "test_ks_statistic",
    "test_ecdf_overlap_proxy",
    "abs_local_to_regional_threshold",
    "plateau_width_normalized",
    "positive_fraction_test",
    "ndwi_one_fraction_test",
    "acquisition_prevalence_sd",
    "annual_prevalence_sd",
    "water_acquisition_median_ndwi_sd",
    "nonwater_acquisition_median_ndwi_sd",
    "log10_Lake_area",
    "Elevation",
    "Depth_avg",
    "log10_Res_time",
    "log10_Wshd_area",
    "log10_Dis_avg",
    "Slope_100",
)
ASSOCIATION_OUTCOMES = ("lolo_test_iou", "lolo_iou_delta")
CASE_STUDIES = (
    ("OPERA", 791, "good discrimination and broad threshold plateau"),
    ("OPERA", 789, "transfer-sensitive threshold geometry"),
    ("JRC", 798, "good discrimination but shifted threshold"),
    ("JRC", 780, "poor class discrimination"),
)

METRIC_DEFINITIONS: dict[str, str] = {
    "outcome_metrics": (
        "Recomputed on authoritative split == test rows with prediction "
        "index_value >= the saved threshold; no threshold is refit."
    ),
    "test_distributions": "Quantiles and class separation use split == test only.",
    "reference_match_availability": (
        "Reference tile counts and match-time deltas are reduced to one metadata "
        "record per acquisition before their distribution is summarized."
    ),
    "discrimination": (
        "ROC AUC, average precision, and two-sample KS use raw held-out NDWI "
        "with reference-water label 1 as positive."
    ),
    "test_ecdf_overlap_proxy": "One minus the held-out two-sample KS statistic.",
    "ndwi_one": "Exact index_value == 1.0 within the named split/label subset.",
    "threshold_grid": (
        "One shared grid per reference: 150 points by default from the minimum "
        "per-lake training 0.5th percentile to the maximum per-lake training "
        "99.5th percentile."
    ),
    "plateau": (
        "Contiguous grid interval containing the first global held-out IoU "
        "maximum where IoU >= maximum - configured tolerance. A zero-width "
        "interval is limited by the shared grid resolution."
    ),
    "temporal_prevalence": (
        "Reference-water fraction is calculated per acquisition; variability "
        "summaries use all cached train and test rows and weight acquisitions "
        "equally. SD uses sample ddof=1."
    ),
    "annual_prevalence": (
        "SD/range of annual means of acquisition prevalence; NaN below two years."
    ),
    "monthly_prevalence": (
        "SD/range of calendar-month means of acquisition prevalence; NaN below "
        "two observed months."
    ),
    "class_temporal_ndwi": (
        "SD of per-acquisition class-specific median NDWI, reported only with "
        f"at least {MIN_CLASS_ACQUISITIONS_FOR_SD} acquisitions."
    ),
    "association": (
        "Spearman rank correlation is descriptive and reference-specific; no "
        "imputation, significance claim, or causal interpretation is made."
    ),
}


class Tier1ExplanatoryError(RuntimeError):
    """Raised when local Tier 1 artifacts are missing or inconsistent."""


@dataclass(frozen=True)
class Tier1Inputs:
    """Validated compact results, metadata, and authoritative cached samples."""

    manifest: pd.DataFrame
    candidates: pd.DataFrame
    results: Mapping[str, Mapping[str, pd.DataFrame]]
    samples: Mapping[tuple[str, int], pd.DataFrame]
    source_paths: tuple[Path, ...]


@dataclass(frozen=True)
class Tier1ExplanatoryResult:
    """All local explanatory products before serialization."""

    table: pd.DataFrame
    associations: pd.DataFrame
    case_studies: pd.DataFrame
    provenance: Mapping[str, Any]
    inputs: Tier1Inputs


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise Tier1ExplanatoryError(f"Missing {label}: {path}")
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise Tier1ExplanatoryError(f"Could not read {label} {path}: {exc}") from exc


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise Tier1ExplanatoryError(
            f"{label} is missing columns: {', '.join(missing)}."
        )


def _require_order(
    frame: pd.DataFrame,
    id_column: str,
    label: str,
    expected: Sequence[int] = TIER1_IDS,
) -> None:
    ids = tuple(pd.to_numeric(frame[id_column], errors="raise").astype(int))
    if ids != tuple(expected):
        raise Tier1ExplanatoryError(
            f"{label} IDs/order {ids} do not match authored Tier 1 {tuple(expected)}."
        )
    if frame[id_column].duplicated().any():
        raise Tier1ExplanatoryError(f"{label} contains duplicate lake IDs.")


def _validate_original_six_local(
    expanded: pd.DataFrame,
    original: pd.DataFrame,
    reference: str,
) -> None:
    """Verify that Tier 1 retains the completed six-lake local benchmarks."""

    _require_order(
        original,
        "Hylak_id",
        f"{reference} original-six local result",
        expected=CURRENT_IDS,
    )
    if set(original["reference"].astype(str).str.upper()) != {reference}:
        raise Tier1ExplanatoryError(
            f"{reference} original-six local result has a conflicting reference."
        )
    expanded_by_id = expanded.set_index("Hylak_id").loc[list(CURRENT_IDS)]
    original_by_id = original.set_index("Hylak_id")
    for column in ("n_samples", "n_train", "n_test", "n_blocks"):
        if (
            not expanded_by_id[column]
            .astype(int)
            .equals(original_by_id[column].astype(int))
        ):
            raise Tier1ExplanatoryError(
                f"{reference} original-six {column} differs in the Tier 1 result."
            )
    for column in (
        "local_threshold",
        "local_cv_iou",
        "local_test_iou",
        "local_test_f1",
    ):
        if not np.allclose(
            expanded_by_id[column].to_numpy(dtype=float),
            original_by_id[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=RESULT_TOLERANCE,
        ):
            raise Tier1ExplanatoryError(
                f"{reference} original-six {column} differs in the Tier 1 result."
            )


def _cache_path(
    reference: str,
    lake_id: int,
    primary_cache_dir: Path,
    expanded_cache_dir: Path,
) -> Path:
    root = primary_cache_dir if lake_id in CURRENT_IDS else expanded_cache_dir
    path = root / f"{reference.lower()}_{lake_id}.csv"
    if not path.is_file():
        raise Tier1ExplanatoryError(f"Missing authoritative sample cache: {path}")
    return path


def load_tier1_inputs(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    candidate_path: str | Path = DEFAULT_CANDIDATE_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    primary_cache_dir: str | Path = DEFAULT_PRIMARY_CACHE_DIR,
    expanded_cache_dir: str | Path = DEFAULT_EXPANDED_CACHE_DIR,
) -> Tier1Inputs:
    """Load every required local artifact and validate its experimental identity."""

    data_root = Path(data_dir)
    candidate_file = Path(candidate_path)
    manifest_file = Path(manifest_path)
    primary_root = Path(primary_cache_dir)
    expanded_root = Path(expanded_cache_dir)
    candidates = _read_csv(candidate_file, "HydroLAKES candidate inventory")
    manifest = _read_csv(manifest_file, "expanded-lake manifest")
    validate_selection_manifest(manifest)
    _require_columns(
        candidates,
        (
            "Hylak_id",
            "Lake_name",
            "state",
            "centroid_lon",
            "centroid_lat",
            *RAW_HYDROLAKES_FIELDS,
        ),
        "HydroLAKES candidate inventory",
    )
    _require_columns(
        manifest,
        (
            "Hylak_id",
            "Lake_name",
            "state",
            "selection_order",
            "analysis_role",
            "tier",
            "Lake_area",
            "Lake_type",
            "lake_type_label",
        ),
        "expanded-lake manifest",
    )
    manifest_tier1 = (
        manifest[manifest["Hylak_id"].isin(TIER1_IDS)]
        .sort_values("selection_order", kind="mergesort")
        .reset_index(drop=True)
    )
    _require_order(manifest_tier1, "Hylak_id", "Tier 1 manifest")
    candidates_tier1 = (
        candidates.set_index("Hylak_id", drop=False)
        .loc[list(TIER1_IDS)]
        .reset_index(drop=True)
    )
    _require_order(candidates_tier1, "Hylak_id", "Tier 1 candidate inventory")

    local_schema = (
        "reference",
        "Hylak_id",
        "Lake_name",
        "state",
        "Lake_area",
        "n_samples",
        "n_train",
        "n_test",
        "n_blocks",
        "positive_fraction",
        "local_threshold",
        "local_cv_iou",
        "local_test_iou",
        "local_test_f1",
    )
    lolo_schema = (
        "reference",
        "held_out_Hylak_id",
        "held_out_lake",
        "threshold",
        "held_out_test_iou",
        "held_out_test_f1",
        "iou_delta_vs_local",
    )
    regional_schema = (
        "reference",
        "Hylak_id",
        "Lake_name",
        "regional_threshold",
        "regional_test_iou",
        "regional_test_f1",
        "iou_delta_vs_local",
    )
    macro_schema = (
        "reference",
        "tier",
        "regional_threshold",
        "macro_regional_test_iou",
        "macro_lolo_iou",
    )
    settings = QuickSamplingSettings()
    results: dict[str, dict[str, pd.DataFrame]] = {}
    samples: dict[tuple[str, int], pd.DataFrame] = {}
    source_paths: list[Path] = [candidate_file, manifest_file]
    for reference in REFERENCES:
        paths = expanded_output_paths(data_root, reference, "tier1")
        tables = {
            "local": _read_csv(paths.local, f"{reference} local result"),
            "lolo": _read_csv(paths.lolo, f"{reference} LOLO result"),
            "regional": _read_csv(paths.regional, f"{reference} regional result"),
            "macro": _read_csv(paths.macro, f"{reference} macro result"),
        }
        original_local_path = (
            data_root / f"western_quick_transfer_{reference.lower()}_local.csv"
        )
        original_local = _read_csv(
            original_local_path, f"{reference} original-six local result"
        )
        _require_columns(
            original_local,
            local_schema,
            f"{reference} original-six local result",
        )
        for name, schema in (
            ("local", local_schema),
            ("lolo", lolo_schema),
            ("regional", regional_schema),
            ("macro", macro_schema),
        ):
            _require_columns(tables[name], schema, f"{reference} {name} result")
            if set(tables[name]["reference"].astype(str).str.upper()) != {reference}:
                raise Tier1ExplanatoryError(
                    f"{reference} {name} result has a conflicting reference."
                )
        _require_order(tables["local"], "Hylak_id", f"{reference} local result")
        _validate_original_six_local(tables["local"], original_local, reference)
        _require_order(
            tables["lolo"],
            "held_out_Hylak_id",
            f"{reference} LOLO result",
        )
        _require_order(tables["regional"], "Hylak_id", f"{reference} regional result")
        if len(tables["macro"]) != 1 or tables["macro"]["tier"].iloc[0] != "tier1":
            raise Tier1ExplanatoryError(
                f"{reference} macro result must contain one Tier 1 row."
            )
        macro = tables["macro"].iloc[0]
        regional_thresholds = tables["regional"]["regional_threshold"].to_numpy(
            dtype=float
        )
        if not np.allclose(
            regional_thresholds,
            float(macro.regional_threshold),
            rtol=0.0,
            atol=RESULT_TOLERANCE,
        ):
            raise Tier1ExplanatoryError(
                f"{reference} regional thresholds disagree with the macro result."
            )
        _assert_close(
            float(tables["regional"]["regional_test_iou"].mean()),
            float(macro.macro_regional_test_iou),
            f"{reference} macro regional test IoU",
        )
        _assert_close(
            float(tables["lolo"]["held_out_test_iou"].mean()),
            float(macro.macro_lolo_iou),
            f"{reference} macro LOLO IoU",
        )
        source_paths.extend(
            (
                original_local_path,
                paths.local,
                paths.lolo,
                paths.regional,
                paths.macro,
            )
        )

        local_by_id = tables["local"].set_index("Hylak_id")
        for candidate in candidates_tier1.itertuples(index=False):
            lake_id = int(candidate.Hylak_id)
            lake = LakeSpec(
                hylak_id=lake_id,
                lake_name=str(candidate.Lake_name),
                state=str(candidate.state),
                lake_area=float(candidate.Lake_area),
                centroid_lon=float(candidate.centroid_lon),
                centroid_lat=float(candidate.centroid_lat),
            )
            row = local_by_id.loc[lake_id]
            expected_counts = (int(row.n_samples), int(row.n_train), int(row.n_test))
            path = _cache_path(reference, lake_id, primary_root, expanded_root)
            frame = read_curve_cache(
                path,
                lake,
                reference,
                settings,
                HYDROLAKES_ASSET,
                expected_counts=expected_counts,
            )
            if int(frame["block_id"].nunique()) != int(row.n_blocks):
                raise Tier1ExplanatoryError(
                    f"{reference} {lake_id} cached block count differs from result."
                )
            samples[(reference, lake_id)] = frame
            source_paths.append(path)
        results[reference] = tables
    return Tier1Inputs(
        manifest=manifest_tier1,
        candidates=candidates_tier1,
        results=results,
        samples=samples,
        source_paths=tuple(source_paths),
    )


def clean_hydrolakes_value(field: str, value: object) -> float:
    """Clean only documented field-specific sentinels; retain valid negatives."""

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float("nan")
    number = float(numeric)
    if number in HYDROLAKES_SENTINEL_VALUES.get(field, ()):
        return float("nan")
    return number


def _positive_log10(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not np.isfinite(number) or float(number) <= 0:
        return float("nan")
    return float(np.log10(float(number)))


def _safe_sd(values: pd.Series, minimum: int = 2) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < minimum:
        return float("nan")
    return float(clean.std(ddof=1))


def sample_explanatory_metrics(samples: pd.DataFrame) -> dict[str, object]:
    """Calculate availability, class, discrimination, endpoint, and time metrics."""

    required = (
        "acquisition_time",
        "sensor",
        "reference_tile_count",
        "match_delta_days",
        "index_value",
        "label",
        "split",
        "block_id",
    )
    _require_columns(samples, required, "cached samples")
    if samples.empty or set(samples["split"].astype(str)) != {"train", "test"}:
        raise Tier1ExplanatoryError("Cached samples must contain train and test rows.")
    if samples.groupby("block_id")["split"].nunique().max() != 1:
        raise Tier1ExplanatoryError("Cached samples contain block/split leakage.")
    frame = samples.copy()
    frame["date"] = pd.to_datetime(frame["acquisition_time"], unit="ms", utc=True)
    test = frame[frame["split"] == "test"]
    train = frame[frame["split"] == "train"]
    water = test.loc[test["label"] == 1, "index_value"].astype(float)
    nonwater = test.loc[test["label"] == 0, "index_value"].astype(float)
    if water.empty or nonwater.empty:
        raise Tier1ExplanatoryError("Held-out samples must contain both classes.")
    curves = discrimination_curves(test.assign(split="test"))
    ks = float(ks_2samp(water, nonwater, alternative="two-sided").statistic)

    acquisition = (
        frame.groupby("acquisition_time", sort=True)
        .agg(
            prevalence=("label", "mean"),
            date=("date", "first"),
            reference_tile_count=("reference_tile_count", "first"),
            match_delta_days=("match_delta_days", "first"),
        )
        .reset_index()
    )
    metadata_variation = frame.groupby("acquisition_time")[
        [
            "reference_tile_count",
            "match_delta_days",
        ]
    ].nunique(dropna=False)
    if (metadata_variation > 1).any().any():
        raise Tier1ExplanatoryError(
            "Reference-match metadata varies within an acquisition."
        )
    acquisition["year"] = acquisition["date"].dt.year
    acquisition["month"] = acquisition["date"].dt.month
    annual = acquisition.groupby("year")["prevalence"].mean()
    monthly = acquisition.groupby("month")["prevalence"].mean()
    class_acquisition = (
        frame.groupby(["acquisition_time", "label"])["index_value"]
        .median()
        .unstack("label")
    )
    water_medians = class_acquisition.get(1, pd.Series(dtype=float)).dropna()
    nonwater_medians = class_acquisition.get(0, pd.Series(dtype=float)).dropna()

    sensors = frame["sensor"].dropna().astype(str)
    sensor_order = {name: order for order, name in enumerate(("L5", "L7", "L8", "L9"))}
    sensor_list = sorted(
        sensors.unique(), key=lambda value: (sensor_order.get(value, 99), value)
    )
    tile_counts = pd.to_numeric(acquisition["reference_tile_count"], errors="coerce")
    match_delta = pd.to_numeric(acquisition["match_delta_days"], errors="coerce")
    test_positive = test[test["label"] == 1]
    return {
        "n_unique_acquisitions": int(frame["acquisition_time"].nunique()),
        "n_unique_years": int(frame["date"].dt.year.nunique()),
        "sample_date_start": frame["date"].min().date().isoformat(),
        "sample_date_end": frame["date"].max().date().isoformat(),
        "sensor_count": len(sensor_list),
        "sensor_list": ",".join(sensor_list),
        "reference_tile_count_median": float(tile_counts.median()),
        "reference_tile_count_max": float(tile_counts.max()),
        "match_delta_days_median": float(match_delta.median()),
        "match_delta_days_p95": float(match_delta.quantile(0.95)),
        "match_delta_days_max": float(match_delta.max()),
        "positive_fraction_all": float(frame["label"].mean()),
        "positive_fraction_train": float(train["label"].mean()),
        "positive_fraction_test": float(test["label"].mean()),
        "n_positive_test": int(test["label"].sum()),
        "n_negative_test": int((test["label"] == 0).sum()),
        "n_positive_test_blocks": int(
            test.loc[test["label"] == 1, "block_id"].nunique()
        ),
        "n_negative_test_blocks": int(
            test.loc[test["label"] == 0, "block_id"].nunique()
        ),
        "water_ndwi_median_test": float(water.median()),
        "nonwater_ndwi_median_test": float(nonwater.median()),
        "water_ndwi_q25_test": float(water.quantile(0.25)),
        "water_ndwi_q75_test": float(water.quantile(0.75)),
        "nonwater_ndwi_q25_test": float(nonwater.quantile(0.25)),
        "nonwater_ndwi_q75_test": float(nonwater.quantile(0.75)),
        "median_class_separation_test": float(water.median() - nonwater.median()),
        "test_roc_auc": curves.roc_auc,
        "test_average_precision": curves.average_precision,
        "test_ks_statistic": ks,
        "test_ecdf_overlap_proxy": 1.0 - ks,
        "ndwi_one_fraction_all": float(frame["index_value"].eq(1.0).mean()),
        "ndwi_one_fraction_train": float(train["index_value"].eq(1.0).mean()),
        "ndwi_one_fraction_test": float(test["index_value"].eq(1.0).mean()),
        "ndwi_one_fraction_positive_test": float(
            test_positive["index_value"].eq(1.0).mean()
        ),
        "acquisition_prevalence_sd": _safe_sd(acquisition["prevalence"]),
        "acquisition_prevalence_iqr": float(
            acquisition["prevalence"].quantile(0.75)
            - acquisition["prevalence"].quantile(0.25)
        ),
        "acquisition_prevalence_range": float(
            acquisition["prevalence"].max() - acquisition["prevalence"].min()
        ),
        "annual_prevalence_sd": _safe_sd(annual),
        "annual_prevalence_range": (
            float(annual.max() - annual.min()) if len(annual) >= 2 else float("nan")
        ),
        "n_years_for_annual_variability": len(annual),
        "monthly_prevalence_sd": _safe_sd(monthly),
        "monthly_prevalence_range": (
            float(monthly.max() - monthly.min()) if len(monthly) >= 2 else float("nan")
        ),
        "n_observed_months": len(monthly),
        "water_acquisition_median_ndwi_sd": _safe_sd(
            water_medians, MIN_CLASS_ACQUISITIONS_FOR_SD
        ),
        "nonwater_acquisition_median_ndwi_sd": _safe_sd(
            nonwater_medians, MIN_CLASS_ACQUISITIONS_FOR_SD
        ),
        "n_water_acquisition_medians": len(water_medians),
        "n_nonwater_acquisition_medians": len(nonwater_medians),
    }


def plateau_metrics(
    response: pd.DataFrame,
    *,
    tolerance: float,
    local_threshold: float,
    lolo_threshold: float,
    regional_threshold: float,
) -> dict[str, object]:
    """Measure the contiguous near-optimal held-out interval around the maximum."""

    if tolerance < 0 or not np.isfinite(tolerance):
        raise Tier1ExplanatoryError("Plateau tolerance must be finite and nonnegative.")
    _require_columns(response, ("threshold", "iou"), "threshold response")
    if len(response) < 2:
        raise Tier1ExplanatoryError("Threshold response needs at least two points.")
    thresholds = response["threshold"].to_numpy(dtype=float)
    iou = response["iou"].to_numpy(dtype=float)
    peak_index = int(np.argmax(iou))
    peak = float(iou[peak_index])
    near = iou >= peak - tolerance
    low_index = peak_index
    high_index = peak_index
    while low_index > 0 and near[low_index - 1]:
        low_index -= 1
    while high_index + 1 < len(near) and near[high_index + 1]:
        high_index += 1
    low = float(thresholds[low_index])
    high = float(thresholds[high_index])
    width = high - low
    grid_range = float(thresholds[-1] - thresholds[0])

    def inside(value: float) -> bool:
        return bool(low - RESULT_TOLERANCE <= value <= high + RESULT_TOLERANCE)

    return {
        "test_grid_max_iou": peak,
        "shared_grid_low_threshold": float(thresholds[0]),
        "shared_grid_high_threshold": float(thresholds[-1]),
        "shared_grid_range": grid_range,
        "plateau_low_threshold": low,
        "plateau_high_threshold": high,
        "plateau_width": width,
        "plateau_width_normalized": width / grid_range,
        "plateau_grid_point_count": int(high_index - low_index + 1),
        "regional_threshold_inside_plateau": inside(regional_threshold),
        "lolo_threshold_inside_plateau": inside(lolo_threshold),
        "local_threshold_inside_plateau": inside(local_threshold),
    }


def _assert_close(
    calculated: float,
    saved: float,
    label: str,
    tolerance: float = RESULT_TOLERANCE,
) -> None:
    if not np.isclose(calculated, saved, rtol=0.0, atol=tolerance):
        raise Tier1ExplanatoryError(
            f"{label} recalculated as {calculated:.17g}, not saved {saved:.17g}."
        )


def _verified_outcomes(
    reference: str,
    lake_id: int,
    samples: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, object]:
    local = tables["local"].set_index("Hylak_id").loc[lake_id]
    lolo = tables["lolo"].set_index("held_out_Hylak_id").loc[lake_id]
    regional = tables["regional"].set_index("Hylak_id").loc[lake_id]
    test = samples[samples["split"] == "test"]
    thresholds = {
        "local": float(local.local_threshold),
        "lolo": float(lolo.threshold),
        "regional": float(regional.regional_threshold),
    }
    metrics = {
        name: evaluate_threshold(test["index_value"], test["label"], threshold)
        for name, threshold in thresholds.items()
    }
    saved = {
        "local": (float(local.local_test_iou), float(local.local_test_f1)),
        "lolo": (float(lolo.held_out_test_iou), float(lolo.held_out_test_f1)),
        "regional": (
            float(regional.regional_test_iou),
            float(regional.regional_test_f1),
        ),
    }
    for name in ("local", "lolo", "regional"):
        _assert_close(
            metrics[name]["iou"], saved[name][0], f"{reference} {lake_id} {name} IoU"
        )
        _assert_close(
            metrics[name]["f1"], saved[name][1], f"{reference} {lake_id} {name} F1"
        )
    _assert_close(
        float(lolo.iou_delta_vs_local),
        metrics["lolo"]["iou"] - metrics["local"]["iou"],
        f"{reference} {lake_id} saved LOLO delta",
    )
    _assert_close(
        float(regional.iou_delta_vs_local),
        metrics["regional"]["iou"] - metrics["local"]["iou"],
        f"{reference} {lake_id} saved regional delta",
    )
    if len(test) != int(local.n_test):
        raise Tier1ExplanatoryError(f"{reference} {lake_id} test count differs.")
    return {
        "local_threshold": thresholds["local"],
        "lolo_threshold": thresholds["lolo"],
        "regional_threshold": thresholds["regional"],
        "local_test_iou": metrics["local"]["iou"],
        "lolo_test_iou": metrics["lolo"]["iou"],
        "regional_test_iou": metrics["regional"]["iou"],
        "local_test_f1": metrics["local"]["f1"],
        "lolo_test_f1": metrics["lolo"]["f1"],
        "regional_test_f1": metrics["regional"]["f1"],
        "lolo_iou_delta": metrics["lolo"]["iou"] - metrics["local"]["iou"],
        "regional_iou_delta": (metrics["regional"]["iou"] - metrics["local"]["iou"]),
        "abs_local_to_lolo_threshold": abs(thresholds["local"] - thresholds["lolo"]),
        "abs_local_to_regional_threshold": abs(
            thresholds["local"] - thresholds["regional"]
        ),
        "local_cv_iou": float(local.local_cv_iou),
        "n_samples": int(local.n_samples),
        "n_train": int(local.n_train),
        "n_test": int(local.n_test),
        "n_blocks": int(local.n_blocks),
    }


def build_explanatory_table(
    inputs: Tier1Inputs,
    *,
    plateau_tolerance: float = DEFAULT_PLATEAU_TOLERANCE,
    grid_steps: int = DEFAULT_GRID_STEPS,
) -> pd.DataFrame:
    """Build and validate one descriptive row per Tier 1 lake/reference."""

    manifest_by_id = inputs.manifest.set_index("Hylak_id")
    candidates_by_id = inputs.candidates.set_index("Hylak_id")
    records: list[dict[str, object]] = []
    for reference in REFERENCES:
        reference_samples = {
            lake_id: inputs.samples[(reference, lake_id)] for lake_id in TIER1_IDS
        }
        grid = training_threshold_grid(reference_samples, grid_steps=grid_steps)
        for selection_order, lake_id in enumerate(TIER1_IDS, start=1):
            manifest = manifest_by_id.loc[lake_id]
            candidate = candidates_by_id.loc[lake_id]
            samples = reference_samples[lake_id]
            outcomes = _verified_outcomes(
                reference, lake_id, samples, inputs.results[reference]
            )
            test = samples[samples["split"] == "test"].copy()
            test["split"] = "test"
            response = threshold_response(test, grid)
            plateau = plateau_metrics(
                response,
                tolerance=plateau_tolerance,
                local_threshold=float(outcomes["local_threshold"]),
                lolo_threshold=float(outcomes["lolo_threshold"]),
                regional_threshold=float(outcomes["regional_threshold"]),
            )
            raw = {f"{field}_raw": candidate[field] for field in RAW_HYDROLAKES_FIELDS}
            cleaned = {
                field: clean_hydrolakes_value(field, candidate[field])
                for field in (
                    "Elevation",
                    "Depth_avg",
                    "Res_time",
                    "Wshd_area",
                    "Dis_avg",
                    "Slope_100",
                )
            }
            lake_area = float(candidate.Lake_area)
            record: dict[str, object] = {
                "reference": reference,
                "Hylak_id": lake_id,
                "Lake_name": str(candidate.Lake_name),
                "state": str(candidate.state),
                "selection_order": selection_order,
                "analysis_role": str(manifest.analysis_role),
                "tier_group": (
                    "original6" if lake_id in CURRENT_IDS else "tier1_addition"
                ),
                "Lake_area": lake_area,
                "Lake_type": int(candidate.Lake_type),
                "lake_type_label": TYPE_LABELS.get(
                    int(candidate.Lake_type), f"unknown ({int(candidate.Lake_type)})"
                ),
                **outcomes,
                **raw,
                **cleaned,
                "log10_Lake_area": _positive_log10(lake_area),
                "log10_Wshd_area": _positive_log10(cleaned["Wshd_area"]),
                "log10_Res_time": _positive_log10(cleaned["Res_time"]),
                "log10_Dis_avg": _positive_log10(cleaned["Dis_avg"]),
                **sample_explanatory_metrics(samples),
                **plateau,
            }
            records.append(record)
    table = pd.DataFrame.from_records(records)
    validate_explanatory_table(table)
    return table


def validate_explanatory_table(table: pd.DataFrame) -> None:
    """Enforce the complete 12-lake x 2-reference analytical design."""

    if len(table) != len(TIER1_IDS) * len(REFERENCES):
        raise Tier1ExplanatoryError("Explanatory table must contain exactly 24 rows.")
    if table[["reference", "Hylak_id"]].duplicated().any():
        raise Tier1ExplanatoryError("Duplicate reference x Hylak_id rows found.")
    expected = [
        (reference, lake_id) for reference in REFERENCES for lake_id in TIER1_IDS
    ]
    actual = list(zip(table["reference"], table["Hylak_id"], strict=True))
    if actual != expected:
        raise Tier1ExplanatoryError(
            "Explanatory table does not preserve reference/Tier 1 authored order."
        )
    outcomes = (
        "local_threshold",
        "lolo_threshold",
        "regional_threshold",
        "local_test_iou",
        "lolo_test_iou",
        "regional_test_iou",
        "local_test_f1",
        "lolo_test_f1",
        "regional_test_f1",
        "lolo_iou_delta",
        "regional_iou_delta",
    )
    if table[list(outcomes)].isna().any().any():
        raise Tier1ExplanatoryError("A required transfer outcome is missing.")
    for reference in REFERENCES:
        subset = table[table["reference"] == reference]
        if len(subset) != 12:
            raise Tier1ExplanatoryError(f"{reference} must contain exactly 12 lakes.")


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(spearmanr(x.to_numpy(float), y.to_numpy(float)).statistic)


def build_association_table(table: pd.DataFrame) -> pd.DataFrame:
    """Calculate pre-specified descriptive Spearman and influence summaries."""

    _require_columns(
        table,
        (
            "reference",
            "Hylak_id",
            "Lake_name",
            *ASSOCIATION_VARIABLES,
            *ASSOCIATION_OUTCOMES,
        ),
        "explanatory table",
    )
    records: list[dict[str, object]] = []
    for reference in REFERENCES:
        ref = table[table["reference"] == reference]
        for outcome in ASSOCIATION_OUTCOMES:
            for variable in ASSOCIATION_VARIABLES:
                complete = ref[["Hylak_id", "Lake_name", variable, outcome]].dropna()
                n_complete = len(complete)
                rho = float("nan")
                leave_out: list[tuple[int, str, float]] = []
                notes: list[str] = [
                    "Descriptive only; no causal or significance claim."
                ]
                if n_complete >= MIN_CORRELATION_LAKES:
                    rho = _spearman(complete[variable], complete[outcome])
                    if np.isnan(rho):
                        notes.append(
                            "Correlation undefined because a variable is constant."
                        )
                    for row in complete.itertuples(index=False):
                        reduced = complete[complete["Hylak_id"] != row.Hylak_id]
                        if len(reduced) < MIN_CORRELATION_LAKES:
                            continue
                        reduced_rho = _spearman(reduced[variable], reduced[outcome])
                        if np.isfinite(reduced_rho):
                            leave_out.append(
                                (int(row.Hylak_id), str(row.Lake_name), reduced_rho)
                            )
                else:
                    notes.append(
                        f"Not calculated: fewer than {MIN_CORRELATION_LAKES} complete lakes."
                    )
                if leave_out:
                    influence = max(
                        leave_out,
                        key=lambda item: abs(item[2] - rho) if np.isfinite(rho) else 0,
                    )
                    rho_min = min(item[2] for item in leave_out)
                    rho_max = max(item[2] for item in leave_out)
                    influential = f"{influence[1]} [{influence[0]}]"
                else:
                    rho_min = rho_max = float("nan")
                    influential = ""
                    if n_complete >= MIN_CORRELATION_LAKES:
                        notes.append(
                            "Leave-one-lake-out range unavailable under the eight-lake minimum."
                        )
                records.append(
                    {
                        "reference": reference,
                        "outcome": outcome,
                        "explanatory_variable": variable,
                        "n_complete": n_complete,
                        "spearman_rho": rho,
                        "leave_one_lake_out_rho_min": rho_min,
                        "leave_one_lake_out_rho_max": rho_max,
                        "most_influential_lake": influential,
                        "notes": " ".join(notes),
                    }
                )
    return pd.DataFrame.from_records(records)


def _case_assessment(row: pd.Series) -> str:
    if (
        row["test_roc_auc"] < 0.85
        or row["test_ks_statistic"] < 0.60
        or row["local_test_iou"] < 0.65
    ):
        return "limited local ceiling / weaker class separation"
    if row["lolo_iou_delta"] <= -0.03 and not bool(
        row["regional_threshold_inside_plateau"]
    ):
        return "good discrimination but shifted or narrow threshold geometry"
    if (
        row["lolo_iou_delta"] >= -0.03
        and bool(row["regional_threshold_inside_plateau"])
        and row["plateau_width_normalized"] >= 0.08
    ):
        return "good discrimination and broad threshold plateau"
    return "mixed or inconclusive threshold/discrimination evidence"


def build_case_studies(table: pd.DataFrame) -> pd.DataFrame:
    """Extract pre-declared cases and report numerical, non-intrinsic diagnoses."""

    indexed = table.set_index(["reference", "Hylak_id"])
    records: list[dict[str, object]] = []
    for reference, lake_id, hypothesis in CASE_STUDIES:
        if (reference, lake_id) not in indexed.index:
            raise Tier1ExplanatoryError(
                f"Missing requested case study {reference} {lake_id}."
            )
        row = indexed.loc[(reference, lake_id)]
        assessment = _case_assessment(row)
        records.append(
            {
                "reference": reference,
                "Hylak_id": lake_id,
                "Lake_name": row["Lake_name"],
                "expected_hypothesis": hypothesis,
                "data_assessment": assessment,
                "local_test_iou": row["local_test_iou"],
                "lolo_test_iou": row["lolo_test_iou"],
                "lolo_iou_delta": row["lolo_iou_delta"],
                "test_roc_auc": row["test_roc_auc"],
                "test_average_precision": row["test_average_precision"],
                "test_ks_statistic": row["test_ks_statistic"],
                "median_class_separation_test": row["median_class_separation_test"],
                "local_threshold": row["local_threshold"],
                "lolo_threshold": row["lolo_threshold"],
                "regional_threshold": row["regional_threshold"],
                "abs_local_to_regional_threshold": row[
                    "abs_local_to_regional_threshold"
                ],
                "plateau_low_threshold": row["plateau_low_threshold"],
                "plateau_high_threshold": row["plateau_high_threshold"],
                "plateau_width_normalized": row["plateau_width_normalized"],
                "plateau_grid_point_count": row["plateau_grid_point_count"],
                "regional_threshold_inside_plateau": row[
                    "regional_threshold_inside_plateau"
                ],
                "n_positive_test": row["n_positive_test"],
                "n_positive_test_blocks": row["n_positive_test_blocks"],
                "test_grid_max_iou": row["test_grid_max_iou"],
                "test_grid_max_iou_gap_vs_lolo": (
                    row["test_grid_max_iou"] - row["lolo_test_iou"]
                ),
                "evidence_note": (
                    "Exploratory mechanism label derived from held-out "
                    "discrimination, saved-threshold distance, and the contiguous "
                    "near-optimal response interval; not an intrinsic lake class."
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def build_provenance(
    inputs: Tier1Inputs,
    *,
    grid_steps: int,
    plateau_tolerance: float,
) -> dict[str, object]:
    """Return a compact, hash-backed provenance record for local inputs."""

    files = []
    for path in inputs.source_paths:
        resolved = path.resolve()
        try:
            label = str(resolved.relative_to(REPOSITORY_ROOT))
        except ValueError:
            label = str(resolved)
        files.append({"path": label.replace("\\", "/"), "sha256": _sha256(resolved)})
    return {
        "analysis": "Tier 1 local threshold-transfer explanation",
        "analysis_date": dt.datetime.now(dt.UTC).date().isoformat(),
        "code_version_git_commit": _git_commit(REPOSITORY_ROOT),
        "references": list(REFERENCES),
        "tier1_hylak_ids": list(TIER1_IDS),
        "grid_steps": int(grid_steps),
        "plateau_tolerance_iou": float(plateau_tolerance),
        "split_authority": (
            "Cached block_id and split columns are authoritative; coordinates "
            "were not used to rebuild spatial blocks."
        ),
        "result_reproduction_absolute_tolerance": RESULT_TOLERANCE,
        "hydrolakes_sentinel_values": {
            key: list(values) for key, values in HYDROLAKES_SENTINEL_VALUES.items()
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "source_files": files,
    }


def run_tier1_explanatory(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    candidate_path: str | Path = DEFAULT_CANDIDATE_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    primary_cache_dir: str | Path = DEFAULT_PRIMARY_CACHE_DIR,
    expanded_cache_dir: str | Path = DEFAULT_EXPANDED_CACHE_DIR,
    plateau_tolerance: float = DEFAULT_PLATEAU_TOLERANCE,
    grid_steps: int = DEFAULT_GRID_STEPS,
) -> Tier1ExplanatoryResult:
    """Run the complete strictly local explanatory analysis in memory."""

    inputs = load_tier1_inputs(
        data_dir=data_dir,
        candidate_path=candidate_path,
        manifest_path=manifest_path,
        primary_cache_dir=primary_cache_dir,
        expanded_cache_dir=expanded_cache_dir,
    )
    table = build_explanatory_table(
        inputs,
        plateau_tolerance=plateau_tolerance,
        grid_steps=grid_steps,
    )
    associations = build_association_table(table)
    case_studies = build_case_studies(table)
    provenance = build_provenance(
        inputs,
        grid_steps=grid_steps,
        plateau_tolerance=plateau_tolerance,
    )
    return Tier1ExplanatoryResult(
        table=table,
        associations=associations,
        case_studies=case_studies,
        provenance=provenance,
        inputs=inputs,
    )


def write_explanatory_outputs(
    result: Tier1ExplanatoryResult,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path, Path, Path]:
    """Write only new explanatory products; never alter source caches/results."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    table_path = root / "tier1_transfer_explanatory.csv"
    associations_path = root / "tier1_transfer_explanatory_associations.csv"
    cases_path = root / "tier1_transfer_case_studies.csv"
    provenance_path = root / "tier1_transfer_explanatory_provenance.json"
    result.table.to_csv(table_path, index=False, float_format="%.17g")
    result.associations.to_csv(associations_path, index=False, float_format="%.17g")
    result.case_studies.to_csv(cases_path, index=False, float_format="%.17g")
    provenance_path.write_text(
        json.dumps(result.provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return table_path, associations_path, cases_path, provenance_path


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 2:
        raise argparse.ArgumentTypeError("must be at least two")
    return number


def _nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be numeric") from exc
    if not np.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Explain completed Tier 1 OPERA/JRC transfer from local compact "
            "results and authoritative cached splits. No Earth Engine call occurs."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--primary-cache-dir", type=Path, default=DEFAULT_PRIMARY_CACHE_DIR
    )
    parser.add_argument(
        "--expanded-cache-dir", type=Path, default=DEFAULT_EXPANDED_CACHE_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--plateau-tolerance",
        type=_nonnegative_float,
        default=DEFAULT_PLATEAU_TOLERANCE,
    )
    parser.add_argument("--grid-steps", type=_positive_int, default=DEFAULT_GRID_STEPS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_tier1_explanatory(
            data_dir=args.data_dir,
            candidate_path=args.candidate_csv,
            manifest_path=args.manifest,
            primary_cache_dir=args.primary_cache_dir,
            expanded_cache_dir=args.expanded_cache_dir,
            plateau_tolerance=args.plateau_tolerance,
            grid_steps=args.grid_steps,
        )
        paths = write_explanatory_outputs(result, args.output_dir)
        print(
            result.table[
                [
                    "reference",
                    "Hylak_id",
                    "Lake_name",
                    "test_roc_auc",
                    "test_ks_statistic",
                    "plateau_width_normalized",
                    "lolo_test_iou",
                    "lolo_iou_delta",
                ]
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )
        print(f"\nValidated {len(result.table)} lake x reference rows.")
        for path in paths:
            print(f"Wrote {path}")
        return 0
    except (OSError, Tier1ExplanatoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
