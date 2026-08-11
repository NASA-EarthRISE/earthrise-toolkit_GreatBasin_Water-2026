"""Cache and evaluate held-out samples for quick-transfer curve plots.

Only cache misses touch Earth Engine, using the existing bounded
``computeFeatures`` path. Threshold response, ROC, precision-recall, and QA are
pure local pandas/NumPy/scikit-learn operations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .discover_lakes import initialize_earth_engine
from .quick_sampling import (
    LakeSpec,
    QuickSamplingSettings,
    build_buffered_lake_aoi,
    build_labelled_collection,
    build_sample_collection,
    evaluate_threshold,
    normalize_reference,
    sample_collection_to_dataframe,
)
from .quick_transfer import TransferError, prepare_lake_samples

CACHE_VERSION = 1
SAMPLE_SCALE_M = 30
SUPPORTED_CURVE_REFERENCES = ("OPERA", "JRC")

CACHE_COLUMNS = (
    "Hylak_id",
    "Lake_name",
    "state",
    "Lake_area",
    "reference",
    "index",
    "index_value",
    "label",
    "split",
    "block_id",
    "lake_block_id",
    "sensor",
    "acquisition_time",
    "reference_time",
    "match_delta_days",
    "reference_tile_count",
    "lon",
    "lat",
    "cache_version",
    "hydrolakes_asset",
    "date_start",
    "date_end",
    "buffer_km",
    "samples_per_image",
    "sample_scale_m",
    "join_window_days",
    "cloud_cover_max",
    "block_size_m",
    "train_fraction",
    "random_seed",
)


class CurveDataError(RuntimeError):
    """Raised when samples or summaries cannot support valid curve plots."""


@dataclass(frozen=True)
class DiscriminationCurves:
    """Held-out ROC and precision-recall coordinates and scalar summaries."""

    fpr: np.ndarray
    tpr: np.ndarray
    recall: np.ndarray
    precision: np.ndarray
    roc_auc: float
    average_precision: float
    positive_fraction: float


@dataclass(frozen=True)
class CurveCacheLoad:
    """Cache-backed samples and paths newly retrieved during this call."""

    samples: Mapping[tuple[str, int], pd.DataFrame]
    created_paths: tuple[Path, ...]
    project: str | None


def cache_path(cache_dir: str | Path, reference: str, lake_id: int) -> Path:
    """Return a stable ignored CSV cache path."""

    normalized = normalize_reference(reference)
    if normalized not in SUPPORTED_CURVE_REFERENCES:
        raise CurveDataError("Curve caches support OPERA and JRC only.")
    return Path(cache_dir) / f"{normalized.lower()}_{int(lake_id)}.csv"


def expected_cache_settings(
    reference: str,
    settings: QuickSamplingSettings,
    hydrolakes_asset: str,
) -> dict[str, object]:
    """Return provenance values that must match before a cache is reused."""

    start, end = settings.reference_window(reference)
    return {
        "cache_version": CACHE_VERSION,
        "hydrolakes_asset": hydrolakes_asset,
        "date_start": start.isoformat(),
        "date_end": end.isoformat(),
        "buffer_km": settings.buffer_km,
        "samples_per_image": settings.samples_per_image,
        "sample_scale_m": SAMPLE_SCALE_M,
        "join_window_days": settings.join_window_days(reference),
        "cloud_cover_max": settings.cloud_cover_max,
        "block_size_m": settings.block_size_m,
        "train_fraction": settings.train_fraction,
        "random_seed": settings.random_seed,
    }


def _constant_matches(series: pd.Series, expected: object) -> bool:
    if series.isna().any() or series.empty:
        return False
    if isinstance(expected, float):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        return bool(
            np.isfinite(values).all()
            and np.allclose(values, expected, rtol=0.0, atol=1e-12)
        )
    return set(series.astype(str)) == {str(expected)}


def validate_cached_samples(
    frame: pd.DataFrame,
    lake: LakeSpec,
    reference: str,
    settings: QuickSamplingSettings,
    hydrolakes_asset: str,
    *,
    expected_counts: tuple[int, int, int] | None = None,
) -> pd.DataFrame:
    """Validate cache identity, provenance, split integrity, and row counts."""

    normalized = normalize_reference(reference)
    missing = sorted(set(CACHE_COLUMNS).difference(frame.columns))
    if missing:
        raise CurveDataError(
            f"{normalized} {lake.hylak_id} cache is missing columns: "
            + ", ".join(missing)
            + "."
        )
    if frame.empty:
        raise CurveDataError(f"{normalized} {lake.hylak_id} cache is empty.")

    result = frame.loc[:, CACHE_COLUMNS].copy()
    numeric_columns = (
        "Hylak_id",
        "Lake_area",
        "index_value",
        "label",
        "acquisition_time",
        "lon",
        "lat",
    )
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[list(numeric_columns)].isna().any().any():
        raise CurveDataError(
            f"{normalized} {lake.hylak_id} cache has invalid numeric values."
        )
    if not np.isfinite(
        result[["index_value", "acquisition_time", "lon", "lat"]].to_numpy(dtype=float)
    ).all():
        raise CurveDataError(
            f"{normalized} {lake.hylak_id} cache has non-finite scores/coordinates."
        )
    if not result["Hylak_id"].eq(lake.hylak_id).all():
        raise CurveDataError(f"Cache lake ID does not match {lake.hylak_id}.")
    if set(result["Lake_name"].astype(str)) != {lake.lake_name}:
        raise CurveDataError(f"Cache lake name does not match {lake.lake_name!r}.")
    if set(result["state"].astype(str)) != {lake.state}:
        raise CurveDataError(f"Cache state does not match {lake.state!r}.")
    if not np.allclose(
        result["Lake_area"].to_numpy(dtype=float),
        lake.lake_area,
        rtol=0.0,
        atol=1e-12,
    ):
        raise CurveDataError(f"Cache lake area does not match {lake.lake_area}.")
    if set(result["reference"].astype(str).str.upper()) != {normalized}:
        raise CurveDataError(f"Cache reference does not match {normalized}.")
    if set(result["index"].astype(str).str.upper()) != {"NDWI"}:
        raise CurveDataError(f"{normalized} {lake.hylak_id} cache is not NDWI.")
    if not result["label"].isin([0, 1]).all():
        raise CurveDataError(f"{normalized} {lake.hylak_id} labels are not binary.")
    result["Hylak_id"] = result["Hylak_id"].astype(int)
    result["label"] = result["label"].astype(int)

    splits = set(result["split"].astype(str))
    if splits != {"train", "test"}:
        raise CurveDataError(
            f"{normalized} {lake.hylak_id} splits must be train/test; found {splits}."
        )
    if result[["block_id", "lake_block_id"]].isna().any().any():
        raise CurveDataError(
            f"{normalized} {lake.hylak_id} cache has missing block IDs."
        )
    block_ids = result["block_id"].astype(str)
    lake_block_ids = result["lake_block_id"].astype(str)
    if block_ids.str.strip().eq("").any() or lake_block_ids.str.strip().eq("").any():
        raise CurveDataError(f"{normalized} {lake.hylak_id} cache has empty block IDs.")
    expected_lake_blocks = str(lake.hylak_id) + ":" + block_ids
    if not lake_block_ids.eq(expected_lake_blocks).all():
        raise CurveDataError(
            f"{normalized} {lake.hylak_id} cache has inconsistent lake_block_id values."
        )
    if (
        result.assign(block_id=block_ids).groupby("block_id")["split"].nunique().max()
        != 1
    ):
        raise CurveDataError(
            f"{normalized} {lake.hylak_id} cache has block split leakage."
        )
    test = result[result["split"] == "test"]
    if test["label"].nunique() != 2:
        raise CurveDataError(
            f"{normalized} {lake.hylak_id} test split must contain both classes."
        )

    for column, expected in expected_cache_settings(
        normalized, settings, hydrolakes_asset
    ).items():
        if not _constant_matches(result[column], expected):
            raise CurveDataError(
                f"{normalized} {lake.hylak_id} cache provenance mismatch for "
                f"{column}; use --refresh-cache."
            )
    if expected_counts is not None:
        actual = (
            len(result),
            int((result["split"] == "train").sum()),
            int((result["split"] == "test").sum()),
        )
        if actual != tuple(int(value) for value in expected_counts):
            raise CurveDataError(
                f"{normalized} {lake.hylak_id} cache counts {actual} do not match "
                f"the completed transfer result {expected_counts}."
            )

    result["acquisition_date"] = pd.to_datetime(
        result["acquisition_time"], unit="ms", utc=True
    )
    return result.sort_values(
        ["acquisition_time", "lon", "lat"], kind="mergesort"
    ).reset_index(drop=True)


def read_curve_cache(
    path: str | Path,
    lake: LakeSpec,
    reference: str,
    settings: QuickSamplingSettings,
    hydrolakes_asset: str,
    *,
    expected_counts: tuple[int, int, int] | None = None,
) -> pd.DataFrame:
    """Read and validate one prepared-sample CSV cache."""

    cache_file = Path(path)
    try:
        frame = pd.read_csv(
            cache_file,
            dtype={"block_id": "string", "lake_block_id": "string"},
        )
    except (OSError, pd.errors.ParserError) as exc:
        raise CurveDataError(f"Could not read curve cache {cache_file}: {exc}") from exc
    return validate_cached_samples(
        frame,
        lake,
        reference,
        settings,
        hydrolakes_asset,
        expected_counts=expected_counts,
    )


def write_curve_cache(
    samples: pd.DataFrame,
    path: str | Path,
    lake: LakeSpec,
    reference: str,
    settings: QuickSamplingSettings,
    hydrolakes_asset: str,
    *,
    expected_counts: tuple[int, int, int] | None = None,
) -> pd.DataFrame:
    """Add provenance, validate, and write one compact CSV cache."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = samples.copy()
    for column, value in expected_cache_settings(
        reference, settings, hydrolakes_asset
    ).items():
        frame[column] = value
    validated = validate_cached_samples(
        frame,
        lake,
        reference,
        settings,
        hydrolakes_asset,
        expected_counts=expected_counts,
    )
    validated.loc[:, CACHE_COLUMNS].to_csv(
        output,
        index=False,
        float_format="%.17g",
    )
    return validated


def retrieve_curve_samples(
    ee_module: Any,
    hydrolakes_asset: str,
    lake: LakeSpec,
    reference: str,
    settings: QuickSamplingSettings,
) -> pd.DataFrame:
    """Retrieve once through computeFeatures and attach the existing block split."""

    normalized = normalize_reference(reference)
    _lake_feature, buffered_aoi = build_buffered_lake_aoi(
        ee_module,
        hydrolakes_asset,
        lake,
        settings.buffer_km,
    )
    labelled = build_labelled_collection(
        ee_module,
        buffered_aoi,
        normalized,
        settings,
    )
    sampled = build_sample_collection(
        ee_module,
        labelled,
        buffered_aoi,
        lake,
        normalized,
        settings,
    )
    transfer = sample_collection_to_dataframe(ee_module, sampled, settings)
    try:
        return prepare_lake_samples(
            transfer.frame,
            lake,
            normalized,
            settings,
        ).samples
    except TransferError as exc:
        raise CurveDataError(
            f"{normalized} {lake.hylak_id} samples could not preserve the "
            f"transfer split: {exc}"
        ) from exc


def load_or_retrieve_curve_samples(
    lakes: Sequence[LakeSpec],
    references: Sequence[str],
    settings: QuickSamplingSettings,
    cache_dir: str | Path,
    expected_counts: Mapping[tuple[str, int], tuple[int, int, int]],
    *,
    hydrolakes_asset: str | None = None,
    refresh_cache: bool = False,
    cache_only: bool = False,
    initializer: Callable[[], tuple[Any, str]] = initialize_earth_engine,
    retriever: Callable[
        [Any, str, LakeSpec, str, QuickSamplingSettings], pd.DataFrame
    ] = retrieve_curve_samples,
    sample_validator: Callable[[str, LakeSpec, pd.DataFrame], None] | None = None,
    verbose: bool = True,
) -> CurveCacheLoad:
    """Load caches and retrieve each missing lake/reference exactly once."""

    normalized_references = tuple(normalize_reference(ref) for ref in references)
    if any(ref not in SUPPORTED_CURVE_REFERENCES for ref in normalized_references):
        raise CurveDataError("Curve plots support OPERA and JRC only.")
    requested = [
        (reference, lake) for reference in normalized_references for lake in lakes
    ]
    requested_keys = [(reference, lake.hylak_id) for reference, lake in requested]
    if len(set(requested_keys)) != len(requested_keys):
        raise CurveDataError("Curve lake/reference requests must be unique.")
    if hydrolakes_asset is None:
        raise CurveDataError("A HydroLAKES asset is required for curve-cache identity.")
    samples: dict[tuple[str, int], pd.DataFrame] = {}
    missing: list[tuple[str, LakeSpec, Path]] = []
    for reference, lake in requested:
        key = (reference, lake.hylak_id)
        if key not in expected_counts:
            raise CurveDataError(
                f"No expected transfer counts are available for {key}."
            )
        path = cache_path(cache_dir, reference, lake.hylak_id)
        if path.is_file() and not refresh_cache:
            samples[key] = read_curve_cache(
                path,
                lake,
                reference,
                settings,
                hydrolakes_asset,
                expected_counts=expected_counts[key],
            )
            if sample_validator is not None:
                sample_validator(reference, lake, samples[key])
            if verbose:
                print(f"Loaded cache: {path} ({len(samples[key])} rows)")
        else:
            missing.append((reference, lake, path))

    if missing and cache_only:
        paths = ", ".join(str(path) for _, _, path in missing)
        raise CurveDataError(
            f"Curve caches are missing and --cache-only was set: {paths}"
        )
    if not missing:
        return CurveCacheLoad(samples=samples, created_paths=(), project=None)
    ee_module, project = initializer()
    if verbose:
        print(f"Earth Engine project: {project}")
    created: list[Path] = []
    for reference, lake, path in missing:
        if verbose:
            print(
                f"Retrieving {reference} samples for {lake.lake_name} "
                f"[{lake.hylak_id}]..."
            )
        frame = retriever(
            ee_module,
            hydrolakes_asset,
            lake,
            reference,
            settings,
        )
        key = (reference, lake.hylak_id)
        if sample_validator is not None:
            sample_validator(reference, lake, frame)
        samples[key] = write_curve_cache(
            frame,
            path,
            lake,
            reference,
            settings,
            hydrolakes_asset,
            expected_counts=expected_counts[key],
        )
        created.append(path)
        if verbose:
            test_count = int((samples[key]["split"] == "test").sum())
            print(f"  cached {len(samples[key])} rows ({test_count} test): {path}")
    return CurveCacheLoad(
        samples=samples,
        created_paths=tuple(created),
        project=project,
    )


def _validated_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if "split" not in frame.columns:
        raise CurveDataError("Samples are missing the split column.")
    actual = set(frame["split"].astype(str))
    if actual != {split}:
        raise CurveDataError(
            f"Expected only split={split!r} observations; found {sorted(actual)}."
        )
    if "index_value" not in frame.columns or "label" not in frame.columns:
        raise CurveDataError("Samples require index_value and label columns.")
    values = pd.to_numeric(frame["index_value"], errors="coerce").to_numpy(float)
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if not np.isfinite(values).all() or labels.isna().any():
        raise CurveDataError("Curve samples contain invalid scores or labels.")
    if not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise CurveDataError("Curve samples must contain both binary classes.")
    result = frame.copy()
    result["index_value"] = values
    result["label"] = labels.astype(int)
    return result


def training_threshold_grid(
    samples_by_lake: Mapping[int, pd.DataFrame],
    *,
    grid_steps: int = 150,
) -> np.ndarray:
    """Build the transfer-style shared grid from training observations only."""

    if grid_steps < 2:
        raise CurveDataError("grid_steps must be at least two.")
    if not samples_by_lake:
        raise CurveDataError("At least one lake is required for a threshold grid.")
    bounds: list[tuple[float, float]] = []
    for lake_id in sorted(samples_by_lake):
        frame = samples_by_lake[lake_id]
        train = frame.loc[frame["split"] == "train"].copy()
        train["split"] = "train"
        validated = _validated_split(train, "train")
        low, high = np.percentile(validated["index_value"], [0.5, 99.5])
        bounds.append((float(low), float(high)))
    low = min(item[0] for item in bounds)
    high = max(item[1] for item in bounds)
    if not np.isfinite([low, high]).all() or low >= high:
        raise CurveDataError("Training scores do not define a valid threshold range.")
    return np.linspace(low, high, grid_steps)


def threshold_response(
    test_samples: pd.DataFrame,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """Evaluate deterministic ``score >= threshold`` metrics on test rows only."""

    test = _validated_split(test_samples, "test")
    grid = np.asarray(thresholds, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all():
        raise CurveDataError("Threshold grid must contain at least two finite values.")
    grid = np.unique(grid)
    rows = []
    for threshold in grid:
        metrics = evaluate_threshold(
            test["index_value"].to_numpy(),
            test["label"].to_numpy(),
            float(threshold),
        )
        rows.append({"threshold": float(threshold), **metrics})
    return pd.DataFrame.from_records(rows)


def discrimination_curves(test_samples: pd.DataFrame) -> DiscriminationCurves:
    """Compute held-out ROC/PR using raw NDWI as the positive-water score."""

    test = _validated_split(test_samples, "test")
    labels = test["label"].to_numpy(dtype=int)
    scores = test["index_value"].to_numpy(dtype=float)
    fpr, tpr, _roc_thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
        drop_intermediate=False,
    )
    precision, recall, _pr_thresholds = precision_recall_curve(
        labels,
        scores,
        pos_label=1,
    )
    return DiscriminationCurves(
        fpr=fpr,
        tpr=tpr,
        recall=recall[::-1],
        precision=precision[::-1],
        roc_auc=float(roc_auc_score(labels, scores)),
        average_precision=float(average_precision_score(labels, scores)),
        positive_fraction=float(labels.mean()),
    )


def build_curve_summary(
    reference: str,
    lakes: Sequence[LakeSpec],
    samples_by_id: Mapping[int, pd.DataFrame],
    local: pd.DataFrame,
    lolo: pd.DataFrame,
    regional: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Verify cached held-out metrics against saved results and summarize curves."""

    normalized = normalize_reference(reference)
    local_by_id = local.set_index("Hylak_id")
    lolo_by_id = lolo.set_index("held_out_Hylak_id")
    regional_by_id = regional.set_index("Hylak_id")
    records: list[dict[str, object]] = []
    for lake in lakes:
        lake_id = lake.hylak_id
        if lake_id not in samples_by_id:
            raise CurveDataError(f"Missing cached samples for {normalized} {lake_id}.")
        test = (
            samples_by_id[lake_id].loc[samples_by_id[lake_id]["split"] == "test"].copy()
        )
        test["split"] = "test"
        test = _validated_split(test, "test")
        local_row = local_by_id.loc[lake_id]
        lolo_row = lolo_by_id.loc[lake_id]
        regional_row = regional_by_id.loc[lake_id]
        thresholds = {
            "local": float(local_row["local_threshold"]),
            "lolo": float(lolo_row["threshold"]),
            "regional": float(regional_row["regional_threshold"]),
        }
        saved_iou = {
            "local": float(local_row["local_test_iou"]),
            "lolo": float(lolo_row["held_out_test_iou"]),
            "regional": float(regional_row["regional_test_iou"]),
        }
        for label, threshold in thresholds.items():
            calculated = evaluate_threshold(
                test["index_value"], test["label"], threshold
            )["iou"]
            if not np.isclose(
                calculated,
                saved_iou[label],
                rtol=0.0,
                atol=tolerance,
            ):
                raise CurveDataError(
                    f"{normalized} {lake_id} cache gives {label} IoU "
                    f"{calculated:.12g}, not saved value {saved_iou[label]:.12g}."
                )
        if len(test) != int(local_row["n_test"]):
            raise CurveDataError(
                f"{normalized} {lake_id} cached test count differs from summary."
            )
        curves = discrimination_curves(test)
        records.append(
            {
                "Hylak_id": lake_id,
                "Lake_name": lake.lake_name,
                "reference": normalized,
                "n_test": len(test),
                "positive_fraction_test": curves.positive_fraction,
                "local_threshold": thresholds["local"],
                "lolo_threshold": thresholds["lolo"],
                "regional_threshold": thresholds["regional"],
                "local_iou": saved_iou["local"],
                "lolo_iou": saved_iou["lolo"],
                "regional_iou": saved_iou["regional"],
                "roc_auc": curves.roc_auc,
                "average_precision": curves.average_precision,
            }
        )
    return pd.DataFrame.from_records(records)
