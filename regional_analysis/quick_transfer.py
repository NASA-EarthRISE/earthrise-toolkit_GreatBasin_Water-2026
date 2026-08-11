"""In-memory cross-lake threshold transfer for the six quick-example lakes.

Earth Engine is used only once per lake/reference to produce bounded samples.
Every transfer fit and evaluation after retrieval is local pandas/NumPy work.
No Earth Engine export task or remote intermediate file is created.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ConfigError, load_config
from .discover_lakes import DiscoveryError, initialize_earth_engine
from .quick_examples import (
    DEFAULT_CANDIDATE_PATH,
    DEFAULT_CONFIG_PATH,
    QUICK_EXAMPLE_SETS,
    CombinationResult,
    run_lake_reference,
)
from .quick_sampling import (
    LakeSpec,
    QuickSamplingError,
    QuickSamplingSettings,
    assign_blocks_and_split,
    evaluate_threshold,
    load_candidate_inventory,
    normalize_reference,
    parse_iso_date,
    select_lake_specs,
    utm_epsg_from_lon_lat,
)

SUPPORTED_TRANSFER_REFERENCES = ("OPERA", "JRC")
DEFAULT_OUTPUT_DIR = Path("data/regional")

LOCAL_COLUMNS = (
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
    "local_threshold_std",
    "local_cv_iou",
    "local_test_iou",
    "local_test_f1",
    "local_test_precision",
    "local_test_recall",
    "sample_retrieval_seconds",
    "local_optimization_seconds",
    "combination_elapsed_seconds",
)

REGIONAL_COLUMNS = (
    "reference",
    "Hylak_id",
    "Lake_name",
    "state",
    "n_train",
    "n_test",
    "regional_threshold",
    "macro_training_iou",
    "lake_training_iou",
    "regional_test_iou",
    "regional_test_f1",
    "regional_test_precision",
    "regional_test_recall",
    "macro_test_iou",
    "local_threshold",
    "local_test_iou",
    "iou_delta_vs_local",
)

LOLO_COLUMNS = (
    "reference",
    "held_out_Hylak_id",
    "held_out_lake",
    "n_training_lakes",
    "threshold",
    "held_out_test_iou",
    "held_out_test_f1",
    "held_out_test_precision",
    "held_out_test_recall",
    "local_threshold",
    "local_test_iou",
    "iou_delta_vs_local",
    "macro_training_iou",
    "n_held_out_test",
    "training_Hylak_ids",
    "macro_lolo_iou",
)


class TransferError(RuntimeError):
    """Raised when samples cannot support a leakage-free transfer analysis."""


@dataclass
class PreparedLake:
    """One locally blocked lake sample retained for all transfer evaluations."""

    lake: LakeSpec
    reference: str
    samples: pd.DataFrame
    local_summary: Mapping[str, Any]
    sample_retrieval_seconds: float
    optimization_seconds: float


@dataclass(frozen=True)
class MacroThresholdResult:
    """Deterministic equal-lake-weighted threshold selection result."""

    threshold: float
    macro_iou: float
    per_lake_iou: Mapping[int, float]
    training_lake_ids: tuple[int, ...]
    grid_scores: pd.DataFrame


@dataclass
class QuickTransferResult:
    """Complete in-memory result for one reference."""

    reference: str
    lake_runs: Mapping[int, CombinationResult]
    prepared_lakes: Mapping[int, PreparedLake]
    local_summary: pd.DataFrame
    cross_iou: pd.DataFrame
    cross_f1: pd.DataFrame
    regional: pd.DataFrame
    lolo: pd.DataFrame
    regional_fit: MacroThresholdResult
    elapsed_seconds: float


@dataclass(frozen=True)
class TransferOutputPaths:
    """Stable local paths for compact transfer outputs."""

    local: Path
    cross_iou: Path
    cross_f1: Path
    regional: Path
    lolo: Path


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise TransferError(f"{label} is missing columns: {', '.join(missing)}.")


def prepare_lake_samples(
    samples: pd.DataFrame,
    lake: LakeSpec,
    reference: str,
    settings: QuickSamplingSettings,
    *,
    local_summary: Mapping[str, Any] | None = None,
    sample_retrieval_seconds: float = float("nan"),
    optimization_seconds: float = float("nan"),
) -> PreparedLake:
    """Validate and retain the existing deterministic 1 km block split."""

    normalized = normalize_reference(reference)
    if normalized not in SUPPORTED_TRANSFER_REFERENCES:
        raise TransferError("Quick transfer supports OPERA and JRC, not Dynamic World.")
    if not isinstance(samples, pd.DataFrame) or samples.empty:
        raise TransferError(f"Hylak_id {lake.hylak_id} has no samples.")

    required = ("index_value", "label", "lon", "lat")
    _require_columns(samples, required, f"Hylak_id {lake.hylak_id} samples")
    frame = samples.copy()
    if "Hylak_id" not in frame.columns:
        frame["Hylak_id"] = lake.hylak_id
    ids = pd.to_numeric(frame["Hylak_id"], errors="coerce")
    if ids.isna().any() or not ids.eq(lake.hylak_id).all():
        raise TransferError(
            f"Sample Hylak_id values conflict with expected ID {lake.hylak_id}."
        )
    frame["Hylak_id"] = ids.astype(int)

    if "reference" not in frame.columns:
        frame["reference"] = normalized
    references = set(frame["reference"].dropna().astype(str).str.upper())
    if references != {normalized}:
        raise TransferError(
            f"Hylak_id {lake.hylak_id} samples contain references {references}; "
            f"expected only {normalized}."
        )

    frame["index_value"] = pd.to_numeric(frame["index_value"], errors="coerce")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    if not np.isfinite(frame["index_value"].to_numpy(dtype=float)).all():
        raise TransferError(f"Hylak_id {lake.hylak_id} has non-finite index values.")
    if frame["label"].isna().any() or not frame["label"].isin([0, 1]).all():
        raise TransferError(f"Hylak_id {lake.hylak_id} labels must be binary.")
    frame["label"] = frame["label"].astype(int)

    epsg = utm_epsg_from_lon_lat(lake.centroid_lon, lake.centroid_lat)
    blocked = assign_blocks_and_split(
        frame,
        epsg,
        block_size_m=settings.block_size_m,
        train_fraction=settings.train_fraction,
        seed=settings.random_seed,
    )
    if blocked.groupby("block_id")["split"].nunique().max() != 1:
        raise TransferError(f"Hylak_id {lake.hylak_id} has block split leakage.")
    blocked["lake_block_id"] = (
        blocked["Hylak_id"].astype(str) + ":" + blocked["block_id"].astype(str)
    )
    for split in ("train", "test"):
        view = blocked[blocked["split"] == split]
        if len(view) < settings.min_split_samples:
            raise TransferError(
                f"Hylak_id {lake.hylak_id} has too few {split} samples."
            )
        if view["label"].nunique() < 2:
            raise TransferError(
                f"Hylak_id {lake.hylak_id} {split} split has only one class."
            )

    summary = {} if local_summary is None else dict(local_summary)
    expected_train = summary.get("n_train")
    expected_test = summary.get("n_test")
    actual_train = int((blocked["split"] == "train").sum())
    actual_test = int((blocked["split"] == "test").sum())
    if expected_train is not None and int(expected_train) != actual_train:
        raise TransferError(
            f"Hylak_id {lake.hylak_id} train split differs from local optimization."
        )
    if expected_test is not None and int(expected_test) != actual_test:
        raise TransferError(
            f"Hylak_id {lake.hylak_id} test split differs from local optimization."
        )

    return PreparedLake(
        lake=lake,
        reference=normalized,
        samples=blocked,
        local_summary=summary,
        sample_retrieval_seconds=sample_retrieval_seconds,
        optimization_seconds=optimization_seconds,
    )


def _split_view(prepared: PreparedLake, split: str) -> pd.DataFrame:
    view = prepared.samples[prepared.samples["split"] == split].copy()
    if view.empty:
        raise TransferError(
            f"Hylak_id {prepared.lake.hylak_id} has no {split} observations."
        )
    return view


def _candidate_grid(
    training_by_lake: Mapping[int, pd.DataFrame], grid_steps: int
) -> np.ndarray:
    if grid_steps < 2:
        raise TransferError("grid_steps must be at least two.")
    bounds = [
        np.percentile(frame["index_value"].to_numpy(dtype=float), [0.5, 99.5])
        for _, frame in sorted(training_by_lake.items())
    ]
    low = float(min(bound[0] for bound in bounds))
    high = float(max(bound[1] for bound in bounds))
    if not np.isfinite([low, high]).all() or low >= high:
        raise TransferError("Training lakes do not provide a valid threshold range.")
    return np.linspace(low, high, grid_steps)


def select_macro_threshold(
    training_by_lake: Mapping[int, pd.DataFrame],
    *,
    candidate_thresholds: Sequence[float] | None = None,
    grid_steps: int = 150,
) -> MacroThresholdResult:
    """Select a threshold by unweighted mean of lake-level training IoUs.

    Production grids span the union of each eligible lake's central 99% range:
    the minimum per-lake 0.5th percentile through the maximum per-lake 99.5th
    percentile. Every lake contributes one IoU to the objective regardless of
    its row count. Ties resolve to the lowest threshold.
    """

    lake_ids = tuple(sorted(int(lake_id) for lake_id in training_by_lake))
    if len(lake_ids) < 2:
        raise TransferError("Macro threshold selection requires at least two lakes.")

    clean: dict[int, pd.DataFrame] = {}
    for lake_id in lake_ids:
        frame = training_by_lake[lake_id]
        _require_columns(
            frame,
            ("Hylak_id", "index_value", "label", "split"),
            f"Training lake {lake_id}",
        )
        if frame.empty:
            raise TransferError(f"Training lake {lake_id} has no observations.")
        if not frame["split"].eq("train").all():
            raise TransferError(
                f"Training lake {lake_id} contains non-training observations."
            )
        ids = pd.to_numeric(frame["Hylak_id"], errors="coerce")
        if ids.isna().any() or not ids.eq(lake_id).all():
            raise TransferError(f"Training lake {lake_id} has conflicting IDs.")
        labels = pd.to_numeric(frame["label"], errors="coerce")
        values = pd.to_numeric(frame["index_value"], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise TransferError(f"Training lake {lake_id} labels must be binary.")
        if labels.nunique() < 2:
            raise TransferError(f"Training lake {lake_id} has only one class.")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise TransferError(f"Training lake {lake_id} has non-finite values.")
        if values.nunique() < 2:
            raise TransferError(f"Training lake {lake_id} has no index variation.")
        normalized = frame.copy()
        normalized["index_value"] = values
        normalized["label"] = labels.astype(int)
        clean[lake_id] = normalized

    if candidate_thresholds is None:
        thresholds = _candidate_grid(clean, grid_steps)
    else:
        thresholds = np.asarray(candidate_thresholds, dtype=float)
        if not len(thresholds) or not np.isfinite(thresholds).all():
            raise TransferError("Candidate thresholds must be finite and non-empty.")
        thresholds = np.unique(thresholds)

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        row: dict[str, float] = {"threshold": float(threshold)}
        lake_scores = []
        for lake_id in lake_ids:
            frame = clean[lake_id]
            iou = evaluate_threshold(
                frame["index_value"], frame["label"], float(threshold)
            )["iou"]
            row[f"iou_{lake_id}"] = iou
            lake_scores.append(iou)
        row["macro_iou"] = float(np.mean(lake_scores))
        rows.append(row)

    scores = pd.DataFrame.from_records(rows).sort_values("threshold", kind="mergesort")
    winner = scores.loc[scores["macro_iou"].idxmax()]
    threshold = float(winner["threshold"])
    per_lake = {lake_id: float(winner[f"iou_{lake_id}"]) for lake_id in lake_ids}
    return MacroThresholdResult(
        threshold=threshold,
        macro_iou=float(winner["macro_iou"]),
        per_lake_iou=per_lake,
        training_lake_ids=lake_ids,
        grid_scores=scores.reset_index(drop=True),
    )


def cross_application_matrices(
    prepared_lakes: Mapping[int, PreparedLake],
    source_thresholds: Mapping[int, float],
    lake_order: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each source-lake threshold on every target lake's test blocks."""

    order = tuple(int(lake_id) for lake_id in lake_order)
    missing = [
        lake_id
        for lake_id in order
        if lake_id not in prepared_lakes or lake_id not in source_thresholds
    ]
    if missing:
        raise TransferError(f"Cross matrix is missing lake IDs: {missing}.")
    iou = pd.DataFrame(index=order, columns=order, dtype=float)
    f1 = pd.DataFrame(index=order, columns=order, dtype=float)
    for source_id in order:
        threshold = float(source_thresholds[source_id])
        if not np.isfinite(threshold):
            raise TransferError(f"Source threshold for {source_id} is not finite.")
        for target_id in order:
            target = _split_view(prepared_lakes[target_id], "test")
            metrics = evaluate_threshold(
                target["index_value"], target["label"], threshold
            )
            iou.loc[source_id, target_id] = metrics["iou"]
            f1.loc[source_id, target_id] = metrics["f1"]
    iou.index.name = "source_Hylak_id"
    iou.columns.name = "target_Hylak_id"
    f1.index.name = "source_Hylak_id"
    f1.columns.name = "target_Hylak_id"
    return iou, f1


def _local_lookup(local_summary: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        local_summary,
        ("Hylak_id", "local_threshold", "local_test_iou"),
        "Local summary",
    )
    if local_summary["Hylak_id"].duplicated().any():
        raise TransferError("Local summary contains duplicate Hylak_id values.")
    return local_summary.set_index("Hylak_id", drop=False)


def evaluate_pooled_regional(
    prepared_lakes: Mapping[int, PreparedLake],
    lake_order: Sequence[int],
    reference: str,
    local_summary: pd.DataFrame,
    *,
    grid_steps: int = 150,
) -> tuple[MacroThresholdResult, pd.DataFrame]:
    """Fit an equal-lake-weighted threshold and score every untouched test set."""

    order = tuple(int(lake_id) for lake_id in lake_order)
    training = {
        lake_id: _split_view(prepared_lakes[lake_id], "train") for lake_id in order
    }
    fit = select_macro_threshold(training, grid_steps=grid_steps)
    local = _local_lookup(local_summary)
    rows: list[dict[str, Any]] = []
    for lake_id in order:
        prepared = prepared_lakes[lake_id]
        testing = _split_view(prepared, "test")
        metrics = evaluate_threshold(
            testing["index_value"], testing["label"], fit.threshold
        )
        local_iou = float(local.loc[lake_id, "local_test_iou"])
        rows.append(
            {
                "reference": reference,
                "Hylak_id": lake_id,
                "Lake_name": prepared.lake.lake_name,
                "state": prepared.lake.state,
                "n_train": len(training[lake_id]),
                "n_test": len(testing),
                "regional_threshold": fit.threshold,
                "macro_training_iou": fit.macro_iou,
                "lake_training_iou": fit.per_lake_iou[lake_id],
                "regional_test_iou": metrics["iou"],
                "regional_test_f1": metrics["f1"],
                "regional_test_precision": metrics["precision"],
                "regional_test_recall": metrics["recall"],
                "local_threshold": float(local.loc[lake_id, "local_threshold"]),
                "local_test_iou": local_iou,
                "iou_delta_vs_local": metrics["iou"] - local_iou,
            }
        )
    result = pd.DataFrame.from_records(rows)
    result["macro_test_iou"] = float(result["regional_test_iou"].mean())
    return fit, result.reindex(columns=REGIONAL_COLUMNS)


def leave_one_lake_out(
    prepared_lakes: Mapping[int, PreparedLake],
    lake_order: Sequence[int],
    reference: str,
    local_summary: pd.DataFrame,
    *,
    grid_steps: int = 150,
) -> pd.DataFrame:
    """Fit on other lakes' training blocks and score only the held-out test set."""

    order = tuple(int(lake_id) for lake_id in lake_order)
    if len(order) < 3:
        raise TransferError("Leave-one-lake-out requires at least three lakes.")
    local = _local_lookup(local_summary)
    rows: list[dict[str, Any]] = []
    for held_out_id in order:
        training_ids = tuple(lake_id for lake_id in order if lake_id != held_out_id)
        training = {
            lake_id: _split_view(prepared_lakes[lake_id], "train")
            for lake_id in training_ids
        }
        fit = select_macro_threshold(training, grid_steps=grid_steps)
        if held_out_id in fit.training_lake_ids:
            raise TransferError("Held-out lake entered threshold fitting.")
        held = prepared_lakes[held_out_id]
        testing = _split_view(held, "test")
        metrics = evaluate_threshold(
            testing["index_value"], testing["label"], fit.threshold
        )
        local_iou = float(local.loc[held_out_id, "local_test_iou"])
        rows.append(
            {
                "reference": reference,
                "held_out_Hylak_id": held_out_id,
                "held_out_lake": held.lake.lake_name,
                "n_training_lakes": len(training_ids),
                "threshold": fit.threshold,
                "held_out_test_iou": metrics["iou"],
                "held_out_test_f1": metrics["f1"],
                "held_out_test_precision": metrics["precision"],
                "held_out_test_recall": metrics["recall"],
                "local_threshold": float(local.loc[held_out_id, "local_threshold"]),
                "local_test_iou": local_iou,
                "iou_delta_vs_local": metrics["iou"] - local_iou,
                "macro_training_iou": fit.macro_iou,
                "n_held_out_test": len(testing),
                "training_Hylak_ids": ",".join(
                    str(lake_id) for lake_id in fit.training_lake_ids
                ),
            }
        )
    result = pd.DataFrame.from_records(rows)
    result["macro_lolo_iou"] = float(result["held_out_test_iou"].mean())
    return result.reindex(columns=LOLO_COLUMNS)


def _local_summary_row(
    result: CombinationResult, prepared: PreparedLake
) -> dict[str, Any]:
    summary = result.summary
    threshold = float(summary.get("_threshold_exact", summary["threshold_mean"]))
    testing = _split_view(prepared, "test")
    exact_test = evaluate_threshold(testing["index_value"], testing["label"], threshold)
    return {
        "reference": result.reference,
        "Hylak_id": result.lake.hylak_id,
        "Lake_name": result.lake.lake_name,
        "state": result.lake.state,
        "Lake_area": result.lake.lake_area,
        "n_samples": len(prepared.samples),
        "n_train": int((prepared.samples["split"] == "train").sum()),
        "n_test": int((prepared.samples["split"] == "test").sum()),
        "n_blocks": int(prepared.samples["block_id"].nunique()),
        "positive_fraction": float(prepared.samples["label"].mean()),
        "local_threshold": threshold,
        "local_threshold_std": summary["threshold_std"],
        "local_cv_iou": summary["cv_iou"],
        "local_test_iou": exact_test["iou"],
        "local_test_f1": exact_test["f1"],
        "local_test_precision": exact_test["precision"],
        "local_test_recall": exact_test["recall"],
        "sample_retrieval_seconds": result.sample_retrieval_seconds,
        "local_optimization_seconds": result.optimization_seconds,
        "combination_elapsed_seconds": summary["elapsed_seconds"],
    }


def run_transfer_experiment(
    ee_module: Any,
    hydrolakes_asset: str,
    lakes: Sequence[LakeSpec],
    reference: str,
    settings: QuickSamplingSettings,
    *,
    verbose: bool = True,
) -> QuickTransferResult:
    """Retrieve each lake once, then run all transfer calculations locally."""

    normalized = normalize_reference(reference)
    if normalized not in SUPPORTED_TRANSFER_REFERENCES:
        raise TransferError("Quick transfer supports OPERA and JRC only.")
    if settings.grid_steps < 2:
        raise TransferError("grid_steps must be at least two.")
    lake_order = tuple(lake.hylak_id for lake in lakes)
    if len(set(lake_order)) != len(lake_order):
        raise TransferError("Transfer lake IDs must be unique.")

    started = time.perf_counter()
    lake_runs: dict[int, CombinationResult] = {}
    prepared: dict[int, PreparedLake] = {}
    local_rows: list[dict[str, Any]] = []
    for lake in lakes:
        if verbose:
            print(
                f"Retrieving {normalized} samples for {lake.lake_name} "
                f"[{lake.hylak_id}]..."
            )
        result = run_lake_reference(
            ee_module, hydrolakes_asset, lake, normalized, settings
        )
        if result.summary["status"] != "ok":
            reason = result.summary.get("skip_reason", "unknown failure")
            raise TransferError(
                f"{lake.lake_name} [{lake.hylak_id}] {normalized} did not "
                f"produce a usable local result: {reason}"
            )
        blocked = prepare_lake_samples(
            result.samples,
            lake,
            normalized,
            settings,
            local_summary=result.summary,
            sample_retrieval_seconds=result.sample_retrieval_seconds,
            optimization_seconds=result.optimization_seconds,
        )
        lake_runs[lake.hylak_id] = result
        prepared[lake.hylak_id] = blocked
        local_rows.append(_local_summary_row(result, blocked))
        if verbose:
            print(
                f"  rows={len(result.samples)}, retrieval="
                f"{result.sample_retrieval_seconds:.2f}s, local threshold="
                f"{result.summary['threshold_mean']:+.4f}, local test IoU="
                f"{result.summary['test_iou']:.3f}"
            )

    local_summary = pd.DataFrame.from_records(local_rows).reindex(columns=LOCAL_COLUMNS)
    thresholds = dict(
        zip(
            local_summary["Hylak_id"].astype(int),
            local_summary["local_threshold"].astype(float),
            strict=True,
        )
    )
    cross_iou, cross_f1 = cross_application_matrices(prepared, thresholds, lake_order)
    local_by_id = local_summary.set_index("Hylak_id")
    for lake_id in lake_order:
        if not np.isclose(
            cross_iou.loc[lake_id, lake_id],
            local_by_id.loc[lake_id, "local_test_iou"],
        ):
            raise TransferError(
                f"Cross-matrix diagonal disagrees with local test IoU for {lake_id}."
            )
    regional_fit, regional = evaluate_pooled_regional(
        prepared,
        lake_order,
        normalized,
        local_summary,
        grid_steps=settings.grid_steps,
    )
    lolo = leave_one_lake_out(
        prepared,
        lake_order,
        normalized,
        local_summary,
        grid_steps=settings.grid_steps,
    )
    return QuickTransferResult(
        reference=normalized,
        lake_runs=lake_runs,
        prepared_lakes=prepared,
        local_summary=local_summary,
        cross_iou=cross_iou,
        cross_f1=cross_f1,
        regional=regional,
        lolo=lolo,
        regional_fit=regional_fit,
        elapsed_seconds=round(time.perf_counter() - started, 2),
    )


def transfer_output_paths(
    output_dir: str | Path, reference: str
) -> TransferOutputPaths:
    normalized_reference = normalize_reference(reference)
    if normalized_reference not in SUPPORTED_TRANSFER_REFERENCES:
        raise TransferError("Quick transfer supports OPERA and JRC only.")
    normalized = normalized_reference.lower()
    prefix = Path(output_dir) / f"western_quick_transfer_{normalized}"
    return TransferOutputPaths(
        local=prefix.with_name(prefix.name + "_local.csv"),
        cross_iou=prefix.with_name(prefix.name + "_cross_matrix_iou.csv"),
        cross_f1=prefix.with_name(prefix.name + "_cross_matrix_f1.csv"),
        regional=prefix.with_name(prefix.name + "_regional.csv"),
        lolo=prefix.with_name(prefix.name + "_lolo.csv"),
    )


def write_transfer_results(
    result: QuickTransferResult, output_dir: str | Path
) -> TransferOutputPaths:
    """Write compact local tables; sampled rows remain only in memory."""

    paths = transfer_output_paths(output_dir, result.reference)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result.local_summary.to_csv(paths.local, index=False)
    result.cross_iou.to_csv(paths.cross_iou, index=True)
    result.cross_f1.to_csv(paths.cross_f1, index=True)
    result.regional.to_csv(paths.regional, index=False)
    result.lolo.to_csv(paths.lolo, index=False)
    return paths


def load_transfer_tables(
    output_dir: str | Path, reference: str
) -> dict[str, pd.DataFrame]:
    """Load compact transfer tables without loading or recreating samples."""

    paths = transfer_output_paths(output_dir, reference)
    cross_iou = pd.read_csv(paths.cross_iou, index_col="source_Hylak_id")
    cross_f1 = pd.read_csv(paths.cross_f1, index_col="source_Hylak_id")
    for matrix in (cross_iou, cross_f1):
        matrix.index = pd.Index(
            pd.to_numeric(matrix.index).astype(int), name="source_Hylak_id"
        )
        matrix.columns = pd.Index(
            [int(column) for column in matrix.columns], name="target_Hylak_id"
        )
    return {
        "local": pd.read_csv(paths.local),
        "cross_iou": cross_iou,
        "cross_f1": cross_f1,
        "regional": pd.read_csv(paths.regional),
        "lolo": pd.read_csv(paths.lolo),
    }


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _at_least_two_int(value: str) -> int:
    number = _positive_int(value)
    if number < 2:
        raise argparse.ArgumentTypeError("must be at least two")
    return number


def _date_argument(value: str):
    try:
        return parse_iso_date(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed-six-lake, one-reference transfer CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Retrieve each saline6 sample once and evaluate NDWI thresholds "
            "between lakes entirely in memory. No Earth Engine export is used."
        )
    )
    parser.add_argument(
        "--reference",
        choices=SUPPORTED_TRANSFER_REFERENCES,
        type=str.upper,
        default="OPERA",
        help="NDWI reference to analyze (default: OPERA; Dynamic World excluded).",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--buffer-km", type=_positive_float, default=5.0)
    parser.add_argument("--samples-per-image", type=_positive_int, default=20)
    parser.add_argument(
        "--max-rows-per-combination", type=_positive_int, default=10_000
    )
    parser.add_argument("--grid-steps", type=_at_least_two_int, default=150)
    parser.add_argument(
        "--analysis-end", type=_date_argument, default=parse_iso_date("2026-08-09")
    )
    parser.add_argument("--jrc-start", type=_date_argument)
    parser.add_argument("--jrc-end", type=_date_argument)
    parser.add_argument("--opera-start", type=_date_argument)
    parser.add_argument("--opera-window-days", type=_positive_int)
    parser.add_argument(
        "--save-results",
        nargs="?",
        const=DEFAULT_OUTPUT_DIR,
        type=Path,
        metavar="DIR",
        help=(
            "Optionally write compact local results to DIR (default when flag "
            "has no value: data/regional). Sample rows are never written."
        ),
    )
    return parser


def _print_result(result: QuickTransferResult) -> None:
    print("\nLocal thresholds and retrieval times:")
    print(
        result.local_summary[
            [
                "Hylak_id",
                "Lake_name",
                "n_samples",
                "sample_retrieval_seconds",
                "local_threshold",
                "local_test_iou",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nCross-lake IoU (rows=threshold source, columns=test target):")
    print(result.cross_iou.to_string(float_format=lambda value: f"{value:.3f}"))
    print(
        f"\nPooled regional threshold={result.regional_fit.threshold:+.4f}; "
        f"macro train IoU={result.regional_fit.macro_iou:.3f}; "
        f"macro test IoU={result.regional['macro_test_iou'].iloc[0]:.3f}"
    )
    print(
        result.regional[
            ["Hylak_id", "Lake_name", "regional_test_iou", "regional_test_f1"]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print("\nLeave-one-lake-out:")
    print(
        result.lolo[
            [
                "held_out_Hylak_id",
                "held_out_lake",
                "threshold",
                "held_out_test_iou",
                "held_out_test_f1",
                "iou_delta_vs_local",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(
        f"Macro LOLO IoU={result.lolo['macro_lolo_iou'].iloc[0]:.3f}; "
        f"total elapsed={result.elapsed_seconds:.2f}s"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        inventory = load_candidate_inventory(args.candidate_csv)
        lakes = select_lake_specs(inventory, QUICK_EXAMPLE_SETS["saline6"])
        settings = QuickSamplingSettings().with_overrides(
            buffer_km=args.buffer_km,
            samples_per_image=args.samples_per_image,
            max_rows_per_combination=args.max_rows_per_combination,
            grid_steps=args.grid_steps,
            analysis_end=args.analysis_end,
            jrc_start=args.jrc_start,
            jrc_end=args.jrc_end,
            opera_start=args.opera_start,
            opera_window_days=args.opera_window_days,
        )
        start, end = settings.reference_window(args.reference)
        print(
            f"{args.reference} six-lake NDWI transfer: {start}..{end}, "
            f"buffer={settings.buffer_km:g} km, "
            f"samples/image={settings.samples_per_image}, "
            f"join window=+/-{settings.join_window_days(args.reference)} day(s)."
        )
        print(
            "Regional thresholds maximize the unweighted mean of per-lake "
            "training IoUs; each LOLO grid uses only its eligible training lakes."
        )
        ee_module, project = initialize_earth_engine()
        print(f"Earth Engine project: {project}")
        result = run_transfer_experiment(
            ee_module,
            config.earth_engine.hydrolakes_asset,
            lakes,
            args.reference,
            settings,
        )
        _print_result(result)
        if args.save_results is not None:
            paths = write_transfer_results(result, args.save_results)
            print("\nWrote compact local results:")
            for path in paths.__dict__.values():
                print(f"  {path}")
        return 0
    except (
        ConfigError,
        DiscoveryError,
        QuickSamplingError,
        TransferError,
        ValueError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
