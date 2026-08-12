"""Cache-aware no-export transfer analysis for the staged lake expansion."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ConfigError, load_config
from .discover_lakes import DiscoveryError, initialize_earth_engine
from .expanded_lake_selection import (
    DEFAULT_MANIFEST_PATH,
    SelectionError,
    selected_ids,
    validate_selection_manifest,
)
from .quick_curve_data import (
    CurveDataError,
    read_curve_cache,
    retrieve_curve_samples,
    write_curve_cache,
)
from .quick_examples import DEFAULT_CANDIDATE_PATH, DEFAULT_CONFIG_PATH
from .quick_sampling import (
    InsufficientSampleError,
    LakeSpec,
    QuickSamplingError,
    QuickSamplingSettings,
    cv_optimal_threshold,
    evaluate_threshold,
    load_candidate_inventory,
    normalize_reference,
    parse_iso_date,
    select_lake_specs,
)
from .quick_transfer import (
    LOCAL_COLUMNS,
    PreparedLake,
    TransferError,
    cross_application_matrices,
    evaluate_pooled_regional,
    leave_one_lake_out,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY_CACHE_DIR = (
    REPOSITORY_ROOT / "data" / "regional" / "quick_transfer_cache"
)
DEFAULT_EXPANDED_CACHE_DIR = (
    REPOSITORY_ROOT / "data" / "regional" / "expanded_transfer_cache"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "regional"
SUPPORTED_REFERENCES = ("OPERA", "JRC")


@dataclass(frozen=True)
class ExpandedOutputPaths:
    local: Path
    cross_iou: Path
    cross_f1: Path
    regional: Path
    lolo: Path
    status: Path
    macro: Path


@dataclass
class ExpandedTransferResult:
    reference: str
    tier: str
    requested_lakes: tuple[LakeSpec, ...]
    prepared_lakes: dict[int, PreparedLake]
    local: pd.DataFrame
    cross_iou: pd.DataFrame
    cross_f1: pd.DataFrame
    regional: pd.DataFrame
    lolo: pd.DataFrame
    status: pd.DataFrame
    macro: pd.DataFrame
    elapsed_seconds: float


def load_manifest_lakes(
    manifest_path: str | Path,
    inventory_path: str | Path,
    tier: str,
) -> tuple[LakeSpec, ...]:
    """Resolve one cumulative tier in its authored selection order."""

    try:
        manifest = pd.read_csv(manifest_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise SelectionError(f"Could not read manifest {manifest_path}: {exc}") from exc
    validate_selection_manifest(manifest)
    ids = selected_ids(tier)
    authored = tuple(
        manifest[manifest["Hylak_id"].isin(ids)]
        .sort_values("selection_order")["Hylak_id"]
        .astype(int)
    )
    if authored != ids:
        raise SelectionError(f"Manifest order for {tier} differs from the design.")
    inventory = load_candidate_inventory(inventory_path)
    return select_lake_specs(inventory, ids)


def expanded_output_paths(
    output_dir: str | Path, reference: str, tier: str
) -> ExpandedOutputPaths:
    """Return deterministic compact result paths for one staged run."""

    normalized = normalize_reference(reference)
    if normalized not in SUPPORTED_REFERENCES:
        raise TransferError("Expanded transfer supports OPERA and JRC only.")
    normalized_tier = str(tier).strip().lower()
    selected_ids(normalized_tier)
    prefix = Path(output_dir) / (
        f"western_quick_transfer_{normalized.lower()}_expanded_{normalized_tier}"
    )
    return ExpandedOutputPaths(
        local=prefix.with_name(prefix.name + "_local.csv"),
        cross_iou=prefix.with_name(prefix.name + "_cross_matrix_iou.csv"),
        cross_f1=prefix.with_name(prefix.name + "_cross_matrix_f1.csv"),
        regional=prefix.with_name(prefix.name + "_regional.csv"),
        lolo=prefix.with_name(prefix.name + "_lolo.csv"),
        status=prefix.with_name(prefix.name + "_status.csv"),
        macro=prefix.with_name(prefix.name + "_macro.csv"),
    )


def _local_row(
    lake: LakeSpec,
    reference: str,
    prepared: PreparedLake,
    summary: dict[str, Any],
    retrieval_seconds: float,
    optimization_seconds: float,
) -> dict[str, Any]:
    threshold = float(summary.get("_threshold_exact", summary["threshold_mean"]))
    test = prepared.samples[prepared.samples["split"] == "test"]
    metrics = evaluate_threshold(test["index_value"], test["label"], threshold)
    return {
        "reference": reference,
        "Hylak_id": lake.hylak_id,
        "Lake_name": lake.lake_name,
        "state": lake.state,
        "Lake_area": lake.lake_area,
        "n_samples": len(prepared.samples),
        "n_train": int(prepared.samples["split"].eq("train").sum()),
        "n_test": len(test),
        "n_blocks": int(prepared.samples["block_id"].nunique()),
        "positive_fraction": float(prepared.samples["label"].mean()),
        "local_threshold": threshold,
        "local_threshold_std": summary["threshold_std"],
        "local_cv_iou": summary["cv_iou"],
        "local_test_iou": metrics["iou"],
        "local_test_f1": metrics["f1"],
        "local_test_precision": metrics["precision"],
        "local_test_recall": metrics["recall"],
        "sample_retrieval_seconds": retrieval_seconds,
        "local_optimization_seconds": optimization_seconds,
        "combination_elapsed_seconds": retrieval_seconds + optimization_seconds,
    }


def _optimize_preblocked_samples(
    samples: pd.DataFrame,
    settings: QuickSamplingSettings,
) -> dict[str, Any]:
    """Fit the local benchmark while preserving cached whole-block splits."""

    if len(samples) < settings.min_samples:
        raise TransferError("too few cached samples for local optimization")
    if samples["block_id"].nunique() < settings.min_blocks:
        raise TransferError("too few cached spatial blocks")
    training = samples[samples["split"] == "train"]
    testing = samples[samples["split"] == "test"]
    for label, view in (("training", training), ("test", testing)):
        if len(view) < settings.min_split_samples:
            raise TransferError(f"too few cached {label} samples")
        if view["label"].nunique() < 2:
            raise TransferError(f"cached {label} split contains one class")
    try:
        threshold, threshold_std, cv_iou = cv_optimal_threshold(
            training,
            n_splits=settings.cv_splits,
            grid_steps=settings.grid_steps,
        )
    except (InsufficientSampleError, ValueError) as exc:
        raise TransferError(f"local grouped CV failed: {exc}") from exc
    metrics = evaluate_threshold(testing["index_value"], testing["label"], threshold)
    return {
        "_threshold_exact": threshold,
        "threshold_mean": round(threshold, 4),
        "threshold_std": round(threshold_std, 4),
        "cv_iou": round(cv_iou, 3),
        "test_iou": metrics["iou"],
        "test_f1": metrics["f1"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "n_train": len(training),
        "n_test": len(testing),
    }


def _existing_cache_path(
    reference: str,
    lake_id: int,
    primary_cache_dir: Path,
    expanded_cache_dir: Path,
) -> Path | None:
    expanded = expanded_cache_dir / f"{reference.lower()}_{lake_id}.csv"
    primary = primary_cache_dir / f"{reference.lower()}_{lake_id}.csv"
    if expanded.is_file():
        return expanded
    if primary.is_file():
        return primary
    return None


def _verify_current_six(
    reference: str,
    local: pd.DataFrame,
    baseline_data_dir: Path,
) -> None:
    """Require unchanged current-six metrics before mixing new lakes."""

    path = baseline_data_dir / f"western_quick_transfer_{reference.lower()}_local.csv"
    if not path.is_file():
        raise TransferError(f"Missing completed six-lake benchmark: {path}")
    baseline = pd.read_csv(path).set_index("Hylak_id")
    current = local[local["Hylak_id"].isin(baseline.index)].set_index("Hylak_id")
    if set(current.index) != set(baseline.index):
        raise TransferError("Expanded run did not preserve all six benchmark lakes.")
    for column in ("local_threshold", "local_test_iou", "local_test_f1"):
        if not np.allclose(
            current.loc[baseline.index, column],
            baseline[column],
            rtol=0,
            atol=1e-12,
        ):
            raise TransferError(
                f"Expanded cached samples do not reproduce six-lake {column}."
            )


def run_expanded_transfer(
    hydrolakes_asset: str,
    lakes: tuple[LakeSpec, ...],
    reference: str,
    settings: QuickSamplingSettings,
    *,
    tier: str = "tier1",
    primary_cache_dir: str | Path = DEFAULT_PRIMARY_CACHE_DIR,
    expanded_cache_dir: str | Path = DEFAULT_EXPANDED_CACHE_DIR,
    baseline_data_dir: str | Path = DEFAULT_OUTPUT_DIR,
    refresh_cache: bool = False,
    cache_only: bool = False,
    initializer: Any = initialize_earth_engine,
    verbose: bool = True,
) -> ExpandedTransferResult:
    """Load/retrieve each pair once, then perform all transfer work locally."""

    normalized = normalize_reference(reference)
    if normalized not in SUPPORTED_REFERENCES:
        raise TransferError("Expanded transfer supports OPERA and JRC only.")
    normalized_tier = str(tier).strip().lower()
    expected_ids = selected_ids(normalized_tier)
    if tuple(lake.hylak_id for lake in lakes) != expected_ids:
        raise TransferError(f"Lake order does not match {normalized_tier} manifest.")

    primary_root = Path(primary_cache_dir)
    expanded_root = Path(expanded_cache_dir)
    started = time.perf_counter()
    ee_module = None
    prepared: dict[int, PreparedLake] = {}
    local_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    benchmark_path = Path(baseline_data_dir) / (
        f"western_quick_transfer_{normalized.lower()}_local.csv"
    )
    if not benchmark_path.is_file():
        raise TransferError(f"Missing completed six-lake benchmark: {benchmark_path}")
    benchmark = pd.read_csv(benchmark_path).set_index("Hylak_id", drop=False)

    for lake in lakes:
        pair_started = time.perf_counter()
        cache = (
            None
            if refresh_cache
            else _existing_cache_path(
                normalized, lake.hylak_id, primary_root, expanded_root
            )
        )
        retrieval_seconds = 0.0
        cache_source = ""
        try:
            if cache is not None:
                frame = read_curve_cache(
                    cache,
                    lake,
                    normalized,
                    settings,
                    hydrolakes_asset,
                    expected_counts=None,
                )
                cache_source = str(cache)
                if verbose:
                    print(f"Loaded {normalized} {lake.lake_name}: {cache}")
            else:
                if cache_only:
                    raise TransferError(
                        f"Missing cache for {normalized} {lake.lake_name} "
                        f"[{lake.hylak_id}] and --cache-only was set."
                    )
                if ee_module is None:
                    ee_module, project = initializer()
                    if verbose:
                        print(f"Earth Engine project: {project}")
                retrieval_started = time.perf_counter()
                frame = retrieve_curve_samples(
                    ee_module,
                    hydrolakes_asset,
                    lake,
                    normalized,
                    settings,
                )
                retrieval_seconds = time.perf_counter() - retrieval_started
                destination = (
                    expanded_root / f"{normalized.lower()}_{lake.hylak_id}.csv"
                )
                frame = write_curve_cache(
                    frame,
                    destination,
                    lake,
                    normalized,
                    settings,
                    hydrolakes_asset,
                    expected_counts=None,
                )
                cache_source = str(destination)
                if verbose:
                    print(
                        f"Retrieved {normalized} {lake.lake_name}: {len(frame)} rows "
                        f"in {retrieval_seconds:.1f}s"
                    )

            if lake.hylak_id in benchmark.index:
                # Preserve the completed six-lake benchmark exactly. Re-running
                # grouped CV is unnecessary and can choose a neighboring grid
                # point when fold-level IoU has a tie.
                benchmark_row = benchmark.loc[lake.hylak_id].to_dict()
                blocked = PreparedLake(
                    lake=lake,
                    reference=normalized,
                    samples=frame.copy(),
                    local_summary=benchmark_row,
                    sample_retrieval_seconds=0.0,
                    optimization_seconds=0.0,
                )
                test = blocked.samples[blocked.samples["split"] == "test"]
                metrics = evaluate_threshold(
                    test["index_value"],
                    test["label"],
                    float(benchmark_row["local_threshold"]),
                )
                if not np.isclose(
                    metrics["iou"],
                    float(benchmark_row["local_test_iou"]),
                    rtol=0,
                    atol=1e-12,
                ):
                    raise TransferError(
                        f"Cached benchmark IoU differs for Hylak_id {lake.hylak_id}."
                    )
                local_rows.append(benchmark_row)
                optimization_seconds = 0.0
            else:
                optimization_started = time.perf_counter()
                summary = _optimize_preblocked_samples(frame, settings)
                optimization_seconds = time.perf_counter() - optimization_started
                blocked = PreparedLake(
                    lake=lake,
                    reference=normalized,
                    samples=frame.copy(),
                    local_summary=summary,
                    sample_retrieval_seconds=retrieval_seconds,
                    optimization_seconds=optimization_seconds,
                )
                local_rows.append(
                    _local_row(
                        lake,
                        normalized,
                        blocked,
                        summary,
                        retrieval_seconds,
                        optimization_seconds,
                    )
                )
            prepared[lake.hylak_id] = blocked
            status = "ok"
            reason = ""
            n_samples = len(frame)
        except (CurveDataError, QuickSamplingError, TransferError, ValueError) as exc:
            status = "skipped"
            reason = str(exc)
            n_samples = 0
            if verbose:
                print(f"Skipped {normalized} {lake.lake_name}: {reason}")
        status_rows.append(
            {
                "reference": normalized,
                "tier": normalized_tier,
                "Hylak_id": lake.hylak_id,
                "Lake_name": lake.lake_name,
                "state": lake.state,
                "status": status,
                "reason": reason,
                "n_samples": n_samples,
                "cache_path": cache_source,
                "retrieval_seconds": retrieval_seconds,
                "elapsed_seconds": time.perf_counter() - pair_started,
            }
        )

    local = pd.DataFrame.from_records(local_rows).reindex(columns=LOCAL_COLUMNS)
    status_table = pd.DataFrame.from_records(status_rows)
    if len(local) < 3:
        raise TransferError("Fewer than three expanded lakes produced usable samples.")
    _verify_current_six(normalized, local, Path(baseline_data_dir))
    usable_order = tuple(lake.hylak_id for lake in lakes if lake.hylak_id in prepared)
    thresholds = dict(
        zip(
            local["Hylak_id"].astype(int),
            local["local_threshold"].astype(float),
            strict=True,
        )
    )
    cross_iou, cross_f1 = cross_application_matrices(prepared, thresholds, usable_order)
    regional_fit, regional = evaluate_pooled_regional(
        prepared,
        usable_order,
        normalized,
        local,
        grid_steps=settings.grid_steps,
    )
    lolo = leave_one_lake_out(
        prepared,
        usable_order,
        normalized,
        local,
        grid_steps=settings.grid_steps,
    )
    elapsed = time.perf_counter() - started
    macro = pd.DataFrame.from_records(
        [
            {
                "reference": normalized,
                "tier": normalized_tier,
                "n_requested_lakes": len(lakes),
                "n_usable_lakes": len(usable_order),
                "n_skipped_lakes": len(lakes) - len(usable_order),
                "regional_threshold": regional_fit.threshold,
                "macro_regional_training_iou": regional_fit.macro_iou,
                "macro_regional_test_iou": float(regional["macro_test_iou"].iloc[0]),
                "macro_lolo_iou": float(lolo["macro_lolo_iou"].iloc[0]),
                "total_samples": int(local["n_samples"].sum()),
                "total_retrieval_seconds": float(
                    status_table["retrieval_seconds"].sum()
                ),
                "elapsed_seconds": elapsed,
            }
        ]
    )
    return ExpandedTransferResult(
        reference=normalized,
        tier=normalized_tier,
        requested_lakes=lakes,
        prepared_lakes=prepared,
        local=local,
        cross_iou=cross_iou,
        cross_f1=cross_f1,
        regional=regional,
        lolo=lolo,
        status=status_table,
        macro=macro,
        elapsed_seconds=elapsed,
    )


def write_expanded_results(
    result: ExpandedTransferResult, output_dir: str | Path
) -> ExpandedOutputPaths:
    """Write only compact local CSVs; no sample table or EE asset is created."""

    paths = expanded_output_paths(output_dir, result.reference, result.tier)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result.local.to_csv(paths.local, index=False, float_format="%.17g")
    result.cross_iou.to_csv(paths.cross_iou, index=True, float_format="%.17g")
    result.cross_f1.to_csv(paths.cross_f1, index=True, float_format="%.17g")
    result.regional.to_csv(paths.regional, index=False, float_format="%.17g")
    result.lolo.to_csv(paths.lolo, index=False, float_format="%.17g")
    result.status.to_csv(paths.status, index=False, float_format="%.17g")
    result.macro.to_csv(paths.macro, index=False, float_format="%.17g")
    return paths


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _date(value: str):
    try:
        return parse_iso_date(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one staged expanded OPERA/JRC transfer analysis from local "
            "caches plus bounded computeFeatures misses. No EE export is used."
        )
    )
    parser.add_argument("--reference", choices=SUPPORTED_REFERENCES, default="OPERA")
    parser.add_argument("--tier", choices=("tier1", "tier2"), default="tier1")
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / DEFAULT_CONFIG_PATH
    )
    parser.add_argument(
        "--candidate-csv", type=Path, default=REPOSITORY_ROOT / DEFAULT_CANDIDATE_PATH
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--primary-cache-dir", type=Path, default=DEFAULT_PRIMARY_CACHE_DIR
    )
    parser.add_argument(
        "--expanded-cache-dir", type=Path, default=DEFAULT_EXPANDED_CACHE_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-image", type=_positive_int, default=20)
    parser.add_argument(
        "--analysis-end", type=_date, default=parse_iso_date("2026-08-09")
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        lakes = load_manifest_lakes(args.manifest, args.candidate_csv, args.tier)
        settings = QuickSamplingSettings(
            samples_per_image=args.samples_per_image,
            analysis_end=args.analysis_end,
        )
        start, end = settings.reference_window(args.reference)
        print(
            f"{args.reference} {args.tier}: {len(lakes)} requested lakes, "
            f"{start}..{end}, 5 km buffer, {settings.samples_per_image} samples/image."
        )
        result = run_expanded_transfer(
            config.earth_engine.hydrolakes_asset,
            lakes,
            args.reference,
            settings,
            tier=args.tier,
            primary_cache_dir=args.primary_cache_dir,
            expanded_cache_dir=args.expanded_cache_dir,
            baseline_data_dir=args.output_dir,
            refresh_cache=args.refresh_cache,
            cache_only=args.cache_only,
        )
        paths = write_expanded_results(result, args.output_dir)
        print("\nLocal results:")
        print(
            result.local[
                [
                    "Hylak_id",
                    "Lake_name",
                    "n_samples",
                    "local_threshold",
                    "local_test_iou",
                ]
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )
        print("\nMacro summary:")
        print(
            result.macro.to_string(
                index=False, float_format=lambda value: f"{value:.4f}"
            )
        )
        print("\nWrote:")
        for path in paths.__dict__.values():
            print(f"  {path}")
        return 0 if result.status["status"].eq("ok").all() else 2
    except (
        ConfigError,
        CurveDataError,
        DiscoveryError,
        OSError,
        QuickSamplingError,
        SelectionError,
        TransferError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
