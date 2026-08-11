"""CLI and sequential orchestration for exploratory multi-lake examples."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ConfigError, load_config
from .discover_lakes import DiscoveryError, initialize_earth_engine
from .quick_sampling import (
    SAMPLE_COLUMNS,
    LakeSpec,
    QuickSamplingError,
    QuickSamplingSettings,
    base_summary,
    build_buffered_lake_aoi,
    build_labelled_collection,
    build_sample_collection,
    load_candidate_inventory,
    normalize_references,
    optimize_samples,
    parse_iso_date,
    sample_collection_to_dataframe,
    sample_diagnostics,
    select_lake_specs,
    summary_dataframe,
)

QUICK_EXAMPLE_SETS = {
    "smoke": (776, 798, 791),
    "saline6": (67, 776, 798, 791, 795, 829),
}

DEFAULT_CONFIG_PATH = Path("configs/western_us.yaml")
DEFAULT_CANDIDATE_PATH = Path("data/regional/western_hydrolakes_candidates.csv")
DEFAULT_RESULTS_PATH = Path("data/regional/western_quick_example_thresholds.csv")


@dataclass
class CombinationResult:
    """One sequential lake/reference execution and its in-memory samples."""

    lake: LakeSpec
    reference: str
    samples: pd.DataFrame
    summary: dict[str, Any]
    diagnostics: dict[str, Any]
    raw_columns: tuple[str, ...]
    lake_feature: Any | None
    buffered_aoi: Any | None
    sample_retrieval_seconds: float = float("nan")
    optimization_seconds: float = float("nan")


@dataclass
class QuickRunResult:
    """Compact threshold table plus each in-memory combination result."""

    summary: pd.DataFrame
    combinations: dict[tuple[int, str], CombinationResult]


def resolve_example_lake_ids(
    *,
    set_name: str | None = None,
    lake_ids: list[int] | tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Resolve a convenience set or explicit IDs without touching Earth Engine."""

    if set_name is not None and lake_ids is not None:
        raise ValueError("Choose either --set or --lake-ids, not both.")
    if set_name is None and lake_ids is None:
        return QUICK_EXAMPLE_SETS["smoke"]
    if set_name is not None:
        normalized = set_name.strip().lower()
        if normalized not in QUICK_EXAMPLE_SETS:
            raise ValueError(
                f"Unknown quick-example set '{set_name}'. Choose from "
                + ", ".join(QUICK_EXAMPLE_SETS)
                + "."
            )
        return QUICK_EXAMPLE_SETS[normalized]
    assert lake_ids is not None
    return tuple(lake_ids)


def fast_mode_notice(settings: QuickSamplingSettings) -> str:
    """Return the methodological warning printed for every quick run."""

    return (
        "Exploratory fast mode: every lake, including Great Salt and Lake Abert, "
        f"uses its HydroLAKES polygon plus a {settings.buffer_km:g} km buffer. "
        "These AOIs and recent windows differ from the historical rectangular-"
        "AOI baseline, so thresholds are not expected to reproduce plugin "
        "defaults. The convenience sets are examples, not a representative "
        "western-lake sample. Dynamic World snow/ice is excluded at sampled "
        "pixels rather than by an AOI-wide scene filter. Same-time tiles are "
        "mosaicked for Dynamic World and the GEE-native OPERA collection."
    )


def run_lake_reference(
    ee_module: Any,
    hydrolakes_asset: str,
    lake: LakeSpec,
    reference: str,
    settings: QuickSamplingSettings,
) -> CombinationResult:
    """Build, transfer, diagnose, and optimize one bounded combination."""

    reference = normalize_references((reference,))[0]
    start_time = time.perf_counter()
    samples = pd.DataFrame(columns=[*SAMPLE_COLUMNS, "acquisition_date"])
    summary = base_summary(lake, reference, settings)
    diagnostics = sample_diagnostics(samples)
    raw_columns: tuple[str, ...] = ()
    lake_feature = None
    buffered_aoi = None
    sample_retrieval_seconds = float("nan")
    optimization_seconds = float("nan")
    try:
        retrieval_start = time.perf_counter()
        lake_feature, buffered_aoi = build_buffered_lake_aoi(
            ee_module,
            hydrolakes_asset,
            lake,
            settings.buffer_km,
        )
        labelled = build_labelled_collection(
            ee_module,
            buffered_aoi,
            reference,
            settings,
        )
        sampled = build_sample_collection(
            ee_module,
            labelled,
            buffered_aoi,
            lake,
            reference,
            settings,
        )
        transfer = sample_collection_to_dataframe(
            ee_module,
            sampled,
            settings,
        )
        samples = transfer.frame
        raw_columns = transfer.raw_columns
        sample_retrieval_seconds = time.perf_counter() - retrieval_start
        diagnostics = sample_diagnostics(samples)
        optimization_start = time.perf_counter()
        summary = optimize_samples(samples, lake, reference, settings)
        optimization_seconds = time.perf_counter() - optimization_start
    # Combination isolation is intentional: Earth Engine raises several public
    # exception types depending on whether graph construction or transfer fails.
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "failed"
        summary["skip_reason"] = f"{type(exc).__name__}: {exc}"
    summary["elapsed_seconds"] = round(time.perf_counter() - start_time, 2)
    return CombinationResult(
        lake=lake,
        reference=reference,
        samples=samples,
        summary=summary,
        diagnostics=diagnostics,
        raw_columns=raw_columns,
        lake_feature=lake_feature,
        buffered_aoi=buffered_aoi,
        sample_retrieval_seconds=round(sample_retrieval_seconds, 2),
        optimization_seconds=round(optimization_seconds, 2),
    )


def _print_combination(result: CombinationResult) -> None:
    summary = result.summary
    name = result.lake.lake_name or f"Hylak_id {result.lake.hylak_id}"
    print(
        f"{name} [{result.lake.hylak_id}] x {result.reference}: "
        f"{summary['status']} in {summary['elapsed_seconds']:.2f}s"
    )
    if result.raw_columns:
        print("  computeFeatures columns: " + ", ".join(result.raw_columns))
    diagnostics = result.diagnostics
    if diagnostics["n_rows"]:
        print(
            f"  rows={diagnostics['n_rows']}, "
            f"positive={diagnostics['positive_fraction']:.3f}, "
            f"dates={diagnostics['date_start']}..{diagnostics['date_end']}, "
            f"sensors={','.join(diagnostics['sensors'])}, "
            f"acquisitions={diagnostics['n_acquisitions']}"
        )
    if summary["status"] == "ok":
        print(
            f"  threshold={summary['threshold_mean']:+.4f}, "
            f"CV IoU={summary['cv_iou']:.3f}, "
            f"test IoU={summary['test_iou']:.3f}, "
            f"test F1={summary['test_f1']:.3f}"
        )
    elif summary["skip_reason"]:
        print(f"  {summary['skip_reason']}")


def run_quick_examples(
    ee_module: Any,
    hydrolakes_asset: str,
    lakes: tuple[LakeSpec, ...],
    references: tuple[str, ...],
    settings: QuickSamplingSettings,
    *,
    verbose: bool = True,
) -> QuickRunResult:
    """Run combinations sequentially and keep bounded samples in memory."""

    combinations: dict[tuple[int, str], CombinationResult] = {}
    rows: list[dict[str, Any]] = []
    for lake in lakes:
        for reference in references:
            if verbose:
                print(
                    f"Running Hylak_id={lake.hylak_id} ({lake.lake_name}) "
                    f"against {reference}..."
                )
            result = run_lake_reference(
                ee_module,
                hydrolakes_asset,
                lake,
                reference,
                settings,
            )
            combinations[(lake.hylak_id, reference)] = result
            rows.append(result.summary)
            if verbose:
                _print_combination(result)
    return QuickRunResult(
        summary=summary_dataframe(rows),
        combinations=combinations,
    )


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


def _date_argument(value: str):
    try:
        return parse_iso_date(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the quick-example CLI without initializing Earth Engine."""

    parser = argparse.ArgumentParser(
        description=(
            "Run bounded, in-memory HydroLAKES-buffer NDWI/NDVI calibration "
            "examples. Samples are retrieved with computeFeatures; no Earth "
            "Engine output task is created."
        )
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--set",
        dest="set_name",
        choices=tuple(QUICK_EXAMPLE_SETS),
        help="Convenience lake set (default: smoke).",
    )
    selection.add_argument(
        "--lake-ids",
        nargs="+",
        type=_positive_int,
        help="Ordered HydroLAKES IDs from the western candidate inventory.",
    )
    parser.add_argument(
        "--references",
        nargs="+",
        default=["JRC", "DW"],
        type=str.upper,
        choices=("JRC", "DW", "OPERA"),
        help="References to run sequentially (default: JRC DW).",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--buffer-km", type=_positive_float, default=5.0)
    parser.add_argument("--samples-per-image", type=_positive_int, default=20)
    parser.add_argument(
        "--max-rows-per-combination", type=_positive_int, default=10_000
    )
    parser.add_argument(
        "--analysis-end", type=_date_argument, default=parse_iso_date("2026-08-09")
    )
    parser.add_argument("--jrc-start", type=_date_argument)
    parser.add_argument("--jrc-end", type=_date_argument)
    parser.add_argument("--dw-start", type=_date_argument)
    parser.add_argument("--opera-start", type=_date_argument)
    parser.add_argument("--opera-window-days", type=_positive_int)
    parser.add_argument(
        "--save-results",
        nargs="?",
        const=DEFAULT_RESULTS_PATH,
        type=Path,
        metavar="PATH",
        help=(
            "Optionally write the compact threshold summary. Intermediate "
            "samples remain in memory and are not cached."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        inventory = load_candidate_inventory(args.candidate_csv)
        lake_ids = resolve_example_lake_ids(
            set_name=args.set_name,
            lake_ids=args.lake_ids,
        )
        lakes = select_lake_specs(inventory, lake_ids)
        references = normalize_references(args.references)
        settings = QuickSamplingSettings().with_overrides(
            buffer_km=args.buffer_km,
            samples_per_image=args.samples_per_image,
            max_rows_per_combination=args.max_rows_per_combination,
            analysis_end=args.analysis_end,
            jrc_start=args.jrc_start,
            jrc_end=args.jrc_end,
            dw_start=args.dw_start,
            opera_start=args.opera_start,
            opera_window_days=args.opera_window_days,
        )
        print(fast_mode_notice(settings))
        for reference in references:
            start, end = settings.reference_window(reference)
            print(
                f"{reference}: {start}..{end}, nearest-time window "
                f"+/-{settings.join_window_days(reference)} day(s)"
            )
        ee_module, project = initialize_earth_engine()
        print(f"Earth Engine project: {project}")
        result = run_quick_examples(
            ee_module,
            config.earth_engine.hydrolakes_asset,
            lakes,
            references,
            settings,
        )
        print("\nCompact threshold summary:")
        print(result.summary.to_string(index=False))
        if args.save_results is not None:
            args.save_results.parent.mkdir(parents=True, exist_ok=True)
            result.summary.to_csv(args.save_results, index=False)
            print(f"Wrote final summary: {args.save_results}")
        return 1 if (result.summary["status"] == "failed").any() else 0
    except (ConfigError, DiscoveryError, QuickSamplingError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
