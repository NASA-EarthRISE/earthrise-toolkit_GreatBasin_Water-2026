"""Create local held-out threshold, ROC, PR, and NDWI-distribution figures."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .config import ConfigError, load_config
from .discover_lakes import DiscoveryError
from .plot_quick_transfer import (
    DEFAULT_DATA_DIR,
    LAKE_IDS,
    LAKES,
    PLOT_STYLE,
    REFERENCE_COLORS,
    PlotDataError,
    ReferencePlotData,
    load_reference_plot_data,
    save_figure_with_no_text,
)
from .quick_curve_data import (
    CurveDataError,
    DiscriminationCurves,
    build_curve_summary,
    discrimination_curves,
    load_or_retrieve_curve_samples,
    threshold_response,
    training_threshold_grid,
)
from .quick_examples import DEFAULT_CANDIDATE_PATH, DEFAULT_CONFIG_PATH
from .quick_sampling import (
    LakeSpec,
    QuickSamplingError,
    QuickSamplingSettings,
    load_candidate_inventory,
    select_lake_specs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPOSITORY_ROOT / "data" / "regional" / "quick_transfer_cache"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "quick_transfer_curves"
CURVE_REFERENCES = ("OPERA", "JRC")

LAKE_LINESTYLES = (
    "-",
    "--",
    ":",
    "-.",
    (0, (5, 1, 1, 1)),
    (0, (2, 1)),
)
THRESHOLD_STYLES = {
    "local": {"color": "#2B2B2B", "linestyle": ":", "marker": "o"},
    "lolo": {"color": "#707070", "linestyle": "-.", "marker": "D"},
    "regional": {"color": "#111111", "linestyle": "--", "marker": "^"},
}


def resolve_curve_references(values: Sequence[str]) -> tuple[str, ...]:
    """Resolve an OPERA/JRC subset in stable reference order."""

    requested = {str(value).strip().upper() for value in values}
    invalid = sorted(requested.difference(CURVE_REFERENCES))
    if invalid or not requested:
        raise ValueError(
            "Curve references must be OPERA and/or JRC; invalid="
            f"{invalid or 'empty selection'}."
        )
    return tuple(reference for reference in CURVE_REFERENCES if reference in requested)


def resolve_curve_lake_ids(values: Sequence[str] | None) -> tuple[int, ...]:
    """Resolve names or IDs while preserving the canonical six-lake order."""

    if values is None:
        return LAKE_IDS
    name_to_id = {name.casefold(): lake_id for lake_id, name in LAKES}
    requested: set[int] = set()
    for value in values:
        token = str(value).strip()
        if token.casefold() in name_to_id:
            lake_id = name_to_id[token.casefold()]
        else:
            try:
                lake_id = int(token)
            except ValueError as exc:
                raise ValueError(f"Unknown curve lake {value!r}.") from exc
        if lake_id not in LAKE_IDS:
            raise ValueError(f"Lake ID {lake_id} is not in the six-lake experiment.")
        if lake_id in requested:
            raise ValueError(f"Duplicate curve lake: {value!r}.")
        requested.add(lake_id)
    if not requested:
        raise ValueError("At least one lake is required.")
    return tuple(lake_id for lake_id in LAKE_IDS if lake_id in requested)


def _require_curve_summary_schema(data: ReferencePlotData) -> None:
    required = {
        "local": {
            "Hylak_id",
            "n_samples",
            "n_train",
            "n_test",
            "local_threshold",
            "local_test_iou",
        },
        "lolo": {
            "held_out_Hylak_id",
            "threshold",
            "held_out_test_iou",
        },
        "regional": {
            "Hylak_id",
            "regional_threshold",
            "regional_test_iou",
        },
    }
    for name, columns in required.items():
        frame = getattr(data, name)
        missing = sorted(columns.difference(frame.columns))
        if missing:
            raise CurveDataError(
                f"{data.reference} {name} summary lacks: {', '.join(missing)}."
            )


def _expected_counts(
    tables: Mapping[str, ReferencePlotData],
    references: Sequence[str],
    lake_ids: Sequence[int],
) -> dict[tuple[str, int], tuple[int, int, int]]:
    result: dict[tuple[str, int], tuple[int, int, int]] = {}
    for reference in references:
        _require_curve_summary_schema(tables[reference])
        local = tables[reference].local.set_index("Hylak_id")
        for lake_id in lake_ids:
            row = local.loc[lake_id]
            result[(reference, lake_id)] = (
                int(row["n_samples"]),
                int(row["n_train"]),
                int(row["n_test"]),
            )
    return result


def _thresholds_for_lake(data: ReferencePlotData, lake_id: int) -> dict[str, float]:
    local = data.local.set_index("Hylak_id").loc[lake_id]
    lolo = data.lolo.set_index("held_out_Hylak_id").loc[lake_id]
    regional = data.regional.set_index("Hylak_id").loc[lake_id]
    return {
        "local": float(local["local_threshold"]),
        "lolo": float(lolo["threshold"]),
        "regional": float(regional["regional_threshold"]),
    }


def _test_samples(samples: pd.DataFrame) -> pd.DataFrame:
    return samples.loc[samples["split"] == "test"].copy()


def _threshold_handles() -> list[Line2D]:
    handles = []
    for label, style in THRESHOLD_STYLES.items():
        handles.append(
            Line2D(
                [],
                [],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor="white",
                markersize=5,
                linewidth=1.0,
                label={
                    "local": "Local threshold",
                    "lolo": "LOLO threshold",
                    "regional": "Pooled regional threshold",
                }[label],
            )
        )
    return handles


def _add_threshold_markers(
    axis: mpl.axes.Axes,
    thresholds: Mapping[str, float],
) -> None:
    lanes = {"local": 0.985, "lolo": 0.945, "regional": 0.905}
    for label, value in thresholds.items():
        style = THRESHOLD_STYLES[label]
        axis.axvline(
            value,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=0.85,
            alpha=0.8,
            zorder=2,
        )
        axis.scatter(
            [value],
            [lanes[label]],
            transform=axis.get_xaxis_transform(),
            marker=style["marker"],
            s=17,
            facecolor="white",
            edgecolor=style["color"],
            linewidth=0.8,
            clip_on=False,
            zorder=4,
        )


def plot_threshold_response_curves(
    references: Sequence[str],
    lakes: Sequence[LakeSpec],
    responses: Mapping[tuple[str, int], pd.DataFrame],
    tables: Mapping[str, ReferencePlotData],
) -> mpl.figure.Figure:
    """Plot held-out IoU/F1 over shared training-derived grids."""

    fig, axes = plt.subplots(
        len(lakes),
        len(references),
        figsize=(5.0 * len(references), 1.85 * len(lakes) + 0.7),
        sharex="col",
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for column, reference in enumerate(references):
        color = REFERENCE_COLORS[reference]
        for row, lake in enumerate(lakes):
            axis = axes[row, column]
            response = responses[(reference, lake.hylak_id)]
            axis.plot(
                response["threshold"],
                response["iou"],
                color=color,
                linewidth=1.35,
                linestyle="-",
                zorder=3,
            )
            axis.plot(
                response["threshold"],
                response["f1"],
                color=color,
                linewidth=1.25,
                linestyle="--",
                zorder=3,
            )
            _add_threshold_markers(
                axis,
                _thresholds_for_lake(tables[reference], lake.hylak_id),
            )
            axis.set_ylim(0.0, 1.02)
            axis.set_yticks(np.arange(0.0, 1.01, 0.25))
            axis.grid(color="#E2E2E2", linewidth=0.55)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            if row == 0:
                axis.set_title(reference)
            if row == len(lakes) - 1:
                axis.set_xlabel("NDWI threshold")
            if column == 0:
                axis.set_ylabel(f"{lake.lake_name}\nPerformance")
    metric_handles = [
        Line2D([], [], color="#444444", linewidth=1.4, label="IoU"),
        Line2D(
            [],
            [],
            color="#444444",
            linewidth=1.3,
            linestyle="--",
            label="F1",
        ),
    ]
    fig.legend(
        handles=[*metric_handles, *_threshold_handles()],
        loc="outside lower center",
        ncol=5,
        frameon=False,
    )
    return fig


def plot_roc_curves(
    references: Sequence[str],
    lakes: Sequence[LakeSpec],
    curves: Mapping[tuple[str, int], DiscriminationCurves],
) -> mpl.figure.Figure:
    """Plot held-out ROC curves; raw NDWI is the positive-water score."""

    fig, axes = plt.subplots(
        1,
        len(references),
        figsize=(5.1 * len(references), 4.9),
        squeeze=False,
        constrained_layout=True,
    )
    for column, reference in enumerate(references):
        axis = axes[0, column]
        color = REFERENCE_COLORS[reference]
        axis.plot([0, 1], [0, 1], color="#999999", linestyle=":", linewidth=0.9)
        for lake, linestyle in zip(lakes, LAKE_LINESTYLES, strict=False):
            result = curves[(reference, lake.hylak_id)]
            axis.plot(
                result.fpr,
                result.tpr,
                color=color,
                linestyle=linestyle,
                linewidth=1.35,
                label=f"{lake.lake_name} ({result.roc_auc:.3f})",
            )
        axis.set_title(reference, loc="left")
        axis.set_xlabel("False-positive rate")
        axis.set_ylabel("True-positive rate")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.01)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#E2E2E2", linewidth=0.55)
        axis.legend(title="Lake (AUC)", frameon=False, fontsize=7.2, title_fontsize=7.5)
        axis.spines[["top", "right"]].set_visible(False)
    return fig


def plot_precision_recall_curves(
    references: Sequence[str],
    lakes: Sequence[LakeSpec],
    curves: Mapping[tuple[str, int], DiscriminationCurves],
) -> mpl.figure.Figure:
    """Plot held-out precision-recall curves and average precision."""

    fig, axes = plt.subplots(
        1,
        len(references),
        figsize=(5.1 * len(references), 4.9),
        squeeze=False,
        constrained_layout=True,
    )
    for column, reference in enumerate(references):
        axis = axes[0, column]
        color = REFERENCE_COLORS[reference]
        for lake, linestyle in zip(lakes, LAKE_LINESTYLES, strict=False):
            result = curves[(reference, lake.hylak_id)]
            axis.plot(
                result.recall,
                result.precision,
                color=color,
                linestyle=linestyle,
                linewidth=1.35,
                label=f"{lake.lake_name} ({result.average_precision:.3f})",
            )
        axis.set_title(reference, loc="left")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.01)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#E2E2E2", linewidth=0.55)
        axis.legend(title="Lake (AP)", frameon=False, fontsize=7.2, title_fontsize=7.5)
        axis.spines[["top", "right"]].set_visible(False)
    return fig


def plot_index_distributions(
    reference: str,
    lakes: Sequence[LakeSpec],
    samples: Mapping[tuple[str, int], pd.DataFrame],
    tables: ReferencePlotData,
) -> mpl.figure.Figure:
    """Plot test-only within-class NDWI line histograms for one reference."""

    selected = [_test_samples(samples[(reference, lake.hylak_id)]) for lake in lakes]
    score_min = min(float(frame["index_value"].min()) for frame in selected)
    score_max = max(float(frame["index_value"].max()) for frame in selected)
    marker_values = [
        value
        for lake in lakes
        for value in _thresholds_for_lake(tables, lake.hylak_id).values()
    ]
    low = min(score_min, min(marker_values))
    high = max(score_max, max(marker_values))
    padding = max((high - low) * 0.025, 0.005)
    bin_edges = np.linspace(low - padding, high + padding, 42)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.8, 6.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    color = REFERENCE_COLORS[reference]
    flat_axes = list(axes.flat)
    for position, (axis, lake, test) in enumerate(
        zip(flat_axes, lakes, selected, strict=False)
    ):
        for label, line_color, linestyle in (
            (0, "#777777", "--"),
            (1, color, "-"),
        ):
            counts, _ = np.histogram(
                test.loc[test["label"] == label, "index_value"],
                bins=bin_edges,
            )
            relative = counts / counts.max() if counts.max() else counts.astype(float)
            axis.stairs(
                relative,
                bin_edges,
                color=line_color,
                linestyle=linestyle,
                linewidth=1.35,
            )
        _add_threshold_markers(
            axis,
            _thresholds_for_lake(tables, lake.hylak_id),
        )
        axis.set_title(lake.lake_name, fontsize=9.5)
        axis.set_ylim(0, 1.04)
        if position + 3 >= len(lakes):
            axis.set_xlabel("NDWI")
        if position % 3 == 0:
            axis.set_ylabel("Within-class relative frequency")
        axis.grid(color="#E4E4E4", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in flat_axes[len(lakes) :]:
        axis.set_visible(False)
    fig.suptitle(f"{reference} held-out NDWI distributions", fontsize=11)
    class_handles = [
        Line2D([], [], color="#777777", linestyle="--", label="Negative label"),
        Line2D([], [], color=color, linestyle="-", label="Positive label"),
    ]
    fig.legend(
        handles=[*class_handles, *_threshold_handles()],
        loc="outside lower center",
        ncol=5,
        frameon=False,
    )
    return fig


def curve_figure_notes(
    references: Sequence[str],
    lakes: Sequence[LakeSpec],
    settings: QuickSamplingSettings,
    grid_steps: int,
) -> str:
    """Build notes that reflect the actual CLI reference/lake selection."""

    windows = "; ".join(
        f"{reference} uses {settings.reference_window(reference)[0].isoformat()} "
        f"through {settings.reference_window(reference)[1].isoformat()}"
        for reference in references
    )
    lake_count = "six" if len(lakes) == 6 else str(len(lakes))
    lake_noun = "examples" if len(lakes) != 1 else "example"
    scope = (
        f"These figures use the untouched 1 km-block test splits from {lake_count} "
        f"selected exploratory western saline/terminal lake {lake_noun}, not a final "
        "western-U.S. validation. Every AOI is a HydroLAKES polygon plus a "
        f"{settings.buffer_km:g} km buffer. {windows}."
    )
    paragraphs = [
        "# Quick-transfer curve figure notes",
        scope,
        "## Threshold-response curves",
        (
            f"IoU and F1 are evaluated descriptively across one common {grid_steps}-"
            "value threshold grid per reference. Each grid is derived only from the "
            "selected lakes' training-score 0.5th–99.5th percentile ranges, while "
            "every plotted metric uses test observations only. Vertical markers are "
            "the saved local grouped-CV, LOLO, and equal-lake-weighted pooled regional "
            "thresholds; test curves never refit or replace them."
        ),
        "## ROC curves",
        (
            "ROC curves use continuous held-out NDWI and report AUC. Score direction "
            "is unchanged from the classifier: higher NDWI is more positive for open "
            "water and predictions use NDWI greater than or equal to the threshold. "
            "The gray diagonal is chance discrimination."
        ),
        "## Precision–recall curves",
        (
            "Precision and recall use the same held-out continuous NDWI scores, with "
            "average precision shown for each lake. They complement ROC because "
            "positive-label prevalence differs among lake/reference samples."
        ),
    ]
    for reference in references:
        paragraphs.extend(
            (
                f"## {reference} index-value distributions",
                (
                    f"Test-only {reference} NDWI values are unsmoothed line "
                    "histograms, scaled to each class's maximum bin so imbalance does "
                    "not hide distribution shape. Class color/line style and saved "
                    "local, LOLO, and pooled-regional markers connect the score "
                    "distributions to classification thresholds."
                ),
            )
        )
    return "\n\n".join(paragraphs) + "\n"


def create_curve_outputs(
    references: Sequence[str],
    lakes: Sequence[LakeSpec],
    samples: Mapping[tuple[str, int], pd.DataFrame],
    tables: Mapping[str, ReferencePlotData],
    summary: pd.DataFrame,
    output_dir: str | Path,
    settings: QuickSamplingSettings,
    *,
    grid_steps: int = 150,
) -> tuple[Path, ...]:
    """Compute local curve coordinates and save all requested outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    responses: dict[tuple[str, int], pd.DataFrame] = {}
    curves: dict[tuple[str, int], DiscriminationCurves] = {}
    for reference in references:
        by_id = {lake.hylak_id: samples[(reference, lake.hylak_id)] for lake in lakes}
        grid = training_threshold_grid(by_id, grid_steps=grid_steps)
        for lake in lakes:
            key = (reference, lake.hylak_id)
            test = _test_samples(samples[key])
            responses[key] = threshold_response(test, grid)
            curves[key] = discrimination_curves(test)

    saved: list[Path] = []
    with mpl.rc_context(PLOT_STYLE):
        plotters: list[tuple[str, mpl.figure.Figure]] = [
            (
                "threshold_response_curves",
                plot_threshold_response_curves(references, lakes, responses, tables),
            ),
            ("roc_curves", plot_roc_curves(references, lakes, curves)),
            (
                "precision_recall_curves",
                plot_precision_recall_curves(references, lakes, curves),
            ),
        ]
        for reference in references:
            plotters.append(
                (
                    f"index_value_distributions_{reference.lower()}",
                    plot_index_distributions(
                        reference,
                        lakes,
                        samples,
                        tables[reference],
                    ),
                )
            )
        for basename, figure in plotters:
            saved.extend(save_figure_with_no_text(figure, output, basename))
    summary_path = output / "curve_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.17g")
    notes_path = output / "figure_notes.md"
    notes_path.write_text(
        curve_figure_notes(references, lakes, settings, grid_steps),
        encoding="utf-8",
    )
    saved.extend((summary_path, notes_path))
    return tuple(saved)


def _positive_grid_steps(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 2:
        raise argparse.ArgumentTypeError("must be at least two")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot held-out threshold response, ROC, PR, and NDWI distributions "
            "from cacheable direct quick-transfer samples. No EE exports are used."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / DEFAULT_CONFIG_PATH
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=REPOSITORY_ROOT / DEFAULT_CANDIDATE_PATH,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--references",
        nargs="+",
        default=list(CURVE_REFERENCES),
        metavar="REFERENCE",
        help="OPERA and/or JRC (default: both)",
    )
    parser.add_argument(
        "--lakes",
        nargs="+",
        metavar="LAKE",
        help="Subset by HydroLAKES ID or exact lake name (default: all six)",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Retrieve requested samples again and replace matching local caches.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Fail rather than initialize Earth Engine if any cache is missing.",
    )
    parser.add_argument(
        "--grid-steps",
        type=_positive_grid_steps,
        default=150,
        help="Shared training-derived threshold grid size (default: 150).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        references = resolve_curve_references(args.references)
        lake_ids = resolve_curve_lake_ids(args.lakes)
        config = load_config(args.config)
        inventory = load_candidate_inventory(args.candidate_csv)
        lakes = select_lake_specs(inventory, lake_ids)
        settings = QuickSamplingSettings(grid_steps=args.grid_steps)
        tables = {
            reference: load_reference_plot_data(args.data_dir, reference)
            for reference in references
        }
        expected = _expected_counts(tables, references, lake_ids)

        def validate_pair(reference: str, lake: LakeSpec, frame: pd.DataFrame) -> None:
            build_curve_summary(
                reference,
                [lake],
                {lake.hylak_id: frame},
                tables[reference].local,
                tables[reference].lolo,
                tables[reference].regional,
            )

        cache_load = load_or_retrieve_curve_samples(
            lakes,
            references,
            settings,
            args.cache_dir,
            expected,
            hydrolakes_asset=config.earth_engine.hydrolakes_asset,
            refresh_cache=args.refresh_cache,
            cache_only=args.cache_only,
            sample_validator=validate_pair,
        )

        summary_parts = []
        for reference in references:
            by_id = {
                lake.hylak_id: cache_load.samples[(reference, lake.hylak_id)]
                for lake in lakes
            }
            summary_parts.append(
                build_curve_summary(
                    reference,
                    lakes,
                    by_id,
                    tables[reference].local,
                    tables[reference].lolo,
                    tables[reference].regional,
                )
            )
        summary = pd.concat(summary_parts, ignore_index=True)
        order = {
            (lake_id, reference): index
            for index, (lake_id, _) in enumerate(LAKES)
            for reference in CURVE_REFERENCES
        }
        summary["_order"] = [
            order[(int(row.Hylak_id), str(row.reference))]
            for row in summary.itertuples(index=False)
        ]
        reference_rank = {
            reference: rank for rank, reference in enumerate(CURVE_REFERENCES)
        }
        summary["_reference_order"] = summary["reference"].map(reference_rank)
        summary = summary.sort_values(
            ["_order", "_reference_order"], kind="mergesort"
        ).drop(columns=["_order", "_reference_order"])

        print("Verified held-out curve data:")
        print(
            summary.to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            )
        )
        saved = create_curve_outputs(
            references,
            lakes,
            cache_load.samples,
            tables,
            summary,
            args.output_dir,
            settings,
            grid_steps=args.grid_steps,
        )
    except (
        ConfigError,
        CurveDataError,
        DiscoveryError,
        OSError,
        PlotDataError,
        QuickSamplingError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Created {len(cache_load.created_paths)} cache file(s), "
        f"{len(saved) - 2} figure file(s), curve_summary.csv, and figure_notes.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
