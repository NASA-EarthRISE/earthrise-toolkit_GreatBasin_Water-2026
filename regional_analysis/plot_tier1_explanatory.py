"""Plot the completed, strictly local Tier 1 explanatory analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .plot_quick_transfer import (
    PLOT_STYLE,
    REFERENCE_COLORS,
    save_figure_with_no_text,
)
from .quick_curve_data import threshold_response, training_threshold_grid
from .tier1_explanatory import (
    CASE_STUDIES,
    DEFAULT_CANDIDATE_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_EXPANDED_CACHE_DIR,
    DEFAULT_GRID_STEPS,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PRIMARY_CACHE_DIR,
    REFERENCES,
    TIER1_IDS,
    Tier1ExplanatoryError,
    load_tier1_inputs,
    validate_explanatory_table,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "tier1_explanatory"

LAKE_ABBREVIATIONS = {
    67: "GS",
    776: "AB",
    798: "MO",
    791: "PY",
    795: "WA",
    829: "SA",
    780: "GO",
    769: "MA",
    789: "HO",
    774: "SU",
    787: "EA",
    788: "UT",
}
FIGURE_NAMES = (
    "transfer_vs_discrimination",
    "threshold_geometry_vs_transfer",
    "temporal_sample_behavior",
    "hydrolakes_context",
    "diagnostic_case_studies",
)
IMPORTANT_LABEL_IDS = {67, 798, 791, 780, 769, 789, 774}


class Tier1PlotError(ValueError):
    """Raised when explanatory products cannot support the requested plots."""


def load_explanatory_products(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and validate the three completed explanatory CSVs."""

    root = Path(data_dir)
    paths = (
        root / "tier1_transfer_explanatory.csv",
        root / "tier1_transfer_explanatory_associations.csv",
        root / "tier1_transfer_case_studies.csv",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise Tier1PlotError(
            "Missing explanatory outputs; run tier1_explanatory first: "
            + ", ".join(missing)
        )
    try:
        table, associations, cases = (pd.read_csv(path) for path in paths)
    except (OSError, pd.errors.ParserError) as exc:
        raise Tier1PlotError(f"Could not read explanatory outputs: {exc}") from exc
    validate_explanatory_table(table)
    association_columns = {
        "reference",
        "outcome",
        "explanatory_variable",
        "n_complete",
        "spearman_rho",
        "leave_one_lake_out_rho_min",
        "leave_one_lake_out_rho_max",
        "most_influential_lake",
    }
    case_columns = {
        "reference",
        "Hylak_id",
        "Lake_name",
        "data_assessment",
        "local_test_iou",
        "lolo_test_iou",
        "test_roc_auc",
        "plateau_width_normalized",
    }
    if missing_columns := sorted(association_columns.difference(associations.columns)):
        raise Tier1PlotError(
            "Association table is missing: " + ", ".join(missing_columns)
        )
    if missing_columns := sorted(case_columns.difference(cases.columns)):
        raise Tier1PlotError("Case table is missing: " + ", ".join(missing_columns))
    if len(cases) != len(CASE_STUDIES):
        raise Tier1PlotError("Case-study table must contain exactly four rows.")
    return table, associations, cases


def _reference_subset(table: pd.DataFrame, reference: str) -> pd.DataFrame:
    subset = table[table["reference"] == reference].copy()
    subset = subset.sort_values("selection_order", kind="mergesort")
    if tuple(subset["Hylak_id"].astype(int)) != TIER1_IDS:
        raise Tier1PlotError(f"{reference} explanatory lake order is invalid.")
    return subset


def _scatter_lakes(
    axis: mpl.axes.Axes,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    reference: str,
    *,
    annotate: bool = True,
) -> None:
    """Draw reference-colored points with lake abbreviations and tier fill."""

    color = REFERENCE_COLORS[reference]
    for row in frame.itertuples(index=False):
        filled = row.tier_group == "original6"
        axis.scatter(
            getattr(row, x_column),
            getattr(row, y_column),
            s=33,
            marker="o" if reference == "OPERA" else "D",
            facecolor=color if filled else "white",
            edgecolor=color,
            linewidth=1.0,
            zorder=3,
        )
        if annotate and int(row.Hylak_id) in IMPORTANT_LABEL_IDS:
            axis.annotate(
                LAKE_ABBREVIATIONS[int(row.Hylak_id)],
                (getattr(row, x_column), getattr(row, y_column)),
                xytext=(4, 3 if int(row.selection_order) % 2 else -8),
                textcoords="offset points",
                fontsize=6.5,
                color="#222222",
            )
    axis.grid(color="#E3E3E3", linewidth=0.55)
    axis.spines[["top", "right"]].set_visible(False)


def plot_transfer_vs_discrimination(table: pd.DataFrame) -> mpl.figure.Figure:
    """Relate held-out transfer performance to AUC and KS separation."""

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.2, 7.4),
        sharey="row",
        constrained_layout=True,
    )
    for column, reference in enumerate(REFERENCES):
        frame = _reference_subset(table, reference)
        for row, (x_column, x_label) in enumerate(
            (
                ("test_roc_auc", "Held-out ROC AUC"),
                ("test_ks_statistic", "Held-out KS separation"),
            )
        ):
            axis = axes[row, column]
            _scatter_lakes(axis, frame, x_column, "lolo_test_iou", reference)
            axis.set_xlabel(x_label)
            axis.set_ylabel("LOLO IoU" if column == 0 else "")
            if row == 0:
                axis.set_title(reference)
            axis.set_ylim(0.35, 1.01)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#555555",
            markeredgecolor="#555555",
            label="Original six",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#555555",
            label="Tier 1 addition",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    return fig


def plot_threshold_geometry(table: pd.DataFrame) -> mpl.figure.Figure:
    """Relate transfer loss to threshold distance and plateau width."""

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.2, 7.4),
        sharey="row",
        constrained_layout=True,
    )
    panels = (
        (
            "abs_local_to_regional_threshold",
            "|Local - regional threshold|",
        ),
        ("plateau_width_normalized", "Normalized near-optimal plateau width"),
    )
    for column, reference in enumerate(REFERENCES):
        frame = _reference_subset(table, reference)
        for row, (x_column, x_label) in enumerate(panels):
            axis = axes[row, column]
            _scatter_lakes(axis, frame, x_column, "lolo_iou_delta", reference)
            axis.axhline(0, color="#222222", linewidth=0.8)
            axis.set_xlabel(x_label)
            axis.set_ylabel("LOLO IoU - local IoU" if column == 0 else "")
            if row == 0:
                axis.set_title(reference)
    return fig


def plot_temporal_sample_behavior(table: pd.DataFrame) -> mpl.figure.Figure:
    """Show three pre-specified temporal/sample diagnostics without selection."""

    panels = (
        ("acquisition_prevalence_sd", "Acquisition water-prevalence SD"),
        ("positive_fraction_test", "Held-out positive fraction"),
        (
            "water_acquisition_median_ndwi_sd",
            "Water acquisition-median NDWI SD",
        ),
    )
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 9.8), constrained_layout=True)
    for column, reference in enumerate(REFERENCES):
        frame = _reference_subset(table, reference)
        for row, (x_column, x_label) in enumerate(panels):
            axis = axes[row, column]
            _scatter_lakes(axis, frame, x_column, "lolo_test_iou", reference)
            axis.set_xlabel(x_label)
            axis.set_ylabel("LOLO IoU" if column == 0 else "")
            axis.set_ylim(0.35, 1.01)
            if row == 0:
                axis.set_title(reference)
    return fig


def plot_hydrolakes_context(table: pd.DataFrame) -> mpl.figure.Figure:
    """Place transfer performance against four fixed HydroLAKES attributes."""

    panels = (
        ("log10_Lake_area", "log10 lake area (km2)"),
        ("Depth_avg", "Mean depth (m)"),
        ("Elevation", "Elevation (m)"),
        ("log10_Wshd_area", "log10 watershed area (km2)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), constrained_layout=True)
    for axis, (x_column, x_label) in zip(axes.flat, panels, strict=True):
        for reference in REFERENCES:
            frame = _reference_subset(table, reference).dropna(subset=[x_column])
            _scatter_lakes(
                axis,
                frame,
                x_column,
                "lolo_test_iou",
                reference,
                annotate=reference == "OPERA",
            )
        n_by_reference = ", ".join(
            f"{reference} n={_reference_subset(table, reference)[x_column].notna().sum()}"
            for reference in REFERENCES
        )
        axis.set_title(n_by_reference, fontsize=8, loc="left")
        axis.set_xlabel(x_label)
        axis.set_ylabel("LOLO IoU")
        axis.set_ylim(0.35, 1.01)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=REFERENCE_COLORS["OPERA"],
            linestyle="none",
            label="OPERA",
        ),
        Line2D(
            [],
            [],
            marker="D",
            color=REFERENCE_COLORS["JRC"],
            linestyle="none",
            label="JRC",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    return fig


def build_case_curve_data(
    table: pd.DataFrame,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    candidate_path: str | Path = DEFAULT_CANDIDATE_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    primary_cache_dir: str | Path = DEFAULT_PRIMARY_CACHE_DIR,
    expanded_cache_dir: str | Path = DEFAULT_EXPANDED_CACHE_DIR,
    grid_steps: int = DEFAULT_GRID_STEPS,
) -> tuple[dict[tuple[str, int], pd.DataFrame], dict[tuple[str, int], pd.DataFrame]]:
    """Load authoritative cached splits and build descriptive held-out responses."""

    inputs = load_tier1_inputs(
        data_dir=data_dir,
        candidate_path=candidate_path,
        manifest_path=manifest_path,
        primary_cache_dir=primary_cache_dir,
        expanded_cache_dir=expanded_cache_dir,
    )
    samples: dict[tuple[str, int], pd.DataFrame] = {}
    responses: dict[tuple[str, int], pd.DataFrame] = {}
    for reference in REFERENCES:
        by_lake = {
            lake_id: inputs.samples[(reference, lake_id)] for lake_id in TIER1_IDS
        }
        grid = training_threshold_grid(by_lake, grid_steps=grid_steps)
        for selected_reference, lake_id, _hypothesis in CASE_STUDIES:
            if selected_reference != reference:
                continue
            frame = by_lake[lake_id]
            test = frame[frame["split"] == "test"].copy()
            if not test["split"].eq("test").all():
                raise Tier1PlotError("Case response contains non-test rows.")
            samples[(reference, lake_id)] = test
            responses[(reference, lake_id)] = threshold_response(test, grid)
    if set(table["Hylak_id"].astype(int)) != set(TIER1_IDS):
        raise Tier1PlotError("Explanatory table and case-cache lake sets differ.")
    return samples, responses


def plot_case_studies(
    table: pd.DataFrame,
    samples: dict[tuple[str, int], pd.DataFrame],
    responses: dict[tuple[str, int], pd.DataFrame],
) -> mpl.figure.Figure:
    """Combine held-out class distributions with response-curve insets."""

    indexed = table.set_index(["reference", "Hylak_id"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    bins = np.linspace(-1, 1, 41)
    threshold_styles = {
        "local_threshold": ("#222222", ":", "Local"),
        "lolo_threshold": ("#666666", "-.", "LOLO"),
        "regional_threshold": ("#111111", "--", "Regional"),
    }
    for axis, (reference, lake_id, _hypothesis) in zip(
        axes.flat, CASE_STUDIES, strict=True
    ):
        row = indexed.loc[(reference, lake_id)]
        test = samples[(reference, lake_id)]
        response = responses[(reference, lake_id)]
        nonwater = test.loc[test["label"] == 0, "index_value"]
        water = test.loc[test["label"] == 1, "index_value"]
        axis.hist(
            nonwater,
            bins=bins,
            density=True,
            histtype="step",
            color="#666666",
            linestyle="--",
            linewidth=1.1,
            label="Non-water",
        )
        axis.hist(
            water,
            bins=bins,
            density=True,
            histtype="step",
            color=REFERENCE_COLORS[reference],
            linewidth=1.3,
            label="Water",
        )
        for column, (color, linestyle, _label) in threshold_styles.items():
            axis.axvline(
                float(row[column]), color=color, linestyle=linestyle, linewidth=0.9
            )
        axis.set_xlim(-1, 1)
        axis.set_xlabel("Held-out NDWI")
        axis.set_ylabel("Class density")
        axis.set_title(f"{row['Lake_name']} - {reference}", loc="left")
        axis.grid(axis="x", color="#E3E3E3", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.02,
            0.96,
            (
                f"AUC {row['test_roc_auc']:.3f}\n"
                f"IoU local/LOLO {row['local_test_iou']:.3f}/"
                f"{row['lolo_test_iou']:.3f}\n"
                f"Plateau {row['plateau_width_normalized']:.3f} of grid"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
        )
        inset = axis.inset_axes([0.57, 0.52, 0.40, 0.38])
        inset.plot(
            response["threshold"],
            response["iou"],
            color=REFERENCE_COLORS[reference],
            linewidth=1.0,
        )
        for column, (color, linestyle, _label) in threshold_styles.items():
            inset.axvline(
                float(row[column]), color=color, linestyle=linestyle, linewidth=0.7
            )
        inset.set_ylim(0, 1.02)
        inset.set_xlim(-0.30, 0.40)
        inset.set_title("Test IoU response", fontsize=6.5)
        inset.tick_params(labelsize=5.5, length=2)
        inset.grid(color="#E8E8E8", linewidth=0.4)
    handles = [
        Line2D([], [], color="#666666", linestyle="--", label="Non-water"),
        Line2D([], [], color="#0072B2", label="Water distribution"),
        *[
            Line2D([], [], color=color, linestyle=linestyle, label=label)
            for color, linestyle, label in threshold_styles.values()
        ],
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    return fig


def _association_row(
    associations: pd.DataFrame,
    reference: str,
    outcome: str,
    variable: str,
) -> pd.Series:
    rows = associations[
        (associations["reference"] == reference)
        & (associations["outcome"] == outcome)
        & (associations["explanatory_variable"] == variable)
    ]
    if len(rows) != 1:
        raise Tier1PlotError(
            f"Expected one association row for {reference}/{outcome}/{variable}."
        )
    return rows.iloc[0]


def _rho_sentence(
    associations: pd.DataFrame,
    reference: str,
    outcome: str,
    variable: str,
    label: str,
) -> str:
    row = _association_row(associations, reference, outcome, variable)
    return (
        f"{label}: rho={row.spearman_rho:.3f}, leave-one-lake-out range "
        f"[{row.leave_one_lake_out_rho_min:.3f}, "
        f"{row.leave_one_lake_out_rho_max:.3f}], most influenced by "
        f"{row.most_influential_lake}."
    )


def findings_text(
    table: pd.DataFrame,
    associations: pd.DataFrame,
    cases: pd.DataFrame,
) -> str:
    """Write a fixed-theme, descriptive interpretation for n=12/reference."""

    case_lines = []
    for row in cases.itertuples(index=False):
        case_lines.append(
            f"- **{row.Lake_name}/{row.reference}:** {row.data_assessment}; "
            f"AUC {row.test_roc_auc:.3f}, local/LOLO IoU "
            f"{row.local_test_iou:.3f}/{row.lolo_test_iou:.3f}, normalized "
            f"plateau width {row.plateau_width_normalized:.3f}."
        )
    endpoint = table.groupby("reference")["ndwi_one_fraction_test"].median()
    return f"""# Tier 1 transfer explanatory findings

This is a descriptive analysis of **n = 12 lakes per reference**. It uses the
completed OPERA and JRC Tier 1 thresholds and authoritative cached spatial-block
test assignments. It does not refit thresholds, reconstruct blocks, or provide a
predictive/causal model.

## Discrimination and absolute transfer performance

{_rho_sentence(associations, "OPERA", "lolo_test_iou", "test_ks_statistic", "OPERA KS versus LOLO IoU")}
{_rho_sentence(associations, "JRC", "lolo_test_iou", "test_ks_statistic", "JRC KS versus LOLO IoU")}
Average precision is similarly aligned with absolute LOLO IoU (OPERA rho
{_association_row(associations, "OPERA", "lolo_test_iou", "test_average_precision").spearman_rho:.3f};
JRC rho {_association_row(associations, "JRC", "lolo_test_iou", "test_average_precision").spearman_rho:.3f}).
This supports poor or weaker class discrimination as one mechanism for a low
absolute IoU. It does not by itself explain degradation relative to each lake's
local threshold.

## Threshold geometry

{_rho_sentence(associations, "OPERA", "lolo_test_iou", "plateau_width_normalized", "OPERA plateau width versus LOLO IoU")}
{_rho_sentence(associations, "JRC", "lolo_test_iou", "plateau_width_normalized", "JRC plateau width versus LOLO IoU")}
Across all 12 lakes, threshold distance is only weakly related to LOLO-minus-local
IoU (OPERA rho {_association_row(associations, "OPERA", "lolo_iou_delta", "abs_local_to_regional_threshold").spearman_rho:.3f};
JRC rho {_association_row(associations, "JRC", "lolo_iou_delta", "abs_local_to_regional_threshold").spearman_rho:.3f})
and its influence ranges are broad. The individual Honey/OPERA, Mono/JRC, and
Goose/JRC response curves nevertheless show that a narrow or displaced operating
region can matter for a particular lake even when a rank association is unstable.

## Sample and temporal behavior

JRC held-out positive fraction has a strong descriptive association with LOLO
IoU: {_rho_sentence(associations, "JRC", "lolo_test_iou", "positive_fraction_test", "positive fraction")}
Acquisition-prevalence variability has moderate positive associations with
absolute IoU (OPERA rho {_association_row(associations, "OPERA", "lolo_test_iou", "acquisition_prevalence_sd").spearman_rho:.3f};
JRC rho {_association_row(associations, "JRC", "lolo_test_iou", "acquisition_prevalence_sd").spearman_rho:.3f}),
but these compact samples do not separate hydrologic variability from sampling
availability or class balance. Median held-out exact-NDWI=1 fractions are
{endpoint["OPERA"]:.3f} for OPERA and {endpoint["JRC"]:.3f} for JRC. The endpoint
is retained; this table does not repeat the completed raw-band diagnosis.

## HydroLAKES context and limitations

Lake area has little descriptive association with OPERA absolute LOLO IoU
(rho {_association_row(associations, "OPERA", "lolo_test_iou", "log10_Lake_area").spearman_rho:.3f}).
Some physical-attribute correlations are larger, including JRC watershed area
versus LOLO-minus-local IoU
(rho {_association_row(associations, "JRC", "lolo_iou_delta", "log10_Wshd_area").spearman_rho:.3f}),
but n=12, linked covariates, and exploratory selection prevent a universal
interpretation. Missing HydroLAKES sentinels were not imputed. Tier 1 has 11
Lake_type 1 records and only one Lake_type 2 record, so it cannot evaluate lake-
type effects; it also contains no type-3 lake.

## Diagnostic cases

{chr(10).join(case_lines)}

These are mechanisms illustrated by particular held-out samples, not definitive
intrinsic lake classes. Pyramid/OPERA represents good discrimination and a broad
tolerance interval; Honey/OPERA and Mono/JRC demonstrate threshold displacement;
Goose/JRC combines a low local ceiling, lower AP than the strongest cases, and a
narrow response interval.

## Decision for Tier 2

Tier 2 is scientifically justified only as a **targeted mechanism test**, not a
routine scale-up. The clearest gaps are the single type-2 lake, absence of type 3,
and limited reservoir, deep-lake, high-elevation, and stable-freshwater contrasts.
A targeted run should test whether class balance/discrimination and plateau width
remain explanatory while adding those contrasts, especially where leave-one-lake-
out influence ranges show sensitivity to Honey, Mono, Goose, or Malheur. The
current n=12 evidence is insufficient for a predictive model or causal claim.
"""


def case_imagery_index_text(table: pd.DataFrame) -> str:
    """Reference existing qualitative images without treating them as validation."""

    indexed = table.set_index(["reference", "Hylak_id"])
    entries = (
        (
            "Lake Abert/OPERA",
            "../imagery_examples/static_lake_abert_opera_2023-06-15.png",
            "strong-transfer visual contrast",
            "OPERA",
            776,
        ),
        (
            "Honey Lake/OPERA",
            "../imagery_examples/static_honey_lake_opera_2023-07-01.png",
            "shifted/narrow weak-transfer case",
            "OPERA",
            789,
        ),
        (
            "Goose Lake/JRC",
            "../imagery_examples/static_goose_lake_jrc_2017-06-30.png",
            "low-ceiling and threshold-sensitive case",
            "JRC",
            780,
        ),
        (
            "Mono Lake/JRC",
            "../imagery_examples/static_mono_lake_jrc_2017-07-18.png",
            "high-discrimination shifted-threshold case",
            "JRC",
            798,
        ),
        (
            "Great Salt/JRC change",
            "../imagery_examples/change_great_salt_jrc.gif",
            "temporal scene context",
            "JRC",
            67,
        ),
        (
            "Mono Lake/JRC change",
            "../imagery_examples/change_mono_lake_jrc.gif",
            "temporal context for the shifted-threshold case",
            "JRC",
            798,
        ),
        (
            "Goose Lake/JRC change",
            "../imagery_examples/change_goose_lake_jrc.gif",
            "temporal context for the weaker case",
            "JRC",
            780,
        ),
        (
            "Great Salt/OPERA endpoint context",
            "../imagery_examples/point_mass_scene_great_salt_opera_2026-01-19.png",
            "NDWI=1 scene context",
            "OPERA",
            67,
        ),
        (
            "Mono Lake/JRC endpoint context",
            "../imagery_examples/point_mass_scene_mono_lake_jrc_2017-07-18.png",
            "NDWI=1 scene context",
            "JRC",
            798,
        ),
    )
    lines = [
        "# Existing imagery index for Tier 1 explanatory cases",
        "",
        (
            "These existing local thumbnail/GIF products are qualitative context, "
            "not independent validation. Their 8-bit rendering does not replace "
            "the cached-sample metrics."
        ),
        "",
    ]
    imagery_root = REPOSITORY_ROOT / "outputs" / "imagery_examples"
    for label, relative, role, reference, lake_id in entries:
        row = indexed.loc[(reference, lake_id)]
        filename = Path(relative).name
        availability = "available" if (imagery_root / filename).is_file() else "missing"
        lines.append(
            f"- [{label}]({relative}) ({availability}): {role}; local/LOLO IoU "
            f"{row.local_test_iou:.3f}/{row.lolo_test_iou:.3f}, local/LOLO/"
            f"regional thresholds {row.local_threshold:.4f}/"
            f"{row.lolo_threshold:.4f}/{row.regional_threshold:.4f}. "
            "Limitation: one or a few presentation scenes cannot establish "
            "the multi-year transfer mechanism."
        )
    return "\n".join(lines) + "\n"


def create_explanatory_figures(
    table: pd.DataFrame,
    associations: pd.DataFrame,
    cases: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    candidate_path: str | Path = DEFAULT_CANDIDATE_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    primary_cache_dir: str | Path = DEFAULT_PRIMARY_CACHE_DIR,
    expanded_cache_dir: str | Path = DEFAULT_EXPANDED_CACHE_DIR,
    grid_steps: int = DEFAULT_GRID_STEPS,
) -> tuple[Path, ...]:
    """Write five PNG/SVG figures, no-text twins, and explanatory notes."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples, responses = build_case_curve_data(
        table,
        data_dir=data_dir,
        candidate_path=candidate_path,
        manifest_path=manifest_path,
        primary_cache_dir=primary_cache_dir,
        expanded_cache_dir=expanded_cache_dir,
        grid_steps=grid_steps,
    )
    saved: list[Path] = []
    with mpl.rc_context(PLOT_STYLE):
        figures = (
            (FIGURE_NAMES[0], plot_transfer_vs_discrimination(table)),
            (FIGURE_NAMES[1], plot_threshold_geometry(table)),
            (FIGURE_NAMES[2], plot_temporal_sample_behavior(table)),
            (FIGURE_NAMES[3], plot_hydrolakes_context(table)),
            (FIGURE_NAMES[4], plot_case_studies(table, samples, responses)),
        )
        for name, figure in figures:
            saved.extend(save_figure_with_no_text(figure, output, name))
    notes = output / "figure_notes.md"
    notes.write_text(
        "# Tier 1 explanatory figure notes\n\n"
        "All panels use the completed 12-lake OPERA/JRC experiment and the "
        "authoritative cached spatial-block test rows. GS=Great Salt, AB=Lake "
        "Abert, MO=Mono, PY=Pyramid, WA=Walker, SA=Salton, GO=Goose, MA=Malheur, "
        "HO=Honey, SU=Summer, EA=Eagle, UT=Utah. Filled symbols are the original "
        "six and open symbols the Tier 1 additions; these are staged design labels, "
        "not randomized strata. Figure 1 relates absolute LOLO IoU to held-out "
        "ranking/separation. Figure 2 shows saved-threshold distance and the "
        "contiguous within-0.02 IoU plateau against transfer loss. Figure 3 uses "
        "three pre-specified temporal/sample diagnostics. Figure 4 provides "
        "HydroLAKES context and reports complete-case n; Lake_type is deliberately "
        "not compared. Figure 5 combines held-out class distributions with a "
        "descriptive 150-point test-response inset for four pre-declared cases. "
        "Plateau widths are grid-based, so zero-width intervals are resolution-"
        "limited rather than proof that no threshold tolerance exists. "
        "Sweeping a held-out response is diagnostic only: it does not refit or "
        "replace any saved threshold. Correlations and visuals are descriptive "
        "at n=12 per reference, not causal or universal.\n",
        encoding="utf-8",
    )
    findings = output / "tier1_explanatory_findings.md"
    findings.write_text(findings_text(table, associations, cases), encoding="utf-8")
    imagery = output / "case_study_imagery_index.md"
    imagery.write_text(case_imagery_index_text(table), encoding="utf-8")
    saved.extend((notes, findings, imagery))
    return tuple(saved)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 2:
        raise argparse.ArgumentTypeError("must be at least two")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot completed Tier 1 explanatory CSVs and authoritative local "
            "test caches. No Earth Engine call occurs."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--primary-cache-dir", type=Path, default=DEFAULT_PRIMARY_CACHE_DIR
    )
    parser.add_argument(
        "--expanded-cache-dir", type=Path, default=DEFAULT_EXPANDED_CACHE_DIR
    )
    parser.add_argument("--grid-steps", type=_positive_int, default=DEFAULT_GRID_STEPS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        table, associations, cases = load_explanatory_products(args.data_dir)
        saved = create_explanatory_figures(
            table,
            associations,
            cases,
            args.output_dir,
            data_dir=args.data_dir,
            candidate_path=args.candidate_csv,
            manifest_path=args.manifest,
            primary_cache_dir=args.primary_cache_dir,
            expanded_cache_dir=args.expanded_cache_dir,
            grid_steps=args.grid_steps,
        )
        print(f"Wrote {len(saved)} figure/note files under {args.output_dir}")
        return 0
    except (OSError, Tier1ExplanatoryError, Tier1PlotError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
