"""Plot compact summaries of the completed six-lake transfer experiment.

This module is deliberately local-only: it reads the compact transfer CSVs,
validates their lake sets and schemas, and creates figures with matplotlib.  It
does not import or initialize Earth Engine and does not recompute thresholds.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data" / "regional"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "quick_transfer_figures"

REFERENCES = ("OPERA", "JRC")
LAKES = (
    (67, "Great Salt"),
    (776, "Lake Abert"),
    (798, "Mono Lake"),
    (791, "Pyramid Lake"),
    (795, "Walker Lake"),
    (829, "Salton"),
)
LAKE_IDS = tuple(lake_id for lake_id, _ in LAKES)
LAKE_NAMES = tuple(name for _, name in LAKES)
LAKE_NAME_BY_ID = dict(LAKES)

# Okabe-Ito blue and vermillion, reinforced with marker differences.
REFERENCE_COLORS = {"OPERA": "#0072B2", "JRC": "#D55E00"}
REFERENCE_MARKERS = {"OPERA": "o", "JRC": "D"}

FIGURE_NAMES = (
    "local_vs_lolo_iou",
    "cross_lake_iou_matrices",
    "threshold_portability",
    "lolo_iou_delta",
)
FIGURE_FORMATS = ("png", "svg")
PNG_DPI = 300

PLOT_STYLE: dict[str, Any] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.0,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
}


class PlotDataError(ValueError):
    """Raised when transfer result files cannot support the requested plots."""


@dataclass(frozen=True)
class ReferencePlotData:
    """Validated compact transfer tables for one reference."""

    reference: str
    local: pd.DataFrame
    cross_iou: pd.DataFrame
    cross_f1: pd.DataFrame
    regional: pd.DataFrame
    lolo: pd.DataFrame


def _source_paths(data_dir: str | Path, reference: str) -> dict[str, Path]:
    normalized = reference.lower()
    prefix = Path(data_dir) / f"western_quick_transfer_{normalized}"
    return {
        "local": prefix.with_name(prefix.name + "_local.csv"),
        "cross_iou": prefix.with_name(prefix.name + "_cross_matrix_iou.csv"),
        "cross_f1": prefix.with_name(prefix.name + "_cross_matrix_f1.csv"),
        "regional": prefix.with_name(prefix.name + "_regional.csv"),
        "lolo": prefix.with_name(prefix.name + "_lolo.csv"),
    }


def source_csv_paths(data_dir: str | Path = DEFAULT_DATA_DIR) -> tuple[Path, ...]:
    """Return all ten compact CSV paths in deterministic reference/table order."""

    return tuple(
        path
        for reference in REFERENCES
        for path in _source_paths(data_dir, reference).values()
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise PlotDataError(f"Required transfer result file is missing: {path}")
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise PlotDataError(
            f"Could not read transfer result file {path}: {exc}"
        ) from exc


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    table_label: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PlotDataError(
            f"{table_label} is missing required columns: {', '.join(missing)}"
        )


def _coerce_ids(
    frame: pd.DataFrame,
    id_column: str,
    *,
    table_label: str,
) -> pd.DataFrame:
    result = frame.copy()
    try:
        numeric = pd.to_numeric(result[id_column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise PlotDataError(
            f"{table_label}.{id_column} must contain integer IDs."
        ) from exc
    if numeric.isna().any() or not np.all(np.equal(numeric, np.floor(numeric))):
        raise PlotDataError(f"{table_label}.{id_column} must contain integer IDs.")
    result[id_column] = numeric.astype(int)
    if result[id_column].duplicated().any():
        duplicates = sorted(
            result.loc[result[id_column].duplicated(), id_column].unique()
        )
        raise PlotDataError(f"{table_label} has duplicate lake IDs: {duplicates}")
    actual_ids = set(result[id_column])
    expected_ids = set(LAKE_IDS)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise PlotDataError(
            f"{table_label} has the wrong lake set; missing={missing}, extra={extra}."
        )
    return (
        result.set_index(id_column, drop=False)
        .loc[list(LAKE_IDS)]
        .reset_index(drop=True)
    )


def _validate_names(
    frame: pd.DataFrame,
    *,
    id_column: str,
    name_column: str,
    table_label: str,
) -> None:
    mismatches = [
        f"{int(row[id_column])}: {row[name_column]!r}"
        for _, row in frame.iterrows()
        if row[name_column] != LAKE_NAME_BY_ID[int(row[id_column])]
    ]
    if mismatches:
        raise PlotDataError(
            f"{table_label} has unexpected lake names: {', '.join(mismatches)}"
        )


def _validate_reference(
    frame: pd.DataFrame,
    reference: str,
    *,
    table_label: str,
) -> None:
    actual = set(frame["reference"].dropna().astype(str))
    if actual != {reference} or frame["reference"].isna().any():
        raise PlotDataError(
            f"{table_label}.reference must contain only {reference}; found {sorted(actual)}."
        )


def _validate_numeric(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    table_label: str,
    unit_interval: bool = False,
) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise PlotDataError(f"{table_label}.{column} must be numeric.") from exc
        if not np.isfinite(values).all():
            raise PlotDataError(
                f"{table_label}.{column} contains missing/non-finite values."
            )
        if unit_interval and ((values < 0.0) | (values > 1.0)).any():
            raise PlotDataError(f"{table_label}.{column} must be within [0, 1].")


def _load_matrix(path: Path, *, table_label: str) -> pd.DataFrame:
    frame = _read_csv(path)
    _require_columns(frame, {"source_Hylak_id"}, table_label=table_label)
    try:
        source_ids = pd.to_numeric(frame["source_Hylak_id"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise PlotDataError(f"{table_label} source IDs must be integers.") from exc
    if source_ids.isna().any() or not np.all(
        np.equal(source_ids, np.floor(source_ids))
    ):
        raise PlotDataError(f"{table_label} source IDs must be integers.")
    if source_ids.duplicated().any():
        raise PlotDataError(f"{table_label} has duplicate source-lake rows.")

    target_columns = [column for column in frame.columns if column != "source_Hylak_id"]
    try:
        target_ids = [int(column) for column in target_columns]
    except (TypeError, ValueError) as exc:
        raise PlotDataError(
            f"{table_label} target columns must be HydroLAKES IDs."
        ) from exc
    if len(set(target_ids)) != len(target_ids):
        raise PlotDataError(f"{table_label} has duplicate target-lake columns.")
    if set(source_ids.astype(int)) != set(LAKE_IDS) or set(target_ids) != set(LAKE_IDS):
        raise PlotDataError(
            f"{table_label} must have the same six source and target lakes: {list(LAKE_IDS)}."
        )

    matrix = frame.drop(columns="source_Hylak_id").copy()
    matrix.index = pd.Index(source_ids.astype(int), name="source_Hylak_id")
    matrix.columns = pd.Index(target_ids, name="target_Hylak_id")
    try:
        matrix = matrix.apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise PlotDataError(f"{table_label} cells must be numeric.") from exc
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise PlotDataError(f"{table_label} cells must be finite values within [0, 1].")
    return matrix.loc[list(LAKE_IDS), list(LAKE_IDS)]


def load_reference_plot_data(
    data_dir: str | Path,
    reference: str,
) -> ReferencePlotData:
    """Read and validate all five compact tables for one reference."""

    if reference not in REFERENCES:
        raise PlotDataError(
            f"Reference must be one of {REFERENCES}; got {reference!r}."
        )
    paths = _source_paths(data_dir, reference)
    local_label = f"{reference} local table"
    lolo_label = f"{reference} LOLO table"
    regional_label = f"{reference} regional table"

    local = _read_csv(paths["local"])
    _require_columns(
        local,
        {"reference", "Hylak_id", "Lake_name", "local_threshold", "local_test_iou"},
        table_label=local_label,
    )
    local = _coerce_ids(local, "Hylak_id", table_label=local_label)
    _validate_reference(local, reference, table_label=local_label)
    _validate_names(
        local,
        id_column="Hylak_id",
        name_column="Lake_name",
        table_label=local_label,
    )
    _validate_numeric(
        local,
        ("local_threshold",),
        table_label=local_label,
    )
    _validate_numeric(
        local,
        ("local_test_iou",),
        table_label=local_label,
        unit_interval=True,
    )

    lolo = _read_csv(paths["lolo"])
    _require_columns(
        lolo,
        {
            "reference",
            "held_out_Hylak_id",
            "held_out_lake",
            "threshold",
            "held_out_test_iou",
            "local_threshold",
            "local_test_iou",
            "iou_delta_vs_local",
        },
        table_label=lolo_label,
    )
    lolo = _coerce_ids(lolo, "held_out_Hylak_id", table_label=lolo_label)
    _validate_reference(lolo, reference, table_label=lolo_label)
    _validate_names(
        lolo,
        id_column="held_out_Hylak_id",
        name_column="held_out_lake",
        table_label=lolo_label,
    )
    _validate_numeric(
        lolo,
        ("threshold", "local_threshold", "iou_delta_vs_local"),
        table_label=lolo_label,
    )
    _validate_numeric(
        lolo,
        ("held_out_test_iou", "local_test_iou"),
        table_label=lolo_label,
        unit_interval=True,
    )

    regional = _read_csv(paths["regional"])
    _require_columns(
        regional,
        {"reference", "Hylak_id", "Lake_name", "regional_threshold"},
        table_label=regional_label,
    )
    regional = _coerce_ids(regional, "Hylak_id", table_label=regional_label)
    _validate_reference(regional, reference, table_label=regional_label)
    _validate_names(
        regional,
        id_column="Hylak_id",
        name_column="Lake_name",
        table_label=regional_label,
    )
    _validate_numeric(
        regional,
        ("regional_threshold",),
        table_label=regional_label,
    )

    cross_iou = _load_matrix(
        paths["cross_iou"], table_label=f"{reference} cross-IoU matrix"
    )
    cross_f1 = _load_matrix(
        paths["cross_f1"], table_label=f"{reference} cross-F1 matrix"
    )

    for column in ("local_threshold", "local_test_iou"):
        local_values = local[column].to_numpy(dtype=float)
        lolo_values = lolo[column].to_numpy(dtype=float)
        if not np.allclose(local_values, lolo_values, rtol=0.0, atol=1e-12):
            raise PlotDataError(
                f"{reference} {column} differs between local and LOLO tables."
            )
    calculated_delta = lolo["held_out_test_iou"].to_numpy(dtype=float) - local[
        "local_test_iou"
    ].to_numpy(dtype=float)
    if not np.allclose(
        calculated_delta,
        lolo["iou_delta_vs_local"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise PlotDataError(
            f"{reference} iou_delta_vs_local does not equal LOLO IoU - local IoU."
        )
    if not np.allclose(
        regional["regional_threshold"].to_numpy(dtype=float),
        float(regional["regional_threshold"].iloc[0]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise PlotDataError(f"{reference} regional_threshold is not constant by lake.")

    return ReferencePlotData(
        reference=reference,
        local=local,
        cross_iou=cross_iou,
        cross_f1=cross_f1,
        regional=regional,
        lolo=lolo,
    )


def load_plot_data(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict[str, ReferencePlotData]:
    """Load both references and ensure their validated lake sets agree."""

    data = {
        reference: load_reference_plot_data(data_dir, reference)
        for reference in REFERENCES
    }
    opera_ids = tuple(data["OPERA"].local["Hylak_id"])
    jrc_ids = tuple(data["JRC"].local["Hylak_id"])
    if opera_ids != jrc_ids or opera_ids != LAKE_IDS:
        raise PlotDataError("OPERA and JRC do not contain the same ordered lake set.")
    return data


def build_verification_table(
    data: dict[str, ReferencePlotData],
) -> pd.DataFrame:
    """Build the compact QA table from validated CSV values."""

    records: list[dict[str, object]] = []
    for lake_id, lake_name in LAKES:
        for reference in REFERENCES:
            reference_data = data[reference]
            local_row = reference_data.local.loc[
                reference_data.local["Hylak_id"] == lake_id
            ].iloc[0]
            lolo_row = reference_data.lolo.loc[
                reference_data.lolo["held_out_Hylak_id"] == lake_id
            ].iloc[0]
            records.append(
                {
                    "lake": lake_name,
                    "reference": reference,
                    "local threshold": float(local_row["local_threshold"]),
                    "local IoU": float(local_row["local_test_iou"]),
                    "LOLO threshold": float(lolo_row["threshold"]),
                    "LOLO IoU": float(lolo_row["held_out_test_iou"]),
                    "IoU delta": float(lolo_row["iou_delta_vs_local"]),
                }
            )
    return pd.DataFrame.from_records(records)


def _ordered_values(
    frame: pd.DataFrame, id_column: str, value_column: str
) -> np.ndarray:
    indexed = frame.set_index(id_column)
    return indexed.loc[list(LAKE_IDS), value_column].to_numpy(dtype=float)


def plot_local_vs_lolo_iou(
    data: dict[str, ReferencePlotData],
) -> mpl.figure.Figure:
    """Plot local and leave-one-lake-out held-out IoU for both references."""

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.5, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    y = np.arange(len(LAKES))
    for axis, reference in zip(axes, REFERENCES, strict=True):
        reference_data = data[reference]
        local_iou = _ordered_values(reference_data.local, "Hylak_id", "local_test_iou")
        lolo_iou = _ordered_values(
            reference_data.lolo, "held_out_Hylak_id", "held_out_test_iou"
        )
        color = REFERENCE_COLORS[reference]
        for y_value, local_value, lolo_value in zip(
            y, local_iou, lolo_iou, strict=True
        ):
            axis.plot(
                [local_value, lolo_value],
                [y_value, y_value],
                color="#8A8A8A",
                linewidth=1.0,
                zorder=1,
            )
        axis.scatter(
            local_iou,
            y,
            s=36,
            marker="o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        axis.scatter(
            lolo_iou,
            y,
            s=38,
            marker="D",
            facecolor="none",
            edgecolor=color,
            linewidth=1.3,
            zorder=4,
        )
        axis.set_title(reference)
        axis.set_xlabel("Held-out test IoU")
        axis.set_xlim(0.78, 1.005)
        axis.set_xticks(np.arange(0.80, 1.001, 0.05))
        axis.set_yticks(y, LAKE_NAMES)
        axis.grid(axis="x", color="#D8D8D8", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Lake")
    axes[1].tick_params(labelleft=False)

    mono_index = LAKE_IDS.index(798)
    mono_delta = float(
        data["JRC"]
        .lolo.loc[data["JRC"].lolo["held_out_Hylak_id"] == 798, "iou_delta_vs_local"]
        .iloc[0]
    )
    mono_local = float(
        data["JRC"]
        .local.loc[data["JRC"].local["Hylak_id"] == 798, "local_test_iou"]
        .iloc[0]
    )
    mono_lolo = float(
        data["JRC"]
        .lolo.loc[data["JRC"].lolo["held_out_Hylak_id"] == 798, "held_out_test_iou"]
        .iloc[0]
    )
    axes[1].annotate(
        f"Δ = {mono_delta:+.3f}",
        xy=((mono_local + mono_lolo) / 2.0, mono_index),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color="#3A3A3A",
        fontsize=8.2,
    )

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=6.5,
            label="Local threshold",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#555555",
            markersize=6,
            label="LOLO threshold",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    return fig


def plot_cross_lake_iou_matrices(
    data: dict[str, ReferencePlotData],
) -> mpl.figure.Figure:
    """Plot source-threshold by target-lake IoU matrices on one color scale."""

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 5.2),
        constrained_layout=True,
    )
    image = None
    for axis, reference in zip(axes, REFERENCES, strict=True):
        values = data[reference].cross_iou.to_numpy(dtype=float)
        image = axis.imshow(
            values,
            cmap="cividis",
            vmin=0.70,
            vmax=1.00,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(reference)
        axis.set_xlabel("Target lake")
        axis.set_ylabel("Threshold-source lake")
        axis.set_xticks(np.arange(len(LAKES)), LAKE_NAMES, rotation=32, ha="right")
        axis.set_yticks(np.arange(len(LAKES)), LAKE_NAMES)
        axis.tick_params(length=0)
        for row in range(len(LAKES)):
            for column in range(len(LAKES)):
                value = values[row, column]
                text_color = "white" if value < 0.84 else "#181818"
                axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7.6,
                )
            axis.add_patch(
                Rectangle(
                    (row - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#2F2F2F",
                    linewidth=0.75,
                    linestyle=(0, (2, 2)),
                )
            )
        for spine in axis.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#666666")
    if image is None:  # pragma: no cover - REFERENCES is a non-empty constant.
        raise RuntimeError("No transfer matrices were plotted.")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.025)
    colorbar.set_label("Test IoU")
    colorbar.set_ticks(np.arange(0.70, 1.001, 0.05))
    colorbar.outline.set_linewidth(0.7)
    return fig


def plot_threshold_portability(
    data: dict[str, ReferencePlotData],
) -> mpl.figure.Figure:
    """Plot local, LOLO, and pooled-regional NDWI thresholds."""

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.7),
        sharey=True,
        constrained_layout=True,
    )
    x = np.arange(len(LAKES))
    offset = 0.09
    for axis, reference in zip(axes, REFERENCES, strict=True):
        reference_data = data[reference]
        local_threshold = _ordered_values(
            reference_data.local, "Hylak_id", "local_threshold"
        )
        lolo_threshold = _ordered_values(
            reference_data.lolo, "held_out_Hylak_id", "threshold"
        )
        regional_threshold = float(
            reference_data.regional["regional_threshold"].iloc[0]
        )
        color = REFERENCE_COLORS[reference]
        for x_value, local_value, lolo_value in zip(
            x, local_threshold, lolo_threshold, strict=True
        ):
            axis.plot(
                [x_value - offset, x_value + offset],
                [local_value, lolo_value],
                color="#999999",
                linewidth=0.8,
                zorder=1,
            )
        axis.scatter(
            x - offset,
            local_threshold,
            marker="o",
            s=35,
            facecolor=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        axis.scatter(
            x + offset,
            lolo_threshold,
            marker="D",
            s=36,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=4,
        )
        axis.axhline(
            regional_threshold,
            color="#333333",
            linewidth=1.1,
            linestyle=(0, (4, 3)),
            zorder=2,
        )
        axis.axhline(0.0, color="#D0D0D0", linewidth=0.7, zorder=0)
        axis.set_title(reference)
        axis.set_xlabel("Lake")
        axis.set_xticks(x, LAKE_NAMES, rotation=28, ha="right")
        axis.set_ylim(-0.105, 0.105)
        axis.set_yticks(np.arange(-0.10, 0.101, 0.05))
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("NDWI threshold")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=6.5,
            label="Local",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#555555",
            markersize=6,
            label="LOLO",
        ),
        Line2D(
            [],
            [],
            color="#333333",
            linestyle=(0, (4, 3)),
            linewidth=1.1,
            label="Pooled regional",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    return fig


def plot_lolo_iou_delta(
    data: dict[str, ReferencePlotData],
) -> mpl.figure.Figure:
    """Plot LOLO-minus-local test IoU for every lake and reference."""

    fig, axis = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    y = np.arange(len(LAKES))
    offsets = {"OPERA": -0.12, "JRC": 0.12}
    for reference in REFERENCES:
        delta = _ordered_values(
            data[reference].lolo, "held_out_Hylak_id", "iou_delta_vs_local"
        )
        y_values = y + offsets[reference]
        color = REFERENCE_COLORS[reference]
        for y_value, delta_value in zip(y_values, delta, strict=True):
            axis.plot(
                [0.0, delta_value],
                [y_value, y_value],
                color=color,
                linewidth=1.1,
                alpha=0.65,
                zorder=1,
            )
        axis.scatter(
            delta,
            y_values,
            s=37,
            marker=REFERENCE_MARKERS[reference],
            facecolor=color if reference == "OPERA" else "white",
            edgecolor=color,
            linewidth=1.2,
            label=reference,
            zorder=3,
        )
    axis.axvline(0.0, color="#333333", linewidth=0.9, zorder=0)
    axis.set_xlim(-0.060, 0.030)
    axis.set_xticks(np.arange(-0.06, 0.031, 0.01))
    axis.set_yticks(y, LAKE_NAMES)
    axis.invert_yaxis()
    axis.set_xlabel("LOLO test IoU − local test IoU")
    axis.set_ylabel("Lake")
    axis.set_title("Transfer difference relative to local thresholds")
    axis.grid(axis="x", color="#DEDEDE", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right", ncol=2)

    mono_delta = float(
        data["JRC"]
        .lolo.loc[data["JRC"].lolo["held_out_Hylak_id"] == 798, "iou_delta_vs_local"]
        .iloc[0]
    )
    axis.annotate(
        f"{mono_delta:+.3f}",
        xy=(mono_delta, LAKE_IDS.index(798) + offsets["JRC"]),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        color=REFERENCE_COLORS["JRC"],
        fontsize=8.2,
    )
    return fig


def _remove_all_text(fig: mpl.figure.Figure) -> None:
    """Strip text while retaining axes, artists, color geometry, and dimensions."""

    for text_artist in list(fig.texts):
        text_artist.set_visible(False)
    for legend in list(fig.legends):
        legend.set_visible(False)
    for axis in fig.axes:
        for location in ("left", "center", "right"):
            axis.set_title("", loc=location)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
        )
        axis.xaxis.get_offset_text().set_visible(False)
        axis.yaxis.get_offset_text().set_visible(False)
        legend = axis.get_legend()
        if legend is not None:
            legend.set_visible(False)
        for text_artist in list(axis.texts):
            text_artist.set_visible(False)


def save_figure_with_no_text(
    fig: mpl.figure.Figure,
    output_dir: str | Path,
    basename: str,
) -> tuple[Path, ...]:
    """Save labeled PNG/SVG files, then equal-size text-free counterparts."""

    destination = Path(output_dir)
    no_text_dir = destination / "no_text"
    destination.mkdir(parents=True, exist_ok=True)
    no_text_dir.mkdir(parents=True, exist_ok=True)

    # Resolve constrained layout once, then freeze positions so removing text does
    # not change graphical geometry or output dimensions.
    fig.canvas.draw()
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)

    saved: list[Path] = []
    with mpl.rc_context({"svg.fonttype": "none"}):
        for suffix in FIGURE_FORMATS:
            path = destination / f"{basename}.{suffix}"
            fig.savefig(path, dpi=PNG_DPI, facecolor="white")
            saved.append(path)
        _remove_all_text(fig)
        for suffix in FIGURE_FORMATS:
            path = no_text_dir / f"{basename}.{suffix}"
            fig.savefig(path, dpi=PNG_DPI, facecolor="white")
            saved.append(path)
    plt.close(fig)
    return tuple(saved)


def figure_notes_text() -> str:
    """Return concise figure captions and the experiment-scope caveat."""

    return """# Quick threshold-transfer figure notes

These figures summarize six exploratory western saline/terminal lake examples, not a representative sample or a final western-U.S. regional validation. Every example uses its HydroLAKES polygon plus a 5 km buffer. OPERA uses the 2023-04-04 through 2026-08-09 window, whereas JRC uses 2017-01-01 through 2021-12-31, so comparisons between references also reflect different temporal coverage.

## Figure 1 — Local versus leave-one-lake-out IoU

For each OPERA- or JRC-labeled NDWI experiment, the filled circle is test-block IoU from a threshold learned with that lake's training blocks, and the open diamond is test-block IoU from a leave-one-lake-out (LOLO) threshold learned using only the other five lakes. OPERA LOLO performance is essentially unchanged or slightly improved across these examples; JRC is also strong overall, with Mono Lake showing the clearest transfer degradation.

## Figure 2 — Cross-lake IoU transfer matrices

Rows identify the lake whose training blocks supplied a local threshold, and columns identify the lake whose untouched test blocks were scored with that threshold; the diagonal is same-lake performance. OPERA thresholds transfer broadly well, while JRC shows more source-target asymmetry, including the relatively poor Mono-source to Lake-Abert application.

## Figure 3 — Threshold portability

Local thresholds, LOLO thresholds, and the equal-lake-weighted pooled regional threshold are shown for OPERA and JRC. OPERA local thresholds occupy a comparatively narrow negative range, whereas JRC local thresholds vary more and Mono Lake's positive local JRC threshold is notably separated from the regional and LOLO solutions.

## Figure 4 — Regional transfer degradation

Each point is LOLO test IoU minus local-threshold test IoU for one lake and reference, so zero indicates equal held-out performance and negative values indicate degradation. Most differences are near zero or slightly positive in this six-lake experiment; Mono Lake under JRC is the conspicuous negative case at about −0.05 IoU.
"""


def create_all_figures(
    data: dict[str, ReferencePlotData],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, ...]:
    """Create all four labeled and text-free PNG/SVG figure pairs."""

    plotters = (
        ("local_vs_lolo_iou", plot_local_vs_lolo_iou),
        ("cross_lake_iou_matrices", plot_cross_lake_iou_matrices),
        ("threshold_portability", plot_threshold_portability),
        ("lolo_iou_delta", plot_lolo_iou_delta),
    )
    saved: list[Path] = []
    with mpl.rc_context(PLOT_STYLE):
        for basename, plotter in plotters:
            saved.extend(save_figure_with_no_text(plotter(data), output_dir, basename))
    notes_path = Path(output_dir) / "figure_notes.md"
    notes_path.write_text(figure_notes_text(), encoding="utf-8")
    saved.append(notes_path)
    return tuple(saved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the completed six-lake OPERA/JRC threshold-transfer CSVs. "
            "This command is local-only and never initializes Earth Engine."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing transfer CSVs (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Figure destination (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        data = load_plot_data(args.data_dir)
        verification = build_verification_table(data)
        print("Verified figure data (CSV precision):")
        print(
            verification.to_string(
                index=False,
                float_format=lambda value: f"{value:.12g}",
            )
        )
        saved = create_all_figures(data, args.output_dir)
    except (PlotDataError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nCreated {len(saved) - 1} figure files and {saved[-1]}.")
    for path in saved[:-1]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
