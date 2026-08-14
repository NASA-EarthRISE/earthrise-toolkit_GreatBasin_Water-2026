"""Matplotlib-only quantitative 2D figures for Bear Lake Phase 2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_phase1 import save_figure_with_no_text
from .spatial_products import BYTE_NODATA, FLOAT_NODATA

FIGURE_NAMES = (
    "bear_lake_topobathymetry_stage_envelope",
    "bear_lake_current_depth",
    "bear_lake_stage_change",
    "bear_lake_eav_validation",
)

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "svg.hashsalt": "hydrometry-phase2",
}


class Phase2PlotError(RuntimeError):
    """Raised when Phase 2 plotting inputs are incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class Phase2FigureInputs:
    """Small rendering-grid arrays plus authoritative validation tables."""

    elevation_m: np.ndarray
    coverage_mask: np.ndarray
    depth_current_m: np.ndarray
    inundation_masks: Mapping[str, np.ndarray]
    depth_change_m: np.ndarray
    inundation_change: np.ndarray
    transform: Any
    validation: pd.DataFrame
    resolution_benchmark: pd.DataFrame
    stage_metadata: Mapping[str, Mapping[str, Any]]
    validation_status: str
    survey_years: str = "official source years; see provenance"
    validation_detail: str = ""

    def __post_init__(self) -> None:
        shape = np.asarray(self.elevation_m).shape
        if len(shape) != 2:
            raise Phase2PlotError("Phase 2 elevation must be two-dimensional")
        grids = (
            self.coverage_mask,
            self.depth_current_m,
            self.depth_change_m,
            self.inundation_change,
            *self.inundation_masks.values(),
        )
        if any(np.asarray(grid).shape != shape for grid in grids):
            raise Phase2PlotError("All Phase 2 figure grids must align")
        missing = {"latest", "recent_low", "recent_high"}.difference(
            self.inundation_masks
        )
        if missing:
            raise Phase2PlotError(f"Missing stage-envelope masks: {sorted(missing)}")
        if not isinstance(self.validation_detail, str):
            raise Phase2PlotError("Phase 2 validation detail must be text")


def _extent(
    shape: tuple[int, int], transform: Any
) -> tuple[float, float, float, float]:
    if abs(float(transform.b)) > 1e-12 or abs(float(transform.d)) > 1e-12:
        raise Phase2PlotError("Phase 2 plotting requires a north-up grid")
    left = float(transform.c)
    top = float(transform.f)
    right = left + float(transform.a) * shape[1]
    bottom = top + float(transform.e) * shape[0]
    return left, right, bottom, top


def _centers(shape: tuple[int, int], transform: Any) -> tuple[np.ndarray, np.ndarray]:
    columns = np.arange(shape[1], dtype=float) + 0.5
    rows = np.arange(shape[0], dtype=float) + 0.5
    x = float(transform.c) + float(transform.a) * columns
    y = float(transform.f) + float(transform.e) * rows
    return x, y


def _map_axes(axis: mpl.axes.Axes) -> None:
    axis.set_aspect("equal")
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")
    axis.ticklabel_format(style="plain", useOffset=False)
    axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6))


def _shoreline(
    axis: mpl.axes.Axes,
    mask: np.ndarray,
    *,
    transform: Any,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str,
) -> None:
    water = np.asarray(mask, dtype=bool)
    if not water.any() or water.all():
        return
    x, y = _centers(water.shape, transform)
    axis.contour(
        x,
        y,
        water.astype(float),
        levels=[0.5],
        colors=[color],
        linestyles=[linestyle],
        linewidths=[linewidth],
    )
    axis.plot(
        [], [], color=color, linestyle=linestyle, linewidth=linewidth, label=label
    )


def plot_topobathymetry_stage_envelope(
    inputs: Phase2FigureInputs,
) -> mpl.figure.Figure:
    """Working topobathymetry with observed low/latest/high shorelines."""

    elevation = np.asarray(inputs.elevation_m, dtype=np.float64)
    coverage = np.asarray(inputs.coverage_mask, dtype=bool)
    shown = np.ma.masked_where(~coverage | ~np.isfinite(elevation), elevation)
    colormap = plt.colormaps["terrain"].with_extremes(bad="#E6E6E6")
    figure, axis = plt.subplots(figsize=(8.8, 7.0), constrained_layout=True)
    image = axis.imshow(
        shown,
        extent=_extent(elevation.shape, inputs.transform),
        origin="upper",
        cmap=colormap,
        interpolation="nearest",
    )
    styles = (
        ("recent_low", "#D55E00", ":", 1.2, "Recent observed low"),
        ("latest", "#0072B2", "-", 1.5, "Latest"),
        ("recent_high", "#009E73", "--", 1.2, "Recent observed high"),
    )
    for state_id, color, linestyle, linewidth, label in styles:
        _shoreline(
            axis,
            inputs.inundation_masks[state_id],
            transform=inputs.transform,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    figure.colorbar(
        image, ax=axis, shrink=0.75, label="Topobathymetric elevation (m NAVD88)"
    )
    legend = axis.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(-0.04, 1.0),
    )
    legend.set_in_layout(False)
    axis.set_title("Bear Lake surveyed topobathymetry and observed stage envelope")
    figure.text(
        0.01,
        0.01,
        f"Source years: {inputs.survey_years}; unsupported cells shown gray",
        ha="left",
        va="bottom",
        fontsize=7,
    )
    _map_axes(axis)
    return figure


def plot_current_depth(inputs: Phase2FigureInputs) -> mpl.figure.Figure:
    """Latest connected depth with shoreline and coverage context."""

    depth = np.asarray(inputs.depth_current_m, dtype=np.float64)
    coverage = np.asarray(inputs.coverage_mask, dtype=bool)
    valid = coverage & np.isfinite(depth) & (depth != FLOAT_NODATA)
    shown = np.ma.masked_where(~valid | (depth <= 0), depth)
    colormap = plt.colormaps["cividis"].with_extremes(bad=(0, 0, 0, 0))
    figure, axis = plt.subplots(figsize=(8.8, 7.0), constrained_layout=True)
    axis.imshow(
        coverage.astype(np.uint8),
        extent=_extent(depth.shape, inputs.transform),
        origin="upper",
        cmap=mpl.colors.ListedColormap(["#D0D0D0", "#FAFAFA"]),
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    image = axis.imshow(
        shown,
        extent=_extent(depth.shape, inputs.transform),
        origin="upper",
        cmap=colormap,
        interpolation="nearest",
        vmin=0,
    )
    _shoreline(
        axis,
        inputs.inundation_masks["latest"],
        transform=inputs.transform,
        color="#111111",
        linestyle="-",
        linewidth=1.0,
        label="Latest connected shoreline",
    )
    latest = inputs.stage_metadata.get("latest", {})
    stage = latest.get("stage_m_navd88")
    date = latest.get("date", latest.get("observation_date", "unknown date"))
    stage_text = "unknown stage" if stage is None else f"{float(stage):.3f} m NAVD88"
    axis.set_title(f"Bear Lake current water depth - {date}")
    figure.colorbar(image, ax=axis, shrink=0.75, label="Water depth (m)")
    legend = axis.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(-0.04, 1.0),
    )
    legend.set_in_layout(False)
    figure.text(
        0.01,
        0.01,
        (
            f"Observed stage: {stage_text}; connected-inundation method; "
            f"{inputs.validation_status}"
            + (
                f"\n{inputs.validation_detail}"
                if inputs.validation_detail.strip()
                else ""
            )
        ),
        ha="left",
        va="bottom",
        fontsize=7,
    )
    _map_axes(axis)
    return figure


def plot_stage_change(inputs: Phase2FigureInputs) -> mpl.figure.Figure:
    """Latest-minus-year-ago depth and categorical inundation changes."""

    depth = np.asarray(inputs.depth_change_m, dtype=np.float64)
    categories = np.asarray(inputs.inundation_change)
    coverage = np.asarray(inputs.coverage_mask, dtype=bool)
    valid_depth = coverage & np.isfinite(depth) & (depth != FLOAT_NODATA)
    maximum = float(np.nanmax(np.abs(depth[valid_depth]))) if valid_depth.any() else 1.0
    maximum = max(maximum, 1e-9)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), constrained_layout=True)
    depth_image = axes[0].imshow(
        np.ma.masked_where(~valid_depth, depth),
        extent=_extent(depth.shape, inputs.transform),
        origin="upper",
        cmap="RdBu",
        norm=mpl.colors.TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
        interpolation="nearest",
    )
    figure.colorbar(depth_image, ax=axes[0], shrink=0.75, label="Depth change (m)")
    axes[0].set_title("Latest minus one-year-prior depth")

    change_colors = ["#E5E5E5", "#0072B2", "#009E73", "#D55E00"]
    change_map = mpl.colors.ListedColormap(change_colors)
    change_norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    supported = categories != BYTE_NODATA
    change_image = axes[1].imshow(
        np.ma.masked_where(~supported, categories),
        extent=_extent(categories.shape, inputs.transform),
        origin="upper",
        cmap=change_map,
        norm=change_norm,
        interpolation="nearest",
    )
    colorbar = figure.colorbar(
        change_image, ax=axes[1], shrink=0.75, ticks=[0, 1, 2, 3]
    )
    colorbar.ax.set_yticklabels(
        ["Unchanged dry", "Persistent water", "Newly inundated", "Exposed/lost"]
    )
    axes[1].set_title("Connected inundation transition")
    for axis in axes:
        _map_axes(axis)
    latest = inputs.stage_metadata.get("latest", {}).get("date", "latest")
    prior = inputs.stage_metadata.get("year_ago", {}).get("date", "one year prior")
    figure.suptitle(f"Bear Lake stage change: {prior} to {latest}")
    return figure


def _normalized_validation(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "area_relative_error": "relative_area_error",
        "storage_relative_error": "relative_storage_error",
        "area_m2": "raster_area_m2",
        "storage_m3": "raster_storage_m3",
        "resolution_m": "working_resolution",
    }
    result = frame.copy()
    for source, target in aliases.items():
        if target not in result and source in result:
            result[target] = result[source]
    required = {
        "stage_m_navd88",
        "official_area_m2",
        "raster_area_m2",
        "relative_area_error",
        "official_storage_m3",
        "raster_storage_m3",
        "relative_storage_error",
    }
    missing = required.difference(result.columns)
    if missing:
        raise Phase2PlotError(f"Validation table is missing columns: {sorted(missing)}")
    if "working_resolution" not in result:
        result["working_resolution"] = "source"
    return result


def plot_eav_validation(
    validation: pd.DataFrame, resolution_benchmark: pd.DataFrame
) -> mpl.figure.Figure:
    """Official and raster E-A-V curves with residuals by resolution."""

    validation_rows = _normalized_validation(validation)
    frames = [validation_rows]
    if not resolution_benchmark.empty:
        try:
            benchmark = _normalized_validation(resolution_benchmark)
        except Phase2PlotError:
            benchmark = pd.DataFrame()
        if not benchmark.empty:
            frames.append(benchmark)
    rows = pd.concat(frames, ignore_index=True).drop_duplicates()
    rows["stage_m_navd88"] = pd.to_numeric(rows["stage_m_navd88"])
    rows = rows.sort_values("stage_m_navd88", kind="stable")
    official = rows.drop_duplicates("stage_m_navd88", keep="first")
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)
    axes[0, 0].plot(
        official["stage_m_navd88"],
        official["official_area_m2"] / 1e6,
        color="#111111",
        linewidth=1.5,
        label="Official E-A-V",
    )
    axes[1, 0].plot(
        official["stage_m_navd88"],
        official["official_storage_m3"] / 1e9,
        color="#111111",
        linewidth=1.5,
        label="Official E-A-V",
    )
    markers = ("o", "s", "D", "^")
    colors = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")
    for index, (resolution, group) in enumerate(
        rows.groupby("working_resolution", sort=False, dropna=False)
    ):
        group = group.sort_values("stage_m_navd88", kind="stable")
        label = "source grid" if str(resolution) == "source" else f"{resolution} m grid"
        marker = markers[index % len(markers)]
        color = colors[index % len(colors)]
        axes[0, 0].plot(
            group["stage_m_navd88"],
            group["raster_area_m2"] / 1e6,
            color=color,
            marker=marker,
            markersize=3,
            linewidth=0.9,
            label=label,
        )
        axes[1, 0].plot(
            group["stage_m_navd88"],
            group["raster_storage_m3"] / 1e9,
            color=color,
            marker=marker,
            markersize=3,
            linewidth=0.9,
            label=label,
        )
        axes[0, 1].plot(
            group["stage_m_navd88"],
            100 * group["relative_area_error"],
            color=color,
            marker=marker,
            markersize=3,
            linewidth=0.9,
            label=label,
        )
        axes[1, 1].plot(
            group["stage_m_navd88"],
            100 * group["relative_storage_error"],
            color=color,
            marker=marker,
            markersize=3,
            linewidth=0.9,
            label=label,
        )
    axes[0, 0].set_ylabel("Area (km2)")
    axes[1, 0].set_ylabel("Storage (km3)")
    axes[0, 1].set_ylabel("Area residual (%)")
    axes[1, 1].set_ylabel("Storage residual (%)")
    axes[0, 0].set_title("Elevation-area relation")
    axes[1, 0].set_title("Elevation-storage relation")
    axes[0, 1].set_title("Signed area residual")
    axes[1, 1].set_title("Signed storage residual")
    for axis in axes.flat:
        axis.set_xlabel("Stage (m NAVD88)")
        axis.grid(color="#DDDDDD", linewidth=0.6)
    axes[0, 1].axhline(0, color="#777777", linewidth=0.8)
    axes[1, 1].axhline(0, color="#777777", linewidth=0.8)
    for axis in (axes[0, 1], axes[1, 1]):
        # Near-zero residuals must remain legible without discarding the
        # physically real, high-relative-error near-empty bottom stage.
        axis.set_yscale("symlog", linthresh=0.01, linscale=1.0)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Bear Lake official E-A-V versus raster calculation")
    return figure


def figure_notes(inputs: Phase2FigureInputs) -> str:
    """Return concise scientific notes for the four figure families."""

    return (
        "# Hydrometry Phase 2 figure notes\n\n"
        "**Topobathymetry and stage envelope.** Surveyed Bear Lake bed and land "
        "elevations are shown on the quantitative working grid with connected "
        "shorelines for the recent observed low, latest observation, and recent "
        f"observed high. Source years: {inputs.survey_years}.\n\n"
        "**Current depth.** Depth is stage minus surveyed bed elevation only inside "
        "the validated, connected inundation and supported source coverage. Dry "
        "and unsupported cells are not portrayed as water.\n\n"
        "**Change through time.** Latest-minus-year-ago depth and categorical "
        "inundation distinguish persistent water, newly inundated water, and exposed "
        "water using exact observed stages.\n\n"
        "**E-A-V validation.** Official USGS elevation-area-volume values are "
        "compared with source/working-grid raster calculations. Residual panels and "
        "resolution series provide the numerical evidence for the render gate.\n\n"
        f"Validation status: `{inputs.validation_status}`. "
        + (
            f"Source-method detail: `{inputs.validation_detail}`. "
            if inputs.validation_detail.strip()
            else ""
        )
        + "Rendering grids are "
        "separate from the quantitative grid. Survey age and unquantified raster/"
        "stage uncertainty limit interpretation.\n"
    )


def create_phase2_figures(inputs: Phase2FigureInputs, output_dir: Path) -> list[Path]:
    """Create four 300-dpi PNG/SVG families with text-free counterparts."""

    # The mandatory numerical validation figure is deliberately written first.
    plotters = (
        (
            "bear_lake_eav_validation",
            lambda: plot_eav_validation(inputs.validation, inputs.resolution_benchmark),
        ),
        (
            "bear_lake_topobathymetry_stage_envelope",
            lambda: plot_topobathymetry_stage_envelope(inputs),
        ),
        ("bear_lake_current_depth", lambda: plot_current_depth(inputs)),
        ("bear_lake_stage_change", lambda: plot_stage_change(inputs)),
    )
    written: list[Path] = []
    with mpl.rc_context(PLOT_STYLE):
        for name, plotter in plotters:
            written.extend(save_figure_with_no_text(plotter(), output_dir, name))
    notes_path = output_dir / "figure_notes.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(figure_notes(inputs), encoding="utf-8")
    written.append(notes_path)
    return written


__all__ = [
    "FIGURE_NAMES",
    "PLOT_STYLE",
    "Phase2FigureInputs",
    "Phase2PlotError",
    "create_phase2_figures",
    "figure_notes",
    "plot_current_depth",
    "plot_eav_validation",
    "plot_stage_change",
    "plot_topobathymetry_stage_envelope",
]
