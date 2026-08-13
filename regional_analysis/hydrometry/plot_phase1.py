"""Matplotlib-only figures for Hydrometry Phase 1 manager products."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    "svg.hashsalt": "hydrometry-phase1",
}
COLORS = {
    "elevation": "#0072B2",
    "reported": "#D55E00",
    "derived": "#009E73",
    "neutral": "#6B6B6B",
    "unavailable": "#B8B8B8",
}
FIGURE_NAMES = (
    "pilot_hydrometry_timeseries",
    "bear_lake_eav_relation",
    "storage_change_summary",
    "seasonal_context",
)


class Phase1PlotError(RuntimeError):
    """Raised when Phase 1 plot inputs are incomplete or inconsistent."""


def _remove_text(figure: mpl.figure.Figure) -> None:
    """Hide every textual layer while retaining plotted geometry."""

    figure.suptitle("")
    for artist in tuple(figure.texts):
        artist.set_visible(False)
    for axis in figure.axes:
        axis.set_title("")
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(
            labelbottom=False, labeltop=False, labelleft=False, labelright=False
        )
        axis.xaxis.offsetText.set_visible(False)
        axis.yaxis.offsetText.set_visible(False)
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
        for artist in tuple(axis.texts):
            artist.set_visible(False)


def save_figure_with_no_text(
    figure: mpl.figure.Figure, output_dir: Path, basename: str
) -> list[Path]:
    """Save labeled and text-free PNG/SVG pairs with identical geometry."""

    output_dir.mkdir(parents=True, exist_ok=True)
    no_text_dir = output_dir / "no_text"
    no_text_dir.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()
    if figure.get_layout_engine() is not None:
        figure.set_layout_engine("none")
    paths = [
        output_dir / f"{basename}.png",
        output_dir / f"{basename}.svg",
        no_text_dir / f"{basename}.png",
        no_text_dir / f"{basename}.svg",
    ]
    figure.savefig(paths[0], dpi=300)
    figure.savefig(paths[1], metadata={"Date": None})
    _remove_text(figure)
    figure.savefig(paths[2], dpi=300)
    figure.savefig(paths[3], metadata={"Date": None})
    plt.close(figure)
    return paths


def _available(frame: pd.DataFrame, subprofile: str, metric: str) -> pd.DataFrame:
    selected = frame.loc[
        frame["subprofile_id"].eq(subprofile)
        & frame["metric"].eq(metric)
        & frame["availability_status"].eq("available")
        & frame["value"].notna()
    ].copy()
    selected["calendar_date"] = pd.to_datetime(selected["calendar_date"])
    return selected.sort_values("calendar_date", kind="stable")


def _format_dates(axis: mpl.axes.Axes) -> None:
    axis.xaxis.set_major_locator(mdates.YearLocator(2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)


def plot_pilot_timeseries(daily: pd.DataFrame) -> mpl.figure.Figure:
    """Six panels separating incompatible elevation and storage units."""

    panels = (
        (
            "great_salt_south_arm",
            "water_elevation",
            "Great Salt south arm",
            "Elevation (ft)",
        ),
        (
            "great_salt_north_arm",
            "water_elevation",
            "Great Salt north arm",
            "Elevation (ft)",
        ),
        ("lake_abert", "water_elevation", "Lake Abert", "Elevation (ft)"),
        ("bear_lake", "water_elevation", "Bear Lake", "Elevation (ft)"),
        ("lake_powell", "total_storage", "Lake Powell", "Storage (acre-ft)"),
        ("flaming_gorge", "total_storage", "Flaming Gorge", "Storage (acre-ft)"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(10.2, 8.2), constrained_layout=True)
    for axis, (subprofile, metric, title, ylabel) in zip(
        axes.flat, panels, strict=True
    ):
        values = _available(daily, subprofile, metric)
        if values.empty:
            raise Phase1PlotError(f"No {metric} rows for {subprofile}")
        method = str(values["method"].iloc[0])
        color = (
            COLORS["derived"]
            if "official_eav" in method
            else (
                COLORS["reported"]
                if method == "agency_reported"
                else COLORS["elevation"]
            )
        )
        linestyle = "--" if "official_eav" in method else "-"
        axis.plot(
            values["calendar_date"],
            values["value"],
            color=color,
            linewidth=1.1,
            linestyle=linestyle,
            label=method.replace("_", " "),
        )
        if subprofile == "bear_lake":
            storage = _available(daily, subprofile, "total_storage")
            if storage.empty:
                raise Phase1PlotError("Bear Lake derived storage is missing")
            twin = axis.twinx()
            twin.plot(
                storage["calendar_date"],
                storage["value"],
                color=COLORS["derived"],
                linewidth=1.0,
                linestyle="--",
                label="official-table-derived storage",
            )
            twin.set_ylabel("Derived storage (acre-ft)", color=COLORS["derived"])
            twin.tick_params(axis="y", colors=COLORS["derived"])
            handles, labels = axis.get_legend_handles_labels()
            other_handles, other_labels = twin.get_legend_handles_labels()
            axis.legend(
                handles + other_handles,
                labels + other_labels,
                frameon=False,
                loc="best",
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False, loc="best") if subprofile != "bear_lake" else None
        _format_dates(axis)
    axes[-1, 0].set_xlabel("Calendar year")
    axes[-1, 1].set_xlabel("Calendar year")
    return figure


def plot_bear_eav_relation(bundle: Any) -> mpl.figure.Figure:
    """Official table relations with observed stage range overlaid."""

    table = pd.DataFrame(bundle.capacity_table.normalized_records())
    stage = _available(bundle.daily, "bear_lake", "water_elevation")
    if stage.empty:
        raise Phase1PlotError("Bear Lake observed elevation rows are missing")
    # Stage is retained in ft NAVD88; convert only units for placement on the
    # table's metre axis. This is not a datum transformation.
    stage_m = stage["value"].astype(float) * 0.3048
    stage_min, stage_max = float(stage_m.min()), float(stage_m.max())
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), constrained_layout=True)
    relationships = (
        ("area_km2", "Surface area (km²)", COLORS["elevation"]),
        ("storage_acre_ft", "Storage (acre-ft)", COLORS["derived"]),
    )
    for axis, (column, ylabel, color) in zip(axes, relationships, strict=True):
        axis.plot(
            table["elevation_m_navd88"],
            table[column],
            color=color,
            linewidth=1.3,
            label="official E-A-V relation",
        )
        axis.axvspan(
            stage_min,
            stage_max,
            color="#E69F00",
            alpha=0.18,
            label="observed 2016–2026 elevation range",
        )
        axis.axvline(
            table["elevation_m_navd88"].min(),
            color=COLORS["neutral"],
            linestyle=":",
            linewidth=0.8,
        )
        axis.axvline(
            table["elevation_m_navd88"].max(),
            color=COLORS["neutral"],
            linestyle=":",
            linewidth=0.8,
        )
        axis.set_xlabel("Elevation (m NAVD88)")
        axis.set_ylabel(ylabel)
        axis.grid(color="#DDDDDD", linewidth=0.6)
        axis.legend(frameon=False)
    axes[0].set_title("Bear Lake elevation–area")
    axes[1].set_title("Bear Lake elevation–storage")
    return figure


def plot_storage_change_summary(changes: pd.DataFrame) -> mpl.figure.Figure:
    """Percent and absolute changes for the three supported storage products."""

    selected = changes.loc[
        changes["metric"].eq("total_storage")
        & changes["subprofile_id"].isin(("bear_lake", "lake_powell", "flaming_gorge"))
    ].copy()
    if selected.empty:
        raise Phase1PlotError("Storage change rows are missing")
    names = {
        "bear_lake": "Bear",
        "lake_powell": "Powell",
        "flaming_gorge": "Flaming Gorge",
    }
    windows = sorted(selected["requested_window_days"].astype(int).unique())
    x = np.arange(len(windows), dtype=float)
    offsets = (-0.18, 0.0, 0.18)
    markers = ("D", "o", "s")
    colors = (COLORS["derived"], COLORS["reported"], "#CC79A7")
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), constrained_layout=True)
    for subprofile, offset, marker, color in zip(
        names, offsets, markers, colors, strict=True
    ):
        rows = selected.loc[selected["subprofile_id"].eq(subprofile)].set_index(
            "requested_window_days"
        )
        percent = [
            rows.loc[window, "percent_change"] if window in rows.index else np.nan
            for window in windows
        ]
        absolute = [
            rows.loc[window, "absolute_change"] if window in rows.index else np.nan
            for window in windows
        ]
        for axis, values in zip(axes, (percent, absolute), strict=True):
            axis.plot(
                x + offset,
                values,
                color=color,
                marker=marker,
                linestyle="--" if subprofile == "bear_lake" else "-",
                linewidth=0.9,
                markersize=5,
                markerfacecolor="white" if subprofile == "bear_lake" else color,
                label=names[subprofile],
            )
    for axis in axes:
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set_xticks(x, [f"{window} d" for window in windows])
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("Storage change (%)")
    axes[0].set_title("Scale-relative change")
    axes[1].set_ylabel("Storage change (acre-ft)")
    axes[1].set_title("Absolute change")
    axes[0].legend(frameon=False)
    return figure


def plot_seasonal_context(seasonal: pd.DataFrame) -> mpl.figure.Figure:
    """Latest empirical seasonal percentiles, retaining unavailable series."""

    rows = seasonal.copy()
    order = (
        "great_salt_south_arm",
        "great_salt_north_arm",
        "lake_abert",
        "bear_lake",
        "lake_powell",
        "flaming_gorge",
    )
    metric_order = {"water_elevation": 0, "surface_area": 1, "total_storage": 2}
    rows["_sub_order"] = rows["subprofile_id"].map(
        {value: index for index, value in enumerate(order)}
    )
    rows["_metric_order"] = rows["metric"].map(metric_order)
    rows = rows.sort_values(["_sub_order", "_metric_order"], kind="stable")
    labels = [
        f"{row.display_name} — {str(row.metric).replace('_', ' ')}"
        for row in rows.itertuples(index=False)
    ]
    y = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    axis.axvspan(25, 75, color="#E5E5E5", alpha=0.7, label="historical middle 50%")
    available = rows["seasonal_status"].eq("available")
    derived = rows["method"].eq("official_eav_piecewise_linear")
    reported = rows["method"].eq("agency_reported")
    groups = (
        (available & ~derived & ~reported, COLORS["elevation"], "o", "observed"),
        (available & derived, COLORS["derived"], "D", "derived"),
        (available & reported, COLORS["reported"], "s", "reported"),
    )
    for mask, color, marker, label in groups:
        axis.scatter(
            rows.loc[mask, "seasonal_percentile"],
            y[mask.to_numpy()],
            color=color,
            marker=marker,
            s=34,
            label=label,
            zorder=3,
        )
    unavailable = ~available
    axis.scatter(
        np.full(unavailable.sum(), -3.0),
        y[unavailable.to_numpy()],
        color=COLORS["unavailable"],
        marker="x",
        s=35,
        label="insufficient/unavailable",
    )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(-7, 102)
    axis.set_xlabel("Latest empirical seasonal percentile")
    axis.set_ylabel("Pilot series")
    axis.set_title("Seasonal context (prior years, ±15 calendar days)")
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, ncol=2, loc="lower center")
    return figure


def figure_notes(bundle: Any) -> str:
    start = bundle.manager_summary["analysis_start"]
    end = bundle.manager_summary["analysis_end"]
    return (
        "\n\n".join(
            (
                "# Hydrometry Phase 1 figure notes",
                (
                    "**Pilot hydrometry time series.** Daily values from "
                    f"{start} through {end}. Great Salt Lake arms, Lake Abert, and Bear "
                    "Lake elevation are observed USGS series. Lake Powell and Flaming "
                    "Gorge storage are agency-reported RISE products; Bear Lake storage "
                    "is derived from observed NAVD88 elevation and the official USGS E-A-V table."
                ),
                (
                    "**Bear Lake E-A-V relation.** The official USGS NAVD88 table is "
                    "shown with the observed Phase 1 elevation range. Piecewise-linear "
                    "interpolation is bounded by the official nodes and never extrapolates."
                ),
                (
                    "**Storage change summary.** Non-interpolated 7-, 30-, 90-, and "
                    "365-day comparisons use the nearest baseline within three days. "
                    "Bear is Class B/derived; Powell and Flaming Gorge are Class A/reported."
                ),
                (
                    "**Seasonal context.** Latest values are ranked against prior-year "
                    "observations within ±15 calendar days. At least five distinct years "
                    "and 20 observations are required; insufficient series remain visible."
                ),
                (
                    "No numeric uncertainty interval is asserted because the source stage "
                    "series does not provide a compatible quantitative uncertainty. Great "
                    "Salt Lake and Lake Abert have no Phase 1 storage series."
                ),
            )
        )
        + "\n"
    )


def create_phase1_figures(bundle: Any, output_dir: Path) -> list[Path]:
    """Create four publication-usable figure families and notes."""

    plotters = (
        ("pilot_hydrometry_timeseries", lambda: plot_pilot_timeseries(bundle.daily)),
        ("bear_lake_eav_relation", lambda: plot_bear_eav_relation(bundle)),
        (
            "storage_change_summary",
            lambda: plot_storage_change_summary(bundle.change_metrics),
        ),
        ("seasonal_context", lambda: plot_seasonal_context(bundle.seasonal_context)),
    )
    written: list[Path] = []
    with mpl.rc_context(PLOT_STYLE):
        for name, plotter in plotters:
            written.extend(save_figure_with_no_text(plotter(), output_dir, name))
    notes_path = output_dir / "figure_notes.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(figure_notes(bundle), encoding="utf-8")
    written.append(notes_path)
    return written


def load_phase1_outputs(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the four figure-facing tables with clear schema failures."""

    paths = {
        "daily": data_dir / "pilot_daily_hydrometry.csv",
        "current": data_dir / "pilot_current_conditions.csv",
        "changes": data_dir / "pilot_change_metrics.csv",
        "seasonal": data_dir / "pilot_seasonal_context.csv",
        "eav": data_dir / "bear_lake_eav_normalized.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise Phase1PlotError("Missing Phase 1 table(s): " + ", ".join(missing))
    return {name: pd.read_csv(path) for name, path in paths.items()}


__all__ = [
    "FIGURE_NAMES",
    "Phase1PlotError",
    "create_phase1_figures",
    "figure_notes",
    "load_phase1_outputs",
    "plot_bear_eav_relation",
    "plot_pilot_timeseries",
    "plot_seasonal_context",
    "plot_storage_change_summary",
    "save_figure_with_no_text",
]
