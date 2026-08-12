"""Local-only summary figures for staged expanded transfer results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .expanded_transfer import expanded_output_paths
from .plot_quick_transfer import PLOT_STYLE, REFERENCE_COLORS, save_figure_with_no_text
from .quick_curve_data import (
    read_curve_cache,
    threshold_response,
    training_threshold_grid,
)
from .quick_sampling import LakeSpec, QuickSamplingSettings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data" / "regional"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "expanded_transfer_figures"
DEFAULT_PRIMARY_CACHE_DIR = (
    REPOSITORY_ROOT / "data" / "regional" / "quick_transfer_cache"
)
DEFAULT_EXPANDED_CACHE_DIR = (
    REPOSITORY_ROOT / "data" / "regional" / "expanded_transfer_cache"
)
HYDROLAKES_ASSET = "projects/sat-io/open-datasets/HydroLakes/lake_poly_v10"
REFERENCES = ("OPERA", "JRC")
GROUP_BOUNDARY = 5.5
CURVE_EXAMPLES = (("OPERA", 791), ("OPERA", 789), ("JRC", 780), ("JRC", 769))


class ExpandedPlotError(ValueError):
    """Raised when compact expanded outputs are missing or inconsistent."""


def load_expanded_plot_tables(
    data_dir: str | Path, tier: str = "tier1"
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load and cross-check compact OPERA/JRC expanded outputs."""

    result: dict[str, dict[str, pd.DataFrame]] = {}
    expected_ids: tuple[int, ...] | None = None
    for reference in REFERENCES:
        paths = expanded_output_paths(data_dir, reference, tier)
        required = {
            "local": paths.local,
            "regional": paths.regional,
            "lolo": paths.lolo,
            "macro": paths.macro,
            "status": paths.status,
        }
        missing_files = [str(path) for path in required.values() if not path.is_file()]
        if missing_files:
            raise ExpandedPlotError(
                "Missing expanded result files: " + ", ".join(missing_files)
            )
        tables = {name: pd.read_csv(path) for name, path in required.items()}
        schemas = {
            "local": {
                "Hylak_id",
                "Lake_name",
                "state",
                "Lake_area",
                "n_samples",
                "n_train",
                "n_test",
                "local_threshold",
                "local_test_iou",
            },
            "regional": {"Hylak_id", "regional_threshold", "regional_test_iou"},
            "lolo": {
                "held_out_Hylak_id",
                "held_out_lake",
                "threshold",
                "held_out_test_iou",
                "iou_delta_vs_local",
            },
            "macro": {
                "reference",
                "regional_threshold",
                "macro_regional_test_iou",
                "macro_lolo_iou",
            },
        }
        for name, columns in schemas.items():
            missing = sorted(columns.difference(tables[name].columns))
            if missing:
                raise ExpandedPlotError(
                    f"{reference} {name} is missing: {', '.join(missing)}."
                )
        local = tables["local"]
        if local["Hylak_id"].duplicated().any():
            raise ExpandedPlotError(f"{reference} local table has duplicate IDs.")
        ids = tuple(local["Hylak_id"].astype(int))
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ExpandedPlotError("OPERA and JRC expanded lake order differs.")
        lolo_ids = tuple(tables["lolo"]["held_out_Hylak_id"].astype(int))
        regional_ids = tuple(tables["regional"]["Hylak_id"].astype(int))
        if lolo_ids != ids or regional_ids != ids:
            raise ExpandedPlotError(f"{reference} table lake orders are inconsistent.")
        result[reference] = tables
    return result


def plot_local_vs_lolo(
    tables: dict[str, dict[str, pd.DataFrame]],
) -> mpl.figure.Figure:
    """Compare local and leave-one-lake-out held-out IoU."""

    lakes = tables["OPERA"]["local"]["Lake_name"].astype(str).tolist()
    y = np.arange(len(lakes))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, max(5.2, 0.42 * len(lakes) + 1.6)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    all_values = []
    for axis, reference in zip(axes, REFERENCES, strict=True):
        local = tables[reference]["local"]["local_test_iou"].to_numpy(float)
        lolo = tables[reference]["lolo"]["held_out_test_iou"].to_numpy(float)
        all_values.extend(local)
        all_values.extend(lolo)
        for position in y:
            axis.plot(
                [local[position], lolo[position]],
                [position, position],
                color="#AAAAAA",
                linewidth=0.8,
            )
        color = REFERENCE_COLORS[reference]
        axis.scatter(local, y, s=30, color=color, marker="o", label="Local")
        axis.scatter(
            lolo,
            y,
            s=34,
            facecolor="white",
            edgecolor=color,
            marker="D",
            label="LOLO",
        )
        axis.set_title(reference)
        axis.set_xlabel("Held-out IoU")
        axis.set_yticks(y, lakes)
        axis.grid(axis="x", color="#E2E2E2", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.axhline(GROUP_BOUNDARY, color="#777777", linestyle=":", linewidth=0.8)
    axes[0].invert_yaxis()
    low = max(0, min(all_values) - 0.04)
    axes[0].set_xlim(low, 1.005)
    axes[1].legend(frameon=False)
    return fig


def plot_lolo_delta(
    tables: dict[str, dict[str, pd.DataFrame]],
) -> mpl.figure.Figure:
    """Plot LOLO minus local held-out IoU for all Tier 1 lakes."""

    lakes = tables["OPERA"]["local"]["Lake_name"].astype(str).tolist()
    y = np.arange(len(lakes))
    fig, axis = plt.subplots(
        figsize=(8.5, max(5.2, 0.42 * len(lakes) + 1.4)),
        constrained_layout=True,
    )
    offsets = {"OPERA": -0.11, "JRC": 0.11}
    for reference in REFERENCES:
        values = tables[reference]["lolo"]["iou_delta_vs_local"].to_numpy(float)
        color = REFERENCE_COLORS[reference]
        positions = y + offsets[reference]
        axis.hlines(positions, 0, values, color=color, linewidth=0.85, alpha=0.55)
        axis.scatter(
            values,
            positions,
            color=color,
            marker="o" if reference == "OPERA" else "D",
            s=34,
            label=reference,
            zorder=3,
        )
    axis.axvline(0, color="#222222", linewidth=0.9)
    axis.axhline(GROUP_BOUNDARY, color="#777777", linestyle=":", linewidth=0.8)
    axis.set_yticks(y, lakes)
    axis.invert_yaxis()
    axis.set_xlabel("LOLO IoU - local IoU")
    axis.grid(axis="x", color="#E2E2E2", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    return fig


def plot_threshold_portability(
    tables: dict[str, dict[str, pd.DataFrame]],
) -> mpl.figure.Figure:
    """Show local, LOLO, and pooled-regional thresholds for every lake."""

    lakes = tables["OPERA"]["local"]["Lake_name"].astype(str).tolist()
    x = np.arange(len(lakes))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.2, 7.4),
        sharex=True,
        constrained_layout=True,
    )
    for axis, reference in zip(axes, REFERENCES, strict=True):
        color = REFERENCE_COLORS[reference]
        local = tables[reference]["local"]["local_threshold"].to_numpy(float)
        lolo = tables[reference]["lolo"]["threshold"].to_numpy(float)
        regional = float(tables[reference]["macro"]["regional_threshold"].iloc[0])
        axis.axhline(
            regional,
            color="#333333",
            linestyle="--",
            linewidth=1.0,
            label="Pooled regional",
        )
        axis.scatter(x, local, color=color, marker="o", s=34, label="Local")
        axis.scatter(
            x,
            lolo,
            edgecolor=color,
            facecolor="white",
            marker="D",
            s=36,
            label="LOLO",
        )
        axis.set_title(reference, loc="left")
        axis.set_ylabel("NDWI threshold")
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.axvline(GROUP_BOUNDARY, color="#777777", linestyle=":", linewidth=0.8)
    axes[-1].set_xticks(x, lakes, rotation=32, ha="right")
    axes[0].legend(frameon=False, ncol=3)
    return fig


def _cache_path(
    reference: str,
    lake_id: int,
    primary_cache_dir: Path,
    expanded_cache_dir: Path,
) -> Path:
    primary = primary_cache_dir / f"{reference.lower()}_{lake_id}.csv"
    expanded = expanded_cache_dir / f"{reference.lower()}_{lake_id}.csv"
    path = primary if primary.is_file() else expanded
    if not path.is_file():
        raise ExpandedPlotError(f"Missing curve cache: {primary} or {expanded}.")
    return path


def build_selected_threshold_responses(
    tables: dict[str, dict[str, pd.DataFrame]],
    *,
    primary_cache_dir: str | Path = DEFAULT_PRIMARY_CACHE_DIR,
    expanded_cache_dir: str | Path = DEFAULT_EXPANDED_CACHE_DIR,
    grid_steps: int = 150,
) -> tuple[dict[tuple[str, int], pd.DataFrame], dict[tuple[str, int], pd.DataFrame]]:
    """Load authoritative cached splits and build selected held-out curves."""

    primary_root = Path(primary_cache_dir)
    expanded_root = Path(expanded_cache_dir)
    settings = QuickSamplingSettings(grid_steps=grid_steps)
    caches: dict[tuple[str, int], pd.DataFrame] = {}
    responses: dict[tuple[str, int], pd.DataFrame] = {}
    for reference in REFERENCES:
        local = tables[reference]["local"]
        lakes_by_id = {int(row.Hylak_id): row for row in local.itertuples(index=False)}
        reference_samples: dict[int, pd.DataFrame] = {}
        for lake_id, row in lakes_by_id.items():
            lake = LakeSpec(
                hylak_id=lake_id,
                lake_name=str(row.Lake_name),
                state=str(row.state),
                lake_area=float(row.Lake_area),
                centroid_lon=0.0,
                centroid_lat=0.0,
            )
            expected = (int(row.n_samples), int(row.n_train), int(row.n_test))
            frame = read_curve_cache(
                _cache_path(reference, lake_id, primary_root, expanded_root),
                lake,
                reference,
                settings,
                HYDROLAKES_ASSET,
                expected_counts=expected,
            )
            reference_samples[lake_id] = frame
            caches[(reference, lake_id)] = frame
        grid = training_threshold_grid(reference_samples, grid_steps=grid_steps)
        for selected_reference, lake_id in CURVE_EXAMPLES:
            if selected_reference != reference:
                continue
            test = reference_samples[lake_id]
            test = test[test["split"] == "test"].copy()
            test["split"] = "test"
            responses[(reference, lake_id)] = threshold_response(test, grid)
    return responses, caches


def plot_selected_threshold_responses(
    tables: dict[str, dict[str, pd.DataFrame]],
    responses: dict[tuple[str, int], pd.DataFrame],
) -> mpl.figure.Figure:
    """Show response shape for selected strong and weak transfer examples."""

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
    threshold_styles = {
        "Local": ("#222222", ":", "o"),
        "LOLO": ("#666666", "-.", "D"),
        "Regional": ("#111111", "--", "^"),
    }
    for axis, (reference, lake_id) in zip(axes.flat, CURVE_EXAMPLES, strict=True):
        response = responses[(reference, lake_id)]
        local_row = tables[reference]["local"].set_index("Hylak_id").loc[lake_id]
        lolo_row = tables[reference]["lolo"].set_index("held_out_Hylak_id").loc[lake_id]
        regional_row = tables[reference]["regional"].set_index("Hylak_id").loc[lake_id]
        thresholds = {
            "Local": float(local_row["local_threshold"]),
            "LOLO": float(lolo_row["threshold"]),
            "Regional": float(regional_row["regional_threshold"]),
        }
        color = REFERENCE_COLORS[reference]
        axis.plot(response["threshold"], response["iou"], color=color, linewidth=1.5)
        axis.plot(
            response["threshold"],
            response["f1"],
            color=color,
            linewidth=1.25,
            linestyle="--",
        )
        for label, threshold in thresholds.items():
            line_color, linestyle, marker = threshold_styles[label]
            axis.axvline(
                threshold, color=line_color, linestyle=linestyle, linewidth=0.9
            )
            axis.scatter(
                [threshold],
                [0.97 - 0.045 * list(thresholds).index(label)],
                transform=axis.get_xaxis_transform(),
                marker=marker,
                s=22,
                facecolor="white",
                edgecolor=line_color,
                zorder=4,
            )
        axis.set_title(f"{local_row['Lake_name']} - {reference}", loc="left")
        axis.set_xlabel("NDWI threshold")
        axis.set_ylabel("Held-out performance")
        axis.set_ylim(0, 1.02)
        axis.grid(color="#E2E2E2", linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([], [], color="#444444", label="IoU"),
        Line2D([], [], color="#444444", linestyle="--", label="F1"),
        *[
            Line2D(
                [],
                [],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markerfacecolor="white",
                label=f"{label} threshold",
            )
            for label, (color, linestyle, marker) in threshold_styles.items()
        ],
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    return fig


def plot_macro_progression(
    data_dir: str | Path,
    tables: dict[str, dict[str, pd.DataFrame]],
) -> mpl.figure.Figure:
    """Compare exact original-six and cumulative Tier 1 macro LOLO IoU."""

    fig, axis = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    x = np.array([0.0, 1.0])
    for reference in REFERENCES:
        six_path = (
            Path(data_dir) / f"western_quick_transfer_{reference.lower()}_lolo.csv"
        )
        if not six_path.is_file():
            raise ExpandedPlotError(f"Missing original-six LOLO table: {six_path}")
        six = pd.read_csv(six_path)
        if "macro_lolo_iou" not in six.columns:
            raise ExpandedPlotError(f"{six_path} lacks macro_lolo_iou.")
        values = [
            float(six["macro_lolo_iou"].iloc[0]),
            float(tables[reference]["macro"]["macro_lolo_iou"].iloc[0]),
        ]
        axis.plot(
            x,
            values,
            color=REFERENCE_COLORS[reference],
            marker="o" if reference == "OPERA" else "D",
            linewidth=1.5,
            markersize=6,
            label=reference,
        )
    axis.set_xticks(x, ["Original six", "Tier 1 (12)"])
    axis.set_ylabel("Macro LOLO IoU")
    axis.set_xlim(-0.15, 1.15)
    axis.set_ylim(0.75, 0.96)
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    return fig


def create_expanded_figures(
    tables: dict[str, dict[str, pd.DataFrame]],
    output_dir: str | Path,
    tier: str,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    primary_cache_dir: str | Path = DEFAULT_PRIMARY_CACHE_DIR,
    expanded_cache_dir: str | Path = DEFAULT_EXPANDED_CACHE_DIR,
) -> tuple[Path, ...]:
    """Save the five key Tier 1 figure pairs plus no-text twins and notes."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    responses, _caches = build_selected_threshold_responses(
        tables,
        primary_cache_dir=primary_cache_dir,
        expanded_cache_dir=expanded_cache_dir,
    )
    with mpl.rc_context(PLOT_STYLE):
        for name, figure in (
            ("expanded_local_vs_lolo_iou", plot_local_vs_lolo(tables)),
            ("expanded_lolo_iou_delta", plot_lolo_delta(tables)),
            ("expanded_threshold_portability", plot_threshold_portability(tables)),
            (
                "expanded_selected_threshold_response",
                plot_selected_threshold_responses(tables, responses),
            ),
            (
                "expanded_macro_progression",
                plot_macro_progression(data_dir, tables),
            ),
        ):
            saved.extend(save_figure_with_no_text(figure, output, name))
    notes = output / "figure_notes.md"
    lake_count = len(tables["OPERA"]["local"])
    notes.write_text(
        "# Expanded transfer figure notes\n\n"
        f"These figures summarize the exploratory {tier} analysis across "
        f"{lake_count} usable lakes. Local thresholds use each lake's training "
        "blocks and grouped CV; LOLO thresholds exclude the target lake entirely; "
        "pooled thresholds maximize equal-lake-weighted training IoU. OPERA and "
        "JRC retain their distinct configured temporal windows. This staged set "
        "does not constitute definitive western-U.S. validation. The dotted "
        "group boundary in lake-wise figures separates the retained original six "
        "from the six Tier 1 additions; it is a visual aid, not a statistical "
        "stratum. Selected threshold-response panels use only authoritative cached "
        "test splits: Pyramid/OPERA (strong), Honey/OPERA (largest OPERA loss), "
        "Goose/JRC (largest JRC loss), and Malheur/JRC (low absolute separation). "
        "The macro-progression figure reads the completed original-six and Tier 1 "
        "LOLO summaries directly.\n",
        encoding="utf-8",
    )
    saved.append(notes)
    return tuple(saved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot compact expanded OPERA/JRC transfer summaries locally."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--primary-cache-dir", type=Path, default=DEFAULT_PRIMARY_CACHE_DIR
    )
    parser.add_argument(
        "--expanded-cache-dir", type=Path, default=DEFAULT_EXPANDED_CACHE_DIR
    )
    parser.add_argument("--tier", choices=("tier1", "tier2"), default="tier1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        tables = load_expanded_plot_tables(args.data_dir, args.tier)
        saved = create_expanded_figures(
            tables,
            args.output_dir,
            args.tier,
            data_dir=args.data_dir,
            primary_cache_dir=args.primary_cache_dir,
            expanded_cache_dir=args.expanded_cache_dir,
        )
        print(f"Wrote {len(saved)} expanded figure/note files under {args.output_dir}")
        return 0
    except (ExpandedPlotError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
