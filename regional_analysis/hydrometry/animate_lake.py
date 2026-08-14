"""Deterministic observed-stage selection and Bear Lake GIF rendering."""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .render_3d import (
    RENDER_STYLE,
    Phase2RenderError,
    RenderConfig,
    RenderState,
    plot_stage_3d,
    require_render_gate,
)

ANIMATION_FRAME_COLUMNS = (
    "frame_number",
    "target_date",
    "observation_date",
    "stage_m_navd88",
    "calculation_stage_m_navd88",
    "stage_ft_navd88",
    "official_area_m2",
    "official_storage_m3",
    "source_resolution_raster_area_m2",
    "source_resolution_raster_storage_m3",
    "working_grid_area_m2",
    "working_grid_storage_m3",
    "quality_status",
    "approval_status",
    "provisional",
    "validation_status",
)


class AnimationError(Phase2RenderError):
    """Raised when the observed-stage animation contract is violated."""


def select_observed_frames(
    observations: pd.DataFrame,
    *,
    frame_count: int,
    date_column: str = "calendar_date",
    stage_column: str = "value",
) -> pd.DataFrame:
    """Select exact observations deterministically, including min/max/latest.

    Evenly spaced target dates guide the selection.  Every selected stage is a
    real observation; no stage or date interpolation is performed.  Duplicate
    dates resolve to the final stable input row.
    """

    if isinstance(frame_count, bool) or not 3 <= frame_count <= 24:
        raise AnimationError("frame_count must be between 3 and 24")
    missing = [
        column for column in (date_column, stage_column) if column not in observations
    ]
    if missing:
        raise AnimationError(f"Observation table is missing columns: {missing}")
    frame = observations.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame[stage_column] = pd.to_numeric(frame[stage_column], errors="coerce")
    frame = frame.dropna(subset=[date_column, stage_column])
    frame = frame.sort_values(date_column, kind="stable").drop_duplicates(
        date_column, keep="last"
    )
    frame = frame.reset_index(drop=True)
    if len(frame) < frame_count:
        raise AnimationError(
            f"Need at least {frame_count} distinct valid observation dates"
        )

    minimum = int(frame[stage_column].idxmin())
    maximum = int(frame[stage_column].idxmax())
    latest = len(frame) - 1
    selected: set[int] = {minimum, maximum, latest}
    targets_by_index: dict[int, pd.Timestamp] = {
        minimum: frame.loc[minimum, date_column],
        maximum: frame.loc[maximum, date_column],
        latest: frame.loc[latest, date_column],
    }

    targets = pd.date_range(
        frame[date_column].iloc[0], frame[date_column].iloc[-1], periods=frame_count
    )
    for target in targets:
        if len(selected) >= frame_count:
            break
        available = frame.index[~frame.index.isin(selected)]
        if available.empty:
            break
        differences = (frame.loc[available, date_column] - target).abs()
        closest = int(
            pd.DataFrame(
                {
                    "index": available,
                    "difference": differences.to_numpy(),
                    "date": frame.loc[available, date_column].to_numpy(),
                }
            )
            .sort_values(["difference", "date", "index"], kind="stable")
            .iloc[0]["index"]
        )
        selected.add(closest)
        targets_by_index[closest] = target

    # Target collisions are filled by a deterministic maximin time-spacing step.
    while len(selected) < frame_count:
        available = [index for index in frame.index if index not in selected]
        selected_dates = frame.loc[sorted(selected), date_column].to_numpy(
            dtype="datetime64[ns]"
        )
        scored: list[tuple[int, int]] = []
        for index in available:
            candidate = np.datetime64(frame.loc[index, date_column])
            distance = int(
                np.min(np.abs(selected_dates - candidate))
                .astype("timedelta64[ns]")
                .astype(np.int64)
            )
            scored.append((distance, index))
        _, chosen = max(scored, key=lambda item: (item[0], -item[1]))
        selected.add(chosen)
        targets_by_index[chosen] = frame.loc[chosen, date_column]

    indices = sorted(selected, key=lambda index: (frame.loc[index, date_column], index))
    result = frame.loc[indices].copy().reset_index(drop=True)
    result.insert(0, "frame_number", np.arange(1, len(result) + 1, dtype=int))
    result.insert(
        1,
        "target_date",
        [targets_by_index[index].date().isoformat() for index in indices],
    )
    result.insert(
        2,
        "observation_date",
        [frame.loc[index, date_column].date().isoformat() for index in indices],
    )
    if stage_column != "stage_m_navd88":
        result = result.rename(columns={stage_column: "stage_m_navd88"})
    return result


def normalize_animation_frame_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the stable animation manifest schema with explicit missing values."""

    result = frame.copy()
    aliases = {
        "stage_original_ft": "stage_ft_navd88",
        "official_table_area_m2": "official_area_m2",
        "official_table_storage_m3": "official_storage_m3",
        "source_quality": "quality_status",
    }
    for source, target in aliases.items():
        if target not in result and source in result:
            result[target] = result[source]
    if "calculation_stage_m_navd88" not in result and "stage_m_navd88" in result:
        result["calculation_stage_m_navd88"] = result["stage_m_navd88"]
    defaults: dict[str, Any] = {
        "frame_number": np.arange(1, len(result) + 1, dtype=int),
        "target_date": None,
        "observation_date": None,
        "stage_m_navd88": np.nan,
        "calculation_stage_m_navd88": np.nan,
        "stage_ft_navd88": np.nan,
        "official_area_m2": np.nan,
        "official_storage_m3": np.nan,
        "source_resolution_raster_area_m2": np.nan,
        "source_resolution_raster_storage_m3": np.nan,
        "working_grid_area_m2": np.nan,
        "working_grid_storage_m3": np.nan,
        "quality_status": "unknown",
        "approval_status": "unknown",
        "provisional": None,
        "validation_status": "not_evaluated",
    }
    for column, default in defaults.items():
        if column not in result:
            result[column] = default
    result = result.loc[:, ANIMATION_FRAME_COLUMNS].copy()
    if result["frame_number"].duplicated().any():
        raise AnimationError("Animation frame numbers must be unique")
    result = result.sort_values("frame_number", kind="stable").reset_index(drop=True)
    dates = pd.to_datetime(result["observation_date"], errors="coerce")
    if dates.isna().any() or not dates.is_monotonic_increasing:
        raise AnimationError("Animation observations must have chronological dates")
    if result["stage_m_navd88"].isna().any():
        raise AnimationError("Every animation frame requires a real observed stage")
    calculation_stages = pd.to_numeric(
        result["calculation_stage_m_navd88"], errors="coerce"
    )
    if calculation_stages.isna().any():
        raise AnimationError("Every animation frame requires a calculation stage")
    result["calculation_stage_m_navd88"] = calculation_stages
    return result


def write_animation_frame_table(frame: pd.DataFrame, path: Path) -> Path:
    """Atomically write the normalized frame manifest."""

    normalized = normalize_animation_frame_table(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".csv.part", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            normalized.to_csv(stream, index=False, lineterminator="\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def _common_render_config(
    elevation: np.ndarray, states: list[RenderState], config: RenderConfig
) -> RenderConfig:
    finite = elevation[np.isfinite(elevation)]
    if finite.size == 0:
        raise AnimationError("Animation bed grid has no finite elevations")
    depths = [
        state.water_surface_stage_m_navd88
        - elevation[np.asarray(state.inundation, dtype=bool)]
        for state in states
    ]
    if any(values.size == 0 for values in depths):
        raise AnimationError("Every animation frame requires connected water pixels")
    maximum_depth = max(float(values.max()) for values in depths)
    return replace(
        config,
        bed_color_limits_m=config.bed_color_limits_m
        or (float(finite.min()), float(finite.max())),
        depth_color_limits_m=config.depth_color_limits_m
        or (0.0, max(maximum_depth, 1e-9)),
    )


def render_gif_frames(
    elevation_m: np.ndarray,
    *,
    transform: Any,
    states: list[RenderState],
    config: RenderConfig,
) -> list[Image.Image]:
    """Render fixed-size RGBA frames with identical camera and color limits."""

    elevation = np.asarray(elevation_m, dtype=np.float64)
    shared = _common_render_config(elevation, states, config)
    images: list[Image.Image] = []
    with mpl.rc_context(RENDER_STYLE):
        for frame_number, state in enumerate(states, start=1):
            figure = plot_stage_3d(
                elevation,
                transform=transform,
                state=state,
                config=shared,
                common_bed_limits_m=shared.bed_color_limits_m,
                common_depth_limits_m=shared.depth_color_limits_m,
            )
            figure.text(
                0.98,
                0.98,
                f"Frame {frame_number}/{len(states)}",
                ha="right",
                va="top",
                fontsize=7,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8,
                    "pad": 1.5,
                },
            )
            figure.canvas.draw()
            pixels = np.asarray(figure.canvas.buffer_rgba()).copy()
            images.append(Image.fromarray(pixels, mode="RGBA"))
            plt.close(figure)
    dimensions = {image.size for image in images}
    if len(dimensions) != 1:
        raise AnimationError("Rendered animation frames have inconsistent dimensions")
    return images


def create_stage_change_gif(
    elevation_m: np.ndarray,
    *,
    transform: Any,
    states: list[RenderState],
    output_path: Path,
    gate: Any,
    config: RenderConfig | None = None,
    duration_ms: int = 650,
    loop: int = 0,
) -> Path:
    """Write a deterministic fixed-camera GIF from observed connected states."""

    require_render_gate(gate)
    config = config or RenderConfig(dpi=120)
    if not states:
        raise AnimationError("At least one animation state is required")
    if duration_ms <= 0 or loop < 0:
        raise AnimationError("GIF duration must be positive and loop nonnegative")
    dates = pd.to_datetime(
        [state.observation_date for state in states], errors="coerce"
    )
    if dates.isna().any() or not dates.is_monotonic_increasing:
        raise AnimationError("Animation states must be ordered chronologically")
    if any(
        state.validation_status.strip().lower()
        not in {"validated", "numerically_consistent"}
        for state in states
    ):
        raise AnimationError("Every animation state must pass quantitative validation")
    images = render_gif_frames(
        elevation_m, transform=transform, states=states, config=config
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".gif.part", dir=output_path.parent
    )
    os.close(handle)
    try:
        images[0].save(
            temporary_name,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=loop,
            disposal=2,
            optimize=False,
        )
        with Image.open(temporary_name) as check:
            if check.n_frames != len(states):
                raise AnimationError(
                    f"GIF retained {check.n_frames} frames; expected {len(states)}"
                )
            if check.info.get("duration") != duration_ms:
                raise AnimationError("GIF frame duration changed during encoding")
        os.replace(temporary_name, output_path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    finally:
        for image in images:
            image.close()
    return output_path


__all__ = [
    "ANIMATION_FRAME_COLUMNS",
    "AnimationError",
    "create_stage_change_gif",
    "normalize_animation_frame_table",
    "render_gif_frames",
    "select_observed_frames",
    "write_animation_frame_table",
]
