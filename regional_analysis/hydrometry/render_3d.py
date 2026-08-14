"""Validation-gated, fixed-camera Phase 2 Bear Lake 3D rendering.

The rendering grid is explicitly decimated and is never used to calculate
authoritative area or storage.  Quantitative rasters may be guarded by hashes
so a render-only run cannot silently alter them.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .spatial_products import QUANTITATIVE_GATE_STATUSES

T = TypeVar("T")


class Phase2RenderError(RuntimeError):
    """Raised when the numerical gate or render contract is violated."""


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """A reusable camera shared by static figures and animation frames."""

    elevation_degrees: float = 34.0
    azimuth_degrees: float = -128.0
    roll_degrees: float = 0.0
    projection: str = "ortho"

    def __post_init__(self) -> None:
        if self.projection not in {"ortho", "persp"}:
            raise Phase2RenderError("Camera projection must be 'ortho' or 'persp'")


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """Rendering-only settings, independent of quantitative calculations."""

    camera: CameraConfig = CameraConfig()
    vertical_exaggeration: float = 10.0
    stride: int = 1
    figure_width_inches: float = 8.0
    figure_height_inches: float = 6.0
    dpi: int = 180
    bed_color_limits_m: tuple[float, float] | None = None
    depth_color_limits_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.vertical_exaggeration <= 0:
            raise Phase2RenderError("vertical_exaggeration must be positive")
        if isinstance(self.stride, bool) or self.stride < 1:
            raise Phase2RenderError("rendering stride must be a positive integer")
        if self.figure_width_inches <= 0 or self.figure_height_inches <= 0:
            raise Phase2RenderError("figure dimensions must be positive")
        if self.dpi <= 0:
            raise Phase2RenderError("dpi must be positive")


@dataclass(frozen=True, slots=True)
class RenderState:
    """State metadata and connected inundation used by one rendered frame."""

    state_id: str
    observation_date: str
    stage_m_navd88: float
    stage_ft_navd88: float
    inundation: np.ndarray
    raster_area_m2: float | None = None
    raster_storage_m3: float | None = None
    validation_status: str = "numerically_consistent"
    source: str = "USGS observed stage + USGS surveyed topobathymetry"
    survey_years: str = "official source years; see provenance"
    method: str = "connected_seed"
    quality_status: str = "unknown"
    approval_status: str = "unknown"
    provenance_label: str = "bear_lake_phase2_provenance.json"
    calculation_stage_m_navd88: float | None = None
    validation_detail: str = ""

    def __post_init__(self) -> None:
        if not self.state_id.strip() or not self.observation_date.strip():
            raise Phase2RenderError("Render state ID and date must be non-empty")
        if not np.isfinite(self.stage_m_navd88):
            raise Phase2RenderError("Render stage must be finite")
        if self.calculation_stage_m_navd88 is not None and not np.isfinite(
            self.calculation_stage_m_navd88
        ):
            raise Phase2RenderError("Render calculation stage must be finite")
        if np.asarray(self.inundation).ndim != 2:
            raise Phase2RenderError("Render inundation must be two-dimensional")
        if not isinstance(self.validation_detail, str):
            raise Phase2RenderError("Render validation detail must be text")

    @property
    def water_surface_stage_m_navd88(self) -> float:
        """Return the numerical plane used for depths and water geometry."""

        if self.calculation_stage_m_navd88 is None:
            return float(self.stage_m_navd88)
        return float(self.calculation_stage_m_navd88)


@dataclass(frozen=True, slots=True)
class InteractiveRenderResult:
    """Outcome of the optional standalone Plotly HTML path."""

    created: bool
    path: Path | None
    file_size_bytes: int
    reason: str | None = None


RENDER_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "svg.fonttype": "none",
    "svg.hashsalt": "hydrometry-phase2-3d",
}


def _status_from_gate(gate: Any) -> str:
    if isinstance(gate, str):
        return gate.strip().lower()
    if isinstance(gate, Mapping):
        return (
            str(gate.get("status", gate.get("validation_status", ""))).strip().lower()
        )
    return (
        str(getattr(gate, "status", getattr(gate, "validation_status", "")))
        .strip()
        .lower()
    )


def require_render_gate(gate: Any, *, validation_figure: Path | None = None) -> str:
    """Require a passing numerical policy and, if supplied, validation figure."""

    status = _status_from_gate(gate)
    explicitly_allowed = getattr(gate, "allowed", None)
    if explicitly_allowed is None:
        explicitly_allowed = getattr(gate, "render_allowed", None)
    if status not in QUANTITATIVE_GATE_STATUSES or explicitly_allowed is False:
        raise Phase2RenderError(
            "Polished rendering is blocked because the raster model did not pass "
            "the numerical validation gate."
        )
    if validation_figure is not None and (
        not validation_figure.is_file() or validation_figure.stat().st_size == 0
    ):
        raise Phase2RenderError(
            "The mandatory E-A-V validation figure must exist before 3D rendering."
        )
    return status


def _file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_hashes(
    expected_hashes: Mapping[str | Path, str], *, base_dir: Path | None = None
) -> dict[Path, str]:
    """Verify quantitative inputs before a render-only operation."""

    verified: dict[Path, str] = {}
    for raw_path, expected in expected_hashes.items():
        path = Path(raw_path)
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        if not path.is_file():
            raise Phase2RenderError(f"Quantitative render input is missing: {path}")
        actual = _file_hash(path)
        if actual.lower() != str(expected).strip().lower():
            raise Phase2RenderError(f"Quantitative input hash mismatch: {path}")
        verified[path] = actual
    return verified


def run_with_quantitative_hash_guard(
    expected_hashes: Mapping[str | Path, str],
    action: Callable[[], T],
    *,
    base_dir: Path | None = None,
) -> T:
    """Run rendering and prove that all quantitative inputs stayed unchanged."""

    before = verify_file_hashes(expected_hashes, base_dir=base_dir)
    result = action()
    after = verify_file_hashes(expected_hashes, base_dir=base_dir)
    if before != after:  # Defensive; verification already checks expected values.
        raise Phase2RenderError("Rendering altered a quantitative input")
    return result


def _xy_grid(
    shape: tuple[int, int], transform: Any, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    if abs(float(transform.b)) > 1e-12 or abs(float(transform.d)) > 1e-12:
        raise Phase2RenderError("3D rendering currently requires a north-up grid")
    rows = np.arange(0, shape[0], stride, dtype=float) + 0.5
    columns = np.arange(0, shape[1], stride, dtype=float) + 0.5
    x = float(transform.c) + float(transform.a) * columns
    y = float(transform.f) + float(transform.e) * rows
    # Plot in local kilometres to avoid unreadable UTM-scale tick labels.
    x = (x - x.min()) / 1000.0
    y = (y - y.min()) / 1000.0
    return np.meshgrid(x, y)


def _limits(
    elevation: np.ndarray,
    states: Sequence[RenderState],
    config: RenderConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    finite = elevation[np.isfinite(elevation)]
    if finite.size == 0:
        raise Phase2RenderError("Rendering grid has no finite bed elevations")
    bed = config.bed_color_limits_m or (float(finite.min()), float(finite.max()))
    maximum_depth = max(
        float(
            np.nanmax(
                np.where(
                    state.inundation,
                    state.water_surface_stage_m_navd88 - elevation,
                    np.nan,
                )
            )
        )
        for state in states
    )
    depth = config.depth_color_limits_m or (0.0, max(maximum_depth, 1e-9))
    if bed[0] >= bed[1] or depth[0] >= depth[1]:
        raise Phase2RenderError("Common bed/depth color limits must increase")
    return bed, depth


def _validate_render_states(
    elevation: np.ndarray, states: Sequence[RenderState]
) -> None:
    for state in states:
        inundation = np.asarray(state.inundation, dtype=bool)
        if inundation.shape != elevation.shape:
            raise Phase2RenderError(
                "Every rendering state must align with the bed grid"
            )
        if state.validation_status.strip().lower() not in QUANTITATIVE_GATE_STATUSES:
            raise Phase2RenderError(
                f"State {state.state_id!r} has not passed quantitative validation"
            )
        if np.any(
            inundation
            & (
                ~np.isfinite(elevation)
                | (elevation > state.water_surface_stage_m_navd88)
            )
        ):
            raise Phase2RenderError(
                f"State {state.state_id!r} includes invalid or above-stage water"
            )


def plot_stage_3d(
    elevation_m: np.ndarray,
    *,
    transform: Any,
    state: RenderState,
    config: RenderConfig | None = None,
    common_bed_limits_m: tuple[float, float] | None = None,
    common_depth_limits_m: tuple[float, float] | None = None,
    include_annotations: bool = True,
) -> mpl.figure.Figure:
    """Render one quantitative state with a restrained, fixed camera."""

    config = config or RenderConfig()
    elevation = np.asarray(elevation_m, dtype=np.float64)
    inundation = np.asarray(state.inundation, dtype=bool)
    if elevation.ndim != 2 or elevation.shape != inundation.shape:
        raise Phase2RenderError("Elevation and inundation rendering grids must align")
    if np.any(
        inundation
        & (~np.isfinite(elevation) | (elevation > state.water_surface_stage_m_navd88))
    ):
        raise Phase2RenderError("Rendered water includes invalid or above-stage cells")
    stride = config.stride
    bed = elevation[::stride, ::stride]
    water_mask = inundation[::stride, ::stride]
    x_grid, y_grid = _xy_grid(elevation.shape, transform, stride)
    bed_limits = common_bed_limits_m or config.bed_color_limits_m
    if bed_limits is None:
        finite = bed[np.isfinite(bed)]
        bed_limits = (float(finite.min()), float(finite.max()))
    calculation_stage = state.water_surface_stage_m_navd88
    depths = np.where(water_mask, calculation_stage - bed, np.nan)
    depth_limits = common_depth_limits_m or config.depth_color_limits_m
    if depth_limits is None:
        maximum = float(np.nanmax(depths)) if np.isfinite(depths).any() else 1.0
        depth_limits = (0.0, max(maximum, 1e-9))
    bed_normalizer = mpl.colors.Normalize(*bed_limits)
    depth_normalizer = mpl.colors.Normalize(*depth_limits)
    figure = plt.figure(
        figsize=(config.figure_width_inches, config.figure_height_inches),
        constrained_layout=True,
    )
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(
        x_grid,
        y_grid,
        np.ma.masked_invalid(bed),
        facecolors=plt.colormaps["terrain"](bed_normalizer(bed)),
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.92,
    )
    water = np.where(water_mask, calculation_stage, np.nan)
    axis.plot_surface(
        x_grid,
        y_grid,
        water,
        facecolors=plt.colormaps["Blues"](depth_normalizer(depths)),
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.82,
    )
    if water_mask.any() and (~water_mask).any():
        axis.contour(
            x_grid,
            y_grid,
            water_mask.astype(float),
            levels=[0.5],
            zdir="z",
            offset=calculation_stage + 0.01,
            colors=["#08306B"],
            linewidths=1.0,
        )
    axis.view_init(
        elev=config.camera.elevation_degrees,
        azim=config.camera.azimuth_degrees,
        roll=config.camera.roll_degrees,
    )
    axis.set_proj_type(config.camera.projection)
    x_span = max(float(np.ptp(x_grid)), 1e-9)
    y_span = max(float(np.ptp(y_grid)), 1e-9)
    z_span_km = max((bed_limits[1] - bed_limits[0]) / 1000.0, 1e-9)
    axis.set_box_aspect(
        (x_span, y_span, z_span_km * config.vertical_exaggeration), zoom=0.9
    )
    axis.set_xlim(float(x_grid.min()), float(x_grid.max()))
    axis.set_ylim(float(y_grid.min()), float(y_grid.max()))
    z_maximum = max(bed_limits[1], calculation_stage)
    axis.set_zlim(bed_limits[0], z_maximum)
    # The disclosed 8x vertical exaggeration remains physically shallow
    # relative to Bear Lake's 32 km length.  Two endpoint labels avoid an
    # unreadable stack of projected z tick labels without changing geometry.
    axis.set_zticks([bed_limits[0], z_maximum])
    axis.tick_params(axis="z", labelsize=7, pad=1)
    axis.set_xlabel("Easting offset (km)")
    axis.set_ylabel("Northing offset (km)")
    axis.set_zlabel("Elevation (m NAVD88)")
    axis.set_title(
        f"Bear Lake - {state.observation_date} - {state.stage_ft_navd88:.2f} ft NAVD88"
    )
    if include_annotations:
        area = (
            "not available"
            if state.raster_area_m2 is None
            else f"{state.raster_area_m2 / 1e6:,.2f} km2"
        )
        storage = (
            "not available"
            if state.raster_storage_m3 is None
            else f"{state.raster_storage_m3 / 1e9:,.3f} km3"
        )
        validation_annotation = state.validation_status
        if state.validation_detail.strip():
            validation_annotation += f"; {state.validation_detail.strip()}"
        figure.text(
            0.02,
            0.02,
            f"Raster area: {area} | Raster storage: {storage}\n"
            f"Vertical exaggeration: {config.vertical_exaggeration:g}x | "
            f"Validation: {validation_annotation}\n"
            f"Method: {state.method}; quality: {state.quality_status}; "
            f"approval: {state.approval_status}\n"
            f"{state.source}; survey years: {state.survey_years}; "
            f"provenance: {state.provenance_label}",
            ha="left",
            va="bottom",
            fontsize=7,
        )
    scalar = mpl.cm.ScalarMappable(norm=depth_normalizer, cmap="Blues")
    scalar.set_array([])
    figure.colorbar(scalar, ax=axis, shrink=0.55, pad=0.08, label="Water depth (m)")
    return figure


def render_static_states(
    elevation_m: np.ndarray,
    *,
    transform: Any,
    states: Sequence[RenderState],
    output_dir: Path,
    gate: Any,
    validation_figure: Path | None = None,
    config: RenderConfig | None = None,
) -> list[Path]:
    """Save fixed-camera low/latest/high PNG views after the render gate."""

    require_render_gate(gate, validation_figure=validation_figure)
    config = config or RenderConfig()
    if not states:
        raise Phase2RenderError("At least one render state is required")
    elevation = np.asarray(elevation_m, dtype=np.float64)
    expected_states = {"recent_low", "latest", "recent_high"}
    state_ids = {state.state_id for state in states}
    if state_ids != expected_states or len(states) != len(expected_states):
        raise Phase2RenderError(
            "Static rendering requires exactly recent_low, latest, and recent_high"
        )
    _validate_render_states(elevation, states)
    bed_limits, depth_limits = _limits(elevation, states, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    aliases = {
        "recent_low": "recent_low",
        "latest": "latest",
        "recent_high": "recent_high",
    }
    output: list[Path] = []
    with mpl.rc_context(RENDER_STYLE):
        for state in states:
            suffix = aliases.get(state.state_id, state.state_id)
            path = output_dir / f"bear_lake_3d_{suffix}.png"
            figure = plot_stage_3d(
                elevation,
                transform=transform,
                state=state,
                config=config,
                common_bed_limits_m=bed_limits,
                common_depth_limits_m=depth_limits,
            )
            figure.savefig(
                path,
                dpi=config.dpi,
                metadata={
                    "Date": None,
                    "StageDate": state.observation_date,
                    "StageMetresNAVD88": str(state.stage_m_navd88),
                    "CalculationStageMetresNAVD88": str(
                        state.water_surface_stage_m_navd88
                    ),
                    "VerticalExaggeration": str(config.vertical_exaggeration),
                    "ValidationStatus": state.validation_status,
                    "ValidationDetail": state.validation_detail,
                    "RenderingGrid": "decimated; not authoritative for area/storage",
                },
            )
            plt.close(figure)
            output.append(path)
    return output


def _physical_scene_aspect_ratio(
    x_grid_km: np.ndarray,
    y_grid_km: np.ndarray,
    elevation_m: np.ndarray,
    states: Sequence[RenderState],
    *,
    vertical_exaggeration: float,
) -> dict[str, float]:
    """Return Plotly box proportions in common physical units.

    Plotly receives horizontal coordinates in kilometres and elevations in
    metres.  Its manual aspect ratio is a display-space ratio, so the vertical
    span must first be converted to kilometres and then multiplied by the
    disclosed vertical exaggeration.  Normalizing all three axes by one common
    horizontal span preserves non-square lake geometry.
    """

    x_values = np.asarray(x_grid_km, dtype=np.float64)
    y_values = np.asarray(y_grid_km, dtype=np.float64)
    bed = np.asarray(elevation_m, dtype=np.float64)
    finite_bed = bed[np.isfinite(bed)]
    if finite_bed.size == 0:
        raise Phase2RenderError("Interactive bed grid has no finite elevations")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise Phase2RenderError("Interactive horizontal coordinates must be finite")
    x_span_km = float(np.ptp(x_values))
    y_span_km = float(np.ptp(y_values))
    if x_span_km <= 0 or y_span_km <= 0:
        raise Phase2RenderError("Interactive rendering requires nonzero x/y spans")
    maximum_stage_m = max(state.water_surface_stage_m_navd88 for state in states)
    z_min_m = float(finite_bed.min())
    z_max_m = max(float(finite_bed.max()), maximum_stage_m)
    z_span_km = max((z_max_m - z_min_m) / 1000.0, np.finfo(float).eps)
    horizontal_reference_km = max(x_span_km, y_span_km)
    return {
        "x": x_span_km / horizontal_reference_km,
        "y": y_span_km / horizontal_reference_km,
        "z": z_span_km * vertical_exaggeration / horizontal_reference_km,
    }


def _interactive_annotation(
    state: RenderState, config: RenderConfig, gate_status: str
) -> dict[str, Any]:
    """Build the state-specific scientific annotation used by Plotly."""

    validation_detail = state.validation_detail.strip()
    validation_line = f"Validation: {gate_status}"
    if validation_detail:
        validation_line += f"; {validation_detail}"
    return {
        "text": (
            "Rendering grid only; "
            f"vertical exaggeration {config.vertical_exaggeration:g}x; "
            f"{validation_line}<br>"
            f"Method: {state.method}; quality: {state.quality_status}; "
            f"approval: {state.approval_status}<br>"
            f"Nominal stage: {state.stage_m_navd88:.9f} m NAVD88; "
            "calculation plane: "
            f"{state.water_surface_stage_m_navd88:.9f} m NAVD88<br>"
            f"Source: {state.source}; survey years: {state.survey_years}; "
            f"provenance: {state.provenance_label}"
        ),
        "xref": "paper",
        "yref": "paper",
        "x": 0,
        "y": 0,
        "showarrow": False,
        "align": "left",
    }


def create_interactive_html(
    elevation_m: np.ndarray,
    *,
    transform: Any,
    states: Sequence[RenderState],
    output_path: Path,
    gate: Any,
    maximum_file_size_mb: float = 50.0,
    config: RenderConfig | None = None,
) -> InteractiveRenderResult:
    """Create optional standalone Plotly HTML under a strict byte-size gate."""

    require_render_gate(gate)
    config = config or RenderConfig(stride=4)
    if maximum_file_size_mb <= 0:
        raise Phase2RenderError("Interactive HTML size limit must be positive")
    try:
        import plotly.io as plotly_io
        from plotly import graph_objects
    except (ImportError, OSError) as error:  # pragma: no cover - environment guard
        output_path.unlink(missing_ok=True)
        return InteractiveRenderResult(False, None, 0, f"Plotly unavailable: {error}")
    if not states:
        raise Phase2RenderError("Interactive rendering requires at least one state")
    elevation = np.asarray(elevation_m, dtype=np.float64)
    _validate_render_states(elevation, states)
    stride = config.stride
    bed = elevation[::stride, ::stride]
    x_grid, y_grid = _xy_grid(elevation.shape, transform, stride)
    aspect_ratio = _physical_scene_aspect_ratio(
        x_grid,
        y_grid,
        bed,
        states,
        vertical_exaggeration=config.vertical_exaggeration,
    )
    gate_status = _status_from_gate(gate)
    state_annotations = [
        [_interactive_annotation(state, config, gate_status)] for state in states
    ]
    sampled_masks = [
        np.asarray(state.inundation, dtype=bool)[::stride, ::stride] for state in states
    ]
    maximum_depth_m = max(
        float(np.nanmax(state.water_surface_stage_m_navd88 - bed[mask]))
        for state, mask in zip(states, sampled_masks, strict=True)
    )
    depth_limits = config.depth_color_limits_m or (0.0, max(maximum_depth_m, 1e-9))
    traces: list[Any] = [
        graph_objects.Surface(
            x=x_grid,
            y=y_grid,
            z=bed,
            colorscale="Earth",
            showscale=True,
            colorbar={
                "title": {"text": "Bed m NAVD88", "side": "right"},
                "x": 0.88,
                "y": 0.76,
                "len": 0.34,
                "thickness": 18,
            },
            name="Surveyed bed",
            hovertemplate="x %{x:.2f} km<br>y %{y:.2f} km<br>bed %{z:.2f} m<extra></extra>",
        )
    ]
    for index, (state, mask) in enumerate(zip(states, sampled_masks, strict=True)):
        calculation_stage = state.water_surface_stage_m_navd88
        water = np.where(mask, calculation_stage, np.nan)
        traces.append(
            graph_objects.Surface(
                x=x_grid,
                y=y_grid,
                z=water,
                surfacecolor=np.where(mask, calculation_stage - bed, np.nan),
                colorscale="Blues",
                cmin=depth_limits[0],
                cmax=depth_limits[1],
                showscale=True,
                colorbar={
                    "title": {"text": "Depth m", "side": "right"},
                    "x": 0.88,
                    "y": 0.27,
                    "len": 0.34,
                    "thickness": 18,
                },
                name=state.state_id,
                visible=index == 0,
                hovertemplate=(
                    f"{state.observation_date}<br>stage {state.stage_m_navd88:.3f} m NAVD88"
                    f"<br>calculation plane {calculation_stage:.9f} m NAVD88"
                    "<extra></extra>"
                ),
            )
        )
    buttons = []
    for index, state in enumerate(states):
        visible = [True] + [item == index for item in range(len(states))]
        buttons.append(
            {
                "label": f"{state.state_id}: {state.observation_date}",
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": (
                            f"Bear Lake - {state.observation_date} - "
                            f"{state.stage_ft_navd88:.2f} ft NAVD88"
                        ),
                        "annotations": state_annotations[index],
                    },
                ],
            }
        )
    figure = graph_objects.Figure(data=traces)
    figure.update_layout(
        title=(
            f"Bear Lake - {states[0].observation_date} - "
            f"{states[0].stage_ft_navd88:.2f} ft NAVD88"
        ),
        updatemenus=[{"type": "dropdown", "buttons": buttons, "x": 0.01, "y": 0.98}],
        annotations=state_annotations[0],
        scene={
            "domain": {"x": [0.0, 0.78], "y": [0.14, 0.98]},
            "xaxis_title": "Easting offset (km)",
            "yaxis_title": "Northing offset (km)",
            "zaxis": {
                "title": "Elevation (m NAVD88)",
                "tickmode": "array",
                "tickvals": [float(np.nanmin(bed)), float(np.nanmax(bed))],
                "ticktext": [
                    f"{float(np.nanmin(bed)):.1f}",
                    f"{float(np.nanmax(bed)):.1f}",
                ],
            },
            "aspectmode": "manual",
            "aspectratio": aspect_ratio,
            "camera": {"projection": {"type": "orthographic"}},
        },
        template="plotly_white",
        height=900,
        margin={"l": 40, "r": 180, "t": 70, "b": 135},
    )
    html = plotly_io.to_html(
        figure,
        full_html=True,
        include_plotlyjs=True,
        auto_play=False,
        config={"responsive": True, "displaylogo": False},
        div_id="bear-lake-phase2-3d",
    ).encode("utf-8")
    limit = int(maximum_file_size_mb * 1024 * 1024)
    if len(html) > limit:
        output_path.unlink(missing_ok=True)
        return InteractiveRenderResult(
            False,
            None,
            len(html),
            f"Standalone HTML ({len(html)} bytes) exceeds configured {limit}-byte limit",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".html.part", dir=output_path.parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(html)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return InteractiveRenderResult(True, output_path, len(html), None)


__all__ = [
    "RENDER_STYLE",
    "CameraConfig",
    "InteractiveRenderResult",
    "Phase2RenderError",
    "RenderConfig",
    "RenderState",
    "create_interactive_html",
    "plot_stage_3d",
    "render_static_states",
    "require_render_gate",
    "run_with_quantitative_hash_guard",
    "verify_file_hashes",
]
