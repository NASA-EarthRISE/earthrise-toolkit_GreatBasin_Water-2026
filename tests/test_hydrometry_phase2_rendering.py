from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import matplotlib
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from PIL import Image

from regional_analysis.hydrometry.animate_lake import (
    AnimationError,
    create_stage_change_gif,
    normalize_animation_frame_table,
    select_observed_frames,
)
from regional_analysis.hydrometry.plot_phase2 import (
    FIGURE_NAMES,
    Phase2FigureInputs,
    create_phase2_figures,
    plot_current_depth,
)
from regional_analysis.hydrometry.raster_storage import RenderGateResult
from regional_analysis.hydrometry.render_3d import (
    Phase2RenderError,
    RenderConfig,
    RenderState,
    _physical_scene_aspect_ratio,
    create_interactive_html,
    plot_stage_3d,
    render_static_states,
    require_render_gate,
    run_with_quantitative_hash_guard,
)
from regional_analysis.hydrometry.spatial_products import (
    FLOAT_NODATA,
    inundation_change_codes,
)

matplotlib.use("Agg", force=True)


def _transform():
    from affine import Affine

    return Affine.translation(500_000.0, 4_650_000.0) * Affine.scale(10.0, -10.0)


def _bed() -> np.ndarray:
    y, x = np.mgrid[-1:1:12j, -1:1:12j]
    return 1740.0 + 8.0 * (x * x + y * y)


def _render_states(elevation: np.ndarray) -> list[RenderState]:
    states = []
    for state_id, day, stage in (
        ("recent_low", "2022-11-05", 1745.5),
        ("latest", "2026-08-12", 1748.0),
        ("recent_high", "2024-06-22", 1751.0),
    ):
        water = elevation <= stage
        states.append(
            RenderState(
                state_id,
                day,
                stage,
                stage / 0.3048,
                water,
                float(water.sum() * 100),
                float(np.maximum(stage - elevation, 0)[water].sum() * 100),
            )
        )
    return states


def _validation() -> pd.DataFrame:
    rows = []
    for resolution in ("source", 5, 10, 30):
        for stage in (1745.0, 1748.0, 1751.0):
            official_area = 1_000_000 + (stage - 1745) * 300_000
            official_storage = 10_000_000 + (stage - 1745) * 2_000_000
            factor = 1.0 if resolution == "source" else 1 + float(resolution) / 10000
            rows.append(
                {
                    "stage_m_navd88": stage,
                    "official_area_m2": official_area,
                    "raster_area_m2": official_area * factor,
                    "relative_area_error": factor - 1,
                    "official_storage_m3": official_storage,
                    "raster_storage_m3": official_storage * factor,
                    "relative_storage_error": factor - 1,
                    "working_resolution": resolution,
                }
            )
    return pd.DataFrame(rows)


class Phase2RenderingTests(unittest.TestCase):
    def test_frame_selection_is_deterministic_and_includes_extrema_latest(self) -> None:
        observations = pd.DataFrame(
            {
                "calendar_date": pd.date_range("2020-01-01", periods=30, freq="30D"),
                "value": np.r_[10.0, np.linspace(11, 18, 27), 25.0, 15.0],
                "approval_status": "approved",
            }
        )
        first = select_observed_frames(observations, frame_count=12)
        second = select_observed_frames(observations, frame_count=12)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 12)
        self.assertTrue(first["observation_date"].is_monotonic_increasing)
        self.assertIn(10.0, first["stage_m_navd88"].to_numpy())
        self.assertIn(25.0, first["stage_m_navd88"].to_numpy())
        self.assertEqual(first.iloc[-1]["stage_m_navd88"], 15.0)

    def test_frame_table_requires_real_chronological_stages(self) -> None:
        valid = pd.DataFrame(
            {
                "frame_number": [1, 2],
                "target_date": ["2025-01-01", "2026-01-01"],
                "observation_date": ["2025-01-02", "2026-01-02"],
                "stage_m_navd88": [1800.0, 1801.0],
                "stage_original_ft": [5905.5, 5908.8],
            }
        )
        normalized = normalize_animation_frame_table(valid)
        self.assertEqual(normalized.iloc[0]["stage_ft_navd88"], 5905.5)
        self.assertEqual(
            normalized.iloc[0]["calculation_stage_m_navd88"],
            normalized.iloc[0]["stage_m_navd88"],
        )
        invalid = valid.copy()
        invalid["observation_date"] = ["2026-01-02", "2025-01-02"]
        with self.assertRaisesRegex(AnimationError, "chronological"):
            normalize_animation_frame_table(invalid)

    def test_render_geometry_uses_calculation_plane_but_labels_nominal_stage(
        self,
    ) -> None:
        nominal_stage = 1805.15
        calculation_stage = float(np.float32(nominal_stage))
        elevation = np.array(
            [[calculation_stage, calculation_stage - 1.0], [1806.0, 1807.0]],
            dtype=np.float64,
        )
        state = RenderState(
            "latest",
            "2026-08-12",
            nominal_stage,
            nominal_stage / 0.3048,
            np.array([[True, True], [False, False]]),
            calculation_stage_m_navd88=calculation_stage,
        )
        figure = plot_stage_3d(
            elevation,
            transform=_transform(),
            state=state,
            config=RenderConfig(
                figure_width_inches=3,
                figure_height_inches=2.5,
                dpi=60,
            ),
        )
        try:
            self.assertIn(f"{state.stage_ft_navd88:.2f} ft", figure.axes[0].get_title())
            self.assertEqual(state.water_surface_stage_m_navd88, calculation_stage)
        finally:
            from matplotlib import pyplot as plt

            plt.close(figure)

    def test_gate_and_hash_guard_block_unsafe_rendering(self) -> None:
        with self.assertRaisesRegex(Phase2RenderError, "validation gate"):
            require_render_gate("failed")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "quantitative.tif"
            path.write_bytes(b"authoritative")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            def mutating_action():
                path.write_bytes(b"changed")

            with self.assertRaisesRegex(Phase2RenderError, "hash mismatch"):
                run_with_quantitative_hash_guard({path: digest}, mutating_action)

    def test_accepts_quantitative_engine_render_gate_result(self) -> None:
        gate = RenderGateResult(
            allowed=True,
            validation_status="numerically_consistent",
            reason="synthetic gate passed",
            validation_policy_sha256="a" * 64,
        )
        self.assertEqual(require_render_gate(gate), "numerically_consistent")

    def test_static_views_use_identical_dimensions_and_metadata(self) -> None:
        elevation = _bed()
        states = _render_states(elevation)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            validation_figure = output / "bear_lake_eav_validation.png"
            validation_figure.write_bytes(b"gate evidence")
            paths = render_static_states(
                elevation,
                transform=_transform(),
                states=states,
                output_dir=output,
                gate="numerically_consistent",
                validation_figure=validation_figure,
                config=RenderConfig(
                    figure_width_inches=3,
                    figure_height_inches=2.5,
                    dpi=60,
                    vertical_exaggeration=8,
                ),
            )
            self.assertEqual(
                [path.name for path in paths],
                [
                    "bear_lake_3d_recent_low.png",
                    "bear_lake_3d_latest.png",
                    "bear_lake_3d_recent_high.png",
                ],
            )
            dimensions = set()
            for path in paths:
                with Image.open(path) as image:
                    dimensions.add(image.size)
                    self.assertEqual(image.text["VerticalExaggeration"], "8")
                    self.assertEqual(
                        image.text["ValidationStatus"], "numerically_consistent"
                    )
            self.assertEqual(len(dimensions), 1)

    def test_gif_frame_order_count_and_duration(self) -> None:
        elevation = _bed()
        states = sorted(
            _render_states(elevation), key=lambda state: state.observation_date
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bear_lake_stage_change.gif"
            create_stage_change_gif(
                elevation,
                transform=_transform(),
                states=states,
                output_path=path,
                gate="numerically_consistent",
                config=RenderConfig(
                    figure_width_inches=2.5,
                    figure_height_inches=2.0,
                    dpi=50,
                    vertical_exaggeration=8,
                ),
                duration_ms=400,
            )
            with Image.open(path) as animation:
                self.assertEqual(animation.n_frames, 3)
                self.assertEqual(animation.info["duration"], 400)
                dimensions = set()
                for frame in range(animation.n_frames):
                    animation.seek(frame)
                    dimensions.add(animation.size)
                self.assertEqual(len(dimensions), 1)
            second = Path(temporary) / "second.gif"
            create_stage_change_gif(
                elevation,
                transform=_transform(),
                states=states,
                output_path=second,
                gate="numerically_consistent",
                config=RenderConfig(
                    figure_width_inches=2.5,
                    figure_height_inches=2.0,
                    dpi=50,
                    vertical_exaggeration=8,
                ),
                duration_ms=400,
            )
            self.assertEqual(path.read_bytes(), second.read_bytes())

    def test_render_annotations_retain_method_quality_and_provenance(self) -> None:
        elevation = _bed()
        states = [
            replace(
                state,
                survey_years="survey-years-2002-2018",
                quality_status=f"quality-{index}",
                approval_status=f"approval-{index}",
                provenance_label="synthetic-provenance.json",
                validation_detail=(
                    "source_method_consistent_with_numeric_area_warning; "
                    f"exact numeric area exceptions={index}"
                ),
            )
            for index, state in enumerate(_render_states(elevation), start=1)
        ]
        state = states[0]
        self.assertEqual(state.method, "connected_seed")
        self.assertEqual(state.quality_status, "quality-1")
        self.assertEqual(state.provenance_label, "synthetic-provenance.json")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "interactive.html"
            result = create_interactive_html(
                elevation,
                transform=_transform(),
                states=states,
                output_path=path,
                gate="numerically_consistent",
                maximum_file_size_mb=10,
                config=RenderConfig(stride=2),
            )
            if not result.created:
                self.skipTest(result.reason)
            html = path.read_text(encoding="utf-8")
            self.assertIn("synthetic-provenance.json", html)
            self.assertIn("USGS observed stage", html)
            self.assertIn("survey-years-2002-2018", html)
            self.assertIn("connected_seed", html)
            for index in range(1, 4):
                self.assertIn(f"quality-{index}", html)
                self.assertIn(f"approval-{index}", html)
                self.assertIn(f"exact numeric area exceptions={index}", html)

        figure = plot_stage_3d(
            elevation,
            transform=_transform(),
            state=state,
            config=RenderConfig(
                figure_width_inches=3,
                figure_height_inches=2.5,
                dpi=60,
            ),
        )
        try:
            annotation = "\n".join(text.get_text() for text in figure.texts)
            self.assertIn("Validation: numerically_consistent", annotation)
            self.assertIn("exact numeric area exceptions=1", annotation)
        finally:
            from matplotlib import pyplot as plt

            plt.close(figure)

    def test_plotly_aspect_preserves_horizontal_shape_and_vertical_scale(self) -> None:
        states = _render_states(_bed())
        x_grid, y_grid = np.meshgrid(
            np.linspace(0.0, 40.0, 5), np.linspace(0.0, 10.0, 3)
        )
        bed = np.linspace(1700.0, 1800.0, 15).reshape(3, 5)
        ratio = _physical_scene_aspect_ratio(
            x_grid,
            y_grid,
            bed,
            states,
            vertical_exaggeration=8.0,
        )
        self.assertAlmostEqual(ratio["x"], 1.0)
        self.assertAlmostEqual(ratio["y"], 0.25)
        # 100 vertical metres = 0.1 km; 8x exaggeration over a 40 km reference.
        self.assertAlmostEqual(ratio["z"], 0.02)

    def test_plotly_uses_common_depth_scale_and_separate_colorbars(self) -> None:
        elevation = _bed()
        states = _render_states(elevation)
        captured = []

        def capture_figure(figure, **_kwargs):
            captured.append(figure)
            return "<html><body>captured</body></html>"

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "interactive.html"
            with patch("plotly.io.to_html", side_effect=capture_figure):
                result = create_interactive_html(
                    elevation,
                    transform=_transform(),
                    states=states,
                    output_path=path,
                    gate="numerically_consistent",
                    maximum_file_size_mb=1,
                    config=RenderConfig(stride=2),
                )
        self.assertTrue(result.created)
        self.assertEqual(len(captured), 1)
        figure = captured[0]
        bed_trace, *water_traces = figure.data
        self.assertTrue(all(trace.showscale for trace in water_traces))
        self.assertEqual(len({float(trace.cmax) for trace in water_traces}), 1)
        self.assertNotEqual(
            float(bed_trace.colorbar.y), float(water_traces[0].colorbar.y)
        )
        sampled_bed = elevation[::2, ::2]
        self.assertEqual(
            list(figure.layout.scene.zaxis.tickvals),
            [float(np.nanmin(sampled_bed)), float(np.nanmax(sampled_bed))],
        )

    def test_plotly_file_size_gate_omits_oversized_standalone_html(self) -> None:
        elevation = _bed()
        states = _render_states(elevation)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "interactive.html"
            result = create_interactive_html(
                elevation,
                transform=_transform(),
                states=states,
                output_path=path,
                gate="numerically_consistent",
                maximum_file_size_mb=0.001,
                config=RenderConfig(stride=2),
            )
            self.assertFalse(result.created)
            self.assertGreater(result.file_size_bytes, 1024)
            self.assertFalse(path.exists())

    def test_plotly_standalone_html_is_byte_reproducible(self) -> None:
        elevation = _bed()
        states = _render_states(elevation)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = output / "first.html"
            second = output / "second.html"
            first_result = create_interactive_html(
                elevation,
                transform=_transform(),
                states=states,
                output_path=first,
                gate="numerically_consistent",
                maximum_file_size_mb=10,
                config=RenderConfig(stride=2),
            )
            second_result = create_interactive_html(
                elevation,
                transform=_transform(),
                states=states,
                output_path=second,
                gate="numerically_consistent",
                maximum_file_size_mb=10,
                config=RenderConfig(stride=2),
            )
            if not first_result.created or not second_result.created:
                self.skipTest(first_result.reason or second_result.reason)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertIn('id="bear-lake-phase2-3d"', first.read_text(encoding="utf-8"))

    def test_four_2d_figure_families_include_text_free_geometry(self) -> None:
        elevation = _bed()
        states = {state.state_id: state for state in _render_states(elevation)}
        latest = states["latest"]
        prior_stage = 1747.0
        prior = elevation <= prior_stage
        coverage = np.isfinite(elevation)
        depth = np.full(elevation.shape, FLOAT_NODATA, dtype=float)
        depth[coverage] = 0.0
        depth[latest.inundation] = latest.stage_m_navd88 - elevation[latest.inundation]
        prior_depth = np.zeros_like(elevation)
        prior_depth[prior] = prior_stage - elevation[prior]
        change_depth = depth - prior_depth
        change = inundation_change_codes(
            latest.inundation, prior, supported_mask=coverage
        )
        metadata = {
            state.state_id: {
                "date": state.observation_date,
                "stage_m_navd88": state.stage_m_navd88,
            }
            for state in states.values()
        }
        metadata["year_ago"] = {"date": "2025-08-12", "stage_m_navd88": prior_stage}
        inputs = Phase2FigureInputs(
            elevation_m=elevation,
            coverage_mask=coverage,
            depth_current_m=depth,
            inundation_masks={
                "latest": latest.inundation,
                "recent_low": states["recent_low"].inundation,
                "recent_high": states["recent_high"].inundation,
                "year_ago": prior,
            },
            depth_change_m=change_depth,
            inundation_change=change,
            transform=_transform(),
            validation=_validation().loc[
                _validation()["working_resolution"].eq("source")
            ],
            resolution_benchmark=_validation().loc[
                ~_validation()["working_resolution"].eq("source")
            ],
            stage_metadata=metadata,
            validation_status="numerically_consistent",
            validation_detail=(
                "source_method_consistent_with_numeric_area_warning; "
                "exact numeric area exceptions=1"
            ),
        )
        current_depth = plot_current_depth(inputs)
        try:
            annotation = "\n".join(text.get_text() for text in current_depth.texts)
            self.assertIn("exact numeric area exceptions=1", annotation)
        finally:
            from matplotlib import pyplot as plt

            plt.close(current_depth)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            written = create_phase2_figures(inputs, output)
            self.assertEqual(len(written), len(FIGURE_NAMES) * 4 + 1)
            for name in FIGURE_NAMES:
                labeled = output / f"{name}.png"
                no_text = output / "no_text" / f"{name}.png"
                self.assertTrue(labeled.is_file())
                self.assertEqual(
                    mpimg.imread(labeled).shape[:2], mpimg.imread(no_text).shape[:2]
                )
                svg = (output / "no_text" / f"{name}.svg").read_text(encoding="utf-8")
                self.assertNotIn("<text", svg)
            notes = (output / "figure_notes.md").read_text(encoding="utf-8")
            self.assertIn("exact numeric area exceptions=1", notes)


if __name__ == "__main__":
    unittest.main()
