from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from regional_analysis.hydrometry.phase2_bear_lake import (
    PHASE2_GDAL_CACHEMAX_BYTES,
    PHASE2_SCHEMA_VERSION,
    PHASE2_SCIENTIFIC_HASH_CONTRACT,
    Phase2Error,
    SelectedStage,
    _archive_request,
    _assess_connected_basin_stage,
    _bounded_rasterio_runtime,
    _calculation_plane_at_stage,
    _load_render_gate,
    _prepare_rasterio_runtime,
    _quantitative_hashes,
    _quantitative_paths,
    _quantitative_scientific_hashes,
    _records_sha256,
    _render_products,
    _render_state,
    _require_matching_source_crs,
    _require_matching_validation_inputs,
    _same_calculation_plane,
    _scalar_at_stage,
    _sciencebase_archive_transport_url,
    _sciencebase_catalog_transport_url,
    _scientific_content_sha256,
    _scientific_hash_contract,
    _select_converged_resolution,
    _selected_stage_frame,
    _set_unique_index,
    _source_year_label,
    _stagewise_precision_gate,
    _stagewise_source_method_consistency_gate,
    _validated_vertical_reference,
    _validation_input_records,
    load_bear_eav_source,
    load_phase2_settings,
    select_animation_stages,
    select_stage_states,
)
from regional_analysis.hydrometry.raster_storage import (
    ValidationPolicy,
    integrate_raster_threshold_events,
)
from regional_analysis.hydrometry.sciencebase import ScienceBaseFile


class Phase2BearLakeConfigTests(unittest.TestCase):
    def test_repository_config_is_narrow_and_pinned(self) -> None:
        settings = load_phase2_settings("configs/hydrometry_pilots.yaml")
        self.assertEqual(settings.profile_id, "bear_lake")
        self.assertEqual(settings.sciencebase_item_id, "698226f4b66b018d00102e42")
        self.assertEqual(settings.candidate_spatial_resolutions_m, (5.0, 10.0, 30.0))
        self.assertEqual(settings.connectivity, 8)
        self.assertEqual(settings.animation_frame_count, 18)
        self.assertEqual(settings.trusted_seed.crs, "EPSG:4326")
        self.assertIn("P14KZS2D", settings.doi)

    def test_runtime_guard_loads_stdlib_dlls_before_rasterio(self) -> None:
        script = (
            "from regional_analysis.hydrometry.phase2_bear_lake import "
            "_prepare_rasterio_runtime; _prepare_rasterio_runtime(); "
            "import rasterio, sqlite3, pyexpat"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_runtime_guard_is_idempotent(self) -> None:
        _prepare_rasterio_runtime()
        _prepare_rasterio_runtime()

    def test_bounded_gdal_cache_is_exact_and_restored(self) -> None:
        _prepare_rasterio_runtime()
        import rasterio
        from rasterio.env import get_gdal_config

        prior_for_test = 64 * 1024**2
        with rasterio.Env(GDAL_CACHEMAX=prior_for_test):
            self.assertEqual(int(get_gdal_config("GDAL_CACHEMAX")), prior_for_test)
            with _bounded_rasterio_runtime():
                self.assertEqual(
                    int(get_gdal_config("GDAL_CACHEMAX")),
                    PHASE2_GDAL_CACHEMAX_BYTES,
                )
            self.assertEqual(int(get_gdal_config("GDAL_CACHEMAX")), prior_for_test)

    def test_bounded_gdal_cache_preserves_file_backed_scientific_results(
        self,
    ) -> None:
        _prepare_rasterio_runtime()
        import rasterio
        from rasterio.transform import from_origin

        elevation = np.arange(32 * 32, dtype=np.float32).reshape(32, 32) / 100.0 - 2.0
        elevation[0, 0] = np.float32(-9999.0)
        stages = (-0.25, 0.75, 1.5)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "synthetic_tiled_float32.tif")
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=elevation.shape[1],
                height=elevation.shape[0],
                count=1,
                dtype="float32",
                nodata=-9999.0,
                transform=from_origin(0.0, 32.0, 1.0, 1.0),
                crs="EPSG:32612",
                tiled=True,
                blockxsize=16,
                blockysize=16,
                compress="deflate",
            ) as dataset:
                dataset.write(elevation, 1)

            source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            baseline = integrate_raster_threshold_events(
                path,
                stages,
                source_resolution_m=1.0,
            )
            with _bounded_rasterio_runtime():
                bounded = integrate_raster_threshold_events(
                    path,
                    stages,
                    source_resolution_m=1.0,
                )

            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), source_sha256
            )
            for baseline_result, bounded_result in zip(baseline, bounded, strict=True):
                baseline_fields = asdict(baseline_result)
                bounded_fields = asdict(bounded_result)
                baseline_fields.pop("processing_seconds")
                bounded_fields.pop("processing_seconds")
                self.assertEqual(bounded_fields, baseline_fields)

    def test_source_raster_crs_must_match_official_prj(self) -> None:
        _require_matching_source_crs(
            SimpleNamespace(horizontal_epsg=6341), {"horizontal_epsg": 6341}
        )
        with self.assertRaisesRegex(Phase2Error, "does not match"):
            _require_matching_source_crs(
                SimpleNamespace(horizontal_epsg=26912),
                {"horizontal_epsg": 6341},
            )

    def test_structured_fgdc_vertical_reference_must_be_navd88_metres(self) -> None:
        datum, units = _validated_vertical_reference(
            {
                "vertical_datum": "North American Vertical Datum of 1988 (EPSG:5703)",
                "vertical_units": "meters",
            }
        )
        self.assertIn("1988", datum)
        self.assertEqual(units, "meters")
        with self.assertRaisesRegex(Phase2Error, "NAVD88"):
            _validated_vertical_reference(
                {"vertical_datum": "NGVD29", "vertical_units": "meters"}
            )
        with self.assertRaisesRegex(Phase2Error, "metre"):
            _validated_vertical_reference(
                {"vertical_datum": "NAVD88", "vertical_units": "feet"}
            )

    def test_eav_replay_must_match_configured_sciencebase_item(self) -> None:
        settings = SimpleNamespace(
            doi="10.5066/P14KZS2D",
            sciencebase_item_id="698226f4b66b018d00102e42",
        )
        replay = SimpleNamespace(item=SimpleNamespace(item_id="different-item"))
        with (
            patch(
                "regional_analysis.hydrometry.phase2_bear_lake.acquire_bear_lake_eav",
                return_value=replay,
            ),
            self.assertRaisesRegex(Phase2Error, "configured item ID"),
        ):
            load_bear_eav_source(settings)


class Phase2ScienceBaseTransportTests(unittest.TestCase):
    @staticmethod
    def _source() -> ScienceBaseFile:
        return ScienceBaseFile(
            name="Bear_Lake_TBDEM_2026_50cm.zip",
            content_type="application/zip",
            size_bytes=1_339_051_329,
            download_url=(
                "https://sciencebase.usgs.gov/manager/download/"
                "cmla2h2m7009d0uob0liecbur"
            ),
            checksum_algorithm=None,
            checksum_value=None,
            cuid="cmla2h2m7009d0uob0liecbur",
            path_on_disk="__s3__",
            s3_download_request_page_url=(
                "https://www.sciencebase.gov/catalog/item/requestDownload/"
                "698226f4b66b018d00102e42?filePath=__s3__"
            ),
            storage_bucket="prod-is-usgs-sb-prod-content",
            storage_key=("698226f4b66b018d00102e42/Bear_Lake_TBDEM_2026_50cm.zip"),
        )

    def test_anonymous_catalog_transport_is_exact_and_needs_no_login(self) -> None:
        source = self._source()
        expected = (
            "https://www.sciencebase.gov/catalog/file/get/"
            "698226f4b66b018d00102e42?name=Bear_Lake_TBDEM_2026_50cm.zip"
        )
        self.assertEqual(
            _sciencebase_catalog_transport_url("698226f4b66b018d00102e42", source),
            expected,
        )
        self.assertEqual(
            _sciencebase_archive_transport_url("698226f4b66b018d00102e42", source),
            expected,
        )

    def test_catalog_transport_requires_exact_provider_key(self) -> None:
        for storage_key in (
            None,
            "other-item/Bear_Lake_TBDEM_2026_50cm.zip",
        ):
            with self.subTest(storage_key=storage_key):
                source = replace(self._source(), storage_key=storage_key)
                with self.assertRaisesRegex(Phase2Error, "storage key"):
                    _sciencebase_catalog_transport_url(
                        "698226f4b66b018d00102e42", source
                    )

    def test_stable_offline_archive_identity_does_not_require_transport(self) -> None:
        source = self._source()
        settings = SimpleNamespace(
            data_dir=Path("ignored-phase2-data"),
            raster_archive_name=source.name,
            spatial_metadata_archive_name="metadata.zip",
        )
        eav = SimpleNamespace(item=SimpleNamespace(item_id="698226f4b66b018d00102e42"))
        selection = SimpleNamespace(raster_archive=source)
        with patch(
            "regional_analysis.hydrometry.phase2_bear_lake."
            "select_bear_lake_phase2_files",
            return_value=selection,
        ):
            request = _archive_request(settings, eav)
            self.assertEqual(
                request.url,
                "https://www.sciencebase.gov/catalog/file/get/"
                "698226f4b66b018d00102e42?name=Bear_Lake_TBDEM_2026_50cm.zip",
            )
            self.assertEqual(request.public_url, request.url)


class Phase2NumericalGateTests(unittest.TestCase):
    @staticmethod
    def _benchmark(
        resolution: float, errors: dict[str, tuple[float, float]]
    ) -> SimpleNamespace:
        metrics = tuple(
            SimpleNamespace(
                stage_id=stage_id,
                area_relative_error_vs_source=area,
                storage_relative_error_vs_source=storage,
            )
            for stage_id, (area, storage) in errors.items()
        )
        return SimpleNamespace(resolution_m=resolution, stage_metrics=metrics)

    def test_stage_specific_absolute_envelopes_are_not_averaged(self) -> None:
        records = (
            SimpleNamespace(
                validation_stage_id="low",
                absolute_area_error_m2=2.0,
                absolute_storage_error_m3=1.0,
            ),
            SimpleNamespace(
                validation_stage_id="high",
                absolute_area_error_m2=1.0,
                absolute_storage_error_m3=8.0,
            ),
        )
        envelopes = (
            {
                "validation_stage_id": "low",
                "stage_m_navd88": 1.0,
                "accepted_area_absolute_error_m2": 1.0,
                "accepted_storage_absolute_error_m3": 2.0,
            },
            {
                "validation_stage_id": "high",
                "stage_m_navd88": 2.0,
                "accepted_area_absolute_error_m2": 2.0,
                "accepted_storage_absolute_error_m3": 9.0,
            },
        )
        passed, audited = _stagewise_precision_gate(records, envelopes)
        self.assertFalse(passed)
        self.assertFalse(audited[0]["area_gate_passed"])
        self.assertTrue(audited[1]["stage_gate_passed"])

    def test_source_method_gate_preserves_numeric_area_exception(self) -> None:
        records = (
            SimpleNamespace(
                validation_stage_id="table_grid_005",
                official_area_m2=1_242_850.0,
                raster_area_m2=1_242_700.0,
                absolute_area_error_m2=150.0,
                absolute_storage_error_m3=34.3396679687,
            ),
        )
        envelopes = (
            {
                "validation_stage_id": "table_grid_005",
                "stage_m_navd88": 1749.43,
                "accepted_area_absolute_error_m2": 88.7509765626,
                "accepted_storage_absolute_error_m3": 151.725087891,
            },
        )
        exact_passed, numeric = _stagewise_precision_gate(records, envelopes)
        method_passed, audited, summary = _stagewise_source_method_consistency_gate(
            records,
            numeric,
            source_pixel_width_m=0.5,
            source_pixel_height_m=0.5,
        )

        self.assertFalse(exact_passed)
        self.assertTrue(method_passed)
        self.assertFalse(audited[0]["numeric_area_precision_gate_passed"])
        self.assertTrue(audited[0]["numeric_area_precision_exception"])
        self.assertTrue(audited[0]["source_method_area_scale_gate_passed"])
        self.assertTrue(audited[0]["source_method_consistency_gate_passed"])
        self.assertAlmostEqual(
            audited[0]["source_area_equivalent_radial_displacement_m"],
            0.037956842090466125,
        )
        self.assertAlmostEqual(
            audited[0]["native_source_cell_diagonal_m"], 0.5 * 2**0.5
        )
        self.assertEqual(summary["source_numeric_area_precision_exception_count"], 1)
        self.assertEqual(
            summary["source_numeric_area_precision_exception_stage_ids"],
            ["table_grid_005"],
        )
        self.assertFalse(summary["source_exact_numeric_gate_passed"])
        self.assertIn("not bound", summary["source_method_consistency_warning"])
        self.assertIn(
            "every source validation stage",
            audited[0]["source_method_consistency_rule"],
        )

    def test_source_method_gate_keeps_storage_precision_strict(self) -> None:
        records = (
            SimpleNamespace(
                validation_stage_id="storage_failure",
                official_area_m2=1_000_000.0,
                raster_area_m2=1_000_000.0,
                absolute_area_error_m2=0.0,
                absolute_storage_error_m3=2.0,
            ),
        )
        envelopes = (
            {
                "validation_stage_id": "storage_failure",
                "stage_m_navd88": 1800.0,
                "accepted_area_absolute_error_m2": 1.0,
                "accepted_storage_absolute_error_m3": 1.0,
            },
        )
        _, numeric = _stagewise_precision_gate(records, envelopes)
        passed, audited, summary = _stagewise_source_method_consistency_gate(
            records,
            numeric,
            source_pixel_width_m=0.5,
            source_pixel_height_m=0.5,
        )

        self.assertFalse(passed)
        self.assertTrue(audited[0]["source_method_area_scale_gate_passed"])
        self.assertFalse(audited[0]["source_method_storage_precision_gate_passed"])
        self.assertFalse(summary["source_storage_numeric_precision_gate_passed"])

    def test_source_method_area_gate_uses_inspected_pixel_dimensions(self) -> None:
        records = (
            SimpleNamespace(
                validation_stage_id="area_failure",
                official_area_m2=math.pi * 10.0**2,
                raster_area_m2=math.pi * 11.0**2,
                absolute_area_error_m2=math.pi * 21.0,
                absolute_storage_error_m3=0.0,
            ),
        )
        envelopes = (
            {
                "validation_stage_id": "area_failure",
                "stage_m_navd88": 1800.0,
                "accepted_area_absolute_error_m2": 1.0,
                "accepted_storage_absolute_error_m3": 1.0,
            },
        )
        _, numeric = _stagewise_precision_gate(records, envelopes)
        passed, audited, _ = _stagewise_source_method_consistency_gate(
            records,
            numeric,
            source_pixel_width_m=0.3,
            source_pixel_height_m=0.4,
        )

        self.assertFalse(passed)
        self.assertEqual(audited[0]["native_source_cell_diagonal_m"], 0.5)
        self.assertAlmostEqual(
            audited[0]["source_area_equivalent_radial_displacement_m"], 1.0
        )
        self.assertFalse(audited[0]["source_method_area_scale_gate_passed"])

    def test_distributional_convergence_retains_local_regressions_diagnostically(
        self,
    ) -> None:
        stage_ids = [f"table_{index:02d}" for index in range(18)] + [
            "observed_low",
            "observed_high",
        ]
        fine = {stage_id: (0.01, 0.01) for stage_id in stage_ids}
        fine["table_00"] = (0.021, 0.01)
        fine["observed_low"] = (0.005, 0.004)
        fine["observed_high"] = (0.006, 0.005)
        medium = {stage_id: (0.02, 0.02) for stage_id in stage_ids}
        medium["observed_low"] = (0.01, 0.008)
        medium["observed_high"] = (0.012, 0.009)
        coarse = {stage_id: (0.04, 0.04) for stage_id in stage_ids}
        benchmarks = {
            5.0: self._benchmark(5.0, fine),
            10.0: self._benchmark(10.0, medium),
            30.0: self._benchmark(30.0, coarse),
        }
        resolution, passed, diagnostics = _select_converged_resolution(
            benchmarks, observed_stage_ids=("observed_low", "observed_high")
        )
        self.assertEqual(resolution, 5.0)
        self.assertTrue(passed)
        self.assertEqual(len(diagnostics), 40)
        local_regression = next(
            row
            for row in diagnostics
            if row["candidate_resolution_m"] == 5.0
            and row["validation_stage_id"] == "table_00"
        )
        self.assertFalse(local_regression["area_convergence_passed"])
        self.assertTrue(local_regression["pair_distributional_convergence_passed"])
        self.assertIn("candidate_full_p95_abs_area_error", local_regression)

    def test_resolution_selection_can_reject_nonconverged_finest_grid(self) -> None:
        stage_ids = ("table", "observed_low", "observed_high")

        def constant(value: float) -> dict[str, tuple[float, float]]:
            return {stage_id: (value, value) for stage_id in stage_ids}

        benchmarks = {
            5.0: self._benchmark(5.0, constant(0.03)),
            10.0: self._benchmark(10.0, constant(0.02)),
            30.0: self._benchmark(30.0, constant(0.04)),
        }
        resolution, passed, diagnostics = _select_converged_resolution(
            benchmarks, observed_stage_ids=("observed_low", "observed_high")
        )
        self.assertEqual(resolution, 10.0)
        self.assertTrue(passed)
        self.assertEqual(len(diagnostics), 6)

    def test_resolution_selection_fails_without_convergence(self) -> None:
        errors = {
            "table": (0.03, 0.03),
            "observed_low": (0.03, 0.03),
        }
        benchmarks = {
            5.0: self._benchmark(5.0, errors),
            10.0: self._benchmark(
                10.0, {stage_id: (0.02, 0.02) for stage_id in errors}
            ),
            30.0: self._benchmark(
                30.0, {stage_id: (0.01, 0.01) for stage_id in errors}
            ),
        }
        resolution, passed, _ = _select_converged_resolution(
            benchmarks, observed_stage_ids=("observed_low",)
        )
        self.assertIsNone(resolution)
        self.assertFalse(passed)

    def test_resolution_selection_requires_fixed_observed_stages(self) -> None:
        errors = {"table": (0.01, 0.01)}
        benchmarks = {
            5.0: self._benchmark(5.0, errors),
            10.0: self._benchmark(10.0, errors),
        }
        with self.assertRaisesRegex(Phase2Error, "fixed observed stages"):
            _select_converged_resolution(benchmarks, observed_stage_ids=())

    def test_connected_basin_gate_uses_same_grid_physical_scales(self) -> None:
        evidence = _assess_connected_basin_stage(
            connected_area_m2=990_000.0,
            connected_storage_m3=9_950_000.0,
            all_valid_area_m2=1_000_000.0,
            all_valid_storage_m3=10_000_000.0,
            selected_grid_cell_diagonal_m=10.0,
        )
        self.assertTrue(evidence["area_scale_gate_passed"])
        self.assertTrue(evidence["storage_shallowness_gate_passed"])
        self.assertTrue(evidence["basin_stage_gate_passed"])
        self.assertAlmostEqual(
            evidence["absolute_area_relative_difference_vs_same_grid_all_valid"],
            0.01,
        )
        self.assertAlmostEqual(
            evidence["absolute_storage_relative_difference_vs_same_grid_all_valid"],
            0.005,
        )

    def test_connected_basin_gate_rejects_deep_or_spatially_large_exclusions(
        self,
    ) -> None:
        deep = _assess_connected_basin_stage(
            connected_area_m2=990_000.0,
            connected_storage_m3=9_000_000.0,
            all_valid_area_m2=1_000_000.0,
            all_valid_storage_m3=10_000_000.0,
            selected_grid_cell_diagonal_m=10.0,
        )
        self.assertTrue(deep["area_scale_gate_passed"])
        self.assertFalse(deep["storage_shallowness_gate_passed"])
        self.assertFalse(deep["basin_stage_gate_passed"])

        spatially_large = _assess_connected_basin_stage(
            connected_area_m2=900_000.0,
            connected_storage_m3=9_500_000.0,
            all_valid_area_m2=1_000_000.0,
            all_valid_storage_m3=10_000_000.0,
            selected_grid_cell_diagonal_m=10.0,
        )
        self.assertFalse(spatially_large["area_scale_gate_passed"])
        self.assertTrue(spatially_large["storage_shallowness_gate_passed"])
        self.assertFalse(spatially_large["basin_stage_gate_passed"])

    def test_connected_basin_gate_rejects_invalid_reference_scalars(self) -> None:
        with self.assertRaisesRegex(Phase2Error, "positive all-valid"):
            _assess_connected_basin_stage(
                connected_area_m2=0.0,
                connected_storage_m3=0.0,
                all_valid_area_m2=0.0,
                all_valid_storage_m3=0.0,
                selected_grid_cell_diagonal_m=5.0,
            )

    def test_unique_index_retains_duplicate_hard_stop(self) -> None:
        indexed = _set_unique_index(
            pd.DataFrame({"state_id": ["low", "high"], "value": [1, 2]}),
            "state_id",
            context="Synthetic table",
        )
        self.assertTrue(indexed.index.is_unique)
        with self.assertRaisesRegex(Phase2Error, "duplicate 'state_id'"):
            _set_unique_index(
                pd.DataFrame({"state_id": ["low", "low"]}),
                "state_id",
                context="Synthetic table",
            )


class Phase2ObservedStateSelectionTests(unittest.TestCase):
    def _frame(self, count: int = 900) -> pd.DataFrame:
        start = date(2020, 1, 1)
        rows = []
        for index in range(count):
            rows.append(
                {
                    "calendar_date": start + timedelta(days=index),
                    "value": 5900.0 + (index % 71) / 10.0,
                    "source_record_id": "USGS-10055000",
                    "source_parameter_code": "62615",
                    "approval_status": "Approved",
                    "provisional": False,
                }
            )
        frame = pd.DataFrame(rows)
        frame.loc[123, "value"] = 5888.0
        frame.loc[678, "value"] = 5933.0
        return frame

    def test_five_states_are_real_and_one_year_uses_nearest_matching_logic(
        self,
    ) -> None:
        frame = self._frame()
        states = select_stage_states(frame, baseline_match_tolerance_days=3)
        self.assertEqual(
            [state.state_id for state in states],
            ["latest", "year_ago", "recent_low", "recent_high", "historical_median"],
        )
        observed = set(frame["calendar_date"])
        self.assertTrue(all(state.observation_date in observed for state in states))
        self.assertEqual(
            states[1].observation_date, states[0].observation_date - timedelta(days=365)
        )
        self.assertEqual(states[2].stage_ft_navd88, 5888.0)
        self.assertEqual(states[3].stage_ft_navd88, 5933.0)

    def test_animation_is_deterministic_and_includes_low_high_latest(self) -> None:
        frame = self._frame()
        first = select_animation_stages(frame, frame_count=18)
        second = select_animation_stages(
            frame.sample(frac=1, random_state=9), frame_count=18
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 18)
        self.assertEqual(
            list(first), sorted(first, key=lambda item: item.observation_date)
        )
        values = {item.stage_ft_navd88 for item in first}
        self.assertIn(5888.0, values)
        self.assertIn(5933.0, values)
        self.assertEqual(first[-1].observation_date, max(frame["calendar_date"]))


class Phase2OrchestrationGateTests(unittest.TestCase):
    @staticmethod
    def _policy() -> ValidationPolicy:
        return ValidationPolicy(
            evidence_used=("synthetic source check", "synthetic convergence check"),
            selected_spatial_resolution_m=10.0,
            source_max_area_relative_error=0.001,
            source_max_storage_relative_error=0.001,
            working_max_area_relative_error_vs_source=0.002,
            working_max_storage_relative_error_vs_source=0.002,
            accepted_source_area_relative_error=0.002,
            accepted_source_storage_relative_error=0.002,
            accepted_working_area_relative_error=0.003,
            accepted_working_storage_relative_error=0.003,
            source_resolution_residuals="synthetic residuals",
            resolution_convergence_behavior="synthetic convergence",
            accepted_error_envelope="synthetic envelope",
            scientific_rationale="Synthetic policy used to test orchestration gates.",
            official_uncertainty_supports_formal_validation=False,
            status="numerically_consistent",
            source_method_consistency_gate_passed=True,
            source_exact_numeric_gate_passed=False,
            source_storage_numeric_precision_gate_passed=True,
            source_numeric_area_precision_exception_count=1,
            source_numeric_area_precision_exception_stage_ids=("table_grid_005",),
            source_method_max_area_equivalent_radial_displacement_m=0.06,
            native_source_cell_diagonal_m=math.sqrt(0.5),
            source_method_consistency_rule=(
                "equal-area radial displacement <= one native-cell diagonal"
            ),
            source_method_consistency_warning=(
                "Equal-area radius is not a bound on local shoreline position."
            ),
            source_method_consistency_status=(
                "source_method_consistent_with_numeric_area_warning"
            ),
        )

    def _settings_and_policy(self, root: Path):
        settings = replace(
            load_phase2_settings("configs/hydrometry_pilots.yaml"), data_dir=root
        )
        policy = self._policy()
        record = {
            **asdict(policy),
            "schema_version": PHASE2_SCHEMA_VERSION,
            "validation_policy_sha256": policy.policy_sha256,
            "render_allowed": True,
        }
        (root / "validation_policy.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
        return settings, record

    def test_persisted_render_policy_is_reconstructed_and_hash_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings, record = self._settings_and_policy(root)
            policy, gate, _ = _load_render_gate(settings)
            self.assertEqual(policy.policy_sha256, record["validation_policy_sha256"])
            self.assertTrue(gate.allowed)

            record["scientific_rationale"] = "tampered"
            (root / "validation_policy.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            with self.assertRaisesRegex(Phase2Error, "policy hash"):
                _load_render_gate(settings)

    def test_persisted_policy_requires_all_source_method_evidence(self) -> None:
        required = (
            "source_method_consistency_gate_passed",
            "source_exact_numeric_gate_passed",
            "source_storage_numeric_precision_gate_passed",
            "source_numeric_area_precision_exception_count",
            "source_numeric_area_precision_exception_stage_ids",
            "source_method_max_area_equivalent_radial_displacement_m",
            "native_source_cell_diagonal_m",
            "source_method_consistency_rule",
            "source_method_consistency_warning",
            "source_method_consistency_status",
        )
        for field in required:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                settings, record = self._settings_and_policy(root)
                record.pop(field)
                (root / "validation_policy.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    Phase2Error, rf"lacks required fields:.*{field}"
                ):
                    _load_render_gate(settings)

    def test_source_method_tamper_is_rejected_by_policy_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings, record = self._settings_and_policy(root)
            record["source_method_consistency_warning"] = (
                "Tampered source-method warning."
            )
            (root / "validation_policy.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            with self.assertRaisesRegex(Phase2Error, "policy hash"):
                _load_render_gate(settings)

    def test_schema_v1_policy_is_rejected_as_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings, record = self._settings_and_policy(root)
            record["schema_version"] = "hydrometry-bear-lake-phase2-v1"
            (root / "validation_policy.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            with self.assertRaisesRegex(Phase2Error, "schema is stale"):
                _load_render_gate(settings)

    def test_render_only_requires_built_provenance_after_valid_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings, _ = self._settings_and_policy(Path(temporary))
            with self.assertRaisesRegex(Phase2Error, "build-spatial-products"):
                _render_products(settings, None, pd.DataFrame())

    def test_render_only_verifies_quantitative_hashes_before_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings, policy_record = self._settings_and_policy(root)
            provenance = {
                "validation_policy_sha256": policy_record["validation_policy_sha256"],
                "quantitative_artifact_sha256": {"missing.tif": "a" * 64},
            }
            (root / "bear_lake_phase2_provenance.json").write_text(
                json.dumps(provenance), encoding="utf-8"
            )
            with self.assertRaisesRegex(Phase2Error, "missing"):
                _render_products(settings, None, pd.DataFrame())

    def test_scalar_lookup_is_exact_and_rejects_ambiguous_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "stage_m_navd88": [1800.0, 1801.0],
                "area_m2": [10.0, 20.0],
                "storage_m3": [30.0, 40.0],
            }
        )
        self.assertEqual(_scalar_at_stage(frame, 1801.0), (20.0, 40.0))
        duplicated = pd.concat((frame, frame.iloc[[1]]), ignore_index=True)
        with self.assertRaisesRegex(Phase2Error, "found 2"):
            _scalar_at_stage(duplicated, 1801.0)

    def test_calculation_plane_reaches_selected_and_render_states(self) -> None:
        requested = 1804.12345
        calculation = float(pd.Series([requested], dtype="float32").iloc[0])
        scalar_frame = pd.DataFrame(
            {
                "stage_m_navd88": [requested],
                "calculation_stage_m_navd88": [calculation],
                "stage_plane_dtype": ["float32"],
                "area_m2": [100.0],
                "storage_m3": [200.0],
            }
        )
        self.assertEqual(
            _calculation_plane_at_stage(scalar_frame, requested),
            (calculation, "float32"),
        )
        state = SelectedStage(
            state_id="latest",
            observation_date=date(2026, 8, 12),
            stage_ft_navd88=5919.0,
            stage_m_navd88=requested,
            source_record_id="USGS-10055000",
            source_parameter_code="62615",
            approval_status="Provisional",
            provisional=True,
            quality_status="provider_quality_not_reported",
        )
        official = pd.DataFrame(
            {
                "state_id": ["latest"],
                "stage_m_navd88": [requested],
                "official_table_area_m2": [99.0],
                "official_table_storage_m3": [199.0],
            }
        )
        selected = _selected_stage_frame(
            official,
            (state,),
            scalar_frame,
            scalar_frame,
            validation_status="numerically_consistent",
        )
        self.assertEqual(selected.iloc[0]["calculation_stage_m_navd88"], calculation)
        self.assertEqual(selected.iloc[0]["stage_plane_dtype"], "float32")
        render = _render_state(
            state,
            pd.DataFrame([[True]]).to_numpy(),
            scalar_frame,
            "numerically_consistent",
            validation_detail=(
                "source_method_consistent_with_numeric_area_warning; "
                "exact numeric area exceptions=1"
            ),
        )
        self.assertEqual(render.stage_m_navd88, requested)
        self.assertEqual(render.water_surface_stage_m_navd88, calculation)
        self.assertIn("exact numeric area exceptions=1", render.validation_detail)

    def test_float32_plane_identity_survives_float64_parser_noise(self) -> None:
        effective = float(np.float32(1804.348944))
        parser_value = float(np.nextafter(effective, -math.inf))
        self.assertNotEqual(parser_value, effective)
        parsed_frame = pd.DataFrame(
            {
                "stage_m_navd88": [1804.348944],
                "calculation_stage_m_navd88": [parser_value],
                "stage_plane_dtype": ["float32"],
            }
        )
        self.assertEqual(
            _calculation_plane_at_stage(parsed_frame, 1804.348944),
            (effective, "float32"),
        )
        self.assertTrue(
            _same_calculation_plane(
                effective,
                "float32",
                parser_value,
                "float32",
            )
        )
        next_float32 = float(np.nextafter(np.float32(effective), np.float32(math.inf)))
        self.assertFalse(
            _same_calculation_plane(
                effective,
                "float32",
                next_float32,
                "float32",
            )
        )
        self.assertFalse(
            _same_calculation_plane(
                effective,
                "float32",
                effective,
                "float64",
            )
        )

    def test_validation_input_binding_rejects_stale_source(self) -> None:
        retained = {
            "source_raster": {"path": "source.tif", "sha256": "a" * 64},
            "phase1_daily_stage": {"path": "daily.csv", "sha256": "b" * 64},
        }
        _require_matching_validation_inputs(
            retained,
            retained,
            retained_sha256=_records_sha256(retained),
        )
        current = {
            **retained,
            "source_raster": {"path": "source.tif", "sha256": "c" * 64},
        }
        with self.assertRaisesRegex(Phase2Error, "stale.*source_raster"):
            _require_matching_validation_inputs(
                retained,
                current,
                retained_sha256=_records_sha256(retained),
            )

    def test_source_year_label_is_provenance_driven(self) -> None:
        self.assertEqual(
            _source_year_label(
                {"survey_years": [1999, 2007], "derived_processing_years": [2025]}
            ),
            "1999, 2007; derived processing 2025",
        )


class Phase2OutputContractTests(unittest.TestCase):
    def test_quantitative_hash_maps_separate_raw_runtime_from_science(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = SimpleNamespace(data_dir=root)
            eav = SimpleNamespace(
                source_path=root / "reference" / "official.csv",
                metadata_path=root / "reference" / "official.metadata.json",
                manifest_path=root / "reference" / "official.manifest.json",
            )
            source_raster = root / "source" / "source.tif"
            archive_manifest = root / "archive.download.json"
            selected = root / "selected_stage_states.csv"
            products = SimpleNamespace(
                rasters={"depth": root / "depth.tif"},
                geopackage=root / "vectors.gpkg",
            )
            with (
                patch(
                    "regional_analysis.hydrometry.phase2_bear_lake._source_raster_path",
                    return_value=source_raster,
                ),
                patch(
                    "regional_analysis.hydrometry.phase2_bear_lake._archive_request",
                    return_value=SimpleNamespace(manifest_path=archive_manifest),
                ),
            ):
                paths = _quantitative_paths(settings, eav, products, selected)
                for path in paths:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    key = path.resolve().relative_to(root.resolve()).as_posix()
                    if key == "source/extraction_manifest.json":
                        value = {
                            "created_at_utc": "2026-08-14T12:00:00Z",
                            "archive_sha256": "a" * 64,
                        }
                        path.write_text(json.dumps(value), encoding="utf-8")
                    elif key == "source_raster_inspection.json":
                        value = {
                            "inspection_seconds": 1.0,
                            "raster_sha256": "a" * 64,
                            "resolution_m": 0.5,
                        }
                        path.write_text(json.dumps(value), encoding="utf-8")
                    elif key == "eav_raster_validation_summary.json":
                        value = {
                            "source_processing_seconds": 1.0,
                            "total_validation_seconds": 2.0,
                            "maximum_absolute_relative_area_error": 0.001,
                        }
                        path.write_text(json.dumps(value), encoding="utf-8")
                    elif key in {
                        "source_stage_scalars.csv",
                        "working_stage_scalars.csv",
                        "resolution_benchmark.csv",
                    }:
                        path.write_text(
                            "stage_m_navd88,processing_seconds,area_m2\n"
                            "1800.0,1.0,100.0\n",
                            encoding="utf-8",
                        )
                    else:
                        path.write_bytes(f"stable:{key}".encode())

                raw_before = _quantitative_hashes(settings, eav, products, selected)
                science_before = _quantitative_scientific_hashes(
                    settings, eav, products, selected
                )
                scalar = root / "source_stage_scalars.csv"
                scalar.write_text(
                    "stage_m_navd88,processing_seconds,area_m2\n1800.0,9.0,100.0\n",
                    encoding="utf-8",
                )
                inspection = root / "source_raster_inspection.json"
                inspection.write_text(
                    json.dumps(
                        {
                            "inspection_seconds": 9.0,
                            "raster_sha256": "a" * 64,
                            "resolution_m": 0.5,
                        }
                    ),
                    encoding="utf-8",
                )
                raw_after_runtime = _quantitative_hashes(
                    settings, eav, products, selected
                )
                science_after_runtime = _quantitative_scientific_hashes(
                    settings, eav, products, selected
                )
                self.assertNotEqual(raw_before, raw_after_runtime)
                self.assertEqual(science_before, science_after_runtime)

                scalar.write_text(
                    "stage_m_navd88,processing_seconds,area_m2\n1800.0,9.0,101.0\n",
                    encoding="utf-8",
                )
                self.assertNotEqual(
                    science_before,
                    _quantitative_scientific_hashes(settings, eav, products, selected),
                )

    def test_inspection_timing_is_excluded_from_bound_and_policy_science(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase1_dir = root / "phase1"
            settings = SimpleNamespace(
                data_dir=root,
                config_path=root / "hydrometry_pilots.yaml",
            )
            eav = SimpleNamespace(
                source_path=root / "reference" / "official.csv",
                metadata_path=root / "reference" / "official.metadata.json",
                manifest_path=root / "reference" / "official.manifest.json",
            )
            archive_manifest = root / "archive.download.json"
            source_raster = root / "source" / "source.tif"
            fixture_paths = (
                settings.config_path,
                phase1_dir / "pilot_daily_hydrometry.csv",
                phase1_dir / "bear_lake_eav_normalized.csv",
                eav.source_path,
                eav.metadata_path,
                eav.manifest_path,
                root / "metadata_audit.json",
                archive_manifest,
                root / "archive_inventory.json",
                source_raster,
            )
            for path in fixture_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"stable:{path.name}".encode())
            extraction = root / "source" / "extraction_manifest.json"
            extraction.write_text(
                json.dumps(
                    {
                        "created_at_utc": "2026-08-14T12:00:00Z",
                        "archive_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            inspection = root / "source_raster_inspection.json"

            def write_inspection(seconds: float, raster_sha256: str) -> None:
                inspection.write_text(
                    json.dumps(
                        {
                            "inspection_seconds": seconds,
                            "raster_sha256": raster_sha256,
                            "resolution_m": 0.5,
                        }
                    ),
                    encoding="utf-8",
                )

            def write_policy(records: dict[str, dict[str, str]]) -> str:
                policy_path = root / "validation_policy.json"
                policy_path.write_text(
                    json.dumps(
                        {
                            "render_allowed": True,
                            "validation_inputs": records,
                            "validation_inputs_sha256": _records_sha256(records),
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                return _scientific_content_sha256(policy_path, root=root)

            patches = (
                patch(
                    "regional_analysis.hydrometry.phase2_bear_lake.DEFAULT_PHASE1_DIR",
                    phase1_dir,
                ),
                patch(
                    "regional_analysis.hydrometry.phase2_bear_lake._archive_request",
                    return_value=SimpleNamespace(manifest_path=archive_manifest),
                ),
                patch(
                    "regional_analysis.hydrometry.phase2_bear_lake._source_raster_path",
                    return_value=source_raster,
                ),
            )
            with patches[0], patches[1], patches[2]:
                write_inspection(1.25, "a" * 64)
                raw_inspection_before = hashlib.sha256(
                    inspection.read_bytes()
                ).hexdigest()
                records_before = _validation_input_records(settings, eav)
                inputs_sha_before = _records_sha256(records_before)
                policy_sha_before = write_policy(records_before)

                write_inspection(9.75, "a" * 64)
                self.assertNotEqual(
                    raw_inspection_before,
                    hashlib.sha256(inspection.read_bytes()).hexdigest(),
                )
                records_after_timing = _validation_input_records(settings, eav)
                self.assertEqual(records_before, records_after_timing)
                self.assertEqual(
                    inputs_sha_before, _records_sha256(records_after_timing)
                )
                self.assertEqual(policy_sha_before, write_policy(records_after_timing))
                self.assertEqual(
                    records_after_timing["source_raster_inspection"]["sha256_scope"],
                    PHASE2_SCIENTIFIC_HASH_CONTRACT,
                )

                write_inspection(9.75, "b" * 64)
                records_after_science = _validation_input_records(settings, eav)
                self.assertNotEqual(records_before, records_after_science)
                self.assertNotEqual(
                    inputs_sha_before, _records_sha256(records_after_science)
                )
                self.assertNotEqual(
                    policy_sha_before, write_policy(records_after_science)
                )
                with self.assertRaisesRegex(Phase2Error, "source_raster_inspection"):
                    _require_matching_validation_inputs(
                        records_before,
                        records_after_science,
                        retained_sha256=inputs_sha_before,
                    )

    def test_scientific_json_hash_excludes_only_declared_runtime_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "source" / "extraction_manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "created_at_utc": "2026-08-14T12:00:00Z",
                        "archive_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            raw_before = hashlib.sha256(path.read_bytes()).hexdigest()
            scientific_before = _scientific_content_sha256(path, root=root)

            path.write_text(
                json.dumps(
                    {
                        "created_at_utc": "2026-08-14T13:00:00Z",
                        "archive_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            self.assertNotEqual(
                raw_before, hashlib.sha256(path.read_bytes()).hexdigest()
            )
            self.assertEqual(
                scientific_before, _scientific_content_sha256(path, root=root)
            )

            path.write_text(
                json.dumps(
                    {
                        "created_at_utc": "2026-08-14T13:00:00Z",
                        "archive_sha256": "b" * 64,
                    }
                ),
                encoding="utf-8",
            )
            self.assertNotEqual(
                scientific_before, _scientific_content_sha256(path, root=root)
            )

    def test_scientific_csv_hash_excludes_runtime_but_not_measurements(self) -> None:
        for relative in (
            "source_stage_scalars.csv",
            "working_stage_scalars.csv",
            "resolution_benchmark.csv",
        ):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path = root / relative
                path.write_text(
                    "stage_m_navd88,processing_seconds,area_m2\n1800.0,1.25,100.0\n",
                    encoding="utf-8",
                )
                raw_before = hashlib.sha256(path.read_bytes()).hexdigest()
                scientific_before = _scientific_content_sha256(path, root=root)

                path.write_text(
                    "stage_m_navd88,processing_seconds,area_m2\n1800.0,9.75,100.0\n",
                    encoding="utf-8",
                )
                self.assertNotEqual(
                    raw_before, hashlib.sha256(path.read_bytes()).hexdigest()
                )
                self.assertEqual(
                    scientific_before,
                    _scientific_content_sha256(path, root=root),
                )

                path.write_text(
                    "stage_m_navd88,processing_seconds,area_m2\n1800.0,9.75,101.0\n",
                    encoding="utf-8",
                )
                self.assertNotEqual(
                    scientific_before,
                    _scientific_content_sha256(path, root=root),
                )

    def test_scientific_hash_allowlist_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            declared = root / "eav_raster_validation_summary.json"
            declared.write_text(
                json.dumps({"source_processing_seconds": 1.0}), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                Phase2Error, "runtime field is missing: total_validation_seconds"
            ):
                _scientific_content_sha256(declared, root=root)

            inspection = root / "source_raster_inspection.json"
            inspection.write_text(
                json.dumps({"raster_sha256": "a" * 64}), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                Phase2Error, "runtime field is missing: inspection_seconds"
            ):
                _scientific_content_sha256(inspection, root=root)

            ordinary = root / "ordinary.json"
            ordinary.write_text(
                json.dumps({"elapsed_seconds": 1.0, "area_m2": 100.0}),
                encoding="utf-8",
            )
            before = _scientific_content_sha256(ordinary, root=root)
            ordinary.write_text(
                json.dumps({"elapsed_seconds": 2.0, "area_m2": 100.0}),
                encoding="utf-8",
            )
            self.assertNotEqual(before, _scientific_content_sha256(ordinary, root=root))

    def test_scientific_hash_contract_retains_raw_render_guard(self) -> None:
        contract = _scientific_hash_contract()
        self.assertEqual(contract["contract_version"], PHASE2_SCIENTIFIC_HASH_CONTRACT)
        self.assertEqual(
            contract["json_runtime_fields_excluded"]["source_raster_inspection.json"],
            ["inspection_seconds"],
        )
        self.assertEqual(contract["raw_hash_field"], "quantitative_artifact_sha256")
        self.assertEqual(
            contract["render_tamper_guard_hash_field"],
            "quantitative_artifact_sha256",
        )
        self.assertEqual(
            contract["scientific_replay_hash_field"],
            "quantitative_scientific_content_sha256",
        )

    def test_phase2_schema_v2_retains_source_method_policy_evidence(self) -> None:
        self.assertEqual(PHASE2_SCHEMA_VERSION, "hydrometry-bear-lake-phase2-v2")
        module = Path("regional_analysis/hydrometry/phase2_bear_lake.py").read_text(
            encoding="utf-8"
        )
        for field in (
            "source_method_consistency_status",
            "source_exact_numeric_gate_passed",
            "source_numeric_area_precision_exception_count",
            "source_numeric_area_precision_exception_stage_ids",
            "source_method_consistency_warning",
            "validation_detail",
        ):
            self.assertIn(f'"{field}"', module)

    def test_selected_stage_and_validation_schemas_are_explicit(self) -> None:
        module = Path("regional_analysis/hydrometry/phase2_bear_lake.py").read_text(
            encoding="utf-8"
        )
        for field in (
            "source_resolution_raster_area_m2",
            "source_resolution_raster_storage_m3",
            "working_grid_area_m2",
            "working_grid_storage_m3",
            "validation_status",
            "official_area_m2",
            "raster_area_m2",
            "official_storage_m3",
            "raster_storage_m3",
        ):
            self.assertIn(f'"{field}"', module)

    def test_provenance_contract_includes_source_hashes_and_survey_years(self) -> None:
        module = Path("regional_analysis/hydrometry/phase2_bear_lake.py").read_text(
            encoding="utf-8"
        )
        for field in (
            "run_mode",
            "source_archive_filename",
            "source_archive_sha256",
            "source_raster_internal_filename",
            "source_raster_sha256",
            "official_eav_sha256",
            "validation_policy_sha256",
            "survey_years",
            "quantitative_artifact_sha256",
            "quantitative_scientific_content_sha256",
            "quantitative_scientific_hash_contract",
            "render_artifact_sha256",
        ):
            self.assertIn(f'"{field}"', module)

    def test_phase2_modules_do_not_import_qgis_or_earth_engine(self) -> None:
        phase2_files = (
            "binary_download.py",
            "topobathymetry.py",
            "raster_storage.py",
            "shoreline.py",
            "spatial_products.py",
            "render_3d.py",
            "animate_lake.py",
            "plot_phase2.py",
            "phase2_bear_lake.py",
            "phase2_metadata.py",
        )
        prohibited = re.compile(
            r"(?:^|\s)(?:from|import)\s+(?:qgis|ee)(?:\.|\s|$)|"
            r"ee\.(?:Initialize|Authenticate)|Export\."
        )
        for name in phase2_files:
            with self.subTest(name=name):
                text = Path("regional_analysis/hydrometry", name).read_text(
                    encoding="utf-8"
                )
                self.assertIsNone(prohibited.search(text))


if __name__ == "__main__":
    unittest.main()
