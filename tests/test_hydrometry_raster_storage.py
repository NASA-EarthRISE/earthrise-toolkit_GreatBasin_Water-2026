from __future__ import annotations

import hashlib
import json
import math
import unittest
from datetime import date

import numpy as np

from regional_analysis.hydrometry.raster_storage import (
    ConnectivityError,
    ElevationWindow,
    OfficialEavPoint,
    RasterScalarResult,
    RasterStorageError,
    ResolutionBenchmark,
    ValidationPolicyError,
    average_aggregate,
    build_eav_validation_records,
    build_elevation_histogram,
    build_resolution_benchmark,
    compute_connected_inundation,
    compute_stage_raster,
    evaluate_elevation_histogram,
    evaluate_render_gate,
    evaluate_validation_policy,
    integrate_elevation_windows,
    integrate_threshold_event_windows,
    summarize_eav_validation,
)


def scalar(
    stage: float,
    area: float,
    storage: float,
    *,
    resolution: float = 0.5,
) -> RasterScalarResult:
    return RasterScalarResult(
        stage_m_navd88=stage,
        area_m2=area,
        storage_m3=storage,
        flooded_cell_count=int(area / (resolution * resolution)),
        valid_cell_count=1_000,
        cell_area_m2=resolution * resolution,
        calculation_method="direct_windowed_float64",
        basin_method="connected_seed",
        source_resolution_m=0.5,
        working_resolution_m=resolution,
        processing_seconds=0.1,
        peak_memory_estimate_bytes=1_024,
    )


class SyntheticRasterIntegrationTests(unittest.TestCase):
    def test_flat_bottom_rectangular_pixels_have_exact_area_and_volume(self) -> None:
        result = compute_stage_raster(
            np.zeros((3, 4)),
            2.0,
            pixel_width_m=2.0,
            pixel_height_m=3.0,
            basin_method="all_valid_below_stage",
        )
        self.assertEqual(result.area_m2, 72.0)
        self.assertEqual(result.storage_m3, 144.0)
        self.assertEqual(result.depth_m[0, 0], 2.0)

    def test_sloped_basin_and_bowl_match_direct_analytical_sum(self) -> None:
        slope = np.array([[0.0, 1.0], [2.0, 3.0]])
        result = compute_stage_raster(
            slope,
            2.5,
            pixel_width_m=1.0,
            pixel_height_m=1.0,
            basin_method="all_valid_below_stage",
        )
        self.assertEqual(result.area_m2, 3.0)
        self.assertEqual(result.storage_m3, 4.5)

        rows, columns = np.indices((9, 9))
        bowl = ((rows - 4) ** 2 + (columns - 4) ** 2).astype(np.float64)
        stage = 10.0
        expected = bowl <= stage
        result = compute_stage_raster(
            bowl,
            stage,
            pixel_width_m=2.0,
            pixel_height_m=2.0,
            basin_mask=np.ones_like(bowl, dtype=bool),
            basin_method="connected_seed",
            trusted_seed=(4, 4),
        )
        self.assertEqual(result.area_m2, np.count_nonzero(expected) * 4.0)
        self.assertEqual(
            result.storage_m3,
            float(np.sum(stage - bowl[expected], dtype=np.float64) * 4.0),
        )

    def test_float32_stage_plane_is_inclusive_and_explicit(self) -> None:
        requested = 1749.03
        effective = float(np.float32(requested))
        elevation = np.asarray([[effective, effective + 0.01]], dtype=np.float32)
        result = compute_stage_raster(
            elevation,
            requested,
            pixel_width_m=1.0,
            pixel_height_m=1.0,
            basin_method="all_valid_below_stage",
        )
        self.assertEqual(result.flooded_cell_count, 1)
        self.assertEqual(result.storage_m3, 0.0)
        self.assertEqual(result.stage_m_navd88, requested)
        self.assertEqual(result.calculation_stage_m_navd88, effective)
        self.assertEqual(result.stage_plane_dtype, "float32")
        self.assertIn("requested_stage_aligned_to_float32_plane", result.warning_flags)

    def test_integer_bed_does_not_truncate_continuous_stage(self) -> None:
        result = compute_stage_raster(
            np.asarray([[0, 1]], dtype=np.int16),
            0.5,
            pixel_width_m=1.0,
            pixel_height_m=1.0,
            basin_method="all_valid_below_stage",
        )
        self.assertEqual(result.flooded_cell_count, 1)
        self.assertEqual(result.calculation_stage_m_navd88, 0.5)
        self.assertEqual(result.stage_plane_dtype, "float64")

    def test_island_and_nodata_hole_remain_dry_or_unsupported(self) -> None:
        elevation = np.zeros((5, 5), dtype=np.float64)
        elevation[2, 2] = 10.0
        valid = np.ones_like(elevation, dtype=bool)
        valid[1, 1] = False
        result = compute_stage_raster(
            elevation,
            5.0,
            pixel_width_m=1.0,
            pixel_height_m=1.0,
            valid_mask=valid,
            basin_mask=np.ones_like(valid),
            trusted_seed=(0, 0),
        )
        self.assertFalse(result.inundation[2, 2])
        self.assertEqual(result.depth_m[2, 2], 0.0)
        self.assertFalse(result.inundation[1, 1])
        self.assertTrue(math.isnan(result.depth_m[1, 1]))
        self.assertEqual(result.area_m2, 23.0)
        self.assertEqual(result.storage_m3, 115.0)

    def test_disconnected_depression_is_excluded_and_connection_is_stage_dependent(
        self,
    ) -> None:
        elevation = np.array([[0.0, 2.0, 0.0]])
        kwargs = {
            "pixel_width_m": 1.0,
            "pixel_height_m": 1.0,
            "basin_mask": np.ones_like(elevation, dtype=bool),
            "trusted_seed": (0, 0),
            "connectivity": 8,
        }
        low = compute_stage_raster(elevation, 1.0, **kwargs)
        high = compute_stage_raster(elevation, 3.0, **kwargs)
        self.assertEqual(low.flooded_cell_count, 1)
        self.assertEqual(high.flooded_cell_count, 3)

    def test_four_and_eight_neighbour_connectivity_differ(self) -> None:
        flooded = np.eye(3, dtype=bool)
        four = compute_connected_inundation(
            flooded, trusted_seed=(0, 0), connectivity=4
        )
        eight = compute_connected_inundation(
            flooded, trusted_seed=(0, 0), connectivity=8
        )
        self.assertEqual(np.count_nonzero(four), 1)
        self.assertEqual(np.count_nonzero(eight), 3)
        with self.assertRaises(ConnectivityError):
            compute_connected_inundation(flooded, trusted_seed=(0, 1), connectivity=8)

    def test_null_range_and_datum_gates_do_not_extrapolate(self) -> None:
        elevation = np.zeros((2, 2))
        common = {
            "pixel_width_m": 1.0,
            "pixel_height_m": 1.0,
            "basin_method": "all_valid_below_stage",
        }
        null = compute_stage_raster(elevation, None, **common)
        self.assertIsNone(null.area_m2)
        self.assertEqual(null.status, "unavailable_null_stage")
        outside = compute_stage_raster(
            elevation, 11.0, supported_stage_range_m=(0.0, 10.0), **common
        )
        self.assertIsNone(outside.storage_m3)
        self.assertIn("no_extrapolation", outside.warning_flags)
        below = compute_stage_raster(elevation, -1.0, **common)
        self.assertEqual(below.area_m2, 0.0)
        self.assertEqual(below.storage_m3, 0.0)
        with self.assertRaisesRegex(RasterStorageError, "datums"):
            compute_stage_raster(
                elevation,
                1.0,
                stage_vertical_datum="NGVD29",
                raster_vertical_datum="NAVD88",
                **common,
            )

    def test_output_is_deterministic(self) -> None:
        elevation = np.arange(25, dtype=np.float64).reshape(5, 5)
        kwargs = {
            "pixel_width_m": 1.0,
            "pixel_height_m": 1.0,
            "basin_mask": np.ones_like(elevation, dtype=bool),
            "trusted_seed": (0, 0),
        }
        first = compute_stage_raster(elevation, 12.5, **kwargs)
        second = compute_stage_raster(elevation, 12.5, **kwargs)
        np.testing.assert_array_equal(first.inundation, second.inundation)
        np.testing.assert_array_equal(first.depth_m, second.depth_m)
        self.assertEqual(first.scalar_dict(), second.scalar_dict())


class WindowedAndHistogramTests(unittest.TestCase):
    def test_threshold_events_match_direct_for_unsorted_arbitrary_stages(self) -> None:
        elevation = np.array([[0.0, 0.15, 0.2], [0.4, 0.9, 1.2]], dtype=np.float64)
        windows = (
            ElevationWindow(elevation[:, :2], np.ones((2, 2), dtype=bool)),
            ElevationWindow(elevation[:, 2:], np.ones((2, 1), dtype=bool)),
        )
        stages = (0.4, 0.16, 1.0)
        results = integrate_threshold_event_windows(
            windows,
            stages,
            pixel_width_m=2.0,
            pixel_height_m=3.0,
            basin_method="all_valid_below_stage",
        )
        self.assertEqual(tuple(result.stage_m_navd88 for result in results), stages)
        for result in results:
            flooded = elevation <= result.stage_m_navd88
            self.assertEqual(result.area_m2, np.count_nonzero(flooded) * 6.0)
            self.assertAlmostEqual(
                result.storage_m3,
                float(np.sum(result.stage_m_navd88 - elevation[flooded]) * 6.0),
            )
        # Equality follows the official raster stage-plane convention.
        self.assertEqual(results[0].flooded_cell_count, 4)

    def test_direct_and_event_engines_share_float32_effective_plane(self) -> None:
        requested = 1749.03
        effective = float(np.float32(requested))
        elevation = np.asarray(
            [[effective - 0.01, effective, effective + 0.01]], dtype=np.float32
        )
        direct = integrate_elevation_windows(
            (ElevationWindow(elevation, np.ones_like(elevation, dtype=bool)),),
            (requested,),
            pixel_width_m=2.0,
            pixel_height_m=3.0,
            basin_method="all_valid_below_stage",
        )[0]
        event = integrate_threshold_event_windows(
            (ElevationWindow(elevation, np.ones_like(elevation, dtype=bool)),),
            (requested,),
            pixel_width_m=2.0,
            pixel_height_m=3.0,
            basin_method="all_valid_below_stage",
        )[0]
        self.assertEqual(direct.flooded_cell_count, 2)
        self.assertEqual(direct.area_m2, 12.0)
        expected_storage = (effective - float(elevation[0, 0])) * 6.0
        self.assertEqual(direct.storage_m3, expected_storage)
        self.assertEqual(direct.area_m2, event.area_m2)
        self.assertAlmostEqual(direct.storage_m3, event.storage_m3)
        self.assertEqual(direct.calculation_stage_m_navd88, effective)
        self.assertEqual(event.calculation_stage_m_navd88, effective)
        self.assertEqual(event.as_dict()["requested_stage_m_navd88"], requested)

    def test_windowed_direct_sum_matches_whole_array(self) -> None:
        elevation = np.arange(16, dtype=np.float64).reshape(4, 4)
        windows = (
            ElevationWindow(elevation[:2], np.ones((2, 4), dtype=bool)),
            ElevationWindow(elevation[2:], np.ones((2, 4), dtype=bool)),
        )
        results = integrate_elevation_windows(
            windows,
            (4.5, 10.5),
            pixel_width_m=2.0,
            pixel_height_m=3.0,
            basin_method="all_valid_below_stage",
        )
        for result in results:
            flooded = elevation < result.stage_m_navd88
            self.assertEqual(result.area_m2, np.count_nonzero(flooded) * 6.0)
            self.assertEqual(
                result.storage_m3,
                float(np.sum(result.stage_m_navd88 - elevation[flooded]) * 6.0),
            )
        self.assertLess(results[0].peak_memory_estimate_bytes, elevation.nbytes * 4)

    def test_connected_windowed_sum_requires_global_component_masks(self) -> None:
        elevation = np.zeros((2, 2))
        with self.assertRaisesRegex(RasterStorageError, "globally"):
            integrate_elevation_windows(
                (ElevationWindow(elevation, np.ones_like(elevation, dtype=bool)),),
                (1.0,),
                pixel_width_m=1.0,
                pixel_height_m=1.0,
                basin_method="connected_seed",
            )
        mask = np.array([[True, False], [True, False]])
        result = integrate_elevation_windows(
            (
                ElevationWindow(
                    elevation,
                    np.ones_like(elevation, dtype=bool),
                    np.ones_like(elevation, dtype=bool),
                    {1.0: mask},
                ),
            ),
            (1.0,),
            pixel_width_m=1.0,
            pixel_height_m=1.0,
            basin_method="connected_seed",
        )[0]
        self.assertEqual(result.area_m2, 2.0)
        self.assertEqual(result.storage_m3, 2.0)

    def test_histogram_bin_edges_and_float64_sum_match_direct(self) -> None:
        elevation = np.array([[0.0, 0.1, 0.19, 0.2]], dtype=np.float64)
        histogram = build_elevation_histogram(
            (ElevationWindow(elevation, np.ones_like(elevation, dtype=bool)),),
            bin_origin_m=0.0,
            bin_width_m=0.1,
            bin_count=3,
            cell_area_m2=2.0,
        )
        result = evaluate_elevation_histogram(
            histogram,
            0.2,
            source_resolution_m=math.sqrt(2),
            basin_method="fixed_basin_below_stage",
        )
        flooded = elevation <= 0.2
        self.assertEqual(result.flooded_cell_count, 4)
        self.assertEqual(result.area_m2, 8.0)
        self.assertAlmostEqual(
            result.storage_m3,
            float(np.sum(0.2 - elevation[flooded], dtype=np.float64) * 2.0),
        )
        self.assertEqual(histogram.counts.dtype, np.dtype("int64"))
        self.assertEqual(histogram.elevation_sums_m.dtype, np.dtype("float64"))
        self.assertIn("(left,right]", histogram.bin_semantics)
        with self.assertRaisesRegex(RasterStorageError, "aligned bin edge"):
            evaluate_elevation_histogram(
                histogram,
                0.15,
                source_resolution_m=1.0,
                basin_method="fixed_basin_below_stage",
            )

    def test_histogram_retains_float32_effective_stage_plane(self) -> None:
        requested = 1749.03
        effective = float(np.float32(requested))
        elevation = np.asarray(
            [[effective, float(np.float32(1749.04))]], dtype=np.float32
        )
        histogram = build_elevation_histogram(
            (ElevationWindow(elevation, np.ones_like(elevation, dtype=bool)),),
            bin_origin_m=requested,
            bin_width_m=0.01,
            bin_count=1,
            cell_area_m2=0.25,
        )
        result = evaluate_elevation_histogram(
            histogram,
            requested,
            source_resolution_m=0.5,
            basin_method="all_valid_below_stage",
        )
        self.assertEqual(result.area_m2, 0.25)
        self.assertEqual(result.storage_m3, 0.0)
        self.assertEqual(result.calculation_stage_m_navd88, effective)
        self.assertEqual(histogram.stage_plane_dtype, "float32")

    def test_average_aggregation_uses_valid_coverage_not_nearest(self) -> None:
        elevation = np.arange(16, dtype=np.float64).reshape(4, 4)
        valid = np.ones_like(elevation, dtype=bool)
        valid[0, 0] = False
        aggregated, output_valid, coverage = average_aggregate(
            elevation,
            valid,
            row_factor=2,
            column_factor=2,
            minimum_valid_fraction=0.75,
        )
        self.assertEqual(aggregated.shape, (2, 2))
        self.assertAlmostEqual(aggregated[0, 0], (1 + 4 + 5) / 3)
        self.assertTrue(output_valid[0, 0])
        self.assertEqual(coverage[0, 0], 0.75)


class ValidationAndResolutionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.official = {
            "low": OfficialEavPoint("low", 10.0, 100.0, 1_000.0, date(2020, 1, 1)),
            "high": OfficialEavPoint("high", 20.0, 200.0, 3_000.0),
        }
        self.source = {
            "low": scalar(10.0, 101.0, 1_010.0),
            "high": scalar(20.0, 198.0, 2_970.0),
        }
        self.candidate = {
            "low": scalar(10.0, 100.0, 1_020.0, resolution=10.0),
            "high": scalar(20.0, 200.0, 3_000.0, resolution=10.0),
        }

    def records(self):
        return build_eav_validation_records(
            self.official.values(), self.source, validation_status="calculated"
        )

    def benchmark(self):
        return build_resolution_benchmark(
            resolution_m=10.0,
            width=20,
            height=10,
            candidate_results=self.candidate,
            source_results=self.source,
            official_points=self.official,
            processing_seconds=0.5,
            peak_memory_estimate_bytes=10_000,
            shoreline_displacement_by_stage_m={"low": 2.0, "high": 3.0},
        )

    def test_validation_records_and_summary_retain_signed_errors(self) -> None:
        records = self.records()
        self.assertEqual(records[0].stage_date, "2020-01-01")
        self.assertAlmostEqual(records[0].relative_area_error, 0.01)
        self.assertAlmostEqual(records[1].relative_area_error, -0.01)
        summary = summarize_eav_validation(records)
        self.assertAlmostEqual(summary.maximum_absolute_relative_area_error, 0.01)
        self.assertEqual(summary.systematic_bias_diagnostics, "mixed_signed_residuals")

    def test_resolution_benchmark_retains_source_and_rejects_rendering_role(
        self,
    ) -> None:
        benchmark = self.benchmark()
        self.assertEqual(benchmark.grid_role, "quantitative_candidate")
        self.assertEqual(benchmark.cell_count, 200)
        self.assertEqual(benchmark.stage_metrics[0].stage_id, "high")
        self.assertAlmostEqual(
            benchmark.stage_metrics[0].area_relative_error_vs_next_finer,
            (200.0 - 198.0) / 198.0,
        )
        with self.assertRaisesRegex(RasterStorageError, "rendering grid"):
            ResolutionBenchmark(
                resolution_m=30.0,
                width=2,
                height=2,
                cell_count=4,
                file_size_bytes=None,
                processing_seconds=0.1,
                peak_memory_estimate_bytes=100,
                resampling_method="average",
                grid_role="rendering",
                stage_metrics=(),
            )

    def test_empty_or_undefined_errors_cannot_pass_the_gate(self) -> None:
        with self.assertRaisesRegex(RasterStorageError, "at least one"):
            ResolutionBenchmark(
                resolution_m=10.0,
                width=2,
                height=2,
                cell_count=4,
                file_size_bytes=None,
                processing_seconds=0.1,
                peak_memory_estimate_bytes=100,
                resampling_method="average",
                grid_role="quantitative_candidate",
                stage_metrics=(),
            )
        zero_official = OfficialEavPoint("zero", 0.0, 0.0, 0.0)
        undefined = build_eav_validation_records(
            (zero_official,), {"zero": scalar(0.0, 1.0, 1.0)}
        )
        with self.assertRaisesRegex(ValidationPolicyError, "finite relative"):
            evaluate_validation_policy(
                undefined,
                self.benchmark(),
                accepted_source_area_relative_error=1.0,
                accepted_source_storage_relative_error=1.0,
                accepted_working_area_relative_error=1.0,
                accepted_working_storage_relative_error=1.0,
                evidence_used=("synthetic zero denominator",),
                scientific_rationale="Undefined relative errors cannot pass.",
                official_uncertainty_supports_formal_validation=False,
                resolution_convergence_behavior="not applicable",
            )

    def test_numerically_consistent_and_validated_are_distinct(self) -> None:
        arguments = {
            "accepted_source_area_relative_error": 0.02,
            "accepted_source_storage_relative_error": 0.02,
            "accepted_working_area_relative_error": 0.03,
            "accepted_working_storage_relative_error": 0.03,
            "evidence_used": ("official E-A-V precision", "resolution convergence"),
            "scientific_rationale": "Envelope reflects measured table and grid precision.",
            "resolution_convergence_behavior": "10 m agrees with source and 5 m",
        }
        numerical = evaluate_validation_policy(
            self.records(),
            self.benchmark(),
            official_uncertainty_supports_formal_validation=False,
            **arguments,
        )
        self.assertEqual(numerical.status, "numerically_consistent")
        self.assertTrue(evaluate_render_gate(numerical).allowed)
        validated = evaluate_validation_policy(
            self.records(),
            self.benchmark(),
            official_uncertainty_supports_formal_validation=True,
            **arguments,
        )
        self.assertEqual(validated.status, "validated")
        self.assertEqual(len(validated.policy_sha256), 64)

    def test_failed_source_or_working_gate_blocks_rendering(self) -> None:
        policy = evaluate_validation_policy(
            self.records(),
            self.benchmark(),
            accepted_source_area_relative_error=0.001,
            accepted_source_storage_relative_error=0.001,
            accepted_working_area_relative_error=0.001,
            accepted_working_storage_relative_error=0.001,
            evidence_used=("deliberately strict synthetic gate",),
            scientific_rationale="Exercise hard-stop behavior.",
            official_uncertainty_supports_formal_validation=False,
            resolution_convergence_behavior="failed synthetic benchmark",
        )
        gate = evaluate_render_gate(policy)
        self.assertEqual(policy.status, "failed")
        self.assertFalse(gate.allowed)
        self.assertIn("blocked", gate.reason)

    def test_stagewise_or_convergence_precondition_blocks_rendering(self) -> None:
        arguments = {
            "accepted_source_area_relative_error": 1.0,
            "accepted_source_storage_relative_error": 1.0,
            "accepted_working_area_relative_error": 1.0,
            "accepted_working_storage_relative_error": 1.0,
            "evidence_used": ("synthetic precondition",),
            "scientific_rationale": "Exercise independent scientific gates.",
            "official_uncertainty_supports_formal_validation": False,
            "resolution_convergence_behavior": "synthetic",
        }
        source_failed = evaluate_validation_policy(
            self.records(),
            self.benchmark(),
            stagewise_source_gate_passed=False,
            **arguments,
        )
        self.assertEqual(source_failed.status, "failed")
        self.assertFalse(source_failed.stagewise_source_gate_passed)
        convergence_failed = evaluate_validation_policy(
            self.records(),
            self.benchmark(),
            resolution_convergence_gate_passed=False,
            **arguments,
        )
        self.assertEqual(convergence_failed.status, "failed")
        self.assertFalse(convergence_failed.resolution_convergence_gate_passed)

    def test_source_method_evidence_is_validated_hashed_and_exposed_by_gate(
        self,
    ) -> None:
        arguments = {
            "accepted_source_area_relative_error": 0.02,
            "accepted_source_storage_relative_error": 0.02,
            "accepted_working_area_relative_error": 0.03,
            "accepted_working_storage_relative_error": 0.03,
            "evidence_used": ("official source", "native-grid method audit"),
            "scientific_rationale": "Retain exact and spatial-scale evidence.",
            "official_uncertainty_supports_formal_validation": False,
            "resolution_convergence_behavior": "synthetic convergence",
            "source_method_consistency_gate_passed": True,
            "source_exact_numeric_gate_passed": False,
            "source_storage_numeric_precision_gate_passed": True,
            "source_numeric_area_precision_exception_count": 1,
            "source_numeric_area_precision_exception_stage_ids": ["low"],
            "source_method_max_area_equivalent_radial_displacement_m": 0.004,
            "native_source_cell_diagonal_m": math.sqrt(0.5),
            "source_method_consistency_rule": (
                "equal-area radial displacement <= one native-cell diagonal"
            ),
            "source_method_consistency_warning": (
                "Equal-area radius is not a bound on local shoreline position."
            ),
            "source_method_consistency_status": (
                "source_method_consistent_with_numeric_area_warning"
            ),
        }
        policy = evaluate_validation_policy(
            self.records(), self.benchmark(), **arguments
        )
        self.assertEqual(
            policy.source_numeric_area_precision_exception_stage_ids, ("low",)
        )
        self.assertEqual(len(policy.policy_sha256), 64)
        hashed_source_method_fields = (
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
        serialized = policy.as_dict()
        canonical = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
        self.assertEqual(
            policy.policy_sha256,
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )
        for field in hashed_source_method_fields:
            with self.subTest(hashed_field=field):
                changed_payload = dict(serialized)
                value = changed_payload[field]
                if isinstance(value, bool):
                    changed_payload[field] = not value
                elif isinstance(value, int):
                    changed_payload[field] = value + 1
                elif isinstance(value, float):
                    changed_payload[field] = value + 0.001
                elif isinstance(value, tuple):
                    changed_payload[field] = (*value, "tampered")
                else:
                    changed_payload[field] = f"{value} tampered"
                changed_canonical = json.dumps(
                    changed_payload, sort_keys=True, separators=(",", ":")
                )
                self.assertNotEqual(
                    policy.policy_sha256,
                    hashlib.sha256(changed_canonical.encode("utf-8")).hexdigest(),
                )
        reconstructed = type(policy)(**json.loads(json.dumps(policy.as_dict())))
        self.assertEqual(reconstructed.policy_sha256, policy.policy_sha256)
        gate = evaluate_render_gate(policy)
        self.assertTrue(gate.allowed)
        self.assertEqual(
            gate.source_method_consistency_status,
            "source_method_consistent_with_numeric_area_warning",
        )
        self.assertEqual(gate.warning, arguments["source_method_consistency_warning"])
        self.assertIn("Warning:", gate.reason)
        self.assertIn("not a bound", gate.reason)

        changed = evaluate_validation_policy(
            self.records(),
            self.benchmark(),
            **{
                **arguments,
                "source_method_consistency_warning": (
                    "A different, still explicit scientific warning."
                ),
            },
        )
        self.assertNotEqual(policy.policy_sha256, changed.policy_sha256)

    def test_source_method_contract_rejects_inconsistent_exception_evidence(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValidationPolicyError, "exception count and unique stage IDs"
        ):
            evaluate_validation_policy(
                self.records(),
                self.benchmark(),
                accepted_source_area_relative_error=0.02,
                accepted_source_storage_relative_error=0.02,
                accepted_working_area_relative_error=0.03,
                accepted_working_storage_relative_error=0.03,
                evidence_used=("official source",),
                scientific_rationale="Reject internally inconsistent evidence.",
                official_uncertainty_supports_formal_validation=False,
                resolution_convergence_behavior="synthetic convergence",
                source_method_consistency_gate_passed=True,
                source_exact_numeric_gate_passed=False,
                source_storage_numeric_precision_gate_passed=True,
                source_numeric_area_precision_exception_count=2,
                source_numeric_area_precision_exception_stage_ids=("low",),
                source_method_max_area_equivalent_radial_displacement_m=0.004,
                native_source_cell_diagonal_m=math.sqrt(0.5),
                source_method_consistency_rule="synthetic native-grid rule",
                source_method_consistency_warning="Synthetic limitation warning.",
                source_method_consistency_status=(
                    "source_method_consistent_with_numeric_area_warning"
                ),
            )


if __name__ == "__main__":
    unittest.main()
