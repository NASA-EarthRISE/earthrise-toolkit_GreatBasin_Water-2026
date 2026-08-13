from __future__ import annotations

import math
import unittest

from regional_analysis.hydrometry.capacity_tables import (
    CapacityTableInputRow,
    CapacityTableMetadata,
    normalize_capacity_table,
)
from regional_analysis.hydrometry.models import (
    CapabilityLevel,
    HeightType,
    HydrometryVariable,
    SourceQualityClass,
)
from regional_analysis.hydrometry.stage_capacity import (
    StageCapacityError,
    StageCapacityInterpolator,
    VerticalCompatibilityError,
    evaluate_vertical_compatibility,
)
from regional_analysis.hydrometry.units import convert_area, convert_volume


def table():
    metadata = CapacityTableMetadata(
        lake_name="Bear Lake",
        source_doi="10.5066/P14KZS2D",
        source_title="Bear Lake official E-A-V relation",
        source_file="bear_eav.csv",
        source_version="fixture-v1",
        source_hash="b" * 64,
        vertical_datum="NAVD88",
        elevation_unit="ft",
        area_unit="acre",
        storage_unit="acre-ft",
    )
    return normalize_capacity_table(
        (
            CapacityTableInputRow(2, 10.0, 100.0, 1_000.0),
            CapacityTableInputRow(3, 20.0, 200.0, 2_500.0),
            CapacityTableInputRow(4, 30.0, 400.0, 5_500.0),
        ),
        metadata,
    )


def derive(value):
    return StageCapacityInterpolator(table()).derive(
        value,
        input_unit="ft",
        input_vertical_datum="NAVD88",
    )


class VerticalGateTests(unittest.TestCase):
    def test_navd88_absolute_elevation_passes_with_unit_conversion(self) -> None:
        decision = evaluate_vertical_compatibility(
            stage_vertical_datum="NAVD 1988",
            stage_unit="ft",
            height_type=HeightType.ORTHOMETRIC_HEIGHT,
            stage_variable=HydrometryVariable.WATER_LEVEL,
            table_vertical_datum="NAVD88",
        )
        self.assertTrue(decision.compatible)
        self.assertFalse(decision.vertical_transformation_applied)
        self.assertEqual(decision.normalized_stage_unit, "m")
        realization = evaluate_vertical_compatibility(
            stage_vertical_datum="NAVD88",
            stage_unit="m",
            height_type=HeightType.ORTHOMETRIC_HEIGHT,
            stage_variable=HydrometryVariable.WATER_LEVEL,
            table_vertical_datum="NAVD88 (Geoid12B)",
        )
        self.assertTrue(realization.compatible)

    def test_different_or_unknown_datum_fails(self) -> None:
        for datum in ("NGVD29", None):
            with (
                self.subTest(datum=datum),
                self.assertRaises(VerticalCompatibilityError),
            ):
                StageCapacityInterpolator(table()).derive(
                    15.0,
                    input_unit="ft",
                    input_vertical_datum=datum or "",
                )

    def test_local_gage_height_and_wrong_variable_fail(self) -> None:
        with self.assertRaises(VerticalCompatibilityError):
            StageCapacityInterpolator(table()).derive(
                15.0,
                input_unit="ft",
                input_vertical_datum="NAVD88",
                height_type=HeightType.LOCAL_GAGE_HEIGHT,
            )
        with self.assertRaises(VerticalCompatibilityError):
            StageCapacityInterpolator(table()).derive(
                15.0,
                input_unit="ft",
                input_vertical_datum="NAVD88",
                stage_variable=HydrometryVariable.GAGE_HEIGHT,
            )

    def test_matching_units_do_not_override_datum_mismatch(self) -> None:
        with self.assertRaisesRegex(VerticalCompatibilityError, "datum_mismatch"):
            StageCapacityInterpolator(table()).derive(
                4.572,
                input_unit="m",
                input_vertical_datum="NGVD29",
            )


class StageCapacityInterpolationTests(unittest.TestCase):
    def test_exact_lower_and_upper_nodes_reproduce_source_values(self) -> None:
        lower = derive(10.0)
        upper = derive(30.0)
        self.assertEqual(lower.derived_area_acres, 100.0)
        self.assertEqual(lower.derived_storage_acre_ft, 1_000.0)
        self.assertEqual(upper.derived_area_acres, 400.0)
        self.assertEqual(upper.derived_storage_acre_ft, 5_500.0)
        self.assertEqual(lower.interpolation_fraction, 0.0)
        self.assertEqual(upper.interpolation_fraction, 1.0)
        self.assertIsNotNone(lower.d_storage_dh_m3_per_m)
        self.assertIsNotNone(upper.d_storage_dh_m3_per_m)

    def test_midpoint_and_arbitrary_interior_are_linear_and_bounded(self) -> None:
        midpoint = derive(15.0)
        self.assertTrue(midpoint.within_supported_range)
        self.assertEqual(midpoint.range_status, "within_supported_range")
        self.assertTrue(math.isclose(midpoint.interpolation_fraction, 0.5))
        self.assertTrue(math.isclose(midpoint.derived_area_acres, 150.0))
        self.assertTrue(math.isclose(midpoint.derived_storage_acre_ft, 1_750.0))

        interior = derive(22.5)
        self.assertTrue(math.isclose(interior.interpolation_fraction, 0.25))
        self.assertTrue(math.isclose(interior.derived_area_acres, 250.0))
        self.assertTrue(math.isclose(interior.derived_storage_acre_ft, 3_250.0))
        self.assertGreaterEqual(interior.derived_storage_acre_ft, 2_500.0)
        self.assertLessEqual(interior.derived_storage_acre_ft, 5_500.0)

    def test_below_and_above_range_are_unavailable(self) -> None:
        for value, expected in (
            (9.99, "below_supported_range"),
            (30.01, "above_supported_range"),
        ):
            with self.subTest(value=value):
                result = derive(value)
                self.assertFalse(result.within_supported_range)
                self.assertEqual(result.range_status, expected)
                self.assertIsNone(result.derived_area_m2)
                self.assertIsNone(result.derived_storage_m3)
                self.assertIn("no_extrapolation", result.warning_flags[0])

    def test_null_remains_null_and_nonfinite_fails(self) -> None:
        result = derive(None)
        self.assertEqual(result.range_status, "null_input")
        self.assertIsNone(result.normalized_elevation_m)
        self.assertIsNone(result.derived_storage_m3)
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(StageCapacityError):
                derive(value)

    def test_provenance_class_capability_and_sensitivity_propagate(self) -> None:
        result = derive(15.0)
        self.assertEqual(result.method, "official_eav_piecewise_linear")
        self.assertIs(
            result.source_quality_class, SourceQualityClass.STAGE_OFFICIAL_CAPACITY
        )
        self.assertIs(result.capability_level, CapabilityLevel.ABSOLUTE_STORAGE)
        self.assertEqual(result.capacity_table_source, "10.5066/P14KZS2D")
        self.assertEqual(result.capacity_table_hash, "b" * 64)
        self.assertEqual(result.capacity_table_version, "fixture-v1")
        expected_slope = convert_volume(1_500.0, "acre-ft", "m3") / (10.0 * 0.3048)
        self.assertTrue(math.isclose(result.d_storage_dh_m3_per_m, expected_slope))
        self.assertTrue(
            math.isclose(
                result.storage_change_per_0_01_m_m3,
                expected_slope * 0.01,
            )
        )
        self.assertTrue(
            math.isclose(result.derived_area_m2, convert_area(150.0, "acre", "m2"))
        )


if __name__ == "__main__":
    unittest.main()
