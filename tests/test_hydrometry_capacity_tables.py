from __future__ import annotations

import math
import unittest
from dataclasses import replace

from regional_analysis.hydrometry.capacity_tables import (
    CapacityTableError,
    CapacityTableInputRow,
    CapacityTableMetadata,
    normalize_capacity_table,
)
from regional_analysis.hydrometry.units import convert_area, convert_volume


def metadata() -> CapacityTableMetadata:
    return CapacityTableMetadata(
        lake_name="Bear Lake",
        source_doi="10.5066/P14KZS2D",
        source_title="Bear Lake bathymetry and elevation-area-volume data",
        source_file="bear_lake_eav.csv",
        source_version="2024 release",
        source_hash="a" * 64,
        vertical_datum="NAVD88",
        elevation_unit="ft",
        area_unit="acres",
        storage_unit="acre-feet",
    )


def rows() -> tuple[CapacityTableInputRow, ...]:
    # Storage increments equal trapezoidal mean area times elevation increment.
    return (
        CapacityTableInputRow(2, 10.0, 10.0, 0.0),
        CapacityTableInputRow(3, 20.0, 20.0, 150.0),
        CapacityTableInputRow(4, 30.0, 30.0, 400.0),
    )


class CapacityTableTests(unittest.TestCase):
    def test_valid_table_preserves_source_and_normalizes_required_fields(self) -> None:
        table = normalize_capacity_table(
            rows(), metadata(), expected_lake_name="Bear Lake"
        )
        self.assertEqual(table.validation.validation_status, "passed")
        self.assertEqual(table.validation.row_count, 3)
        first = table.normalized_records()[0]
        self.assertEqual(first["source_row"], 2)
        self.assertEqual(first["elevation_original"], 10.0)
        self.assertEqual(first["elevation_original_unit"], "ft")
        self.assertEqual(first["area_original_unit"], "acre")
        self.assertEqual(first["storage_original_unit"], "acre_ft")
        self.assertEqual(first["vertical_datum"], "NAVD88")
        self.assertEqual(first["source_doi"], "10.5066/P14KZS2D")
        self.assertEqual(table.to_dataframe().shape, (3, len(first)))

    def test_area_volume_consistency_diagnostic(self) -> None:
        table = normalize_capacity_table(rows(), metadata())
        diagnostics, summary = table.area_volume_diagnostics()
        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(summary.segment_count, 2)
        # Consistency remains near machine precision after exact unit conversions.
        self.assertTrue(
            math.isclose(summary.maximum_relative_discrepancy, 0.0, abs_tol=1e-14)
        )

    def test_duplicate_or_nonincreasing_elevation_fails(self) -> None:
        duplicate = rows()[:2] + (replace(rows()[2], elevation=20.0),)
        with self.assertRaisesRegex(CapacityTableError, "duplicate elevations"):
            normalize_capacity_table(duplicate, metadata())
        descending = (rows()[0], rows()[2], rows()[1])
        with self.assertRaisesRegex(CapacityTableError, "strictly increase"):
            normalize_capacity_table(descending, metadata())

    def test_nonmonotonic_area_or_storage_fails(self) -> None:
        bad_area = rows()[:2] + (replace(rows()[2], area=19.0),)
        with self.assertRaisesRegex(CapacityTableError, "area must be nondecreasing"):
            normalize_capacity_table(bad_area, metadata())
        bad_storage = rows()[:2] + (replace(rows()[2], storage=149.0),)
        with self.assertRaisesRegex(
            CapacityTableError, "storage must be nondecreasing"
        ):
            normalize_capacity_table(bad_storage, metadata())

    def test_missing_nonfinite_negative_or_wrong_identity_fails(self) -> None:
        with self.assertRaisesRegex(CapacityTableError, "at least two"):
            normalize_capacity_table(rows()[:1], metadata())
        with self.assertRaisesRegex(CapacityTableError, "finite"):
            normalize_capacity_table(
                (rows()[0], replace(rows()[1], area=float("nan"))), metadata()
            )
        with self.assertRaisesRegex(CapacityTableError, "negative"):
            normalize_capacity_table(
                (rows()[0], replace(rows()[1], storage=-1.0)), metadata()
            )
        with self.assertRaisesRegex(CapacityTableError, "not 'Wrong Lake'"):
            normalize_capacity_table(
                rows(), metadata(), expected_lake_name="Wrong Lake"
            )

    def test_metadata_requires_datum_units_version_and_sha256(self) -> None:
        for field in ("vertical_datum", "source_version", "source_doi"):
            with self.subTest(field=field), self.assertRaises(CapacityTableError):
                replace(metadata(), **{field: ""})
        with self.assertRaisesRegex(CapacityTableError, "SHA-256"):
            replace(metadata(), source_hash="not-a-hash")
        with self.assertRaises(CapacityTableError):
            replace(metadata(), area_unit="ft")

    def test_source_units_normalize_without_rounding_originals(self) -> None:
        table = normalize_capacity_table(rows(), metadata())
        second = table.rows[1]
        self.assertEqual(second.area_original, 20.0)
        self.assertEqual(second.storage_original, 150.0)
        self.assertEqual(second.area_m2, convert_area(20.0, "acre", "m2"))
        self.assertEqual(second.storage_m3, convert_volume(150.0, "acre_ft", "m3"))


if __name__ == "__main__":
    unittest.main()
