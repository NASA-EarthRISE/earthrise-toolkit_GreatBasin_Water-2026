from __future__ import annotations

import os
import pyexpat  # noqa: F401  # Windows GDAL DLL import-order requirement
import sqlite3  # noqa: F401  # Windows GDAL DLL import-order requirement
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.transform import from_origin

from regional_analysis.hydrometry.raster_storage import (
    build_raster_elevation_histogram,
    evaluate_elevation_histogram,
)
from regional_analysis.hydrometry.topobathymetry import (
    AmbiguousRasterError,
    ArchiveRasterCandidate,
    RasterInspectionError,
    inspect_elevation_raster,
    select_archive_raster_candidate,
)


class ArchiveRasterSelectionTests(unittest.TestCase):
    def test_documented_candidate_is_selected_not_first_tif(self) -> None:
        candidates = (
            ArchiveRasterCandidate("uncertainty.tif", 50),
            ArchiveRasterCandidate(
                "data/Bear_Lake_TBDEM.tif",
                500,
                expected_by_metadata=True,
                resolution_m=0.5,
            ),
        )
        selected = select_archive_raster_candidate(
            candidates, expected_resolution_m=0.5
        )
        self.assertEqual(selected.archive_path, "data/Bear_Lake_TBDEM.tif")

    def test_ambiguous_candidates_fail(self) -> None:
        candidates = (
            ArchiveRasterCandidate("a.tif", 50, resolution_m=0.5),
            ArchiveRasterCandidate("b.tif", 60, resolution_m=0.5),
        )
        with self.assertRaises(AmbiguousRasterError):
            select_archive_raster_candidate(candidates, expected_resolution_m=0.5)

    def test_expected_filename_is_exact_and_unsafe_path_fails(self) -> None:
        candidates = (
            ArchiveRasterCandidate("nested/a.tif", 50),
            ArchiveRasterCandidate("nested/b.tif", 60),
        )
        selected = select_archive_raster_candidate(
            candidates, expected_filename="b.tif"
        )
        self.assertEqual(selected.archive_path, "nested/b.tif")
        with self.assertRaises(RasterInspectionError):
            ArchiveRasterCandidate("../escape.tif", 2)


class RasterInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_proj_lib = os.environ.get("PROJ_LIB")
        cls.original_gdal_data = os.environ.get("GDAL_DATA")
        package_root = Path(rasterio.__file__).resolve().parent
        os.environ["PROJ_LIB"] = str(package_root / "proj_data")
        os.environ["GDAL_DATA"] = str(package_root / "gdal_data")

    @classmethod
    def tearDownClass(cls) -> None:
        for key, value in (
            ("PROJ_LIB", cls.original_proj_lib),
            ("GDAL_DATA", cls.original_gdal_data),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_raster(
        self,
        name: str,
        *,
        crs: str = "EPSG:26912",
        transform: Affine | None = None,
        count: int = 1,
        nodata: float | None = -9999.0,
        vertical_datum: str | None = "NAVD88",
        vertical_units: str | None = "metres",
        scale: float = 1.0,
        offset: float = 0.0,
    ) -> Path:
        path = self.root / name
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        data[0, 0] = nodata if nodata is not None else 0.0
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=count,
            dtype="float32",
            crs=crs,
            transform=transform or from_origin(500_000, 4_600_000, 0.5, 0.5),
            nodata=nodata,
        ) as target:
            for band in range(1, count + 1):
                target.write(data, band)
            tags = {}
            if vertical_datum is not None:
                tags["vertical_datum"] = vertical_datum
            if vertical_units is not None:
                tags["vertical_units"] = vertical_units
            target.update_tags(**tags)
            target.scales = tuple(scale for _ in range(count))
            target.offsets = tuple(offset for _ in range(count))
        return path

    def test_valid_projected_navd88_raster_is_inspected_windowwise(self) -> None:
        path = self.write_raster("valid.tif")
        result = inspect_elevation_raster(path, expected_resolution_m=0.5)
        self.assertEqual(result.band_count, 1)
        self.assertEqual(result.horizontal_epsg, 26912)
        self.assertEqual(result.vertical_datum, "NAVD88")
        self.assertEqual(result.vertical_units, "metres")
        self.assertEqual(result.valid_cell_count, 15)
        self.assertEqual(result.minimum_valid_elevation_m, 1.0)
        self.assertEqual(result.maximum_valid_elevation_m, 15.0)
        self.assertAlmostEqual(result.cell_area_m2, 0.25)
        self.assertGreater(result.peak_window_memory_bytes, 0)

    def test_file_backed_histogram_uses_windows_and_exact_bed_sums(self) -> None:
        path = self.write_raster("histogram.tif")
        histogram = build_raster_elevation_histogram(
            path,
            bin_origin_m=0.0,
            bin_width_m=1.0,
            bin_count=16,
        )
        result = evaluate_elevation_histogram(
            histogram,
            10.0,
            source_resolution_m=0.5,
            basin_method="all_valid_below_stage",
        )
        self.assertEqual(histogram.valid_cell_count, 15)
        self.assertEqual(result.area_m2, 10 * 0.25)
        self.assertEqual(result.storage_m3, 45 * 0.25)
        self.assertGreater(histogram.peak_memory_estimate_bytes, 0)

    def test_external_official_vertical_metadata_is_allowed_but_not_guessed(
        self,
    ) -> None:
        path = self.write_raster(
            "external_metadata.tif", vertical_datum=None, vertical_units=None
        )
        with self.assertRaisesRegex(RasterInspectionError, "[Vv]ertical datum"):
            inspect_elevation_raster(path, expected_resolution_m=0.5)
        result = inspect_elevation_raster(
            path,
            expected_resolution_m=0.5,
            vertical_datum="NAVD 88 (Geoid12B)",
            vertical_units="m",
        )
        self.assertEqual(result.vertical_datum, "NAVD 88 (Geoid12B)")
        official = inspect_elevation_raster(
            path,
            expected_resolution_m=0.5,
            vertical_datum=(
                "North American Vertical Datum of 1988 (EPSG:5703), GEOID12B"
            ),
            vertical_units="meters",
        )
        self.assertIn("1988", official.vertical_datum)

    def test_wrong_crs_units_band_count_mask_scale_and_resolution_fail(self) -> None:
        cases = (
            ("geographic.tif", {"crs": "EPSG:4326"}, "projected"),
            ("two_band.tif", {"count": 2}, "one authoritative"),
            ("no_mask.tif", {"nodata": None}, "nodata"),
            ("scaled.tif", {"scale": 2.0}, "scale/offset"),
            (
                "wrong_resolution.tif",
                {"transform": from_origin(500_000, 4_600_000, 1.0, 1.0)},
                "pixel dimensions",
            ),
        )
        for name, options, message in cases:
            with self.subTest(name=name):
                path = self.write_raster(name, **options)
                with self.assertRaisesRegex(RasterInspectionError, message):
                    inspect_elevation_raster(path, expected_resolution_m=0.5)

    def test_rotated_grid_and_vertical_conflict_fail(self) -> None:
        rotated = self.write_raster(
            "rotated.tif",
            transform=Affine(0.5, 0.1, 500_000, 0.0, -0.5, 4_600_000),
        )
        with self.assertRaisesRegex(RasterInspectionError, "Rotated"):
            inspect_elevation_raster(rotated, expected_resolution_m=0.5)
        conflict = self.write_raster("conflict.tif", vertical_datum="NGVD29")
        with self.assertRaisesRegex(RasterInspectionError, "[Vv]ertical datum"):
            inspect_elevation_raster(
                conflict,
                expected_resolution_m=0.5,
                vertical_datum="NAVD88",
            )

    def test_expected_extent_must_cover_the_official_domain(self) -> None:
        path = self.write_raster("extent.tif")
        result = inspect_elevation_raster(
            path,
            expected_resolution_m=0.5,
            expected_bounds=(500_000.0, 4_599_998.0, 500_002.0, 4_600_000.0),
        )
        self.assertEqual(result.width, 4)
        with self.assertRaisesRegex(RasterInspectionError, "expected lake domain"):
            inspect_elevation_raster(
                path,
                expected_resolution_m=0.5,
                expected_bounds=(600_000.0, 4_699_998.0, 600_002.0, 4_700_000.0),
            )


if __name__ == "__main__":
    unittest.main()
