from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np

from regional_analysis.hydrometry.raster_storage import compute_stage_raster
from regional_analysis.hydrometry.spatial_products import (
    BYTE_NODATA,
    CHANGE_CODES,
    FLOAT_NODATA,
    OptionalVectorDependencyError,
    SpatialProductError,
    SpatialState,
    build_phase2_vector_layers,
    depth_and_inundation_arrays,
    inundation_change_codes,
    polygonize_inundation,
    require_quantitative_gate,
    write_geopackage,
    write_quantitative_spatial_products,
    write_tiled_geotiff,
)


def _rasterio():
    import pyexpat

    _ = pyexpat
    import rasterio
    from affine import Affine

    return rasterio, Affine


class Phase2SpatialProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, affine = _rasterio()
        cls.transform = affine.translation(500_000.0, 4_650_000.0) * affine.scale(
            10.0, -10.0
        )
        cls.crs = "EPSG:32612"

    def test_depth_arrays_and_change_codes_preserve_support(self) -> None:
        elevation = np.array([[0.0, 1.0, 2.0], [0.0, 1.5, np.nan]])
        support = np.array([[1, 1, 1], [1, 1, 0]], dtype=bool)
        water = np.array([[1, 1, 0], [1, 0, 0]], dtype=bool)
        depth, inundation = depth_and_inundation_arrays(
            elevation,
            connected_inundation=water,
            supported_mask=support,
            stage_m_navd88=1.2,
        )
        np.testing.assert_allclose(depth[water], [1.2, 0.2, 1.2])
        self.assertEqual(depth[0, 2], 0.0)
        self.assertEqual(depth[1, 2], FLOAT_NODATA)
        self.assertEqual(inundation[1, 2], BYTE_NODATA)

        prior = np.array([[1, 0, 0], [0, 1, 0]], dtype=bool)
        change = inundation_change_codes(water, prior, supported_mask=support)
        np.testing.assert_array_equal(
            change, np.array([[1, 2, 0], [2, 3, 255]], dtype=np.uint8)
        )
        self.assertEqual(CHANGE_CODES[3], "exposed_or_lost_water")

    def test_accepts_connected_mask_from_quantitative_raster_engine(self) -> None:
        elevation = np.array([[0.0, 1.0, 9.0, 0.0], [0.0, 1.0, 9.0, 0.0]], dtype=float)
        result = compute_stage_raster(
            elevation,
            2.0,
            pixel_width_m=10.0,
            pixel_height_m=10.0,
            basin_method="connected_seed",
            trusted_seed=(0, 0),
            connectivity=8,
        )
        self.assertFalse(result.inundation[:, 3].any())
        depth, categories = depth_and_inundation_arrays(
            elevation,
            connected_inundation=result.inundation,
            supported_mask=result.supported_mask,
            stage_m_navd88=2.0,
        )
        self.assertEqual(np.count_nonzero(categories == 1), 4)
        self.assertEqual(float(depth[categories == 1].sum()) * 100.0, result.storage_m3)

    def test_invalid_connected_mask_is_rejected(self) -> None:
        with self.assertRaisesRegex(SpatialProductError, "dry or invalid"):
            depth_and_inundation_arrays(
                np.array([[2.0]]),
                connected_inundation=np.array([[True]]),
                supported_mask=np.array([[True]]),
                stage_m_navd88=1.0,
            )

    def test_calculation_plane_preserves_nominal_stage_and_float32_equality(
        self,
    ) -> None:
        nominal_stage = 1805.15
        calculation_stage = float(np.float32(nominal_stage))
        self.assertGreater(calculation_stage, nominal_stage)
        elevation = np.array([[calculation_stage]], dtype=np.float64)
        water = np.array([[True]])
        with self.assertRaisesRegex(SpatialProductError, "dry or invalid"):
            depth_and_inundation_arrays(
                elevation,
                connected_inundation=water,
                supported_mask=water,
                stage_m_navd88=nominal_stage,
            )
        depth, categories = depth_and_inundation_arrays(
            elevation,
            connected_inundation=water,
            supported_mask=water,
            stage_m_navd88=nominal_stage,
            calculation_stage_m_navd88=calculation_stage,
        )
        self.assertEqual(float(depth[0, 0]), 0.0)
        self.assertEqual(int(categories[0, 0]), 1)

        state = SpatialState(
            "latest",
            nominal_stage,
            nominal_stage / 0.3048,
            water,
            calculation_stage_m_navd88=calculation_stage,
        )
        properties = build_phase2_vector_layers(
            (
                SpatialState(
                    "year_ago",
                    nominal_stage,
                    nominal_stage / 0.3048,
                    water,
                    calculation_stage_m_navd88=calculation_stage,
                ),
                state,
            ),
            transform=self.transform,
        )["water_surface_current"][0].properties
        self.assertEqual(properties["stage_m_navd88"], nominal_stage)
        self.assertEqual(properties["calculation_stage_m_navd88"], calculation_stage)

    def test_polygonization_preserves_island_hole_and_exact_area(self) -> None:
        water = np.ones((7, 7), dtype=bool)
        water[2:5, 2:5] = False
        result = polygonize_inundation(water, transform=self.transform)
        self.assertAlmostEqual(result.raster_area_m2, 40 * 100.0)
        self.assertAlmostEqual(result.polygon_area_m2, result.raster_area_m2)
        self.assertTrue(result.geometry.is_valid)
        self.assertEqual(len(result.geometry.interiors), 1)

    def test_polygonization_repairs_eight_neighbor_point_touches(self) -> None:
        water = np.zeros((3, 3), dtype=bool)
        water[0, 0] = True
        water[1, 1] = True
        result = polygonize_inundation(water, transform=self.transform)
        self.assertTrue(result.geometry.is_valid)
        self.assertEqual(result.geometry.geom_type, "MultiPolygon")
        self.assertAlmostEqual(result.polygon_area_m2, 200.0)
        self.assertEqual(
            result.repair_method,
            "shapely_make_valid_for_8_neighbor_point_touches",
        )

    def test_geotiff_is_tiled_compressed_tagged_and_hashed(self) -> None:
        rasterio, _ = _rasterio()
        array = np.arange(100, dtype=np.float32).reshape(10, 10)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "depth.tif"
            write_tiled_geotiff(
                path,
                array,
                transform=self.transform,
                crs=self.crs,
                nodata=FLOAT_NODATA,
                band_description="Synthetic water depth (m)",
                tags={
                    "PROVENANCE_PATH": "bear_lake_phase2_provenance.json",
                    "VERTICAL_DATUM": "NAVD88",
                },
                dtype="float32",
            )
            with rasterio.open(path) as dataset:
                self.assertTrue(dataset.profile["tiled"])
                self.assertEqual(dataset.compression.name.lower(), "deflate")
                self.assertEqual(dataset.nodata, FLOAT_NODATA)
                self.assertEqual(dataset.descriptions[0], "Synthetic water depth (m)")
                self.assertEqual(dataset.tags()["VERTICAL_DATUM"], "NAVD88")
                self.assertEqual(dataset.crs.to_epsg(), 32612)

    def test_full_raster_writer_uses_stable_names_and_embeds_codes(self) -> None:
        rasterio, _ = _rasterio()
        elevation = np.tile(np.arange(8, dtype=float), (8, 1))
        coverage = np.ones_like(elevation, dtype=bool)
        basin = coverage.copy()
        states = (
            SpatialState("recent_low", 2.5, 2.5 / 0.3048, elevation <= 2.5),
            SpatialState("year_ago", 3.5, 3.5 / 0.3048, elevation <= 3.5),
            SpatialState(
                "latest",
                4.5,
                4.5 / 0.3048,
                elevation <= 4.5,
                calculation_stage_m_navd88=4.5001,
            ),
            SpatialState("recent_high", 5.5, 5.5 / 0.3048, elevation <= 5.5),
            SpatialState("historical_median", 4.0, 4.0 / 0.3048, elevation <= 4.0),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            provenance = output / "bear_lake_phase2_provenance.json"
            provenance.write_text("{}", encoding="utf-8")
            result = write_quantitative_spatial_products(
                output,
                elevation_m=elevation,
                basin_mask=basin,
                coverage_mask=coverage,
                transform=self.transform,
                crs=self.crs,
                states=states,
                validation_status="numerically_consistent",
                provenance_path=provenance,
                write_vectors=False,
            )
            expected = {
                "topobathy_working",
                "basin_mask",
                "coverage",
                "depth_current",
                "inundation_current",
                "depth_recent_low",
                "depth_recent_high",
                "depth_year_ago",
                "depth_historical_median",
                "inundation_change_latest_vs_year_ago",
            }
            self.assertTrue(expected.issubset(result.rasters))
            self.assertEqual(set(result.rasters), set(result.sha256))
            change_path = result.rasters["inundation_change_latest_vs_year_ago"]
            with rasterio.open(change_path) as dataset:
                codes = json.loads(dataset.tags()["CATEGORY_CODES"])
                self.assertEqual(codes["2"], "newly_inundated")
                self.assertEqual(dataset.tags()["PROVENANCE_PATH"], provenance.name)
            with rasterio.open(result.rasters["depth_current"]) as dataset:
                self.assertEqual(dataset.tags()["STAGE_M_NAVD88"], "4.5")
                self.assertEqual(dataset.tags()["CALCULATION_STAGE_M_NAVD88"], "4.5001")
            self.assertEqual(
                set(result.vector_layers),
                {
                    "shoreline_current",
                    "shorelines_selected",
                    "shoreline_change_latest_vs_year_ago",
                    "depth_contours_current",
                    "water_surface_current",
                    "water_surfaces_selected",
                },
            )

    def test_vector_layers_include_temporal_and_quantitative_attributes(self) -> None:
        elevation = np.tile(np.arange(6, dtype=float), (6, 1))
        states = (
            SpatialState(
                "year_ago",
                2.5,
                2.5 / 0.3048,
                elevation <= 2.5,
                "2025-08-12",
                1,
                1800.0,
                12000.0,
            ),
            SpatialState(
                "latest",
                3.5,
                3.5 / 0.3048,
                elevation <= 3.5,
                "2026-08-12",
                2,
                2400.0,
                16000.0,
            ),
        )
        depth, _ = depth_and_inundation_arrays(
            elevation,
            connected_inundation=states[1].inundation,
            supported_mask=np.ones_like(elevation, dtype=bool),
            stage_m_navd88=3.5,
        )
        layers = build_phase2_vector_layers(
            states,
            transform=self.transform,
            current_depth_m=depth,
            contour_levels_m=(1.0, 2.0),
        )
        current = layers["water_surface_current"][0]
        self.assertEqual(current.properties["start_datetime"], "2026-08-12")
        self.assertEqual(current.properties["frame_order"], 2)
        self.assertEqual(current.properties["derived_storage_m3"], 16000.0)
        self.assertTrue(current.geometry.is_valid)
        self.assertGreater(len(layers["depth_contours_current"]), 0)

    def test_ogr2ogr_fallback_uses_explicit_argv_and_atomic_result(self) -> None:
        water = np.ones((2, 2), dtype=bool)
        feature = build_phase2_vector_layers(
            (
                SpatialState("year_ago", 1.0, 3.28, water, "2025-01-01"),
                SpatialState("latest", 1.0, 3.28, water, "2026-01-01"),
            ),
            transform=self.transform,
        )["water_surface_current"]
        calls: list[list[str]] = []

        def fake_runner(command, **kwargs):
            self.assertNotIsInstance(command, str)
            self.assertTrue(kwargs["check"])
            self.assertNotIn("shell", kwargs)
            calls.append(command)
            with closing(sqlite3.connect(command[3])) as database:
                database.execute("CREATE TABLE gpkg_contents (last_change TEXT)")
                database.execute(
                    "INSERT INTO gpkg_contents VALUES ('2026-08-13T12:00:00.000Z')"
                )
                database.execute(
                    "CREATE TABLE gpkg_metadata_reference (timestamp TEXT)"
                )
                database.execute(
                    "INSERT INTO gpkg_metadata_reference VALUES "
                    "('2026-08-13T12:00:00.000Z')"
                )
                database.commit()

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            executable = directory / "ogr2ogr.exe"
            executable.write_bytes(b"stub")
            target = directory / "test.gpkg"
            write_geopackage(
                target,
                {"water_surface_current": feature},
                crs=self.crs,
                ogr2ogr_executable=executable,
                runner=fake_runner,
            )
            self.assertTrue(target.is_file())
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], str(executable))
            self.assertIn("-nln", calls[0])
            with closing(sqlite3.connect(target)) as database:
                self.assertEqual(
                    database.execute(
                        "SELECT last_change FROM gpkg_contents"
                    ).fetchone()[0],
                    "1980-01-01T00:00:00.000Z",
                )

    def test_missing_vector_writer_and_failed_gate_are_clear(self) -> None:
        with self.assertRaisesRegex(SpatialProductError, "gate is closed"):
            require_quantitative_gate("failed")
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaises(OptionalVectorDependencyError),
        ):
            write_geopackage(
                Path(temporary) / "missing.gpkg",
                {},
                crs=self.crs,
                ogr2ogr_executable=Path(temporary) / "missing.exe",
            )


if __name__ == "__main__":
    unittest.main()
