from __future__ import annotations

import unittest

import numpy as np

from regional_analysis.hydrometry.shoreline import (
    ShorelineError,
    inundation_change_codes,
    shoreline_displacement_proxy,
    shoreline_mask,
)


class InundationChangeTests(unittest.TestCase):
    def test_all_change_codes_and_nodata_are_explicit(self) -> None:
        latest = np.array([[False, True, True], [False, False, True]])
        prior = np.array([[False, True, False], [True, False, False]])
        supported = np.array([[True, True, True], [True, False, True]])
        result = inundation_change_codes(latest, prior, supported_mask=supported)
        expected = np.array([[0, 1, 2], [3, 255, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_water_outside_supported_domain_fails(self) -> None:
        wet = np.array([[True]])
        with self.assertRaises(ShorelineError):
            inundation_change_codes(wet, ~wet, supported_mask=np.array([[False]]))


class ShorelineGeometryProxyTests(unittest.TestCase):
    def test_island_hole_is_retained_as_wet_side_shoreline(self) -> None:
        wet = np.ones((7, 7), dtype=bool)
        wet[3, 3] = False
        shore = shoreline_mask(wet, connectivity=8)
        self.assertFalse(shore[3, 3])
        self.assertTrue(shore[3, 2])
        self.assertTrue(shore[2, 3])
        self.assertTrue(shore[0, 0])

    def test_identical_shoreline_has_zero_displacement(self) -> None:
        wet = np.zeros((7, 7), dtype=bool)
        wet[2:5, 2:5] = True
        result = shoreline_displacement_proxy(
            wet, wet, pixel_width_m=3.0, pixel_height_m=2.0
        )
        self.assertEqual(result.status, "calculated_raster_proxy")
        self.assertEqual(result.mean_m, 0.0)
        self.assertEqual(result.maximum_m, 0.0)

    def test_rectangular_pixel_sampling_controls_displacement(self) -> None:
        first = np.zeros((7, 7), dtype=bool)
        second = np.zeros((7, 7), dtype=bool)
        first[2:5, 2:5] = True
        second[2:5, 3:6] = True
        horizontal = shoreline_displacement_proxy(
            first, second, pixel_width_m=4.0, pixel_height_m=1.0
        )
        self.assertGreater(horizontal.maximum_m, 0.0)
        self.assertLessEqual(horizontal.maximum_m, 4.0)

    def test_empty_shoreline_is_reported_not_fabricated(self) -> None:
        empty = np.zeros((3, 3), dtype=bool)
        wet = np.ones((3, 3), dtype=bool)
        result = shoreline_displacement_proxy(
            empty, wet, pixel_width_m=1.0, pixel_height_m=1.0
        )
        self.assertEqual(result.status, "unavailable_empty_shoreline")
        self.assertIsNone(result.mean_m)


if __name__ == "__main__":
    unittest.main()
