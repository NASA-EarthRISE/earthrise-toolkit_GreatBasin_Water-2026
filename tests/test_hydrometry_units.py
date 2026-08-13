from __future__ import annotations

import math
import unittest

from regional_analysis.hydrometry.units import (
    ACRE_FOOT_TO_CUBIC_METRE,
    ACRE_TO_SQUARE_METRE,
    FOOT_TO_METRE,
    UnitConversionError,
    canonical_unit,
    convert_area,
    convert_length,
    convert_vertical_value,
    convert_volume,
)


class HydrometryUnitTests(unittest.TestCase):
    def test_documented_exact_constants(self) -> None:
        self.assertEqual(FOOT_TO_METRE, 0.3048)
        self.assertEqual(ACRE_TO_SQUARE_METRE, 4_046.856_422_4)
        self.assertEqual(ACRE_FOOT_TO_CUBIC_METRE, 1_233.481_837_547_52)

    def test_length_round_trip(self) -> None:
        source = -18.25  # Negative elevations are valid physical values.
        metres = convert_length(source, "ft", "m")
        self.assertTrue(math.isclose(convert_length(metres, "m", "ft"), source))

    def test_area_round_trips(self) -> None:
        acres = 19_321.75
        square_metres = convert_area(acres, "acres", "m2")
        self.assertTrue(math.isclose(convert_area(square_metres, "m²", "acre"), acres))
        square_kilometres = convert_area(square_metres, "m2", "km2")
        self.assertTrue(
            math.isclose(convert_area(square_kilometres, "km^2", "m2"), square_metres)
        )

    def test_volume_round_trip(self) -> None:
        acre_feet = 1_234_567.89
        cubic_metres = convert_volume(acre_feet, "acre-feet", "m3")
        self.assertTrue(
            math.isclose(
                convert_volume(cubic_metres, "cubic metres", "acre_ft"),
                acre_feet,
            )
        )

    def test_vertical_unit_conversion_retains_datum(self) -> None:
        converted = convert_vertical_value(
            5_923.4,
            from_unit="ft",
            to_unit="m",
            vertical_datum="NAVD88",
        )
        self.assertEqual(converted.vertical_datum, "NAVD88")
        self.assertFalse(converted.datum_transformation_applied)
        self.assertEqual(converted.unit, "m")

    def test_cross_dimension_and_nonfinite_values_fail(self) -> None:
        with self.assertRaises(UnitConversionError):
            convert_length(1.0, "acre", "m")
        with self.assertRaises(UnitConversionError):
            convert_volume(float("nan"), "m3", "acre-ft")
        with self.assertRaises(UnitConversionError):
            canonical_unit("furlongs")


if __name__ == "__main__":
    unittest.main()
