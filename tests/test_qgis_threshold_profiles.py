"""Credential-free tests for the shipped QGIS threshold registry."""

from __future__ import annotations

import dataclasses
import math
import unittest

from Lake_Classifer_FD.threshold_profiles import (
    DEFAULT_THRESHOLDS,
    GREAT_SALT_LAKE_PROFILE,
    LAKE_ABERT_PROFILE,
    SHIPPED_THRESHOLD_PROFILES,
    default_thresholds_copy,
    get_profile_for_hylak_id,
    get_profile_for_preset,
    get_threshold_profile,
)


class ThresholdProfileTests(unittest.TestCase):
    def test_registry_contains_only_two_shipped_lake_profiles(self) -> None:
        self.assertEqual(
            SHIPPED_THRESHOLD_PROFILES,
            (GREAT_SALT_LAKE_PROFILE, LAKE_ABERT_PROFILE),
        )
        self.assertEqual(
            [profile.preset_key for profile in SHIPPED_THRESHOLD_PROFILES],
            ["Great Salt Lake, UT", "Lake Abert, OR"],
        )
        self.assertEqual(
            [
                (profile.Hylak_id, profile.scope, profile.status)
                for profile in SHIPPED_THRESHOLD_PROFILES
            ],
            [
                (67, "lake_specific", "shipped"),
                (776, "lake_specific", "shipped"),
            ],
        )
        self.assertIn(
            "Hylak_id",
            {field.name for field in dataclasses.fields(GREAT_SALT_LAKE_PROFILE)},
        )
        self.assertFalse(
            any(
                "regional" in profile.profile_id
                for profile in SHIPPED_THRESHOLD_PROFILES
            )
        )

    def test_compatibility_defaults_preserve_exact_shipped_values(self) -> None:
        self.assertEqual(
            DEFAULT_THRESHOLDS,
            {
                "Great Salt Lake, UT": {
                    "ndwi": -0.055,
                    "ndvi": 0.165,
                    "ndmi": 0.053,
                },
                "Lake Abert, OR": {
                    "ndwi": -0.063,
                    "ndvi": 0.130,
                    "ndmi": 0.123,
                },
            },
        )
        for values in DEFAULT_THRESHOLDS.values():
            self.assertEqual(set(values), {"ndwi", "ndvi", "ndmi"})
            self.assertTrue(
                all(
                    math.isfinite(value) and -1 <= value <= 1
                    for value in values.values()
                )
            )

    def test_references_sources_and_legacy_aois_are_explicit(self) -> None:
        for profile in SHIPPED_THRESHOLD_PROFILES:
            self.assertEqual(
                profile.references,
                {"ndwi": "OPERA", "ndvi": "Dynamic World", "ndmi": "NWI"},
            )
            self.assertEqual(
                profile.sources,
                {
                    "ndwi": "data/threshold_cv_summary.csv",
                    "ndvi": "data/threshold_cv_summary.csv",
                    "ndmi": "data/ndmi_threshold_cv_all.csv",
                },
            )
            self.assertIn("Legacy QGIS rectangular preset", profile.aoi_definition)
            self.assertEqual(len(profile.aoi_bounds_wgs84), 4)
            self.assertIn(
                "superseded threshold_optimization_cv.py", profile.provenance_note
            )
            self.assertIn("not present", profile.provenance_note)

    def test_lookup_is_optional_or_strict(self) -> None:
        self.assertIs(
            get_threshold_profile("great_salt_lake_ut"),
            GREAT_SALT_LAKE_PROFILE,
        )
        self.assertIs(
            get_profile_for_preset("Lake Abert, OR"),
            LAKE_ABERT_PROFILE,
        )
        self.assertIs(get_profile_for_hylak_id(67), GREAT_SALT_LAKE_PROFILE)
        self.assertIsNone(get_threshold_profile("missing"))
        self.assertIsNone(get_profile_for_preset("missing"))
        self.assertIsNone(get_profile_for_hylak_id(-1))
        with self.assertRaises(KeyError):
            get_threshold_profile("missing", strict=True)

    def test_profiles_are_immutable_and_mapping_copies_are_detached(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            GREAT_SALT_LAKE_PROFILE.ndwi = 0.5

        thresholds = GREAT_SALT_LAKE_PROFILE.thresholds
        thresholds["ndwi"] = 0.5
        self.assertEqual(GREAT_SALT_LAKE_PROFILE.ndwi, -0.055)

        defaults = default_thresholds_copy()
        defaults["Great Salt Lake, UT"]["ndwi"] = 0.5
        self.assertEqual(
            default_thresholds_copy()["Great Salt Lake, UT"]["ndwi"],
            -0.055,
        )


if __name__ == "__main__":
    unittest.main()
