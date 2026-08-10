"""Credential-free tests for quick-example lake and setting resolution."""

from __future__ import annotations

import datetime as dt
import unittest

import pandas as pd

from regional_analysis.quick_examples import (
    QUICK_EXAMPLE_SETS,
    resolve_example_lake_ids,
)
from regional_analysis.quick_sampling import (
    QuickSamplingSettings,
    select_lake_specs,
)


def candidate_inventory() -> pd.DataFrame:
    """Return the minimum local inventory needed by ``select_lake_specs``."""

    return pd.DataFrame(
        {
            "Hylak_id": [776, 798, 791],
            "Lake_name": ["Lake Abert", "Mono Lake", "Pyramid Lake"],
            "state": ["OR", "CA", "NV"],
            "Lake_area": [174.35, 185.96, 476.87],
            "centroid_lon": [-120.2199, -119.0146, -119.5658],
            "centroid_lat": [42.6519, 38.0079, 40.0480],
        }
    )


class QuickExampleResolutionTests(unittest.TestCase):
    def test_default_resolves_to_smoke_set(self) -> None:
        self.assertEqual(
            resolve_example_lake_ids(),
            QUICK_EXAMPLE_SETS["smoke"],
        )

    def test_convenience_sets_resolve_case_insensitively(self) -> None:
        self.assertEqual(
            resolve_example_lake_ids(set_name=" SMOKE "),
            (776, 798, 791),
        )
        self.assertEqual(
            resolve_example_lake_ids(set_name="Saline6"),
            (67, 776, 798, 791, 795, 829),
        )

    def test_explicit_ids_preserve_user_order(self) -> None:
        self.assertEqual(
            resolve_example_lake_ids(lake_ids=[791, 776, 798]),
            (791, 776, 798),
        )

    def test_set_and_explicit_ids_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "either --set or --lake-ids"):
            resolve_example_lake_ids(set_name="smoke", lake_ids=[776])

    def test_invalid_hydrolakes_ids_are_rejected(self) -> None:
        inventory = candidate_inventory()
        for value in (0, -1, 1.5, "not-an-id", True):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    (TypeError, ValueError), "Invalid HydroLAKES ID"
                ),
            ):
                select_lake_specs(inventory, [value])

    def test_duplicate_hydrolakes_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate HydroLAKES IDs: 776"):
            select_lake_specs(candidate_inventory(), [776, 776])

    def test_ids_missing_from_inventory_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "not in the western candidate inventory: 999999"
        ):
            select_lake_specs(candidate_inventory(), [999999])

    def test_resolved_lake_specs_retain_inventory_metadata(self) -> None:
        specs = select_lake_specs(candidate_inventory(), [798, 776])

        self.assertEqual([spec.hylak_id for spec in specs], [798, 776])
        self.assertEqual(specs[0].lake_name, "Mono Lake")
        self.assertEqual(specs[0].state, "CA")
        self.assertAlmostEqual(specs[0].lake_area, 185.96)


class QuickSamplingSettingsTests(unittest.TestCase):
    def test_overrides_return_a_validated_copy(self) -> None:
        defaults = QuickSamplingSettings()
        overridden = defaults.with_overrides(
            buffer_km=7.5,
            samples_per_image=12,
            analysis_end=dt.date(2025, 12, 31),
            opera_window_days=7,
        )

        self.assertEqual(defaults.buffer_km, 5.0)
        self.assertEqual(defaults.samples_per_image, 20)
        self.assertEqual(overridden.buffer_km, 7.5)
        self.assertEqual(overridden.samples_per_image, 12)
        self.assertEqual(overridden.analysis_end, dt.date(2025, 12, 31))
        self.assertEqual(overridden.opera_window_days, 7)
        self.assertEqual(
            overridden.reference_window("DW"),
            (dt.date(2017, 1, 1), dt.date(2025, 12, 31)),
        )

    def test_none_override_keeps_existing_value(self) -> None:
        settings = QuickSamplingSettings(buffer_km=8.0)

        self.assertEqual(settings.with_overrides(buffer_km=None).buffer_km, 8.0)

    def test_non_positive_settings_are_rejected(self) -> None:
        for field, value in (
            ("buffer_km", 0),
            ("samples_per_image", -1),
            ("max_rows_per_combination", 0),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                QuickSamplingSettings(**{field: value})

    def test_invalid_fraction_and_empty_windows_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "train_fraction"):
            QuickSamplingSettings(train_fraction=1.0)
        with self.assertRaisesRegex(ValueError, "cv_splits"):
            QuickSamplingSettings(cv_splits=1)
        settings = QuickSamplingSettings(
            analysis_end=dt.date(2017, 12, 31),
            dw_start=dt.date(2018, 1, 1),
            opera_start=dt.date(2017, 1, 1),
        )
        with self.assertRaisesRegex(ValueError, "Dynamic World window is empty"):
            settings.reference_window("DW")

    def test_unselected_later_reference_does_not_block_jrc_override(self) -> None:
        settings = QuickSamplingSettings(
            analysis_end=dt.date(2021, 12, 31),
        )

        self.assertEqual(
            settings.reference_window("JRC"),
            (dt.date(2017, 1, 1), dt.date(2021, 12, 31)),
        )


if __name__ == "__main__":
    unittest.main()
