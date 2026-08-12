"""Credential-free tests for the staged expanded-lake manifest."""

from __future__ import annotations

import unittest

import pandas as pd

from regional_analysis.expanded_lake_selection import (
    CURRENT_IDS,
    DEFERRED,
    TIER1_ADDITIONS,
    TIER1_IDS,
    TIER2_ADDITIONS,
    TIER2_IDS,
    SelectionError,
    build_selection_manifest,
    selected_ids,
    selection_notes,
    validate_selection_manifest,
)


def inventory_fixture() -> pd.DataFrame:
    ids = list(dict.fromkeys([*TIER2_IDS, *DEFERRED]))
    return pd.DataFrame(
        {
            "Hylak_id": ids,
            "Lake_name": [f"Lake {lake_id}" for lake_id in ids],
            "state": ["OR"] * len(ids),
            "Lake_area": [float(1000 - index) for index in range(len(ids))],
            "Lake_type": [1] * len(ids),
        }
    )


class ExpandedSelectionTests(unittest.TestCase):
    def test_tiers_are_cumulative_and_stably_ordered(self) -> None:
        self.assertEqual(selected_ids("tier1"), TIER1_IDS)
        self.assertEqual(selected_ids("tier2"), TIER2_IDS)
        self.assertEqual(TIER1_IDS[: len(CURRENT_IDS)], CURRENT_IDS)
        self.assertEqual(TIER1_IDS[len(CURRENT_IDS) :], TIER1_ADDITIONS)
        self.assertEqual(TIER2_IDS[len(TIER1_IDS) :], TIER2_ADDITIONS)

    def test_manifest_marks_selected_and_reviewed_exclusions(self) -> None:
        manifest = build_selection_manifest(inventory_fixture())
        validate_selection_manifest(manifest)

        selected = manifest[manifest["include"]]
        self.assertEqual(set(selected["Hylak_id"]), set(TIER2_IDS))
        self.assertFalse(manifest.loc[manifest["Hylak_id"] == 783, "include"].item())
        self.assertIn(
            "Salt-flat",
            manifest.loc[manifest["Hylak_id"] == 783, "exclude_reason"].item(),
        )

    def test_missing_required_design_id_is_rejected(self) -> None:
        inventory = inventory_fixture()
        inventory = inventory[inventory["Hylak_id"] != TIER1_ADDITIONS[0]]

        with self.assertRaisesRegex(SelectionError, "absent from inventory"):
            build_selection_manifest(inventory)

    def test_unknown_tier_is_rejected(self) -> None:
        with self.assertRaisesRegex(SelectionError, "tier must"):
            selected_ids("all")

    def test_notes_freeze_candidate_inventory_provenance(self) -> None:
        notes = selection_notes(build_selection_manifest(inventory_fixture()))

        self.assertIn(
            "projects/sat-io/open-datasets/HydroLakes/lake_poly_v10",
            notes,
        )
        self.assertIn("2026-08-10", notes)
        self.assertIn("25 km2", notes)
        self.assertIn("selection_order <= 12", notes)


if __name__ == "__main__":
    unittest.main()
