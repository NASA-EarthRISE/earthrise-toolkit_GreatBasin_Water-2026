"""Credential-free tests for the Tier 1 expanded plotting helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from regional_analysis.plot_expanded_transfer import (
    CURVE_EXAMPLES,
    DEFAULT_DATA_DIR,
    DEFAULT_EXPANDED_CACHE_DIR,
    DEFAULT_PRIMARY_CACHE_DIR,
    build_selected_threshold_responses,
    load_expanded_plot_tables,
)


@unittest.skipUnless(
    (DEFAULT_PRIMARY_CACHE_DIR / "opera_67.csv").is_file()
    and (
        Path(DEFAULT_DATA_DIR) / "western_quick_transfer_opera_expanded_tier1_local.csv"
    ).is_file(),
    "requires ignored completed Tier 1 local artifacts",
)
class ExpandedPlotLocalIntegrationTests(unittest.TestCase):
    def test_selected_curves_use_authoritative_cached_test_splits(self) -> None:
        tables = load_expanded_plot_tables(DEFAULT_DATA_DIR, "tier1")
        responses, caches = build_selected_threshold_responses(
            tables,
            primary_cache_dir=DEFAULT_PRIMARY_CACHE_DIR,
            expanded_cache_dir=DEFAULT_EXPANDED_CACHE_DIR,
        )

        self.assertEqual(set(responses), set(CURVE_EXAMPLES))
        for reference, lake_id in CURVE_EXAMPLES:
            cache = caches[(reference, lake_id)]
            self.assertEqual(set(cache["split"]), {"train", "test"})
            self.assertEqual(
                cache.groupby("block_id")["split"].nunique().max(),
                1,
            )
            response = responses[(reference, lake_id)]
            self.assertEqual(len(response), 150)
            self.assertTrue(
                np.isfinite(response[["threshold", "iou", "f1"]]).all().all()
            )

    def test_macro_sources_preserve_exact_completed_values(self) -> None:
        tables = load_expanded_plot_tables(DEFAULT_DATA_DIR, "tier1")
        expected = {
            "OPERA": (0.9380496911408241, 0.8453853644720456),
            "JRC": (0.9252922701916546, 0.7910505347621574),
        }
        for reference, (six_expected, tier1_expected) in expected.items():
            six = pd.read_csv(
                Path(DEFAULT_DATA_DIR)
                / f"western_quick_transfer_{reference.lower()}_lolo.csv"
            )
            self.assertAlmostEqual(float(six["macro_lolo_iou"].iloc[0]), six_expected)
            self.assertAlmostEqual(
                float(tables[reference]["macro"]["macro_lolo_iou"].iloc[0]),
                tier1_expected,
            )


if __name__ == "__main__":
    unittest.main()
