from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from regional_analysis.tier1_explanatory import (
    ASSOCIATION_OUTCOMES,
    ASSOCIATION_VARIABLES,
    DEFAULT_DATA_DIR,
    DEFAULT_EXPANDED_CACHE_DIR,
    DEFAULT_PRIMARY_CACHE_DIR,
    Tier1ExplanatoryError,
    build_association_table,
    clean_hydrolakes_value,
    plateau_metrics,
    run_tier1_explanatory,
    sample_explanatory_metrics,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    for index, (split, label, score) in enumerate(
        (
            ("train", 0, -0.4),
            ("train", 1, 0.5),
            ("train", 0, -0.2),
            ("train", 1, 1.0),
            ("test", 0, -0.3),
            ("test", 1, 0.4),
            ("test", 0, -0.1),
            ("test", 1, 1.0),
        )
    ):
        rows.append(
            {
                "acquisition_time": 1_600_000_000_000 + (index // 2) * 86_400_000,
                "sensor": "L8",
                "reference_tile_count": 1 + (index // 2) % 2,
                "match_delta_days": (index // 2) / 10,
                "index_value": score,
                "label": label,
                "split": split,
                "block_id": f"b{index}",
            }
        )
    return pd.DataFrame.from_records(rows)


class SentinelTests(unittest.TestCase):
    def test_only_field_specific_sentinels_are_cleaned(self) -> None:
        self.assertTrue(np.isnan(clean_hydrolakes_value("Res_time", -1)))
        self.assertTrue(np.isnan(clean_hydrolakes_value("Dis_avg", -9999)))
        self.assertEqual(clean_hydrolakes_value("Elevation", -71), -71)
        self.assertEqual(clean_hydrolakes_value("Depth_avg", -1), -1)


class SampleMetricTests(unittest.TestCase):
    def test_sample_metrics_use_authoritative_splits_and_exact_endpoint(self) -> None:
        metrics = sample_explanatory_metrics(sample_frame())
        self.assertEqual(metrics["n_unique_acquisitions"], 4)
        self.assertEqual(metrics["n_positive_test"], 2)
        self.assertEqual(metrics["n_positive_test_blocks"], 2)
        self.assertAlmostEqual(metrics["positive_fraction_test"], 0.5)
        self.assertAlmostEqual(metrics["ndwi_one_fraction_test"], 0.25)
        self.assertAlmostEqual(metrics["ndwi_one_fraction_positive_test"], 0.5)
        self.assertAlmostEqual(metrics["median_class_separation_test"], 0.9)
        self.assertAlmostEqual(metrics["test_roc_auc"], 1.0)

    def test_block_split_leakage_is_rejected(self) -> None:
        frame = sample_frame()
        frame.loc[4, "block_id"] = frame.loc[0, "block_id"]
        with self.assertRaisesRegex(Tier1ExplanatoryError, "block/split leakage"):
            sample_explanatory_metrics(frame)


class PlateauTests(unittest.TestCase):
    def test_plateau_uses_contiguous_component_containing_first_global_max(
        self,
    ) -> None:
        response = pd.DataFrame(
            {
                "threshold": [0, 1, 2, 3, 4, 5],
                "iou": [0.94, 0.94, 0.2, 0.95, 0.94, 0.1],
            }
        )
        result = plateau_metrics(
            response,
            tolerance=0.02,
            local_threshold=3.5,
            lolo_threshold=1.0,
            regional_threshold=4.0,
        )
        self.assertEqual(result["plateau_low_threshold"], 3)
        self.assertEqual(result["plateau_high_threshold"], 4)
        self.assertEqual(result["plateau_grid_point_count"], 2)
        self.assertAlmostEqual(result["plateau_width_normalized"], 0.2)
        self.assertTrue(result["local_threshold_inside_plateau"])
        self.assertTrue(result["regional_threshold_inside_plateau"])
        self.assertFalse(result["lolo_threshold_inside_plateau"])


class AssociationTests(unittest.TestCase):
    def test_minimum_eight_lakes_and_leave_one_lake_out_range(self) -> None:
        rows = []
        for reference in ("OPERA", "JRC"):
            for index in range(9):
                row = {
                    "reference": reference,
                    "Hylak_id": index,
                    "Lake_name": f"Lake {index}",
                    "lolo_test_iou": float(index),
                    "lolo_iou_delta": float(index),
                }
                row.update(
                    {variable: float(index) for variable in ASSOCIATION_VARIABLES}
                )
                rows.append(row)
        associations = build_association_table(pd.DataFrame.from_records(rows))
        self.assertEqual(
            len(associations),
            2 * len(ASSOCIATION_OUTCOMES) * len(ASSOCIATION_VARIABLES),
        )
        self.assertTrue(associations["spearman_rho"].eq(1.0).all())
        self.assertTrue(associations["leave_one_lake_out_rho_min"].eq(1.0).all())
        self.assertTrue(associations["leave_one_lake_out_rho_max"].eq(1.0).all())

    def test_fewer_than_eight_complete_lakes_is_not_correlated(self) -> None:
        rows = []
        for reference in ("OPERA", "JRC"):
            for index in range(7):
                row = {
                    "reference": reference,
                    "Hylak_id": index,
                    "Lake_name": f"Lake {index}",
                    "lolo_test_iou": float(index),
                    "lolo_iou_delta": float(index),
                }
                row.update(
                    {variable: float(index) for variable in ASSOCIATION_VARIABLES}
                )
                rows.append(row)
        associations = build_association_table(pd.DataFrame.from_records(rows))
        self.assertTrue(associations["spearman_rho"].isna().all())

    def test_eight_lakes_do_not_claim_a_seven_lake_influence_range(self) -> None:
        rows = []
        for reference in ("OPERA", "JRC"):
            for index in range(8):
                row = {
                    "reference": reference,
                    "Hylak_id": index,
                    "Lake_name": f"Lake {index}",
                    "lolo_test_iou": float(index),
                    "lolo_iou_delta": float(index),
                }
                row.update(
                    {variable: float(index) for variable in ASSOCIATION_VARIABLES}
                )
                rows.append(row)
        associations = build_association_table(pd.DataFrame.from_records(rows))
        self.assertTrue(associations["spearman_rho"].eq(1.0).all())
        self.assertTrue(associations["leave_one_lake_out_rho_min"].isna().all())


@unittest.skipUnless(
    (Path(DEFAULT_DATA_DIR) / "expanded_lake_selection.csv").is_file()
    and Path(DEFAULT_PRIMARY_CACHE_DIR).is_dir()
    and Path(DEFAULT_EXPANDED_CACHE_DIR).is_dir(),
    "ignored Tier 1 local artifacts are unavailable",
)
class LocalArtifactIntegrationTests(unittest.TestCase):
    def test_completed_experiment_reproduces_exactly(self) -> None:
        result = run_tier1_explanatory()
        self.assertEqual(result.table.shape[0], 24)
        self.assertEqual(
            result.table.groupby("reference").size().to_dict(), {"JRC": 12, "OPERA": 12}
        )
        self.assertFalse(
            result.table[
                [
                    "local_test_iou",
                    "lolo_test_iou",
                    "regional_test_iou",
                    "local_test_f1",
                    "lolo_test_f1",
                    "regional_test_f1",
                ]
            ]
            .isna()
            .any()
            .any()
        )
        self.assertEqual(len(result.case_studies), 4)


if __name__ == "__main__":
    unittest.main()
