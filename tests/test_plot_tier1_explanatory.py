from __future__ import annotations

import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regional_analysis.plot_tier1_explanatory import (
    Tier1PlotError,
    case_imagery_index_text,
    load_explanatory_products,
    plot_case_studies,
    plot_hydrolakes_context,
    plot_temporal_sample_behavior,
    plot_threshold_geometry,
    plot_transfer_vs_discrimination,
)
from regional_analysis.tier1_explanatory import (
    CASE_STUDIES,
    REFERENCES,
    TIER1_IDS,
)


def synthetic_table() -> pd.DataFrame:
    rows = []
    for reference_index, reference in enumerate(REFERENCES):
        for order, lake_id in enumerate(TIER1_IDS, start=1):
            value = order / 13
            rows.append(
                {
                    "reference": reference,
                    "Hylak_id": lake_id,
                    "Lake_name": f"Lake {lake_id}",
                    "selection_order": order,
                    "tier_group": "original6" if order <= 6 else "tier1_addition",
                    "test_roc_auc": 0.80 + 0.015 * order,
                    "test_ks_statistic": 0.65 + 0.02 * order,
                    "lolo_test_iou": 0.45 + 0.04 * order,
                    "lolo_iou_delta": -0.12 + 0.015 * order,
                    "abs_local_to_regional_threshold": 0.005 + 0.006 * order,
                    "plateau_width_normalized": 0.01 + 0.015 * order,
                    "acquisition_prevalence_sd": 0.1 + value / 10,
                    "positive_fraction_test": 0.1 + value / 2,
                    "water_acquisition_median_ndwi_sd": 0.08 + value / 4,
                    "log10_Lake_area": 2 + value,
                    "Depth_avg": 2 + order,
                    "Elevation": -50 + 150 * order,
                    "log10_Wshd_area": 3 + value,
                    "local_threshold": -0.10 + 0.01 * order,
                    "lolo_threshold": -0.06 + 0.004 * order,
                    "regional_threshold": -0.04 - 0.005 * reference_index,
                    "local_test_iou": 0.55 + 0.03 * order,
                }
            )
    return pd.DataFrame.from_records(rows)


def synthetic_case_data() -> tuple[
    dict[tuple[str, int], pd.DataFrame], dict[tuple[str, int], pd.DataFrame]
]:
    samples = {}
    responses = {}
    thresholds = np.linspace(-0.3, 0.4, 40)
    for reference, lake_id, _hypothesis in CASE_STUDIES:
        values = np.linspace(-0.8, 1.0, 60)
        labels = (values >= -0.05).astype(int)
        samples[(reference, lake_id)] = pd.DataFrame(
            {"index_value": values, "label": labels, "split": "test"}
        )
        responses[(reference, lake_id)] = pd.DataFrame(
            {
                "threshold": thresholds,
                "iou": np.clip(1 - np.abs(thresholds + 0.05), 0, 1),
            }
        )
    return samples, responses


class PlotTests(unittest.TestCase):
    def test_missing_products_fail_clearly(self) -> None:
        with self.assertRaisesRegex(Tier1PlotError, "Missing explanatory outputs"):
            load_explanatory_products("does-not-exist")

    def test_all_plotters_render_synthetic_data(self) -> None:
        table = synthetic_table()
        samples, responses = synthetic_case_data()
        figures = (
            plot_transfer_vs_discrimination(table),
            plot_threshold_geometry(table),
            plot_temporal_sample_behavior(table),
            plot_hydrolakes_context(table),
            plot_case_studies(table, samples, responses),
        )
        try:
            for figure in figures:
                figure.canvas.draw()
                self.assertGreater(figure.get_figwidth(), 0)
                self.assertGreater(figure.get_figheight(), 0)
        finally:
            for figure in figures:
                plt.close(figure)

    def test_imagery_index_is_a_compact_markdown_list(self) -> None:
        text = case_imagery_index_text(synthetic_table())
        link_lines = [line for line in text.splitlines() if line.startswith("- [")]
        self.assertEqual(len(link_lines), 9)
        self.assertIn("static_honey_lake_opera_2023-07-01.png", text)
        self.assertLess(len(text.splitlines()), 20)


if __name__ == "__main__":
    unittest.main()
