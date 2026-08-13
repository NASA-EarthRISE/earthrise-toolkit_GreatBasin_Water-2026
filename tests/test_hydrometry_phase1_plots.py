from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

from regional_analysis.hydrometry.capacity_tables import (
    CapacityTableInputRow,
    CapacityTableMetadata,
    normalize_capacity_table,
)
from regional_analysis.hydrometry.plot_phase1 import (
    FIGURE_NAMES,
    Phase1PlotError,
    create_phase1_figures,
    load_phase1_outputs,
    plot_bear_eav_relation,
    plot_pilot_timeseries,
    plot_seasonal_context,
    plot_storage_change_summary,
)

matplotlib.use("Agg", force=True)


def _capacity_table():
    metadata = CapacityTableMetadata(
        lake_name="Bear Lake",
        source_doi="10.5066/P14KZS2D",
        source_title="Synthetic official Bear Lake E-A-V fixture",
        source_file="bear_lake_eav.csv",
        source_version="test-v1",
        source_hash="a" * 64,
        vertical_datum="NAVD88",
        elevation_unit="m",
        area_unit="km2",
        storage_unit="acre_ft",
    )
    rows = (
        CapacityTableInputRow(2, 1795.0, 200.0, 900_000.0),
        CapacityTableInputRow(3, 1805.0, 250.0, 1_500_000.0),
        CapacityTableInputRow(4, 1815.0, 300.0, 2_300_000.0),
    )
    return normalize_capacity_table(rows, metadata, expected_lake_name="Bear Lake")


def _daily_frame() -> pd.DataFrame:
    series = (
        ("great_salt_south_arm", "water_elevation", "agency_observed", 4190.0),
        ("great_salt_north_arm", "water_elevation", "agency_observed", 4188.0),
        ("lake_abert", "water_elevation", "agency_observed", 4250.0),
        ("bear_lake", "water_elevation", "agency_observed", 5920.0),
        (
            "bear_lake",
            "total_storage",
            "official_eav_piecewise_linear",
            1_450_000.0,
        ),
        ("lake_powell", "total_storage", "agency_reported", 8_500_000.0),
        ("flaming_gorge", "total_storage", "agency_reported", 3_200_000.0),
    )
    dates = ("2016-01-01", "2021-01-01", "2026-08-13")
    return pd.DataFrame(
        [
            {
                "subprofile_id": subprofile,
                "metric": metric,
                "availability_status": "available",
                "value": value + index * 1000.0,
                "calendar_date": day,
                "method": method,
            }
            for subprofile, metric, method, value in series
            for index, day in enumerate(dates)
        ]
    )


def _change_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subprofile_id": subprofile,
                "metric": "total_storage",
                "requested_window_days": window,
                "percent_change": multiplier * window / 100.0,
                "absolute_change": multiplier * window * 1000.0,
            }
            for multiplier, subprofile in enumerate(
                ("bear_lake", "lake_powell", "flaming_gorge"), start=1
            )
            for window in (7, 30, 90, 365)
        ]
    )


def _seasonal_frame() -> pd.DataFrame:
    rows = [
        (
            "great_salt_south_arm",
            "Great Salt south arm",
            "water_elevation",
            "agency_observed",
            45.0,
        ),
        (
            "great_salt_north_arm",
            "Great Salt north arm",
            "water_elevation",
            "agency_observed",
            55.0,
        ),
        ("lake_abert", "Lake Abert", "water_elevation", "agency_observed", 35.0),
        ("bear_lake", "Bear Lake", "water_elevation", "agency_observed", 60.0),
        (
            "bear_lake",
            "Bear Lake",
            "surface_area",
            "official_eav_piecewise_linear",
            62.0,
        ),
        (
            "bear_lake",
            "Bear Lake",
            "total_storage",
            "official_eav_piecewise_linear",
            64.0,
        ),
        ("lake_powell", "Lake Powell", "total_storage", "agency_reported", 20.0),
        ("flaming_gorge", "Flaming Gorge", "total_storage", "agency_reported", 75.0),
    ]
    frame = pd.DataFrame(
        rows,
        columns=(
            "subprofile_id",
            "display_name",
            "metric",
            "method",
            "seasonal_percentile",
        ),
    )
    frame["seasonal_status"] = "available"
    unavailable = pd.DataFrame(
        [
            {
                "subprofile_id": "lake_abert",
                "display_name": "Lake Abert",
                "metric": "total_storage",
                "method": "unavailable",
                "seasonal_percentile": None,
                "seasonal_status": "unavailable_for_profile",
            }
        ]
    )
    return pd.concat((frame, unavailable), ignore_index=True)


def _bundle():
    return SimpleNamespace(
        daily=_daily_frame(),
        change_metrics=_change_frame(),
        seasonal_context=_seasonal_frame(),
        capacity_table=_capacity_table(),
        manager_summary={
            "analysis_start": "2016-01-01",
            "analysis_end": "2026-08-13",
        },
    )


class Phase1PlotTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_each_plotter_accepts_phase1_output_schema(self) -> None:
        bundle = _bundle()
        figures = (
            plot_pilot_timeseries(bundle.daily),
            plot_bear_eav_relation(bundle),
            plot_storage_change_summary(bundle.change_metrics),
            plot_seasonal_context(bundle.seasonal_context),
        )
        self.assertEqual([len(figure.axes) for figure in figures], [7, 2, 2, 1])

    def test_create_outputs_png_svg_and_text_free_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            written = create_phase1_figures(_bundle(), output_dir)

            self.assertEqual(len(written), len(FIGURE_NAMES) * 4 + 1)
            for path in written:
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0, path)

            for name in FIGURE_NAMES:
                labeled_png = output_dir / f"{name}.png"
                no_text_png = output_dir / "no_text" / f"{name}.png"
                self.assertEqual(
                    mpimg.imread(labeled_png).shape[:2],
                    mpimg.imread(no_text_png).shape[:2],
                )
                no_text_svg = (output_dir / "no_text" / f"{name}.svg").read_text(
                    encoding="utf-8"
                )
                self.assertNotIn("<text", no_text_svg)

    def test_missing_output_table_fails_clearly(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(Phase1PlotError, "Missing Phase 1 table"),
        ):
            load_phase1_outputs(Path(temporary))

    def test_missing_required_series_fails_clearly(self) -> None:
        daily = _daily_frame()
        daily = daily.loc[~daily["subprofile_id"].eq("lake_powell")]
        with self.assertRaisesRegex(
            Phase1PlotError, "No total_storage rows for lake_powell"
        ):
            plot_pilot_timeseries(daily)


if __name__ == "__main__":
    unittest.main()
