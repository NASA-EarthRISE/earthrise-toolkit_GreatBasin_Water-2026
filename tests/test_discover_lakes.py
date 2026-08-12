"""Credential-free tests for local candidate-table handling."""

from __future__ import annotations

import unittest

import pandas as pd

from regional_analysis.discover_lakes import (
    CORE_OUTPUT_COLUMNS,
    CandidateTableError,
    candidate_output_columns,
    prepare_candidate_table,
    summarize_candidate_table,
    validate_candidate_table,
)

CONFIGURED_STATES = ("AZ", "CA", "UT")


def valid_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Hylak_id": [101, 102],
            "Lake_name": ["Small Lake", None],
            "Lake_area": [25.1, 80.0],
            "Lake_type": [1, 3],
            "Elevation": [1200, 1280],
            "state": ["AZ", "UT"],
            "centroid_lon": [-111.2, -112.4],
            "centroid_lat": [34.2, 40.8],
        }
    )


class CandidateTableTests(unittest.TestCase):
    def test_valid_candidate_table(self) -> None:
        validate_candidate_table(valid_table(), CONFIGURED_STATES)

    def test_missing_required_column_is_rejected(self) -> None:
        table = valid_table().drop(columns="centroid_lat")

        with self.assertRaisesRegex(CandidateTableError, "missing required columns"):
            validate_candidate_table(table, CONFIGURED_STATES)

    def test_duplicate_hylak_id_is_rejected(self) -> None:
        table = valid_table()
        table.loc[1, "Hylak_id"] = table.loc[0, "Hylak_id"]

        with self.assertRaisesRegex(CandidateTableError, "duplicate Hylak_id"):
            validate_candidate_table(table, CONFIGURED_STATES)

    def test_missing_hylak_id_is_rejected(self) -> None:
        table = valid_table()
        table.loc[0, "Hylak_id"] = None

        with self.assertRaisesRegex(CandidateTableError, "missing Hylak_id"):
            validate_candidate_table(table, CONFIGURED_STATES)

    def test_coordinate_ranges_are_validated(self) -> None:
        cases = (
            ("centroid_lon", 180.01, "centroid_lon"),
            ("centroid_lon", -180.01, "centroid_lon"),
            ("centroid_lat", 90.01, "centroid_lat"),
            ("centroid_lat", -90.01, "centroid_lat"),
        )
        for column, value, message in cases:
            with self.subTest(column=column, value=value):
                table = valid_table()
                table.loc[0, column] = value
                with self.assertRaisesRegex(CandidateTableError, message):
                    validate_candidate_table(table, CONFIGURED_STATES)

    def test_invalid_state_is_rejected(self) -> None:
        table = valid_table()
        table.loc[0, "state"] = "TX"

        with self.assertRaisesRegex(CandidateTableError, "outside the configured"):
            validate_candidate_table(table, CONFIGURED_STATES)

    def test_non_positive_area_is_rejected(self) -> None:
        table = valid_table()
        table.loc[0, "Lake_area"] = 0

        with self.assertRaisesRegex(CandidateTableError, "Lake_area must be positive"):
            validate_candidate_table(table, CONFIGURED_STATES)

    def test_output_columns_and_sorting_are_deterministic(self) -> None:
        raw = valid_table().assign(
            Country=["United States", "United States"],
            Depth_avg=[3.0, 4.0],
            Vol_total=[0.2, 0.7],
        )
        raw = raw[
            [
                "Depth_avg",
                "centroid_lat",
                "Lake_name",
                "Hylak_id",
                "Vol_total",
                "state",
                "Lake_type",
                "centroid_lon",
                "Country",
                "Elevation",
                "Lake_area",
            ]
        ]
        available = ("Depth_avg", "Country", "Vol_total")

        prepared = prepare_candidate_table(raw, reversed(available))

        expected_columns = [
            *CORE_OUTPUT_COLUMNS,
            "Country",
            "Vol_total",
            "Depth_avg",
        ]
        self.assertEqual(
            candidate_output_columns(reversed(available)), expected_columns
        )
        self.assertEqual(prepared.columns.tolist(), expected_columns)
        self.assertEqual(prepared["Hylak_id"].tolist(), [102, 101])

    def test_equal_area_sort_uses_hylak_id_tie_break(self) -> None:
        raw = valid_table()
        raw["Lake_area"] = 50.0
        raw = raw.iloc[::-1]

        prepared = prepare_candidate_table(raw)

        self.assertEqual(prepared["Hylak_id"].tolist(), [101, 102])

    def test_hydrolakes_sentinels_are_reported_but_preserved(self) -> None:
        table = valid_table().assign(
            Dis_avg=[-9999, 2.0],
            Res_time=[-1, -9999],
            Slope_100=[-1, 3.2],
            Wshd_area=[-9999, 120.0],
        )
        summary = summarize_candidate_table(table)

        self.assertEqual(
            summary["hydrolakes_sentinel_values"],
            {
                "Dis_avg": {"-9999": 1},
                "Res_time": {"-9999": 1, "-1": 1},
                "Slope_100": {"-1": 1},
                "Wshd_area": {"-9999": 1},
            },
        )
        self.assertEqual(table.loc[0, "Dis_avg"], -9999)


if __name__ == "__main__":
    unittest.main()
