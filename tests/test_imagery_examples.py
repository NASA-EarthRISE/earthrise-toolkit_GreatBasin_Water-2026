"""Credential-free tests for thumbnail/GIF planning and local assembly."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from regional_analysis.expanded_transfer import expanded_output_paths
from regional_analysis.imagery_examples import (
    ImageryError,
    compose_gif_frame,
    load_regional_threshold,
    rank_point_mass_acquisitions,
    save_gif,
    select_representative_acquisitions,
)


def cache_fixture() -> pd.DataFrame:
    rows = []
    for year, time_ms, count, point_count in (
        (2020, 1_580_000_000_000, 3, 1),
        (2020, 1_590_000_000_000, 5, 2),
        (2021, 1_620_000_000_000, 4, 4),
        (2022, 1_650_000_000_000, 2, 0),
    ):
        for index in range(count):
            rows.append(
                {
                    "acquisition_time": time_ms,
                    "acquisition_date": pd.Timestamp(
                        year=year, month=7, day=1, tz="UTC"
                    ),
                    "sensor": "L8",
                    "block_id": f"b{index % 3}",
                    "index_value": 1.0 if index < point_count else 0.2,
                }
            )
    return pd.DataFrame(rows)


class SceneSelectionTests(unittest.TestCase):
    def test_one_highest_coverage_acquisition_per_year(self) -> None:
        selected = select_representative_acquisitions(
            cache_fixture(), n_frames=5, one_per_year=True
        )

        self.assertEqual(selected["year"].tolist(), [2020, 2021, 2022])
        self.assertEqual(selected.loc[0, "acquisition_time"], 1_590_000_000_000)

    def test_seasonal_window_excludes_higher_coverage_spring_scene(self) -> None:
        spring = int(pd.Timestamp("2020-03-15", tz="UTC").timestamp() * 1000)
        summer = int(pd.Timestamp("2020-07-15", tz="UTC").timestamp() * 1000)
        frame = pd.DataFrame.from_records(
            [
                {
                    "acquisition_time": acquisition,
                    "acquisition_date": pd.to_datetime(
                        acquisition, unit="ms", utc=True
                    ),
                    "sensor": "L8",
                    "block_id": f"b{index}",
                    "index_value": 0.5,
                }
                for acquisition, count in ((spring, 6), (summer, 3))
                for index in range(count)
            ]
        )
        selected = select_representative_acquisitions(
            frame,
            n_frames=1,
            one_per_year=True,
            months=(6, 7, 8, 9),
        )

        self.assertEqual(selected.loc[0, "acquisition_time"], summer)

    def test_point_mass_ranking_uses_count_then_time(self) -> None:
        ranked = rank_point_mass_acquisitions(cache_fixture(), n_scenes=2)

        self.assertEqual(ranked.iloc[0]["point_mass_rows"], 4)
        self.assertEqual(ranked.iloc[1]["point_mass_rows"], 2)


class LocalImageAssemblyTests(unittest.TestCase):
    def test_gif_has_expected_frame_count(self) -> None:
        left = Image.new("RGB", (60, 40), "blue")
        right = Image.new("RGB", (55, 35), "white")
        frames = [
            compose_gif_frame(left, right, heading=f"frame {index}")
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = save_gif(frames, Path(temp) / "change.gif", duration_ms=50)
            with Image.open(path) as image:
                self.assertEqual(image.n_frames, 3)
                self.assertEqual(image.size, frames[0].size)

    def test_regional_threshold_must_be_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = expanded_output_paths(temp, "OPERA", "tier1")
            pd.DataFrame(
                {"Hylak_id": [1, 2], "regional_threshold": [-0.05, -0.05]}
            ).to_csv(paths.regional, index=False)
            self.assertEqual(load_regional_threshold(temp, "OPERA"), -0.05)

            pd.DataFrame(
                {"Hylak_id": [1, 2], "regional_threshold": [-0.05, -0.04]}
            ).to_csv(paths.regional, index=False)
            with self.assertRaisesRegex(ImageryError, "one stable"):
                load_regional_threshold(temp, "OPERA")


if __name__ == "__main__":
    unittest.main()
