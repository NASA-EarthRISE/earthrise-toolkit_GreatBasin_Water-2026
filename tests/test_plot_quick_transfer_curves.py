"""Credential-free tests for quick-transfer curve data and plotting inputs."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import numpy as np
import pandas as pd

from regional_analysis.plot_quick_transfer_curves import (
    resolve_curve_lake_ids,
    resolve_curve_references,
)
from regional_analysis.quick_curve_data import (
    CurveDataError,
    build_curve_summary,
    discrimination_curves,
    load_or_retrieve_curve_samples,
    read_curve_cache,
    threshold_response,
    training_threshold_grid,
    validate_cached_samples,
    write_curve_cache,
)
from regional_analysis.quick_sampling import LakeSpec, QuickSamplingSettings

LAKE = LakeSpec(
    hylak_id=776,
    lake_name="Lake Abert",
    state="OR",
    lake_area=55.0,
    centroid_lon=-120.2,
    centroid_lat=42.6,
)


class CurveSelectionTests(unittest.TestCase):
    def test_reference_and_lake_subsets_keep_canonical_order(self) -> None:
        self.assertEqual(resolve_curve_references(["jrc", "opera"]), ("OPERA", "JRC"))
        self.assertEqual(
            resolve_curve_lake_ids(["829", "Great Salt", "Mono Lake"]),
            (67, 798, 829),
        )

    def test_invalid_or_duplicate_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Curve references"):
            resolve_curve_references(["DW"])
        with self.assertRaisesRegex(ValueError, "Duplicate curve lake"):
            resolve_curve_lake_ids(["67", "Great Salt"])


def prepared_samples() -> pd.DataFrame:
    rows = []
    values = (-0.5, 0.0, 0.5, 1.0)
    labels = (0, 1, 0, 1)
    splits = ("train", "train", "test", "test")
    for index, (value, label, split) in enumerate(
        zip(values, labels, splits, strict=True)
    ):
        block = f"block_{index}"
        rows.append(
            {
                "Hylak_id": 776,
                "Lake_name": "Lake Abert",
                "state": "OR",
                "Lake_area": 55.0,
                "reference": "OPERA",
                "index": "NDWI",
                "index_value": value,
                "label": label,
                "split": split,
                "block_id": block,
                "lake_block_id": f"776:{block}",
                "sensor": "L8",
                "acquisition_time": 1_700_000_000_000 + index,
                "reference_time": 1_700_000_000_000 + index,
                "match_delta_days": 0.0,
                "reference_tile_count": 1,
                "lon": -120.2 + index * 0.001,
                "lat": 42.6 + index * 0.001,
            }
        )
    return pd.DataFrame.from_records(rows)


def test_only_frame(
    values: tuple[float, ...] = (-0.5, 0.0, 0.5, 1.0),
    labels: tuple[int, ...] = (0, 1, 0, 1),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index_value": values,
            "label": labels,
            "split": "test",
        }
    )


class CurveMetricTests(unittest.TestCase):
    def test_threshold_response_is_test_only_and_uses_greater_equal(self) -> None:
        response = threshold_response(test_only_frame(), [-1.0, 0.0, 2.0])
        at_zero = response.set_index("threshold").loc[0.0]
        self.assertAlmostEqual(at_zero["iou"], 2 / 3)
        self.assertAlmostEqual(at_zero["f1"], 0.8)

        mixed = test_only_frame()
        mixed.loc[0, "split"] = "train"
        with self.assertRaisesRegex(CurveDataError, "Expected only split='test'"):
            threshold_response(mixed, [-1.0, 0.0])

    def test_raw_score_orientation_is_not_auto_flipped(self) -> None:
        perfect = discrimination_curves(
            test_only_frame(values=(-1.0, -0.5, 0.5, 1.0), labels=(0, 0, 1, 1))
        )
        inverse = discrimination_curves(
            test_only_frame(values=(1.0, 0.5, -0.5, -1.0), labels=(0, 0, 1, 1))
        )
        self.assertEqual(perfect.roc_auc, 1.0)
        self.assertEqual(perfect.average_precision, 1.0)
        self.assertLess(inverse.roc_auc, 0.5)
        self.assertLess(inverse.average_precision, 0.5)

    def test_training_grid_is_deterministic_and_ignores_test_values(self) -> None:
        lake_one = pd.DataFrame(
            {
                "index_value": [-0.4, 0.2, 1000.0],
                "label": [0, 1, 1],
                "split": ["train", "train", "test"],
            }
        )
        lake_two = pd.DataFrame(
            {
                "index_value": [-0.2, 0.7, -1000.0],
                "label": [0, 1, 0],
                "split": ["train", "train", "test"],
            }
        )
        first = training_threshold_grid({2: lake_two, 1: lake_one}, grid_steps=7)
        poisoned = lake_one.copy()
        poisoned.loc[poisoned["split"] == "test", "index_value"] = -999999.0
        second = training_threshold_grid(
            {1: poisoned.sample(frac=1, random_state=3), 2: lake_two},
            grid_steps=7,
        )
        np.testing.assert_allclose(first, second)
        self.assertEqual(len(first), 7)


class CurveCacheTests(unittest.TestCase):
    def test_cache_round_trip_preserves_split_and_provenance(self) -> None:
        settings = QuickSamplingSettings(min_split_samples=1)
        with TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "opera_776.csv"
            written = write_curve_cache(
                prepared_samples(),
                path,
                LAKE,
                "OPERA",
                settings,
                "test/asset",
                expected_counts=(4, 2, 2),
            )
            loaded = read_curve_cache(
                path,
                LAKE,
                "OPERA",
                settings,
                "test/asset",
                expected_counts=(4, 2, 2),
            )
            self.assertEqual(len(written), len(loaded))
            self.assertEqual(set(loaded["split"]), {"train", "test"})
            np.testing.assert_allclose(written["index_value"], loaded["index_value"])
            with self.assertRaisesRegex(CurveDataError, "hydrolakes_asset"):
                read_curve_cache(
                    path,
                    LAKE,
                    "OPERA",
                    settings,
                    "different/asset",
                    expected_counts=(4, 2, 2),
                )

    def test_cache_rejects_block_split_leakage(self) -> None:
        settings = QuickSamplingSettings(min_split_samples=1)
        frame = prepared_samples()
        frame.loc[1, "block_id"] = frame.loc[2, "block_id"]
        frame.loc[1, "lake_block_id"] = frame.loc[2, "lake_block_id"]
        for column, value in {
            "cache_version": 1,
            "hydrolakes_asset": "test/asset",
            "date_start": "2023-04-04",
            "date_end": "2026-08-09",
            "buffer_km": 5.0,
            "samples_per_image": 20,
            "sample_scale_m": 30,
            "join_window_days": 5,
            "cloud_cover_max": 20.0,
            "block_size_m": 1000,
            "train_fraction": 0.8,
            "random_seed": 42,
        }.items():
            frame[column] = value
        with self.assertRaisesRegex(CurveDataError, "block split leakage"):
            validate_cached_samples(frame, LAKE, "OPERA", settings, "test/asset")

    def test_complete_cache_never_initializes_or_retrieves(self) -> None:
        settings = QuickSamplingSettings(min_split_samples=1)
        initializer = Mock(side_effect=AssertionError("initializer called"))
        retriever = Mock(side_effect=AssertionError("retriever called"))
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            write_curve_cache(
                prepared_samples(),
                directory / "opera_776.csv",
                LAKE,
                "OPERA",
                settings,
                "test/asset",
                expected_counts=(4, 2, 2),
            )
            result = load_or_retrieve_curve_samples(
                [LAKE],
                ["OPERA"],
                settings,
                directory,
                {("OPERA", 776): (4, 2, 2)},
                hydrolakes_asset="test/asset",
                initializer=initializer,
                retriever=retriever,
                cache_only=True,
                verbose=False,
            )
            self.assertIsNone(result.project)
            self.assertEqual(result.created_paths, ())
            initializer.assert_not_called()
            retriever.assert_not_called()

    def test_cache_miss_retrieves_exactly_once(self) -> None:
        settings = QuickSamplingSettings(min_split_samples=1)
        ee_object = object()
        initializer = Mock(return_value=(ee_object, "test-project"))
        retriever = Mock(return_value=prepared_samples())
        with TemporaryDirectory() as directory_name:
            result = load_or_retrieve_curve_samples(
                [LAKE],
                ["OPERA"],
                settings,
                directory_name,
                {("OPERA", 776): (4, 2, 2)},
                hydrolakes_asset="test/asset",
                initializer=initializer,
                retriever=retriever,
                verbose=False,
            )
            self.assertEqual(len(result.created_paths), 1)
            initializer.assert_called_once_with()
            retriever.assert_called_once_with(
                ee_object,
                "test/asset",
                LAKE,
                "OPERA",
                settings,
            )

    def test_failed_scientific_validation_does_not_poison_cache(self) -> None:
        settings = QuickSamplingSettings(min_split_samples=1)
        validator = Mock(side_effect=CurveDataError("summary mismatch"))
        with TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "opera_776.csv"
            with self.assertRaisesRegex(CurveDataError, "summary mismatch"):
                load_or_retrieve_curve_samples(
                    [LAKE],
                    ["OPERA"],
                    settings,
                    directory_name,
                    {("OPERA", 776): (4, 2, 2)},
                    hydrolakes_asset="test/asset",
                    initializer=lambda: (object(), "test-project"),
                    retriever=lambda *_args: prepared_samples(),
                    sample_validator=validator,
                    verbose=False,
                )
            self.assertFalse(path.exists())
            validator.assert_called_once()


class CurveSummaryTests(unittest.TestCase):
    def test_summary_verifies_saved_threshold_metrics(self) -> None:
        samples = test_only_frame()
        local = pd.DataFrame(
            {
                "Hylak_id": [776],
                "n_test": [4],
                "local_threshold": [0.0],
                "local_test_iou": [2 / 3],
            }
        )
        lolo = pd.DataFrame(
            {
                "held_out_Hylak_id": [776],
                "threshold": [0.5],
                "held_out_test_iou": [1 / 3],
            }
        )
        regional = pd.DataFrame(
            {
                "Hylak_id": [776],
                "regional_threshold": [-0.5],
                "regional_test_iou": [0.5],
            }
        )
        summary = build_curve_summary(
            "OPERA", [LAKE], {776: samples}, local, lolo, regional
        )
        self.assertEqual(summary.loc[0, "n_test"], 4)
        self.assertAlmostEqual(summary.loc[0, "local_iou"], 2 / 3)

        stale = local.copy()
        stale["local_test_iou"] = 0.99
        with self.assertRaisesRegex(CurveDataError, "not saved value"):
            build_curve_summary("OPERA", [LAKE], {776: samples}, stale, lolo, regional)


if __name__ == "__main__":
    unittest.main()
