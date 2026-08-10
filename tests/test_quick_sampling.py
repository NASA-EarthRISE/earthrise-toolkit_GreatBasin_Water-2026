"""Credential-free tests for sample normalization and local optimization."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from regional_analysis.quick_sampling import (
    JOIN_MEASURE_PROPERTY,
    SAMPLE_COLUMNS,
    InsufficientSampleError,
    LakeSpec,
    QuickSamplingError,
    QuickSamplingSettings,
    _matched_image_and_measure,
    _nearest_time_join,
    assign_blocks_and_split,
    cv_optimal_threshold,
    normalize_sample_dataframe,
    optimize_samples,
    utm_epsg_from_lon_lat,
)

LAKE = LakeSpec(
    hylak_id=776,
    lake_name="Lake Abert",
    state="OR",
    lake_area=174.35,
    centroid_lon=-120.2199,
    centroid_lat=42.6519,
)


class FakePropertySource:
    """Minimal computed-object stand-in that records property reads."""

    def __init__(self, properties: dict[str, object]) -> None:
        self.properties = properties
        self.reads: list[str] = []

    def get(self, key: str) -> object:
        self.reads.append(key)
        return self.properties.get(key)


def raw_compute_features_frame() -> pd.DataFrame:
    """Mimic the schema returned by EE's PANDAS_DATAFRAME converter."""

    common = {
        "Hylak_id": 776,
        "Lake_name": "Lake Abert",
        "state": "OR",
        "Lake_area": 174.35,
        "reference": "JRC",
        "index": "NDWI",
        "reference_time": 1_514_764_800_000,
        "match_delta_days": 0.0,
        "reference_tile_count": 1,
        "sensor": "L8",
    }
    return pd.DataFrame.from_records(
        [
            {
                "geo": None,
                "system:index": "second",
                **common,
                "index_value": 0.4,
                "label": 1,
                "acquisition_time": 1_514_851_200_000,
                "lon": -120.20,
                "lat": 42.65,
            },
            {
                "geo": None,
                "system:index": "first",
                **common,
                "index_value": -0.3,
                "label": 0,
                "acquisition_time": 1_514_764_800_000,
                "lon": -120.21,
                "lat": 42.64,
            },
        ]
    )


def synthetic_separable_samples(n_blocks: int = 12) -> pd.DataFrame:
    """Build well-separated locations with both classes inside every block."""

    rows: list[dict[str, object]] = []
    time_ms = 1_600_000_000_000
    for block_index in range(n_blocks):
        # At this latitude 0.03 degrees longitude is over 2 km, while repeated
        # positive/negative rows share one exact location and therefore a block.
        lon = -119.90 + 0.03 * block_index
        lat = 42.00
        for replicate in range(5):
            rows.append(
                {
                    "index_value": -0.40 + replicate * 0.02,
                    "label": 0,
                    "lon": lon,
                    "lat": lat,
                    "acquisition_time": time_ms + block_index * 86_400_000,
                    "sensor": "L8",
                }
            )
            rows.append(
                {
                    "index_value": 0.35 + replicate * 0.02,
                    "label": 1,
                    "lon": lon,
                    "lat": lat,
                    "acquisition_time": time_ms + block_index * 86_400_000,
                    "sensor": "L8",
                }
            )
    return pd.DataFrame.from_records(rows)


class SampleSchemaTests(unittest.TestCase):
    def test_compute_features_schema_is_normalized_and_geometry_is_dropped(
        self,
    ) -> None:
        normalized = normalize_sample_dataframe(raw_compute_features_frame())

        self.assertEqual(
            normalized.columns.tolist(),
            [*SAMPLE_COLUMNS, "acquisition_date"],
        )
        self.assertNotIn("geo", normalized.columns)
        self.assertNotIn("system:index", normalized.columns)
        self.assertEqual(normalized["label"].dtype, np.dtype("int64"))
        self.assertEqual(
            normalized["acquisition_time"].tolist(),
            sorted(normalized["acquisition_time"].tolist()),
        )
        self.assertTrue(isinstance(normalized.loc[0, "acquisition_date"], pd.Timestamp))

    def test_missing_compute_features_property_is_rejected(self) -> None:
        raw = raw_compute_features_frame().drop(columns="lat")

        with self.assertRaisesRegex(
            QuickSamplingError, "Unexpected computeFeatures schema.*lat"
        ):
            normalize_sample_dataframe(raw)

    def test_empty_compute_features_frame_has_stable_schema(self) -> None:
        normalized = normalize_sample_dataframe(pd.DataFrame())

        self.assertTrue(normalized.empty)
        self.assertEqual(
            normalized.columns.tolist(),
            [*SAMPLE_COLUMNS, "acquisition_date"],
        )

    def test_projected_coordinates_are_rejected(self) -> None:
        raw = raw_compute_features_frame()
        raw.loc[0, ["lon", "lat"]] = [653_000, 4_725_000]

        with self.assertRaisesRegex(
            QuickSamplingError, "outside longitude/latitude ranges"
        ):
            normalize_sample_dataframe(raw)


class TemporalJoinTests(unittest.TestCase):
    def test_measure_is_read_from_matched_secondary(self) -> None:
        matched = FakePropertySource({JOIN_MEASURE_PROPERTY: 86_400_000})
        primary = FakePropertySource({"jrc": matched, JOIN_MEASURE_PROPERTY: None})
        fake_ee = SimpleNamespace(Image=lambda value: value)

        resolved, measure = _matched_image_and_measure(fake_ee, primary, "jrc")

        self.assertIs(resolved, matched)
        self.assertEqual(measure, 86_400_000)
        self.assertEqual(primary.reads, ["jrc"])
        self.assertEqual(matched.reads, [JOIN_MEASURE_PROPERTY])

    def test_nearest_join_excludes_unmatched_primary_rows(self) -> None:
        calls: dict[str, object] = {}

        class FakeFilter:
            @staticmethod
            def maxDifference(**kwargs: object) -> str:
                calls["filter"] = kwargs
                return "time-filter"

        class FakeJoinOperation:
            def apply(
                self, primary: object, secondary: object, condition: object
            ) -> str:
                calls["apply"] = (primary, secondary, condition)
                return "joined"

        class FakeJoin:
            @staticmethod
            def saveBest(**kwargs: object) -> FakeJoinOperation:
                calls["save_best"] = kwargs
                return FakeJoinOperation()

        fake_ee = SimpleNamespace(Filter=FakeFilter, Join=FakeJoin)

        result = _nearest_time_join(fake_ee, "landsat", "reference", 31, "jrc")

        self.assertEqual(result, "joined")
        self.assertEqual(
            calls["save_best"],
            {
                "matchKey": "jrc",
                "measureKey": JOIN_MEASURE_PROPERTY,
                "outer": False,
            },
        )
        self.assertEqual(calls["apply"], ("landsat", "reference", "time-filter"))


class SpatialBlockTests(unittest.TestCase):
    def test_block_assignment_is_deterministic_and_blocks_do_not_cross_splits(
        self,
    ) -> None:
        samples = synthetic_separable_samples()
        epsg = utm_epsg_from_lon_lat(LAKE.centroid_lon, LAKE.centroid_lat)

        first = assign_blocks_and_split(samples, epsg, seed=42)
        second = assign_blocks_and_split(samples, epsg, seed=42)

        self.assertEqual(first["block_id"].tolist(), second["block_id"].tolist())
        self.assertEqual(first["split"].tolist(), second["split"].tolist())
        self.assertEqual(first.groupby("block_id")["split"].nunique().max(), 1)
        self.assertEqual(set(first["split"]), {"train", "test"})
        self.assertGreaterEqual(first["block_id"].nunique(), 5)


class ThresholdOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = QuickSamplingSettings(grid_steps=80)
        self.samples = synthetic_separable_samples()

    def test_separable_sample_produces_threshold_and_held_out_metrics(self) -> None:
        summary = optimize_samples(self.samples, LAKE, "JRC", self.settings)

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["n_samples"], len(self.samples))
        self.assertGreaterEqual(summary["n_blocks"], self.settings.min_blocks)
        self.assertLess(summary["threshold_mean"], 0.35)
        self.assertGreater(summary["threshold_mean"], -0.40)
        self.assertEqual(summary["test_iou"], 1.0)
        self.assertEqual(summary["test_f1"], 1.0)
        self.assertEqual(summary["test_precision"], 1.0)
        self.assertEqual(summary["test_recall"], 1.0)

    def test_too_small_sample_is_skipped_cleanly(self) -> None:
        small = self.samples.iloc[: self.settings.min_samples - 1]

        summary = optimize_samples(small, LAKE, "JRC", self.settings)

        self.assertEqual(summary["status"], "skipped")
        self.assertIn("too few samples", summary["skip_reason"])
        self.assertEqual(summary["n_samples"], len(small))

    def test_single_class_sample_is_skipped_cleanly(self) -> None:
        single_class = self.samples.copy()
        single_class["label"] = 0

        summary = optimize_samples(single_class, LAKE, "JRC", self.settings)

        self.assertEqual(summary["status"], "skipped")
        self.assertEqual(
            summary["skip_reason"],
            "sample contains only one label class",
        )

    def test_one_class_grouped_cv_training_fold_is_rejected(self) -> None:
        training = pd.DataFrame(
            {
                "index_value": [-0.4, -0.3, 0.3, 0.4],
                "label": [0, 0, 1, 1],
                "block_id": ["negative", "negative", "positive", "positive"],
            }
        )

        with self.assertRaisesRegex(ValueError, "training fold.*one label class"):
            cv_optimal_threshold(training, n_splits=2)

    def test_grouped_cv_class_insufficiency_skips_cleanly(self) -> None:
        with patch(
            "regional_analysis.quick_sampling.cv_optimal_threshold",
            side_effect=InsufficientSampleError("fold lost one class"),
        ):
            summary = optimize_samples(self.samples, LAKE, "JRC", self.settings)

        self.assertEqual(summary["status"], "skipped")
        self.assertEqual(summary["skip_reason"], "fold lost one class")


if __name__ == "__main__":
    unittest.main()
