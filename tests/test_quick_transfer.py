"""Credential-free tests for local cross-lake threshold transfer."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from regional_analysis.quick_sampling import LakeSpec
from regional_analysis.quick_transfer import (
    PreparedLake,
    TransferError,
    cross_application_matrices,
    evaluate_pooled_regional,
    leave_one_lake_out,
    load_transfer_tables,
    select_macro_threshold,
    transfer_output_paths,
)


def training_frame(
    lake_id: int,
    points: list[tuple[float, int]],
    *,
    repeats: int = 1,
) -> pd.DataFrame:
    rows = [
        {
            "Hylak_id": lake_id,
            "index_value": value,
            "label": label,
            "split": "train",
        }
        for _ in range(repeats)
        for value, label in points
    ]
    return pd.DataFrame.from_records(rows)


def prepared_lake(
    lake_id: int,
    train: list[tuple[float, int]],
    test: list[tuple[float, int]],
) -> PreparedLake:
    rows: list[dict[str, object]] = []
    for split, points in (("train", train), ("test", test)):
        for index, (value, label) in enumerate(points):
            rows.append(
                {
                    "Hylak_id": lake_id,
                    "Lake_name": f"Lake {lake_id}",
                    "reference": "OPERA",
                    "index_value": value,
                    "label": label,
                    "split": split,
                    "block_id": f"{split}_{index}",
                }
            )
    lake = LakeSpec(
        hylak_id=lake_id,
        lake_name=f"Lake {lake_id}",
        state="NV",
        lake_area=100.0,
        centroid_lon=-119.0,
        centroid_lat=40.0,
    )
    return PreparedLake(
        lake=lake,
        reference="OPERA",
        samples=pd.DataFrame.from_records(rows),
        local_summary={},
        sample_retrieval_seconds=1.0,
        optimization_seconds=0.1,
    )


def local_summary(lake_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Hylak_id": lake_ids,
            "local_threshold": [0.25] * len(lake_ids),
            "local_test_iou": [1.0] * len(lake_ids),
        }
    )


class MacroThresholdTests(unittest.TestCase):
    def test_equal_lake_weighting_ignores_unequal_sample_counts(self) -> None:
        lake_101 = training_frame(101, [(0.0, 0), (0.5, 1)])
        small_202 = training_frame(202, [(0.5, 0), (1.0, 1)])
        large_202 = training_frame(202, [(0.5, 0), (1.0, 1)], repeats=100)

        small = select_macro_threshold(
            {101: lake_101, 202: small_202},
            candidate_thresholds=[0.25, 0.75],
        )
        unequal = select_macro_threshold(
            {101: lake_101, 202: large_202},
            candidate_thresholds=[0.25, 0.75],
        )

        self.assertEqual(small.threshold, 0.25)
        self.assertEqual(unequal.threshold, 0.25)
        self.assertEqual(small.macro_iou, 0.75)
        self.assertEqual(unequal.macro_iou, 0.75)
        self.assertEqual(unequal.per_lake_iou, {101: 1.0, 202: 0.5})

    def test_selection_is_deterministic_and_uses_lowest_tie(self) -> None:
        first = training_frame(101, [(0.0, 0), (1.0, 1)])
        second = training_frame(202, [(0.0, 0), (1.0, 1)])

        expected = select_macro_threshold(
            {101: first, 202: second}, candidate_thresholds=[0.75, 0.25]
        )
        shuffled = select_macro_threshold(
            {
                202: second.sample(frac=1, random_state=9),
                101: first.sample(frac=1, random_state=4),
            },
            candidate_thresholds=[0.25, 0.75],
        )

        self.assertEqual(expected.threshold, 0.25)
        self.assertEqual(shuffled.threshold, expected.threshold)
        self.assertEqual(shuffled.macro_iou, expected.macro_iou)

    def test_insufficient_or_single_class_lakes_are_rejected(self) -> None:
        valid = training_frame(101, [(0.0, 0), (1.0, 1)])
        single_class = training_frame(202, [(0.0, 0), (1.0, 0)])

        with self.assertRaisesRegex(TransferError, "at least two lakes"):
            select_macro_threshold({101: valid})
        with self.assertRaisesRegex(TransferError, "lake 202.*one class"):
            select_macro_threshold({101: valid, 202: single_class})

    def test_selector_rejects_test_rows_in_fitting_input(self) -> None:
        first = training_frame(101, [(0.0, 0), (1.0, 1)])
        second = training_frame(202, [(0.0, 0), (1.0, 1)])
        second.loc[second.index[0], "split"] = "test"

        with self.assertRaisesRegex(TransferError, "non-training observations"):
            select_macro_threshold({101: first, 202: second})


class CrossApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prepared = {
            101: prepared_lake(
                101,
                [(0.0, 0), (0.5, 1)],
                [(0.0, 0), (0.5, 1)],
            ),
            202: prepared_lake(
                202,
                [(0.5, 0), (1.0, 1)],
                [(0.5, 0), (1.0, 1)],
            ),
        }

    def test_matrix_rows_are_sources_and_columns_are_targets(self) -> None:
        iou, f1 = cross_application_matrices(
            self.prepared, {101: 0.25, 202: 0.75}, [101, 202]
        )

        self.assertEqual(iou.index.tolist(), [101, 202])
        self.assertEqual(iou.columns.tolist(), [101, 202])
        self.assertEqual(iou.index.name, "source_Hylak_id")
        self.assertEqual(iou.columns.name, "target_Hylak_id")
        np.testing.assert_allclose(iou.to_numpy(), [[1.0, 0.5], [0.0, 1.0]])
        np.testing.assert_allclose(f1.to_numpy(), [[1.0, 2 / 3], [0.0, 1.0]])

    def test_cross_application_uses_only_target_test_rows(self) -> None:
        before, _ = cross_application_matrices(
            self.prepared, {101: 0.25, 202: 0.75}, [101, 202]
        )
        poisoned = dict(self.prepared)
        poisoned[202] = prepared_lake(
            202,
            [(-100.0, 1), (100.0, 0)] * 100,
            [(0.5, 0), (1.0, 1)],
        )
        after, _ = cross_application_matrices(
            poisoned, {101: 0.25, 202: 0.75}, [101, 202]
        )

        pd.testing.assert_series_equal(before[202], after[202])


class RegionalAndLoloTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prepared = {
            101: prepared_lake(
                101,
                [(0.0, 0), (0.5, 1)],
                [(0.0, 0), (0.5, 1)],
            ),
            202: prepared_lake(
                202,
                [(0.5, 0), (1.0, 1)],
                [(0.5, 0), (1.0, 1)],
            ),
            303: prepared_lake(
                303,
                [(0.1, 0), (0.9, 1)],
                [(0.1, 0), (0.9, 1)],
            ),
        }
        self.local = local_summary([101, 202, 303])

    def test_pooled_fit_ignores_every_test_split(self) -> None:
        fit_before, _ = evaluate_pooled_regional(
            self.prepared, [101, 202, 303], "OPERA", self.local, grid_steps=31
        )
        poisoned = {
            lake_id: prepared_lake(
                lake_id,
                list(
                    zip(
                        prepared.samples.query("split == 'train'")["index_value"],
                        prepared.samples.query("split == 'train'")["label"],
                        strict=True,
                    )
                ),
                [(-100.0, 1), (100.0, 0)] * 100,
            )
            for lake_id, prepared in self.prepared.items()
        }
        fit_after, _ = evaluate_pooled_regional(
            poisoned, [101, 202, 303], "OPERA", self.local, grid_steps=31
        )

        self.assertEqual(fit_before.threshold, fit_after.threshold)
        self.assertEqual(fit_before.macro_iou, fit_after.macro_iou)

    def test_lolo_excludes_held_out_lake_from_fit(self) -> None:
        with patch(
            "regional_analysis.quick_transfer.select_macro_threshold",
            wraps=select_macro_threshold,
        ) as selector:
            baseline = leave_one_lake_out(
                self.prepared,
                [101, 202, 303],
                "OPERA",
                self.local,
                grid_steps=31,
            )
        for held_out_id, call in zip(
            baseline["held_out_Hylak_id"], selector.call_args_list, strict=True
        ):
            fitting = call.args[0]
            self.assertNotIn(held_out_id, fitting)
            self.assertTrue(
                all(frame["split"].eq("train").all() for frame in fitting.values())
            )
        poisoned = dict(self.prepared)
        poisoned[303] = prepared_lake(
            303,
            [(-100.0, 0), (100.0, 1)] * 100,
            [(0.1, 0), (0.9, 1)],
        )
        changed = leave_one_lake_out(
            poisoned, [101, 202, 303], "OPERA", self.local, grid_steps=31
        )

        baseline_303 = baseline.set_index("held_out_Hylak_id").loc[303]
        changed_303 = changed.set_index("held_out_Hylak_id").loc[303]
        self.assertEqual(baseline_303["threshold"], changed_303["threshold"])
        self.assertEqual(
            baseline_303["held_out_test_iou"],
            changed_303["held_out_test_iou"],
        )
        for row in baseline.itertuples(index=False):
            training_ids = {int(value) for value in row.training_Hylak_ids.split(",")}
            self.assertNotIn(row.held_out_Hylak_id, training_ids)
            self.assertEqual(len(training_ids), row.n_training_lakes)

    def test_lolo_rejects_fewer_than_three_lakes(self) -> None:
        with self.assertRaisesRegex(TransferError, "at least three lakes"):
            leave_one_lake_out(
                {101: self.prepared[101], 202: self.prepared[202]},
                [101, 202],
                "OPERA",
                self.local,
            )


class OutputContractTests(unittest.TestCase):
    def test_matrix_ids_roundtrip_as_integers(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = transfer_output_paths(output_dir, "OPERA")
            pd.DataFrame({"Hylak_id": [101]}).to_csv(paths.local, index=False)
            matrix = pd.DataFrame(
                [[1.0, 0.5], [0.0, 1.0]],
                index=pd.Index([101, 202], name="source_Hylak_id"),
                columns=pd.Index([101, 202], name="target_Hylak_id"),
            )
            matrix.to_csv(paths.cross_iou)
            matrix.to_csv(paths.cross_f1)
            pd.DataFrame({"Hylak_id": [101]}).to_csv(paths.regional, index=False)
            pd.DataFrame({"held_out_Hylak_id": [101]}).to_csv(paths.lolo, index=False)

            loaded = load_transfer_tables(output_dir, "OPERA")

        self.assertEqual(loaded["cross_iou"].index.tolist(), [101, 202])
        self.assertEqual(loaded["cross_iou"].columns.tolist(), [101, 202])
        self.assertEqual(loaded["cross_iou"].index.name, "source_Hylak_id")
        self.assertEqual(loaded["cross_iou"].columns.name, "target_Hylak_id")

    def test_dynamic_world_output_paths_are_rejected(self) -> None:
        with self.assertRaisesRegex(TransferError, "OPERA and JRC only"):
            transfer_output_paths("data/regional", "DW")


if __name__ == "__main__":
    unittest.main()
