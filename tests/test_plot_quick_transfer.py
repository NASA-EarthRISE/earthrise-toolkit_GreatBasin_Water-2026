"""Credential-free tests for the local quick-transfer plotting helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regional_analysis.plot_quick_transfer import (
    LAKE_IDS,
    LAKES,
    PlotDataError,
    build_verification_table,
    load_plot_data,
    save_figure_with_no_text,
)


def _write_reference_tables(
    directory: Path,
    reference: str,
    *,
    omit_lake_id: int | None = None,
    drop_local_column: str | None = None,
) -> None:
    rows = [item for item in reversed(LAKES) if item[0] != omit_lake_id]
    local_records = []
    lolo_records = []
    regional_records = []
    for position, (lake_id, lake_name) in enumerate(rows):
        local_threshold = -0.08 + position * 0.01
        local_iou = 0.80 + position * 0.02
        lolo_threshold = -0.04 + position * 0.002
        lolo_iou = local_iou - 0.01
        local_records.append(
            {
                "reference": reference,
                "Hylak_id": lake_id,
                "Lake_name": lake_name,
                "local_threshold": local_threshold,
                "local_test_iou": local_iou,
            }
        )
        lolo_records.append(
            {
                "reference": reference,
                "held_out_Hylak_id": lake_id,
                "held_out_lake": lake_name,
                "threshold": lolo_threshold,
                "held_out_test_iou": lolo_iou,
                "local_threshold": local_threshold,
                "local_test_iou": local_iou,
                "iou_delta_vs_local": lolo_iou - local_iou,
            }
        )
        regional_records.append(
            {
                "reference": reference,
                "Hylak_id": lake_id,
                "Lake_name": lake_name,
                "regional_threshold": -0.035,
            }
        )

    prefix = directory / f"western_quick_transfer_{reference.lower()}"
    local = pd.DataFrame.from_records(local_records)
    if drop_local_column is not None:
        local = local.drop(columns=drop_local_column)
    local.to_csv(prefix.with_name(prefix.name + "_local.csv"), index=False)
    pd.DataFrame.from_records(lolo_records).to_csv(
        prefix.with_name(prefix.name + "_lolo.csv"), index=False
    )
    pd.DataFrame.from_records(regional_records).to_csv(
        prefix.with_name(prefix.name + "_regional.csv"), index=False
    )

    matrix_ids = [lake_id for lake_id, _ in rows]
    iou = pd.DataFrame(
        0.75,
        index=pd.Index(matrix_ids, name="source_Hylak_id"),
        columns=matrix_ids,
    )
    f1 = pd.DataFrame(
        0.82,
        index=pd.Index(matrix_ids, name="source_Hylak_id"),
        columns=matrix_ids,
    )
    if omit_lake_id is None:
        iou.loc[67, 776] = 0.12
        iou.loc[776, 67] = 0.87
    iou.to_csv(prefix.with_name(prefix.name + "_cross_matrix_iou.csv"))
    f1.to_csv(prefix.with_name(prefix.name + "_cross_matrix_f1.csv"))


class PlotDataTests(unittest.TestCase):
    def test_load_orders_lakes_and_preserves_matrix_orientation(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            for reference in ("OPERA", "JRC"):
                _write_reference_tables(directory, reference)

            data = load_plot_data(directory)
            verification = build_verification_table(data)

            self.assertEqual(tuple(data["OPERA"].cross_iou.index), LAKE_IDS)
            self.assertEqual(tuple(data["OPERA"].cross_iou.columns), LAKE_IDS)
            self.assertEqual(data["OPERA"].cross_iou.loc[67, 776], 0.12)
            self.assertEqual(data["OPERA"].cross_iou.loc[776, 67], 0.87)
            self.assertEqual(
                verification["lake"].tolist()[::2],
                [lake_name for _, lake_name in LAKES],
            )
            self.assertEqual(
                verification["reference"].tolist(),
                ["OPERA", "JRC"] * len(LAKES),
            )
            self.assertTrue(
                np.allclose(
                    verification["IoU delta"].to_numpy(dtype=float),
                    -0.01,
                    atol=1e-12,
                )
            )

    def test_missing_lake_is_rejected_clearly(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            _write_reference_tables(directory, "OPERA", omit_lake_id=829)
            _write_reference_tables(directory, "JRC")

            with self.assertRaisesRegex(PlotDataError, "wrong lake set"):
                load_plot_data(directory)

    def test_missing_required_column_is_rejected_clearly(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            _write_reference_tables(
                directory,
                "OPERA",
                drop_local_column="local_threshold",
            )
            _write_reference_tables(directory, "JRC")

            with self.assertRaisesRegex(PlotDataError, "local_threshold"):
                load_plot_data(directory)

    def test_no_text_export_keeps_dimensions_and_removes_svg_text(self) -> None:
        with TemporaryDirectory() as directory_name:
            destination = Path(directory_name)
            fig, axis = plt.subplots(figsize=(3.0, 2.0), constrained_layout=True)
            axis.plot([0, 1], [0, 1], label="line")
            axis.set_title("Title")
            axis.set_title("Left title", loc="left")
            axis.set_xlabel("X label")
            axis.set_ylabel("Y label")
            axis.annotate("annotation", (0.5, 0.5))
            axis.legend()

            save_figure_with_no_text(fig, destination, "test_figure")
            normal_png = destination / "test_figure.png"
            no_text_png = destination / "no_text" / "test_figure.png"
            normal_svg = destination / "test_figure.svg"
            no_text_svg = destination / "no_text" / "test_figure.svg"

            for path in (normal_png, no_text_png, normal_svg, no_text_svg):
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(
                mpimg.imread(normal_png).shape, mpimg.imread(no_text_png).shape
            )
            self.assertIn("<text", normal_svg.read_text(encoding="utf-8"))
            self.assertNotIn("<text", no_text_svg.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
