"""Credential-free tests for exact-NDWI endpoint diagnosis."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from regional_analysis.point_mass_diagnostics import (
    DIAGNOSTIC_COLUMNS,
    PointMassError,
    _row_key,
    derive_reflectance_and_qa,
    point_mass_summary,
    validate_against_source,
)


def source_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Hylak_id": [776, 776, 776],
            "Lake_name": ["Lake Abert"] * 3,
            "reference": ["OPERA"] * 3,
            "index_value": [1.0, 0.999999, -0.2],
            "label": [1, 1, 0],
            "split": ["train", "test", "test"],
            "sensor": ["L8", "L8", "L9"],
            "acquisition_time": [1_700_000_000_000] * 3,
            "reference_time": [1_700_000_000_000] * 3,
            "lon": [-120.1, -120.2, -120.3],
            "lat": [42.6, 42.7, 42.8],
        }
    )
    frame["row_key"] = _row_key(frame)
    frame["acquisition_date"] = pd.to_datetime(
        frame["acquisition_time"], unit="ms", utc=True
    )
    return frame


def diagnostic_frame() -> pd.DataFrame:
    source = source_frame()
    rows = []
    for index, row in source.iterrows():
        if index == 0:
            green_dn, nir_dn = 9000, 7000
        elif index == 1:
            green_dn, nir_dn = 9000, 7273
        else:
            green_dn, nir_dn = 8000, 9000
        record = {column: 0 for column in DIAGNOSTIC_COLUMNS}
        record.update(
            {
                "Hylak_id": 776,
                "Lake_name": "Lake Abert",
                "reference": "OPERA",
                "index_value": row["index_value"],
                "label": row["label"],
                "acquisition_time": row["acquisition_time"],
                "reference_time": row["acquisition_time"],
                "sensor": row["sensor"],
                "landsat_scene_id": "scene",
                "lon": row["lon"],
                "lat": row["lat"],
                "green_dn": green_dn,
                "nir_dn": nir_dn,
                "green_unclamped": green_dn * 0.0000275 - 0.2,
                "nir_unclamped": nir_dn * 0.0000275 - 0.2,
                "green_reflectance": np.clip(green_dn * 0.0000275 - 0.2, 0, 1),
                "nir_reflectance": np.clip(nir_dn * 0.0000275 - 0.2, 0, 1),
                "qa_pixel": 1 << 7,
                "qa_radsat": 0,
            }
        )
        rows.append(record)
    frame = derive_reflectance_and_qa(pd.DataFrame(rows))
    frame["row_key"] = _row_key(frame)
    return frame


class PointMassSummaryTests(unittest.TestCase):
    def test_exact_one_is_not_near_one(self) -> None:
        summary = point_mass_summary({("OPERA", 776): source_frame()})
        overall = summary[summary["dimension"] == "overall"].iloc[0]

        self.assertEqual(overall["n_rows"], 3)
        self.assertEqual(overall["n_exact_one"], 1)
        self.assertAlmostEqual(overall["exact_one_fraction"], 1 / 3)

    def test_split_counts_are_preserved(self) -> None:
        summary = point_mass_summary({("OPERA", 776): source_frame()})
        split = summary[summary["dimension"] == "split"].set_index("split")

        self.assertEqual(split.loc["train", "n_exact_one"], 1)
        self.assertEqual(split.loc["test", "n_exact_one"], 0)


class ReflectanceDiagnosticTests(unittest.TestCase):
    def test_dn_boundary_and_qa_bits(self) -> None:
        frame = pd.DataFrame(
            {
                "green_dn": [9000, 9000],
                "nir_dn": [7272, 7273],
                "qa_pixel": [(1 << 0) | (1 << 4) | (1 << 5), 1 << 7],
                "qa_radsat": [1, 0],
            }
        )
        result = derive_reflectance_and_qa(frame)

        self.assertTrue(result.loc[0, "nir_clamped_low"])
        self.assertFalse(result.loc[1, "nir_clamped_low"])
        self.assertEqual(result.loc[0, "nir_clamped_local"], 0)
        self.assertGreater(result.loc[1, "nir_clamped_local"], 0)
        self.assertTrue(result.loc[0, "qa_fill"])
        self.assertTrue(result.loc[0, "qa_shadow"])
        self.assertTrue(result.loc[0, "qa_snow"])
        self.assertTrue(result.loc[0, "qa_any_radsat"])
        self.assertTrue(result.loc[1, "qa_water"])

    def test_source_key_mismatch_is_rejected(self) -> None:
        source = source_frame()
        diagnostic = diagnostic_frame()
        diagnostic.loc[0, "lon"] += 0.01
        diagnostic["row_key"] = _row_key(diagnostic)

        with self.assertRaisesRegex(PointMassError, "sample keys differ"):
            validate_against_source(diagnostic, source, reference="OPERA", lake_id=776)

    def test_exact_one_requires_clamped_nir_zero(self) -> None:
        source = source_frame().iloc[[0]].copy()
        diagnostic = diagnostic_frame().iloc[[0]].copy()
        validated = validate_against_source(
            diagnostic, source, reference="OPERA", lake_id=776
        )

        self.assertTrue(validated.loc[0, "nir_clamped_low"])
        self.assertGreater(validated.loc[0, "green_clamped_local"], 0)
        self.assertEqual(validated.loc[0, "ndwi_recomputed"], 1.0)


if __name__ == "__main__":
    unittest.main()
