"""Static, credential-free contracts for QGIS threshold integration."""

from __future__ import annotations

import ast
import csv
import unittest
from pathlib import Path

from Lake_Classifer_FD.threshold_profiles import SHIPPED_THRESHOLD_PROFILES

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCK_PATH = REPOSITORY_ROOT / "Lake_Classifer_FD" / "dock.py"


class QgisPluginContractTests(unittest.TestCase):
    @staticmethod
    def _number(node: ast.expr) -> float:
        value = ast.literal_eval(node)
        if not isinstance(value, (int, float)):
            raise TypeError("Expected a numeric rectangle coordinate.")
        return float(value)

    def test_dock_imports_registry_and_no_longer_defines_thresholds(self) -> None:
        tree = ast.parse(DOCK_PATH.read_text(encoding="utf-8"))
        imports_default = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "threshold_profiles"
            and any(alias.name == "DEFAULT_THRESHOLDS" for alias in node.names)
            for node in ast.walk(tree)
        )
        assignments = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
            if isinstance(target, ast.Name)
        }
        self.assertTrue(imports_default)
        self.assertNotIn("DEFAULT_THRESHOLDS", assignments)

    def test_profile_aois_match_existing_qgis_presets(self) -> None:
        tree = ast.parse(DOCK_PATH.read_text(encoding="utf-8"))
        preset_node = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "PRESETS"
                for target in node.targets
            )
        )
        self.assertIsInstance(preset_node, ast.Dict)
        presets = {
            ast.literal_eval(key): tuple(self._number(arg) for arg in value.args)
            for key, value in zip(
                preset_node.keys,
                preset_node.values,
                strict=True,
            )
        }
        expected_presets = {
            "Great Salt Lake, UT": (-113.31, 40.51, -111.85, 41.87),
            "Lake Abert, OR": (-120.61, 42.40, -120.00, 42.85),
        }
        self.assertEqual(presets, expected_presets)
        self.assertEqual(
            {
                profile.preset_key: profile.aoi_bounds_wgs84
                for profile in SHIPPED_THRESHOLD_PROFILES
            },
            expected_presets,
        )

    def test_classification_rule_is_unchanged(self) -> None:
        source = DOCK_PATH.read_text(encoding="utf-8")
        self.assertIn("water = valid & (ndwi >= ndwi_thr)", source)
        self.assertIn("wet = valid & (~water) & (ndvi >= ndvi_thr)", source)
        self.assertIn("wet &= ndmi >= ndmi_thr", source)
        self.assertIn("land = valid & (~water) & (~wet)", source)
        self.assertLess(source.index("cls[water] = 1"), source.index("cls[wet] = 2"))
        self.assertLess(source.index("cls[wet] = 2"), source.index("cls[land] = 3"))
        self.assertIn('(1, "#2166ac", "Open water")', source)
        self.assertIn('(2, "#8a9b54", "Wetland")', source)
        self.assertIn('(3, "#d6cab2", "Land")', source)
        self.assertNotIn("regional_analysis", source)

    def test_profile_source_rows_exist_with_expected_references(self) -> None:
        with (REPOSITORY_ROOT / "data" / "threshold_cv_summary.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            primary = list(csv.DictReader(stream))
        with (REPOSITORY_ROOT / "data" / "ndmi_threshold_cv_all.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            ndmi = list(csv.DictReader(stream))

        for profile in SHIPPED_THRESHOLD_PROFILES:
            ndwi_rows = [
                row
                for row in primary
                if row["site"] == profile.display_name
                and row["index"] == "NDWI"
                and row["reference"] == "OPERA"
            ]
            ndvi_rows = [
                row
                for row in primary
                if row["site"] == profile.display_name
                and row["index"] == "NDVI"
                and row["reference"] == "DW"
            ]
            ndmi_rows = [
                row
                for row in ndmi
                if row["site"] == profile.display_name and row["reference"] == "NWI"
            ]
            self.assertEqual(len(ndwi_rows), 1)
            self.assertEqual(len(ndvi_rows), 1)
            self.assertEqual(len(ndmi_rows), 1)

    def test_plugin_threshold_registry_has_no_regional_dependency(self) -> None:
        registry = (
            REPOSITORY_ROOT / "Lake_Classifer_FD" / "threshold_profiles.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(registry)
        imported_roots = {
            (
                node.module.split(".")[0]
                if isinstance(node, ast.ImportFrom)
                else alias.name.split(".")[0]
            )
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
            if not isinstance(node, ast.ImportFrom) or node.module is not None
        }
        self.assertTrue(
            imported_roots.issubset(
                {"__future__", "collections", "dataclasses", "types", "typing"}
            )
        )
        self.assertNotIn("regional_analysis", registry)


if __name__ == "__main__":
    unittest.main()
