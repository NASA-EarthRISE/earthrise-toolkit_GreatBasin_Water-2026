"""Credential-free tests for regional YAML configuration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from regional_analysis.config import ConfigError, load_config, validate_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def valid_mapping(min_area_km2: float = 25) -> dict[str, object]:
    return {
        "earth_engine": {
            "hydrolakes_asset": "projects/example/hydrolakes",
            "states_asset": "TIGER/2018/States",
        },
        "region": {"states": ["AZ", "CA", "UT"]},
        "selection": {"min_area_km2": min_area_km2},
    }


class ConfigTests(unittest.TestCase):
    def test_repository_yaml_is_valid(self) -> None:
        config = load_config(REPOSITORY_ROOT / "configs" / "western_us.yaml")

        self.assertEqual(config.selection.min_area_km2, 25.0)
        self.assertEqual(
            config.region.states,
            ("AZ", "CA", "CO", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY"),
        )
        self.assertEqual(config.earth_engine.states_asset, "TIGER/2018/States")

    def test_malformed_yaml_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.yaml"
            path.write_text("region:\n  states: [AZ\n", encoding="utf-8")

            with self.assertRaisesRegex(ConfigError, "Invalid YAML"):
                load_config(path)

    def test_missing_required_section_is_rejected(self) -> None:
        mapping = valid_mapping()
        del mapping["selection"]

        with self.assertRaisesRegex(ConfigError, "selection"):
            validate_config(mapping)

    def test_empty_state_list_is_rejected(self) -> None:
        mapping = valid_mapping()
        mapping["region"] = {"states": []}

        with self.assertRaisesRegex(ConfigError, "non-empty YAML list"):
            validate_config(mapping)

    def test_non_positive_minimum_area_is_rejected(self) -> None:
        for value in (0, -1, -0.01):
            with self.subTest(value=value), self.assertRaisesRegex(
                ConfigError, "greater than zero"
            ):
                validate_config(valid_mapping(value))

    def test_non_numeric_minimum_area_is_rejected(self) -> None:
        mapping = valid_mapping()
        mapping["selection"] = {"min_area_km2": "25"}

        with self.assertRaisesRegex(ConfigError, "positive number"):
            validate_config(mapping)


if __name__ == "__main__":
    unittest.main()
