"""Credential-free tests for the Phase 0 hydrometry pilot registry."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

from regional_analysis.hydrometry.models import HydrometryVariable, Provider
from regional_analysis.hydrometry.registry import (
    HydrometryRegistryError,
    load_registry,
    validate_registry_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "hydrometry_pilots.yaml"


class HydrometryRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry(CONFIG_PATH)

    def test_registry_contains_exactly_the_five_phase0_pilots(self) -> None:
        self.assertEqual(len(self.registry.profiles), 5)
        self.assertEqual(
            [
                (profile.profile_id, profile.Hylak_id)
                for profile in self.registry.profiles
            ],
            [
                ("great_salt", 67),
                ("lake_abert", 776),
                ("bear_lake", 778),
                ("lake_powell", 802),
                ("flaming_gorge", 784),
            ],
        )
        self.assertEqual(
            len({profile.profile_id for profile in self.registry.profiles}), 5
        )
        self.assertEqual(
            len({profile.Hylak_id for profile in self.registry.profiles}), 5
        )

    def test_great_salt_arms_are_distinct_non_interchangeable_sources(self) -> None:
        profile = self.registry.get_profile("great_salt", strict=True)
        assert profile is not None
        self.assertEqual(
            {source.source_record_id for source in profile.sources},
            {"USGS-10010000", "USGS-10010100"},
        )
        self.assertEqual(
            {source.metadata["arm"] for source in profile.sources}, {"south", "north"}
        )
        self.assertTrue(
            all(
                source.metadata["interchangeable_with_other_arm"] is False
                for source in profile.sources
            )
        )

    def test_lookup_requires_exact_source_and_never_falls_back(self) -> None:
        first = self.registry.get_source(
            "great_salt", "usgs_great_salt_south_arm_level"
        )
        self.assertIsNotNone(first)
        self.assertIsNone(self.registry.get_source("great_salt", "missing"))
        self.assertIsNone(self.registry.get_source("missing", "missing"))
        with self.assertRaises(KeyError):
            self.registry.get_source("great_salt", "missing", strict=True)
        self.assertEqual(
            [source.priority for source in self.registry.ordered_sources("great_salt")],
            [10, 20],
        )

    def test_provider_ids_are_not_hydrolakes_or_swot_ids(self) -> None:
        powell = self.registry.get_profile("lake_powell", strict=True)
        assert powell is not None
        self.assertEqual(powell.Hylak_id, 802)
        self.assertEqual(powell.sources[0].source_record_id, "2362")
        for profile in self.registry.profiles:
            self.assertIsNone(profile.swot.swot_pld_id)
            self.assertEqual(profile.swot.hydrocron_status, "requires_crosswalk")

    def test_config_and_serialized_registry_contain_no_credentials(self) -> None:
        serialized = json.dumps(self.registry.as_public_dict()).lower()
        for forbidden in (
            "api_key",
            "authorization",
            "cookie",
            "password",
            "secret",
            "token",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_registry_is_defensive_and_metadata_is_immutable(self) -> None:
        profile = self.registry.get_profile("great_salt", strict=True)
        assert profile is not None
        with self.assertRaises(TypeError):
            profile.sources[0].metadata["arm"] = "changed"
        public = self.registry.as_public_dict()
        public["profiles"][0]["sources"][0]["metadata"]["arm"] = "changed"
        self.assertEqual(profile.sources[0].metadata["arm"], "south")

    def test_flaming_gorge_bank_storage_is_not_total_storage(self) -> None:
        source = self.registry.get_source(
            "flaming_gorge", "rise_flaming_gorge_bank_storage", strict=True
        )
        assert source is not None
        self.assertIs(source.provider, Provider.USBR_RISE)
        self.assertIs(source.variable, HydrometryVariable.BANK_STORAGE)
        self.assertFalse(source.metadata["confers_absolute_storage_capability"])

    def test_duplicate_source_ids_are_rejected(self) -> None:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        raw["profiles"][0]["sources"][1]["source_id"] = raw["profiles"][0]["sources"][
            0
        ]["source_id"]
        with self.assertRaisesRegex(HydrometryRegistryError, "duplicate source IDs"):
            validate_registry_config(raw)

    def test_credential_fields_are_rejected_during_validation(self) -> None:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        raw["profiles"][0]["sources"][0]["metadata"]["api_key"] = "not-allowed"
        with self.assertRaisesRegex(HydrometryRegistryError, "credential"):
            validate_registry_config(raw)


if __name__ == "__main__":
    unittest.main()
