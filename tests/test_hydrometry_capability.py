"""Tests for conservative Phase 0 capability classification."""

from __future__ import annotations

import unittest
from pathlib import Path

from regional_analysis.hydrometry.capability import (
    CapabilityEvidence,
    classify_capability,
)
from regional_analysis.hydrometry.models import (
    CapabilityLevel,
    HydrometryModelError,
    HydrometryVariable,
    SourceQualityClass,
)
from regional_analysis.hydrometry.registry import load_registry

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class CapabilityClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry(
            REPOSITORY_ROOT / "configs" / "hydrometry_pilots.yaml"
        )

    def profile(self, profile_id: str):
        profile = self.registry.get_profile(profile_id, strict=True)
        assert profile is not None
        return profile

    def test_hydrolakes_modeled_volume_does_not_confer_level4(self) -> None:
        record = classify_capability(
            self.profile("lake_abert"),
            CapabilityEvidence(
                water_level_available=True,
                hydrolakes_modeled_volume_present=True,
                potential_relative_storage=True,
                datum_status="NAVD88 source retained; no capacity relation",
            ),
        )
        self.assertEqual(record.highest_current_capability, CapabilityLevel.WATER_LEVEL)
        self.assertEqual(
            record.highest_potential_capability,
            CapabilityLevel.RELATIVE_STORAGE_CHANGE,
        )
        self.assertFalse(record.absolute_storage_available)

    def test_great_salt_remains_blocked_by_datum_and_arm_constraints(self) -> None:
        record = classify_capability(
            self.profile("great_salt"),
            CapabilityEvidence(
                water_level_available=True,
                surveyed_bathymetry_available=True,
                source_vertical_datum_known=True,
                vertical_relation_documented=False,
                multiple_arm_constraint_resolved=False,
                potential_surveyed_bathymetry=True,
                potential_quantitative_3d=True,
                datum_status="blocked: NGVD29 levels versus NAVD88 bathymetry",
            ),
        )
        self.assertEqual(record.highest_current_capability, CapabilityLevel.WATER_LEVEL)
        self.assertEqual(
            record.highest_potential_capability, CapabilityLevel.QUANTITATIVE_3D
        )
        self.assertFalse(record.absolute_storage_available)
        self.assertIn(
            "independent water-body arm handling", record.missing_requirements
        )

    def test_direct_validated_total_storage_confers_level4(self) -> None:
        record = classify_capability(
            self.profile("flaming_gorge"),
            CapabilityEvidence(
                direct_authoritative_storage_available=True,
                direct_storage_variable=HydrometryVariable.STORAGE,
                datum_status="not required for direct agency storage",
            ),
        )
        self.assertEqual(
            record.highest_current_capability, CapabilityLevel.ABSOLUTE_STORAGE
        )
        self.assertTrue(record.absolute_storage_available)
        self.assertIs(
            record.source_quality_class,
            SourceQualityClass.DIRECT_AUTHORITATIVE_STORAGE,
        )

    def test_direct_storage_does_not_inherit_stage_relation_blockers(self) -> None:
        record = classify_capability(
            self.profile("flaming_gorge"),
            CapabilityEvidence(
                water_level_available=True,
                direct_authoritative_storage_available=True,
                direct_storage_variable=HydrometryVariable.STORAGE,
                official_capacity_relation_available=True,
                source_vertical_datum_known=False,
                vertical_relation_documented=False,
                storage_computation_implemented=False,
                datum_status="direct storage validated; elevation datum unresolved",
            ),
        )
        self.assertEqual(record.missing_requirements, ())

    def test_bank_storage_cannot_be_declared_direct_total_storage(self) -> None:
        with self.assertRaisesRegex(HydrometryModelError, "bank storage"):
            CapabilityEvidence(
                direct_authoritative_storage_available=True,
                direct_storage_variable=HydrometryVariable.BANK_STORAGE,
            )

    def test_unknown_datum_blocks_stage_capacity_level4(self) -> None:
        record = classify_capability(
            self.profile("bear_lake"),
            CapabilityEvidence(
                water_level_available=True,
                official_capacity_relation_available=True,
                storage_computation_implemented=True,
                source_vertical_datum_known=False,
                vertical_relation_documented=False,
                datum_status="unknown",
            ),
        )
        self.assertEqual(record.highest_current_capability, CapabilityLevel.WATER_LEVEL)
        self.assertFalse(record.absolute_storage_available)

    def test_missing_bathymetry_does_not_prevent_level2(self) -> None:
        record = classify_capability(
            self.profile("lake_abert"),
            CapabilityEvidence(
                water_level_available=True,
                datum_status="NAVD88 retained",
            ),
        )
        self.assertTrue(record.water_level_available)
        self.assertEqual(record.highest_current_capability, CapabilityLevel.WATER_LEVEL)
        self.assertFalse(record.absolute_storage_available)


if __name__ == "__main__":
    unittest.main()
