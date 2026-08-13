"""Tests for normalized hydrometry and vertical-reference models."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

from regional_analysis.hydrometry.models import (
    HeightType,
    HydrometryModelError,
    HydrometryVariable,
    NormalizationStatus,
    NormalizedTimeSeriesRecord,
    Provider,
    VerticalReference,
    vertical_reference_supports_absolute_storage,
)


class HydrometryModelTests(unittest.TestCase):
    def test_local_gage_height_cannot_be_normalized_without_offset(self) -> None:
        with self.assertRaisesRegex(HydrometryModelError, "published gage"):
            VerticalReference(
                height_type=HeightType.LOCAL_GAGE_HEIGHT,
                source_vertical_datum="local gage datum",
                source_vertical_units="ft",
                target_vertical_datum="NAVD88",
                transformation_method="published gage offset",
                normalization_status=NormalizationStatus.NORMALIZED,
            )

    def test_same_units_do_not_make_different_datums_compatible(self) -> None:
        reference = VerticalReference(
            height_type=HeightType.ORTHOMETRIC_HEIGHT,
            source_vertical_datum="NGVD29",
            source_vertical_units="ft",
            target_vertical_datum="NAVD88",
            normalization_status=NormalizationStatus.POSSIBLE_NOT_IMPLEMENTED,
        )
        self.assertFalse(vertical_reference_supports_absolute_storage(reference))

    def test_unknown_datum_blocks_absolute_storage_relation(self) -> None:
        reference = VerticalReference(
            height_type=HeightType.UNKNOWN,
            source_vertical_datum=None,
            source_vertical_units="ft",
            normalization_status=NormalizationStatus.BLOCKED,
        )
        self.assertFalse(vertical_reference_supports_absolute_storage(reference))

    def test_blocked_status_overrides_matching_datum_labels(self) -> None:
        reference = VerticalReference(
            height_type=HeightType.ORTHOMETRIC_HEIGHT,
            source_vertical_datum="NAVD88",
            source_vertical_units="ft",
            target_vertical_datum="NAVD88",
            normalization_status=NormalizationStatus.BLOCKED,
        )
        self.assertFalse(vertical_reference_supports_absolute_storage(reference))

    def test_negative_elevation_is_valid(self) -> None:
        reference = VerticalReference(
            height_type=HeightType.ORTHOMETRIC_HEIGHT,
            source_vertical_datum="NAVD88",
            source_vertical_units="ft",
            target_vertical_datum="NAVD88",
            normalization_status=NormalizationStatus.SOURCE_VALUE_RETAINED,
        )
        record = NormalizedTimeSeriesRecord(
            profile_id="example",
            Hylak_id=1,
            display_name="Below sea level example",
            provider=Provider.USGS_WATERDATA,
            source_id="example-level",
            source_record_id="USGS-00000000",
            source_item_id=None,
            variable=HydrometryVariable.WATER_LEVEL,
            value=-71.0,
            unit="ft",
            timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
            quality_status="approved",
            provisional=False,
            approval_status="approved",
            height_type=HeightType.ORTHOMETRIC_HEIGHT,
            vertical_datum=reference,
            retrieved_at_utc=datetime(2026, 1, 2, tzinfo=UTC),
            source_version="fixture-v1",
            source_url_public="https://example.test/public",
            raw_parameter_code="example",
            raw_statistic_code="daily_mean",
            raw_metadata={"site_name": "example"},
        )
        self.assertEqual(record.value, -71.0)
        self.assertTrue(vertical_reference_supports_absolute_storage(reference))
        self.assertEqual(
            record.as_public_dict()["timestamp_utc"], "2026-01-01T00:00:00Z"
        )

    def test_normalized_record_rejects_sensitive_raw_metadata(self) -> None:
        reference = VerticalReference(
            height_type=HeightType.UNKNOWN,
            source_vertical_datum=None,
            source_vertical_units=None,
        )
        with self.assertRaisesRegex(HydrometryModelError, "credential"):
            NormalizedTimeSeriesRecord(
                profile_id="example",
                Hylak_id=1,
                display_name="Example",
                provider=Provider.HYDROCRON,
                source_id="example",
                source_record_id="record",
                source_item_id=None,
                variable=HydrometryVariable.WATER_LEVEL,
                value=1.0,
                unit="m",
                timestamp_utc=datetime.now(UTC),
                quality_status="unknown",
                provisional=None,
                approval_status=None,
                height_type=HeightType.UNKNOWN,
                vertical_datum=reference,
                retrieved_at_utc=datetime.now(UTC),
                source_version=None,
                source_url_public="https://example.test",
                raw_parameter_code=None,
                raw_statistic_code=None,
                raw_metadata={"access_token": "not-allowed"},
            )


if __name__ == "__main__":
    unittest.main()
