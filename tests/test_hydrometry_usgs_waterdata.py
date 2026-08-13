from __future__ import annotations

import inspect
import json
import unittest
from copy import deepcopy
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import Mock

from regional_analysis.hydrometry.adapters import usgs_waterdata
from regional_analysis.hydrometry.adapters.usgs_waterdata import (
    SeriesSelectionStatus,
    UsgsDataKind,
    UsgsWaterDataAdapter,
    parse_monitoring_location,
    parse_observation,
    parse_series_metadata,
    select_water_level_series,
)
from regional_analysis.hydrometry.models import (
    HeightType,
    HydrometryVariable,
    NormalizationStatus,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "hydrometry"


def fixture(name: str):
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def parsed_series():
    payload = fixture("usgs_series.json")
    return tuple(
        parse_series_metadata(
            feature,
            expected_location_id="USGS-10010000",
            source_url_public="https://api.waterdata.usgs.gov/ogcapi/v0/collections/time-series-metadata/items",
        )
        for feature in payload["features"]
    )


class UsgsParsingTests(unittest.TestCase):
    def test_monitoring_location_retains_ngvd29(self) -> None:
        location = parse_monitoring_location(
            fixture("usgs_monitoring_location.json"),
            expected_location_id="USGS-10010000",
            source_url_public="https://api.waterdata.usgs.gov/ogcapi/v0/collections/monitoring-locations/items/USGS-10010000",
        )
        self.assertEqual(location.vertical_datum, "NGVD29")
        self.assertEqual(location.monitoring_location_number, "10010000")
        self.assertIn("GREAT SALT LAKE", location.name)

    def test_discovers_daily_continuous_and_keeps_gage_height_distinct(self) -> None:
        daily, continuous, gage = parsed_series()
        self.assertIs(daily.data_kind, UsgsDataKind.DAILY)
        self.assertIs(continuous.data_kind, UsgsDataKind.CONTINUOUS)
        self.assertEqual(daily.vertical_datum, "NGVD29")
        self.assertIs(daily.variable, HydrometryVariable.WATER_LEVEL)
        self.assertIs(gage.variable, HydrometryVariable.GAGE_HEIGHT)
        self.assertIs(gage.height_type, HeightType.LOCAL_GAGE_HEIGHT)
        self.assertFalse(gage.is_absolute_water_level)

    def test_selection_prefers_validated_daily_series_not_parameter_code_alone(
        self,
    ) -> None:
        selection = select_water_level_series(
            parsed_series(), expected_unit="ft", expected_vertical_datum="NGVD29"
        )
        self.assertIs(selection.status, SeriesSelectionStatus.SELECTED)
        self.assertEqual(selection.selected.time_series_id, "gsl-daily-62614")

    def test_navd88_is_retained_and_ambiguous_series_requires_review(self) -> None:
        payload = fixture("usgs_series.json")["features"][0]
        payload["id"] = "navd-a"
        payload["properties"]["parameter_description"] = (
            "Lake or reservoir water surface elevation above NAVD 1988, feet"
        )
        first = parse_series_metadata(
            payload,
            expected_location_id="USGS-10010000",
            source_url_public="https://api.waterdata.usgs.gov/ogcapi/v0/test",
        )
        payload["id"] = "navd-b"
        second = parse_series_metadata(
            payload,
            expected_location_id="USGS-10010000",
            source_url_public="https://api.waterdata.usgs.gov/ogcapi/v0/test",
        )
        selection = select_water_level_series(
            (first, second), expected_unit="ft", expected_vertical_datum="NAVD88"
        )
        self.assertEqual(first.vertical_datum, "NAVD88")
        self.assertIs(selection.status, SeriesSelectionStatus.REQUIRES_REVIEW)
        self.assertIsNone(selection.selected)
        self.assertEqual(len(selection.candidates), 2)

    def test_unknown_or_wrong_datum_is_not_guessed(self) -> None:
        selection = select_water_level_series(
            parsed_series(), expected_unit="ft", expected_vertical_datum="NAVD88"
        )
        self.assertIs(selection.status, SeriesSelectionStatus.REQUIRES_REVIEW)
        self.assertIsNone(selection.selected)

    def test_v0_null_parameter_name_falls_back_to_required_description(self) -> None:
        payload = fixture("usgs_series.json")["features"][0]
        payload["properties"]["parameter_name"] = None
        payload["properties"]["parameter"] = None
        parsed = parse_series_metadata(
            payload,
            expected_location_id="USGS-10010000",
            source_url_public="https://api.waterdata.usgs.gov/ogcapi/v0/test",
        )
        self.assertEqual(parsed.parameter_name, parsed.parameter_description)
        self.assertIs(parsed.variable, HydrometryVariable.WATER_LEVEL)

    def test_observation_retains_approval_and_provisional_status(self) -> None:
        series = parsed_series()[0]
        features = fixture("usgs_daily.json")["features"]
        provisional = parse_observation(features[0], expected_series=series)
        approved = parse_observation(features[1], expected_series=series)
        self.assertTrue(provisional.provisional)
        self.assertEqual(provisional.approval_status, "Provisional")
        self.assertEqual(provisional.qualifier, "Estimated")
        self.assertFalse(approved.provisional)
        self.assertEqual(approved.approval_status, "Approved")

    def test_fetch_skips_null_value_after_validating_row_identity(self) -> None:
        series = parsed_series()[0]
        features = deepcopy(fixture("usgs_daily.json")["features"])
        null_row = deepcopy(features[0])
        null_row["properties"]["value"] = None
        adapter = Mock(spec=UsgsWaterDataAdapter)
        adapter._feature_pages.return_value = [
            (features[0], "https://api.waterdata.usgs.gov/ogcapi/v0/test"),
            (null_row, "https://api.waterdata.usgs.gov/ogcapi/v0/test"),
            (features[1], "https://api.waterdata.usgs.gov/ogcapi/v0/test"),
        ]

        observations = UsgsWaterDataAdapter.fetch_observations(
            adapter,
            series,
            start=date(2026, 8, 1),
            end=date(2026, 8, 2),
        )

        self.assertEqual(len(observations), 2)
        self.assertEqual([row.value for row in observations], [4192.31, 4192.28])

        foreign_null = deepcopy(null_row)
        foreign_null["properties"]["monitoring_location_id"] = "USGS-99999999"
        adapter._feature_pages.return_value = [
            (foreign_null, "https://api.waterdata.usgs.gov/ogcapi/v0/test")
        ]
        with self.assertRaises(usgs_waterdata.UsgsIdentityError):
            UsgsWaterDataAdapter.fetch_observations(
                adapter,
                series,
                start=date(2026, 8, 1),
                end=date(2026, 8, 2),
            )

    def test_normalizes_records_without_changing_source_datum(self) -> None:
        location = parse_monitoring_location(
            fixture("usgs_monitoring_location.json"),
            expected_location_id="USGS-10010000",
            source_url_public="https://api.waterdata.usgs.gov/ogcapi/v0/location",
        )
        series = parsed_series()[0]
        observations = tuple(
            parse_observation(item, expected_series=series)
            for item in fixture("usgs_daily.json")["features"]
        )
        records = UsgsWaterDataAdapter.normalize_observations(
            observations,
            series=series,
            location=location,
            profile_id="great_salt",
            hylak_id=67,
            display_name="Great Salt",
            source_id="usgs_great_salt_south_arm_level",
            retrieved_at_utc=datetime(2026, 8, 13, tzinfo=UTC),
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].vertical_datum.source_vertical_datum, "NGVD29")
        self.assertIs(
            records[0].vertical_datum.normalization_status,
            NormalizationStatus.SOURCE_VALUE_RETAINED,
        )
        self.assertEqual(records[0].raw_parameter_code, "62614")
        self.assertTrue(records[0].provisional)

    def test_bounded_interval_validation(self) -> None:
        # This check happens before any HTTP call.
        adapter = object.__new__(UsgsWaterDataAdapter)
        with self.assertRaises(ValueError):
            adapter.fetch_observations(
                parsed_series()[0],
                start=date(2026, 8, 2),
                end=date(2026, 8, 1),
            )

    def test_adapter_contains_only_modern_waterdata_base_url(self) -> None:
        source = inspect.getsource(usgs_waterdata).casefold()
        self.assertIn("api.waterdata.usgs.gov/ogcapi/v0", source)
        legacy_host_fragment = "water" + "services"
        self.assertNotIn(legacy_host_fragment, source)


if __name__ == "__main__":
    unittest.main()
