"""Credential-free orchestration tests for the Phase 0 inventory CLI."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

from regional_analysis.hydrometry.adapters.hydrocron import HydrocronProbeResult
from regional_analysis.hydrometry.adapters.usbr_rise import (
    RiseCatalogItem,
    RiseCatalogRecord,
    RiseItemInventory,
    RiseLocation,
    RiseObservation,
)
from regional_analysis.hydrometry.adapters.usgs_waterdata import (
    MonitoringLocation,
    UsgsDataKind,
    UsgsObservation,
    UsgsParameterMetadata,
    UsgsSeriesMetadata,
    UsgsWaterDataAdapter,
)
from regional_analysis.hydrometry.bathymetry_catalog import BathymetryProbeResult
from regional_analysis.hydrometry.cache import CacheMode, CacheStore
from regional_analysis.hydrometry.capability_inventory import (
    InventoryAdapters,
    build_parser,
    render_capability_report,
    run_inventory,
    write_inventory_outputs,
)
from regional_analysis.hydrometry.http import HttpResponseData, public_url
from regional_analysis.hydrometry.models import HeightType, HydrometryVariable
from regional_analysis.hydrometry.registry import load_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "hydrometry_pilots.yaml"
NOW = datetime(2026, 8, 13, 12, tzinfo=UTC)


class _NoNetworkClient:
    request_count = 0


class _FixtureUsgs:
    names: ClassVar[dict[str, str]] = {
        "USGS-10010000": "Great Salt Lake at Saltair Boat Harbor, UT",
        "USGS-10010100": "Great Salt Lake near Saline, UT",
        "USGS-10387150": "Lake Abert near Valley Falls, OR",
        "USGS-10055000": "Bear Lake at Lifton Pumping Station, ID",
    }
    datums: ClassVar[dict[str, str]] = {
        "USGS-10010000": "NGVD29",
        "USGS-10010100": "NGVD29",
        "USGS-10387150": "NAVD88",
        "USGS-10055000": "NAVD88",
    }

    def fetch_monitoring_location(self, location_id: str) -> MonitoringLocation:
        return MonitoringLocation(
            location_id=location_id,
            monitoring_location_number=location_id.removeprefix("USGS-"),
            name=self.names[location_id],
            site_type="Lake, Reservoir, Impoundment",
            state_name="fixture",
            longitude=-112.0,
            latitude=41.0,
            altitude=4200.0,
            altitude_unit="ft",
            vertical_datum=self.datums[location_id],
            vertical_datum_name=self.datums[location_id],
            source_url_public=f"https://api.waterdata.usgs.gov/fixture/{location_id}",
        )

    def discover_series(self, location_id: str):
        datum = self.datums[location_id]
        parameter = "62614" if datum == "NGVD29" else "62615"
        return (
            UsgsSeriesMetadata(
                time_series_id=f"{location_id}-{parameter}-daily",
                monitoring_location_id=location_id,
                parameter_code=parameter,
                parameter_name="Lake or reservoir water surface elevation",
                parameter_description=f"Water surface elevation, feet above {datum}",
                unit="ft",
                statistic_code="00003",
                computation_identifier="Mean",
                computation_period_identifier="Daily",
                begin_utc=datetime(2020, 1, 1, tzinfo=UTC),
                end_utc=NOW,
                primary=True,
                web_description=f"Water surface elevation above {datum}",
                variable=HydrometryVariable.WATER_LEVEL,
                height_type=HeightType.ORTHOMETRIC_HEIGHT,
                vertical_datum=datum,
                data_kind=UsgsDataKind.DAILY,
                source_url_public=f"https://api.waterdata.usgs.gov/fixture/{location_id}/series",
            ),
        )

    def fetch_parameter_metadata(self, parameter_code: str):
        return UsgsParameterMetadata(
            parameter_code=parameter_code,
            parameter_name="Water surface elevation",
            parameter_description="Fixture absolute water elevation",
            unit="ft",
            medium="surface water",
            source_url_public="https://api.waterdata.usgs.gov/fixture/parameter",
        )

    def fetch_observations(self, series, *, start, end):
        del start, end
        return (
            UsgsObservation(
                time_series_id=series.time_series_id,
                monitoring_location_id=series.monitoring_location_id,
                timestamp_utc=NOW,
                value=4200.0,
                unit="ft",
                parameter_code=series.parameter_code,
                statistic_code=series.statistic_code,
                approval_status="Provisional",
                provisional=True,
                qualifier="P",
                last_modified_utc=NOW,
            ),
        )

    vertical_reference = staticmethod(UsgsWaterDataAdapter.vertical_reference)
    normalize_observations = staticmethod(UsgsWaterDataAdapter.normalize_observations)


def _rise_item(
    item_id: int,
    record_id: int,
    lake: str,
    variable: HydrometryVariable,
    unit: str,
) -> RiseCatalogItem:
    parameter = {
        HydrometryVariable.RESERVOIR_ELEVATION: "Lake/Reservoir Elevation",
        HydrometryVariable.SURFACE_AREA: "Lake/Reservoir Area",
        HydrometryVariable.STORAGE: "Lake/Reservoir Storage",
        HydrometryVariable.BANK_STORAGE: "Lake/Reservoir Bank Storage",
    }[variable]
    return RiseCatalogItem(
        item_id=item_id,
        record_id=record_id,
        title=f"{lake} - {parameter}",
        description="fixture",
        variable=variable,
        provider_variable=variable.value,
        parameter_name=parameter,
        parameter_group="Reservoir",
        parameter_id=item_id + 1000,
        unit=unit,
        timestep="Daily",
        transformation="Instantaneous",
        temporal_start=datetime(2020, 1, 1, tzinfo=UTC),
        temporal_end=NOW,
        update_frequency="Daily",
        updated_at=NOW,
        vertical_datum=None,
        provisional=True,
        disclaimer="These data are provisional and subject to revision.",
        data_structure="Time series",
        binary_file_url=None,
        validation_issues=(),
        source_url_public=f"https://data.usbr.gov/rise/api/catalog-item/{item_id}",
    )


class _FixtureRise:
    items: ClassVar[dict[int, tuple[RiseCatalogItem, ...]]] = {
        2362: (
            _rise_item(
                508, 2362, "Lake Powell", HydrometryVariable.RESERVOIR_ELEVATION, "ft"
            ),
            _rise_item(509, 2362, "Lake Powell", HydrometryVariable.STORAGE, "af"),
            _rise_item(
                4784, 2362, "Lake Powell", HydrometryVariable.SURFACE_AREA, "acres"
            ),
        ),
        2300: (
            _rise_item(
                341, 2300, "Flaming Gorge", HydrometryVariable.RESERVOIR_ELEVATION, "ft"
            ),
            _rise_item(337, 2300, "Flaming Gorge", HydrometryVariable.STORAGE, "af"),
            _rise_item(
                4275, 2300, "Flaming Gorge", HydrometryVariable.BANK_STORAGE, "af"
            ),
            _rise_item(
                4782, 2300, "Flaming Gorge", HydrometryVariable.SURFACE_AREA, "acres"
            ),
        ),
    }

    def fetch_record(self, record_id: int, *, expected_title_contains=None):
        lake = "Lake Powell" if record_id == 2362 else "Flaming Gorge"
        self.assert_contains(lake, expected_title_contains)
        return RiseCatalogRecord(
            record_id=record_id,
            title=f"{lake} Reservoir and River Operations",
            description="fixture",
            generation_effort=None,
            location_id=record_id + 100,
            item_ids=tuple(item.item_id for item in self.items[record_id]),
            updated_at=NOW,
            source_url_public=f"https://data.usbr.gov/rise/api/catalog-record/{record_id}",
        )

    @staticmethod
    def assert_contains(actual: str, expected: str | None) -> None:
        if expected and expected.casefold() not in actual.casefold():
            raise AssertionError((actual, expected))

    def inventory_record_items(self, record, *, expected_location_name=None):
        self.assert_contains(record.title, expected_location_name)
        items = self.items[record.record_id]
        by_variable = {
            variable: tuple(item.item_id for item in items if item.variable is variable)
            for variable in HydrometryVariable
            if any(item.variable is variable for item in items)
        }
        return RiseItemInventory(
            record=record,
            items=items,
            candidates_by_variable=MappingProxyType(by_variable),
            requires_review=False,
        )

    def fetch_location(
        self,
        location_id,
        *,
        expected_name_contains=None,
        expected_record_id=None,
    ):
        record_id = location_id - 100
        lake = "Lake Powell" if record_id == 2362 else "Flaming Gorge"
        self.assert_contains(lake, expected_name_contains)
        if expected_record_id != record_id:
            raise AssertionError((expected_record_id, record_id))
        return RiseLocation(
            location_id=location_id,
            name=lake,
            location_type="Reservoir",
            longitude=-110.0,
            latitude=40.0,
            horizontal_datum="WGS84",
            vertical_datum=None,
            states=("UT",),
            record_ids=(record_id,),
            source_url_public=f"https://data.usbr.gov/rise/api/location/{location_id}",
        )

    def fetch_timeseries(self, item, *, start, end):
        del start, end
        return (
            RiseObservation(
                result_id=item.item_id * 10,
                item_id=item.item_id,
                location_id=item.record_id + 100,
                parameter_id=item.parameter_id,
                timestamp_utc=NOW,
                value=100.0,
                status="provisional",
                source_code="fixture",
                created_at=NOW,
                updated_at=NOW,
            ),
        )

    from regional_analysis.hydrometry.adapters.usbr_rise import RiseAdapter

    normalize_observations = staticmethod(RiseAdapter.normalize_observations)


class _FixtureHydrocron:
    def probe(self, *, swot_pld_id, start, end):
        del start, end
        self._last_pld_id = swot_pld_id
        return HydrocronProbeResult(
            status="requires_crosswalk",
            swot_pld_id=None,
            collection_name=None,
            product_versions=(),
            observations=(),
            message="HydroLAKES ID and SWOT PLD ID are separate.",
            source_url_public=None,
        )


class _FixtureBathymetry:
    def verify(self, entry):
        return BathymetryProbeResult(
            catalog_id=entry.catalog_id,
            status=(
                "confirmed_metadata_only"
                if entry.provider == "usbr_rise"
                else "confirmed"
            ),
            returned_title=entry.title,
            returned_identifier=entry.doi or entry.source_record_id,
            retrieved_at_utc=NOW.isoformat(),
            source_url_public=entry.landing_url,
            warning=None,
        )


def _fixture_adapters(usgs=None) -> InventoryAdapters:
    return InventoryAdapters(
        usgs=usgs or _FixtureUsgs(),
        rise=_FixtureRise(),
        hydrocron=_FixtureHydrocron(),
        bathymetry=_FixtureBathymetry(),
        client=_NoNetworkClient(),
    )


class CapabilityInventoryTests(unittest.TestCase):
    def test_cli_requires_explicit_live_or_offline(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        self.assertTrue(parser.parse_args(["--offline"]).offline)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--offline", "--live"])

    def test_offline_fixture_orchestration_writes_all_outputs(self):
        registry = load_registry(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = run_inventory(
                registry,
                mode=CacheMode.OFFLINE,
                start=date(2026, 5, 16),
                end=date(2026, 8, 13),
                cache_dir=root / "cache",
                adapters=_fixture_adapters(),
                now=NOW,
            )
            levels = {
                record.profile_id: int(record.highest_current_capability)
                for record in bundle.capability_records
            }
            self.assertEqual(
                levels,
                {
                    "great_salt": 2,
                    "lake_abert": 2,
                    "bear_lake": 2,
                    "lake_powell": 4,
                    "flaming_gorge": 4,
                },
            )
            powell = next(
                record
                for record in bundle.capability_records
                if record.profile_id == "lake_powell"
            )
            self.assertTrue(powell.absolute_storage_available)
            self.assertEqual(powell.source_quality_class.value, "A")
            paths = write_inventory_outputs(
                bundle,
                output_dir=root / "data",
                report_dir=root / "report",
            )
            self.assertEqual(len(paths), 8)
            self.assertTrue(
                all(path.is_file() and path.stat().st_size for path in paths.values())
            )
            provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
            self.assertEqual(provenance["request_count"], 0)
            self.assertFalse(provenance["volume_calculated"])
            self.assertNotIn(
                "api_key", paths["provenance"].read_text(encoding="utf-8").casefold()
            )
            with paths["capability"].open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 5)
            with paths["bathymetry"].open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertIn("Hylak_id", reader.fieldnames or ())
                self.assertNotIn("hylak_id", reader.fieldnames or ())
                self.assertEqual(len(list(reader)), 6)

    def test_one_source_failure_is_reported_without_stopping_inventory(self):
        class OneFailureUsgs(_FixtureUsgs):
            def fetch_monitoring_location(self, location_id):
                if location_id == "USGS-10010100":
                    raise RuntimeError("fixture source unavailable")
                return super().fetch_monitoring_location(location_id)

        bundle = run_inventory(
            load_registry(CONFIG),
            mode=CacheMode.OFFLINE,
            start=date(2026, 8, 1),
            end=date(2026, 8, 13),
            adapters=_fixture_adapters(OneFailureUsgs()),
            now=NOW,
        )
        failures = [
            row
            for row in bundle.probe_rows
            if row["source_id"] == "usgs_great_salt_north_arm_level"
            and row["status"] == "failed"
        ]
        self.assertEqual(len(failures), 1)
        self.assertEqual(len(bundle.capability_records), 5)

    def test_parameter_unit_conflict_prevents_source_confirmation(self):
        class ConflictingParameterUsgs(_FixtureUsgs):
            def fetch_parameter_metadata(self, parameter_code):
                metadata = super().fetch_parameter_metadata(parameter_code)
                return UsgsParameterMetadata(
                    parameter_code=metadata.parameter_code,
                    parameter_name=metadata.parameter_name,
                    parameter_description=metadata.parameter_description,
                    unit="m",
                    medium=metadata.medium,
                    source_url_public=metadata.source_url_public,
                )

        bundle = run_inventory(
            load_registry(CONFIG),
            mode=CacheMode.OFFLINE,
            start=date(2026, 8, 1),
            end=date(2026, 8, 13),
            adapters=_fixture_adapters(ConflictingParameterUsgs()),
            now=NOW,
        )
        usgs_capabilities = {
            row.profile_id: int(row.highest_current_capability)
            for row in bundle.capability_records
            if row.profile_id in {"great_salt", "lake_abert", "bear_lake"}
        }
        self.assertEqual(
            usgs_capabilities,
            {"great_salt": 1, "lake_abert": 1, "bear_lake": 1},
        )
        self.assertTrue(
            all(
                "Observed-property unit" in row["message"]
                for row in bundle.probe_rows
                if row["provider"] == "usgs_waterdata" and row["status"] == "failed"
            )
        )

    def test_historical_series_without_recent_rows_remains_level_two(self):
        class HistoricalAbertUsgs(_FixtureUsgs):
            def fetch_observations(self, series, *, start, end):
                if series.monitoring_location_id == "USGS-10387150":
                    return ()
                return super().fetch_observations(series, start=start, end=end)

        bundle = run_inventory(
            load_registry(CONFIG),
            mode=CacheMode.OFFLINE,
            start=date(2026, 8, 1),
            end=date(2026, 8, 13),
            adapters=_fixture_adapters(HistoricalAbertUsgs()),
            now=NOW,
        )
        abert = next(
            row for row in bundle.capability_records if row.profile_id == "lake_abert"
        )
        self.assertEqual(int(abert.highest_current_capability), 2)
        report = render_capability_report(bundle)
        self.assertIn("USGS-10387150-62615-daily", report)
        self.assertIn("datum=NAVD88", report)

    def test_normalized_link_does_not_repoint_stale_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = CacheStore(Path(temporary))
            responses = []
            for suffix in ("used", "stale"):
                url = f"https://example.test/{suffix}"
                responses.append(
                    cache.write(
                        provider="fixture",
                        url=url,
                        params=None,
                        response=HttpResponseData(
                            body=b"{}",
                            status_code=200,
                            content_type="application/json",
                            retrieved_at_utc=NOW,
                            url_public=public_url(url),
                            headers={},
                        ),
                    )
                )
            used = responses[0].manifest.raw_response_path
            linked = cache.link_normalized_output(
                "data/hydrometry/pilot_hydrometry_provenance.json",
                raw_response_paths=(used,),
            )
            self.assertEqual(linked, 1)
            manifests = []
            for path in sorted(Path(temporary).glob("*/*.meta.json")):
                manifests.append(json.loads(path.read_text(encoding="utf-8")))
            by_raw = {item["raw_response_path"]: item for item in manifests}
            self.assertEqual(
                by_raw[used]["normalized_output_path"],
                "data/hydrometry/pilot_hydrometry_provenance.json",
            )
            stale = responses[1].manifest.raw_response_path
            self.assertIsNone(by_raw[stale]["normalized_output_path"])


if __name__ == "__main__":
    unittest.main()
