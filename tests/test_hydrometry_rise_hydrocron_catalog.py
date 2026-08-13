"""Credential-free RISE, Hydrocron, and bathymetry metadata tests."""

from __future__ import annotations

import copy
import json
import unittest
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from regional_analysis.hydrometry.adapters.hydrocron import (
    HydrocronAdapter,
    HydrocronIdentityError,
    build_hydrocron_query,
    parse_hydrocron_response,
    validate_pld_id,
)
from regional_analysis.hydrometry.adapters.usbr_rise import (
    RISE_ACCEPT,
    RiseAdapter,
    RiseIdentityError,
    RiseSchemaError,
    RiseUnitError,
    parse_catalog_item,
    parse_catalog_record,
    parse_location,
    parse_results_page,
)
from regional_analysis.hydrometry.bathymetry_catalog import (
    BATHYMETRY_CATALOG,
    BathymetryCatalogError,
    entries_for_profile,
    get_bathymetry_entry,
    parse_doi_citation,
)
from regional_analysis.hydrometry.cache import CacheMode, CacheStore
from regional_analysis.hydrometry.http import HttpResponseData, public_url
from regional_analysis.hydrometry.models import HydrometryVariable
from regional_analysis.hydrometry.registry import load_registry

FIXTURES = Path(__file__).parent / "fixtures" / "hydrometry"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FixtureClient:
    """Minimal HttpClient-compatible fixture transport."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        expected_content_types: tuple[str, ...] = (),
        timeout: float | None = None,
    ) -> HttpResponseData:
        del expected_content_types, timeout
        clean_params = dict(params or {})
        clean_headers = dict(headers or {})
        self.calls.append((url, clean_params, clean_headers))
        key = f"{url}|{clean_params.get('page', '')}"
        payload = self.responses.get(key, self.responses.get(url))
        if payload is None:
            raise AssertionError(f"Unexpected fixture request: {url} {clean_params}")
        return HttpResponseData(
            body=json.dumps(payload).encode(),
            status_code=200,
            content_type=RISE_ACCEPT,
            retrieved_at_utc=datetime(2026, 8, 13, tzinfo=UTC),
            url_public=public_url(url, clean_params),
            headers={},
        )


class NeverNetworkClient:
    def get(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("A requires-crosswalk probe must not use the network")


class RiseParsingTests(unittest.TestCase):
    def test_adapter_record_cache_supports_network_free_offline_mode(self) -> None:
        url = "https://data.usbr.gov/rise/api/catalog-record/2300"
        client = FixtureClient({url: _fixture("rise_record_2300.json")})
        with TemporaryDirectory() as directory:
            cache = CacheStore(directory)
            live = RiseAdapter(
                client=client, mode=CacheMode.LIVE, cache=cache
            ).fetch_record(2300, expected_title_contains="Flaming Gorge")
            self.assertEqual(live.record_id, 2300)
            offline = RiseAdapter(
                client=NeverNetworkClient(), mode=CacheMode.OFFLINE, cache=cache
            ).fetch_record(2300, expected_title_contains="Flaming Gorge")
            self.assertEqual(offline, live)
            self.assertEqual(len(client.calls), 1)

    def test_record_and_relationship_item_discovery(self) -> None:
        record = parse_catalog_record(
            _fixture("rise_record_2300.json"),
            expected_record_id=2300,
            expected_title_contains="Flaming Gorge",
        )
        self.assertEqual(record.location_id, 1535)
        self.assertEqual(record.item_ids, (341, 4275, 4782))

        location = parse_location(
            _fixture("rise_location_1535.json"),
            expected_location_id=1535,
            expected_name_contains="Flaming Gorge",
            expected_record_id=2300,
        )
        self.assertEqual(location.name, "Flaming Gorge Reservoir Dam and Powerplant")
        self.assertEqual(location.states, ("UT", "WY"))
        self.assertEqual(location.vertical_datum, "EGM96")

        base = "https://data.usbr.gov/rise/api"
        client = FixtureClient(
            {
                f"{base}/catalog-item/341": _fixture("rise_item_341.json"),
                f"{base}/catalog-item/4275": _fixture("rise_item_4275.json"),
                f"{base}/catalog-item/4782": _fixture("rise_item_4782.json"),
            }
        )
        inventory = RiseAdapter(
            client=client, mode=CacheMode.LIVE
        ).inventory_record_items(record, expected_location_name="Flaming Gorge")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(
            inventory.candidates_by_variable[HydrometryVariable.RESERVOIR_ELEVATION],
            (341,),
        )
        self.assertEqual(
            inventory.candidates_by_variable[HydrometryVariable.BANK_STORAGE],
            (4275,),
        )
        self.assertEqual(
            inventory.candidates_by_variable[HydrometryVariable.SURFACE_AREA],
            (4782,),
        )
        self.assertNotIn(HydrometryVariable.STORAGE, inventory.candidates_by_variable)
        self.assertTrue(inventory.requires_review)

    def test_item_identity_unit_alias_and_provisional(self) -> None:
        bank = parse_catalog_item(
            _fixture("rise_item_4275.json"),
            expected_item_id=4275,
            expected_record_id=2300,
            expected_location_name="Flaming Gorge",
            expected_parameter_name="Lake/Reservoir Bank Storage",
            expected_unit="acre-feet",
        )
        self.assertIs(bank.variable, HydrometryVariable.BANK_STORAGE)
        self.assertEqual(bank.provider_variable, "bank_storage")
        self.assertEqual(bank.unit, "af")
        self.assertTrue(bank.provisional)
        self.assertEqual(bank.update_frequency, "daily")

        total = parse_catalog_item(
            _fixture("rise_item_337.json"),
            expected_item_id=337,
            expected_record_id=2300,
            expected_parameter_name="Lake/Reservoir Storage",
            expected_unit="acre-feet",
        )
        self.assertIs(total.variable, HydrometryVariable.STORAGE)
        self.assertEqual(total.provider_variable, "total_reservoir_storage")
        self.assertNotEqual(total.variable, bank.variable)

    def test_unexpected_unit_fails(self) -> None:
        with self.assertRaises(RiseUnitError):
            parse_catalog_item(
                _fixture("rise_item_341.json"),
                expected_item_id=341,
                expected_record_id=2300,
                expected_unit="m",
            )

    def test_result_identity_is_validated(self) -> None:
        item = parse_catalog_item(
            _fixture("rise_item_341.json"),
            expected_item_id=341,
            expected_record_id=2300,
        )
        wrong = copy.deepcopy(_fixture("rise_results_341_page1.json"))
        wrong["data"][0]["attributes"]["itemId"] = 999
        with self.assertRaises(RiseIdentityError):
            parse_results_page(wrong, expected_item=item)

        documented_alias = copy.deepcopy(_fixture("rise_results_341_page1.json"))
        attributes = documented_alias["data"][0]["attributes"]
        attributes["catalogItemId"] = attributes.pop("itemId")
        observations, _total, _size, _page = parse_results_page(
            documented_alias, expected_item=item
        )
        self.assertEqual(observations[0].item_id, 341)

    def test_result_pagination_is_bounded_and_uses_item_id(self) -> None:
        item = parse_catalog_item(
            _fixture("rise_item_341.json"),
            expected_item_id=341,
            expected_record_id=2300,
        )
        url = "https://data.usbr.gov/rise/api/result"
        client = FixtureClient(
            {
                f"{url}|1": _fixture("rise_results_341_page1.json"),
                f"{url}|2": _fixture("rise_results_341_page2.json"),
            }
        )
        observations = RiseAdapter(client=client, mode=CacheMode.LIVE).fetch_timeseries(
            item,
            start=date(2026, 8, 1),
            end=date(2026, 8, 2),
            page_size=1,
            max_pages=2,
        )
        self.assertEqual(len(observations), 2)
        self.assertTrue(all(call[1]["itemId"] == 341 for call in client.calls))
        self.assertTrue(all("catalogItemId" not in call[1] for call in client.calls))

    def test_result_outside_requested_interval_fails(self) -> None:
        item = parse_catalog_item(
            _fixture("rise_item_341.json"),
            expected_item_id=341,
            expected_record_id=2300,
        )
        payload = copy.deepcopy(_fixture("rise_results_341_page1.json"))
        payload["data"][0]["attributes"]["dateTime"] = "2026-07-31T07:00:00Z"
        url = "https://data.usbr.gov/rise/api/result"
        client = FixtureClient({f"{url}|1": payload})
        with self.assertRaisesRegex(RiseSchemaError, "outside"):
            RiseAdapter(client=client, mode=CacheMode.LIVE).fetch_timeseries(
                item,
                start=date(2026, 8, 1),
                end=date(2026, 8, 2),
                page_size=1,
                max_pages=2,
            )


class HydrocronTests(unittest.TestCase):
    def test_missing_pld_is_structured_and_network_free(self) -> None:
        result = HydrocronAdapter(
            client=NeverNetworkClient(), mode=CacheMode.LIVE
        ).probe(
            swot_pld_id=None,
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=datetime(2025, 2, 1, tzinfo=UTC),
        )
        self.assertEqual(result.status, "requires_crosswalk")
        self.assertEqual(result.observations, ())
        self.assertIn("separate", result.message)

    def test_pld_validation_and_query_provenance_fields(self) -> None:
        with self.assertRaises(ValueError):
            validate_pld_id("67")
        query = build_hydrocron_query(
            "2710046612",
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=datetime(2025, 2, 1, tzinfo=UTC),
        )
        self.assertEqual(query["feature"], "PriorLake")
        self.assertEqual(query["feature_id"], "2710046612")
        self.assertIn("collection_shortname", query["fields"])
        self.assertIn("PLD_version", query["fields"])

    def test_hydrocron_response_normalization_and_identity(self) -> None:
        payload = _fixture("hydrocron_prior_lake.json")
        observations = parse_hydrocron_response(payload, expected_pld_id="2710046612")
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].wse_unit, "m")
        self.assertEqual(observations[0].area_unit, "km2")
        self.assertEqual(observations[0].pld_version, "PriorLakeDb_v10")

        records = HydrocronAdapter.normalize_observations(
            observations,
            profile_id="explicit_pld_fixture",
            hylak_id=67,
            display_name="Fixture Lake",
            source_id="hydrocron:explicit_pld_fixture",
            retrieved_at_utc=datetime(2026, 8, 13, tzinfo=UTC),
            source_url_public="https://example.test/hydrocron",
        )
        self.assertEqual(
            {record.variable for record in records},
            {HydrometryVariable.WATER_LEVEL, HydrometryVariable.SURFACE_AREA},
        )
        self.assertTrue(all(record.source_version == "D" for record in records))

        wrong = copy.deepcopy(payload)
        wrong["results"]["geojson"]["features"][0]["properties"]["lake_id"] = (
            "9999999999"
        )
        with self.assertRaises(HydrocronIdentityError):
            parse_hydrocron_response(wrong, expected_pld_id="2710046612")


class BathymetryCatalogTests(unittest.TestCase):
    def test_catalog_is_metadata_only_and_covers_official_records(self) -> None:
        self.assertEqual(len(BATHYMETRY_CATALOG), 5)
        self.assertEqual(len(entries_for_profile("lake_powell")), 2)
        great_salt = get_bathymetry_entry("usgs_great_salt_topobathymetry", strict=True)
        self.assertEqual(great_salt.vertical_datum, "NAVD88 (Geoid12B)")
        self.assertTrue(great_salt.raster_available)
        self.assertTrue(great_salt.relation_table_available)
        flaming = get_bathymetry_entry(
            "rise_flaming_gorge_sedimentation_survey", strict=True
        )
        self.assertIsNone(flaming.vertical_datum)
        self.assertTrue(flaming.relation_table_available)

    def test_static_catalog_ids_match_pilot_registry(self) -> None:
        registry = load_registry("configs/hydrometry_pilots.yaml")
        configured = {
            source.catalog_id
            for profile in registry.profiles
            for source in profile.bathymetry_sources
        }
        static = {entry.catalog_id for entry in BATHYMETRY_CATALOG}
        self.assertEqual(static, configured)

    def test_doi_citation_identity_and_title(self) -> None:
        identifier, title = parse_doi_citation(
            {
                "DOI": "10.5066/P9O3IPG3",
                "title": "Elevation-area-capacity tables for Lake Powell, 2018",
            },
            expected_doi="10.5066/P9O3IPG3",
            expected_title="Elevation-area-capacity tables for Lake Powell, 2018",
        )
        self.assertEqual(identifier, "10.5066/P9O3IPG3")
        self.assertIn("Lake Powell", title)
        with self.assertRaises(BathymetryCatalogError):
            parse_doi_citation(
                {"DOI": "10.0/wrong", "title": title},
                expected_doi="10.5066/P9O3IPG3",
                expected_title=title,
            )


if __name__ == "__main__":
    unittest.main()
