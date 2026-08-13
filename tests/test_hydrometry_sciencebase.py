"""Credential-free ScienceBase reference-source tests."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

from regional_analysis.hydrometry.cache import CacheMode
from regional_analysis.hydrometry.http import (
    HttpClient,
    HttpResponseData,
    public_url,
)
from regional_analysis.hydrometry.sciencebase import (
    BEAR_LAKE_TITLE,
    SCIENCEBASE_ITEMS_URL,
    ScienceBaseContentError,
    ScienceBaseFileSelectionError,
    ScienceBaseIdentityError,
    acquire_bear_lake_eav,
    parse_bear_lake_eav_csv,
    parse_doi_metadata,
    parse_sciencebase_item,
    parse_sciencebase_search,
    safe_extract_expected_zip,
    select_eav_file,
)

FIXTURES = Path(__file__).parent / "fixtures" / "hydrometry_phase1"
BEAR_LAKE_DOI = "10.5066/P14KZS2D"
ITEM_URL = "https://www.sciencebase.gov/catalog/item/698226f4b66b018d00102e42"
FILE_URL = (
    "https://www.sciencebase.gov/catalog/file/get/"
    "698226f4b66b018d00102e42?f=fixture-eav"
)


def _json_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _response(
    body: bytes,
    *,
    url: str,
    params: dict[str, object] | None = None,
    content_type: str,
) -> HttpResponseData:
    return HttpResponseData(
        body=body,
        status_code=200,
        content_type=content_type,
        retrieved_at_utc=datetime(2026, 8, 13, tzinfo=UTC),
        url_public=public_url(url, params),
        headers={"Content-Type": content_type, "Content-Length": str(len(body))},
    )


class ScienceBaseParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.search = _json_fixture("bear_lake_sciencebase_search.json")
        self.item_payload = _json_fixture("bear_lake_sciencebase_item.json")
        self.csv_bytes = (FIXTURES / "bear_lake_eav.csv").read_bytes()

    def test_doi_and_structured_metadata_parsing(self) -> None:
        doi = parse_doi_metadata(
            _json_fixture("bear_lake_doi.json"),
            expected_doi=BEAR_LAKE_DOI,
            expected_title=BEAR_LAKE_TITLE,
        )
        self.assertEqual(doi.doi, BEAR_LAKE_DOI)
        self.assertEqual(doi.published, "2026-3-31")

        item_id, metadata_url = parse_sciencebase_search(
            self.search, expected_title=BEAR_LAKE_TITLE
        )
        self.assertEqual(item_id, "698226f4b66b018d00102e42")
        item = parse_sciencebase_item(
            self.item_payload,
            expected_doi=BEAR_LAKE_DOI,
            expected_title=BEAR_LAKE_TITLE,
            metadata_url=metadata_url,
        )
        self.assertEqual(item.publication_date, "2026-03-31")
        self.assertEqual(item.source_version, "2026-03-31")
        self.assertEqual(len(item.files), 3)

    def test_wrong_title_or_nonofficial_item_url_fails(self) -> None:
        wrong = dict(self.search)
        wrong["items"] = [dict(self.search["items"][0], title="Other Lake")]
        with self.assertRaises(ScienceBaseIdentityError):
            parse_sciencebase_search(wrong, expected_title=BEAR_LAKE_TITLE)

        unsafe = dict(self.search)
        unsafe_item = dict(self.search["items"][0])
        unsafe_item["link"] = {"url": "https://example.test/item/abc"}
        unsafe["items"] = [unsafe_item]
        with self.assertRaises(ScienceBaseIdentityError):
            parse_sciencebase_search(unsafe, expected_title=BEAR_LAKE_TITLE)

        incomplete = dict(self.search, total=2)
        with self.assertRaises(ScienceBaseIdentityError):
            parse_sciencebase_search(incomplete, expected_title=BEAR_LAKE_TITLE)

    def test_expected_file_selection_is_identity_based_and_ambiguous_fails(
        self,
    ) -> None:
        item = parse_sciencebase_item(
            self.item_payload,
            expected_doi=BEAR_LAKE_DOI,
            expected_title=BEAR_LAKE_TITLE,
        )
        selected = select_eav_file(item)
        self.assertEqual(selected.name, "Bear_Lake_TBDEM_2026_ElevAreaVolume.csv")
        self.assertEqual(selected.content_type, "text/csv")
        self.assertLess(selected.size_bytes, 1_000_000)

        duplicate = replace(
            selected,
            name="Bear_Lake_TBDEM_2027_ElevAreaVolume.csv",
        )
        with self.assertRaises(ScienceBaseFileSelectionError):
            select_eav_file(replace(item, files=item.files + (duplicate,)))

        with self.assertRaises(ScienceBaseFileSelectionError):
            select_eav_file(replace(item, summary="Undocumented generic table."))

    def test_csv_schema_hash_blank_rows_and_normalization(self) -> None:
        parsed = parse_bear_lake_eav_csv(
            self.csv_bytes,
            source_file="Bear_Lake_TBDEM_2026_ElevAreaVolume.csv",
            source_title=BEAR_LAKE_TITLE,
            source_version="2026-03-31",
            source_doi=BEAR_LAKE_DOI,
            source_sha256=(
                "283506b418943f4fe578b389e00326eb0f74dc453f4a2cf4d9903a6945795f60"
            ),
        )
        self.assertEqual(parsed.valid_row_count, 3)
        self.assertEqual(parsed.fully_blank_row_count, 2)
        self.assertEqual(parsed.physical_record_count, 5)
        self.assertEqual(parsed.table.validation.validation_status, "passed")
        self.assertEqual(parsed.table.elevation_min_m, 1747.39)
        self.assertEqual(parsed.table.elevation_max_m, 1747.41)

        partial = self.csv_bytes.replace(
            b"1747.40,5732.94,4305.56,400.00",
            b"1747.40,5732.94,,400.00",
        )
        with self.assertRaises(ScienceBaseContentError):
            parse_bear_lake_eav_csv(
                partial,
                source_file="Bear_Lake_TBDEM_2026_ElevAreaVolume.csv",
                source_title=BEAR_LAKE_TITLE,
                source_version="2026-03-31",
                source_doi=BEAR_LAKE_DOI,
            )


class ScienceBaseZipTests(unittest.TestCase):
    @staticmethod
    def _zip(members: dict[str, bytes]) -> bytes:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
            for name, body in members.items():
                bundle.writestr(name, body)
        return output.getvalue()

    def test_zip_path_traversal_is_blocked(self) -> None:
        archive = self._zip({"relation.csv": b"ok", "../outside.txt": b"bad"})
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ScienceBaseContentError):
                safe_extract_expected_zip(
                    archive,
                    output_dir=temporary,
                    expected_filenames=("relation.csv",),
                )
            self.assertFalse((Path(temporary).parent / "outside.txt").exists())

    def test_only_explicit_expected_members_are_extracted(self) -> None:
        archive = self._zip(
            {"nested/relation.csv": b"expected", "large-raster.tif": b"not extracted"}
        )
        with tempfile.TemporaryDirectory() as temporary:
            outputs = safe_extract_expected_zip(
                archive,
                output_dir=temporary,
                expected_filenames=("relation.csv",),
            )
            self.assertEqual(outputs, (Path(temporary) / "relation.csv",))
            self.assertEqual(outputs[0].read_bytes(), b"expected")
            self.assertFalse((Path(temporary) / "large-raster.tif").exists())


class ScienceBaseAcquisitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.search_bytes = json.dumps(
            _json_fixture("bear_lake_sciencebase_search.json")
        ).encode()
        self.item_bytes = json.dumps(
            _json_fixture("bear_lake_sciencebase_item.json")
        ).encode()
        self.csv_bytes = (FIXTURES / "bear_lake_eav.csv").read_bytes()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def _client(self, *, csv_content_type: str = "text/csv") -> Mock:
        requested: list[str] = []

        def get(
            url: str,
            *,
            params: dict[str, object] | None = None,
            **_kwargs: object,
        ) -> HttpResponseData:
            requested.append(public_url(url, params))
            if url == SCIENCEBASE_ITEMS_URL:
                return _response(
                    self.search_bytes,
                    url=url,
                    params=params,
                    content_type="application/json",
                )
            if url == ITEM_URL:
                return _response(
                    self.item_bytes,
                    url=url,
                    params=params,
                    content_type="application/json",
                )
            if url == FILE_URL:
                return _response(
                    self.csv_bytes,
                    url=url,
                    params=params,
                    content_type=csv_content_type,
                )
            raise AssertionError(f"Unexpected request {url}")

        client = Mock(spec=HttpClient)
        client.get.side_effect = get
        client.requested_urls = requested
        return client

    def test_live_acquisition_then_zero_network_offline_replay(self) -> None:
        live_client = self._client()
        source = acquire_bear_lake_eav(
            client=live_client,
            mode=CacheMode.LIVE,
            reference_dir=self.temporary.name,
            doi=BEAR_LAKE_DOI,
        )
        self.assertFalse(source.cache_hit)
        self.assertEqual(live_client.get.call_count, 3)
        self.assertFalse(
            any("fixture-raster" in url for url in live_client.requested_urls)
        )
        self.assertEqual(source.manifest.source_row_count, 3)
        self.assertEqual(source.manifest.ignored_fully_blank_row_count, 2)
        self.assertEqual(
            source.manifest.sha256,
            "283506b418943f4fe578b389e00326eb0f74dc453f4a2cf4d9903a6945795f60",
        )
        self.assertTrue(source.source_path.is_file())
        self.assertTrue(source.metadata_path.is_file())
        self.assertTrue(source.manifest_path.is_file())

        http_cache = Path(self.temporary.name) / "bear_lake" / ".http_cache"
        http_cache.rename(http_cache.with_name("disabled_http_cache"))
        offline_client = Mock(spec=HttpClient)
        replay = acquire_bear_lake_eav(
            client=offline_client,
            mode=CacheMode.OFFLINE,
            reference_dir=self.temporary.name,
            doi=BEAR_LAKE_DOI,
        )
        self.assertTrue(replay.cache_hit)
        offline_client.get.assert_not_called()
        self.assertEqual(
            replay.table.normalized_records(), source.table.normalized_records()
        )
        self.assertEqual(replay.manifest.sha256, source.manifest.sha256)

    def test_offline_replay_rejects_tampered_content_addressed_source(self) -> None:
        source = acquire_bear_lake_eav(
            client=self._client(),
            mode=CacheMode.LIVE,
            reference_dir=self.temporary.name,
            doi=BEAR_LAKE_DOI,
        )
        source.source_path.write_bytes(b"tampered")
        offline_client = Mock(spec=HttpClient)
        with self.assertRaises(ScienceBaseContentError):
            acquire_bear_lake_eav(
                client=offline_client,
                mode=CacheMode.OFFLINE,
                reference_dir=self.temporary.name,
                doi=BEAR_LAKE_DOI,
            )
        offline_client.get.assert_not_called()

    def test_wrong_download_content_type_fails(self) -> None:
        with self.assertRaises(ScienceBaseContentError):
            acquire_bear_lake_eav(
                client=self._client(csv_content_type="application/json"),
                mode=CacheMode.LIVE,
                reference_dir=self.temporary.name,
                doi=BEAR_LAKE_DOI,
            )

    def test_failed_refresh_preserves_previous_reference(self) -> None:
        source = acquire_bear_lake_eav(
            client=self._client(),
            mode=CacheMode.LIVE,
            reference_dir=self.temporary.name,
            doi=BEAR_LAKE_DOI,
        )
        source_before = source.source_path.read_bytes()
        manifest_before = source.manifest_path.read_bytes()
        failed = Mock(spec=HttpClient)
        failed.get.side_effect = RuntimeError("network failed")
        with self.assertRaises(RuntimeError):
            acquire_bear_lake_eav(
                client=failed,
                mode=CacheMode.LIVE,
                reference_dir=self.temporary.name,
                doi=BEAR_LAKE_DOI,
                refresh=True,
            )
        self.assertEqual(source.source_path.read_bytes(), source_before)
        self.assertEqual(source.manifest_path.read_bytes(), manifest_before)


if __name__ == "__main__":
    unittest.main()
