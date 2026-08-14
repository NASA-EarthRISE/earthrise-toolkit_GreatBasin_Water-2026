"""Credential-free tests for Phase 2 bounded binary acquisition."""

from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import traceback
import unittest
import zipfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import requests

from regional_analysis.hydrometry.binary_download import (
    BinaryCacheError,
    BinaryContentError,
    BinaryDownloadError,
    BinaryDownloadRequest,
    BinaryHttpStatusError,
    DownloadCancelledError,
    InsufficientDiskSpaceError,
    ScienceBaseBinarySelectionError,
    StreamingBinaryDownloader,
    ZipInventoryError,
    calculate_disk_requirements,
    inspect_zip_archive,
    metadata_file_records,
    require_sufficient_disk_space,
    safe_extract_selected,
    select_archive_raster,
    select_bear_lake_phase2_files,
    select_sciencebase_file,
    write_extraction_manifest,
)
from regional_analysis.hydrometry.sciencebase import (
    ScienceBaseFile,
    ScienceBaseItem,
)


class _Response:
    def __init__(
        self,
        status: int,
        body: bytes = b"",
        *,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.status_code = status
        self.headers = headers or {}
        self._chunks = chunks if chunks is not None else [body]
        self.closed = False

    def iter_content(self, *, chunk_size: int):
        del chunk_size
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("Unexpected HTTP request")
        return self.responses.pop(0)


def _headers(
    size: int,
    *,
    etag: str = '"release-1"',
    content_range: str | None = None,
) -> dict[str, str]:
    values = {
        "Content-Type": "application/zip",
        "Content-Length": str(size),
        "ETag": etag,
        "Last-Modified": "Thu, 13 Aug 2026 12:00:00 GMT",
        "Accept-Ranges": "bytes",
        "Set-Cookie": "never-serialize=this",
    }
    if content_range:
        values["Content-Range"] = content_range
    return values


def _source_file(name: str, content_type: str) -> ScienceBaseFile:
    return ScienceBaseFile(
        name=name,
        content_type=content_type,
        size_bytes=123,
        download_url=f"https://sciencebase.usgs.gov/files/{name}",
        checksum_algorithm="MD5",
        checksum_value="abc123",
    )


def _sciencebase_item() -> ScienceBaseItem:
    return ScienceBaseItem(
        item_id="698226f4b66b018d00102e42",
        title="Bear Lake official release",
        summary="Official metadata",
        doi="10.5066/P14KZS2D",
        citation=None,
        rights="CC0",
        publication_date="2026-03-31",
        item_last_updated="2026-03-31T16:09:48Z",
        metadata_url=(
            "https://www.sciencebase.gov/catalog/item/698226f4b66b018d00102e42"
        ),
        files=(
            _source_file("Bear_Lake_TBDEM_2026_50cm.zip", "application/zip"),
            _source_file(
                "Bear_Lake_TBDEM_2026_Spatial_Metadata.zip", "application/zip"
            ),
            _source_file("Bear_Lake_TBDEM_2026_thumbnail.png", "image/png"),
            _source_file("Bear_Lake_TBDEM_2026.xml", "application/fgdc+xml"),
        ),
    )


class ScienceBaseBinarySelectionTests(unittest.TestCase):
    def test_metadata_files_are_selected_by_identity_not_order(self) -> None:
        item = _sciencebase_item()
        selection = select_bear_lake_phase2_files(
            replace(item, files=tuple(reversed(item.files))),
            raster_archive_name="Bear_Lake_TBDEM_2026_50cm.zip",
            spatial_metadata_archive_name=("Bear_Lake_TBDEM_2026_Spatial_Metadata.zip"),
        )
        self.assertEqual(selection.raster_archive.name, "Bear_Lake_TBDEM_2026_50cm.zip")
        self.assertEqual(selection.fgdc_xml.content_type, "application/fgdc+xml")
        self.assertEqual(selection.thumbnail.content_type, "image/png")
        rows = metadata_file_records(
            (selection.raster_archive, selection.spatial_metadata_archive)
        )
        self.assertEqual(rows[0]["size_bytes"], 123)
        self.assertEqual(rows[0]["download_mode"], "direct")
        self.assertIsNone(rows[0]["s3_download_request_page_url"])

        request_page = "https://www.sciencebase.gov/catalog/item/requestDownload/id"
        staged = replace(
            selection.raster_archive,
            cuid="file-cuid",
            path_on_disk="__s3__",
            s3_download_request_page_url=request_page,
            storage_bucket="provider-bucket",
            storage_key="item/archive.zip",
        )
        staged_row = metadata_file_records((staged,))[0]
        self.assertEqual(staged_row["download_mode"], "catalog_file_proxy")
        self.assertEqual(staged_row["s3_download_request_page_url"], request_page)
        self.assertEqual(staged_row["sciencebase_file_cuid"], "file-cuid")
        self.assertEqual(staged_row["storage_bucket"], "provider-bucket")
        self.assertEqual(staged_row["storage_key"], "item/archive.zip")
        self.assertEqual(staged_row["path_on_disk"], "__s3__")

    def test_missing_or_ambiguous_metadata_file_fails_clearly(self) -> None:
        item = _sciencebase_item()
        duplicate = replace(item.files[-1], name="other.xml")
        with self.assertRaises(ScienceBaseBinarySelectionError):
            select_bear_lake_phase2_files(
                replace(item, files=item.files + (duplicate,)),
                raster_archive_name="Bear_Lake_TBDEM_2026_50cm.zip",
                spatial_metadata_archive_name=(
                    "Bear_Lake_TBDEM_2026_Spatial_Metadata.zip"
                ),
            )
        with self.assertRaises(ScienceBaseBinarySelectionError):
            select_sciencebase_file(item, expected_name="missing.zip")


class StreamingBinaryDownloaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.body = b"abcdef"
        self.digest = hashlib.sha256(self.body).hexdigest()

    def _request(self) -> BinaryDownloadRequest:
        return BinaryDownloadRequest(
            provider="usgs_sciencebase",
            url="https://sciencebase.usgs.gov/file/archive.zip?token=secret&f=public",
            destination=self.root / "archive.zip",
            expected_size_bytes=len(self.body),
            expected_sha256=self.digest,
            require_zip_integrity=False,
        )

    def test_streams_validates_hash_and_loads_offline_without_network(self) -> None:
        response = _Response(
            200,
            headers=_headers(len(self.body)),
            chunks=[b"ab", b"cd", b"ef"],
        )
        session = _Session([response])
        progress = []
        downloader = StreamingBinaryDownloader(
            session=session, chunk_size=2, sleep=lambda _seconds: None
        )
        result = downloader.download(self._request(), progress_callback=progress.append)
        self.assertEqual(result.path.read_bytes(), self.body)
        self.assertEqual(result.manifest.sha256, self.digest)
        self.assertEqual(result.bytes_transferred, len(self.body))
        self.assertNotIn("token", result.manifest.public_url)
        self.assertNotIn("Set-Cookie", result.manifest.response_headers)
        self.assertTrue(progress[-1].complete)
        self.assertFalse(self._request().part_path.exists())

        offline = StreamingBinaryDownloader(session=_Session([]))
        cached = offline.load_cached(self._request())
        self.assertTrue(cached.cache_hit)
        self.assertEqual(offline.request_count, 0)

    def test_chunked_response_uses_identity_validated_expected_size(self) -> None:
        headers = _headers(len(self.body))
        headers.pop("Content-Length")
        response = _Response(200, headers=headers, chunks=[b"ab", b"cd", b"ef"])
        result = StreamingBinaryDownloader(
            session=_Session([response]), chunk_size=2
        ).download(self._request())
        self.assertEqual(result.path.read_bytes(), self.body)
        self.assertEqual(result.manifest.content_length, len(self.body))
        self.assertNotIn("Content-Length", result.manifest.response_headers)

    def test_chunked_response_requires_an_expected_size(self) -> None:
        headers = _headers(len(self.body))
        headers.pop("Content-Length")
        request = replace(
            self._request(),
            expected_size_bytes=None,
            expected_sha256=None,
        )
        with self.assertRaisesRegex(BinaryContentError, "expected size"):
            StreamingBinaryDownloader(
                session=_Session([_Response(200, self.body, headers=headers)])
            ).download(request)
        self.assertFalse(request.part_path.exists())

    def test_chunked_response_rejects_underflow_and_overflow(self) -> None:
        headers = _headers(len(self.body))
        headers.pop("Content-Length")
        for label, chunks in (
            ("underflow", [b"abc"]),
            ("overflow", [b"abcdefg"]),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                request = replace(
                    self._request(), destination=Path(temporary) / "archive.zip"
                )
                with self.assertRaises(BinaryContentError):
                    StreamingBinaryDownloader(
                        session=_Session(
                            [_Response(200, headers=headers, chunks=chunks)]
                        )
                    ).download(request)
                self.assertFalse(request.destination.exists())

    def test_chunked_partial_restarts_without_range_validators(self) -> None:
        headers = _headers(len(self.body))
        for name in ("Content-Length", "ETag", "Last-Modified", "Accept-Ranges"):
            headers.pop(name)
        request = self._request()
        calls = 0

        def cancel() -> bool:
            nonlocal calls
            calls += 1
            return calls >= 3

        with self.assertRaises(DownloadCancelledError):
            StreamingBinaryDownloader(
                session=_Session(
                    [_Response(200, headers=headers, chunks=[b"abc", b"def"])]
                )
            ).download(request, cancel_check=cancel)
        self.assertEqual(request.part_path.read_bytes(), b"abc")

        session = _Session([_Response(200, headers=headers, chunks=[self.body])])
        result = StreamingBinaryDownloader(session=session).download(request)
        self.assertEqual(result.path.read_bytes(), self.body)
        self.assertNotIn("Range", session.calls[0]["headers"])
        self.assertEqual(result.manifest.resumed_from_bytes, 0)

    def test_signed_transport_uses_stable_identity_in_all_sidecars(self) -> None:
        signed_url = (
            "https://example-bucket.s3.us-west-2.amazonaws.com/archive.zip?"
            "X-Amz-Credential=credential-secret&X-Amz-Signature=signature-secret&"
            "X-Amz-Security-Token=token-secret"
        )
        identity_url = (
            "https://sciencebase.usgs.gov/manager/download/cmla2h2m7009d0uob0liecbur"
        )
        request = replace(
            self._request(),
            url=signed_url,
            identity_url=identity_url,
        )
        self.assertNotIn(signed_url, repr(request))
        self.assertNotIn("signature-secret", repr(request))
        session = _Session([_Response(200, self.body, headers=_headers(6))])
        result = StreamingBinaryDownloader(session=session).download(request)

        self.assertEqual(session.calls[0]["url"], signed_url)
        self.assertEqual(result.manifest.public_url, identity_url)
        serialized = request.manifest_path.read_text(encoding="utf-8")
        self.assertNotIn("credential-secret", serialized)
        self.assertNotIn("signature-secret", serialized)
        self.assertNotIn("token-secret", serialized)
        self.assertNotIn("amazonaws.com", serialized)

        rotated = replace(
            request,
            url=(
                "https://example-bucket.s3.us-west-2.amazonaws.com/archive.zip?"
                "X-Amz-Signature=rotated-secret"
            ),
        )
        offline = StreamingBinaryDownloader(session=_Session([]))
        cached = offline.load_cached(rotated)
        self.assertTrue(cached.cache_hit)
        self.assertEqual(offline.request_count, 0)

    def test_signed_transport_partial_records_only_stable_identity(self) -> None:
        signed_url = (
            "https://example-bucket.s3.us-west-2.amazonaws.com/archive.zip?"
            "X-Amz-Signature=signature-secret"
        )
        identity_url = "https://sciencebase.usgs.gov/manager/download/archive-cuid"
        request = replace(self._request(), url=signed_url, identity_url=identity_url)
        calls = 0

        def cancel() -> bool:
            nonlocal calls
            calls += 1
            return calls >= 3

        with self.assertRaises(DownloadCancelledError):
            StreamingBinaryDownloader(
                session=_Session(
                    [_Response(200, headers=_headers(6), chunks=[b"abc", b"def"])]
                )
            ).download(request, cancel_check=cancel)

        serialized = request.part_manifest_path.read_text(encoding="utf-8")
        self.assertIn(identity_url, serialized)
        self.assertNotIn("signature-secret", serialized)
        self.assertNotIn("amazonaws.com", serialized)

    def test_transport_exception_does_not_echo_signed_url(self) -> None:
        signed_url = (
            "https://example-bucket.s3.us-west-2.amazonaws.com/archive.zip?"
            "X-Amz-Signature=signature-secret"
        )
        identity_url = "https://sciencebase.usgs.gov/manager/download/archive-cuid"
        request = replace(self._request(), url=signed_url, identity_url=identity_url)
        session = _Session([])

        def fail(*_args: object, **_kwargs: object) -> None:
            raise requests.ConnectionError(f"failed request for {signed_url}")

        session.get = fail  # type: ignore[method-assign]
        with self.assertRaises(BinaryDownloadError) as raised:
            StreamingBinaryDownloader(session=session).download(request)
        message = str(raised.exception)
        self.assertIn(identity_url, message)
        self.assertNotIn("signature-secret", message)
        self.assertNotIn("amazonaws.com", message)
        formatted = "".join(traceback.format_exception(raised.exception))
        self.assertNotIn("signature-secret", formatted)
        self.assertNotIn("amazonaws.com", formatted)

    def test_streaming_exception_does_not_echo_signed_url(self) -> None:
        signed_url = (
            "https://example-bucket.s3.us-west-2.amazonaws.com/archive.zip?"
            "X-Amz-Signature=signature-secret"
        )
        identity_url = "https://sciencebase.usgs.gov/manager/download/archive-cuid"
        request = replace(self._request(), url=signed_url, identity_url=identity_url)

        class FailingResponse(_Response):
            def iter_content(self, *, chunk_size: int):
                del chunk_size
                yield b"abc"
                raise requests.ConnectionError(f"failed stream for {signed_url}")

        response = FailingResponse(200, headers=_headers(6))
        with self.assertRaises(BinaryDownloadError) as raised:
            StreamingBinaryDownloader(session=_Session([response])).download(request)
        formatted = "".join(traceback.format_exception(raised.exception))
        self.assertIn(identity_url, formatted)
        self.assertNotIn("signature-secret", formatted)
        self.assertNotIn("amazonaws.com", formatted)
        partial = request.part_manifest_path.read_text(encoding="utf-8")
        self.assertIn(identity_url, partial)
        self.assertNotIn("signature-secret", partial)

    def test_content_length_failure_preserves_previous_valid_archive(self) -> None:
        request = self._request()
        first = StreamingBinaryDownloader(
            session=_Session([_Response(200, self.body, headers=_headers(6))])
        )
        first.download(request)
        old_archive = request.destination.read_bytes()
        old_manifest = request.manifest_path.read_bytes()

        failed = StreamingBinaryDownloader(
            session=_Session([_Response(200, b"abc", headers=_headers(6))])
        )
        with self.assertRaises(BinaryContentError):
            failed.download(request, refresh=True)
        self.assertEqual(request.destination.read_bytes(), old_archive)
        self.assertEqual(request.manifest_path.read_bytes(), old_manifest)
        self.assertEqual(request.part_path.read_bytes(), b"abc")

    def test_cancelled_partial_is_retained_then_resumed_safely(self) -> None:
        request = self._request()
        calls = 0

        def cancel() -> bool:
            nonlocal calls
            calls += 1
            return calls >= 3

        initial_session = _Session(
            [_Response(200, headers=_headers(6), chunks=[b"abc", b"def"])]
        )
        with self.assertRaises(DownloadCancelledError):
            StreamingBinaryDownloader(session=initial_session).download(
                request, cancel_check=cancel
            )
        self.assertEqual(request.part_path.read_bytes(), b"abc")

        resumed_session = _Session(
            [
                _Response(
                    206,
                    b"def",
                    headers=_headers(3, content_range="bytes 3-5/6"),
                )
            ]
        )
        resumed = StreamingBinaryDownloader(session=resumed_session).download(request)
        self.assertEqual(resumed.path.read_bytes(), self.body)
        self.assertEqual(resumed.manifest.resumed_from_bytes, 3)
        headers = resumed_session.calls[0]["headers"]
        self.assertEqual(headers["Range"], "bytes=3-")
        self.assertEqual(headers["If-Range"], '"release-1"')

    def test_changed_etag_rejects_partial_and_restarts_from_zero(self) -> None:
        request = self._request()
        request.part_path.write_bytes(b"abc")
        request.part_manifest_path.write_text(
            json.dumps(
                {
                    "public_url": request.public_url,
                    "expected_size_bytes": 6,
                    "etag": '"release-1"',
                    "last_modified": "Thu, 13 Aug 2026 12:00:00 GMT",
                    "downloaded_bytes": 3,
                }
            ),
            encoding="utf-8",
        )
        session = _Session(
            [
                _Response(
                    206,
                    b"XYZ",
                    headers=_headers(
                        3,
                        etag='"release-2"',
                        content_range="bytes 3-5/6",
                    ),
                ),
                _Response(
                    200,
                    self.body,
                    headers=_headers(6, etag='"release-2"'),
                ),
            ]
        )
        result = StreamingBinaryDownloader(session=session).download(request)
        self.assertEqual(result.path.read_bytes(), self.body)
        self.assertEqual(len(session.calls), 2)
        self.assertIn("Range", session.calls[0]["headers"])
        self.assertNotIn("Range", session.calls[1]["headers"])
        self.assertEqual(result.manifest.resumed_from_bytes, 0)

    def test_retries_only_transient_statuses(self) -> None:
        session = _Session(
            [
                _Response(503),
                _Response(200, self.body, headers=_headers(6)),
            ]
        )
        sleeps = []
        downloader = StreamingBinaryDownloader(
            session=session, sleep=sleeps.append, max_retries=2
        )
        downloader.download(self._request())
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(sleeps, [0.5])

        bad_request = replace(self._request(), destination=self.root / "not-found.zip")
        not_found = _Session([_Response(404), _Response(200)])
        with self.assertRaises(BinaryHttpStatusError):
            StreamingBinaryDownloader(session=not_found).download(bad_request)
        self.assertEqual(len(not_found.calls), 1)

    def test_transport_error_is_not_retried(self) -> None:
        session = _Session([])

        def fail(*_args: object, **_kwargs: object) -> None:
            session.calls.append({"failed": True})
            raise requests.ConnectionError("offline")

        session.get = fail  # type: ignore[method-assign]
        with self.assertRaisesRegex(BinaryDownloadError, "Request failed"):
            StreamingBinaryDownloader(session=session, max_retries=3).download(
                self._request()
            )
        self.assertEqual(len(session.calls), 1)

    def test_cache_hash_tampering_is_rejected(self) -> None:
        request = self._request()
        downloader = StreamingBinaryDownloader(
            session=_Session([_Response(200, self.body, headers=_headers(6))])
        )
        downloader.download(request)
        request.destination.write_bytes(b"tampered")
        with self.assertRaises(BinaryCacheError):
            downloader.load_cached(request)

    def test_refresh_finalization_failure_restores_prior_archive_and_manifest(
        self,
    ) -> None:
        request = replace(self._request(), expected_sha256=None)
        StreamingBinaryDownloader(
            session=_Session([_Response(200, self.body, headers=_headers(6))])
        ).download(request)
        old_archive = request.destination.read_bytes()
        old_manifest = request.manifest_path.read_bytes()
        pending_manifest = request.manifest_path.with_name(
            f".{request.manifest_path.name}.pending"
        )
        replacement = b"ABCDEF"
        real_replace = __import__("os").replace
        failed = False

        def fail_manifest_promotion(source: object, destination: object) -> None:
            nonlocal failed
            if (
                not failed
                and Path(source) == pending_manifest
                and Path(destination) == request.manifest_path
            ):
                failed = True
                raise OSError("injected manifest promotion failure")
            real_replace(source, destination)

        downloader = StreamingBinaryDownloader(
            session=_Session(
                [_Response(200, replacement, headers=_headers(len(replacement)))]
            )
        )
        with (
            patch(
                "regional_analysis.hydrometry.binary_download.os.replace",
                side_effect=fail_manifest_promotion,
            ),
            self.assertRaisesRegex(BinaryCacheError, "prior valid cache was restored"),
        ):
            downloader.download(request, refresh=True)
        self.assertTrue(failed)
        self.assertEqual(request.destination.read_bytes(), old_archive)
        self.assertEqual(request.manifest_path.read_bytes(), old_manifest)
        self.assertFalse(request.previous_path.exists())
        self.assertFalse(request.previous_manifest_path.exists())
        self.assertFalse(request.transaction_path.exists())

    def test_corrupt_zip_crc_fails_before_replacing_prior_valid_cache(self) -> None:
        with BytesIO() as stream:
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_STORED) as bundle:
                bundle.writestr("bed.tif", b"surveyed-raster-values" * 20)
            valid = stream.getvalue()
        corrupt = bytearray(valid)
        with zipfile.ZipFile(BytesIO(valid), "r") as bundle:
            info = bundle.getinfo("bed.tif")
        name_length = int.from_bytes(
            valid[info.header_offset + 26 : info.header_offset + 28], "little"
        )
        extra_length = int.from_bytes(
            valid[info.header_offset + 28 : info.header_offset + 30], "little"
        )
        payload_start = info.header_offset + 30 + name_length + extra_length
        corrupt[payload_start] ^= 0x01
        self.assertEqual(len(valid), len(corrupt))

        request = BinaryDownloadRequest(
            provider="usgs_sciencebase",
            url="https://sciencebase.usgs.gov/file/topobathy.zip",
            destination=self.root / "topobathy.zip",
            expected_size_bytes=len(valid),
        )
        StreamingBinaryDownloader(
            session=_Session([_Response(200, valid, headers=_headers(len(valid)))])
        ).download(request)
        old_archive = request.destination.read_bytes()
        old_manifest = request.manifest_path.read_bytes()

        with self.assertRaisesRegex(BinaryContentError, "integrity validation failed"):
            StreamingBinaryDownloader(
                session=_Session(
                    [
                        _Response(
                            200,
                            bytes(corrupt),
                            headers=_headers(len(corrupt), etag='"release-2"'),
                        )
                    ]
                )
            ).download(request, refresh=True)
        self.assertEqual(request.destination.read_bytes(), old_archive)
        self.assertEqual(request.manifest_path.read_bytes(), old_manifest)
        manifest = json.loads(old_manifest)
        self.assertTrue(manifest["zip_crc_validated"])


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as bundle:
        for name, body in members.items():
            bundle.writestr(name, body)


class ZipAcquisitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_zip64_inventory_disk_gate_and_selective_extraction(self) -> None:
        archive = self.root / "source.zip"
        with zipfile.ZipFile(archive, "w", allowZip64=True) as bundle:
            with bundle.open("nested/bed.tif", "w", force_zip64=True) as stream:
                stream.write(b"raster-data")
            bundle.writestr("metadata/readme.xml", b"<metadata />")
            bundle.writestr("unrelated.bin", b"do-not-extract")
        inventory = inspect_zip_archive(archive)
        selected = select_archive_raster(inventory)
        self.assertEqual(selected.name, "nested/bed.tif")
        self.assertEqual(inventory.total_uncompressed_bytes, 37)

        insufficient = calculate_disk_requirements(
            inventory,
            selected_members=(selected.name, "metadata/readme.xml"),
            working_directory=self.root,
            quantitative_derivative_bytes=100,
            rendering_output_bytes=50,
            available_free_bytes=1,
        )
        self.assertFalse(insufficient.sufficient)
        with self.assertRaises(InsufficientDiskSpaceError):
            require_sufficient_disk_space(insufficient)

        sufficient = replace(insufficient, available_free_bytes=10_000, sufficient=True)
        outputs = safe_extract_selected(
            archive,
            output_dir=self.root / "extracted",
            selected_members=(selected.name, "metadata/readme.xml"),
            inventory=inventory,
            disk_requirements=sufficient,
            chunk_size=3,
        )
        self.assertEqual(outputs[0].output_path.read_bytes(), b"raster-data")
        self.assertEqual(outputs[1].output_path.read_bytes(), b"<metadata />")
        self.assertFalse((self.root / "extracted" / "unrelated.bin").exists())
        manifest = write_extraction_manifest(
            self.root / "source_manifest.json",
            inventory=inventory,
            extracted_members=outputs,
            relative_to=self.root,
        )
        self.assertEqual(manifest.archive_sha256, inventory.archive_sha256)
        self.assertEqual(len(manifest.extracted_members), 2)

    def test_ambiguous_raster_selection_fails(self) -> None:
        archive = self.root / "ambiguous.zip"
        _write_zip(archive, {"one.tif": b"1", "two.tiff": b"2"})
        inventory = inspect_zip_archive(archive)
        with self.assertRaises(ZipInventoryError):
            select_archive_raster(inventory)
        self.assertEqual(
            select_archive_raster(inventory, expected_name="two.tiff").name,
            "two.tiff",
        )

    def test_path_traversal_is_rejected_before_any_extraction(self) -> None:
        archive = self.root / "traversal.zip"
        _write_zip(archive, {"bed.tif": b"ok", "../outside.txt": b"bad"})
        with self.assertRaises(ZipInventoryError):
            inspect_zip_archive(archive)
        self.assertFalse((self.root.parent / "outside.txt").exists())

    def test_symlink_is_rejected_and_prior_file_is_preserved(self) -> None:
        archive = self.root / "symlink.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("bed.tif", b"new")
            link = zipfile.ZipInfo("link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            bundle.writestr(link, b"bed.tif")
        inventory = inspect_zip_archive(archive)
        destination = self.root / "out" / "bed.tif"
        destination.parent.mkdir()
        destination.write_bytes(b"prior")
        with self.assertRaises(ZipInventoryError):
            safe_extract_selected(
                archive,
                output_dir=destination.parent,
                selected_members=("bed.tif",),
                inventory=inventory,
            )
        self.assertEqual(destination.read_bytes(), b"prior")


if __name__ == "__main__":
    unittest.main()
