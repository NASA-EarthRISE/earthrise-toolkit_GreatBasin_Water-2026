"""Credential-free tests for the Phase 2 metadata-first audit."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pyproj import CRS

from regional_analysis.hydrometry.phase2_bear_lake import _run_metadata_audit
from regional_analysis.hydrometry.phase2_metadata import (
    inspect_spatial_coverage_bundle,
    metadata_audit_markdown,
)
from regional_analysis.hydrometry.sciencebase import (
    ScienceBaseFile,
    ScienceBaseItem,
)


def _dbf_bytes(rows: list[dict[str, str]]) -> bytes:
    fields = (
        ("Name", "C", 20, 0),
        ("Title", "C", 70, 0),
        ("Date_Acq", "C", 10, 0),
        ("Date_Pub", "C", 10, 0),
        ("Data_Type", "C", 70, 0),
        ("Source_VD", "C", 15, 0),
        ("Source_Res", "C", 35, 0),
        ("Source_HD", "C", 35, 0),
        ("Source_Unt", "C", 20, 0),
        ("Source_Geo", "C", 20, 0),
        ("Area_km2", "N", 13, 5),
    )
    header_length = 32 + 32 * len(fields) + 1
    record_length = 1 + sum(field[2] for field in fields)
    header = bytearray(header_length)
    header[0] = 0x03
    struct.pack_into("<I", header, 4, len(rows))
    struct.pack_into("<H", header, 8, header_length)
    struct.pack_into("<H", header, 10, record_length)
    for index, (name, field_type, width, decimals) in enumerate(fields):
        descriptor = bytearray(32)
        encoded_name = name.encode("ascii")[:11]
        descriptor[: len(encoded_name)] = encoded_name
        descriptor[11] = ord(field_type)
        descriptor[16] = width
        descriptor[17] = decimals
        start = 32 + index * 32
        header[start : start + 32] = descriptor
    header[-1] = 0x0D

    records = bytearray()
    for row in rows:
        record = bytearray(b" " + b" " * (record_length - 1))
        position = 1
        for name, field_type, width, decimals in fields:
            value = row.get(name, "")
            if field_type == "N" and value:
                encoded = f"{float(value):{width}.{decimals}f}".encode("ascii")
            else:
                encoded = value.encode("cp1252")[:width].ljust(width, b" ")
            record[position : position + width] = encoded
            position += width
        records.extend(record)
    records.append(0x1A)
    return bytes(header + records)


def _shp_header() -> bytes:
    header = bytearray(100)
    struct.pack_into(">I", header, 0, 9994)
    struct.pack_into(">I", header, 24, 50)
    struct.pack_into("<I", header, 28, 1000)
    struct.pack_into("<I", header, 32, 5)
    struct.pack_into("<4d", header, 36, 465930.0, 4631143.5, 479149.0, 4663558.0)
    return bytes(header)


def _spatial_zip() -> bytes:
    rows = [
        {
            "Name": "bathy",
            "Title": "2002 10-meter interpolated bathymetry",
            "Date_Acq": "2002",
            "Date_Pub": "2003",
            "Data_Type": "Interpolated bathymetry from field observations",
            "Source_VD": "NAVD88",
            "Source_Res": "10 m",
            "Source_HD": "NAD83 / UTM zone 12N",
            "Source_Unt": "meters",
            "Source_Geo": "NA",
            "Area_km2": "223.441",
        },
        {
            "Name": "lidar_2016",
            "Title": "2016 0.5-meter QL1 lidar",
            "Date_Acq": "2016",
            "Date_Pub": "2017",
            "Data_Type": "Topographic lidar",
            "Source_VD": "NAVD88",
            "Source_Res": "0.5 m",
            "Source_HD": "NAD83(2011) / UTM zone 12N",
            "Source_Unt": "meters",
            "Source_Geo": "GEOID12B",
            "Area_km2": "37.5702",
        },
        {
            "Name": "lidar_2018",
            "Title": "2018 0.5-meter QL1 lidar",
            "Date_Acq": "2018",
            "Date_Pub": "2019",
            "Data_Type": "Topographic lidar",
            "Source_VD": "NAVD88",
            "Source_Res": "0.5 m",
            "Source_HD": "NAD83(2011) / UTM zone 12N",
            "Source_Unt": "meters",
            "Source_Geo": "GEOID12B",
            "Area_km2": "1.37255",
        },
        {
            "Name": "gap_interp",
            "Title": "2026 0.5-meter shoreline gap interpolation",
            "Date_Acq": "2026",
            "Date_Pub": "2026",
            "Data_Type": "Interpolated gap between lidar and bathymetry",
            "Source_VD": "NAVD88",
            "Source_Res": "0.5 m",
            "Source_HD": "NAD83(2011) / UTM zone 12N",
            "Source_Unt": "meters",
            "Source_Geo": "GEOID12B",
            "Area_km2": "40.308",
        },
    ]
    projection = CRS.from_epsg(6341).to_wkt()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("coverage.prj", projection)
        archive.writestr("coverage.dbf", _dbf_bytes(rows))
        archive.writestr("coverage.shp", _shp_header())
    return stream.getvalue()


def _fgdc_xml() -> bytes:
    return b"""<?xml version='1.0' encoding='UTF-8'?>
<metadata>
  <idinfo>
    <citation><citeinfo><title>Bear Lake official metadata</title></citeinfo></citation>
    <descript><abstract>Whole-lake topobathymetry in NAVD88 meters.</abstract></descript>
    <spdom><bounding>
      <westbc>-111.4335</westbc><eastbc>-111.2464</eastbc>
      <northbc>42.1278</northbc><southbc>41.8211</southbc>
    </bounding></spdom>
  </idinfo>
  <spref><horizsys><planar><utm><utmzone>12</utmzone></utm>
    <planci><plandu>meters</plandu><coordrep><absres>0.5</absres></coordrep></planci>
  </planar><geodetic><horizdn>NAD83 (2011)</horizdn></geodetic></horizsys>
  <vertdef><altsys><altdatum>North American Vertical Datum of 1988 (NAVD88) GEOID12B</altdatum>
    <altunits>meters</altunits></altsys></vertdef></spref>
  <eainfo><detailed><enttyp><enttypd>north arm of Bear Lake</enttypd></enttyp></detailed></eainfo>
</metadata>"""


class Phase2SpatialMetadataTests(unittest.TestCase):
    def test_structured_coverage_separates_surveys_from_2026_processing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "coverage.zip"
            path.write_bytes(_spatial_zip())
            result = inspect_spatial_coverage_bundle(path)

        self.assertEqual(result["horizontal_epsg"], 6341)
        self.assertEqual(result["survey_source_years"], [2002, 2016, 2018])
        self.assertEqual(result["derived_processing_years"], [2026])
        self.assertEqual(result["coverage_record_count"], 4)
        self.assertAlmostEqual(result["projected_width_m"], 13219.0)
        self.assertAlmostEqual(result["projected_height_m"], 32414.5)
        gap = result["gap_interpolation"]
        self.assertTrue(gap["present"])
        self.assertIn("not a 2026 field survey", gap["interpretation"])
        self.assertFalse(
            any(
                source["numeric_source_uncertainty_published"]
                for source in result["coverage_records"]
            )
        )

    def test_markdown_explains_scope_and_archive_gate(self) -> None:
        record = {
            "status": "passed",
            "run_mode": "offline",
            "doi": "10.5066/P14KZS2D",
            "sciencebase_item_id": "698226f4b66b018d00102e42",
            "archive_name": "Bear_Lake_TBDEM_2026_50cm.zip",
            "archive_size_bytes": 1_339_051_329,
            "archive_downloaded": False,
            "vertical_datum": "NAVD88",
            "vertical_units": "metres",
            "bounded_metadata_request_count": 0,
            "fgdc_metadata": {
                "geographic_bounds": {
                    "west_longitude": -111.4335,
                    "east_longitude": -111.2464,
                    "south_latitude": 41.8211,
                    "north_latitude": 42.1278,
                }
            },
            "spatial_metadata_archive": {
                "horizontal_crs_name": "NAD83(2011) / UTM zone 12N",
                "horizontal_epsg": 6341,
                "survey_source_years": [2002, 2016, 2018],
                "derived_processing_years": [2026],
                "coverage_records": [],
                "gap_interpolation": {
                    "interpretation": "2026 is derived interpolation."
                },
            },
        }
        text = metadata_audit_markdown(record)
        self.assertIn("EPSG:6341", text)
        self.assertIn("whole-lake spatial coverage", text)
        self.assertIn("large archive is absent", text)
        self.assertIn("No storage volume was calculated", text)


class _FakeDownloader:
    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.calls: list[tuple[str, bool, bool]] = []
        self.request_count = 0

    def _result(self, request, *, refresh: bool, live: bool):
        body = self.bodies[request.destination.name]
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        request.destination.write_bytes(body)
        manifest_path = request.destination.with_name(
            f"{request.destination.name}.manifest.json"
        )
        manifest_path.write_text("{}", encoding="utf-8")
        self.calls.append((request.destination.name, refresh, live))
        if live:
            self.request_count += 1
        return SimpleNamespace(
            path=request.destination,
            manifest_path=manifest_path,
            manifest=SimpleNamespace(sha256=hashlib.sha256(body).hexdigest()),
            bytes_transferred=len(body) if live else 0,
            cache_hit=not live,
        )

    def download(self, request, *, refresh=False):
        return self._result(request, refresh=refresh, live=True)

    def load_cached(self, request):
        return self._result(request, refresh=False, live=False)


class Phase2MetadataAuditFlowTests(unittest.TestCase):
    def _source(self, name: str, content_type: str, body: bytes) -> ScienceBaseFile:
        return ScienceBaseFile(
            name=name,
            content_type=content_type,
            size_bytes=len(body),
            download_url=f"https://sciencebase.example.test/{name}",
            checksum_algorithm=None,
            checksum_value=None,
        )

    def test_refresh_reaches_each_bounded_download_and_offline_preserves_live_run(
        self,
    ) -> None:
        spatial = _spatial_zip()
        xml = _fgdc_xml()
        thumbnail = b"\x89PNG\r\n\x1a\nsmall-test-thumbnail"
        raster = b"large-raster-placeholder"
        files = (
            replace(
                self._source("source.zip", "application/zip", raster),
                path_on_disk="__s3__",
                storage_key="698226f4b66b018d00102e42/source.zip",
            ),
            self._source("spatial.zip", "application/zip", spatial),
            self._source("metadata.xml", "application/fgdc+xml", xml),
            self._source("thumbnail.png", "image/png", thumbnail),
        )
        item = ScienceBaseItem(
            item_id="698226f4b66b018d00102e42",
            title="Bear Lake official release",
            summary=(
                "NAVD 88; UTM zone 12; North American Datum of 1983; "
                "GEOID12B; 0.5-meter; surveys 2002, 2016, and 2018"
            ),
            doi="10.5066/P14KZS2D",
            citation=None,
            rights="CC0",
            publication_date="2026-03-31",
            item_last_updated=None,
            metadata_url="https://sciencebase.example.test/item",
            files=files,
        )
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "phase2"
            settings = SimpleNamespace(
                raster_archive_name="source.zip",
                spatial_metadata_archive_name="spatial.zip",
                fgdc_metadata_name="metadata.xml",
                thumbnail_name="thumbnail.png",
                data_dir=data_dir,
                doi="10.5066/P14KZS2D",
            )
            bodies = {
                "spatial.zip": spatial,
                "metadata.xml": xml,
                "thumbnail.png": thumbnail,
            }
            live_downloader = _FakeDownloader(bodies)
            with patch(
                "regional_analysis.hydrometry.phase2_bear_lake.StreamingBinaryDownloader",
                return_value=live_downloader,
            ):
                live = _run_metadata_audit(
                    settings, SimpleNamespace(item=item), live=True, refresh=True
                )
            self.assertEqual(live["horizontal_epsg"], 6341)
            self.assertEqual(live["geographic_bounds"]["north_latitude"], 42.1278)
            self.assertEqual(live["bounded_metadata_request_count"], 3)
            self.assertFalse(live["archive_downloaded_by_metadata_action"])
            self.assertFalse(live["archive_present_at_audit"])
            self.assertEqual(
                live["metadata_product_inventory"][
                    "expected_internal_raster_filename_status"
                ],
                "requires_archive_central_directory_inspection",
            )
            self.assertTrue(all(call[1] for call in live_downloader.calls))
            self.assertTrue((data_dir / "metadata_audit.md").is_file())

            offline_downloader = _FakeDownloader(bodies)
            with patch(
                "regional_analysis.hydrometry.phase2_bear_lake.StreamingBinaryDownloader",
                return_value=offline_downloader,
            ):
                offline = _run_metadata_audit(
                    settings, SimpleNamespace(item=item), live=False
                )
            self.assertEqual(offline["bounded_metadata_request_count"], 0)
            self.assertEqual(offline["latest_live_run"]["request_count"], 3)
            saved = json.loads(
                (data_dir / "metadata_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["latest_live_run"]["request_count"], 3)


if __name__ == "__main__":
    unittest.main()
