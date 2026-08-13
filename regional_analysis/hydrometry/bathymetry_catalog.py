"""Metadata-only catalog for official pilot bathymetry/capacity products.

No raster, table, report, or archive is downloaded in Phase 0.  DOI probes use
small citation-metadata responses, while the Flaming Gorge probe validates its
RISE catalog record and does not follow linked binary-file URLs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from regional_analysis.hydrometry.adapters.usbr_rise import RiseAdapter
from regional_analysis.hydrometry.cache import CacheMode, CacheStore
from regional_analysis.hydrometry.http import HttpClient
from regional_analysis.hydrometry.models import SourceQualityClass

DOI_CSL_CONTENT_TYPE = "application/vnd.citationstyles.csl+json"


class BathymetryCatalogError(RuntimeError):
    """Raised when metadata does not match a configured official record."""


@dataclass(frozen=True, slots=True)
class BathymetryCatalogEntry:
    """Static, cited metadata for one official Phase 0 source."""

    catalog_id: str
    profile_id: str
    hylak_id: int
    display_name: str
    title: str
    provider: str
    doi: str | None
    source_record_id: str | None
    product_type: str
    survey_source_dates: str | None
    horizontal_crs: str | None
    vertical_datum: str | None
    vertical_units: str | None
    spatial_resolution: str | None
    valid_elevation_range: str | None
    raster_available: bool | None
    relation_table_available: bool | None
    relation_table_type: str | None
    license: str | None
    source_quality_candidate: SourceQualityClass
    warnings: tuple[str, ...]
    landing_url: str

    @property
    def Hylak_id(self) -> int:
        """Return the persistent HydroLAKES identifier with output casing."""

        return self.hylak_id


@dataclass(frozen=True, slots=True)
class BathymetryProbeResult:
    """Small landing-record verification result."""

    catalog_id: str
    status: str
    returned_title: str | None
    returned_identifier: str | None
    retrieved_at_utc: str | None
    source_url_public: str
    warning: str | None


BATHYMETRY_CATALOG: tuple[BathymetryCatalogEntry, ...] = (
    BathymetryCatalogEntry(
        catalog_id="usgs_great_salt_topobathymetry",
        profile_id="great_salt",
        hylak_id=67,
        display_name="Great Salt",
        title=(
            "Half-meter topobathymetric elevation model and elevation-area-volume "
            "tables for Great Salt Lake, Utah, 2002-2016"
        ),
        provider="usgs_sciencebase",
        doi="10.5066/P9DGG75W",
        source_record_id=None,
        product_type="topobathymetric_dem_and_elevation_area_volume",
        survey_source_dates="2002-2016",
        horizontal_crs="NAD83 / UTM zone 12N",
        vertical_datum="NAVD88 (Geoid12B)",
        vertical_units="meters (TBDEM); feet (published E-A-V tables)",
        spatial_resolution="0.5 m TBDEM; E-A-V calculations used a 5 m resample",
        valid_elevation_range="4,170.00-4,215.00 ft NAVD88 (E-A-V tables)",
        raster_available=True,
        relation_table_available=True,
        relation_table_type="elevation-area-volume",
        license="CC0 1.0 / U.S. Government public-domain data",
        source_quality_candidate=SourceQualityClass.STAGE_SURVEYED_BATHYMETRY,
        warnings=(
            "USGS gauges are NGVD29 while this product is NAVD88.",
            (
                "North arm, south arm, Bear River Bay, and Farmington Bay require "
                "explicit handling; the two arm gauges are not interchangeable."
            ),
        ),
        landing_url="https://doi.org/10.5066/P9DGG75W",
    ),
    BathymetryCatalogEntry(
        catalog_id="usgs_bear_lake_eav_tbdem",
        profile_id="bear_lake",
        hylak_id=778,
        display_name="Bear Lake",
        title=(
            "Half-meter topobathymetric elevation model and elevation-area-volume "
            "tables for Bear Lake, Utah and Idaho, 2002-2018"
        ),
        provider="usgs_sciencebase",
        doi="10.5066/P14KZS2D",
        source_record_id=None,
        product_type="elevation_area_volume_and_topobathymetric_dem",
        survey_source_dates="2002-2018",
        horizontal_crs="NAD83 / UTM zone 12N",
        vertical_datum="NAVD88 (Geoid12B)",
        vertical_units="meters",
        spatial_resolution="0.5 m TBDEM; E-A-V table at 0.01 m intervals",
        valid_elevation_range="1,747.39-1,808.00 m NAVD88",
        raster_available=True,
        relation_table_available=True,
        relation_table_type="elevation-area-volume",
        license="U.S. Government public-domain data",
        source_quality_candidate=SourceQualityClass.STAGE_OFFICIAL_CAPACITY,
        warnings=(
            (
                "Potential Level 4/5 requires confirmation that the operational "
                "level series has a compatible documented vertical reference."
            ),
        ),
        landing_url="https://doi.org/10.5066/P14KZS2D",
    ),
    BathymetryCatalogEntry(
        catalog_id="usgs_lake_powell_topobathymetry",
        profile_id="lake_powell",
        hylak_id=802,
        display_name="Lake Powell",
        title=(
            "One Meter Topobathymetric Digital Elevation Model for Lake Powell, "
            "Arizona-Utah, 1947-2018"
        ),
        provider="usgs_sciencebase",
        doi="10.5066/P9XX0J1Y",
        source_record_id=None,
        product_type="topobathymetric_dem",
        survey_source_dates="1947-2018 (bathymetry 2017; lidar 2018)",
        horizontal_crs="NAD83 / UTM zone 12N",
        vertical_datum="NAVD88 (Geoid12B)",
        vertical_units="meters",
        spatial_resolution="1 m",
        valid_elevation_range="948.28-1,440.94 m NAVD88",
        raster_available=True,
        relation_table_available=False,
        relation_table_type=None,
        license="U.S. Government public-domain data",
        source_quality_candidate=SourceQualityClass.STAGE_SURVEYED_BATHYMETRY,
        warnings=(
            (
                "A modified TBDEM was used for the 2018 capacity release; product "
                "lineage must be fixed before any future integration."
            ),
        ),
        landing_url="https://doi.org/10.5066/P9XX0J1Y",
    ),
    BathymetryCatalogEntry(
        catalog_id="usgs_lake_powell_capacity_table",
        profile_id="lake_powell",
        hylak_id=802,
        display_name="Lake Powell",
        title="Elevation-area-capacity tables for Lake Powell, 2018",
        provider="usgs_sciencebase",
        doi="10.5066/P9O3IPG3",
        source_record_id=None,
        product_type="elevation_area_capacity_table",
        survey_source_dates="2017-2018",
        horizontal_crs="Derived from NAD83(2011) / UTM zone 12N TBDEM",
        vertical_datum="NAVD88 (Geoid12B)",
        vertical_units="feet in operational table; calculations at 0.10 m intervals",
        spatial_resolution=None,
        valid_elevation_range=None,
        raster_available=False,
        relation_table_available=True,
        relation_table_type="elevation-area-capacity",
        license="CC0 1.0 / U.S. Government public-domain data",
        source_quality_candidate=SourceQualityClass.STAGE_OFFICIAL_CAPACITY,
        warnings=(
            (
                "RISE operational elevations do not report a vertical datum in item "
                "metadata; compatibility with NAVD88 is not established in Phase 0."
            ),
            "No stage-capacity interpolation is performed in Phase 0.",
        ),
        landing_url="https://doi.org/10.5066/P9O3IPG3",
    ),
    BathymetryCatalogEntry(
        catalog_id="rise_flaming_gorge_sedimentation_survey",
        profile_id="flaming_gorge",
        hylak_id=784,
        display_name="Flaming Gorge",
        title="Flaming Gorge Reservoir (Utah/Wyoming) Sedimentation Survey Data",
        provider="usbr_rise",
        doi=None,
        source_record_id="8065",
        product_type="sedimentation_survey_acap_metadata",
        survey_source_dates="May-June 2019 and September 2020",
        horizontal_crs=None,
        vertical_datum=None,
        vertical_units=None,
        spatial_resolution=None,
        valid_elevation_range=None,
        raster_available=None,
        relation_table_available=True,
        relation_table_type="ACAP elevation-area-capacity",
        license=None,
        source_quality_candidate=SourceQualityClass.STAGE_OFFICIAL_CAPACITY,
        warnings=(
            (
                "RISE record metadata does not publish the ACAP vertical datum; read "
                "and validate the survey report before future calculations."
            ),
            (
                "Phase 0 validates metadata only and does not download the report, "
                "ACAP table, or geospatial files."
            ),
        ),
        landing_url="https://data.usbr.gov/rise/api/catalog-record/8065",
    ),
)

_BY_CATALOG_ID: Mapping[str, BathymetryCatalogEntry] = MappingProxyType(
    {entry.catalog_id: entry for entry in BATHYMETRY_CATALOG}
)


def get_bathymetry_entry(
    catalog_id: str,
    *,
    strict: bool = False,
) -> BathymetryCatalogEntry | None:
    """Look up an official metadata record without a network request."""

    entry = _BY_CATALOG_ID.get(catalog_id)
    if entry is None and strict:
        raise KeyError(catalog_id)
    return entry


def entries_for_profile(profile_id: str) -> tuple[BathymetryCatalogEntry, ...]:
    """Return the static entries for one research profile."""

    return tuple(
        entry for entry in BATHYMETRY_CATALOG if entry.profile_id == profile_id
    )


def parse_doi_citation(
    payload: Mapping[str, Any],
    *,
    expected_doi: str,
    expected_title: str,
) -> tuple[str, str]:
    """Validate Crosscite citation metadata without following data-file links."""

    returned_doi = str(payload.get("DOI") or payload.get("doi") or "").strip()
    if returned_doi.casefold() != expected_doi.casefold():
        raise BathymetryCatalogError(
            f"DOI metadata returned {returned_doi!r}, expected {expected_doi!r}."
        )
    title = payload.get("title")
    if isinstance(title, list):
        title = title[0] if title else None
    if not isinstance(title, str) or not title.strip():
        raise BathymetryCatalogError(f"DOI {expected_doi} has no title metadata.")
    expected_words = {
        word.casefold().strip(",:-")
        for word in expected_title.split()
        if len(word.strip(",:-")) >= 5
    }
    returned_words = {word.casefold().strip(",:-") for word in title.split()}
    if expected_words and len(expected_words & returned_words) < min(
        3, len(expected_words)
    ):
        raise BathymetryCatalogError(
            f"DOI {expected_doi} title {title!r} does not match configured product."
        )
    return returned_doi, title.strip()


class BathymetryCatalogProbe:
    """Verify small public metadata records under explicit cache/network mode."""

    def __init__(
        self,
        *,
        client: HttpClient,
        mode: CacheMode,
        cache: CacheStore | None = None,
        refresh: bool = False,
    ) -> None:
        if mode is CacheMode.OFFLINE and cache is None:
            raise ValueError("offline bathymetry probes require a CacheStore")
        self.client = client
        self.mode = mode
        self.cache = cache
        self.refresh = refresh

    def _doi_response(self, entry: BathymetryCatalogEntry) -> Any:
        headers = {"Accept": DOI_CSL_CONTENT_TYPE}
        expected_types = (DOI_CSL_CONTENT_TYPE, "application/json")
        if self.cache is not None:
            return self.cache.request(
                provider=entry.provider,
                client=self.client,
                url=entry.landing_url,
                params=None,
                headers=headers,
                mode=self.mode,
                refresh=self.refresh,
                expected_content_types=expected_types,
                response_schema="crosscite-csl-json-v1",
            )
        if self.mode is CacheMode.OFFLINE:
            raise AssertionError("offline probe created without cache")
        return self.client.get(
            entry.landing_url,
            headers=headers,
            expected_content_types=expected_types,
        )

    def verify(self, entry: BathymetryCatalogEntry) -> BathymetryProbeResult:
        """Verify one DOI or RISE record without downloading its products."""

        if entry.doi:
            response = self._doi_response(entry)
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise BathymetryCatalogError("DOI response must be a JSON object.")
            identifier, title = parse_doi_citation(
                payload, expected_doi=entry.doi, expected_title=entry.title
            )
            retrieved = getattr(response, "retrieved_at_utc", None)
            return BathymetryProbeResult(
                catalog_id=entry.catalog_id,
                status="confirmed",
                returned_title=title,
                returned_identifier=identifier,
                retrieved_at_utc=(retrieved.isoformat() if retrieved else None),
                source_url_public=response.url_public,
                warning=None,
            )

        if entry.provider == "usbr_rise" and entry.source_record_id:
            adapter = RiseAdapter(
                client=self.client,
                mode=self.mode,
                cache=self.cache,
                refresh=self.refresh,
            )
            record = adapter.fetch_record(
                int(entry.source_record_id),
                expected_title_contains=entry.display_name,
            )
            if record.title != entry.title:
                raise BathymetryCatalogError(
                    f"RISE record title {record.title!r} does not match configured "
                    f"title {entry.title!r}."
                )
            return BathymetryProbeResult(
                catalog_id=entry.catalog_id,
                status="confirmed_metadata_only",
                returned_title=record.title,
                returned_identifier=str(record.record_id),
                retrieved_at_utc=None,
                source_url_public=record.source_url_public,
                warning=(
                    "Binary survey products were intentionally not downloaded in Phase 0."
                ),
            )
        raise BathymetryCatalogError(
            f"Catalog entry {entry.catalog_id!r} has no supported metadata probe."
        )
