"""Bounded ScienceBase acquisition for official hydrometry reference tables.

The module deliberately handles only small structured metadata and reference
tables.  It never downloads a bathymetric raster.  Network behavior remains
explicit through :class:`~regional_analysis.hydrometry.cache.CacheMode`, and
all HTTP requests use the Phase 0 client/cache implementation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from .cache import CacheMissError, CacheMode, CacheStore
from .capacity_tables import (
    CapacityTableInputRow,
    CapacityTableMetadata,
    NormalizedCapacityTable,
    normalize_capacity_table,
)
from .http import HttpClient

SCIENCEBASE_ITEMS_URL = "https://www.sciencebase.gov/catalog/items"
SCIENCEBASE_ITEM_URL = "https://www.sciencebase.gov/catalog/item/{item_id}"
BEAR_LAKE_TITLE = (
    "Half-meter topobathymetric elevation model and elevation-area-volume "
    "tables for Bear Lake, Utah and Idaho, 2002-2018"
)
BEAR_LAKE_FILENAME_PATTERN = re.compile(
    r"^Bear_Lake_TBDEM_\d{4}_ElevAreaVolume\.csv$", re.IGNORECASE
)
BEAR_LAKE_COLUMNS = (
    "elev_m_NAVD88",
    "elev_ft_NAVD88",
    "area_ft2",
    "area_m2",
    "area_km2",
    "area_mi2",
    "area_acre",
    "volume_ft3",
    "volume_m3",
    "volume_km3",
    "volume_mi3",
    "volume_acreft",
)
PARSER_VERSION = "bear-lake-eav-csv-v1"
REFERENCE_MANIFEST_VERSION = "hydrometry-reference-source-v1"
MAX_REFERENCE_BYTES = 10_000_000
MAX_ZIP_MEMBER_BYTES = 25_000_000
_ITEM_ID = re.compile(r"^[0-9a-f]{24}$", re.IGNORECASE)
_OFFICIAL_HOSTS = frozenset(
    {
        "www.sciencebase.gov",
        "sciencebase.gov",
        "sciencebase.usgs.gov",
    }
)


class ScienceBaseError(RuntimeError):
    """Base error for reference-source discovery or acquisition."""


class ScienceBaseSchemaError(ScienceBaseError):
    """Raised when official structured metadata has an unexpected schema."""


class ScienceBaseIdentityError(ScienceBaseError):
    """Raised when a DOI, item, title, or file does not match expectations."""


class ScienceBaseFileSelectionError(ScienceBaseError):
    """Raised when an official E-A-V file cannot be selected unambiguously."""


class ScienceBaseContentError(ScienceBaseError):
    """Raised when downloaded bytes fail type, length, hash, or CSV checks."""


@dataclass(frozen=True, slots=True)
class DoiMetadata:
    """Small normalized DOI citation record."""

    doi: str
    title: str
    published: str | None
    license_url: str | None


@dataclass(frozen=True, slots=True)
class ScienceBaseFile:
    """One public file advertised by a ScienceBase item."""

    name: str
    content_type: str
    size_bytes: int
    download_url: str
    checksum_algorithm: str | None
    checksum_value: str | None


@dataclass(frozen=True, slots=True)
class ScienceBaseItem:
    """Validated ScienceBase data-release identity and file inventory."""

    item_id: str
    title: str
    summary: str
    doi: str
    citation: str | None
    rights: str | None
    publication_date: str | None
    item_last_updated: str | None
    metadata_url: str
    files: tuple[ScienceBaseFile, ...]

    @property
    def source_version(self) -> str:
        """Return a deterministic release version suitable for provenance."""

        if self.publication_date:
            return self.publication_date
        if self.item_last_updated:
            return self.item_last_updated
        return self.item_id


@dataclass(frozen=True, slots=True)
class ParsedBearLakeEav:
    """Parsed official table plus source-layout diagnostics."""

    table: NormalizedCapacityTable
    valid_row_count: int
    fully_blank_row_count: int
    physical_record_count: int
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BearLakeEavManifest:
    """Stable, public provenance for the locally retained reference table."""

    schema_version: str
    doi: str
    landing_url: str
    sciencebase_item_id: str
    resolved_metadata_url: str
    download_url: str
    source_filename: str
    provider_title: str
    retrieved_at_utc: str
    content_type: str
    content_length: int
    sha256: str
    provider_checksum_algorithm: str | None
    provider_checksum_value: str | None
    source_release_version: str
    item_last_updated: str | None
    license: str | None
    vertical_datum: str
    geoid_model: str
    source_elevation_units: tuple[str, ...]
    source_area_units: tuple[str, ...]
    source_storage_units: tuple[str, ...]
    source_row_count: int
    ignored_fully_blank_row_count: int
    source_path: str
    metadata_path: str
    metadata_sha256: str
    parser_version: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe detached manifest."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BearLakeEavSource:
    """Acquired Bear Lake relation and all paths needed by orchestration."""

    table: NormalizedCapacityTable
    item: ScienceBaseItem
    source_file: ScienceBaseFile
    manifest: BearLakeEavManifest
    source_path: Path
    metadata_path: Path
    manifest_path: Path
    cache_hit: bool


def normalize_doi(value: str) -> str:
    """Normalize a DOI or DOI URL without accepting an empty identifier."""

    if not isinstance(value, str) or not value.strip():
        raise ScienceBaseIdentityError("DOI must be non-empty.")
    normalized = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.casefold().startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    if not normalized.casefold().startswith("10.") or "/" not in normalized:
        raise ScienceBaseIdentityError(f"Invalid DOI {value!r}.")
    return normalized.upper()


def _title(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = next((item for item in value if isinstance(item, str)), None)
    if not isinstance(value, str) or not value.strip():
        raise ScienceBaseSchemaError("Metadata title is missing.")
    return value.strip()


def parse_doi_metadata(
    payload: Mapping[str, Any],
    *,
    expected_doi: str,
    expected_title: str,
) -> DoiMetadata:
    """Validate a DOI citation payload such as Crosscite CSL JSON."""

    returned_doi = normalize_doi(str(payload.get("DOI") or payload.get("doi") or ""))
    wanted_doi = normalize_doi(expected_doi)
    if returned_doi != wanted_doi:
        raise ScienceBaseIdentityError(
            f"DOI metadata returned {returned_doi!r}, expected {wanted_doi!r}."
        )
    returned_title = _title(payload.get("title"))
    if returned_title.casefold() != expected_title.strip().casefold():
        raise ScienceBaseIdentityError(
            f"DOI title {returned_title!r} does not match {expected_title!r}."
        )
    published = payload.get("published") or payload.get("issued")
    if isinstance(published, Mapping):
        date_parts = published.get("date-parts")
        if (
            isinstance(date_parts, Sequence)
            and date_parts
            and isinstance(date_parts[0], Sequence)
        ):
            published = "-".join(str(part) for part in date_parts[0])
        else:
            published = None
    elif published is not None:
        published = str(published)
    license_url = payload.get("license")
    return DoiMetadata(
        doi=returned_doi,
        title=returned_title,
        published=published,
        license_url=str(license_url) if license_url else None,
    )


def _require_official_url(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScienceBaseSchemaError(f"{label} URL is missing.")
    url = value.strip()
    parts = urlsplit(url)
    if parts.scheme.lower() != "https" or (parts.hostname or "").lower() not in (
        _OFFICIAL_HOSTS
    ):
        raise ScienceBaseIdentityError(
            f"{label} URL is not an approved official ScienceBase URL: {url}"
        )
    return url


def parse_sciencebase_search(
    payload: Mapping[str, Any],
    *,
    expected_title: str,
) -> tuple[str, str]:
    """Require exactly one expected item from a bounded DOI search response."""

    items = payload.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ScienceBaseSchemaError("ScienceBase search response lacks an items list.")
    total = payload.get("total")
    if isinstance(total, bool):
        raise ScienceBaseSchemaError("ScienceBase search total is invalid.")
    try:
        total_count = int(total)
    except (TypeError, ValueError) as exc:
        raise ScienceBaseSchemaError("ScienceBase search total is invalid.") from exc
    if total_count != len(items):
        raise ScienceBaseIdentityError(
            "ScienceBase DOI search was incomplete or paginated unexpectedly: "
            f"total={total_count}, returned={len(items)}."
        )
    matches: list[tuple[str, str]] = []
    for raw in items:
        if not isinstance(raw, Mapping):
            raise ScienceBaseSchemaError("ScienceBase search item is not an object.")
        title = _title(raw.get("title"))
        if title.casefold() != expected_title.strip().casefold():
            continue
        item_id = str(raw.get("id") or "").strip()
        if not _ITEM_ID.fullmatch(item_id):
            raise ScienceBaseIdentityError(
                f"ScienceBase returned invalid item ID {item_id!r}."
            )
        link = raw.get("link")
        if not isinstance(link, Mapping):
            raise ScienceBaseSchemaError("ScienceBase search item lacks its link.")
        metadata_url = _require_official_url(link.get("url"), "item metadata")
        if not metadata_url.rstrip("/").endswith(f"/{item_id}"):
            raise ScienceBaseIdentityError(
                "ScienceBase item ID and metadata URL do not agree."
            )
        matches.append((item_id, metadata_url))
    if len(matches) != 1:
        raise ScienceBaseIdentityError(
            "ScienceBase DOI search did not resolve to exactly one item with the "
            f"expected title; found {len(matches)}."
        )
    return matches[0]


def _publication_date(payload: Mapping[str, Any]) -> str | None:
    dates = payload.get("dates")
    if not isinstance(dates, Sequence) or isinstance(dates, (str, bytes)):
        return None
    for raw in dates:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("type") or "").casefold() != "publication":
            continue
        value = raw.get("dateString")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _item_doi(payload: Mapping[str, Any]) -> str:
    identifiers = payload.get("identifiers")
    if not isinstance(identifiers, Sequence) or isinstance(identifiers, (str, bytes)):
        raise ScienceBaseSchemaError("ScienceBase item lacks identifiers.")
    for raw in identifiers:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("type") or "").casefold() != "doi":
            continue
        return normalize_doi(str(raw.get("key") or ""))
    raise ScienceBaseIdentityError("ScienceBase item does not advertise a DOI.")


def _parse_file(raw: Mapping[str, Any]) -> ScienceBaseFile:
    name = str(raw.get("name") or "").strip()
    content_type = str(raw.get("contentType") or "").strip().lower()
    download_url = _require_official_url(
        raw.get("downloadUri") or raw.get("url"), f"file {name!r} download"
    )
    try:
        size = int(raw.get("size"))
    except (TypeError, ValueError) as exc:
        raise ScienceBaseSchemaError(f"File {name!r} lacks a valid size.") from exc
    if not name or size <= 0 or not content_type:
        raise ScienceBaseSchemaError(
            "ScienceBase files require a name, positive size, and content type."
        )
    checksum = raw.get("checksum")
    algorithm = value = None
    if checksum is not None:
        if not isinstance(checksum, Mapping):
            raise ScienceBaseSchemaError(f"File {name!r} checksum is malformed.")
        algorithm = str(checksum.get("type") or "").strip().upper() or None
        value = str(checksum.get("value") or "").strip().lower() or None
        if bool(algorithm) != bool(value):
            raise ScienceBaseSchemaError(f"File {name!r} checksum is incomplete.")
    return ScienceBaseFile(
        name=name,
        content_type=content_type,
        size_bytes=size,
        download_url=download_url,
        checksum_algorithm=algorithm,
        checksum_value=value,
    )


def parse_sciencebase_item(
    payload: Mapping[str, Any],
    *,
    expected_doi: str,
    expected_title: str,
    metadata_url: str | None = None,
) -> ScienceBaseItem:
    """Validate one complete ScienceBase item and its public file inventory."""

    item_id = str(payload.get("id") or "").strip()
    if not _ITEM_ID.fullmatch(item_id):
        raise ScienceBaseIdentityError(f"Invalid ScienceBase item ID {item_id!r}.")
    title = _title(payload.get("title"))
    if title.casefold() != expected_title.strip().casefold():
        raise ScienceBaseIdentityError(
            f"ScienceBase title {title!r} does not match {expected_title!r}."
        )
    doi = _item_doi(payload)
    if doi != normalize_doi(expected_doi):
        raise ScienceBaseIdentityError(
            f"ScienceBase DOI {doi!r} does not match {expected_doi!r}."
        )
    if metadata_url is None:
        metadata_url = SCIENCEBASE_ITEM_URL.format(item_id=item_id)
    metadata_url = _require_official_url(metadata_url, "item metadata")
    if not metadata_url.rstrip("/").endswith(f"/{item_id}"):
        raise ScienceBaseIdentityError(
            "ScienceBase item ID and metadata URL do not agree."
        )
    raw_files = payload.get("files")
    if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes)):
        raise ScienceBaseSchemaError("ScienceBase item lacks a files list.")
    files = tuple(_parse_file(raw) for raw in raw_files if isinstance(raw, Mapping))
    if len(files) != len(raw_files):
        raise ScienceBaseSchemaError("ScienceBase files list contains a non-object.")
    if not files:
        raise ScienceBaseSchemaError("ScienceBase item has no files.")
    summary = payload.get("body") or payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ScienceBaseSchemaError("ScienceBase item lacks a product summary.")
    provenance = payload.get("provenance")
    last_updated = None
    if isinstance(provenance, Mapping):
        value = provenance.get("lastUpdated")
        if isinstance(value, str) and value.strip():
            last_updated = value.strip()
    return ScienceBaseItem(
        item_id=item_id,
        title=title,
        summary=summary.strip(),
        doi=doi,
        citation=(
            str(payload.get("citation")).strip() if payload.get("citation") else None
        ),
        rights=str(payload.get("rights")).strip() if payload.get("rights") else None,
        publication_date=_publication_date(payload),
        item_last_updated=last_updated,
        metadata_url=metadata_url,
        files=files,
    )


def select_eav_file(
    item: ScienceBaseItem,
    *,
    expected_filename: str | None = None,
    max_bytes: int = MAX_REFERENCE_BYTES,
) -> ScienceBaseFile:
    """Select exactly one small documented E-A-V CSV; never choose by order."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")
    documented = item.summary.casefold()
    missing_markers = [
        marker
        for marker in ("bear lake", "elevation-area-volume", "navd88", "0.01-meter")
        if marker not in documented
    ]
    if missing_markers:
        raise ScienceBaseFileSelectionError(
            "ScienceBase metadata does not document the expected E-A-V identity, "
            f"datum, and interval; missing {missing_markers}."
        )
    candidates = []
    for source_file in item.files:
        name_match = (
            source_file.name.casefold() == expected_filename.casefold()
            if expected_filename
            else bool(BEAR_LAKE_FILENAME_PATTERN.fullmatch(source_file.name))
        )
        if (
            name_match
            and source_file.content_type.split(";", 1)[0] == "text/csv"
            and source_file.name.lower().endswith(".csv")
        ):
            candidates.append(source_file)
    if len(candidates) != 1:
        raise ScienceBaseFileSelectionError(
            "Expected exactly one documented Bear Lake elevation-area-volume "
            f"CSV, found {len(candidates)}."
        )
    selected = candidates[0]
    if selected.size_bytes > max_bytes:
        raise ScienceBaseFileSelectionError(
            f"Reference table {selected.name!r} is unexpectedly large "
            f"({selected.size_bytes:,} bytes)."
        )
    return selected


def _numeric(value: str, *, row: int, column: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ScienceBaseContentError(
            f"CSV row {row}, column {column!r} is not numeric."
        ) from exc
    if not math.isfinite(result):
        raise ScienceBaseContentError(
            f"CSV row {row}, column {column!r} is not finite."
        )
    return result


def parse_bear_lake_eav_csv(
    body: bytes,
    *,
    source_file: str,
    source_title: str,
    source_version: str,
    source_doi: str,
    source_sha256: str | None = None,
) -> ParsedBearLakeEav:
    """Parse the official table, ignoring only fully blank trailing records."""

    if not body:
        raise ScienceBaseContentError("Bear Lake E-A-V CSV is empty.")
    digest = hashlib.sha256(body).hexdigest()
    if source_sha256 is not None and digest != source_sha256.casefold():
        raise ScienceBaseContentError("Bear Lake E-A-V SHA-256 does not match.")
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ScienceBaseContentError("Bear Lake E-A-V CSV is not UTF-8.") from exc
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        columns = tuple(part.strip() for part in next(reader))
    except StopIteration as exc:
        raise ScienceBaseContentError("Bear Lake E-A-V CSV has no header.") from exc
    if columns != BEAR_LAKE_COLUMNS:
        raise ScienceBaseContentError(
            f"Unexpected Bear Lake E-A-V columns: {columns!r}."
        )
    inputs: list[CapacityTableInputRow] = []
    blank_count = 0
    physical_count = 0
    for source_row, values in enumerate(reader, start=2):
        physical_count += 1
        stripped = tuple(value.strip() for value in values)
        if not stripped or all(not value for value in stripped):
            blank_count += 1
            continue
        if len(stripped) != len(BEAR_LAKE_COLUMNS):
            raise ScienceBaseContentError(
                f"CSV row {source_row} has {len(stripped)} fields; expected "
                f"{len(BEAR_LAKE_COLUMNS)}."
            )
        missing = [
            column
            for column, value in zip(BEAR_LAKE_COLUMNS, stripped, strict=True)
            if not value
        ]
        if missing:
            raise ScienceBaseContentError(
                f"CSV row {source_row} has missing fields: {missing}."
            )
        numeric = {
            column: _numeric(value, row=source_row, column=column)
            for column, value in zip(BEAR_LAKE_COLUMNS, stripped, strict=True)
        }
        inputs.append(
            CapacityTableInputRow(
                source_row=source_row,
                elevation=numeric["elev_m_NAVD88"],
                area=numeric["area_m2"],
                storage=numeric["volume_m3"],
            )
        )
    if len(inputs) < 2:
        raise ScienceBaseContentError(
            "Bear Lake E-A-V CSV has fewer than two valid rows."
        )
    metadata = CapacityTableMetadata(
        lake_name="Bear Lake",
        source_doi=normalize_doi(source_doi),
        source_title=source_title,
        source_file=source_file,
        source_version=source_version,
        source_hash=digest,
        vertical_datum="NAVD88",
        elevation_unit="m",
        area_unit="m2",
        storage_unit="m3",
        parser_version=PARSER_VERSION,
    )
    table = normalize_capacity_table(inputs, metadata, expected_lake_name="Bear Lake")
    return ParsedBearLakeEav(
        table=table,
        valid_row_count=len(inputs),
        fully_blank_row_count=blank_count,
        physical_record_count=physical_count,
        columns=columns,
    )


def _safe_zip_name(name: str) -> PurePosixPath:
    if "\\" in name:
        raise ScienceBaseContentError(f"Unsafe ZIP member path {name!r}.")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or any(":" in part for part in path.parts)
    ):
        raise ScienceBaseContentError(f"Unsafe ZIP member path {name!r}.")
    return path


def safe_extract_expected_zip(
    archive: bytes,
    *,
    output_dir: str | Path,
    expected_filenames: Sequence[str],
    max_member_bytes: int = MAX_ZIP_MEMBER_BYTES,
) -> tuple[Path, ...]:
    """Extract only explicitly named small ZIP members without path traversal."""

    expected = {str(name) for name in expected_filenames}
    if not expected or any(Path(name).name != name for name in expected):
        raise ValueError("expected_filenames must contain plain basenames.")
    if max_member_bytes <= 0:
        raise ValueError("max_member_bytes must be positive.")
    try:
        bundle = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as exc:
        raise ScienceBaseContentError("Reference archive is not a valid ZIP.") from exc
    selected: dict[str, zipfile.ZipInfo] = {}
    with bundle:
        for member in bundle.infolist():
            path = _safe_zip_name(member.filename)
            if member.is_dir() or path.name not in expected:
                continue
            if path.name in selected:
                raise ScienceBaseContentError(
                    f"ZIP contains duplicate expected file {path.name!r}."
                )
            if member.file_size > max_member_bytes:
                raise ScienceBaseContentError(
                    f"ZIP member {path.name!r} is unexpectedly large."
                )
            selected[path.name] = member
        missing = expected.difference(selected)
        if missing:
            raise ScienceBaseContentError(
                f"ZIP lacks expected reference files: {sorted(missing)}."
            )
        destination = Path(output_dir)
        outputs: list[Path] = []
        for name in sorted(expected):
            data = bundle.read(selected[name])
            if len(data) != selected[name].file_size:
                raise ScienceBaseContentError(
                    f"ZIP member {name!r} length does not match metadata."
                )
            target = destination / name
            _atomic_write(target, data)
            outputs.append(target)
    return tuple(outputs)


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reference_path(reference_dir: Path, relative_path: str, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ScienceBaseContentError(f"Reference manifest {label} is missing.")
    root = reference_dir.resolve()
    candidate = (reference_dir / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ScienceBaseContentError(
            f"Reference manifest {label} escapes the reference directory."
        ) from exc
    return candidate


def _manifest_from_json(body: bytes) -> BearLakeEavManifest:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScienceBaseContentError(
            "Bear Lake reference manifest is invalid JSON."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ScienceBaseContentError("Bear Lake reference manifest is not an object.")
    normalized = dict(payload)
    for field in (
        "source_elevation_units",
        "source_area_units",
        "source_storage_units",
    ):
        value = normalized.get(field)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ScienceBaseContentError(
                f"Bear Lake reference manifest field {field!r} is invalid."
            )
        normalized[field] = tuple(str(part) for part in value)
    try:
        manifest = BearLakeEavManifest(**normalized)
    except TypeError as exc:
        raise ScienceBaseContentError(
            f"Bear Lake reference manifest schema mismatch: {exc}"
        ) from exc
    if manifest.schema_version != REFERENCE_MANIFEST_VERSION:
        raise ScienceBaseContentError(
            f"Unsupported reference manifest {manifest.schema_version!r}."
        )
    return manifest


def _load_bear_lake_reference(
    *,
    reference_dir: Path,
    doi: str,
    expected_title: str,
    expected_filename: str | None,
) -> BearLakeEavSource:
    manifest_path = reference_dir / "bear_lake" / "bear_lake_eav_manifest.json"
    if not manifest_path.is_file():
        raise CacheMissError(f"No stable Bear Lake reference at {manifest_path}")
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ScienceBaseContentError(
            f"Cannot read Bear Lake reference manifest: {exc}"
        ) from exc
    manifest = _manifest_from_json(manifest_bytes)
    if normalize_doi(manifest.doi) != normalize_doi(doi):
        raise ScienceBaseIdentityError("Cached Bear Lake reference DOI does not match.")
    if manifest.provider_title.casefold() != expected_title.strip().casefold():
        raise ScienceBaseIdentityError(
            "Cached Bear Lake reference title does not match."
        )
    source_path = _reference_path(reference_dir, manifest.source_path, "source_path")
    metadata_path = _reference_path(
        reference_dir, manifest.metadata_path, "metadata_path"
    )
    try:
        source_body = source_path.read_bytes()
        metadata_body = metadata_path.read_bytes()
    except OSError as exc:
        raise ScienceBaseContentError(
            f"Cannot read cached Bear Lake reference: {exc}"
        ) from exc
    if hashlib.sha256(source_body).hexdigest() != manifest.sha256:
        raise ScienceBaseContentError("Cached Bear Lake E-A-V SHA-256 mismatch.")
    if len(source_body) != manifest.content_length:
        raise ScienceBaseContentError("Cached Bear Lake E-A-V length mismatch.")
    if hashlib.sha256(metadata_body).hexdigest() != manifest.metadata_sha256:
        raise ScienceBaseContentError("Cached ScienceBase item SHA-256 mismatch.")
    try:
        item_payload = json.loads(metadata_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScienceBaseContentError(
            "Cached ScienceBase item JSON is invalid."
        ) from exc
    if not isinstance(item_payload, Mapping):
        raise ScienceBaseContentError("Cached ScienceBase item is not an object.")
    item = parse_sciencebase_item(
        item_payload,
        expected_doi=doi,
        expected_title=expected_title,
        metadata_url=manifest.resolved_metadata_url.split("?", 1)[0],
    )
    if item.item_id != manifest.sciencebase_item_id:
        raise ScienceBaseIdentityError(
            "Cached ScienceBase item and reference manifest IDs disagree."
        )
    source_file = select_eav_file(item, expected_filename=expected_filename)
    if source_file.name != manifest.source_filename:
        raise ScienceBaseIdentityError(
            "Cached ScienceBase file and reference manifest names disagree."
        )
    _validate_source_bytes(source_file, source_body)
    parsed = parse_bear_lake_eav_csv(
        source_body,
        source_file=source_file.name,
        source_title=item.title,
        source_version=manifest.source_release_version,
        source_sha256=manifest.sha256,
        source_doi=item.doi,
    )
    if (
        parsed.valid_row_count != manifest.source_row_count
        or parsed.fully_blank_row_count != manifest.ignored_fully_blank_row_count
    ):
        raise ScienceBaseContentError(
            "Cached Bear Lake E-A-V row counts disagree with the manifest."
        )
    return BearLakeEavSource(
        table=parsed.table,
        item=item,
        source_file=source_file,
        manifest=manifest,
        source_path=source_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        cache_hit=True,
    )


def _validate_source_bytes(
    source_file: ScienceBaseFile, body: bytes
) -> tuple[str, str]:
    if len(body) != source_file.size_bytes:
        raise ScienceBaseContentError(
            f"Downloaded {len(body):,} bytes for {source_file.name!r}; "
            f"ScienceBase advertises {source_file.size_bytes:,}."
        )
    sha256 = hashlib.sha256(body).hexdigest()
    if source_file.checksum_algorithm:
        algorithm = source_file.checksum_algorithm.casefold().replace("-", "")
        try:
            calculated = hashlib.new(algorithm, body).hexdigest()
        except ValueError as exc:
            raise ScienceBaseContentError(
                f"Unsupported ScienceBase checksum {source_file.checksum_algorithm!r}."
            ) from exc
        if calculated.casefold() != (source_file.checksum_value or "").casefold():
            raise ScienceBaseContentError(
                f"ScienceBase {source_file.checksum_algorithm} checksum mismatch "
                f"for {source_file.name!r}."
            )
    return sha256, source_file.checksum_value or ""


def acquire_bear_lake_eav(
    *,
    client: HttpClient,
    mode: CacheMode,
    reference_dir: str | Path,
    doi: str,
    expected_title: str = BEAR_LAKE_TITLE,
    expected_filename: str | None = None,
    refresh: bool = False,
) -> BearLakeEavSource:
    """Acquire, validate, and retain only Bear Lake's official E-A-V CSV.

    A content-addressed source and item JSON are written before a stable
    manifest is atomically replaced.  Consequently, a failed refresh cannot
    invalidate or overwrite the previous valid reference.
    """

    if not isinstance(mode, CacheMode):
        raise TypeError("mode must be CacheMode.LIVE or CacheMode.OFFLINE.")
    if mode is CacheMode.OFFLINE and refresh:
        raise ValueError("refresh cannot be used in offline mode.")
    normalized_doi = normalize_doi(doi)
    reference_root = Path(reference_dir)
    root = reference_root / "bear_lake"
    if mode is CacheMode.OFFLINE or (
        not refresh and (root / "bear_lake_eav_manifest.json").is_file()
    ):
        return _load_bear_lake_reference(
            reference_dir=reference_root,
            doi=normalized_doi,
            expected_title=expected_title,
            expected_filename=expected_filename,
        )
    cache = CacheStore(root / ".http_cache")
    search_params: dict[str, Any] = {
        "filter": "itemIdentifier="
        + json.dumps(
            {"type": "DOI", "key": f"doi:{normalized_doi}"},
            separators=(",", ":"),
        ),
        "format": "json",
        "max": 5,
    }
    search_response = cache.request(
        provider="usgs_sciencebase",
        client=client,
        url=SCIENCEBASE_ITEMS_URL,
        params=search_params,
        mode=mode,
        refresh=refresh,
        expected_content_types=("application/json",),
        response_schema="sciencebase-doi-search-v1",
    )
    search_payload = search_response.json()
    if not isinstance(search_payload, Mapping):
        raise ScienceBaseSchemaError("ScienceBase DOI response is not an object.")
    item_id, metadata_url = parse_sciencebase_search(
        search_payload, expected_title=expected_title
    )
    item_response = cache.request(
        provider="usgs_sciencebase",
        client=client,
        url=metadata_url,
        params={"format": "json"},
        mode=mode,
        refresh=refresh,
        expected_content_types=("application/json",),
        response_schema="sciencebase-item-v1",
    )
    item_payload = item_response.json()
    if not isinstance(item_payload, Mapping):
        raise ScienceBaseSchemaError("ScienceBase item response is not an object.")
    item = parse_sciencebase_item(
        item_payload,
        expected_doi=normalized_doi,
        expected_title=expected_title,
        metadata_url=metadata_url,
    )
    if item.item_id != item_id:
        raise ScienceBaseIdentityError("DOI search and item response IDs disagree.")
    source_file = select_eav_file(item, expected_filename=expected_filename)
    source_response = client.get(
        url=source_file.download_url,
        expected_content_types=("text/csv",),
    )
    media_type = source_response.content_type.split(";", 1)[0].strip().lower()
    if media_type != "text/csv":
        raise ScienceBaseContentError(
            f"Bear Lake E-A-V response is {source_response.content_type!r}, not CSV."
        )
    sha256, _provider_hash = _validate_source_bytes(source_file, source_response.body)
    parsed = parse_bear_lake_eav_csv(
        source_response.body,
        source_file=source_file.name,
        source_title=item.title,
        source_version=item.source_version,
        source_sha256=sha256,
        source_doi=item.doi,
    )

    item_bytes = (
        json.dumps(item_payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    item_hash = hashlib.sha256(item_bytes).hexdigest()
    source_path = root / "sources" / sha256 / source_file.name
    metadata_path = root / "metadata" / f"{item.item_id}_{item_hash}.json"
    manifest_path = root / "bear_lake_eav_manifest.json"
    _atomic_write(source_path, source_response.body)
    _atomic_write(metadata_path, item_bytes)
    manifest = BearLakeEavManifest(
        schema_version=REFERENCE_MANIFEST_VERSION,
        doi=item.doi,
        landing_url=f"https://doi.org/{item.doi}",
        sciencebase_item_id=item.item_id,
        resolved_metadata_url=item.metadata_url + "?format=json",
        download_url=source_file.download_url,
        source_filename=source_file.name,
        provider_title=item.title,
        retrieved_at_utc=(
            source_response.retrieved_at_utc.astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z")
        ),
        content_type=source_response.content_type,
        content_length=len(source_response.body),
        sha256=sha256,
        provider_checksum_algorithm=source_file.checksum_algorithm,
        provider_checksum_value=source_file.checksum_value,
        source_release_version=item.source_version,
        item_last_updated=item.item_last_updated,
        license=item.rights,
        vertical_datum="NAVD88 (EPSG:5703)",
        geoid_model="GEOID12B",
        source_elevation_units=("m NAVD88", "ft NAVD88"),
        source_area_units=("ft2", "m2", "km2", "mi2", "acre"),
        source_storage_units=("ft3", "m3", "km3", "mi3", "acre-ft"),
        source_row_count=parsed.valid_row_count,
        ignored_fully_blank_row_count=parsed.fully_blank_row_count,
        source_path=source_path.relative_to(reference_root).as_posix(),
        metadata_path=metadata_path.relative_to(reference_root).as_posix(),
        metadata_sha256=item_hash,
        parser_version=PARSER_VERSION,
    )
    _atomic_write(
        manifest_path,
        (json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n").encode(),
    )
    return BearLakeEavSource(
        table=parsed.table,
        item=item,
        source_file=source_file,
        manifest=manifest,
        source_path=source_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        cache_hit=False,
    )


__all__ = [
    "BEAR_LAKE_COLUMNS",
    "BEAR_LAKE_TITLE",
    "BearLakeEavManifest",
    "BearLakeEavSource",
    "DoiMetadata",
    "ParsedBearLakeEav",
    "ScienceBaseContentError",
    "ScienceBaseError",
    "ScienceBaseFile",
    "ScienceBaseFileSelectionError",
    "ScienceBaseIdentityError",
    "ScienceBaseItem",
    "ScienceBaseSchemaError",
    "acquire_bear_lake_eav",
    "normalize_doi",
    "parse_bear_lake_eav_csv",
    "parse_doi_metadata",
    "parse_sciencebase_item",
    "parse_sciencebase_search",
    "safe_extract_expected_zip",
    "select_eav_file",
]
