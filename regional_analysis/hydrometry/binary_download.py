"""Bounded streaming acquisition and safe ZIP handling for Phase 2 sources.

The Phase 0/1 :mod:`regional_analysis.hydrometry.http` client intentionally
materializes small responses in memory.  This module is the separate binary
path for large official source files.  It streams to a sibling ``.part`` file,
validates the completed representation, and only then atomically replaces the
destination.  A failed refresh therefore cannot overwrite a previously
validated archive.

ZIP inspection uses only the central directory.  Selected members are copied
in chunks and no function in this module reads an entire archive or extracted
member into memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import urlsplit

import requests

from .http import (
    DEFAULT_USER_AGENT,
    TRANSIENT_STATUS_CODES,
    public_url,
)
from .sciencebase import ScienceBaseFile, ScienceBaseItem

BINARY_MANIFEST_SCHEMA = "hydrometry-binary-download-v1"
EXTRACTION_MANIFEST_SCHEMA = "hydrometry-binary-extraction-v1"
DEFAULT_BINARY_CONTENT_TYPES = (
    "application/octet-stream",
    "application/zip",
    "application/x-zip-compressed",
    "application/x-zip",
)
DEFAULT_RASTER_SUFFIXES = (".tif", ".tiff", ".img")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE = re.compile(r"^[a-zA-Z]:[/\\]")
_PUBLIC_RESPONSE_HEADERS = frozenset(
    {
        "accept-ranges",
        "content-length",
        "content-range",
        "content-type",
        "etag",
        "last-modified",
    }
)


class BinaryDownloadError(RuntimeError):
    """Base error for bounded binary acquisition."""


class BinaryHttpStatusError(BinaryDownloadError):
    """Raised for a non-success response after bounded status retries."""

    def __init__(self, status_code: int, url_public: str) -> None:
        self.status_code = int(status_code)
        self.url_public = url_public
        super().__init__(f"HTTP {status_code} from {url_public}")


class BinaryTransportError(BinaryDownloadError):
    """Raised when a streaming request or response body cannot be read."""


class BinaryContentError(BinaryDownloadError):
    """Raised when size, type, range, or digest validation fails."""


class BinaryCacheError(BinaryDownloadError):
    """Raised when a retained binary or its manifest fails validation."""


class DownloadCancelledError(BinaryDownloadError):
    """Raised after a caller cancellation request, leaving resumable bytes."""


class ScienceBaseBinarySelectionError(BinaryDownloadError):
    """Raised when an advertised official file cannot be selected uniquely."""


class ZipInventoryError(BinaryDownloadError):
    """Raised for an invalid, ambiguous, or unsafe ZIP inventory."""


class InsufficientDiskSpaceError(BinaryDownloadError):
    """Raised when the explicit extraction disk-space gate fails."""


@dataclass(frozen=True, slots=True)
class BinaryDownloadRequest:
    """Validated transport, stable public identity, and binary destination.

    ``url`` may be a short-lived provider-signed transport URL.  Callers using
    such URLs must set ``identity_url`` to the stable, public provider record
    that identifies the same file.  Only that sanitized identity is used in
    cache sidecars, manifests, and controlled error messages.
    """

    provider: str
    url: str = field(repr=False)
    destination: Path
    identity_url: str | None = None
    expected_size_bytes: int | None = None
    expected_sha256: str | None = None
    expected_content_types: tuple[str, ...] = DEFAULT_BINARY_CONTENT_TYPES
    require_zip_integrity: bool | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if not self.url.strip():
            raise ValueError("url must be non-empty")
        parsed_url = urlsplit(self.url)
        if (
            parsed_url.scheme.casefold() not in {"http", "https"}
            or not parsed_url.hostname
        ):
            raise ValueError("url must be an absolute HTTP(S) URL")
        if parsed_url.username is not None or parsed_url.password is not None:
            raise ValueError("credentials must not be embedded in a binary URL")
        if self.identity_url is not None:
            identity = self.identity_url.strip()
            if not identity:
                raise ValueError("identity_url must be non-empty when provided")
            parsed_identity = urlsplit(identity)
            if (
                parsed_identity.scheme.casefold() not in {"http", "https"}
                or not parsed_identity.hostname
            ):
                raise ValueError("identity_url must be an absolute HTTP(S) URL")
            if (
                parsed_identity.username is not None
                or parsed_identity.password is not None
            ):
                raise ValueError("credentials must not be embedded in identity_url")
            object.__setattr__(self, "identity_url", identity)
        object.__setattr__(self, "destination", Path(self.destination))
        if self.expected_size_bytes is not None and self.expected_size_bytes <= 0:
            raise ValueError("expected_size_bytes must be positive when provided")
        if self.expected_sha256 is not None:
            digest = self.expected_sha256.strip().lower()
            if not _SHA256.fullmatch(digest):
                raise ValueError("expected_sha256 must contain 64 hexadecimal digits")
            object.__setattr__(self, "expected_sha256", digest)
        normalized_types = tuple(
            _media_type(value) for value in self.expected_content_types if value.strip()
        )
        if not normalized_types:
            raise ValueError("expected_content_types must be non-empty")
        object.__setattr__(self, "expected_content_types", normalized_types)
        require_zip_integrity = self.require_zip_integrity
        if require_zip_integrity is None:
            require_zip_integrity = self.destination.suffix.casefold() == ".zip"
        elif not isinstance(require_zip_integrity, bool):
            raise ValueError("require_zip_integrity must be boolean when provided")
        object.__setattr__(self, "require_zip_integrity", require_zip_integrity)

    @property
    def public_url(self) -> str:
        """Return the stable identity with secret query parameters removed."""

        return public_url(self.identity_url or self.url)

    @property
    def part_path(self) -> Path:
        """Return the stable sibling partial-download path."""

        return self.destination.with_name(f"{self.destination.name}.part")

    @property
    def part_manifest_path(self) -> Path:
        """Return the validator sidecar used for bounded resume."""

        return self.destination.with_name(f"{self.destination.name}.part.json")

    @property
    def manifest_path(self) -> Path:
        """Return the finalized public-provenance sidecar path."""

        return self.destination.with_name(f"{self.destination.name}.manifest.json")

    @property
    def previous_path(self) -> Path:
        """Return the stable rollback path for a prior finalized binary."""

        return self.destination.with_name(f".{self.destination.name}.previous")

    @property
    def previous_manifest_path(self) -> Path:
        """Return the stable rollback path for a prior finalized manifest."""

        return self.manifest_path.with_name(f".{self.manifest_path.name}.previous")

    @property
    def transaction_path(self) -> Path:
        """Return the crash-recovery journal for a two-file finalization."""

        return self.destination.with_name(f".{self.destination.name}.transaction.json")


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    """One deterministic progress update from a streaming transfer."""

    downloaded_bytes: int
    expected_bytes: int | None
    resumed_from_bytes: int
    attempt: int
    complete: bool = False


@dataclass(frozen=True, slots=True)
class BinaryDownloadManifest:
    """Serializable, secret-free provenance for one validated binary."""

    schema_version: str
    provider: str
    public_url: str
    retrieved_at_utc: str
    http_status: int
    content_type: str
    content_length: int
    sha256: str
    etag: str | None
    last_modified: str | None
    accept_ranges: str | None
    response_headers: Mapping[str, str]
    destination_path: str
    resumed_from_bytes: int
    zip_crc_validated: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a detached JSON-safe representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BinaryDownloadResult:
    """A validated final path and its public provenance."""

    path: Path
    manifest_path: Path
    manifest: BinaryDownloadManifest
    cache_hit: bool
    bytes_transferred: int


@dataclass(frozen=True, slots=True)
class BearLakeMetadataFiles:
    """Identity-selected Phase 2 files from the parsed ScienceBase item."""

    raster_archive: ScienceBaseFile
    spatial_metadata_archive: ScienceBaseFile
    fgdc_xml: ScienceBaseFile
    thumbnail: ScienceBaseFile


@dataclass(frozen=True, slots=True)
class ZipMember:
    """One central-directory record without extracted content."""

    name: str
    compressed_size_bytes: int
    uncompressed_size_bytes: int
    crc32: int
    compression_method: int
    is_directory: bool
    is_encrypted: bool
    is_symlink: bool
    is_special: bool

    @property
    def suffix(self) -> str:
        """Return the lowercase final suffix of the normalized member name."""

        return PurePosixPath(self.name.replace("\\", "/")).suffix.lower()


@dataclass(frozen=True, slots=True)
class ZipInventory:
    """Central-directory inventory and exact byte totals."""

    archive_path: Path
    archive_size_bytes: int
    archive_sha256: str
    members: tuple[ZipMember, ...]
    total_compressed_member_bytes: int
    total_uncompressed_bytes: int
    raster_candidates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiskRequirements:
    """Explicit disk budget calculated after central-directory inspection."""

    archive_bytes: int
    selected_extracted_bytes: int
    temporary_bytes: int
    quantitative_derivative_bytes: int
    rendering_output_bytes: int
    safety_margin_bytes: int
    projected_total_bytes: int
    required_additional_free_bytes: int
    available_free_bytes: int
    sufficient: bool


@dataclass(frozen=True, slots=True)
class ExtractedMember:
    """Hash and path of one safely extracted selected member."""

    archive_member: str
    output_path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ExtractionManifest:
    """Source archive and selective-extraction provenance."""

    schema_version: str
    created_at_utc: str
    archive_path: str
    archive_size_bytes: int
    archive_sha256: str
    total_uncompressed_bytes: int
    raster_candidates: tuple[str, ...]
    extracted_members: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a detached JSON-safe representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class _PartialState:
    public_url: str
    expected_size_bytes: int
    etag: str | None
    last_modified: str | None
    downloaded_bytes: int


def _media_type(value: str) -> str:
    return value.split(";", 1)[0].strip().lower()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _hash_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _public_headers(headers: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in headers.items()
        if str(key).strip().lower() in _PUBLIC_RESPONSE_HEADERS
    }


def _header(headers: Mapping[str, Any], name: str) -> str | None:
    wanted = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == wanted:
            text = str(value).strip()
            return text or None
    return None


def _parse_positive_header(headers: Mapping[str, Any], name: str) -> int | None:
    value = _header(headers, name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise BinaryContentError(f"Invalid {name} header {value!r}.") from exc
    if parsed < 0:
        raise BinaryContentError(f"Invalid {name} header {value!r}.")
    return parsed


def _parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+)", value.strip())
    if not match:
        raise BinaryContentError(f"Invalid Content-Range header {value!r}.")
    start, end, total = (int(part) for part in match.groups())
    if start > end or end >= total:
        raise BinaryContentError(f"Invalid Content-Range header {value!r}.")
    return start, end, total


def _strong_resume_validator(etag: str | None, modified: str | None) -> str | None:
    if etag and not etag.strip().casefold().startswith("w/"):
        return etag.strip()
    return modified.strip() if modified else None


def _cancelled(callback: Callable[[], bool] | None) -> bool:
    return bool(callback is not None and callback())


def _load_partial_state(request: BinaryDownloadRequest) -> _PartialState | None:
    if not request.part_path.is_file() or not request.part_manifest_path.is_file():
        return None
    try:
        raw = json.loads(request.part_manifest_path.read_text(encoding="utf-8"))
        state = _PartialState(
            public_url=str(raw["public_url"]),
            expected_size_bytes=int(raw["expected_size_bytes"]),
            etag=str(raw["etag"]) if raw.get("etag") else None,
            last_modified=(
                str(raw["last_modified"]) if raw.get("last_modified") else None
            ),
            downloaded_bytes=int(raw["downloaded_bytes"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    actual_size = request.part_path.stat().st_size
    if (
        state.public_url != request.public_url
        or state.downloaded_bytes != actual_size
        or actual_size <= 0
        or actual_size >= state.expected_size_bytes
        or (
            request.expected_size_bytes is not None
            and state.expected_size_bytes != request.expected_size_bytes
        )
        or _strong_resume_validator(state.etag, state.last_modified) is None
    ):
        return None
    return state


def _write_partial_state(
    request: BinaryDownloadRequest,
    *,
    expected_size_bytes: int,
    etag: str | None,
    last_modified: str | None,
    downloaded_bytes: int,
) -> None:
    _atomic_json(
        request.part_manifest_path,
        {
            "schema_version": "hydrometry-binary-part-v1",
            "public_url": request.public_url,
            "expected_size_bytes": expected_size_bytes,
            "etag": etag,
            "last_modified": last_modified,
            "downloaded_bytes": downloaded_bytes,
        },
    )


def _load_manifest(path: Path) -> BinaryDownloadManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return BinaryDownloadManifest(**payload)
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise BinaryCacheError(f"Invalid binary manifest {path}: {exc}") from exc


def validate_zip_integrity(
    path: str | Path, *, chunk_size: int = 4 * 1024 * 1024
) -> None:
    """Stream every ZIP member and require its recorded CRC to validate.

    Reading a member through :mod:`zipfile` checks its CRC when EOF is reached.
    The loop deliberately discards bounded chunks, so even the multi-gigabyte
    topobathymetry member is never materialized in memory.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    archive = Path(path)
    try:
        with zipfile.ZipFile(archive, "r", allowZip64=True) as bundle:
            for info in bundle.infolist():
                if info.is_dir():
                    continue
                if info.flag_bits & 0x1:
                    raise BinaryContentError(
                        f"Encrypted ZIP member is unsupported: {info.filename!r}"
                    )
                with bundle.open(info, "r") as stream:
                    while stream.read(chunk_size):
                        pass
    except BinaryContentError:
        raise
    except (OSError, RuntimeError, NotImplementedError, zipfile.BadZipFile) as exc:
        raise BinaryContentError(
            f"ZIP member integrity validation failed for {archive}: {exc}"
        ) from exc


def _recover_binary_transaction(request: BinaryDownloadRequest) -> None:
    """Recover an interrupted archive/manifest two-file transaction."""

    journal = request.transaction_path
    if not journal.is_file():
        return
    try:
        transaction = json.loads(journal.read_text(encoding="utf-8"))
        state = str(transaction["state"])
        had_prior = transaction["had_prior"]
        if state not in {"preparing", "committed"} or not isinstance(had_prior, bool):
            raise ValueError("invalid transaction state")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BinaryCacheError(
            f"Invalid binary finalization journal {journal}: {exc}"
        ) from exc

    try:
        final_pair_exists = (
            request.destination.is_file() and request.manifest_path.is_file()
        )
        if state == "committed" and final_pair_exists:
            request.previous_path.unlink(missing_ok=True)
            request.previous_manifest_path.unlink(missing_ok=True)
            journal.unlink(missing_ok=True)
            return

        if had_prior:
            if request.previous_path.is_file():
                os.replace(request.previous_path, request.destination)
            elif not request.destination.is_file():
                raise BinaryCacheError(
                    "Interrupted binary transaction lost the prior archive."
                )
            if request.previous_manifest_path.is_file():
                os.replace(request.previous_manifest_path, request.manifest_path)
            elif not request.manifest_path.is_file():
                raise BinaryCacheError(
                    "Interrupted binary transaction lost the prior manifest."
                )
        else:
            request.destination.unlink(missing_ok=True)
            request.manifest_path.unlink(missing_ok=True)
        journal.unlink(missing_ok=True)
    except OSError as exc:
        raise BinaryCacheError(
            f"Could not recover interrupted binary finalization: {exc}"
        ) from exc


def _finalize_binary_pair(
    request: BinaryDownloadRequest, pending_manifest: Path
) -> None:
    """Install a binary and manifest together with rollback and crash recovery."""

    _recover_binary_transaction(request)
    destination_exists = request.destination.is_file()
    manifest_exists = request.manifest_path.is_file()
    if destination_exists != manifest_exists:
        raise BinaryCacheError("Cannot refresh an incomplete prior binary cache pair.")
    had_prior = destination_exists and manifest_exists
    if request.previous_path.exists() or request.previous_manifest_path.exists():
        raise BinaryCacheError("Stale binary rollback files require manual review.")

    try:
        _atomic_json(
            request.transaction_path,
            {
                "schema_version": "hydrometry-binary-transaction-v1",
                "state": "preparing",
                "had_prior": had_prior,
            },
        )
        if had_prior:
            os.replace(request.destination, request.previous_path)
            os.replace(request.manifest_path, request.previous_manifest_path)
        os.replace(request.part_path, request.destination)
        os.replace(pending_manifest, request.manifest_path)
        _atomic_json(
            request.transaction_path,
            {
                "schema_version": "hydrometry-binary-transaction-v1",
                "state": "committed",
                "had_prior": had_prior,
            },
        )
    except (OSError, BinaryCacheError) as exc:
        try:
            _recover_binary_transaction(request)
        except BinaryCacheError as recovery_error:
            raise BinaryCacheError(
                "Binary finalization failed and automatic rollback also failed: "
                f"{recovery_error}"
            ) from exc
        raise BinaryCacheError(
            f"Binary finalization failed; the prior valid cache was restored: {exc}"
        ) from exc

    # A committed journal makes cleanup restart-safe. If cleanup is interrupted,
    # the next cache access keeps the complete new pair and removes leftovers.
    try:
        request.previous_path.unlink(missing_ok=True)
        request.previous_manifest_path.unlink(missing_ok=True)
        request.transaction_path.unlink(missing_ok=True)
    except OSError:
        pass


class StreamingBinaryDownloader:
    """Synchronous, bounded, status-retrying streaming downloader."""

    def __init__(
        self,
        *,
        connect_timeout: float = 15.0,
        read_timeout: float = 120.0,
        chunk_size: int = 4 * 1024 * 1024,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5,
        max_backoff_seconds: float = 8.0,
        user_agent: str = DEFAULT_USER_AGENT,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if connect_timeout <= 0 or read_timeout <= 0:
            raise ValueError("connect_timeout and read_timeout must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if backoff_base_seconds < 0 or max_backoff_seconds < 0:
            raise ValueError("backoff durations cannot be negative")
        if not user_agent.strip():
            raise ValueError("user_agent must be non-empty")
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
        self.chunk_size = int(chunk_size)
        self.max_retries = int(max_retries)
        self.backoff_base_seconds = float(backoff_base_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self.user_agent = user_agent.strip()
        self.session = session or requests.Session()
        self._sleep = sleep
        self.request_count = 0

    def load_cached(self, request: BinaryDownloadRequest) -> BinaryDownloadResult:
        """Validate and return a finalized binary without a network request."""

        _recover_binary_transaction(request)
        if not request.destination.is_file() or not request.manifest_path.is_file():
            raise BinaryCacheError(
                f"No validated binary cache at {request.destination}"
            )
        manifest = _load_manifest(request.manifest_path)
        size = request.destination.stat().st_size
        digest = _hash_file(request.destination, chunk_size=self.chunk_size)
        if manifest.schema_version != BINARY_MANIFEST_SCHEMA:
            raise BinaryCacheError("Unsupported binary manifest schema.")
        if (
            manifest.provider != request.provider
            or manifest.public_url != request.public_url
        ):
            raise BinaryCacheError("Binary cache request identity does not match.")
        if manifest.destination_path != request.destination.name:
            raise BinaryCacheError("Binary cache destination identity does not match.")
        if _media_type(manifest.content_type) not in request.expected_content_types:
            raise BinaryCacheError("Binary cache content type is no longer accepted.")
        if size != manifest.content_length or digest != manifest.sha256:
            raise BinaryCacheError("Binary cache size or SHA-256 does not match.")
        if (
            request.expected_size_bytes is not None
            and size != request.expected_size_bytes
        ):
            raise BinaryCacheError("Binary cache does not match the expected size.")
        if request.expected_sha256 is not None and digest != request.expected_sha256:
            raise BinaryCacheError("Binary cache does not match the expected SHA-256.")
        if request.require_zip_integrity and not manifest.zip_crc_validated:
            try:
                validate_zip_integrity(request.destination, chunk_size=self.chunk_size)
            except BinaryContentError as exc:
                raise BinaryCacheError(
                    f"Binary cache ZIP integrity validation failed: {exc}"
                ) from exc
            manifest = replace(manifest, zip_crc_validated=True)
            _atomic_json(request.manifest_path, manifest.as_dict())
        return BinaryDownloadResult(
            path=request.destination,
            manifest_path=request.manifest_path,
            manifest=manifest,
            cache_hit=True,
            bytes_transferred=0,
        )

    def download(
        self,
        request: BinaryDownloadRequest,
        *,
        refresh: bool = False,
        allow_resume: bool = True,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[DownloadProgress], None] | None = None,
    ) -> BinaryDownloadResult:
        """Stream and validate one response, preserving prior valid content.

        Transport failures are deliberately not retried.  Only HTTP 429 and
        transient 5xx statuses enter the bounded retry loop.  A partial body is
        retained with public validators so a later invocation can use Range
        only when identity can be checked safely.
        """

        if not isinstance(request, BinaryDownloadRequest):
            raise TypeError("request must be a BinaryDownloadRequest")
        _recover_binary_transaction(request)
        if not refresh and request.destination.exists():
            return self.load_cached(request)
        if refresh and (request.destination.exists() or request.manifest_path.exists()):
            # Establish that the pair we promise to preserve is valid before
            # starting the replacement transfer.
            self.load_cached(request)
        if _cancelled(cancel_check):
            raise DownloadCancelledError("Download cancelled before request.")
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        partial = _load_partial_state(request) if allow_resume else None
        resume_offset = partial.downloaded_bytes if partial else 0
        resume_validator = (
            _strong_resume_validator(partial.etag, partial.last_modified)
            if partial
            else None
        )
        for attempt in range(self.max_retries + 1):
            headers = {
                "User-Agent": self.user_agent,
                "Accept": ", ".join(request.expected_content_types),
                "Accept-Encoding": "identity",
            }
            if resume_offset and resume_validator:
                headers["Range"] = f"bytes={resume_offset}-"
                headers["If-Range"] = resume_validator
            self.request_count += 1
            try:
                response = self.session.get(
                    request.url,
                    headers=headers,
                    timeout=(self.connect_timeout, self.read_timeout),
                    stream=True,
                )
            except requests.RequestException as exc:
                raise BinaryTransportError(
                    f"Request failed for {request.public_url} ({type(exc).__name__})."
                ) from None

            status = int(response.status_code)
            if status in TRANSIENT_STATUS_CODES and attempt < self.max_retries:
                response.close()
                delay = min(
                    self.max_backoff_seconds,
                    self.backoff_base_seconds * (2**attempt),
                )
                retry_after = _header(response.headers, "Retry-After")
                if retry_after:
                    try:
                        delay = min(
                            self.max_backoff_seconds,
                            max(delay, float(retry_after)),
                        )
                    except ValueError:
                        pass
                self._sleep(delay)
                continue
            if not 200 <= status < 300:
                response.close()
                raise BinaryHttpStatusError(status, request.public_url)

            if resume_offset and status == 206:
                response_etag = _header(response.headers, "ETag")
                response_modified = _header(response.headers, "Last-Modified")
                returned_range = _parse_content_range(
                    _header(response.headers, "Content-Range")
                )
                uses_etag = partial is not None and resume_validator == partial.etag
                validator_compatible = bool(
                    partial is not None
                    and returned_range is not None
                    and returned_range[2] == partial.expected_size_bytes
                    and (
                        response_etag == partial.etag
                        if uses_etag
                        else response_modified == partial.last_modified
                    )
                )
                if not validator_compatible:
                    response.close()
                    request.part_path.unlink(missing_ok=True)
                    request.part_manifest_path.unlink(missing_ok=True)
                    return self.download(
                        request,
                        refresh=refresh,
                        allow_resume=False,
                        cancel_check=cancel_check,
                        progress_callback=progress_callback,
                    )

            try:
                return self._consume_response(
                    request,
                    response=response,
                    attempt=attempt + 1,
                    requested_resume_offset=resume_offset,
                    cancel_check=cancel_check,
                    progress_callback=progress_callback,
                )
            finally:
                response.close()
        raise AssertionError("binary retry loop terminated unexpectedly")

    def _consume_response(
        self,
        request: BinaryDownloadRequest,
        *,
        response: requests.Response,
        attempt: int,
        requested_resume_offset: int,
        cancel_check: Callable[[], bool] | None,
        progress_callback: Callable[[DownloadProgress], None] | None,
    ) -> BinaryDownloadResult:
        status = int(response.status_code)
        headers = response.headers
        content_type = str(_header(headers, "Content-Type") or "")
        if _media_type(content_type) not in request.expected_content_types:
            raise BinaryContentError(
                f"Unexpected content type {content_type!r} from {request.public_url}."
            )
        content_length = _parse_positive_header(headers, "Content-Length")

        append = requested_resume_offset > 0 and status == 206
        content_range = _parse_content_range(_header(headers, "Content-Range"))
        if append:
            if content_length is None:
                raise BinaryContentError("A resumed response requires Content-Length.")
            if content_range is None or content_range[0] != requested_resume_offset:
                raise BinaryContentError(
                    "Server did not honor the requested byte range."
                )
            expected_total = content_range[2]
            if content_range[1] - content_range[0] + 1 != content_length:
                raise BinaryContentError("Content-Length and Content-Range disagree.")
        elif status == 206:
            raise BinaryContentError(
                "Unexpected partial response without a resume request."
            )
        else:
            expected_total = (
                content_length
                if content_length is not None
                else request.expected_size_bytes
            )
            if expected_total is None:
                raise BinaryContentError(
                    "Streaming response lacks Content-Length and no "
                    "identity-validated expected size was supplied."
                )

        if (
            request.expected_size_bytes is not None
            and expected_total != request.expected_size_bytes
        ):
            raise BinaryContentError(
                f"Remote size {expected_total:,} does not match advertised size "
                f"{request.expected_size_bytes:,}."
            )
        etag = _header(headers, "ETag")
        last_modified = _header(headers, "Last-Modified")
        resumed_from = requested_resume_offset if append else 0
        mode = "ab" if append else "wb"
        digest = hashlib.sha256()
        if append:
            with request.part_path.open("rb") as existing:
                while chunk := existing.read(self.chunk_size):
                    digest.update(chunk)

        downloaded = resumed_from
        transferred = 0
        _write_partial_state(
            request,
            expected_size_bytes=expected_total,
            etag=etag,
            last_modified=last_modified,
            downloaded_bytes=downloaded,
        )
        if progress_callback:
            progress_callback(
                DownloadProgress(downloaded, expected_total, resumed_from, attempt)
            )
        try:
            with request.part_path.open(mode) as stream:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if _cancelled(cancel_check):
                        raise DownloadCancelledError(
                            f"Download cancelled after {downloaded:,} bytes."
                        )
                    if not chunk:
                        continue
                    stream.write(chunk)
                    digest.update(chunk)
                    chunk_length = len(chunk)
                    downloaded += chunk_length
                    transferred += chunk_length
                    if downloaded > expected_total:
                        raise BinaryContentError(
                            "Response body exceeded the validated content length."
                        )
                    if progress_callback:
                        progress_callback(
                            DownloadProgress(
                                downloaded,
                                expected_total,
                                resumed_from,
                                attempt,
                            )
                        )
                stream.flush()
                os.fsync(stream.fileno())
        except requests.RequestException as exc:
            _write_partial_state(
                request,
                expected_size_bytes=expected_total,
                etag=etag,
                last_modified=last_modified,
                downloaded_bytes=request.part_path.stat().st_size,
            )
            raise BinaryTransportError(
                f"Streaming body failed for {request.public_url} "
                f"({type(exc).__name__})."
            ) from None
        except (BinaryContentError, DownloadCancelledError):
            _write_partial_state(
                request,
                expected_size_bytes=expected_total,
                etag=etag,
                last_modified=last_modified,
                downloaded_bytes=request.part_path.stat().st_size,
            )
            raise

        if (
            content_length is not None and transferred != content_length
        ) or downloaded != expected_total:
            _write_partial_state(
                request,
                expected_size_bytes=expected_total,
                etag=etag,
                last_modified=last_modified,
                downloaded_bytes=request.part_path.stat().st_size,
            )
            raise BinaryContentError(
                "Response body length does not match the validated size: "
                f"received {downloaded:,}, expected {expected_total:,}."
            )
        sha256 = digest.hexdigest()
        if request.expected_sha256 is not None and sha256 != request.expected_sha256:
            raise BinaryContentError(
                f"SHA-256 mismatch for {request.public_url}: "
                f"{sha256} != {request.expected_sha256}."
            )
        zip_crc_validated = False
        if request.require_zip_integrity:
            _write_partial_state(
                request,
                expected_size_bytes=expected_total,
                etag=etag,
                last_modified=last_modified,
                downloaded_bytes=downloaded,
            )
            validate_zip_integrity(request.part_path, chunk_size=self.chunk_size)
            zip_crc_validated = True
        manifest = BinaryDownloadManifest(
            schema_version=BINARY_MANIFEST_SCHEMA,
            provider=request.provider,
            public_url=request.public_url,
            retrieved_at_utc=_utc_now(),
            http_status=status,
            content_type=content_type,
            content_length=expected_total,
            sha256=sha256,
            etag=etag,
            last_modified=last_modified,
            accept_ranges=_header(headers, "Accept-Ranges"),
            response_headers=_public_headers(headers),
            destination_path=request.destination.name,
            resumed_from_bytes=resumed_from,
            zip_crc_validated=zip_crc_validated,
        )
        # Prepare the manifest before replacing a prior valid final archive.
        pending_manifest = request.manifest_path.with_name(
            f".{request.manifest_path.name}.pending"
        )
        _atomic_json(pending_manifest, manifest.as_dict())
        try:
            _finalize_binary_pair(request, pending_manifest)
        finally:
            pending_manifest.unlink(missing_ok=True)
        request.part_manifest_path.unlink(missing_ok=True)
        if progress_callback:
            progress_callback(
                DownloadProgress(
                    downloaded,
                    expected_total,
                    resumed_from,
                    attempt,
                    complete=True,
                )
            )
        return BinaryDownloadResult(
            path=request.destination,
            manifest_path=request.manifest_path,
            manifest=manifest,
            cache_hit=False,
            bytes_transferred=transferred,
        )


def select_sciencebase_file(
    item: ScienceBaseItem,
    *,
    expected_name: str | None = None,
    content_types: Sequence[str] = (),
    suffixes: Sequence[str] = (),
    required_name_marker: str | None = None,
) -> ScienceBaseFile:
    """Select exactly one advertised file by documented identity, never order."""

    if expected_name is None and not (
        content_types or suffixes or required_name_marker
    ):
        raise ValueError("file selection requires at least one explicit criterion")
    allowed_types = {_media_type(value) for value in content_types}
    allowed_suffixes = {value.casefold() for value in suffixes}
    marker = required_name_marker.casefold() if required_name_marker else None
    matches = []
    for candidate in item.files:
        if (
            expected_name is not None
            and candidate.name.casefold() != expected_name.casefold()
        ):
            continue
        if allowed_types and _media_type(candidate.content_type) not in allowed_types:
            continue
        if (
            allowed_suffixes
            and Path(candidate.name).suffix.casefold() not in allowed_suffixes
        ):
            continue
        if marker and marker not in candidate.name.casefold():
            continue
        matches.append(candidate)
    if len(matches) != 1:
        description = expected_name or required_name_marker or "requested metadata file"
        raise ScienceBaseBinarySelectionError(
            f"Expected exactly one ScienceBase file matching {description!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def select_sciencebase_files(
    item: ScienceBaseItem, expected_names: Sequence[str]
) -> tuple[ScienceBaseFile, ...]:
    """Select a caller-ordered set of exact advertised filenames."""

    if not expected_names:
        raise ValueError("expected_names must be non-empty")
    folded = [name.casefold() for name in expected_names]
    if len(folded) != len(set(folded)):
        raise ValueError("expected_names must be unique")
    return tuple(
        select_sciencebase_file(item, expected_name=name) for name in expected_names
    )


def select_bear_lake_phase2_files(
    item: ScienceBaseItem,
    *,
    raster_archive_name: str,
    spatial_metadata_archive_name: str,
) -> BearLakeMetadataFiles:
    """Select the four bounded metadata-first products plus raster identity."""

    raster_archive, spatial_archive = select_sciencebase_files(
        item, (raster_archive_name, spatial_metadata_archive_name)
    )
    for label, selected in (
        ("raster archive", raster_archive),
        ("spatial metadata archive", spatial_archive),
    ):
        if _media_type(selected.content_type) not in {
            "application/zip",
            "application/x-zip-compressed",
            "application/octet-stream",
        } or not selected.name.casefold().endswith(".zip"):
            raise ScienceBaseBinarySelectionError(
                f"Selected {label} {selected.name!r} is not an advertised ZIP."
            )
    fgdc_xml = select_sciencebase_file(
        item,
        content_types=("application/fgdc+xml", "application/xml", "text/xml"),
        suffixes=(".xml",),
    )
    thumbnail = select_sciencebase_file(
        item,
        content_types=("image/png", "image/jpeg"),
        required_name_marker="thumbnail",
    )
    return BearLakeMetadataFiles(
        raster_archive=raster_archive,
        spatial_metadata_archive=spatial_archive,
        fgdc_xml=fgdc_xml,
        thumbnail=thumbnail,
    )


def _member_security(info: zipfile.ZipInfo) -> tuple[bool, bool]:
    mode = (info.external_attr >> 16) & 0xFFFF
    kind = stat.S_IFMT(mode)
    is_symlink = kind == stat.S_IFLNK
    is_special = kind not in (0, stat.S_IFREG, stat.S_IFDIR, stat.S_IFLNK)
    return is_symlink, is_special


def _safe_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith(("/", "//"))
        or _WINDOWS_ABSOLUTE.match(normalized)
        or path.is_absolute()
        or any(part in ("", "..") for part in path.parts)
        or any(":" in part for part in path.parts)
    ):
        raise ZipInventoryError(f"Unsafe ZIP member path {name!r}.")
    return path


def inspect_zip_archive(
    archive_path: str | Path,
    *,
    raster_suffixes: Sequence[str] = DEFAULT_RASTER_SUFFIXES,
) -> ZipInventory:
    """Inspect a ZIP/ZIP64 central directory without extracting its members."""

    archive = Path(archive_path)
    if not archive.is_file():
        raise ZipInventoryError(f"ZIP archive does not exist: {archive}")
    allowed_rasters = {suffix.casefold() for suffix in raster_suffixes}
    members: list[ZipMember] = []
    raster_candidates: list[str] = []
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(archive, "r", allowZip64=True) as bundle:
            for info in bundle.infolist():
                path = _safe_member_path(info.filename)
                normalized = path.as_posix()
                if normalized.casefold() in seen:
                    raise ZipInventoryError(
                        f"ZIP contains duplicate member path {normalized!r}."
                    )
                seen.add(normalized.casefold())
                symlink, special = _member_security(info)
                member = ZipMember(
                    name=normalized,
                    compressed_size_bytes=int(info.compress_size),
                    uncompressed_size_bytes=int(info.file_size),
                    crc32=int(info.CRC),
                    compression_method=int(info.compress_type),
                    is_directory=info.is_dir(),
                    is_encrypted=bool(info.flag_bits & 0x1),
                    is_symlink=symlink,
                    is_special=special,
                )
                members.append(member)
                if not member.is_directory and member.suffix in allowed_rasters:
                    raster_candidates.append(member.name)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ZipInventoryError(f"Cannot inspect ZIP {archive}: {exc}") from exc
    return ZipInventory(
        archive_path=archive,
        archive_size_bytes=archive.stat().st_size,
        archive_sha256=_hash_file(archive),
        members=tuple(members),
        total_compressed_member_bytes=sum(
            member.compressed_size_bytes for member in members
        ),
        total_uncompressed_bytes=sum(
            member.uncompressed_size_bytes for member in members
        ),
        raster_candidates=tuple(raster_candidates),
    )


def select_archive_raster(
    inventory: ZipInventory,
    *,
    expected_name: str | None = None,
) -> ZipMember:
    """Select one raster central-directory record without choosing by order."""

    candidates = [
        member
        for member in inventory.members
        if member.name in inventory.raster_candidates
    ]
    if expected_name is not None:
        normalized = expected_name.replace("\\", "/").casefold()
        candidates = [
            member
            for member in candidates
            if member.name.casefold() == normalized
            or PurePosixPath(member.name).name.casefold()
            == PurePosixPath(normalized).name.casefold()
        ]
    if len(candidates) != 1:
        raise ZipInventoryError(
            "Expected exactly one unambiguous raster candidate; "
            f"found {len(candidates)}: {[item.name for item in candidates]}."
        )
    selected = candidates[0]
    if selected.is_encrypted or selected.is_symlink or selected.is_special:
        raise ZipInventoryError(f"Raster member {selected.name!r} is unsupported.")
    return selected


def calculate_disk_requirements(
    inventory: ZipInventory,
    *,
    selected_members: Sequence[str],
    working_directory: str | Path,
    quantitative_derivative_bytes: int = 0,
    rendering_output_bytes: int = 0,
    temporary_bytes: int | None = None,
    safety_fraction: float = 0.1,
    available_free_bytes: int | None = None,
    archive_already_present: bool = True,
) -> DiskRequirements:
    """Calculate an auditable extraction budget from exact ZIP member sizes."""

    if not selected_members:
        raise ValueError("selected_members must be non-empty")
    if any(
        value < 0 for value in (quantitative_derivative_bytes, rendering_output_bytes)
    ):
        raise ValueError("derived-output byte estimates cannot be negative")
    if safety_fraction < 0:
        raise ValueError("safety_fraction cannot be negative")
    by_name = {member.name: member for member in inventory.members}
    missing = [name for name in selected_members if name not in by_name]
    if missing:
        raise ZipInventoryError(f"Selected ZIP members are missing: {missing}")
    selected_bytes = sum(
        by_name[name].uncompressed_size_bytes for name in selected_members
    )
    temp_bytes = selected_bytes if temporary_bytes is None else int(temporary_bytes)
    if temp_bytes < 0:
        raise ValueError("temporary_bytes cannot be negative")
    base_additional = (
        selected_bytes
        + temp_bytes
        + int(quantitative_derivative_bytes)
        + int(rendering_output_bytes)
    )
    if not archive_already_present:
        base_additional += inventory.archive_size_bytes
    safety = int(base_additional * safety_fraction + 0.999999)
    required = base_additional + safety
    if available_free_bytes is None:
        free = shutil.disk_usage(Path(working_directory)).free
    else:
        free = int(available_free_bytes)
    if free < 0:
        raise ValueError("available_free_bytes cannot be negative")
    projected_total = (
        inventory.archive_size_bytes
        + selected_bytes
        + temp_bytes
        + int(quantitative_derivative_bytes)
        + int(rendering_output_bytes)
        + safety
    )
    return DiskRequirements(
        archive_bytes=inventory.archive_size_bytes,
        selected_extracted_bytes=selected_bytes,
        temporary_bytes=temp_bytes,
        quantitative_derivative_bytes=int(quantitative_derivative_bytes),
        rendering_output_bytes=int(rendering_output_bytes),
        safety_margin_bytes=safety,
        projected_total_bytes=projected_total,
        required_additional_free_bytes=required,
        available_free_bytes=free,
        sufficient=free >= required,
    )


def require_sufficient_disk_space(requirements: DiskRequirements) -> None:
    """Enforce the explicit pre-extraction disk gate."""

    if not requirements.sufficient:
        raise InsufficientDiskSpaceError(
            "Insufficient free space for selective extraction and derivatives: "
            f"need {requirements.required_additional_free_bytes:,} bytes, have "
            f"{requirements.available_free_bytes:,} bytes."
        )


def _validate_extractable(member: ZipMember) -> None:
    _safe_member_path(member.name)
    if member.is_encrypted:
        raise ZipInventoryError(f"Encrypted ZIP member is unsupported: {member.name!r}")
    if member.is_symlink:
        raise ZipInventoryError(f"ZIP symlink is rejected: {member.name!r}")
    if member.is_special:
        raise ZipInventoryError(
            f"Unsupported special ZIP member is rejected: {member.name!r}"
        )


def _copy_and_hash(
    source: BinaryIO, destination: BinaryIO, chunk_size: int
) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    while chunk := source.read(chunk_size):
        destination.write(chunk)
        digest.update(chunk)
        count += len(chunk)
    return count, digest.hexdigest()


def safe_extract_selected(
    archive_path: str | Path,
    *,
    output_dir: str | Path,
    selected_members: Sequence[str],
    inventory: ZipInventory | None = None,
    disk_requirements: DiskRequirements | None = None,
    chunk_size: int = 4 * 1024 * 1024,
) -> tuple[ExtractedMember, ...]:
    """Safely extract only explicitly selected files, using atomic writes."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    archive = Path(archive_path)
    inventory = inventory or inspect_zip_archive(archive)
    if inventory.archive_path.resolve() != archive.resolve():
        raise ZipInventoryError("ZIP inventory belongs to a different archive.")
    if _hash_file(archive) != inventory.archive_sha256:
        raise ZipInventoryError(
            "ZIP archive changed after central-directory inspection."
        )
    if disk_requirements is not None:
        require_sufficient_disk_space(disk_requirements)
    if not selected_members:
        raise ValueError("selected_members must be non-empty")
    by_name = {member.name: member for member in inventory.members}
    if len(set(selected_members)) != len(selected_members):
        raise ZipInventoryError("Selected ZIP member names must be unique.")
    selected = []
    for name in selected_members:
        member = by_name.get(name)
        if member is None:
            raise ZipInventoryError(f"Selected ZIP member is missing: {name!r}")
        if member.is_directory:
            raise ZipInventoryError(f"Cannot extract a directory as a file: {name!r}")
        _validate_extractable(member)
        selected.append(member)
    # Reject a malicious archive even when the unsafe entry was not requested.
    for member in inventory.members:
        _validate_extractable(member)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    results: list[ExtractedMember] = []
    try:
        with zipfile.ZipFile(archive, "r", allowZip64=True) as bundle:
            infos = {
                info.filename.replace("\\", "/"): info for info in bundle.infolist()
            }
            for member in selected:
                relative = _safe_member_path(member.name)
                destination = root.joinpath(*relative.parts)
                try:
                    destination.resolve(strict=False).relative_to(resolved_root)
                except ValueError as exc:
                    raise ZipInventoryError(
                        f"ZIP destination escapes the extraction root: {member.name!r}"
                    ) from exc
                destination.parent.mkdir(parents=True, exist_ok=True)
                part = destination.with_name(f"{destination.name}.part")
                if part.is_symlink():
                    raise ZipInventoryError(
                        f"Partial extraction path is a symlink: {part}"
                    )
                try:
                    with (
                        bundle.open(infos[member.name], "r") as source,
                        part.open("wb") as target,
                    ):
                        size, sha256 = _copy_and_hash(source, target, chunk_size)
                        target.flush()
                        os.fsync(target.fileno())
                    if size != member.uncompressed_size_bytes:
                        raise ZipInventoryError(
                            f"Extracted size mismatch for {member.name!r}: "
                            f"{size} != {member.uncompressed_size_bytes}."
                        )
                    os.replace(part, destination)
                finally:
                    part.unlink(missing_ok=True)
                results.append(
                    ExtractedMember(
                        archive_member=member.name,
                        output_path=destination,
                        size_bytes=size,
                        sha256=sha256,
                    )
                )
    except (OSError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ZipInventoryError):
            raise
        raise ZipInventoryError(f"Selective ZIP extraction failed: {exc}") from exc
    return tuple(results)


def write_extraction_manifest(
    path: str | Path,
    *,
    inventory: ZipInventory,
    extracted_members: Sequence[ExtractedMember],
    relative_to: str | Path | None = None,
) -> ExtractionManifest:
    """Atomically write archive and extracted-file hashes for provenance."""

    base = Path(relative_to).resolve() if relative_to is not None else None

    def display_path(value: Path) -> str:
        resolved = value.resolve()
        if base is not None:
            try:
                return resolved.relative_to(base).as_posix()
            except ValueError:
                pass
        return value.as_posix()

    manifest = ExtractionManifest(
        schema_version=EXTRACTION_MANIFEST_SCHEMA,
        created_at_utc=_utc_now(),
        archive_path=display_path(inventory.archive_path),
        archive_size_bytes=inventory.archive_size_bytes,
        archive_sha256=inventory.archive_sha256,
        total_uncompressed_bytes=inventory.total_uncompressed_bytes,
        raster_candidates=inventory.raster_candidates,
        extracted_members=tuple(
            {
                "archive_member": member.archive_member,
                "output_path": display_path(member.output_path),
                "size_bytes": member.size_bytes,
                "sha256": member.sha256,
            }
            for member in extracted_members
        ),
    )
    _atomic_json(Path(path), manifest.as_dict())
    return manifest


def inventory_records(inventory: ZipInventory) -> tuple[dict[str, Any], ...]:
    """Return detached central-directory rows suitable for CSV/JSON output."""

    return tuple(asdict(member) for member in inventory.members)


def metadata_file_records(
    files: Iterable[ScienceBaseFile],
) -> tuple[dict[str, Any], ...]:
    """Return compact, public ScienceBase file metadata without credentials."""

    return tuple(
        {
            "name": source.name,
            "content_type": source.content_type,
            "size_bytes": source.size_bytes,
            "public_download_url": public_url(source.download_url),
            "download_mode": source.download_mode.value,
            "s3_download_request_page_url": (
                public_url(source.s3_download_request_page_url)
                if source.s3_download_request_page_url
                else None
            ),
            "sciencebase_file_cuid": source.cuid,
            "storage_bucket": source.storage_bucket,
            "storage_key": source.storage_key,
            "path_on_disk": source.path_on_disk,
            "provider_checksum_algorithm": source.checksum_algorithm,
            "provider_checksum_value": source.checksum_value,
        }
        for source in files
    )
