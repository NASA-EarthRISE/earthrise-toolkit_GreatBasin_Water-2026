"""Immutable-response cache and public request provenance for hydrometry probes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .http import (
    DEFAULT_JSON_CONTENT_TYPES,
    HttpClient,
    HttpResponseData,
    ResponseJsonError,
    public_url,
    sanitize_mapping,
)


class CacheError(RuntimeError):
    """Base exception for hydrometry cache failures."""


class CacheMissError(CacheError):
    """Raised when an offline request has no valid matching cache."""


class CacheIntegrityError(CacheError):
    """Raised when cached bytes do not match their manifest."""


class CacheMode(str, Enum):
    """Explicit network behavior; callers must choose one."""

    LIVE = "live"
    OFFLINE = "offline"


@dataclass(frozen=True, slots=True)
class CacheManifest:
    provider: str
    request_method: str
    request_parameters: Mapping[str, Any]
    public_url: str
    retrieved_at_utc: str
    http_status: int
    content_type: str
    sha256: str
    response_schema: str | None
    response_headers: Mapping[str, str]
    raw_response_path: str
    normalized_output_path: str | None


@dataclass(frozen=True, slots=True)
class CachedResponseData:
    """Response plus persisted public provenance, from cache or network."""

    body: bytes
    manifest: CacheManifest
    cache_hit: bool

    @property
    def status_code(self) -> int:
        return self.manifest.http_status

    @property
    def content_type(self) -> str:
        return self.manifest.content_type

    @property
    def url_public(self) -> str:
        return self.manifest.public_url

    @property
    def retrieved_at_utc(self) -> datetime:
        return datetime.fromisoformat(
            self.manifest.retrieved_at_utc.replace("Z", "+00:00")
        ).astimezone(UTC)

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResponseJsonError(
                f"Invalid cached JSON for {self.manifest.public_url}: {exc}"
            ) from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _atomic_write(path: Path, body: bytes) -> None:
    """Write one file with fsync + replace, preserving any prior valid file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class CacheStore:
    """Content-validated ignored cache rooted at ``data/hydrometry/cache``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._accessed_raw_response_paths: set[str] = set()

    @property
    def accessed_raw_response_paths(self) -> tuple[str, ...]:
        """Raw response paths successfully loaded or written by this instance."""

        return tuple(sorted(self._accessed_raw_response_paths))

    def _identity(
        self,
        provider: str,
        url: str,
        params: Mapping[str, Any] | None,
    ) -> tuple[str, dict[str, Any], str]:
        safe_params = sanitize_mapping(params)
        safe_url = public_url(url, params)
        request = {
            "provider": provider,
            "method": "GET",
            "url": safe_url,
            "parameters": safe_params,
        }
        digest = hashlib.sha256(_canonical_json(request).encode()).hexdigest()
        return digest, safe_params, safe_url

    def paths(
        self,
        provider: str,
        url: str,
        params: Mapping[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        """Return stable raw/manifest paths without exposing authentication."""

        digest, _safe_params, _safe_url = self._identity(provider, url, params)
        directory = self.root / provider
        return directory / f"{digest}.json", directory / f"{digest}.meta.json"

    def load(
        self,
        *,
        provider: str,
        url: str,
        params: Mapping[str, Any] | None = None,
    ) -> CachedResponseData:
        """Load a cache only after validating request identity and SHA-256."""

        raw_path, manifest_path = self.paths(provider, url, params)
        if not raw_path.is_file() or not manifest_path.is_file():
            raise CacheMissError(f"No cache for {public_url(url, params)}")
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = CacheManifest(**data)
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            raise CacheIntegrityError(
                f"Invalid cache manifest {manifest_path}: {exc}"
            ) from exc
        _digest, safe_params, safe_url = self._identity(provider, url, params)
        if (
            manifest.provider != provider
            or manifest.request_method != "GET"
            or dict(manifest.request_parameters) != safe_params
            or manifest.public_url != safe_url
        ):
            raise CacheIntegrityError(
                f"Cache request identity mismatch: {manifest_path}"
            )
        try:
            body = raw_path.read_bytes()
        except OSError as exc:
            raise CacheIntegrityError(
                f"Cannot read cached response {raw_path}: {exc}"
            ) from exc
        actual_hash = hashlib.sha256(body).hexdigest()
        if actual_hash != manifest.sha256:
            raise CacheIntegrityError(
                f"Cached response hash mismatch for {raw_path}: "
                f"{actual_hash} != {manifest.sha256}"
            )
        self._accessed_raw_response_paths.add(manifest.raw_response_path)
        return CachedResponseData(body=body, manifest=manifest, cache_hit=True)

    def write(
        self,
        *,
        provider: str,
        url: str,
        params: Mapping[str, Any] | None,
        response: HttpResponseData,
        response_schema: str | None = None,
        normalized_output_path: str | Path | None = None,
    ) -> CachedResponseData:
        """Persist a successful response and manifest using atomic replaces."""

        raw_path, manifest_path = self.paths(provider, url, params)
        _digest, safe_params, safe_url = self._identity(provider, url, params)
        if response.url_public != safe_url:
            raise CacheError(
                "HTTP response public URL does not match cache request identity."
            )
        raw_relative = raw_path.relative_to(self.root).as_posix()
        normalized = (
            str(Path(normalized_output_path)).replace("\\", "/")
            if normalized_output_path is not None
            else None
        )
        manifest = CacheManifest(
            provider=provider,
            request_method="GET",
            request_parameters=safe_params,
            public_url=safe_url,
            retrieved_at_utc=response.retrieved_at_utc.astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            http_status=response.status_code,
            content_type=response.content_type,
            sha256=hashlib.sha256(response.body).hexdigest(),
            response_schema=response_schema,
            response_headers=sanitize_mapping(response.headers),
            raw_response_path=raw_relative,
            normalized_output_path=normalized,
        )
        # A failed request never reaches this method.  Write raw bytes first and
        # the manifest last; hash validation rejects any interrupted pair.
        _atomic_write(raw_path, response.body)
        _atomic_write(
            manifest_path,
            (json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n").encode(),
        )
        self._accessed_raw_response_paths.add(manifest.raw_response_path)
        return CachedResponseData(
            body=response.body, manifest=manifest, cache_hit=False
        )

    def request(
        self,
        *,
        provider: str,
        client: HttpClient,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        mode: CacheMode,
        refresh: bool = False,
        expected_content_types: Sequence[str] = DEFAULT_JSON_CONTENT_TYPES,
        response_schema: str | None = None,
        normalized_output_path: str | Path | None = None,
    ) -> CachedResponseData:
        """Load or fetch one request under explicit live/offline semantics."""

        if not isinstance(mode, CacheMode):
            raise TypeError("mode must be explicitly set to CacheMode.LIVE or OFFLINE")
        if mode is CacheMode.OFFLINE:
            if refresh:
                raise ValueError("refresh cannot be used in offline mode")
            return self.load(provider=provider, url=url, params=params)
        if not refresh:
            try:
                return self.load(provider=provider, url=url, params=params)
            except CacheMissError:
                pass
        response = client.get(
            url,
            params=params,
            headers=headers,
            expected_content_types=expected_content_types,
        )
        return self.write(
            provider=provider,
            url=url,
            params=params,
            response=response,
            response_schema=response_schema,
            normalized_output_path=normalized_output_path,
        )

    def link_normalized_output(
        self,
        normalized_output_path: str | Path,
        *,
        raw_response_paths: Sequence[str] | None = None,
    ) -> int:
        """Atomically link each valid manifest to a normalized inventory output.

        Provider requests occur before the inventory output path exists, so the
        link is filled only after a successful normalized write. Corrupt or
        incomplete manifest/raw pairs are left untouched and reported through
        the returned count rather than overwriting valid cache state.
        """

        normalized = str(Path(normalized_output_path)).replace("\\", "/")
        count = 0
        if raw_response_paths is None:
            allowed = set(self._accessed_raw_response_paths)
        else:
            allowed = set(raw_response_paths)
        if not allowed:
            return 0
        for manifest_path in sorted(self.root.glob("*/*.meta.json")):
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest = CacheManifest(**data)
                if manifest.raw_response_path not in allowed:
                    continue
                raw_path = self.root / manifest.raw_response_path
                body = raw_path.read_bytes()
                if hashlib.sha256(body).hexdigest() != manifest.sha256:
                    continue
                updated = asdict(manifest)
                updated["normalized_output_path"] = normalized
                _atomic_write(
                    manifest_path,
                    (json.dumps(updated, indent=2, sort_keys=True) + "\n").encode(),
                )
                count += 1
            except (OSError, TypeError, json.JSONDecodeError):
                continue
        return count
