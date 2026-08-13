"""Bounded HTTP utilities shared by Phase 0 hydrometry adapters.

The client is deliberately small.  Provider adapters remain responsible for
validating payload schemas and identities; this module handles transport,
content negotiation, bounded retries, and common OGC/JSON:API pagination.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

DEFAULT_USER_AGENT = (
    "GreatBasinWater-hydrometry-phase0/1.0 "
    "(+https://github.com/; research capability probe)"
)
DEFAULT_JSON_CONTENT_TYPES = (
    "application/json",
    "application/geo+json",
    "application/ld+json",
    "application/schema+json",
    "application/vnd.api+json",
)
TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)


class HydrometryHttpError(RuntimeError):
    """Base exception for HTTP transport failures."""


class HttpStatusError(HydrometryHttpError):
    """Raised for a non-success response after any allowed retries."""

    def __init__(self, status_code: int, public_url: str) -> None:
        self.status_code = status_code
        self.public_url = public_url
        super().__init__(f"HTTP {status_code} from {public_url}")


class HttpTransportError(HydrometryHttpError):
    """Raised when a request cannot be completed."""


class ContentTypeError(HydrometryHttpError):
    """Raised when a provider returns an unexpected representation."""


class PaginationError(HydrometryHttpError):
    """Raised for a malformed, cyclic, or unbounded pagination chain."""


class ResponseJsonError(HydrometryHttpError):
    """Raised when an accepted JSON response cannot be decoded."""


def _normalized_secret_key(key: object) -> str:
    return str(key).strip().lower().replace("-", "_")


def is_secret_key(key: object) -> bool:
    """Return whether a request key can contain authentication material."""

    normalized = _normalized_secret_key(key)
    return any(part in normalized for part in SECRET_KEY_PARTS)


def sanitize_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a detached request mapping with secret-bearing fields removed."""

    if not value:
        return {}
    clean: dict[str, Any] = {}
    for key, item in value.items():
        if is_secret_key(key):
            continue
        if isinstance(item, Mapping):
            clean[str(key)] = sanitize_mapping(item)
        elif isinstance(item, (list, tuple)):
            clean[str(key)] = [
                sanitize_mapping(part) if isinstance(part, Mapping) else part
                for part in item
            ]
        else:
            clean[str(key)] = item
    return clean


def public_url(url: str, params: Mapping[str, Any] | None = None) -> str:
    """Build a deterministic request URL with authentication fields removed."""

    parts = urlsplit(url)
    query = [
        (key, value) for key, value in parse_qsl(parts.query) if not is_secret_key(key)
    ]
    for key, value in sanitize_mapping(params).items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            query.extend((key, str(part)) for part in value)
        else:
            query.append((key, str(value)))
    query.sort()
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def _media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


@dataclass(frozen=True, slots=True)
class HttpResponseData:
    """Detached, serializable-enough HTTP response used by adapters/caches."""

    body: bytes
    status_code: int
    content_type: str
    retrieved_at_utc: datetime
    url_public: str
    headers: Mapping[str, str]

    def json(self) -> Any:
        """Decode the response body as JSON with a transport-specific error."""

        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResponseJsonError(
                f"Invalid JSON returned by {self.url_public}: {exc}"
            ) from exc


def next_page_url(payload: Mapping[str, Any]) -> str | None:
    """Return an OGC or JSON:API next-page URL, if present."""

    links = payload.get("links")
    if isinstance(links, Mapping):
        value = links.get("next")
        if isinstance(value, Mapping):
            value = value.get("href")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None
    if not isinstance(links, Sequence) or isinstance(links, (str, bytes)):
        return None
    for link in links:
        if not isinstance(link, Mapping):
            continue
        if str(link.get("rel", "")).lower() != "next":
            continue
        href = link.get("href")
        if isinstance(href, str) and href.strip():
            return href.strip()
    return None


class HttpClient:
    """Synchronous client with narrow retries and bounded pagination."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        user_agent: str = DEFAULT_USER_AGENT,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5,
        max_backoff_seconds: float = 8.0,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if backoff_base_seconds < 0 or max_backoff_seconds < 0:
            raise ValueError("backoff durations cannot be negative")
        if not user_agent.strip():
            raise ValueError("user_agent must be non-empty")
        self.timeout = float(timeout)
        self.user_agent = user_agent.strip()
        self.max_retries = int(max_retries)
        self.backoff_base_seconds = float(backoff_base_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self.session = session or requests.Session()
        self._sleep = sleep
        self.request_count = 0

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        expected_content_types: Sequence[str] = DEFAULT_JSON_CONTENT_TYPES,
        timeout: float | None = None,
    ) -> HttpResponseData:
        """GET one representation, retrying only throttling/transient failures."""

        request_headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        request_headers.update(headers or {})
        safe_url = public_url(url, params)
        request_timeout = self.timeout if timeout is None else float(timeout)
        if request_timeout <= 0:
            raise ValueError("timeout must be positive")

        for attempt in range(self.max_retries + 1):
            self.request_count += 1
            try:
                response = self.session.get(
                    url,
                    params=dict(params or {}),
                    headers=request_headers,
                    timeout=request_timeout,
                )
            except requests.RequestException as exc:
                # Transport errors are not retried: retry policy is deliberately
                # limited to explicit 429/transient HTTP responses.
                raise HttpTransportError(
                    f"Request failed for {safe_url}: {exc}"
                ) from exc

            status = int(response.status_code)
            if status in TRANSIENT_STATUS_CODES and attempt < self.max_retries:
                delay = min(
                    self.max_backoff_seconds,
                    self.backoff_base_seconds * (2**attempt),
                )
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = min(
                            self.max_backoff_seconds, max(delay, float(retry_after))
                        )
                    except ValueError:
                        pass
                self._sleep(delay)
                continue
            if not 200 <= status < 300:
                raise HttpStatusError(status, safe_url)

            content_type = str(response.headers.get("Content-Type", ""))
            actual_media_type = _media_type(content_type)
            allowed = {_media_type(item) for item in expected_content_types}
            if actual_media_type not in allowed:
                raise ContentTypeError(
                    f"Unexpected content type {content_type!r} from {safe_url}; "
                    f"expected one of {sorted(allowed)}."
                )
            public_headers = {
                str(key): str(value)
                for key, value in response.headers.items()
                if not is_secret_key(key)
            }
            return HttpResponseData(
                body=bytes(response.content),
                status_code=status,
                content_type=content_type,
                retrieved_at_utc=datetime.now(UTC),
                url_public=safe_url,
                headers=public_headers,
            )
        raise AssertionError("retry loop terminated unexpectedly")

    def iter_json_pages(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        expected_content_types: Sequence[str] = DEFAULT_JSON_CONTENT_TYPES,
        max_pages: int = 50,
    ) -> Iterator[HttpResponseData]:
        """Yield bounded OGC/JSON:API pages while following explicit next links."""

        if max_pages <= 0:
            raise ValueError("max_pages must be positive")
        next_url: str | None = url
        next_params = params
        seen: set[str] = set()
        for _page_number in range(max_pages):
            if next_url is None:
                return
            identity = public_url(next_url, next_params)
            if identity in seen:
                raise PaginationError(f"Cyclic pagination link: {identity}")
            seen.add(identity)
            response = self.get(
                next_url,
                params=next_params,
                headers=headers,
                expected_content_types=expected_content_types,
            )
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise PaginationError(
                    f"Pagination payload from {response.url_public} is not an object."
                )
            yield response
            next_url = next_page_url(payload)
            next_params = None
        if next_url is not None:
            raise PaginationError(f"Pagination exceeded maximum of {max_pages} pages.")
