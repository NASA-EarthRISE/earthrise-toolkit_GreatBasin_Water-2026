"""Common adapter contracts and provider-specific validation errors."""

from __future__ import annotations

from abc import ABC

from ..cache import CachedResponseData, CacheMode, CacheStore
from ..http import HttpClient


class AdapterError(RuntimeError):
    """Base exception for provider adapter failures."""


class ProviderSchemaError(AdapterError):
    """Raised when a successful response violates the provider schema."""


class ProviderIdentityError(AdapterError):
    """Raised when returned source metadata does not match the requested source."""


class ProviderValidationError(AdapterError):
    """Raised when parameter, unit, datum, or date validation fails."""


class BaseAdapter(ABC):
    """Base carrying explicit network mode and optional response cache."""

    provider: str

    def __init__(
        self,
        *,
        client: HttpClient,
        mode: CacheMode,
        cache: CacheStore | None = None,
        refresh: bool = False,
    ) -> None:
        if not isinstance(mode, CacheMode):
            raise TypeError("mode must be explicitly set to CacheMode.LIVE or OFFLINE")
        if mode is CacheMode.OFFLINE and cache is None:
            raise ValueError("offline adapters require a CacheStore")
        if mode is CacheMode.OFFLINE and refresh:
            raise ValueError("refresh cannot be used in offline mode")
        self.client = client
        self.mode = mode
        self.cache = cache
        self.refresh = refresh

    def _get(
        self,
        url: str,
        *,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        response_schema: str | None = None,
    ) -> CachedResponseData:
        if self.cache is None:
            if self.mode is CacheMode.OFFLINE:
                raise AssertionError("offline adapter created without cache")
            response = self.client.get(url, params=params, headers=headers)
            from ..cache import CacheManifest  # local import keeps module surface small

            manifest = CacheManifest(
                provider=self.provider,
                request_method="GET",
                request_parameters={},
                public_url=response.url_public,
                retrieved_at_utc=response.retrieved_at_utc.isoformat().replace(
                    "+00:00", "Z"
                ),
                http_status=response.status_code,
                content_type=response.content_type,
                sha256="",
                response_schema=response_schema,
                response_headers={},
                raw_response_path="",
                normalized_output_path=None,
            )
            return CachedResponseData(response.body, manifest, cache_hit=False)
        return self.cache.request(
            provider=self.provider,
            client=self.client,
            url=url,
            params=params,
            headers=headers,
            mode=self.mode,
            refresh=self.refresh,
            response_schema=response_schema,
        )
