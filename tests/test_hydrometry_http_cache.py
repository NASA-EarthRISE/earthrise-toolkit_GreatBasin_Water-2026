from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import requests

from regional_analysis.hydrometry.cache import (
    CacheIntegrityError,
    CacheMissError,
    CacheMode,
    CacheStore,
)
from regional_analysis.hydrometry.http import (
    ContentTypeError,
    HttpClient,
    HttpResponseData,
    HttpStatusError,
    HttpTransportError,
    PaginationError,
    public_url,
)


def response(status: int, payload: object, *, content_type: str = "application/json"):
    item = Mock()
    item.status_code = status
    item.content = json.dumps(payload).encode()
    item.headers = {"Content-Type": content_type}
    return item


class HttpClientTests(unittest.TestCase):
    def test_retries_only_transient_statuses_with_exponential_backoff(self) -> None:
        session = Mock()
        session.get.side_effect = [
            response(429, {}),
            response(503, {}),
            response(200, {"ok": True}),
        ]
        delays: list[float] = []
        client = HttpClient(
            session=session,
            max_retries=3,
            backoff_base_seconds=0.25,
            sleep=delays.append,
        )
        result = client.get("https://example.test/items")
        self.assertEqual(result.json(), {"ok": True})
        self.assertEqual(client.request_count, 3)
        self.assertEqual(delays, [0.25, 0.5])

    def test_does_not_retry_client_or_transport_errors(self) -> None:
        session = Mock()
        session.get.return_value = response(400, {"error": "bad"})
        client = HttpClient(session=session, max_retries=3, sleep=Mock())
        with self.assertRaises(HttpStatusError):
            client.get("https://example.test/items")
        self.assertEqual(session.get.call_count, 1)

        session.reset_mock()
        session.get.side_effect = requests.ConnectionError("offline")
        with self.assertRaises(HttpTransportError):
            client.get("https://example.test/items")
        self.assertEqual(session.get.call_count, 1)

    def test_rejects_unexpected_content_type(self) -> None:
        session = Mock()
        session.get.return_value = response(200, {}, content_type="text/html")
        with self.assertRaises(ContentTypeError):
            HttpClient(session=session).get("https://example.test/items")

    def test_pagination_is_bounded_and_detects_cycle(self) -> None:
        session = Mock()
        same = "https://example.test/items?offset=1"
        session.get.side_effect = [
            response(200, {"features": [], "links": [{"rel": "next", "href": same}]}),
            response(200, {"features": [], "links": [{"rel": "next", "href": same}]}),
        ]
        client = HttpClient(session=session)
        with self.assertRaises(PaginationError):
            list(client.iter_json_pages("https://example.test/items", max_pages=5))

    def test_public_url_removes_credentials(self) -> None:
        url = public_url(
            "https://example.test/items?token=hidden&f=json",
            {"api_key": "hidden", "site": "123"},
        )
        self.assertEqual(url, "https://example.test/items?f=json&site=123")
        self.assertNotIn("hidden", url)


class CacheStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = CacheStore(Path(self.temporary.name))
        self.url = "https://example.test/items"

    def _successful_response(self, value: int = 1) -> HttpResponseData:
        body = json.dumps({"value": value}).encode()
        return HttpResponseData(
            body=body,
            status_code=200,
            content_type="application/json",
            retrieved_at_utc=datetime(2026, 8, 13, tzinfo=UTC),
            url_public="https://example.test/items?site=123",
            headers={"Content-Type": "application/json", "X-Api-Key": "hidden"},
        )

    def test_atomic_cache_round_trip_hash_and_secret_redaction(self) -> None:
        written = self.store.write(
            provider="fixture",
            url=self.url,
            params={"site": "123", "api_key": "hidden"},
            response=self._successful_response(),
            response_schema="fixture-v1",
        )
        loaded = self.store.load(
            provider="fixture",
            url=self.url,
            params={"site": "123", "api_key": "different-secret"},
        )
        self.assertFalse(written.cache_hit)
        self.assertTrue(loaded.cache_hit)
        self.assertEqual(loaded.json(), {"value": 1})
        manifest_text = next(Path(self.temporary.name).rglob("*.meta.json")).read_text()
        self.assertNotIn("hidden", manifest_text)
        self.assertNotIn("different-secret", manifest_text)
        self.assertNotIn("X-Api-Key", manifest_text)

        raw_path, _manifest = self.store.paths("fixture", self.url, {"site": "123"})
        raw_path.write_bytes(b"tampered")
        with self.assertRaises(CacheIntegrityError):
            self.store.load(provider="fixture", url=self.url, params={"site": "123"})

    def test_offline_uses_cache_only_and_miss_is_clear(self) -> None:
        client = Mock(spec=HttpClient)
        with self.assertRaises(CacheMissError):
            self.store.request(
                provider="fixture",
                client=client,
                url=self.url,
                mode=CacheMode.OFFLINE,
            )
        client.get.assert_not_called()

    def test_failed_refresh_preserves_prior_valid_cache(self) -> None:
        self.store.write(
            provider="fixture",
            url=self.url,
            params={"site": "123"},
            response=self._successful_response(),
        )
        client = Mock(spec=HttpClient)
        client.get.side_effect = HttpTransportError("failed")
        with self.assertRaises(HttpTransportError):
            self.store.request(
                provider="fixture",
                client=client,
                url=self.url,
                params={"site": "123"},
                mode=CacheMode.LIVE,
                refresh=True,
            )
        preserved = self.store.load(
            provider="fixture", url=self.url, params={"site": "123"}
        )
        self.assertEqual(preserved.json(), {"value": 1})

    def test_refresh_is_rejected_offline(self) -> None:
        with self.assertRaises(ValueError):
            self.store.request(
                provider="fixture",
                client=Mock(spec=HttpClient),
                url=self.url,
                mode=CacheMode.OFFLINE,
                refresh=True,
            )


if __name__ == "__main__":
    unittest.main()
