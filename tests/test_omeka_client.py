"""Fetch-truncation reconciliation (safety rail B1) — no network involved."""

import asyncio
import json
import os

import pytest

from iwac_common.omeka_client import Cache, Config, OmekaApiClient, TruncatedFetchError


class StubClient(OmekaApiClient):
    """Serves canned pages; overrides the two network entry points."""

    def __init__(self, pages, total):
        super().__init__(Config(), use_cache=False)
        self._pages = pages
        self._total = total

    async def fetch_items_page(self, rcid, page, per=100):
        try:
            return self._pages[page - 1]
        except IndexError:
            return []

    async def fetch_total_results(self, rcid):
        return self._total


def _items(n, start=0):
    return [{"o:id": i} for i in range(start, start + n)]


class TestDiskCache:
    def test_atomic_round_trip_leaves_no_temp_file(self, tmp_path):
        cache = Cache(str(tmp_path))
        asyncio.run(cache.set("key", {"value": 1}))
        assert asyncio.run(cache.get("key")) == {"value": 1}
        assert list(tmp_path.glob("*.tmp")) == []

    def test_corrupt_entry_is_deleted_and_treated_as_miss(self, tmp_path):
        cache = Cache(str(tmp_path))
        path = cache._path("key")
        with open(path, "wb") as handle:
            handle.write(b"not gzip")
        assert asyncio.run(cache.get("key")) is None
        assert not os.path.exists(path)


class TestFetchReconciliation:
    def test_complete_fetch_passes(self):
        pages = [_items(100), _items(37, start=100)]
        client = StubClient(pages, total=137)
        items = asyncio.run(client.fetch_items(36))
        assert len(items) == 137

    def test_truncated_fetch_raises(self):
        # Server reports 250 items but pagination stopped after a short page.
        pages = [_items(100), _items(40, start=100)]
        client = StubClient(pages, total=250)
        with pytest.raises(TruncatedFetchError):
            asyncio.run(client.fetch_items(36))

    def test_unknown_total_fails_closed(self):
        pages = [_items(42)]
        client = StubClient(pages, total=None)
        with pytest.raises(TruncatedFetchError, match="no Omeka-S-Total-Results"):
            asyncio.run(client.fetch_items(36))

    def test_reconciliation_has_no_opt_out(self):
        client = StubClient([_items(10)], total=999)
        with pytest.raises(TypeError):
            asyncio.run(client.fetch_items(36, verify_total=False))

    def test_the_reported_total_fixes_the_pages_requested(self):
        requested = []

        class Recording(StubClient):
            async def fetch_items_page(self, rcid, page, per=100):
                requested.append(page)
                return await super().fetch_items_page(rcid, page, per)

        pages = [_items(100), _items(100, start=100), _items(5, start=200), _items(9)]
        items = asyncio.run(Recording(pages, total=205).fetch_items(36))
        assert len(items) == 205
        assert sorted(requested) == [1, 2, 3]


class _FakeResponse:
    def __init__(self, status, payload=None, error=None):
        self.status = status
        self._payload = payload
        self._error = error

    async def json(self):
        if self._error is not None:
            raise self._error
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Plays one scripted outcome per request: a response or an exception."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


MANIFEST = {"thumbnail": [{"id": "https://example.test/thumb.jpg"}]}


class TestIIIFThumbnailRetry:
    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch):
        from iwac_common import omeka_client as oc

        self.sleeps = []

        async def no_sleep(seconds):
            self.sleeps.append(seconds)

        monkeypatch.setattr(oc.asyncio, "sleep", no_sleep)
        oc.media_stats.reset()
        yield
        oc.media_stats.reset()

    def _fetch(self, session, **kwargs):
        from iwac_common import omeka_client as oc

        return asyncio.run(oc.fetch_iiif_thumbnail_url(7, session, **kwargs))

    def test_transient_failures_are_retried_before_success(self):
        import aiohttp

        from iwac_common import omeka_client as oc

        session = _FakeSession(
            asyncio.TimeoutError(),
            aiohttp.ClientConnectionError("reset"),
            _FakeResponse(503),
            _FakeResponse(429),
            _FakeResponse(200, MANIFEST),
        )
        assert self._fetch(session) == "https://example.test/thumb.jpg"
        assert session.calls == 5
        assert self.sleeps == [1, 2, 4, 8]
        assert (oc.media_stats.attempted, oc.media_stats.failed) == (1, 0)

    def test_persistent_transient_failure_is_recorded_once(self):
        from iwac_common import omeka_client as oc

        session = _FakeSession(*(_FakeResponse(502) for _ in range(3)))
        assert self._fetch(session, max_tries=3) == ""
        assert session.calls == 3
        assert oc.media_stats.failed == 1
        assert oc.media_stats.failed_fields_by_item == {"7": {"thumbnail", "iiif_manifest"}}

    def test_failure_restricts_only_the_fields_the_caller_names(self):
        from iwac_common import omeka_client as oc

        session = _FakeSession(_FakeResponse(500))
        assert self._fetch(session, max_tries=1, affected_fields=("thumbnail",)) == ""
        assert oc.media_stats.failed_fields_by_item == {"7": {"thumbnail"}}

    @pytest.mark.parametrize("status", [404, 410])
    def test_absence_is_not_retried_or_failed(self, status):
        from iwac_common import omeka_client as oc

        session = _FakeSession(_FakeResponse(status))
        assert self._fetch(session) == ""
        assert session.calls == 1
        assert oc.media_stats.failed == 0

    @pytest.mark.parametrize("outcome", [
        _FakeResponse(403),
        _FakeResponse(200, error=json.JSONDecodeError("bad", "x", 0)),
    ])
    def test_permanent_failure_is_recorded_without_retry(self, outcome):
        from iwac_common import omeka_client as oc

        session = _FakeSession(outcome)
        assert self._fetch(session) == ""
        assert session.calls == 1
        assert oc.media_stats.failed == 1

    def test_manifest_without_thumbnail_is_an_absence(self):
        from iwac_common import omeka_client as oc

        assert self._fetch(_FakeSession(_FakeResponse(200, {"thumbnail": [{"id": None}]}))) == ""
        assert oc.media_stats.failed == 0

    def test_session_defaults_to_the_connection_manager(self, monkeypatch):
        from iwac_common import omeka_client as oc

        session = _FakeSession(_FakeResponse(200, MANIFEST))

        async def shared():
            return session

        monkeypatch.setattr(oc.conn_manager, "get", shared)
        assert asyncio.run(oc.fetch_iiif_thumbnail_url(7)) == "https://example.test/thumb.jpg"
        assert session.calls == 1


class TestMediaPointers:
    """One helper for the file / thumbnail / manifest triple of five mappers."""

    class _Api:
        def __init__(self, original_url):
            self.original_url = original_url

        async def fetch_media_data(self, media_id):
            return {"o:is_public": True, "o:original_url": self.original_url}

    def _run(self, monkeypatch, item, api, thumbnail="", **kwargs):
        from iwac_common import omeka_client as oc

        monkeypatch.setenv("OMEKA_BASE_URL", "https://staging.example/api")
        monkeypatch.delenv("IWAC_PUBLIC_BASE_URL", raising=False)
        calls = []

        async def fake_thumbnail(omeka_id, session=None):
            calls.append(omeka_id)
            return thumbnail

        monkeypatch.setattr(oc, "fetch_iiif_thumbnail_url", fake_thumbnail)
        result = asyncio.run(oc.fetch_media_pointers(item, api, **kwargs))
        return result, calls

    MEDIA_ITEM = {
        "o:id": 3,
        "o:primary_media": {"@id": "https://staging.example/api/media/4"},
        "thumbnail_display_urls": {"large": "https://staging.example/large.jpg"},
    }
    MANIFEST_URL = "https://staging.example/iiif/3/3/manifest"

    def test_no_media_no_lookups(self, monkeypatch):
        result, calls = self._run(monkeypatch, {"o:id": 3}, self._Api("x"))
        assert result == ("", "", "") and calls == []

    def test_file_gives_all_three_pointers(self, monkeypatch):
        result, calls = self._run(
            monkeypatch, dict(self.MEDIA_ITEM), self._Api("https://f/a.pdf"), thumbnail="T"
        )
        assert result == ("https://f/a.pdf", "T", self.MANIFEST_URL)
        assert calls == [3]

    def test_fallback_only_when_enabled(self, monkeypatch):
        api = self._Api("https://f/a.jpg")
        off, _ = self._run(monkeypatch, dict(self.MEDIA_ITEM), api)
        on, _ = self._run(monkeypatch, dict(self.MEDIA_ITEM), api, thumbnail_fallback=True)
        assert off[1] == ""
        assert on[1] == "https://staging.example/large.jpg"

    def test_media_without_file(self, monkeypatch):
        """A YouTube media: thumbnail only if the caller does not require a file."""
        api = self._Api(None)
        strict, strict_calls = self._run(monkeypatch, dict(self.MEDIA_ITEM), api, thumbnail="T")
        assert strict == ("", "", "") and strict_calls == []
        loose, _ = self._run(
            monkeypatch, dict(self.MEDIA_ITEM), api, thumbnail="T",
            thumbnail_requires_file=False,
        )
        assert loose == ("", "T", "")


class TestMediaFetchGuard:
    """Mass media-lookup failure blanks PDF/thumbnail while leaving row counts
    intact, so neither the shrink tripwire nor the total-count reconciliation
    sees it. This guard is the only thing standing between a network outage
    and a wiped media column."""

    def _stats(self, attempted, failed):
        from iwac_common.omeka_client import MediaFetchStats

        s = MediaFetchStats()
        s.attempted, s.failed = attempted, failed
        return s

    def test_no_failures_is_silent(self):
        assert self._stats(500, 0).check() is None

    def test_total_outage_aborts(self):
        from iwac_common.omeka_client import MediaFetchGuardError

        with pytest.raises(MediaFetchGuardError):
            self._stats(1501, 1501).check()

    def test_a_few_bad_manifests_abort(self):
        """Any lookup error could overwrite a valid prior URL."""
        from iwac_common.omeka_client import MediaFetchGuardError

        with pytest.raises(MediaFetchGuardError):
            self._stats(500, 5).check()

    def test_small_subset_also_fails_closed(self):
        from iwac_common.omeka_client import MediaFetchGuardError

        with pytest.raises(MediaFetchGuardError):
            self._stats(3, 1).check()

    def test_override_downgrades_to_warning(self):
        summary = self._stats(1501, 1501).check(allow_failures=True)
        assert summary and "100%" in summary

    def test_absences_are_not_failures(self):
        """An item with no primary media makes no attempt, so sparse media
        coverage must never look like failure."""
        from iwac_common.omeka_client import MediaFetchStats

        s = MediaFetchStats()
        for _ in range(50):
            s.record_attempt()          # only items that HAVE media
        assert s.failure_rate == 0.0
        assert s.check() is None


class TestCredentialRedaction:
    """Omeka authenticates through query parameters, so an unsanitized aiohttp
    error prints ``key_credential=…`` into every retry warning and the final
    abort panel. Reproduced against a local server; no external network."""

    SECRET = "SECRET-CRED-xyz/+="

    def _run_against(self, status, caplog):
        from aiohttp import web

        from iwac_common import omeka_client as oc

        hits = {"n": 0}

        async def handler(request):
            hits["n"] += 1
            return web.Response(status=status)

        async def scenario():
            app = web.Application()
            app.router.add_get("/api/items", handler)
            runner = web.AppRunner(app)
            await runner.setup()
            import socket

            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            site = web.TCPSite(runner, "127.0.0.1", port)
            await site.start()
            cfg = Config(
                API_URL=f"http://127.0.0.1:{port}/api",
                API_KEY_IDENTITY="IDENT-abc",
                API_KEY_CREDENTIAL=self.SECRET,
            )
            client = OmekaApiClient(cfg, use_cache=False)
            real_sleep = asyncio.sleep

            async def no_backoff(_seconds):
                await real_sleep(0)

            oc.asyncio.sleep = no_backoff
            try:
                with pytest.raises(TruncatedFetchError) as info:
                    await client.fetch_items(36)
            finally:
                oc.asyncio.sleep = real_sleep
                await oc.conn_manager.close()
                await runner.cleanup()
            return info.value

        with caplog.at_level("WARNING"):
            error = asyncio.run(scenario())
        return error, hits["n"]

    def _assert_clean(self, text):
        from urllib.parse import quote_plus

        assert self.SECRET not in text
        assert quote_plus(self.SECRET) not in text
        assert "IDENT-abc" not in text

    def test_retry_warnings_and_final_error_carry_no_credentials(self, caplog):
        error, hits = self._run_against(503, caplog)
        assert hits == 5  # 503 is transient: the full ladder runs
        self._assert_clean(caplog.text)
        self._assert_clean(str(error))
        cause = error.__cause__
        assert cause is not None and cause.__suppress_context__
        self._assert_clean(str(cause))
        assert "HTTP 503" in str(error)

    def test_permanent_status_is_not_retried(self, caplog):
        error, hits = self._run_against(403, caplog)
        assert hits == 1
        self._assert_clean(str(error))

    def test_redact_secrets_masks_query_and_literal_forms(self):
        from iwac_common.omeka_client import redact_secrets

        text = (
            "url='https://x/api/items?page=1&key_identity=abc&key_credential=s3cr%2Bt'"
            " raw s3cr+t"
        )
        out = redact_secrets(text, ["s3cr+t"])
        assert "abc" not in out and "s3cr" not in out
        assert "key_credential=***" in out and "page=1" in out
