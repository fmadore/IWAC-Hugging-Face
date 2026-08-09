"""Fetch-truncation reconciliation (safety rail B1) — no network involved."""

import asyncio
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

    def test_verify_can_be_disabled(self):
        pages = [_items(10)]
        client = StubClient(pages, total=999)
        items = asyncio.run(client.fetch_items(36, verify_total=False))
        assert len(items) == 10


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
