"""Fetch-truncation reconciliation (safety rail B1) — no network involved."""

import asyncio

import pytest

from iwac_common.omeka_client import Config, OmekaApiClient, TruncatedFetchError


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

    def test_unknown_total_skips_check(self):
        pages = [_items(42)]
        client = StubClient(pages, total=None)
        items = asyncio.run(client.fetch_items(36))
        assert len(items) == 42

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

    def test_a_few_bad_manifests_pass(self):
        """5/500 is ordinary attrition, not an outage — must not abort."""
        assert "5/500" in self._stats(500, 5).check().replace(",", "")

    def test_small_subset_never_trips(self):
        """Below MIN_ATTEMPTS the rate is too noisy: 3/3 on a tiny subset
        must report but not abort, or `documents` (26 rows) would be
        unrunnable whenever a couple of manifests are missing."""
        assert self._stats(3, 3).check() is not None

    def test_threshold_boundary(self):
        from iwac_common.omeka_client import MediaFetchGuardError, MediaFetchStats

        n = MediaFetchStats.MIN_ATTEMPTS * 10
        at_limit = int(n * MediaFetchStats.MAX_FAILURE_RATE)
        self._stats(n, at_limit).check()          # exactly at threshold: allowed
        with pytest.raises(MediaFetchGuardError):
            self._stats(n, at_limit + 1).check()  # one over: aborts

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
