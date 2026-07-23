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
