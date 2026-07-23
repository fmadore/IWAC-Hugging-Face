"""End-to-end test of the shared upload orchestration in --dry-run mode.

Stubs the Omeka fetch and the Hub load so nothing hits the network; exercises
fetch -> map -> post_map -> merge (with the real safety rails) -> dry-run.
"""

import asyncio

import pandas as pd
import pytest

import iwac_common.upload_runner as ur
import iwac_common.hub_merge as hub_merge
from iwac_common.upload_runner import UploadSpec, build_parser


class _FakeDS:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


@pytest.fixture
def stub_omeka(monkeypatch):
    """Make OmekaApiClient.fetch_items return canned items, no network."""

    def _install(items_by_class):
        async def fake_fetch(self, rcid, verify_total=True):
            return items_by_class.get(rcid, [])

        monkeypatch.setattr(ur.OmekaApiClient, "fetch_items", fake_fetch)

    return _install


@pytest.fixture
def stub_hub(monkeypatch):
    """Serve a canned existing Hub frame to the merge helper."""

    def _install(existing_df):
        monkeypatch.setattr(hub_merge, "load_dataset", lambda *a, **k: _FakeDS(existing_df))

    return _install


async def _map(item, api):
    return {"o:id": item["o:id"], "title": item["title"].upper()}


def _spec(**kw):
    base = dict(
        config_name="articles",
        resource_class_ids=(36,),
        map_item=_map,
        title="Test Upload",
        cache_dir=".cache_test",
    )
    base.update(kw)
    return UploadSpec(**base)


def _run(spec, argv):
    return ur.run_upload(spec, argv)


class TestDryRun:
    def test_happy_path_dry_run_returns_zero(self, stub_omeka, stub_hub, monkeypatch):
        stub_omeka({36: [{"o:id": 1, "title": "a"}, {"o:id": 2, "title": "b"}]})
        stub_hub(pd.DataFrame({"o:id": ["1", "2"], "title": ["a", "b"], "embedding_OCR": [[0.1], [0.2]]}))
        # no HF token needed in dry-run; disable cache to avoid disk writes
        assert _run(_spec(), ["--dry-run", "--no-cache"]) == 0

    def test_empty_omeka_leaves_hub_untouched(self, stub_omeka, monkeypatch):
        stub_omeka({36: []})
        assert _run(_spec(), ["--dry-run", "--no-cache"]) == 0  # warns, no push, exit 0

    def test_truncated_fetch_aborts_nonzero(self, stub_omeka, stub_hub, monkeypatch):
        # fetch reports 1 item but verify_total will see a mismatch → we simulate
        # by having the stubbed fetch raise TruncatedFetchError.
        async def boom(self, rcid, verify_total=True):
            raise ur.TruncatedFetchError("simulated truncation")

        monkeypatch.setattr(ur.OmekaApiClient, "fetch_items", boom)
        assert _run(_spec(), ["--dry-run", "--no-cache"]) == 1

    def test_shrink_guard_aborts_nonzero(self, stub_omeka, stub_hub):
        stub_omeka({36: [{"o:id": 1, "title": "a"}]})  # 1 fresh row
        stub_hub(pd.DataFrame({"o:id": [str(i) for i in range(100)], "title": ["x"] * 100}))
        assert _run(_spec(), ["--dry-run", "--no-cache"]) == 1  # 1 << 100

    def test_force_shrink_overrides(self, stub_omeka, stub_hub):
        stub_omeka({36: [{"o:id": 1, "title": "a"}]})
        stub_hub(pd.DataFrame({"o:id": [str(i) for i in range(100)], "title": ["x"] * 100}))
        assert _run(_spec(), ["--dry-run", "--no-cache", "--force-shrink"]) == 0

    def test_post_map_hook_runs(self, stub_omeka, stub_hub):
        stub_omeka({36: [{"o:id": 1, "title": "a"}]})
        stub_hub(pd.DataFrame())
        seen = {}

        async def post_map(df, api, repo, token):
            seen["cols"] = list(df.columns)
            seen["repo"] = repo
            df["extra"] = 1
            return df

        assert _run(_spec(post_map=post_map), ["--dry-run", "--no-cache", "--repo", "scratch/x"]) == 0
        assert seen["repo"] == "scratch/x"
        assert "o:id" in seen["cols"]

    def test_multi_class_fetch_concatenates(self, stub_omeka, stub_hub):
        stub_omeka({35: [{"o:id": 1, "title": "a"}], 43: [{"o:id": 2, "title": "b"}]})
        stub_hub(pd.DataFrame())
        assert _run(_spec(resource_class_ids=(35, 43)), ["--dry-run", "--no-cache"]) == 0


class TestParser:
    def test_stale_rows_flag_only_when_enabled(self):
        assert "--stale-rows" not in build_parser(_spec()).format_help()
        assert "--stale-rows" in build_parser(_spec(supports_stale_rows=True)).format_help()

    def test_standard_flags_present(self):
        help_text = build_parser(_spec()).format_help()
        for flag in ("--repo", "--max-shard-size", "--no-cache", "--dry-run", "--force-shrink"):
            assert flag in help_text
