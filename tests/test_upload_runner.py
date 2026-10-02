"""End-to-end test of the shared upload orchestration in --dry-run mode.

Stubs the Omeka fetch and the Hub load so nothing hits the network; exercises
fetch -> map -> post_map -> merge (with the real safety rails) -> dry-run.
"""


import pandas as pd
import pytest

import iwac_common.upload_runner as ur
import iwac_common.hub_merge as hub_merge
from iwac_common.upload_runner import UploadSpec, build_parser


@pytest.fixture(autouse=True)
def source_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("OMEKA_KEY_IDENTITY", "synthetic-identity")
    monkeypatch.setenv("OMEKA_KEY_CREDENTIAL", "synthetic-credential")
    monkeypatch.setenv("IWAC_STATE_DIR", str(tmp_path / "state"))


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
        monkeypatch.setattr(hub_merge, "get_repo_revision", lambda *a, **k: "rev-1")
        monkeypatch.setattr(ur, "resolve_hf_token", lambda *a, **k: "token")

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
        stub_hub(pd.DataFrame({
            "o:id": ["1", "2"],
            "title": ["a", "b"],
            "embedding_OCR": [[0.1] * 768, [0.2] * 768],
        }))
        # Hub baseline reads are authenticated even in dry-run.
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

    def test_mapper_failure_aborts_by_default(self, stub_omeka, stub_hub):
        stub_omeka({36: [{"o:id": 1, "title": "a"}, {"o:id": 2, "title": "b"}]})
        stub_hub(pd.DataFrame({"o:id": ["1", "2"], "title": ["old-a", "old-b"]}))

        async def sometimes_fails(item, api):
            if item["o:id"] == 2:
                raise ValueError("broken field")
            return {"o:id": item["o:id"], "title": item["title"].upper()}

        assert _run(_spec(map_item=sometimes_fails), ["--dry-run", "--no-cache"]) == 1

    def test_allowed_mapper_failure_preserves_hub_row(self, stub_omeka, stub_hub):
        stub_omeka({36: [{"o:id": 1, "title": "a"}, {"o:id": 2, "title": "b"}]})
        stub_hub(pd.DataFrame({"o:id": ["1", "2"], "title": ["old-a", "old-b"]}))

        async def sometimes_fails(item, api):
            if item["o:id"] == 2:
                raise ValueError("broken field")
            return {"o:id": item["o:id"], "title": item["title"].upper()}

        assert _run(
            _spec(map_item=sometimes_fails),
            ["--dry-run", "--no-cache", "--allow-map-failures"],
        ) == 0


class TestParser:
    def test_stale_rows_flag_only_when_enabled(self):
        assert "--stale-rows" not in build_parser(_spec()).format_help()
        assert "--stale-rows" in build_parser(_spec(supports_stale_rows=True)).format_help()

    def test_standard_flags_present(self):
        help_text = build_parser(_spec()).format_help()
        for flag in ("--repo", "--max-shard-size", "--no-cache", "--dry-run", "--force-shrink"):
            assert flag in help_text

    def test_invalidation_is_default_and_preservation_explicit(self):
        parser = build_parser(_spec())
        assert parser.parse_args([]).invalidate_derived is True
        assert parser.parse_args(["--preserve-derived"]).invalidate_derived is False
        assert build_parser(_spec(supports_stale_rows=True)).parse_args([]).stale_rows == "drop"


def test_missing_source_credentials_abort_before_fetch(monkeypatch):
    monkeypatch.delenv("OMEKA_KEY_CREDENTIAL")

    async def forbidden(*args, **kwargs):
        raise AssertionError("must not fetch anonymous data")

    monkeypatch.setattr(ur.OmekaApiClient, "fetch_items", forbidden)
    assert _run(_spec(), ["--dry-run", "--no-cache"]) == 1


def test_preserved_stale_worklist_is_staged_before_write_and_replayed(
    stub_omeka, stub_hub, monkeypatch,
):
    stub_omeka({36: [{"o:id": 1, "title": "a"}]})
    baseline = pd.DataFrame({"o:id": ["1"], "OCR": ["before"], "lemma_text": ["old"]})
    stub_hub(baseline)

    async def mapper(item, api):
        return {"o:id": item["o:id"], "OCR": "after"}

    writes = []

    def push(ds, **kwargs):
        assert ur.load_stale_derived(kwargs["repo_id"], "articles") == {"lemma_text": ["1"]}
        writes.append(ds.to_pandas())

    monkeypatch.setattr(ur, "push_dataset_verified", push)
    argv = ["--no-cache", "--repo", "scratch/full"]
    assert _run(_spec(map_item=mapper), [*argv, "--preserve-derived"]) == 0
    assert writes[-1].iloc[0]["lemma_text"] == "old"
    # The source change is already in the Hub. Comparison alone sees nothing;
    # durable recovery must still clear the stale enrichment on a later run.
    stub_hub(writes[-1])
    assert _run(_spec(map_item=mapper), argv) == 0
    assert pd.isna(writes[-1].iloc[0]["lemma_text"])
    assert not ur.stale_worklist_path("scratch/full", "articles").exists()


def test_failed_push_does_not_erase_stale_worklist(stub_omeka, stub_hub, monkeypatch):
    stub_omeka({36: [{"o:id": 1, "title": "a"}]})
    stub_hub(pd.DataFrame({"o:id": ["1"], "OCR": ["old"], "lemma_text": ["old"]}))

    async def mapper(item, api):
        return {"o:id": 1, "OCR": "changed"}

    def push(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(ur, "push_dataset_verified", push)
    assert _run(_spec(map_item=mapper), ["--no-cache", "--repo", "scratch/full"]) == 1
    assert ur.load_stale_derived("scratch/full", "articles") == {"lemma_text": ["1"]}


def test_corrupt_worklist_fails_closed(monkeypatch, tmp_path):
    path = ur.stale_worklist_path("scratch/full", "articles")
    path.parent.mkdir(parents=True)
    path.write_text('{"repository":"different","config":"articles","columns":{}}')
    with pytest.raises(ValueError, match="identity mismatch"):
        ur.load_stale_derived("scratch/full", "articles")


def test_another_ingest_cannot_mutate_same_recovery_queue(monkeypatch):
    ur.record_stale_derived("scratch/full", "articles", {
        "OCR": {"ids": ["1"], "derived": ["lemma_text"]},
    }, "old-revision")
    path = ur.stale_worklist_path("scratch/full", "articles")
    snapshot = path.read_bytes()
    with ur.hub_write_lock("ingest-state::scratch/full::articles", root_dir=ur._state_root() / "locks"):
        assert _run(_spec(), ["--no-cache", "--repo", "scratch/full"]) == 1
    assert path.read_bytes() == snapshot


@pytest.mark.parametrize("absolute", [False, True])
def test_cache_respects_workspace_and_explicit_absolute_path(monkeypatch, tmp_path, absolute):
    workspace = tmp_path / "research"
    monkeypatch.setenv("IWAC_WORK_DIR", str(workspace))
    cache = tmp_path / "separate-cache" if absolute else ".cache_subset"
    seen = []

    async def empty_fetch(self, rcid, verify_total=True):
        seen.append(self.cfg.CACHE_DIR)
        return []

    monkeypatch.setattr(ur.OmekaApiClient, "fetch_items", empty_fetch)
    assert _run(_spec(cache_dir=str(cache)), ["--dry-run", "--no-cache"]) == 0
    assert seen == [str(cache if absolute else workspace / cache)]
