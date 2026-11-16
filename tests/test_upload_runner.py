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
    # The token is resolved before the Omeka fetch; never fall into login().
    monkeypatch.setattr(ur, "resolve_hf_token", lambda *a, **k: "token")


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
        # references spans several Omeka classes, taken from the registry.
        from iwac_common.schema import SUBSETS

        assert _spec(config_name="references").resource_class_ids == SUBSETS["references"].resource_class_ids
        assert _run(_spec(config_name="references"), ["--dry-run", "--no-cache"]) == 0

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


def test_missing_hf_token_fails_before_the_omeka_fetch(monkeypatch):
    monkeypatch.setattr(ur, "resolve_hf_token", lambda *a, **k: None)

    def failed_login():
        raise RuntimeError("no terminal")

    async def forbidden(*args, **kwargs):
        raise AssertionError("must not fetch before the token is known")

    monkeypatch.setattr(ur, "login", failed_login)
    monkeypatch.setattr(ur.OmekaApiClient, "fetch_items", forbidden)
    assert _run(_spec(), ["--dry-run", "--no-cache"]) == 1


def _capture_panels(monkeypatch):
    panels = []
    real_panel = ur.Panel

    def panel(renderable, **kwargs):
        panels.append((str(renderable), kwargs.get("title")))
        return real_panel(renderable, **kwargs)

    monkeypatch.setattr(ur, "Panel", panel)
    return panels


@pytest.mark.parametrize("error, headline", [
    (lambda: ur.HubWriteError("Commit abc landed atomically, but verification failed"),
     "Hub safety guard"),
    (lambda: ur.ConcurrentHubWriteError("stale parent"), "Hub safety guard"),
    (lambda: ur.HubWriteLockedError("locked"), "Hub safety guard"),
    (lambda: RuntimeError("boom"), "Unexpected pipeline failure"),
])
def test_gateway_errors_reach_the_typed_handler(stub_omeka, stub_hub, monkeypatch,
                                                error, headline):
    stub_omeka({36: [{"o:id": 1, "title": "a"}]})
    stub_hub(pd.DataFrame({"o:id": ["1"], "title": ["a"]}))

    def push(*args, **kwargs):
        raise error()

    monkeypatch.setattr(ur, "push_dataset_verified", push)
    panels = _capture_panels(monkeypatch)
    assert _run(_spec(), ["--no-cache", "--repo", "scratch/full"]) == 1
    assert headline in panels[-1][0]


def test_duplicate_mapped_ids_are_reported_as_duplicates(stub_omeka, stub_hub, monkeypatch):
    stub_omeka({36: [{"o:id": 1, "title": "a"}, {"o:id": 1, "title": "b"}]})
    stub_hub(pd.DataFrame({"o:id": ["1"], "title": ["a"]}))
    panels = _capture_panels(monkeypatch)
    assert _run(_spec(), ["--dry-run", "--no-cache"]) == 1
    assert "DuplicateIdError" in panels[-1][0]


def test_registry_int_columns_are_cast_after_merge(stub_omeka, stub_hub, monkeypatch):
    stub_omeka({36: [{"o:id": 1, "title": "a"}, {"o:id": 2, "title": "b"}]})

    async def mapper(item, api):
        return {"o:id": item["o:id"], "pub_year": 1990 if item["o:id"] == 1 else None,
                "nb_pages": ""}

    stub_hub(pd.DataFrame({"o:id": ["1", "2"], "title": ["a", "b"]}))
    seen = {}

    def post_merge(df):
        seen["df"] = df
        return df

    assert _run(_spec(map_item=mapper, post_merge=post_merge), ["--dry-run", "--no-cache"]) == 0
    # post_merge runs before the cast; the cast mutates the same frame.
    assert str(seen["df"]["pub_year"].dtype) == "Int64"
    assert str(seen["df"]["nb_pages"].dtype) == "Int64"
    assert seen["df"]["nb_pages"].isna().all()


class TestSpecFromRegistry:
    def test_classes_and_int_columns_come_from_the_registry(self):
        from iwac_common.schema import SUBSETS

        for name, subset in SUBSETS.items():
            spec = _spec(config_name=name)
            assert spec.resource_class_ids == subset.resource_class_ids
            assert spec.int_columns == subset.int_columns

    def test_unknown_subset_is_refused(self):
        with pytest.raises(ValueError, match="Unknown IWAC subset"):
            _spec(config_name="nope")

    def test_removed_knobs_are_gone(self):
        import dataclasses

        fields = {f.name for f in dataclasses.fields(UploadSpec)}
        assert not fields & {"resource_class_ids", "int_columns", "merge_suffixes",
                             "extra_config_rows"}

    def test_every_upload_script_spec_matches_its_subset(self):
        import importlib

        from iwac_pipeline.cli import UPLOAD_MODULES

        for config, module_name in UPLOAD_MODULES.items():
            spec = importlib.import_module(module_name).SPEC
            assert spec.config_name == config
            assert spec.resource_class_ids  # derived, never restated


class TestCastIntColumns:
    def test_blank_and_null_become_null(self):
        df = pd.DataFrame({"n": pd.Series([1, None, "", "  ", 3.0], dtype=object)})
        out = ur.cast_int_columns(df, ["n", "absent"])
        assert out["n"].tolist()[0] == 1 and out["n"].tolist()[4] == 3
        assert out["n"].isna().tolist() == [False, True, True, True, False]

    def test_fractional_value_raises(self):
        with pytest.raises(TypeError):
            ur.cast_int_columns(pd.DataFrame({"n": [1.5, None]}), ["n"])


def test_unmapped_report_uses_runner_console_and_generic_hint():
    import asyncio
    import io
    from types import SimpleNamespace

    from rich.console import Console

    buffer = io.StringIO()
    api = SimpleNamespace(console=Console(file=buffer, width=200))
    df = pd.DataFrame({"publisher": ["Known", "Unknown"], "region": ["X", ""]})
    asyncio.run(ur.report_unmapped_values("publisher", "region")(df, api, "r", None))
    text = buffer.getvalue()
    assert "Unknown" in text
    assert "country_mapper" not in text
    assert "derives region from publisher" in text

    buffer.truncate(0)
    asyncio.run(ur.report_unmapped_values("publisher", "region", hint="Fix X.")(df, api, "r", None))
    assert "Fix X." in buffer.getvalue()
