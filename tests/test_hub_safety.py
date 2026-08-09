"""Contracts for the single Hugging Face write gateway."""

from __future__ import annotations

import pytest

import iwac_common.card_sync as card_sync
import iwac_common.hub as hub


class FakeDataset:
    def __init__(self, data):
        self.data = data
        self.column_names = list(data)
        self.pushes = []

    def __getitem__(self, column):
        return self.data[column]

    def push_to_hub(self, **kwargs):
        self.pushes.append(kwargs)


@pytest.fixture
def lock_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("IWAC_LOCK_DIR", str(tmp_path / "locks"))


def test_revision_change_aborts_before_push(monkeypatch, lock_dir):
    ds = FakeDataset({"o:id": ["1"], "title": ["a"]})
    monkeypatch.setattr(hub, "get_repo_revision", lambda *a, **k: "new")
    with pytest.raises(hub.ConcurrentHubWriteError):
        hub.push_dataset_verified(
            ds,
            repo_id="owner/repo",
            config_name="articles",
            token="token",
            commit_message="test",
            expected_revision="old",
        )
    assert ds.pushes == []


def test_push_repairs_card_and_reload_verifies(monkeypatch, lock_dir):
    ds = FakeDataset({"o:id": ["1", "2"], "title": ["a", "b"]})
    revisions = iter(["before", "after", "after"])
    monkeypatch.setattr(hub, "get_repo_revision", lambda *a, **k: next(revisions))
    monkeypatch.setattr(card_sync, "sync_card_features", lambda *a, **k: True)

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: FakeDataset(ds.data))
    result = hub.push_dataset_verified(
        ds,
        repo_id="owner/repo",
        config_name="articles",
        token="token",
        commit_message="test",
        expected_revision="before",
    )
    assert len(ds.pushes) == 1
    assert result.before_revision == "before"
    assert result.after_revision == "after"


def test_bad_embedding_dimension_is_rejected(monkeypatch, lock_dir):
    ds = FakeDataset({"o:id": ["1"], "embedding_OCR": [[0.1] * 3]})
    with pytest.raises(hub.HubWriteError, match="dimension 3"):
        hub.push_dataset_verified(
            ds,
            repo_id="owner/repo",
            config_name="articles",
            token="token",
            commit_message="test",
        )


def test_local_lock_blocks_overlapping_writer(monkeypatch, lock_dir):
    with hub.hub_write_lock("owner/repo"):
        with pytest.raises(hub.HubWriteLockedError):
            with hub.hub_write_lock("owner/repo"):
                pass


def test_config_discovery_includes_parquet_missing_from_card(monkeypatch):
    class Info:
        config_names = ["articles"]

    class Api:
        def __init__(self, *args, **kwargs):
            pass

        def dataset_info(self, **kwargs):
            return Info()

        def list_repo_files(self, **kwargs):
            return ["README.md", "references/train-00000-of-00001.parquet"]

    monkeypatch.setattr(hub, "HfApi", Api)
    assert hub.get_repo_configs("owner/repo", token="token") == {
        "articles", "references"
    }
