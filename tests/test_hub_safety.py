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


def test_columnar_verification_avoids_a_full_reload(monkeypatch, lock_dir):
    ds = FakeDataset({"o:id": ["1", "2"], "title": ["a", "b"]})
    revisions = iter(["before", "after", "after"])
    monkeypatch.setattr(hub, "get_repo_revision", lambda *a, **k: next(revisions))
    monkeypatch.setattr(card_sync, "sync_card_features", lambda *a, **k: True)
    monkeypatch.setattr(
        hub, "_published_ids_columnar", lambda *a, **k: ["1", "2"]
    )

    import datasets

    def forbidden(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("the columnar path must not trigger a full reload")

    monkeypatch.setattr(datasets, "load_dataset", forbidden)
    hub.push_dataset_verified(
        ds,
        repo_id="owner/repo",
        config_name="articles",
        token="token",
        commit_message="test",
        expected_revision="before",
    )
    assert len(ds.pushes) == 1


def test_columnar_verification_catches_a_short_write(monkeypatch, lock_dir):
    ds = FakeDataset({"o:id": ["1", "2"], "title": ["a", "b"]})
    revisions = iter(["before", "after", "after"])
    monkeypatch.setattr(hub, "get_repo_revision", lambda *a, **k: next(revisions))
    monkeypatch.setattr(card_sync, "sync_card_features", lambda *a, **k: True)
    monkeypatch.setattr(hub, "_published_ids_columnar", lambda *a, **k: ["1"])
    with pytest.raises(hub.HubWriteError, match="id set differs"):
        hub.push_dataset_verified(
            ds,
            repo_id="owner/repo",
            config_name="articles",
            token="token",
            commit_message="test",
            expected_revision="before",
        )


def test_columnar_failure_falls_back_to_the_full_reload(monkeypatch, lock_dir):
    ds = FakeDataset({"o:id": ["1", "2"], "title": ["a", "b"]})
    revisions = iter(["before", "after", "after"])
    monkeypatch.setattr(hub, "get_repo_revision", lambda *a, **k: next(revisions))
    monkeypatch.setattr(card_sync, "sync_card_features", lambda *a, **k: True)

    def unavailable(*args, **kwargs):
        raise RuntimeError("HfFileSystem is unavailable")

    monkeypatch.setattr(hub, "_published_ids_columnar", unavailable)

    import datasets

    reloads = []

    def fake_load(*args, **kwargs):
        reloads.append(kwargs)
        return FakeDataset(ds.data)

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    hub.push_dataset_verified(
        ds,
        repo_id="owner/repo",
        config_name="articles",
        token="token",
        commit_message="test",
        expected_revision="before",
    )
    assert len(reloads) == 1, "verification must never be silently skipped"


def test_local_lock_blocks_overlapping_writer(monkeypatch, lock_dir):
    with hub.hub_write_lock("owner/repo"):
        with pytest.raises(hub.HubWriteLockedError):
            with hub.hub_write_lock("owner/repo"):
                pass


def test_lock_held_by_a_live_process_is_not_reclaimed(lock_dir):
    with hub.hub_write_lock("owner/repo"):
        # The outer lock records this very process, which is obviously alive.
        with pytest.raises(hub.HubWriteLockedError, match="still running"):
            with hub.hub_write_lock("owner/repo"):
                pass


def test_stale_lock_from_a_dead_local_process_is_reclaimed(monkeypatch, lock_dir):
    monkeypatch.setattr(hub, "_process_alive", lambda pid: False)
    root = hub._lock_root()
    root.mkdir(parents=True, exist_ok=True)
    import hashlib
    import socket

    slug = hashlib.sha256(b"owner/repo").hexdigest()[:16]
    path = root / f"{slug}.lock"
    path.write_text(
        f"repo=owner/repo\npid=999999\nhost={socket.gethostname()}\nstarted=x\n",
        encoding="utf-8",
    )
    with hub.hub_write_lock("owner/repo"):
        pass
    assert not path.exists()


def test_stale_lock_from_another_host_is_never_reclaimed(monkeypatch, lock_dir):
    monkeypatch.setattr(hub, "_process_alive", lambda pid: False)
    root = hub._lock_root()
    root.mkdir(parents=True, exist_ok=True)
    import hashlib

    slug = hashlib.sha256(b"owner/repo").hexdigest()[:16]
    path = root / f"{slug}.lock"
    path.write_text(
        "repo=owner/repo\npid=999999\nhost=some-other-machine\nstarted=x\n",
        encoding="utf-8",
    )
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
