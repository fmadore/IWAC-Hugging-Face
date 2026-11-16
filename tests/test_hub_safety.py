"""Behavioral contracts for atomic, privacy-aware Hugging Face writes."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset
from huggingface_hub import CommitOperationAdd, DatasetCard

import iwac_common.card_sync as card_sync
import iwac_common.hub as hub


@pytest.fixture
def atomic_hub(monkeypatch, tmp_path):
    monkeypatch.setenv("IWAC_LOCK_DIR", str(tmp_path / "locks"))
    state = {"head": "before", "private": True, "files": {}, "commits": [], "conflict": False}

    class Api:
        def __init__(self, **kwargs):
            pass

        def dataset_info(self, **kwargs):
            return SimpleNamespace(sha=state["head"], private=state["private"])

        def list_repo_files(self, **kwargs):
            assert kwargs["revision"] == "before"
            return list(state["files"])

        def create_commit(self, **kwargs):
            assert kwargs["parent_commit"] == "before"
            if state["conflict"]:
                error = RuntimeError("parent commit mismatch")
                error.response = SimpleNamespace(status_code=409)
                raise error
            contents = dict(state["files"])
            for operation in kwargs["operations"]:
                if isinstance(operation, CommitOperationAdd):
                    value = operation.path_or_fileobj
                    contents[operation.path_in_repo] = value if isinstance(value, bytes) else Path(value).read_bytes()
                else:
                    contents.pop(operation.path_in_repo, None)
            state["files"] = contents
            state["commits"].append(kwargs)
            state["head"] = "after"
            return SimpleNamespace(oid="after")

    def read_schema(repo, config, token, revision=None):
        assert revision == "after"
        schemas = [pq.read_schema(pa.BufferReader(data)) for path, data in state["files"].items()
                   if path.startswith(config + "/") and path.endswith(".parquet")]
        assert schemas
        assert all(schema.equals(schemas[0], check_metadata=False) for schema in schemas)
        return card_sync._normalize_schema(schemas[0])

    def read_card(repo, token, revision):
        assert revision == state["head"]
        return DatasetCard(state["files"]["README.md"].decode())

    def read_columnar(repo, config, revision, token):
        values = []
        for path, data in sorted(state["files"].items()):
            if path.startswith(config + "/") and path.endswith(".parquet"):
                values.extend(pq.read_table(pa.BufferReader(data), columns=["o:id"])["o:id"].to_pylist())
        return read_schema(repo, config, token, revision), values

    monkeypatch.setattr(hub, "HfApi", Api)
    monkeypatch.setattr(card_sync, "_parquet_schema", read_schema)
    monkeypatch.setattr(card_sync, "_load_card", read_card)
    monkeypatch.setattr(hub, "_published_columnar", read_columnar)
    return state


def sample():
    return Dataset.from_dict({"o:id": ["1", "2"], "title": ["a", "b"]})


def push(ds=None, **kwargs):
    return hub.push_dataset_verified(
        ds if ds is not None else sample(), repo_id=kwargs.pop("repo_id", "owner/repo"),
        config_name="articles", token="token", commit_message="test", **kwargs,
    )


def test_revision_change_aborts_before_commit(atomic_hub):
    with pytest.raises(hub.ConcurrentHubWriteError):
        push(expected_revision="old")
    assert atomic_hub["commits"] == []


def test_atomic_commit_has_data_and_exact_card(atomic_hub):
    result = push(expected_revision="before")
    assert len(atomic_hub["commits"]) == 1
    paths = set(atomic_hub["files"])
    assert paths == {"README.md", "articles/train-00000-of-00001.parquet"}
    assert (result.before_revision, result.after_revision) == ("before", "after")
    card = DatasetCard(atomic_hub["files"]["README.md"].decode())
    assert card.data["dataset_info"][0]["splits"][0]["num_examples"] == 2
    assert card.data["configs"][0]["data_files"][0]["path"] == "articles/train-*.parquet"


def test_all_subsets_in_one_commit_and_old_shards_removed(atomic_hub):
    atomic_hub["files"]["articles/train-00001-of-00002.parquet"] = b"old shard"
    atomic_hub["files"]["unrelated/note.txt"] = b"keep"
    hub.push_datasets_verified(
        {"articles": sample(), "images": sample()}, repo_id="owner/repo", token="t",
        commit_message="batch", expected_revision="before",
    )
    assert len(atomic_hub["commits"]) == 1
    assert "articles/train-00001-of-00002.parquet" not in atomic_hub["files"]
    assert "images/train-00000-of-00001.parquet" in atomic_hub["files"]
    assert atomic_hub["files"]["unrelated/note.txt"] == b"keep"


def test_invalid_second_subset_never_commits_first(atomic_hub):
    broken = Dataset.from_dict({"o:id": ["1"], "embedding_image": [[1.0]]})
    with pytest.raises(hub.HubWriteError, match="dimension"):
        hub.push_datasets_verified(
            {"articles": sample(), "images": broken}, repo_id="owner/repo", token="t", commit_message="batch",
        )
    assert atomic_hub["commits"] == []


def test_server_parent_conflict_never_changes_repository(atomic_hub):
    atomic_hub["conflict"] = True
    with pytest.raises(hub.ConcurrentHubWriteError, match="stale parent"):
        push()
    assert atomic_hub["files"] == {}
    assert atomic_hub["commits"] == []


def test_raw_write_to_any_public_repo_is_refused(atomic_hub):
    atomic_hub["private"] = False
    with pytest.raises(hub.HubWriteError, match="verified private"):
        push(repo_id="owner/public-scratch")
    assert not atomic_hub["commits"]


def test_raw_write_to_known_public_destination_even_when_private_is_refused(atomic_hub):
    with pytest.raises(hub.HubWriteError, match="verified private"):
        push(repo_id="fmadore/islam-west-africa-collection")


def test_unknown_destination_privacy_is_refused(atomic_hub):
    atomic_hub["private"] = None
    with pytest.raises(hub.HubWriteError, match="verified private"):
        push()


def test_public_mode_requires_masked_projection(atomic_hub):
    ds = Dataset.from_dict({"o:id": ["1"], "item_is_public": [True], "private_fields": [[]],
                            "OCR_is_public": [False], "OCR": ["private text"]})
    with pytest.raises(hub.HubWriteError, match="Private value"):
        push(ds, mode="public_projection", repo_id="owner/public")
    assert not atomic_hub["commits"]


def test_explicit_public_projection_is_revalidated_and_committed(atomic_hub):
    atomic_hub["private"] = False
    ds = Dataset.from_dict({"o:id": ["1"], "item_is_public": [True], "private_fields": [[]],
                            "OCR_is_public": [False], "OCR": [""]})
    push(ds, mode="public_projection", repo_id="owner/public")
    assert len(atomic_hub["commits"]) == 1


def test_columnar_verification_catches_short_write(monkeypatch, atomic_hub):
    original = hub._published_columnar

    def short(*args, **kwargs):
        schema, _ = original(*args, **kwargs)
        return schema, ["1"]

    monkeypatch.setattr(hub, "_published_columnar", short)
    with pytest.raises(hub.HubWriteError, match="landed atomically.*id set differs"):
        push()


def test_columnar_failure_falls_back_to_pinned_full_reload(monkeypatch, atomic_hub):
    import datasets
    calls = []

    def fail(*args, **kwargs):
        raise RuntimeError("range read unavailable")

    def load(*args, **kwargs):
        calls.append(kwargs)
        return sample()

    monkeypatch.setattr(hub, "_published_columnar", fail)
    monkeypatch.setattr(datasets, "load_dataset", load)
    push()
    assert calls[0]["revision"] == "after"


def test_one_open_per_shard_serves_footer_and_ids(monkeypatch):
    """The fast path must not open each shard twice (once for the card check,
    once for the ids)."""
    import io

    from rich.console import Console

    monkeypatch.setattr(card_sync, "resolve_hf_token", lambda *a, **k: "t")
    opened = []
    bodies = {}
    for index, ids in enumerate((["1"], ["2"])):
        out = io.BytesIO()
        pq.write_table(pa.table({"o:id": ids, "title": ["x"]}), out)
        bodies[f"shard-{index}"] = out.getvalue()

    class Files:
        def __init__(self, **kwargs):
            pass

        def glob(self, pattern):
            assert "owner/repo@rev/articles" in pattern
            return sorted(bodies)

        def open(self, path, mode):
            opened.append(path)
            return io.BytesIO(bodies[path])

    monkeypatch.setattr(hub, "HfFileSystem", Files)
    card = DatasetCard("---\n{}\n---\n")
    monkeypatch.setattr(card_sync, "_load_card", lambda *a, **k: card)
    ds = sample()
    hub._verify_committed_dataset(
        ds, repo_id="owner/repo", config_name="articles", token="t",
        revision="rev", console=Console(),
    )
    assert opened == ["shard-0", "shard-1"]


def test_iter_parquet_shards_raises_the_requested_error_when_empty():
    import fsspec

    with pytest.raises(hub.HubWriteError, match="No parquet found"):
        list(hub.iter_parquet_shards("owner/none", "articles", revision="r",
                                     fs=fsspec.filesystem("memory"), error=hub.HubWriteError))


def test_gateway_has_no_lock_or_verification_bypass():
    import inspect

    for function in (hub.push_dataset_verified, hub.push_datasets_verified):
        parameters = inspect.signature(function).parameters
        assert "verify_reload" not in parameters
        assert "acquire_lock" not in parameters


def test_gateway_conforms_types_before_staging(atomic_hub):
    ds = Dataset(pa.table({"o:id": ["1", "2"], "lda_topic_id": [4.0, None],
                           "embedding_OCR": pa.array([[0.25] * 768, None], pa.list_(pa.float64()))}))
    push(ds)
    raw = atomic_hub["files"]["articles/train-00000-of-00001.parquet"]
    schema = pq.read_schema(pa.BufferReader(raw))
    assert schema.field("lda_topic_id").type == pa.int64()
    assert schema.field("embedding_OCR").type.value_type == pa.float32()


def test_fractional_integer_refused_before_write(atomic_hub):
    ds = Dataset.from_dict({"o:id": ["1"], "nb_pages": [2.5]})
    with pytest.raises(hub.HubWriteError, match="fractional"):
        push(ds)
    assert not atomic_hub["commits"]


def test_replacing_card_config_preserves_siblings_and_prose(atomic_hub):
    atomic_hub["files"]["README.md"] = b"---\nlicense: cc-by-4.0\ndataset_info:\n- config_name: index\n  features:\n  - name: o:id\n    dtype: string\nconfigs:\n- config_name: index\n  data_files:\n  - split: train\n    path: index/*.parquet\n---\n\nScholarship remains.\n"
    push()
    card = DatasetCard(atomic_hub["files"]["README.md"].decode())
    assert card.data["dataset_info"][0]["config_name"] == "index"
    assert card.data["configs"][0]["config_name"] == "index"
    assert "Scholarship remains." in card.text
    assert card.data["license"] == "cc-by-4.0"


def test_empty_subset_keeps_typed_parquet(atomic_hub):
    push(sample().select([]))
    raw = atomic_hub["files"]["articles/train-00000-of-00001.parquet"]
    assert pq.read_table(pa.BufferReader(raw)).num_rows == 0


def test_small_shards_are_all_verified(atomic_hub):
    push(max_shard_size="1B")
    assert sum(p.endswith(".parquet") for p in atomic_hub["files"]) == 2


def test_local_lock_blocks_overlapping_writer(atomic_hub):
    with hub.hub_write_lock("owner/repo"):
        with pytest.raises(hub.HubWriteLockedError):
            with hub.hub_write_lock("owner/repo"):
                pass


def test_stale_lock_from_dead_local_process_is_reclaimed(monkeypatch, atomic_hub):
    import hashlib
    import socket
    monkeypatch.setattr(hub, "_process_alive", lambda pid: False)
    root = hub._lock_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / (hashlib.sha256(b"owner/repo").hexdigest()[:16] + ".lock")
    path.write_text(f"repo=owner/repo\npid=999999\nhost={socket.gethostname()}\nstarted=x\n")
    with hub.hub_write_lock("owner/repo"):
        pass
    assert not path.exists()


def test_reclaim_race_loser_restores_the_winners_fresh_lock(monkeypatch, atomic_hub):
    """Two processes read the same dead owner. The winner reclaims it and
    writes its own lock; the loser, acting on its stale read, must not delete
    that fresh lock and must not acquire."""
    import hashlib
    import os
    import socket

    host = socket.gethostname()
    root = hub._lock_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / (hashlib.sha256(b"owner/repo").hexdigest()[:16] + ".lock")
    dead = {"repo": "owner/repo", "pid": "999999", "host": host, "started": "x"}
    winner = f"repo=owner/repo\npid={os.getpid()}\nhost={host}\nstarted=fresh\n"
    path.write_text(winner)  # the winner already reclaimed and holds the lock

    real_owner = hub._lock_owner
    reads = iter([dead])  # the loser's first read predates the winner's reclaim
    monkeypatch.setattr(hub, "_lock_owner", lambda p: next(reads, None) or real_owner(p))
    monkeypatch.setattr(hub, "_process_alive", lambda pid: pid != 999999)

    assert hub._reclaim_if_dead(path, "owner/repo") is False
    assert path.read_text() == winner
    assert list(root.glob("*.stale")) == []

    reads = iter([dead])
    with pytest.raises(hub.HubWriteLockedError):
        with hub.hub_write_lock("owner/repo"):
            pass
    assert path.read_text() == winner


def test_reclaim_race_with_occupied_slot_fails_closed(monkeypatch, atomic_hub):
    import hashlib
    import os
    import socket

    host = socket.gethostname()
    root = hub._lock_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / (hashlib.sha256(b"owner/repo").hexdigest()[:16] + ".lock")
    dead = {"repo": "owner/repo", "pid": "999999", "host": host, "started": "x"}
    path.write_text(f"repo=owner/repo\npid={os.getpid()}\nhost={host}\nstarted=fresh\n")
    monkeypatch.setattr(hub, "_lock_owner", lambda p: dead if p == path else {})
    monkeypatch.setattr(hub, "_process_alive", lambda pid: pid != 999999)

    def occupied(src, dst):
        raise FileExistsError(dst)

    monkeypatch.setattr(hub.os, "link", occupied)
    with pytest.raises(hub.HubWriteLockedError, match="reclaim race"):
        hub._reclaim_if_dead(path, "owner/repo")


def test_stale_lock_from_another_host_is_not_reclaimed(monkeypatch, atomic_hub):
    import hashlib
    monkeypatch.setattr(hub, "_process_alive", lambda pid: False)
    root = hub._lock_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / (hashlib.sha256(b"owner/repo").hexdigest()[:16] + ".lock")
    path.write_text("repo=owner/repo\npid=999999\nhost=elsewhere\n")
    with pytest.raises(hub.HubWriteLockedError):
        with hub.hub_write_lock("owner/repo"):
            pass


def test_config_discovery_includes_parquet_missing_from_card(monkeypatch):
    class Api:
        def __init__(self, **kwargs):
            pass
        def dataset_info(self, **kwargs):
            return SimpleNamespace(config_names=["articles"])
        def list_repo_files(self, **kwargs):
            return ["README.md", "references/train-00000-of-00001.parquet"]
    monkeypatch.setattr(hub, "HfApi", Api)
    assert hub.get_repo_configs("owner/repo", token="t") == {"articles", "references"}


def test_committed_layout_loads_with_real_datasets_reader(atomic_hub, tmp_path):
    import datasets

    hub.push_datasets_verified(
        {"articles": sample(), "images": sample()}, repo_id="owner/repo", token="t", commit_message="batch",
    )
    directory = tmp_path / "snapshot"
    for relative, value in atomic_hub["files"].items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    for config in ("articles", "images"):
        loaded = datasets.load_dataset(str(directory), name=config, split="train")
        assert list(loaded["o:id"]) == ["1", "2"]
        assert loaded.features == sample().features


def test_default_lock_directory_uses_workspace_not_installed_package(monkeypatch, tmp_path):
    import iwac_common.paths as paths
    monkeypatch.delenv("IWAC_LOCK_DIR", raising=False)
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    assert hub._lock_root() == tmp_path / ".iwac_locks"
    monkeypatch.setenv("IWAC_LOCK_DIR", str(tmp_path / "explicit"))
    assert hub._lock_root() == tmp_path / "explicit"


def test_worklist_lock_can_use_shared_state_directory(atomic_hub, tmp_path):
    shared_state = tmp_path / "shared-state-locks"
    with hub.hub_write_lock("ingest-state::owner/repo::articles", root_dir=shared_state):
        assert len(list(shared_state.glob("*.lock"))) == 1
        with pytest.raises(hub.HubWriteLockedError):
            with hub.hub_write_lock("ingest-state::owner/repo::articles", root_dir=shared_state):
                pass
    assert not list(shared_state.glob("*.lock"))
