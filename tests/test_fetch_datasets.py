"""The local mirror downloader's CLI, token and failure reporting (offline)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def mirror():
    spec = importlib.util.spec_from_file_location("_mirror_cli_ut", ROOT / "data" / "fetch_datasets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_private_mirror_uses_the_shared_token_resolver(mirror, monkeypatch):
    monkeypatch.setattr(mirror, "resolve_hf_token", lambda *a, **k: None)

    def unexpected(*args, **kwargs):
        pytest.fail("no Hub read without a token")

    monkeypatch.setattr(mirror, "get_repo_revision", unexpected)
    assert mirror.main(dataset_id="owner/repo", label="private") == 1


def test_default_repository_is_resolved_lazily(mirror, monkeypatch):
    monkeypatch.setenv("IWAC_HF_PRIVATE_REPO", "scratch/private")
    monkeypatch.setenv("IWAC_HF_PUBLIC_REPO", "scratch/public")
    assert mirror.choose_dataset("private") == ("scratch/private", "private")
    assert mirror.choose_dataset("public") == ("scratch/public", "public")
    seen = []
    monkeypatch.setattr(mirror, "resolve_hf_token", lambda *a, **k: "t")
    monkeypatch.setattr(mirror, "get_repo_revision",
                        lambda repo, **k: seen.append(repo) or (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError):
        mirror.main()
    assert seen == ["scratch/private"]


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_abort_message_names_the_actual_format(mirror, monkeypatch, tmp_path, capsys, fmt):
    from rich.console import Console

    monkeypatch.setattr(mirror, "console", Console(width=400))
    monkeypatch.setattr(mirror, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mirror, "get_repo_revision", lambda *a, **k: "rev")

    def fail(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(mirror, "load_dataset", fail)
    assert mirror.main(dataset_id="owner/repo", label="public", fmt=fmt) == 1
    out = capsys.readouterr().out
    assert f"partially replaced {fmt} files" in out
    assert not (tmp_path / mirror.MANIFEST_NAME).exists()


def test_cli_parses_and_dispatches(mirror, monkeypatch):
    calls = []
    monkeypatch.setattr(mirror, "choose_dataset", lambda choice: ("owner/repo", choice))
    monkeypatch.setattr(mirror, "main", lambda **kwargs: calls.append(kwargs) or 0)
    assert mirror.cli(["--dataset", "private"]) == 0
    assert calls == [{"dataset_id": "owner/repo", "label": "private", "fmt": "parquet"}]


def test_unknown_format_is_refused(mirror):
    assert mirror.main(dataset_id="owner/repo", label="public", fmt="xlsx") == 2
