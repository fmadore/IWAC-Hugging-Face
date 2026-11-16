"""Shared stage helpers in post-processing/_common.py."""
from __future__ import annotations

import hashlib
import io
import json

import pytest
from rich.console import Console

import _common


def test_load_environment_honours_iwac_env_file(monkeypatch, tmp_path):
    env = tmp_path / "custom.env"
    env.write_text("IWAC_TEST_ONLY_VARIABLE=from-env-file\n", encoding="utf-8")
    monkeypatch.setenv("IWAC_ENV_FILE", str(env))
    monkeypatch.delenv("IWAC_TEST_ONLY_VARIABLE", raising=False)
    _common.load_environment()
    import os
    assert os.environ["IWAC_TEST_ONLY_VARIABLE"] == "from-env-file"
    monkeypatch.setenv("IWAC_TEST_ONLY_VARIABLE", "explicit")
    _common.load_environment()
    assert os.environ["IWAC_TEST_ONLY_VARIABLE"] == "explicit"  # process env wins


def test_stage_modules_use_the_shared_loader():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "post-processing"
    for name in ("publish_public.py", "related_articles.py", "sentiment_agreement.py",
                 "semantic_embedding.py", "semantic_embedding_images.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "load_environment()" in source, name
        assert "load_dotenv" not in source, name


@pytest.mark.parametrize("given,expected", [("hub", "hub"), ("local", "local"), ("csv", "local")])
def test_normalize_source(given, expected):
    assert _common.normalize_source(given) == expected


def test_normalize_source_rejects_unknown():
    with pytest.raises(ValueError):
        _common.normalize_source("s3")


def test_verify_mirror_manifest_checks_repository_and_hash(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    payload = b"o:id\n1\n"
    (data / "iwac_articles.csv").write_bytes(payload)
    manifest = {"schema_version": 1, "revision": "r1", "repository": "owner/repo",
                "configs": {"articles": {"file": "iwac_articles.csv", "rows": 1,
                                         "sha256": hashlib.sha256(payload).hexdigest()}}}
    (data / "mirror_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    path, loaded, entry = _common.verify_mirror_manifest("owner/repo", "articles", data_dir=data)
    assert path.name == "iwac_articles.csv" and loaded["revision"] == "r1" and entry["rows"] == 1
    with pytest.raises(RuntimeError, match="belongs to"):
        _common.verify_mirror_manifest("other/repo", "articles", data_dir=data)
    (data / "iwac_articles.csv").write_bytes(b"o:id\n2\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _common.verify_mirror_manifest("owner/repo", "articles", data_dir=data)


def test_display_text_stats_counts_bytes_for_chunking():
    out = io.StringIO()
    console = Console(file=out, width=200)
    # 4 characters but 8 UTF-8 bytes: chunked by a byte budget of 6.
    assert _common.display_text_stats(["éééé", "", None, "ab"], "OCR", chunk_bytes=6, console=console) == 2
    assert "Will be chunked" in out.getvalue()
