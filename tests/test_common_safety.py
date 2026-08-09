"""ID alignment and local-mirror integrity regressions."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest
from datasets import Dataset

import _common


def test_add_columns_aligns_by_id_not_row_position():
    ds = Dataset.from_dict({"o:id": ["2", "1"], "title": ["b", "a"]})
    values = pd.DataFrame({"o:id": ["1", "2"], "score": [10, 20]})
    updated = _common.add_columns_by_id(ds, values)
    assert updated["score"] == [20, 10]


def test_add_columns_rejects_id_set_mismatch():
    ds = Dataset.from_dict({"o:id": ["1", "2"]})
    values = pd.DataFrame({"o:id": ["1", "3"], "score": [10, 30]})
    with pytest.raises(ValueError, match="ID mismatch"):
        _common.add_columns_by_id(ds, values)


def _write_manifest(root, csv_bytes: bytes, *, digest: str | None = None):
    data_dir = root / "data"
    data_dir.mkdir()
    path = data_dir / "iwac_articles.csv"
    path.write_bytes(csv_bytes)
    sha = digest or hashlib.sha256(csv_bytes).hexdigest()
    (data_dir / "mirror_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "revision": "abc123",
        "repository": "owner/repo",
        "configs": {
            "articles": {
                "file": path.name,
                "rows": 1,
                "columns": 2,
                "sha256": sha,
            }
        },
    }), encoding="utf-8")


def test_default_csv_mirror_requires_matching_hash(monkeypatch, tmp_path):
    payload = b"o:id,title\n1,hello\n"
    _write_manifest(tmp_path, payload)
    monkeypatch.setattr(_common, "REPO_ROOT", tmp_path)
    frame = _common.load_subset_dataframe("owner/repo", "articles", source="csv")
    assert frame.attrs["iwac_source_revision"] == "abc123"


def test_default_csv_mirror_rejects_mixed_files(monkeypatch, tmp_path):
    _write_manifest(tmp_path, b"o:id,title\n1,hello\n", digest="0" * 64)
    monkeypatch.setattr(_common, "REPO_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _common.load_subset_dataframe("owner/repo", "articles", source="csv")


def test_multi_subset_read_rejects_revision_change(monkeypatch, tmp_path):
    _write_manifest(tmp_path, b"o:id,title\n1,hello\n")
    monkeypatch.setattr(_common, "REPO_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="revision changed"):
        _common.load_subset_dataframe(
            "owner/repo", "articles", source="csv", revision="older"
        )
