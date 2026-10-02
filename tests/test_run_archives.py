"""Research results stay recoverable after the next run replaces latest CSVs."""

import hashlib
import json

import pytest

import _common
from iwac_common.paths import workspace_root


def test_each_run_archives_its_actual_output_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(_common, "_git_state", lambda: {"sha": "code", "dirty": False})
    output = tmp_path / "result.csv"
    output.write_bytes(b"year,value\n2000,1\n")
    manifest = _common.write_run_manifest(
        tmp_path, script="study", repo_id="owner/data", revision="dataset-sha",
        outputs=[output],
    )
    first = json.loads(manifest.read_text())
    output.write_bytes(b"year,value\n2000,2\n")
    _common.write_run_manifest(
        tmp_path, script="study", repo_id="owner/data", revision="new-sha",
        outputs=[output],
    )
    second = json.loads(manifest.read_text())
    assert first["archive"] != second["archive"]
    archived = tmp_path / first["archive"] / "result.csv"
    assert archived.read_bytes() == b"year,value\n2000,1\n"
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == first["outputs"]["result.csv"]["sha256"]
    assert (archived.parent / "environment.json").is_file()
    assert json.loads((archived.parent / "manifest.json").read_text()) == first


def test_manifest_refuses_missing_output(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing run output"):
        _common.write_run_manifest(
            tmp_path, script="study", repo_id=None, revision=None,
            outputs=[tmp_path / "missing.csv"],
        )
    assert not (tmp_path / "study.manifest.json").exists()


def test_workspace_override_is_resolved_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setenv("IWAC_WORK_DIR", str(tmp_path))
    assert workspace_root() == tmp_path


def test_manifest_refuses_ambiguous_basenames(tmp_path):
    with pytest.raises(ValueError, match="distinct filenames"):
        _common.write_run_manifest(
            tmp_path, script="study", repo_id=None, revision=None,
            outputs=[tmp_path / "one" / "result.csv", tmp_path / "two" / "result.csv"],
        )


def test_output_cannot_overwrite_archive_metadata(tmp_path):
    with pytest.raises(ValueError, match="archive metadata"):
        _common.write_run_manifest(
            tmp_path, script="study", repo_id=None, revision=None,
            outputs=[tmp_path / "environment.json"],
        )
