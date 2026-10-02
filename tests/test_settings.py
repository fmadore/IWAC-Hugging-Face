"""Configuration is loaded before repository defaults in a fresh process."""

import json
import os
from pathlib import Path
import subprocess
import sys


def test_dotenv_targets_resolve_before_import_defaults(tmp_path):
    (tmp_path / ".env").write_text(
        "IWAC_HF_PRIVATE_REPO=scratch/full\nIWAC_HF_PUBLIC_REPO=scratch/public\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    for key in ("IWAC_HF_PRIVATE_REPO", "IWAC_HF_PUBLIC_REPO", "IWAC_ENV_FILE"):
        environment.pop(key, None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    script = """
import json
from iwac_common.repos import PRIVATE_REPO_ID, PUBLIC_REPO_ID
from iwac_common.upload_runner import UploadSpec, build_parser
spec = UploadSpec('articles', [36], None, 'test', '.cache')
print(json.dumps([PRIVATE_REPO_ID, PUBLIC_REPO_ID, build_parser(spec).parse_args([]).repo]))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            env=environment, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == ["scratch/full", "scratch/public", "scratch/full"]


def test_runtime_repo_getters_honor_explicit_environment(monkeypatch):
    from iwac_common.repos import get_private_repo_id, get_public_repo_id

    monkeypatch.setenv("IWAC_HF_PRIVATE_REPO", "scratch/new-full")
    monkeypatch.setenv("IWAC_HF_PUBLIC_REPO", "scratch/new-public")
    assert get_private_repo_id() == "scratch/new-full"
    assert get_public_repo_id() == "scratch/new-public"


def test_workspace_dotenv_is_fallback_when_cwd_has_none(tmp_path):
    working_directory = tmp_path / "launch"
    working_directory.mkdir()
    workspace = tmp_path / "research"
    workspace.mkdir()
    (workspace / ".env").write_text(
        "IWAC_HF_PRIVATE_REPO=workspace/full\nIWAC_HF_PUBLIC_REPO=workspace/public\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.pop("IWAC_ENV_FILE", None)
    environment.pop("IWAC_HF_PUBLIC_REPO", None)
    # Explicit process configuration still takes precedence over workspace .env.
    environment["IWAC_HF_PRIVATE_REPO"] = "explicit/full"
    environment["IWAC_WORK_DIR"] = str(workspace)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    script = """
import json
from iwac_common.repos import PRIVATE_REPO_ID, PUBLIC_REPO_ID
print(json.dumps([PRIVATE_REPO_ID, PUBLIC_REPO_ID]))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=working_directory,
                            env=environment, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == ["explicit/full", "workspace/public"]
