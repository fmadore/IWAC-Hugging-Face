"""UTF-8 standard streams, forced whenever iwac_common is imported."""
from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from iwac_common.stdio import force_utf8_stdio

ROOT = Path(__file__).resolve().parents[1]


def _stream(encoding: str) -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict")


def test_cp1252_streams_become_utf8(monkeypatch):
    out, err = _stream("cp1252"), _stream("cp1252")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    force_utf8_stdio()
    assert out.encoding == "utf-8" and err.encoding == "utf-8"
    out.write("✓ ⚠ ✗ → ℹ")  # would raise UnicodeEncodeError under cp1252
    out.flush()
    assert out.buffer.getvalue().decode("utf-8") == "✓ ⚠ ✗ → ℹ"


def test_utf8_and_foreign_streams_are_left_alone(monkeypatch):
    utf8 = _stream("UTF-8")
    notebook = io.StringIO()  # no reconfigure()
    monkeypatch.setattr(sys, "stdout", utf8)
    monkeypatch.setattr(sys, "stderr", notebook)
    force_utf8_stdio()
    assert utf8.encoding == "UTF-8" and utf8.errors == "strict"
    assert sys.stderr is notebook


def test_detached_stream_is_skipped(monkeypatch):
    detached = _stream("cp1252")
    detached.detach()
    monkeypatch.setattr(sys, "stdout", detached)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    force_utf8_stdio()  # ValueError from the detached stream is swallowed


def test_pipeline_alias_is_the_same_function():
    from iwac_pipeline._stdio import force_utf8_stdio as alias

    assert alias is force_utf8_stdio


@pytest.mark.parametrize("module", ["iwac_common.repos", "iwac_common.hub"])
def test_importing_iwac_common_forces_utf8_on_a_redirected_cp1252_stdout(module):
    """The direct `python script.py > file` route never imports iwac_pipeline."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env["PYTHONIOENCODING"] = "cp1252"
    env["PYTHONUTF8"] = "0"
    code = f"import {module}, sys; print(sys.stdout.encoding); print('✓')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, cwd=ROOT, env=env)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    encoding, mark = result.stdout.decode("utf-8").split()
    assert encoding.lower().replace("-", "") == "utf8"
    assert mark == "✓"
