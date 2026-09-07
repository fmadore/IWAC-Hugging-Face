from iwac_common.schema import ALL_CONFIGS
from iwac_pipeline.cli import UPLOAD_SCRIPTS, REPO_ROOT

import pytest
from types import SimpleNamespace
from iwac_pipeline import cli


def test_unified_upload_cli_covers_every_subset():
    assert set(UPLOAD_SCRIPTS) == set(ALL_CONFIGS)
    for relative_path in UPLOAD_SCRIPTS.values():
        assert (REPO_ROOT / relative_path).is_file()


def test_subset_help_reaches_subset_parser(monkeypatch):
    spec = object()
    monkeypatch.setattr(cli, "_load_script", lambda *args: SimpleNamespace(SPEC=spec))
    calls = []
    monkeypatch.setattr(cli, "run_upload", lambda *args: calls.append(args) or 0)
    assert cli.upload_main(["articles", "--help"]) == 0
    assert calls == [(spec, ["--help"])]


def test_top_level_help_does_not_load_a_script(monkeypatch, capsys):
    def unexpected(*args):
        pytest.fail("Top-level help must not load an upload script")
    monkeypatch.setattr(cli, "_load_script", unexpected)
    with pytest.raises(SystemExit) as exc:
        cli.upload_main(["--help"])
    assert exc.value.code == 0
    assert "articles" in capsys.readouterr().out
