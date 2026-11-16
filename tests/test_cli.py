from iwac_common.schema import ALL_CONFIGS
from iwac_pipeline.cli import UPLOAD_MODULES

import pytest
from types import SimpleNamespace
from iwac_pipeline import cli


def test_unified_upload_cli_covers_every_subset():
    assert set(UPLOAD_MODULES) == set(ALL_CONFIGS)
    for module in UPLOAD_MODULES.values():
        assert cli.importlib.util.find_spec(module) is not None


def test_subset_help_reaches_subset_parser(monkeypatch):
    spec = object()
    monkeypatch.setattr(cli.importlib, "import_module", lambda *args: SimpleNamespace(SPEC=spec))
    calls = []
    monkeypatch.setattr(cli, "run_upload", lambda *args: calls.append(args) or 0)
    assert cli.upload_main(["articles", "--help"]) == 0
    assert calls == [(spec, ["--help"])]


#: Top-level names of this repository's own modules. A missing one of these is
#: a broken route (renamed/moved module), never an optional dependency.
_OWN_MODULES = ("iwac_pipeline", "iwac_common", "country_mapper", "lemmatize_update_hf",
                "_common", "_research", "_stats", "_gemma_encoder", "_embedding_utils")


def _import_route(module_name):
    """Import a dispatch target, skipping only when a third-party extra is absent."""
    try:
        return cli.importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        if not missing or missing in _OWN_MODULES or module_name.startswith(f"{missing}."):
            raise
        pytest.skip(f"{module_name} needs the optional dependency {missing!r}")


@pytest.mark.parametrize("command, module_name", sorted(
    {**cli.PROCESS_MODULES, **cli.ANALYSIS_MODULES}.items()
))
def test_every_process_and_analysis_route_exposes_main(command, module_name):
    module = _import_route(module_name)
    assert callable(getattr(module, "main", None)), f"{command}: {module_name}.main"


def test_dispatch_tables_have_no_duplicate_targets():
    targets = [*cli.PROCESS_MODULES.values(), *cli.ANALYSIS_MODULES.values()]
    assert len(targets) == len(set(targets))


def test_mirror_command_uses_the_script_parser(monkeypatch):
    calls = []
    module = _import_route("iwac_pipeline.mirror.fetch_datasets")
    monkeypatch.setattr(module, "choose_dataset", lambda choice: ("owner/repo", choice))
    monkeypatch.setattr(module, "main", lambda **kwargs: calls.append(kwargs) or 0)
    assert cli.mirror_main(["--dataset", "public", "--format", "csv"]) == 0
    assert calls == [{"dataset_id": "owner/repo", "label": "public", "fmt": "csv"}]
    with pytest.raises(SystemExit):
        cli.mirror_main(["--format", "xlsx"])


def test_mirror_formats_come_from_the_module(capsys):
    module = _import_route("iwac_pipeline.mirror.fetch_datasets")
    with pytest.raises(SystemExit) as exc:
        cli.mirror_main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "iwac-mirror" in out
    for fmt in module.MIRROR_FORMATS:
        assert fmt in out


def test_top_level_help_does_not_load_a_script(monkeypatch, capsys):
    def unexpected(*args):
        pytest.fail("Top-level help must not load an upload script")
    monkeypatch.setattr(cli.importlib, "import_module", unexpected)
    with pytest.raises(SystemExit) as exc:
        cli.upload_main(["--help"])
    assert exc.value.code == 0
    assert "articles" in capsys.readouterr().out
