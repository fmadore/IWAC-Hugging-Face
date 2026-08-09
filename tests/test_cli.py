from iwac_common.schema import ALL_CONFIGS
from iwac_pipeline.cli import UPLOAD_SCRIPTS, REPO_ROOT


def test_unified_upload_cli_covers_every_subset():
    assert set(UPLOAD_SCRIPTS) == set(ALL_CONFIGS)
    for relative_path in UPLOAD_SCRIPTS.values():
        assert (REPO_ROOT / relative_path).is_file()
