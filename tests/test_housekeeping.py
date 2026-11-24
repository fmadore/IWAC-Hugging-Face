"""Small regressions from the housekeeping pass (issue #52)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_review_sheet_links_use_the_public_site_slug(monkeypatch):
    from iwac_common.omeka_client import item_page_url
    from iwac_pipeline.analyses._research import source_link

    assert source_link({"o:id": "42", "iwac_url": "https://x.test/given"}) == "https://x.test/given"
    built = source_link({"o:id": "42", "iwac_url": ""})
    assert built == item_page_url("42")
    assert "/s/westafrica/" not in built


def test_tqdm_is_not_a_direct_dependency():
    core = (ROOT / "requirements-core.txt").read_text(encoding="utf-8")
    assert "tqdm" not in core
    for path in ROOT.joinpath("post-processing").rglob("*.py"):
        assert "from tqdm" not in path.read_text(encoding="utf-8"), path.name


def test_logs_are_ignored():
    assert "logs/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_shingle_words_survive_nfd_and_combining_marks():
    """Issue #41: NFD text and Arabic harakat must not split words."""
    import unicodedata

    from iwac_pipeline.analyses._research import shingles, words

    nfd = unicodedata.normalize("NFD", "La mosquée célèbre de Lomé")
    assert words(nfd) == words("La mosquée célèbre de Lomé") == ["la", "mosquée", "célèbre", "de", "lomé"]
    assert words("بِسْمِ اللَّهِ") == ["بِسْمِ", "اللَّهِ"]
    assert shingles(nfd) == shingles("La mosquée célèbre de Lomé")


def test_lda_script_runs_without_the_installed_package_mapping():
    """Issue #52: direct runs resolve ``iwac_pipeline.processing`` from source."""
    import os
    import subprocess
    import sys

    script = ROOT / "post-processing" / "lda_topic_modeling" / "lda_topic_modeling.py"
    code = (
        "import runpy, sys\n"
        "sys.modules['iwac_pipeline.processing'] = None  # as if not installed\n"
        "script = sys.argv[1]\n"
        "sys.argv = ['lda', '--help']\n"
        "try:\n"
        "    runpy.run_path(script, run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 0, exc.code\n"
        "print(sys.modules['iwac_pipeline.processing.lda_topic_modeling.modeling'].__file__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)], capture_output=True, text=True,
        encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"}, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip().splitlines()[-1]).parent == script.parent
