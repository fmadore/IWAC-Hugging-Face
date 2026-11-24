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
