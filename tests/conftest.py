"""Shared test setup: make `post-processing/` (hyphenated, not importable as a
package) and the repo root importable the same way the scripts do it."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "post-processing"))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _production_public_urls(monkeypatch):
    """Item/IIIF URLs derive from the configured Omeka host, and the upload
    scripts load ``.env`` on import. Pin the production host so a developer's
    staging ``.env`` cannot change expected URLs; tests of the derivation
    itself delete this variable."""
    monkeypatch.setenv("IWAC_PUBLIC_BASE_URL", "https://islam.zmo.de")
