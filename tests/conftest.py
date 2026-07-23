"""Shared test setup: make `post-processing/` (hyphenated, not importable as a
package) and the repo root importable the same way the scripts do it."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "post-processing"))
